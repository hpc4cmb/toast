# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
import os
import re
import warnings
from time import time

import astropy.io.fits as pf
import numpy as np
import scipy.signal
import traitlets
from astropy import units as u

from .. import qarray as qa
from .._libtoast import filter_poly2D, filter_polynomial, subtract_mean, sum_detectors
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..observation import default_values as defaults
from ..timing import function_timer
from ..intervals import Interval
from ..traits import (
    Bool,
    Dict,
    Instance,
    Int,
    List,
    Quantity,
    Tuple,
    Unicode,
    trait_docs,
)
from ..utils import (
    AlignedF64,
    AlignedU8,
    Environment,
    GlobalTimers,
    Logger,
    Timer,
    dtype_to_aligned,
)
from .noise_estimation_utils import autocov_psd, crosscov_psd, flagged_running_average
from .operator import Operator
from .scan_healpix import ScanHealpixMap, ScanHealpixMask


@trait_docs
class NoiseEstim(Operator):
    """Noise estimation operator"""

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame.  "
        "Only relevant if `maskfile` and/or `mapfile` are set",
    )

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDistribution object is located.  "
        "Only relevant if `maskfile` and/or `mapfile` are set",
    )

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="An instance of a pixel pointing operator.  "
        "Only relevant if `maskfile` and/or `mapfile` are set",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="An instance of a Stokes weights operator.  "
        "Only relevant if `mapfile` is set",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    mask_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for processing mask flags",
    )

    mask_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask for raising processing mask flags"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    output_dir = Unicode(
        None,
        allow_none=True,
        help="If specified, write output data products to this directory",
    )

    maskfile = Unicode(
        None,
        allow_none=True,
        help="Optional HEALPix processing mask",
    )

    mapfile = Unicode(
        None,
        allow_none=True,
        help="Optional HEALPix map to sample and subtract from the signal",
    )

    pol = Bool(True, help="Sample also the polarized part of the map")

    save_cov = Bool(False, help="Save also the sample covariance")

    nbin_psd = Int(1000, allow_none=True, help="Bin the resulting PSD")

    lagmax = Int(10000, help="Maximum lag to consider for the covariance function")

    stationary_period = Quantity(
        86400 * u.s,
        help="Break the observation into several estimation periods of this length",
    )

    nosingle = Bool(
        False, help="Do not evaluate individual PSDs.  Overridden by `pairs`"
    )

    nocross = Bool(True, help="Do not evaluate cross-PSDs.  Overridden by `pairs`")

    calibrate = Bool(False, help="Regress, not just subtract the signal estimate")

    nsum = Int(1, help="Downsampling factor for decimated data")

    naverage = Int(100, help="Smoothing kernel width for downsampled data")

    view = Unicode(
        None, allow_none=True, help="Only measure the covariance within each view"
    )

    pairs = List(
        default_value=None,
        trait=Tuple,
        allow_none=True,
        help="Detector pairs to estimate noise for.  Overrides `nosingle` and `nocross`",
    )

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _redistribute(self, data, obs):
        log = Logger.get()
        timer = Timer()
        timer.start()
        if obs.comm_col is not None and obs.comm_col.size > 1:
            self.redistribute = True
            # Redistribute the data so each process has all detectors for some sample range
            # Duplicate just the fields of the observation we will use
            dup_shared = list()
            if self.shared_flags is not None:
                dup_shared.append(self.shared_flags)
            dup_detdata = [self.det_data]
            if self.det_flags is not None:
                dup_detdata.append(self.det_flags)
            dup_intervals = list()
            temp_obs = obs.duplicate(
                times=self.times,
                meta=list(),
                shared=dup_shared,
                detdata=dup_detdata,
                intervals=dup_intervals,
            )
            log.debug_rank(
                f"{data.comm.group:4} : Duplicated observation in",
                comm=temp_obs.comm,
                timer=timer,
            )
            # Redistribute this temporary observation to be distributed by sample sets
            temp_obs.redistribute(1, times=self.times, override_sample_sets=None)
            log.debug_rank(
                f"{data.comm.group:4} : Redistributed observation in",
                comm=temp_obs.comm,
                timer=timer,
            )
            comm = None
        else:
            self.redistribute = False
            temp_obs = obs

        return temp_obs

    @function_timer
    def _re_redistribute(self, data, obs, temp_obs):
        log = Logger.get()
        timer = Timer()
        timer.start()
        if self.redistribute:
            # Redistribute data back
            temp_ob.redistribute(
                obs.dist.process_rows,
                times=self.times,
                override_sample_sets=obs.dist.sample_sets,
            )
            log.debug_rank(
                f"{data.comm.group:4} : Re-redistributed observation in",
                comm=temp_obs.comm,
                timer=timer,
            )
            # Copy data to original observation
            obs.detdata[self.det_data][:] = temp_obs.detdata[self.det_data][:]
            log.debug_rank(
                f"{data.comm.group:4} : Copied observation data in",
                comm=temp_obs.comm,
                timer=timer,
            )
            self.redistribute = False
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if detectors is not None:
            msg = "NoiseEstim cannot be run in batch mode"
            raise RuntimeError(msg)

        log = Logger.get()

        self.rank = data.comm.world_rank

        if self.maskfile is None:
            scan_mask = None
        else:
            scan_mask = ScanHealpixMask(
                file=self.maskfile,
                det_flags=self.mask_flags,
                def_flags_value=self.mask_bit,
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
            )

        if self.mapfile is None:
            scan_map = None
        else:
            if self.pol:
                weights = self.stokes_weights
            else:
                weights = None
            scan_map = ScanHealpixMap(
                file=self.mapfile,
                det_data=self.det_data,
                subtract=True,
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                stokes_weights=weights,
            )

        for orig_obs in data.obs:
            obs = self._redistribute(data, orig_obs)

            det_names = obs.all_detectors
            det_ids = {}
            for idet, det in enumerate(det_names):
                det_ids[det] = idet
            ndet = len(det_ids)
            if self.pairs is not None:
                pairs = self.pairs
            else:
                # Construct a list of detector pairs
                pairs = []
                for idet1 in range(ndet):
                    det1 = det_names[idet1]
                    for idet2 in range(idet1, ndet):
                        det2 = det_names[idet2]
                        if det1 == det2 and self.nosingle:
                            continue
                        if det1 != det2 and self.nocross:
                            continue
                        pairs.append([det1, det2])

            times = np.array(obs.shared[self.times])
            nsample = times.size

            if self.shared_flags is None:
                shared_flags = np.zeros(times.size, dtype=bool)
            else:
                shared_flags = obs.shared[self.shared_flags].data
                shared_flags = (shared_flags & self.shared_flag_mask) != 0
            fsample = obs.telescope.focalplane.sample_rate.to_value(u.Hz)

            fileroot = f"{self.name}_{obs.name}"

            if self.view is None:
                intervals = [
                    Interval(start=times[0], stop=times[-1], first=0, last=nsample - 1)
                ]
            else:
                intervals = obs.intervals[self.view]

            self.subtract_signal(data, obs, scan_mask, scan_map)

            # self.highpass_signal(obs, comm, intervals)

            # Extend the gap between intervals to prevent sample pairs
            # that cross the gap.

            gap_min = self.lagmax + 1
            # Downsampled data requires longer gaps
            gap_min_nsum = self.lagmax * self.nsum + 1

            gapflags = np.zeros_like(shared_flags)
            gapflags_nsum = np.zeros_like(shared_flags)
            for ival1, ival2 in zip(intervals[:-1], intervals[1:]):
                gap_start = ival1.last + 1
                gap_stop = max(gap_start + gap_min, ival2.first)
                gap_stop_nsum = max(gap_start + gap_min_nsum, ival2.first)

                gapflags[gap_start:gap_stop] = True
                gapflags_nsum[gap_start:gap_stop_nsum] = True

            for det1, det2 in pairs:
                if det1 not in det_names or det2 not in det_names:
                    # User-specified pair is invalid
                    continue
                signal1 = obs.detdata[self.det_data][det1]
                flags1 = obs.detdata[self.det_flags][det1]
                flags = flags1 & self.det_flag_mask != 0
                signal2 = None
                flags2 = None
                if det1 != det2:
                    signal2 = obs.detdata[self.det_data][det2]
                    flags2 = obs.detdata[self.det_flags][det2]
                    flags[flags2 & self._detmask != 0] = True
                flags[shared_flags] = True

                self.process_noise_estimate(
                    obs,
                    signal1,
                    signal2,
                    flags,
                    gapflags,
                    gapflags_nsum,
                    times,
                    fsample,
                    fileroot,
                    det1,
                    det2,
                    intervals,
                )

            self._re_redistribute(data, orig_obs, obs)

        return

    @function_timer
    def highpass_signal(self, obs, comm, intervals):
        """Suppress the sub-harmonic modes in the TOD by high-pass
        filtering.
        """
        log = Logger.get()
        timer = Timer()
        timer.start()
        if self.rank == 0:
            log.info("High-pass-filtering signal")
        for det in obs.local_detectors:
            signal = obs.detdata[self.det_data][det]
            flags = obs.detdata[self.det_flags][det] & self.det_flag_mask
            for ival in intervals:
                ind = slice(ival.first, ival.last + 1)
                sig = signal[ind]
                flg = flags[ind]
                trend = flagged_running_average(
                    sig, flg, self.lagmax, return_flags=False
                )
                sig -= trend
        if self.rank == 0:
            timer.report_clear("TOD high pass")
        return

    @function_timer
    def subtract_signal(self, data, obs, scan_mask, scan_map):
        """Subtract a signal estimate from the TOD and update the
        flags for noise estimation.
        """
        if scan_map is None and scan_mask is None:
            return
        log = Logger.get()
        log.debug_rank("Subtracting signal", comm=obs.comm.comm_group)
        for det in obs.local_detectors:
            if det.endswith("-diff") and not self.pol:
                continue
            obs_data = data.select(obs_uid=obs.uid)
            if self.calibrate:
                # If we are using linear regression, we must scan the
                # signal onto another TOD object first
                msg = "Linear regression of signal estimate not implemented"
                raise RuntimeError(msg)
            if scan_map is not None:
                scan_map.apply(obs_data, detectors=[det])
            if scan_mask is not None:
                scan_mask.apply(obs_data, detectors=[det])
        log.debug_rank("Subtracted signal", comm=obs.comm.comm_group)
        return

    @function_timer
    def decimate(self, x, flg, gapflg, intervals):
        # Low-pass filter with running average, then downsample
        xx = x.copy()
        flags = flg.copy()
        for ival in intervals:
            ind = slice(ival.first, ival.last + 1)
            xx[ind], flags[ind] = flagged_running_average(
                x[ind], flg[ind], self.naverage, return_flags=True
            )
        return xx[:: self.nsum].copy(), (flags + gapflg)[:: self.nsum].copy()

    # def highpass(self, x, flg):
    #     # Flagged real-space high pass filter
    #     xx = x.copy()
    #
    #     j = 0
    #     while j < x.size and flg[j]: j += 1
    #
    #     alpha = .999
    #
    #     for i in range(j+1, x.size):
    #         if flg[i]:
    #             xx[i] = x[j]
    #         else:
    #             xx[i] = alpha*(xx[j] + x[i] - x[j])
    #             j = i
    #
    #     xx /= alpha
    #     return xx

    @function_timer
    def log_bin(self, freq, nbin=100, fmin=None, fmax=None):
        if np.any(freq == 0):
            msg = "Logarithmic binning should not include zero frequency"
            raise Exception(msg)

        if fmin is None:
            fmin = np.amin(freq)
        if fmax is None:
            fmax = np.amax(freq)

        bins = np.logspace(
            np.log(fmin), np.log(fmax), num=nbin + 1, endpoint=True, base=np.e
        )
        bins[-1] *= 1.01  # Widen the last bin not to have a bin with one entry

        locs = np.digitize(freq, bins).astype(np.int32)
        hits = np.zeros(nbin + 2, dtype=np.int32)
        for loc in locs:
            hits[loc] += 1
        return locs, hits

    @function_timer
    def bin_psds(self, my_psds, fmin=None, fmax=None):
        my_binned_psds = []
        my_times = []
        binfreq0 = None

        for i in range(len(my_psds)):
            t0, _, freq, psd = my_psds[i]

            good = freq != 0

            if self.nbin_psd is not None:
                locs, hits = self.log_bin(
                    freq[good], nbin=self.nbin_psd, fmin=fmin, fmax=fmax
                )
                binfreq = np.zeros(hits.size)
                for loc, f in zip(locs, freq[good]):
                    binfreq[loc] += f
                binfreq = binfreq[hits != 0] / hits[hits != 0]
            else:
                binfreq = freq
                hits = np.ones(len(binfreq))

            if binfreq0 is None:
                binfreq0 = binfreq
            else:
                if np.any(binfreq != binfreq0):
                    msg = "Binned PSD frequencies change"
                    raise Exception(msg)

            if self.nbin_psd is not None:
                binpsd = np.zeros(hits.size)
                for loc, p in zip(locs, psd[good]):
                    binpsd[loc] += p
                binpsd = binpsd[hits != 0] / hits[hits != 0]
            else:
                binpsd = psd

            my_times.append(t0)
            my_binned_psds.append(binpsd)
        return my_binned_psds, my_times, binfreq0

    @function_timer
    def discard_outliers(self, binfreq, all_psds, all_times, all_cov):
        log = Logger.get()

        all_psds = copy.deepcopy(all_psds)
        all_times = copy.deepcopy(all_times)
        if self.save_cov:
            all_cov = copy.deepcopy(all_cov)

        nrow, ncol = np.shape(all_psds)

        # Discard empty PSDs

        i = 1
        nempty = 0
        while i < nrow:
            p = all_psds[i]
            if np.all(p == 0) or np.any(np.isnan(p)):
                del all_psds[i]
                del all_times[i]
                if self.save_cov:
                    del all_cov[i]
                nrow -= 1
                nempty += 1
            else:
                i += 1

        if nempty > 0:
            log.info(f"Discarded {nempty} empty or NaN psds")

        # Throw away outlier PSDs by comparing the PSDs in specific bins

        if nrow < 10:
            nbad = 0
        else:
            all_good = np.isfinite(np.sum(all_psds, 1))
            for col in range(ncol - 1):
                if binfreq[col] < 0.001:
                    continue

                # Local outliers

                psdvalues = np.array([x[col] for x in all_psds])
                smooth_values = scipy.signal.medfilt(psdvalues, 11)
                good = np.ones(psdvalues.size, dtype=np.bool)
                good[psdvalues == 0] = False

                for i in range(10):
                    # Local test
                    diff = np.zeros(psdvalues.size)
                    diff[good] = np.log(psdvalues[good]) - np.log(smooth_values[good])
                    sdev = np.std(diff[good])
                    good[np.abs(diff) > 5 * sdev] = False
                    # Global test
                    diff = np.zeros(psdvalues.size)
                    diff[good] = np.log(psdvalues[good]) - np.mean(
                        np.log(psdvalues[good])
                    )
                    sdev = np.std(diff[good])
                    good[np.abs(diff) > 5 * sdev] = False

                all_good[np.logical_not(good)] = False

            bad = np.logical_not(all_good)
            nbad = np.sum(bad)
            if nbad > 0:
                for ii in np.argwhere(bad).ravel()[::-1]:
                    del all_psds[ii]
                    del all_times[ii]
                    if self.save_cov:
                        del all_cov[ii]

            if nbad > 0:
                log.info(f"Masked extra {nbad} psds due to outliers.")
        return all_psds, all_times, nempty + nbad, all_cov

    @function_timer
    def save_psds(
        self, binfreq, all_psds, all_times, det1, det2, fsample, rootname, all_cov
    ):
        timer = Timer()
        timer.start()
        if det1 == det2:
            fn_out = os.path.join(self.output_dir, f"{rootname}_{det1}.fits")
        else:
            fn_out = os.path.join(self.output_dir, f"{rootname}_{det1}_{det2}.fits")
        all_psds = np.vstack([binfreq, all_psds])

        hdulist = [pf.PrimaryHDU()]

        cols = []
        cols.append(pf.Column(name="OBT", format="D", array=all_times))
        coldefs = pf.ColDefs(cols)
        hdu1 = pf.BinTableHDU.from_columns(coldefs)
        hdu1.header["RATE"] = fsample, "Sampling rate"
        hdulist.append(hdu1)

        cols = []
        cols.append(pf.Column(name="PSD", format=f"{binfreq.size}E", array=all_psds))
        coldefs = pf.ColDefs(cols)
        hdu2 = pf.BinTableHDU.from_columns(coldefs)
        hdu2.header["EXTNAME"] = str(det1), "Detector"
        hdu2.header["DET1"] = str(det1), "Detector1"
        hdu2.header["DET2"] = str(det2), "Detector2"
        hdulist.append(hdu2)

        if self.save_cov:
            all_cov = np.array(all_cov)
            cols = []
            nrow, ncol, nsamp = np.shape(all_cov)
            cols.append(
                pf.Column(
                    name="HITS",
                    format=f"{nsamp}J",
                    array=np.ascontiguousarray(all_cov[:, 0, :]),
                )
            )
            cols.append(
                pf.Column(
                    name="COV",
                    format=f"{nsamp}E",
                    array=np.ascontiguousarray(all_cov[:, 1, :]),
                )
            )
            coldefs = pf.ColDefs(cols)
            hdu3 = pf.BinTableHDU.from_columns(coldefs)
            hdu3.header["EXTNAME"] = str(det1), "Detector"
            hdu3.header["DET1"] = str(det1), "Detector1"
            hdu3.header["DET2"] = str(det2), "Detector2"
            hdulist.append(hdu3)

        hdulist = pf.HDUList(hdulist)

        with open(fn_out, "wb") as fits_out:
            hdulist.writeto(fits_out, overwrite=True)

        print(f"Detector {det1} vs. {det2} PSDs stored in {fn_out}")
        return

    @function_timer
    def process_downsampled_noise_estimate(
        self,
        timestamps,
        fsample,
        signal1,
        signal2,
        flags,
        gapflags_nsum,
        local_intervals,
        my_psds1,
        my_cov1,
        comm,
    ):
        # Get another PSD for a down-sampled TOD to measure the
        # low frequency power

        timestamps_decim = timestamps[:: self.nsum]
        # decimate() will smooth and downsample the signal in
        # each valid interval separately
        signal1_decim, flags_decim = self.decimate(
            signal1, flags, gapflags_nsum, local_intervals
        )
        if signal2 is not None:
            signal2_decim, flags_decim = self.decimate(
                signal2, flags, gapflags_nsum, local_intervals
            )

        stationary_period = self.stationary_period.to_value(u.s)
        lagmax = min(self.lagmax, timestamps_decim.size)
        if signal2 is None:
            result = autocov_psd(
                timestamps_decim,
                signal1_decim,
                flags_decim,
                lagmax,
                stationary_period,
                fsample / self.nsum,
                comm=comm,
                return_cov=self.save_cov,
            )
        else:
            result = crosscov_psd(
                timestamps_decim,
                signal1_decim,
                signal2_decim,
                flags_decim,
                lagmax,
                stationary_period,
                fsample / self.nsum,
                comm=comm,
                return_cov=self.save_cov,
            )
        if self.save_cov:
            my_psds2, my_cov2 = result
        else:
            my_psds2, my_cov2 = result, None

        # Ensure the two sets of PSDs are of equal length

        my_new_psds1 = []
        my_new_psds2 = []
        if self.save_cov:
            my_new_cov1 = []
            my_new_cov2 = []
        i = 0
        while i < min(len(my_psds1), len(my_psds2)):
            t1 = my_psds1[i][0]
            t2 = my_psds2[i][0]
            if np.isclose(t1, t2):
                my_new_psds1.append(my_psds1[i])
                my_new_psds2.append(my_psds2[i])
                if self.save_cov:
                    my_new_cov1.append(my_cov1[i])
                    my_new_cov2.append(my_cov2[i])
                i += 1
            else:
                if t1 < t2:
                    del my_psds1[i]
                    if self._cov:
                        del my_cov1[i]
                else:
                    del my_psds2[i]
                    if self._cov:
                        del my_cov1[i]
        my_psds1 = my_new_psds1
        my_psds2 = my_new_psds2
        if self.save_cov:
            my_cov1 = my_new_cov1
            my_cov2 = my_new_cov2

        if len(my_psds1) != len(my_psds2):
            while my_psds1[-1][0] > my_psds2[-1][0]:
                del my_psds1[-1]
                if self.save_cov:
                    del my_cov1[-1]
            while my_psds1[-1][0] < my_psds2[-1][0]:
                del my_psds2[-1]
                if self.save_cov:
                    del my_cov2[-1]
        return my_psds1, my_cov1, my_psds2, my_cov2

    @function_timer
    def process_noise_estimate(
        self,
        obs,
        signal1,
        signal2,
        flags,
        gapflags,
        gapflags_nsum,
        timestamps,
        fsample,
        fileroot,
        det1,
        det2,
        local_intervals,
    ):
        # High pass filter the signal to avoid aliasing
        # self.highpass(signal1, noise_flags)
        # self.highpass(signal2, noise_flags)

        # Compute the autocovariance function and the matching
        # PSD for each stationary interval

        timer = Timer()
        timer.start()
        comm = obs.comm_row
        stationary_period = self.stationary_period.to_value(u.s)
        if signal2 is None:
            result = autocov_psd(
                timestamps,
                signal1,
                flags + gapflags,
                self.lagmax,
                stationary_period,
                fsample,
                comm=comm,
                return_cov=self.save_cov,
            )
        else:
            result = crosscov_psd(
                timestamps,
                signal1,
                signal2,
                flags + gapflags,
                self.lagmax,
                stationary_period,
                fsample,
                comm=comm,
                return_cov=self.save_cov,
            )
        if self.save_cov:
            my_psds1, my_cov1 = result
        else:
            my_psds1, my_cov1 = result, None

        if self.nsum > 1:
            (
                my_psds1,
                my_cov1,
                my_psds2,
                my_cov2,
            ) = self.process_downsampled_noise_estimate(
                timestamps,
                fsample,
                signal1,
                signal2,
                flags,
                gapflags_nsum,
                local_intervals,
                my_psds1,
                my_cov1,
                comm,
            )

        if self.rank == 0:
            timer.report_clear("Compute Correlators and PSDs")

        # Now bin the PSDs

        fmin = 1 / stationary_period
        fmax = fsample / 2

        my_binned_psds1, my_times1, binfreq10 = self.bin_psds(my_psds1, fmin, fmax)
        if self.nsum > 1:
            my_binned_psds2, _, binfreq20 = self.bin_psds(my_psds2, fmin, fmax)
        if self.rank == 0:
            timer.report_clear("Bin PSDs")

        # concatenate

        if binfreq10 is None:
            my_times = []
            my_binned_psds = []
            binfreq0 = None
        else:
            my_times = my_times1
            if self.save_cov:
                my_cov = my_cov1  # Only store the fully sampled covariance
            if self.nsum > 1:
                # frequencies that are usable in the down-sampled PSD
                fcut = fsample / 2 / self.naverage / 100
                ind1 = binfreq10 > fcut
                ind2 = binfreq20 <= fcut
                binfreq0 = np.hstack([binfreq20[ind2], binfreq10[ind1]])
                my_binned_psds = []
                for psd1, psd2 in zip(my_binned_psds1, my_binned_psds2):
                    my_binned_psds.append(np.hstack([psd2[ind2], psd1[ind1]]))
            else:
                binfreq0 = binfreq10
                my_binned_psds = my_binned_psds1

        # Collect and write the PSDs.  Start by determining the first
        # process to have a valid PSD to determine binning

        have_bins = binfreq0 is not None
        have_bins_all = None
        if comm is None:
            have_bins_all = [have_bins]
        else:
            have_bins_all = comm.allgather(have_bins)
        root = 0
        if np.any(have_bins_all):
            while not have_bins_all[root]:
                root += 1
        else:
            msg = "None of the processes have valid PSDs"
            raise RuntimeError(msg)
        binfreq = None
        if comm is None:
            binfreq = binfreq0
        else:
            binfreq = comm.bcast(binfreq0, root=root)
        if binfreq0 is not None and np.any(binfreq != binfreq0):
            msg = (
                f"{self.rank:4} : Binned PSD frequencies change. "
                f"len(binfreq0) = {binfreq0.size}, "
                f"len(binfreq) = {binfreq.size}, binfreq0={binfreq0}, "
                f"binfreq = {binfreq}. len(my_psds) = {len(my_psds1)}"
            )
            raise Exception(msg)

        if len(my_times) != len(my_binned_psds):
            msg = (
                f"ERROR: Process {self.rank} has len(my_times) = {len(my_times)}, "
                f"len(my_binned_psds) = {len(my_binned_psds)}"
            )
            raise Exception(msg)

        all_times = None
        all_psds = None
        if comm is None:
            all_times = [my_times]
            all_psds = [my_binned_psds]
        else:
            all_times = comm.gather(my_times, root=0)
            all_psds = comm.gather(my_binned_psds, root=0)
        all_cov = None
        if self.save_cov:
            if comm is None:
                all_cov = [my_cov]
            else:
                all_cov = comm.gather(my_cov, root=0)
        if self.rank == 0:
            timer.report_clear("Collect PSDs")

        if self.rank == 0:
            # FIXME: original code had no timing report here for the previous block.
            # Was one intended?
            if len(all_times) != len(all_psds):
                msg = (
                    f"ERROR: Process {self.rank} has len(all_times) = {len(all_times)},"
                    f" len(all_psds) = {len(all_psds)} before deglitch"
                )
                raise Exception(msg)

            # De-glitch the binned PSDs and write them to file
            i = 0
            while i < len(all_times):
                if len(all_times[i]) == 0:
                    del all_times[i]
                    del all_psds[i]
                    if self.save_cov:
                        del all_cov[i]
                else:
                    i += 1

            if len(all_times) != len(all_psds):
                msg = (
                    f"ERROR: Process {self.rank} has len(all_times) = {len(all_times)}, "
                    f"len(all_psds) = {len(all_psds)} AFTER deglitch"
                )
                raise Exception(msg)

            all_times = list(np.hstack(all_times))
            all_psds = list(np.hstack(all_psds))
            if self.save_cov:
                all_cov = list(np.hstack(all_cov))

            good_psds, good_times, nbad, good_cov = self.discard_outliers(
                binfreq, all_psds, all_times, all_cov
            )
            timer.report_clear("Discard outliers")

            self.save_psds(
                binfreq, all_psds, all_times, det1, det2, fsample, fileroot, all_cov
            )

            if nbad > 0:
                self.save_psds(
                    binfreq,
                    good_psds,
                    good_times,
                    det1,
                    det2,
                    fsample,
                    fileroot + "_good",
                    good_cov,
                )
            timer.report_clear("Write PSDs")

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.detector_pointing is not None:
            req.update(self.detector_pointing.requires())
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov

    def _accelerators(self):
        return list()
