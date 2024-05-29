# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
import os

import astropy.io.fits as pf
import numpy as np
import scipy.signal
import traitlets
from astropy import units as u

from ..noise import Noise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Instance, Int, List, Quantity, Tuple, Unicode, trait_docs
from ..utils import Logger
from .arithmetic import Combine
from .copy import Copy
from .delete import Delete
from .noise_estimation_utils import (
    autocov_psd,
    communicate_overlap,
    crosscov_psd,
    flagged_running_average,
    highpass_flagged_signal,
)
from .operator import Operator
from .polyfilter import CommonModeFilter
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

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
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
        defaults.det_mask_invalid,
        help="Bit mask value for detector sample flagging",
    )

    mask_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for processing mask flags",
    )

    mask_flag_mask = Int(
        defaults.det_mask_processing, help="Bit mask for raising processing mask flags"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional shared flagging",
    )

    out_model = Unicode(
        None, allow_none=True, help="Create a new noise model with this name"
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

    symmetric = Bool(
        False,
        help="If True, treat positive and negative lags as equivalent "
        "in the cross correlator",
    )

    nbin_psd = Int(1000, allow_none=True, help="Bin the resulting PSD")

    lagmax = Int(
        10000,
        help="Maximum lag to consider for the covariance function. "
        "Will be truncated the length of the longest view.",
    )

    stationary_period = Quantity(
        86400 * u.s,
        help="Break the observation into several estimation periods of this length",
    )

    nosingle = Bool(
        False, help="Do not evaluate individual PSDs.  Overridden by `pairs`"
    )

    nocross = Bool(True, help="Do not evaluate cross-PSDs.  Overridden by `pairs`")

    nsum = Int(1, help="Downsampling factor for decimated data")

    naverage = Int(100, help="Smoothing kernel width for downsampled data")

    view = Unicode(
        None, allow_none=True, help="Only measure the covariance within each view"
    )

    pairs = List(
        [],
        help="Detector pairs to estimate noise for.  Overrides `nosingle` and `nocross`",
    )

    focalplane_key = Unicode(
        None, allow_none=True, help="When set, PSDs are measured over averaged TODs"
    )

    remove_common_mode = Bool(False, help="Remove common mode signal before estimation")

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
                "det_mask",
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

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _redistribute(self, obs):
        log = Logger.get()
        timer = Timer()
        timer.start()
        if (len(self.pairs) > 0 or (not self.nocross)) and (
            obs.comm_col is not None and obs.comm_col.size > 1
        ):
            self.redistribute = True
            # Redistribute the data so each process has all detectors
            # for some sample range
            # Duplicate just the fields of the observation we will use
            dup_shared = [self.times]
            if self.shared_flags is not None:
                dup_shared.append(self.shared_flags)
            dup_detdata = [self.det_data]
            if self.det_flags is not None:
                dup_detdata.append(self.det_flags)
            dup_intervals = list()
            if self.view is not None:
                dup_intervals.append(self.view)
            temp_obs = obs.duplicate(
                times=self.times,
                meta=list(),
                shared=dup_shared,
                detdata=dup_detdata,
                intervals=dup_intervals,
            )
            log.debug_rank(
                f"{obs.comm.group:4} : Duplicated observation in",
                comm=temp_obs.comm.comm_group,
                timer=timer,
            )
            # Redistribute this temporary observation to be distributed by samples
            global_intervals = temp_obs.redistribute(
                1,
                times=self.times,
                override_sample_sets=None,
                return_global_intervals=True,
            )
            if self.view is not None:
                global_intervals = global_intervals[self.view]
            log.debug_rank(
                f"{obs.comm.group:4} : Redistributed observation in",
                comm=temp_obs.comm.comm_group,
                timer=timer,
            )
        else:
            self.redistribute = False
            temp_obs = obs
            global_intervals = []
            if self.view is not None:
                for ival in obs.intervals[self.view]:
                    global_intervals.append((ival.start, ival.stop))
        if self.view is None:
            global_intervals = [(None, None)]

        return temp_obs, global_intervals

    @function_timer
    def _re_redistribute(self, obs, temp_obs):
        log = Logger.get()
        timer = Timer()
        timer.start()
        if self.redistribute:
            # Redistribute data back
            temp_obs.redistribute(
                obs.dist.process_rows,
                times=self.times,
                override_sample_sets=obs.dist.sample_sets,
            )
            log.debug_rank(
                f"{temp_obs.comm.group:4} : Re-redistributed observation in",
                comm=temp_obs.comm.comm_group,
                timer=timer,
            )
            # Copy data to original observation
            obs.detdata[self.det_data][:] = temp_obs.detdata[self.det_data][:]
            log.debug_rank(
                f"{temp_obs.comm.group:4} : Copied observation data in",
                comm=temp_obs.comm.comm_group,
                timer=timer,
            )
            if self.out_model is not None:
                obs[self.out_model] = temp_obs[self.out_model]
            self.redistribute = False
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if detectors is not None:
            msg = "NoiseEstim cannot be run with subsets of detectors"
            raise RuntimeError(msg)

        log = Logger.get()

        if self.focalplane_key is not None:
            if len(self.pairs) > 0:
                msg = "focalplane_key is not compatible with pairs"
                raise RuntimeError(msg)
            if self.remove_common_mode:
                # Measure and subtract the common mode signal across the focalplane.
                Copy(detdata=[(self.det_data, "temp_signal")]).apply(data)
                CommonModeFilter(
                    det_data="temp_signal",
                    det_mask=self.det_mask,
                    det_flags=self.det_flags,
                    det_flag_mask=self.det_flag_mask,
                    focalplane_key=self.focalplane_key,
                ).apply(data)
                Combine(
                    op="subtract",
                    first=self.det_data,
                    second="temp_signal",
                    output=self.det_data,
                ).apply(data)
                Delete(detdata="temp_signal")

        if self.mapfile is not None:
            if self.pol:
                weights = self.stokes_weights
            else:
                weights = None
            scan_map = ScanHealpixMap(
                file=self.mapfile,
                det_data=self.det_data,
                det_mask=self.det_mask,
                subtract=True,
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                stokes_weights=weights,
            )
            scan_map.apply(data, detectors=detectors)

        if self.maskfile is not None:
            scan_mask = ScanHealpixMask(
                file=self.maskfile,
                det_mask=self.det_mask,
                det_flags=self.mask_flags,
                def_flags_value=self.mask_flag_mask,
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
            )
            scan_mask.apply(data, detectors=detectors)

        for orig_obs in data.obs:
            obs, global_intervals = self._redistribute(orig_obs)

            # Get the set of all detectors we are considering for this obs.  Since
            # we have already redistributed the data, every process has a time slice
            # and the same set of local detectors.
            local_dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            good_dets = set(local_dets)

            if self.focalplane_key is not None:
                # Pick just one detector to represent each key value
                fp = obs.telescope.focalplane
                det_names = []
                key2det = {}
                det2key = {}
                for det in local_dets:
                    key = fp[det][self.focalplane_key]
                    if key not in key2det:
                        det_names.append(det)
                        key2det[key] = det
                        det2key[det] = key
                pairs = []
                for det1 in key2det.values():
                    for det2 in key2det.values():
                        if det1 == det2 and self.nosingle:
                            continue
                        if det1 != det2 and self.nocross:
                            continue
                        pairs.append([det1, det2])
            else:
                det2key = None
                det_names = obs.all_detectors
                ndet = len(det_names)
                if len(self.pairs) > 0:
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

            if self.symmetric:
                # Remove duplicate entries in pair list
                unordered_pairs = set()
                for pair in pairs:
                    unordered_pairs.add(tuple(sorted(pair)))
                pairs = list(unordered_pairs)

            times = np.array(obs.shared[self.times])
            nsample = times.size

            shared_flags = np.zeros(times.size, dtype=bool)
            if self.shared_flags is not None:
                shared_flags[:] = (
                    obs.shared[self.shared_flags].data & self.shared_flag_mask
                ) != 0

            fsample = obs.telescope.focalplane.sample_rate.to_value(u.Hz)

            fileroot = f"{self.name}_{obs.name}"

            # Re-use this flag array
            flags = np.zeros(times.size, dtype=bool)

            noise_dets = list()
            noise_freqs = dict()
            noise_psds = dict()
            noise_indices = dict()

            for det1, det2 in pairs:
                if det1 not in det_names or det2 not in det_names:
                    # User-specified pair is invalid
                    continue
                if det1 not in good_dets or (
                    det2 is not None and det2 not in good_dets
                ):
                    # One of our detectors is cut.  Store a zero PSD.
                    nse_freqs = np.array(
                        [
                            0.0,
                            1.0e-5,
                            fsample / 4,
                            fsample / 2,
                        ],
                        dtype=np.float64,
                    )
                    nse_psd = np.zeros_like(nse_freqs)
                else:
                    signal1 = obs.detdata[self.det_data][det1]

                    flags[:] = shared_flags
                    if self.det_flags is not None:
                        flags[:] |= (
                            obs.detdata[self.det_flags][det1] & self.det_flag_mask
                        ) != 0

                    signal2 = None
                    if det1 != det2:
                        signal2 = obs.detdata[self.det_data][det2]
                        if self.det_flags is not None:
                            flags[:] |= (
                                obs.detdata[self.det_flags][det2] & self.det_flag_mask
                            ) != 0

                    if det2key is None:
                        det1_name = det1
                        det2_name = det2
                    else:
                        det1_name = det2key[det1]
                        det2_name = det2key[det2]

                    nse_freqs, nse_psd = self.process_noise_estimate(
                        obs,
                        global_intervals,
                        signal1,
                        signal2,
                        flags,
                        times,
                        fsample,
                        fileroot,
                        det1_name,
                        det2_name,
                        self.lagmax,
                    )

                if obs.comm.group_rank == 0:
                    det_units = obs.detdata[self.det_data].units
                    if det_units == u.dimensionless_unscaled:
                        msg = f"Observation {obs.name}, detector data '{self.det_data}'"
                        msg += f" has no units.  Assuming Kelvin."
                        log.warning(msg)
                        det_units = u.K
                    psd_unit = det_units**2 * u.second
                    noise_dets.append(det1)
                    noise_freqs[det1] = nse_freqs[1:] * u.Hz
                    noise_psds[det1] = nse_psd[1:] * psd_unit
                    noise_indices[det1] = obs.telescope.focalplane[det1]["uid"]

            if self.out_model is not None:
                # Create a noise model.  Our observation is currently distributed
                # so that every process has all detectors.
                if data.comm.comm_group is not None:
                    noise_dets = data.comm.comm_group.bcast(noise_dets, root=0)
                    noise_freqs = data.comm.comm_group.bcast(noise_freqs, root=0)
                    noise_psds = data.comm.comm_group.bcast(noise_psds, root=0)
                    noise_indices = data.comm.comm_group.bcast(noise_indices, root=0)
                obs[self.out_model] = Noise(
                    detectors=noise_dets,
                    freqs=noise_freqs,
                    psds=noise_psds,
                    indices=noise_indices,
                )

            # Redistribute the observation, replacing the input TOD with the filtered
            # one and redistributing the noise model.
            self._re_redistribute(orig_obs, obs)

            # Delete temporary obs
            del obs

    @function_timer
    def decimate(self, signal, flags):
        """Downsample previously highpass-filtered signal"""
        return signal[:: self.nsum].copy(), flags[:: self.nsum].copy()

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
                    raise RuntimeError(msg)

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
            log.debug(f"Discarded {nempty} empty or NaN psds")

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
                good = np.ones(psdvalues.size, dtype=bool)
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
                log.debug(f"Masked extra {nbad} psds due to outliers.")
        return all_psds, all_times, nempty + nbad, all_cov

    @function_timer
    def save_psds(
        self, binfreq, all_psds, all_times, det1, det2, fsample, rootname, all_cov
    ):
        log = Logger.get()
        timer = Timer()
        timer.start()
        os.makedirs(self.output_dir, exist_ok=True)
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

        log.debug(f"Detector {det1} vs. {det2} PSDs stored in {fn_out}")

        return

    @function_timer
    def process_downsampled_noise_estimate(
        self,
        obs,
        global_intervals,
        timestamps,
        fsample,
        signal1,
        signal2,
        flags,
        my_psds1,
        my_cov1,
        comm,
        lagmax,
    ):
        # Get another PSD for a down-sampled TOD to measure the
        # low frequency power

        timestamps_decim = timestamps[:: self.nsum]
        # decimate() will smooth and downsample the signal in
        # each valid interval separately
        signal1_decim, flags_decim = self.decimate(signal1, flags)
        if signal2 is None:
            signal2_decim = None
        else:
            signal2_decim, flags_decim = self.decimate(signal2, flags)

        stationary_period = self.stationary_period.to_value(u.s)
        lagmax = min(lagmax, timestamps_decim.size)

        # We apply a prewhitening filter to the signal.  To accommodate the
        # quality flags, the filter is a moving average that only accounts
        # for the unflagged samples
        naverage = lagmax

        # Extend the local arrays to remove boundary effects from filtering
        comm = obs.comm_row
        (
            extended_times,
            extended_flags,
            extended_signal1,
            extended_signal2,
        ) = communicate_overlap(
            timestamps_decim,
            signal1_decim,
            signal2_decim,
            flags_decim,
            lagmax,
            naverage,
            comm,
            obs.comm.group,
        )
        # High pass filter the signal to avoid aliasing
        extended_signal1 = highpass_flagged_signal(
            extended_signal1,
            extended_flags == 0,
            naverage,
        )
        if signal2 is not None:
            extended_signal2 = highpass_flagged_signal(
                extended_signal2,
                extended_flags == 0,
                naverage,
            )
        # Crop the filtering margin but keep up to lagmax samples
        half_average = naverage // 2 + 1
        if comm is not None and comm.rank > 0:
            extended_times = extended_times[half_average:]
            extended_flags = extended_flags[half_average:]
            extended_signal1 = extended_signal1[half_average:]
            if extended_signal2 is not None:
                extended_signal2 = extended_signal2[half_average:]
        if comm is not None and comm.rank < comm.size - 1:
            extended_times = extended_times[:-half_average]
            extended_flags = extended_flags[:-half_average]
            extended_signal1 = extended_signal1[:-half_average]
            if extended_signal2 is not None:
                extended_signal2 = extended_signal2[:-half_average]

        if signal2 is None:
            result = autocov_psd(
                timestamps_decim,
                extended_times,
                global_intervals,
                extended_signal1,
                extended_flags,
                lagmax,
                naverage,
                stationary_period,
                fsample / self.nsum,
                comm=comm,
                return_cov=self.save_cov,
            )
        else:
            result = crosscov_psd(
                timestamps_decim,
                extended_times,
                global_intervals,
                extended_signal1,
                extended_signal2,
                extended_flags,
                lagmax,
                naverage,
                stationary_period,
                fsample / self.nsum,
                comm=comm,
                return_cov=self.save_cov,
                symmetric=self.symmetric,
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
        global_intervals,
        signal1,
        signal2,
        flags,
        timestamps,
        fsample,
        fileroot,
        det1,
        det2,
        lagmax,
    ):
        """Measure the sample (cross) covariance in the signal-subtracted
        TOD and Fourier-transform it for noise PSD.
        """

        log = Logger.get()

        # We apply a prewhitening filter to the signal.  To accommodate the
        # quality flags, the filter is a moving average that only accounts
        # for the unflagged samples
        naverage = lagmax

        # Extend the local arrays to remove boundary effects from filtering
        comm = obs.comm_row
        (
            extended_times,
            extended_flags,
            extended_signal1,
            extended_signal2,
        ) = communicate_overlap(
            timestamps, signal1, signal2, flags, lagmax, naverage, comm, obs.comm.group
        )
        # High pass filter the signal to avoid aliasing
        extended_signal1 = highpass_flagged_signal(
            extended_signal1,
            extended_flags == 0,
            naverage,
        )
        if signal2 is not None:
            extended_signal2 = highpass_flagged_signal(
                extended_signal2,
                extended_flags == 0,
                naverage,
            )
        # Crop the filtering margin but keep up to lagmax samples
        half_average = naverage // 2 + 1
        if comm is not None and comm.rank > 0:
            extended_times = extended_times[half_average:]
            extended_flags = extended_flags[half_average:]
            extended_signal1 = extended_signal1[half_average:]
            if extended_signal2 is not None:
                extended_signal2 = extended_signal2[half_average:]
        if comm is not None and comm.rank < comm.size - 1:
            extended_times = extended_times[:-half_average]
            extended_flags = extended_flags[:-half_average]
            extended_signal1 = extended_signal1[:-half_average]
            if extended_signal2 is not None:
                extended_signal2 = extended_signal2[:-half_average]

        # Compute the autocovariance function and the matching
        # PSD for each stationary interval

        timer = Timer()
        timer.start()
        stationary_period = self.stationary_period.to_value(u.s)
        if signal2 is None:
            result = autocov_psd(
                timestamps,
                extended_times,
                global_intervals,
                extended_signal1,
                extended_flags,
                lagmax,
                naverage,
                stationary_period,
                fsample,
                comm=comm,
                return_cov=self.save_cov,
            )
        else:
            result = crosscov_psd(
                timestamps,
                extended_times,
                global_intervals,
                extended_signal1,
                extended_signal2,
                extended_flags,
                lagmax,
                naverage,
                stationary_period,
                fsample,
                comm=comm,
                return_cov=self.save_cov,
                symmetric=self.symmetric,
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
                obs,
                global_intervals,
                extended_times,
                fsample,
                extended_signal1,
                extended_signal2,
                extended_flags,
                my_psds1,
                my_cov1,
                comm,
                lagmax,
            )

        log.debug_rank(
            "Compute Correlators and PSDs",
            comm=obs.comm.comm_group,
            rank=0,
            timer=timer,
        )

        # Now bin the PSDs

        fmin = 1 / stationary_period
        fmax = fsample / 2

        my_binned_psds1, my_times1, binfreq10 = self.bin_psds(my_psds1, fmin, fmax)
        if self.nsum > 1:
            my_binned_psds2, _, binfreq20 = self.bin_psds(my_psds2, fmin, fmax)

        log.debug_rank(
            "Bin PSDs",
            comm=obs.comm.comm_group,
            rank=0,
            timer=timer,
        )

        # concatenate

        if self.save_cov:
            my_cov = my_cov1  # Only store the fully sampled covariance

        if binfreq10 is None:
            my_times = []
            my_binned_psds = []
            binfreq0 = None
        else:
            my_times = my_times1
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
        if obs.comm_row is None:
            have_bins_all = [have_bins]
        else:
            have_bins_all = obs.comm_row.allgather(have_bins)
        root = 0
        if np.any(have_bins_all):
            while not have_bins_all[root]:
                root += 1
        else:
            msg = "None of the processes have valid PSDs"
            raise RuntimeError(msg)
        binfreq = None
        if obs.comm_row is None:
            binfreq = binfreq0
        else:
            binfreq = obs.comm_row.bcast(binfreq0, root=root)
        if binfreq0 is not None and np.any(binfreq != binfreq0):
            msg = (
                f"{obs.comm.world_rank:4} : Binned PSD frequencies change. "
                f"len(binfreq0) = {binfreq0.size}, "
                f"len(binfreq) = {binfreq.size}, binfreq0={binfreq0}, "
                f"binfreq = {binfreq}. len(my_psds) = {len(my_psds1)}"
            )
            raise RuntimeError(msg)

        if len(my_times) != len(my_binned_psds):
            msg = (
                f"ERROR: Process {obs.comm.world_rank} has len(my_times) = "
                f"{len(my_times)}, "
                f"len(my_binned_psds) = {len(my_binned_psds)}"
            )
            raise RuntimeError(msg)

        all_times = None
        all_psds = None
        if obs.comm_row is None:
            all_times = [my_times]
            all_psds = [my_binned_psds]
        else:
            all_times = obs.comm_row.gather(my_times, root=0)
            all_psds = obs.comm_row.gather(my_binned_psds, root=0)
        all_cov = None
        if self.save_cov:
            if obs.comm_row is None:
                all_cov = [my_cov]
            else:
                all_cov = obs.comm_row.gather(my_cov, root=0)

        log.debug_rank(
            "Collect PSDs",
            comm=obs.comm.comm_group,
            rank=0,
            timer=timer,
        )

        final_freqs = None
        final_psd = None
        if obs.comm.group_rank == 0:
            if len(all_times) != len(all_psds):
                msg = (
                    f"ERROR: Process {obs.comm.world_rank} has len(all_times) = "
                    f"{len(all_times)},"
                    f" len(all_psds) = {len(all_psds)} before deglitch"
                )
                raise RuntimeError(msg)

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
                    f"ERROR: Process {obs.comm.world_rank} has len(all_times) = "
                    f"{len(all_times)}, "
                    f"len(all_psds) = {len(all_psds)} AFTER deglitch"
                )
                raise RuntimeError(msg)

            all_times = list(np.hstack(all_times))
            all_psds = list(np.hstack(all_psds))
            if self.save_cov:
                all_cov = list(np.hstack(all_cov))

            good_psds, good_times, nbad, good_cov = self.discard_outliers(
                binfreq, all_psds, all_times, all_cov
            )
            log.debug_rank("Discard outliers", timer=timer)

            if self.output_dir is not None:
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

            final_freqs = binfreq
            final_psd = np.mean(np.array(good_psds), axis=0)
            log.debug_rank("Write PSDs", timer=timer)

        return final_freqs, final_psd

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
        if self.out_model is not None:
            prov["meta"].append(self.out_model)
        return prov
