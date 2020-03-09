# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import os

import numpy as np

import scipy.signal

import astropy.io.fits as pf

from .. import qarray as qa

from ..timing import Timer, function_timer

from ..todmap import MapSampler

from ..tod import flagged_running_average, Interval

from .psd_math import autocov_psd, crosscov_psd


class OpNoiseEstim:
    """Noise estimation operator.

    Args:
        signal(str):  Cache object name to analyze
        flags(str):  Cached flags to apply
        detmask(byte):  Flag bits to consider
        commonmask(byte):  Flag bits to consider
        out(str):  Output directory to write the PSDs to.
        maskfile(str):  FITS file name to read the mask from.
        mapfile(str):  FITS map to sample and subtract from the signal.
        pol(bool):  Sample also the polarized part of the map.
        nbin_psd(int):  Bin the resulting PSD.
        lagmax(int):  Maximum lag to consider for the covariance function.
        stationary_period(float):  Break the observation into several
            estimation periods of this length [s].
        nosingle(bool):  Do not evaluate individual PSDs.  Overridden by pairs.
        nocross(bool):  Do not evaluate cross-PSDs.  Overridden by pairs.
        calibrate_signal_estimate(bool):  Regress, not just subtract the
            signal estimate.
        nsum(int):  Downsampling factor for decimated data.
        naverage(int):  Smoothing kernel width for downsampled data.
        apply_intervals(bool):  If true, only measure the covariance
             within each interval
        pairs(iterable):  Detector pairs to estimate noise for.  Overrides
            nosingle and nocross.
        save_cov(bool):  Save also the sample covariance.

    """

    def __init__(
        self,
        signal=None,
        flags=None,
        detmask=1,
        commonmask=1,
        out=None,
        maskfile=None,
        mapfile=None,
        pol=True,
        nbin_psd=1000,
        lagmax=10000,
        stationary_period=86400.0,
        nosingle=False,
        nocross=True,
        calibrate_signal_estimate=False,
        nsum=10,
        naverage=100,
        apply_intervals=False,
        pairs=None,
        save_cov=False,
    ):
        self._signal = signal
        self._flags = flags
        self._detmask = detmask
        self._commonmask = commonmask
        self._out = out
        self._maskfile = maskfile
        self._mapfile = mapfile
        self._pol = pol
        self._nbin_psd = nbin_psd
        self._lagmax = lagmax
        self._stationary_period = stationary_period
        self._nosingle = nosingle
        self._nocross = nocross
        self._calibrate_signal_estimate = calibrate_signal_estimate
        self._apply_intervals = apply_intervals
        self._pairs = pairs
        # Parameters for downsampling the data
        self._nsum = nsum
        self._naverage = naverage
        self._save_cov = save_cov

    @function_timer
    def exec(self, data):
        comm = data.comm.comm_group

        masksampler = None
        if self._maskfile:
            masksampler = MapSampler(self._maskfile, comm=comm)
        mapsampler = None
        if self._mapfile:
            mapsampler = MapSampler(self._mapfile, comm=comm, pol=True)

        for obs in data.obs:
            tod = obs["tod"]
            if len(tod.local_dets) < len(tod.detectors):
                raise RuntimeError(
                    "Noise estimation does not work on " "detector-split data"
                )
            dets = {}
            for idet, det in enumerate(tod.detectors):
                dets[det] = idet
            ndet = len(dets)
            if self._pairs is not None:
                pairs = self._pairs
            else:
                # Construct a list of detector pairs
                pairs = []
                for idet1 in range(ndet):
                    det1 = tod.detectors[idet1]
                    for idet2 in range(idet1, ndet):
                        det2 = tod.detectors[idet2]
                        if det1 == det2 and self._nosingle:
                            continue
                        if det1 != det2 and self._nocross:
                            continue
                        pairs.append([det1, det2])
            timestamps = tod.local_times()
            commonflags = tod.local_common_flags()
            commonflags = commonflags & self._commonmask != 0
            fsample = 1 / np.median(np.diff(timestamps))

            if "name" in obs:
                fileroot = "noise_{}".format(obs["name"])
            elif "id" in obs:
                fileroot = "noise_{}".format(obs["id"])
            else:
                fileroot = "noise_{}".format(int(timestamps[0]))

            if self._apply_intervals:
                intervals = tod.local_intervals(obs["intervals"])
            else:
                intervals = [
                    Interval(
                        start=timestamps[0],
                        stop=timestamps[-1],
                        first=0,
                        last=timestamps.size - 1,
                    )
                ]

            self.subtract_signal(tod, comm, masksampler, mapsampler, intervals)

            # self.highpass_signal(tod, comm, intervals)

            # Extend the gap between intervals to prevent sample pairs
            # that cross the gap.

            gap_min = np.int(self._lagmax) + 1
            # Downsampled data requires longer gaps
            gap_min_nsum = np.int(self._lagmax * self._nsum) + 1
            offset, nsamp = tod.local_samples
            gapflags = np.zeros_like(commonflags)
            gapflags_nsum = np.zeros_like(commonflags)
            for ival1, ival2 in zip(intervals[:-1], intervals[1:]):
                gap_start = ival1.last + 1
                gap_stop = max(gap_start + gap_min, ival2.first)
                gap_stop_nsum = max(gap_start + gap_min_nsum, ival2.first)
                if gap_start < offset + nsamp and gap_stop > offset:
                    gap_start = max(0, gap_start - offset)
                    gap_stop = min(offset + nsamp, gap_stop - offset)
                    gapflags[gap_start:gap_stop] = True
                    gap_stop_nsum = min(offset + nsamp, gap_stop_nsum - offset)
                    gapflags_nsum[gap_start:gap_stop_nsum] = True

            # FIXME: This operator needs to handle situations where
            # det1 and det2 are not on the same process.  Then the
            # check at the top of the loop can be removed.

            for det1, det2 in pairs:
                if det1 not in dets or det2 not in dets:
                    # User-specified pair is invalid
                    continue
                signal1 = tod.local_signal(det1)
                flags1 = tod.local_flags(det1, name=self._flags)
                flags = flags1 & self._detmask != 0
                signal2 = None
                flags2 = None
                if det1 != det2:
                    signal2 = tod.local_signal(det2)
                    flags2 = tod.local_flags(det2, name=self._flags)
                    flags[flags2 & self._detmask != 0] = True
                flags[commonflags] = True

                self.process_noise_estimate(
                    signal1,
                    signal2,
                    flags,
                    gapflags,
                    gapflags_nsum,
                    timestamps,
                    fsample,
                    comm,
                    fileroot,
                    det1,
                    det2,
                    intervals,
                )

        return

    def highpass_signal(self, tod, comm, intervals):
        """ Suppress the sub-harmonic modes in the TOD by high-pass
        filtering.
        """
        timer = Timer()
        timer.start()
        rank = 0
        if comm is not None:
            rank = comm.rank
        if rank == 0:
            print("High-pass-filtering signal", flush=True)
        for det in tod.local_dets:
            signal = tod.local_signal(det, name=self._signal)
            flags = tod.local_flags(det, name=self._flags)
            flags &= self._detmask
            for ival in intervals:
                ind = slice(ival.first, ival.last + 1)
                sig = signal[ind]
                flg = flags[ind]
                trend = flagged_running_average(
                    sig, flg, self._lagmax, return_flags=False
                )
                sig -= trend
        if comm is not None:
            comm.barrier()
        if rank == 0:
            timer.report_clear("TOD high pass")
        return

    def subtract_signal(self, tod, comm, masksampler, mapsampler, intervals):
        """ Subtract a signal estimate from the TOD and update the
        flags for noise estimation.
        """
        if mapsampler is None and masksampler is None:
            return
        timer = Timer()
        timer.start()
        rank = 0
        if comm is not None:
            rank = comm.rank
        if rank == 0:
            print("Subtracting signal", flush=True)
        for det in tod.local_dets:
            if det.endswith("-diff") and not self._pol:
                continue
            # if comm.rank == 0:
            #    print('Subtracting signal for {}'.format(det), flush=True)
            #    tod.cache.report()
            epsilon = 0  # FIXME: Where can we get this for every detector?
            eta = (1 - epsilon) / (1 + epsilon)
            signal = tod.local_signal(det, name=self._signal)
            flags = tod.local_flags(det, name=self._flags)
            flags &= self._detmask
            try:
                quats = tod.local_pointing(det)
            except AttributeError:
                quats = None
            if quats is None:
                continue
            iquweights = tod.local_weights(det)
            for ival in intervals:
                ind = slice(ival.first, ival.last + 1)
                sig = signal[ind]
                flg = flags[ind]
                quat = quats[ind]
                theta, phi = qa.to_position(quat)
                iquw = iquweights[ind, :]
                if masksampler is not None:
                    maskflg = masksampler.at(theta, phi) < 0.5
                    flg[maskflg] |= 255
                if mapsampler is not None:
                    if self._pol:
                        bg = mapsampler.atpol(theta, phi, iquw)
                    else:
                        bg = mapsampler.at(theta, phi) * iquw[:, 0]
                    if self._calibrate_signal_estimate:
                        good = flg == 0
                        ngood = np.sum(good)
                        if ngood > 1:
                            templates = np.vstack([np.ones(ngood), bg[good]])
                            invcov = np.dot(templates, templates.T)
                            cov = np.linalg.inv(invcov)
                            proj = np.dot(templates, sig[good])
                            coeff = np.dot(cov, proj)
                            bg = coeff[0] + coeff[1] * bg
                    sig -= bg
        if comm is not None:
            comm.barrier()
        if rank == 0:
            timer.report_clear("TOD signal subtraction")
        return

    def decimate(self, x, flg, gapflg, intervals):
        # Low-pass filter with running average, then downsample
        xx = x.copy()
        flags = flg.copy()
        for ival in intervals:
            ind = slice(ival.first, ival.last + 1)
            xx[ind], flags[ind] = flagged_running_average(
                x[ind], flg[ind], self._naverage, return_flags=True
            )
        return xx[:: self._nsum].copy(), (flags + gapflg)[:: self._nsum].copy()

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

    def log_bin(self, freq, nbin=100, fmin=None, fmax=None):
        if np.any(freq == 0):
            raise Exception("Logarithmic binning should not include " "zero frequency")

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

    def bin_psds(self, my_psds, fmin=None, fmax=None):
        my_binned_psds = []
        my_times = []
        binfreq0 = None

        for i in range(len(my_psds)):
            t0, _, freq, psd = my_psds[i]

            good = freq != 0

            if self._nbin_psd is not None:
                locs, hits = self.log_bin(
                    freq[good], nbin=self._nbin_psd, fmin=fmin, fmax=fmax
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
                    raise Exception("Binned PSD frequencies change")

            if self._nbin_psd is not None:
                binpsd = np.zeros(hits.size)
                for loc, p in zip(locs, psd[good]):
                    binpsd[loc] += p
                binpsd = binpsd[hits != 0] / hits[hits != 0]
            else:
                binpsd = psd

            my_times.append(t0)
            my_binned_psds.append(binpsd)
        return my_binned_psds, my_times, binfreq0

    def discard_outliers(self, binfreq, all_psds, all_times, all_cov):
        all_psds = copy.deepcopy(all_psds)
        all_times = copy.deepcopy(all_times)
        if self._save_cov:
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
                if self._save_cov:
                    del all_cov[i]
                nrow -= 1
                nempty += 1
            else:
                i += 1

        if nempty > 0:
            print("Discarded {} empty or NaN psds".format(nempty), flush=True)

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
                    if self._save_cov:
                        del all_cov[ii]

            if nbad > 0:
                print("Masked extra {} psds due to outliers." "".format(nbad))
        return all_psds, all_times, nempty + nbad, all_cov

    def save_psds(
        self, binfreq, all_psds, all_times, det1, det2, fsample, rootname, all_cov
    ):
        timer = Timer()
        timer.start()
        if det1 == det2:
            fn_out = os.path.join(self._out, "{}_{}.fits".format(rootname, det1))
        else:
            fn_out = os.path.join(
                self._out, "{}_{}_{}.fits".format(rootname, det1, det2)
            )
        all_psds = np.vstack([binfreq, all_psds])

        hdulist = [pf.PrimaryHDU()]

        cols = []
        cols.append(pf.Column(name="OBT", format="D", array=all_times))
        coldefs = pf.ColDefs(cols)
        hdu1 = pf.BinTableHDU.from_columns(coldefs)
        hdu1.header["RATE"] = fsample, "Sampling rate"
        hdulist.append(hdu1)

        cols = []
        cols.append(
            pf.Column(name="PSD", format="{}E".format(binfreq.size), array=all_psds)
        )
        coldefs = pf.ColDefs(cols)
        hdu2 = pf.BinTableHDU.from_columns(coldefs)
        hdu2.header["EXTNAME"] = str(det1), "Detector"
        hdu2.header["DET1"] = str(det1), "Detector1"
        hdu2.header["DET2"] = str(det2), "Detector2"
        hdulist.append(hdu2)
        timer.report_clear("Create header")

        if self._save_cov:
            all_cov = np.array(all_cov)
            cols = []
            nrow, ncol, nsamp = np.shape(all_cov)
            cols.append(
                pf.Column(
                    name="HITS",
                    format="{}J".format(nsamp),
                    array=np.ascontiguousarray(all_cov[:, 0, :]),
                )
            )
            cols.append(
                pf.Column(
                    name="COV",
                    format="{}E".format(nsamp),
                    array=np.ascontiguousarray(all_cov[:, 1, :]),
                )
            )
            coldefs = pf.ColDefs(cols)
            hdu3 = pf.BinTableHDU.from_columns(coldefs)
            hdu3.header["EXTNAME"] = str(det1), "Detector"
            hdu3.header["DET1"] = str(det1), "Detector1"
            hdu3.header["DET2"] = str(det2), "Detector2"
            hdulist.append(hdu3)
            timer.report_clear("Append covariance")

        hdulist = pf.HDUList(hdulist)
        timer.report_clear("Assemble HDU list")

        with open(fn_out, "wb") as fits_out:
            hdulist.writeto(fits_out, overwrite=True)
        timer.report_clear("Write PSD")

        print("Detector {} vs. {} PSDs stored in {}".format(det1, det2, fn_out))
        return

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

        timestamps_decim = timestamps[:: self._nsum]
        # decimate() will smooth and downsample the signal in
        # each valid interval separately
        signal1_decim, flags_decim = self.decimate(
            signal1, flags, gapflags_nsum, local_intervals
        )
        if signal2 is not None:
            signal2_decim, flags_decim = self.decimate(
                signal2, flags, gapflags_nsum, local_intervals
            )

        if signal2 is None:
            result = autocov_psd(
                timestamps_decim,
                signal1_decim,
                flags_decim,
                min(self._lagmax, timestamps_decim.size),
                self._stationary_period,
                fsample / self._nsum,
                comm=comm,
                return_cov=self._save_cov,
            )
        else:
            result = crosscov_psd(
                timestamps_decim,
                signal1_decim,
                signal2_decim,
                flags_decim,
                min(self._lagmax, timestamps_decim.size),
                self._stationary_period,
                fsample / self._nsum,
                comm=comm,
                return_cov=self._save_cov,
            )
        if self._save_cov:
            my_psds2, my_cov2 = result
        else:
            my_psds2, my_cov2 = result, None

        # Ensure the two sets of PSDs are of equal length

        my_new_psds1 = []
        my_new_psds2 = []
        if self._save_cov:
            my_new_cov1 = []
            my_new_cov2 = []
        i = 0
        while i < min(len(my_psds1), len(my_psds2)):
            t1 = my_psds1[i][0]
            t2 = my_psds2[i][0]
            if np.isclose(t1, t2):
                my_new_psds1.append(my_psds1[i])
                my_new_psds2.append(my_psds2[i])
                if self._save_cov:
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
        if self._save_cov:
            my_cov1 = my_new_cov1
            my_cov2 = my_new_cov2

        if len(my_psds1) != len(my_psds2):
            while my_psds1[-1][0] > my_psds2[-1][0]:
                del my_psds1[-1]
                if self._save_cov:
                    del my_cov1[-1]
            while my_psds1[-1][0] < my_psds2[-1][0]:
                del my_psds2[-1]
                if self._save_cov:
                    del my_cov2[-1]
        return my_psds1, my_cov1, my_psds2, my_cov2

    def process_noise_estimate(
        self,
        signal1,
        signal2,
        flags,
        gapflags,
        gapflags_nsum,
        timestamps,
        fsample,
        comm,
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

        rank = 0
        if comm is not None:
            rank = comm.rank

        timer = Timer()
        timer.start()
        if signal2 is None:
            result = autocov_psd(
                timestamps,
                signal1,
                flags + gapflags,
                self._lagmax,
                self._stationary_period,
                fsample,
                comm=comm,
                return_cov=self._save_cov,
            )
        else:
            result = crosscov_psd(
                timestamps,
                signal1,
                signal2,
                flags + gapflags,
                self._lagmax,
                self._stationary_period,
                fsample,
                comm=comm,
                return_cov=self._save_cov,
            )
        if self._save_cov:
            my_psds1, my_cov1 = result
        else:
            my_psds1, my_cov1 = result, None

        if self._nsum > 1:
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

        if rank == 0:
            timer.report_clear("Compute Correlators and PSDs")

        # Now bin the PSDs

        fmin = 1 / self._stationary_period
        fmax = fsample / 2

        my_binned_psds1, my_times1, binfreq10 = self.bin_psds(my_psds1, fmin, fmax)
        if self._nsum > 1:
            my_binned_psds2, _, binfreq20 = self.bin_psds(my_psds2, fmin, fmax)
        if rank == 0:
            timer.report_clear("Bin PSDs")

        # concatenate

        if binfreq10 is None:
            my_times = []
            my_binned_psds = []
            binfreq0 = None
        else:
            my_times = my_times1
            if self._save_cov:
                my_cov = my_cov1  # Only store the fully sampled covariance
            if self._nsum > 1:
                # frequencies that are usable in the down-sampled PSD
                fcut = fsample / 2 / self._naverage / 100
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
            raise RuntimeError("None of the processes have valid PSDs")
        binfreq = None
        if comm is None:
            binfreq = binfreq0
        else:
            binfreq = comm.bcast(binfreq0, root=root)
        if binfreq0 is not None and np.any(binfreq != binfreq0):
            raise Exception(
                "{:4} : Binned PSD frequencies change. len(binfreq0)={}"
                ", len(binfreq)={}, binfreq0={}, binfreq={}. "
                "len(my_psds)={}".format(
                    rank, binfreq0.size, binfreq.size, binfreq0, binfreq, len(my_psds1)
                )
            )
        if len(my_times) != len(my_binned_psds):
            raise Exception(
                "ERROR: Process {} has len(my_times) = {}, len(my_binned_psds)"
                " = {}".format(rank, len(my_times), len(my_binned_psds))
            )
        all_times = None
        all_psds = None
        if comm is None:
            all_times = [my_times]
            all_psds = [my_binned_psds]
        else:
            all_times = comm.gather(my_times, root=0)
            all_psds = comm.gather(my_binned_psds, root=0)
        all_cov = None
        if self._save_cov:
            if comm is None:
                all_cov = [my_cov]
            else:
                all_cov = comm.gather(my_cov, root=0)
        if rank == 0:
            timer.report_clear("Collect PSDs")

        if rank == 0:
            # FIXME: original code had no timing report here for the previous block.
            # Was one intended?
            if len(all_times) != len(all_psds):
                raise Exception(
                    "ERROR: Process {} has len(all_times) = {}, len(all_psds)"
                    " = {} before deglitch".format(rank, len(all_times), len(all_psds))
                )
            # De-glitch the binned PSDs and write them to file
            i = 0
            while i < len(all_times):
                if len(all_times[i]) == 0:
                    del all_times[i]
                    del all_psds[i]
                    if self._save_cov:
                        del all_cov[i]
                else:
                    i += 1

            if len(all_times) != len(all_psds):
                raise Exception(
                    "ERROR: Process {} has len(all_times) = {}, len(all_psds)"
                    " = {} AFTER deglitch".format(rank, len(all_times), len(all_psds))
                )

            all_times = list(np.hstack(all_times))
            all_psds = list(np.hstack(all_psds))
            if self._save_cov:
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
