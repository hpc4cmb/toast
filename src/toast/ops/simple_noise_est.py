# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..noise import Noise
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Unicode, trait_docs
from ..utils import Logger, rate_from_times

from .operator import Operator
from .copy import Copy
from .detrend import Detrend

try:
    import finufft

    have_finufft = True
except ImportError:
    have_finufft = False


@trait_docs
class SimpleNoiseEstim(Operator):
    """Estimate the PSDs of detector data using Fourier domain autospectra.

    This uses a non-uniform FFT to handle flagged samples in the time domain.
    The low-frequency limit of the estimation is set by smallest interval
    used.  The estimated PSD is binned to average down noise.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode("noise_model", help="The output noise model to create")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for timestreams",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
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

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    view = Unicode(
        None,
        allow_none=True,
        help="Use these intervals to define uncorrelated time spans",
    )

    binned = Bool(True, help="Bin the estimated PSD")

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
        if not have_finufft:
            msg = "The SimpleNoiseEstim operator requires the finufft package."
            raise RuntimeError(msg)
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            msg = "SimpleNoiseEstim uses all detectors- ignoring input detector list"
            log.warning(msg)

        temp_data = f"{self.name}_temp_data"
        temp_flags = f"{self.name}_temp_flags"

        for ob in data.obs:
            n_samp = ob.n_local_samples
            local_dets = ob.select_local_detectors(flagmask=self.det_mask)
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[self.times].data
            )

            # If any independent intervals are too small, log a warning.
            warn_small = 2 * int(rate)

            # Copy the data to a temp location so that we can detrend
            obdata = data.select(obs_uid=ob.uid)
            Copy(
                detdata=[
                    (self.det_data, temp_data),
                    (self.det_flags, temp_flags),
                ]
            ).apply(obdata)

            # Detrend
            detrend_edge = ob.n_local_samples // 100
            Detrend(
                det_data=temp_data,
                det_flags=temp_flags,
                shared_flags=self.shared_flags,
                det_mask=self.det_mask,
                det_flag_mask=self.det_flag_mask,
                shared_flag_mask=self.shared_flag_mask,
                method="linear",
                edge_nsample_method="mean",
                edge_nsample=int(detrend_edge),
            ).apply(obdata)

            # Find the smallest independent interval
            min_len = n_samp
            ranges = list()
            for ivw, vw in enumerate(ob.intervals[self.view]):
                vlen = vw.last - vw.first
                print(f"DBG interval {ivw}: {vw.first} .. {vw.last}", flush=True)
                if vlen < warn_small:
                    msg = f"{ob.name} interval {ivw} is too small ({vlen} samples)"
                    msg += " this interval will be ignored for noise estimation"
                    log.warning(msg)
                    continue
                ranges.append((vw.first, vw.last))
                min_len = min(min_len, vlen)
            print(f"DBG min_len = {min_len}, ranges = {ranges}", flush=True)

            # Compute Fourier domain resolution
            order = int(np.ceil(np.log(min_len) / np.log(2)))
            n_fft = 2**order
            print(f"DBG order = {order}, n_fft = {n_fft}", flush=True)

            # Compute the binning.  We start with a small number of logarithmically
            # spaced bins.  We double the number of bins until we reach a point where
            # some bins at low frequency have no hits.
            raw_freq = np.fft.rfftfreq(n_fft, d=dt)
            bin_locs, bin_hits = self._get_binning(raw_freq)

            print(f"DBG raw_freq = {raw_freq}", flush=True)

            if self.binned:
                bin_freq = self._bin_data(bin_locs, bin_hits, raw_freq)
                print(
                    f"DBG binned: {len(bin_hits)} bins, {bin_locs}, {bin_hits}, freq = {bin_freq}",
                    flush=True,
                )
            else:
                bin_freq = raw_freq

            # Compute binned PSDs.
            psds = dict()

            for det in local_dets:
                if self.shared_flags is None:
                    flags = np.zeros(n_samp, dtype=np.uint8)
                else:
                    flags = np.copy(ob.shared[self.shared_flags].data)
                    flags &= self.shared_flag_mask
                if self.det_flags is not None:
                    flags |= ob.detdata[temp_flags][det] & self.det_flag_mask

                psds[det] = self._compute_det_psd(
                    rate,
                    n_fft,
                    bin_locs,
                    bin_hits,
                    ranges,
                    ob.shared[self.times].data,
                    ob.detdata[temp_data][det],
                    flags,
                )
                print(f"psd {det}: {len(psds[det])} bins, {psds[det]}", flush=True)

            del ob.detdata[temp_data]
            del ob.detdata[temp_flags]
            del obdata

            # Create the noise model
            units = ob.detdata[self.det_data].units ** 2 * u.second
            print(units)
            final_psds = {x: psds[x] * units for x in psds.keys()}
            ob[self.noise_model] = Noise(
                detectors=local_dets,
                freqs={x: bin_freq * u.Hz for x in local_dets},
                psds=final_psds,
            )

    def _compute_det_psd(self, rate, n_fft, bin_locs, bin_hits, ranges, times, tod, flags):
        if self.binned:
            n_bins = len(bin_hits)
        else:
            n_bins = n_fft // 2 + 1

        # Temp buffers, re-used for each view
        ftemp = np.empty(n_fft, dtype=np.complex128)
        temp_psd = np.zeros(n_fft // 2 + 1, dtype=np.float64)

        # Accumulated PSD (minus zero frequency)
        psd = np.zeros(n_bins, dtype=np.float64)

        n_range = len(ranges)
        for first, last in ranges:
            slc = slice(first, last, 1)
            nslc = last - first
            if flags is None:
                good = np.ones(nslc, dtype=bool)
            else:
                good = flags[slc] == 0
            n_good = np.count_nonzero(good)
            good_times = times[slc][good]
            good_data = tod[slc][good] + 1j * np.zeros(n_good)
            tnorm = np.sum((tod[slc][good]) ** 2)

            finufft.nufft1d1(
                good_times, good_data, n_modes=n_fft, out=ftemp, eps=1.0e-15
            )

            # Compute Power spectrum.  finufft Fourier domain values are indexed from
            # -N/2 to N/2 - 1 for even values of nfft.

            # DC
            temp_psd[0] = 0

            # Nyquist
            temp_psd[-1] = ftemp[0].real ** 2

            # Other frequencies.  The data is symmetric since the input is real,
            # so we just use one half.
            fmid = ftemp[n_fft // 2 + 1 : n_fft]
            temp_psd[1:-1] = 2.0 * (np.abs(fmid)).real ** 2

            # Normalize power spectrum
            pnorm = np.sum(temp_psd) / n_fft
            temp_psd *= tnorm / pnorm

            # Convert to a PSD
            temp_psd /= (1.85 * rate * n_fft)

            # Bin the PSD and accumulate.
            if self.binned:
                psd[:] += self._bin_data(bin_locs, bin_hits, temp_psd)
            else:
                psd[:] += temp_psd

        rg = 1 / n_range
        psd[:] *= rg
        del temp_psd
        del ftemp

        # Normalize PSD to ensure that Parseval's theorem holds
        #pnorm = np.sum(psd) / n_bins
        # pnorm = np.sum(psd) * rate / n_fft
        # norm = tnorm / pnorm
        # psd[:] *= norm
        return psd

    def _get_binning(self, freq):
        """Compute optional binning of the PSD."""
        # Exclude zero frequency
        good_freq = freq[1:]

        half = (good_freq[1] - good_freq[0]) / 2
        max_incr = 0.2 * freq[-1]

        # Set first bin to include a reasonable number of points
        first_bin_points = int(0.01 * len(freq))
        if first_bin_points == 0:
            first_bin_points = 1
        first_width = 2 * half * first_bin_points
        epsilon = 0.001 * half
        bounds = [good_freq[0] - epsilon, good_freq[0] + first_width + epsilon]

        cur = bounds[-1]
        incr = first_width
        while cur + incr < good_freq[-1]:
            bounds.append(cur + incr)
            cur += incr
            if incr < max_incr:
                # Increase the bin size up to some threshold
                incr *= np.e
            if cur + incr > good_freq[-1]:
                incr = good_freq[-1] - cur
        # Adjust last bin
        bounds[-1] = good_freq[-1] + epsilon
        for ib, bnd in enumerate(bounds):
            print(f"BND {ib}: {bnd}", flush=True)
        nbins = len(bounds)

        locs = np.digitize(good_freq, bounds).astype(np.int32)
        print(locs)
        hits = np.zeros(nbins, dtype=np.int32)
        for loc in locs:
            hits[loc] += 1
        # Trim lowest bin
        return locs - 1, hits[1:]

    def _bin_data(self, bin_locs, bin_hits, data):
        """Bin spectral domain data, cutting the zero frequency."""
        binned = np.zeros(bin_hits.size)
        for loc, val in zip(bin_locs, data[1:]):
            binned[loc] += val
        good_bins = bin_hits != 0
        binned = binned[good_bins] / bin_hits[good_bins]
        return binned

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.times],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": [self.noise_model],
            "shared": list(),
            "detdata": list(),
        }
        return prov
