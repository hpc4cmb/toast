# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u
from scipy import interpolate

from .. import rng
from .._libtoast import tod_sim_noise_timestream, tod_sim_noise_timestream_batch
from ..fft import FFTPlanReal1DStore
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Unicode, Unit, trait_docs
from ..utils import AlignedF64, Logger, rate_from_times, unit_conversion
from .operator import Operator


@function_timer
def sim_noise_timestream(
    realization=0,
    telescope=0,
    component=0,
    sindx=0,
    detindx=0,
    rate=1.0,
    firstsamp=0,
    samples=0,
    oversample=2,
    freq=None,
    psd=None,
    py=False,
):
    """Generate a noise timestream, given a starting RNG state.

    Use the RNG parameters to generate unit-variance Gaussian samples
    and then modify the Fourier domain amplitudes to match the desired
    PSD.

    The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
    which each consist of two unsigned 64bit integers.  These four
    numbers together uniquely identify a single sample.  We construct
    those four numbers in the following way:

    key1 = realization * 2^32 + telescope * 2^16 + component
    key2 = sindx * 2^32 + detindx
    counter1 = currently unused (0)
    counter2 = sample in stream

    counter2 is incremented internally by the RNG function as it calls
    the underlying Random123 library for each sample.

    Args:
        realization (int): the Monte Carlo realization.
        telescope (int): a unique index assigned to a telescope.
        component (int): a number representing the type of timestream
            we are generating (detector noise, common mode noise,
            atmosphere, etc).
        sindx (int): the global index of this observing session.
        detindx (int): the global index of this detector.
        rate (float): the sample rate.
        firstsamp (int): the start sample in the stream.
        samples (int): the number of samples to generate.
        oversample (int): the factor by which to expand the FFT length
            beyond the number of samples.
        freq (array): the frequency points of the PSD.
        psd (array): the PSD values.
        py (bool): if True, use a pure-python implementation.  This is useful
            for testing.  If True, also return the interpolated PSD.

    Returns:
        (AlignedF64):  the noise timestream.  If py=True, returns a tuple of timestream,
            interpolated frequencies, and interpolated PSD.

    """
    tdata = AlignedF64(samples)
    if py:
        fftlen = 2
        while fftlen <= (oversample * samples):
            fftlen *= 2
        npsd = fftlen // 2 + 1
        norm = rate * float(npsd - 1)

        interp_freq = np.fft.rfftfreq(fftlen, 1 / rate)
        if interp_freq.size != npsd:
            raise RuntimeError(
                "interpolated PSD frequencies do not have expected length"
            )

        # Ensure that the input frequency range includes all the frequencies
        # we need.  Otherwise the extrapolation is not well defined.

        if np.amin(freq) < 0.0:
            raise RuntimeError("input PSD frequencies should be >= zero")

        if np.amin(psd) < 0.0:
            raise RuntimeError("input PSD values should be >= zero")

        increment = rate / fftlen

        if freq[0] > increment:
            raise RuntimeError(
                "input PSD does not go to low enough frequency to "
                "allow for interpolation"
            )

        nyquist = rate / 2
        if np.abs((freq[-1] - nyquist) / nyquist) > 0.01:
            raise RuntimeError(
                "last frequency element does not match Nyquist "
                "frequency for given sample rate: {} != {}".format(freq[-1], nyquist)
            )

        # Perform a logarithmic interpolation.  In order to avoid zero values, we
        # shift the PSD by a fixed amount in frequency and amplitude.

        psdshift = 0.01 * np.amin(psd[(psd > 0.0)])
        freqshift = increment

        loginterp_freq = np.log10(interp_freq + freqshift)
        logfreq = np.log10(freq + freqshift)
        logpsd = np.log10(psd + psdshift)

        interp = interpolate.interp1d(
            logfreq, logpsd, kind="linear", fill_value="extrapolate"
        )

        loginterp_psd = interp(loginterp_freq)
        interp_psd = np.power(10.0, loginterp_psd) - psdshift

        # Zero out DC value

        interp_psd[0] = 0.0

        scale = np.sqrt(interp_psd * norm)

        # gaussian Re/Im randoms, packed into a complex valued array

        key1 = (
            int(realization) * int(4294967296)
            + int(telescope) * int(65536)
            + int(component)
        )
        key2 = int(sindx) * int(4294967296) + int(detindx)
        counter1 = 0
        counter2 = firstsamp * oversample

        rngdata = rng.random(
            fftlen, sampler="gaussian", key=(key1, key2), counter=(counter1, counter2)
        ).array()

        fdata = np.zeros(npsd, dtype=np.complex128)

        # Set the DC and Nyquist frequency imaginary part to zero
        fdata[0] = rngdata[0] + 0.0j
        fdata[-1] = rngdata[npsd - 1] + 0.0j

        # Repack the other values.
        fdata[1:-1] = rngdata[1 : npsd - 1] + 1j * rngdata[-1 : npsd - 1 : -1]

        # scale by PSD
        fdata *= scale

        # inverse FFT
        tempdata = np.fft.irfft(fdata)

        # subtract the DC level- for just the samples that we are returning
        offset = (fftlen - samples) // 2

        DC = np.mean(tempdata[offset : offset + samples])
        tdata[:] = tempdata[offset : offset + samples] - DC
        return (tdata, interp_freq, interp_psd)
    else:
        tod_sim_noise_timestream(
            realization,
            telescope,
            component,
            sindx,
            detindx,
            rate,
            firstsamp,
            oversample,
            freq.astype(np.float64),
            psd.astype(np.float64),
            tdata,
        )
        return tdata


@trait_docs
class SimNoise(Operator):
    """Operator which generates noise timestreams.

    This passes through each observation and every process generates data
    for its assigned samples.  The observation unique ID is used in the random
    number generation.

    This operator intentionally does not provide a "view" trait.  To avoid
    discontinuities, the full observation must be simulated regardless of any data
    views that will be used for subsequent analysis.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    realization = Int(0, help="The noise realization index")

    component = Int(0, help="The noise component index")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating noise timestreams",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    serial = Bool(True, help="Use legacy serial implementation instead of batched")

    @traitlets.validate("realization")
    def _check_realization(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("realization index must be positive")
        return check

    @traitlets.validate("component")
    def _check_component(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("component index must be positive")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._oversample = 2

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Unique session ID
            sindx = ob.session.uid

            # Telescope UID
            telescope = ob.telescope.uid

            if self.noise_model not in ob:
                msg = "Observation does not contain noise model key '{}'".format(
                    self.noise_model
                )
                log.error(msg)
                raise KeyError(msg)

            nse = ob[self.noise_model]

            # Eventually we'll redistribute, to allow long correlations...
            if ob.comm_row_size != 1:
                msg = "Noise simulation for process grids with multiple ranks in the sample direction not implemented"
                log.error(msg)
                raise NotImplementedError(msg)

            # The previous code verified that a single process has whole
            # detectors within the observation...

            # Make sure correct output exists
            exists = ob.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )

            # The units of the output timestream
            data_units = ob.detdata[self.det_data].units

            # The target units of the PSD needed to produce the timestream units
            sim_units = data_units**2 * u.second

            # Get the sample rate from the data.  We also have nominal sample rates
            # from the noise model and also from the focalplane.  Perhaps we should add
            # a check here that they are all consistent.
            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(
                ob.shared[self.times].data
            )

            if self.serial:
                # Original serial implementation (for testing / comparison)
                for key in nse.all_keys_for_dets(dets):
                    # Simulate the noise matching this key
                    nsedata = sim_noise_timestream(
                        realization=self.realization,
                        telescope=telescope,
                        component=self.component,
                        sindx=sindx,
                        detindx=nse.index(key),
                        rate=rate,
                        firstsamp=ob.local_index_offset,
                        samples=ob.n_local_samples,
                        oversample=self._oversample,
                        freq=nse.freq(key).to_value(u.Hz),
                        psd=nse.psd(key).to_value(sim_units),
                        py=False,
                    )

                    # Add the noise to all detectors that have nonzero weights
                    for det in dets:
                        weight = nse.weight(det, key)
                        if weight == 0:
                            continue
                        ob.detdata[self.det_data][det] += weight * nsedata.array()

                    nsedata.clear()
                    del nsedata

                # Release the work space allocated in the FFT plan store.
                store = FFTPlanReal1DStore.get()
                store.clear()
            else:
                # Build up the list of noise stream indices and verify that the
                # frequency data for all psds is consistent.
                strm_names = list()
                freq_zero = nse.freq(nse.keys[0])
                for ikey, key in enumerate(nse.keys):
                    weight = 0.0
                    for det in dets:
                        weight += np.abs(nse.weight(det, key))
                    if weight == 0:
                        continue
                    test_freq = nse.freq(key)
                    if (
                        len(test_freq) != len(freq_zero)
                        or test_freq[0] != freq_zero[0]
                        or test_freq[-1] != freq_zero[-1]
                    ):
                        msg = "All psds must have the same frequency values"
                        log.error(msg)
                        raise RuntimeError(msg)
                    strm_names.append(key)

                freq = AlignedF64(len(freq_zero))
                freq[:] = freq_zero.to_value(u.Hz)

                strmindices = np.array(
                    [nse.index(x) for x in strm_names], dtype=np.uint64
                )

                psdbuf = AlignedF64(len(freq_zero) * len(strmindices))
                psds = psdbuf.array().reshape((len(strmindices), len(freq_zero)))
                for ikey, key in enumerate(strm_names):
                    psds[ikey][:] = nse.psd(key).to_value(sim_units)

                noisebuf = AlignedF64(ob.n_local_samples * len(strmindices))
                noise = noisebuf.array().reshape((len(strmindices), ob.n_local_samples))

                tod_sim_noise_timestream_batch(
                    self.realization,
                    telescope,
                    self.component,
                    sindx,
                    rate,
                    ob.local_index_offset,
                    self._oversample,
                    strmindices,
                    freq,
                    psds,
                    noise,
                )

                del psds
                psdbuf.clear()
                del psdbuf

                freq.clear()
                del freq

                # Add the noise to all detectors that have nonzero weights
                for ikey, key in enumerate(strm_names):
                    for det in dets:
                        weight = nse.weight(det, key)
                        if weight == 0:
                            continue
                        ob.detdata[self.det_data][det] += weight * noise[ikey]

                del noise
                noisebuf.clear()
                del noisebuf

                # Save memory by clearing the fft plans
                store = FFTPlanReal1DStore.get()
                store.clear()

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return {
            "meta": [
                self.noise_model,
            ],
            "shared": [
                self.times,
            ],
            "detdata": [
                self.det_data,
            ],
        }

    def _provides(self):
        return {
            "detdata": [
                self.det_data,
            ]
        }
