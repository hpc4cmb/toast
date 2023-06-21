# Copyright (c) 2019-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u
from scipy import interpolate, signal

from .. import rng
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


class InjectCosmicRays(Operator):
    """
    Inject the cosmic rays  signal into the TOD. So far we inject two kinds of cosmic ray noise:

    Wafer noise, due to ~400 impacts per second in the wafer   undistinguishable  individually.
    For each observation and for each detector we inject low noise component as a _white noise signal_,
    i.e. noraml distributed  random samples following the observed properties from simulations and read from disc.
    This component is then coadded to the sky signal (as if it were a noise term) .

    Common mode noise

    A common mode  noise within each detector pair can be simulated given the properties of the wafer noise.
    Will use the informations of correlations can be found in the file provided as an input to the simulations, if
    present, otherwise 50% detector correlation is assumed.

    Direct  hits (or Glitches)

    Given the size of the detector we can derive the cosmic ray event rate and simulate the profile of a cosmic ray glitch.
    We assume the glitch to be described as

    .. math::
    \gamma (t) = C_1 +C_2 e^{-t/\tau }

    where :math:C_1 and :math:C_2 and the time constant :math:\tau  are drawn from a distribution of estimated values
    from simulations.   For each observation and each detector, we estimate the number of hits expected
    theroretically and draw a random integer, `N`,  with a Poissonian distribution given the expected number
    of events, `Nexp`.  We then  select randomly `N` timestamps where   the hits will be injected into the tod simulated in TOAST.
    Evaluate the function :math:\gamma at a higher sampling rate (~150 Hz), decimate it to the TOD sample rate and coadd  it.

    Args:
        crfile (string):  A `*.npz`  file encoding 4 attributes,
            `low_noise` (mean and std. dev. of  the wafer noise)
            `sampling_rate` sampling rate of the glitch simulations
            `direct_hits` distribution of the glitch parameters
            `correlation_matrix` correlation matrix for common mode
            must have a tag {detector} that will be replaced with the detector index.
        signal_name (string): the cache reference of the TOD data where the cosmic ray will be stored
        realization (int): to run several    Monte-Carlo realizations of cosmic ray noise
        eventrate (float) : the expected event rate of hits in a detector
        inject_direct_hits (bool): will include also direct hits if set to True
        conversion_factor (float): factor to convert the cosmic ray units to temperature units
        common_mode (bool) :  will include also common mode per pixel pair  if set to True
    """

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key to inject the gain drift"
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    crfile = Unicode(None, help="Path to the *.npz file encoding cosmic ray infos")

    crdata_units = Unit(u.W, help="The units of the input amplitudes")

    realization = Int(0, help="integer to set a different random seed ")

    eventrate = Float(
        0.0015,
        help="the expected event rate of hits in a detector",
    )

    inject_direct_hits = Bool(False, help="inject  direct hits as glitches in the TODs")

    conversion_factor = Quantity(
        1 * u.K / u.W,
        help="factor to convert the cosmic ray signal (usually Watts) into temperature units",
    )

    include_common_mode = Bool(
        False, help="will include also common mode per pixel pair  if set to True"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_cosmic_ray_data(self, filename):
        data_dic = np.load(filename)

        return data_dic

    def resample_cosmic_ray_statistics(self, arr, Nresamples, key, counter):
        resampled = np.zeros((Nresamples, arr.shape[1]))

        for ii in range(arr.shape[1]):
            # Resample by considering the bulk of  the Parameter distribution ~2sigma central interval
            bins = np.linspace(
                np.quantile(
                    arr[:, ii],
                    0.025,
                ),
                np.quantile(arr[:, ii], 0.975),
                30,
            )

            binned, edges = np.histogram(arr[:, ii], bins=bins)

            xb = 0.5 * (edges[:-1] + edges[1:])
            CDF = np.cumsum(binned) / binned.sum()

            pinv = interpolate.interp1d(CDF, xb, fill_value="extrapolate")
            # r = np.random.rand(Nresamples)
            r = rng.random(
                Nresamples,
                sampler="uniform_01",
                key=key,
                counter=counter,
            )

            resampled[:, ii] = pinv(r)

        return resampled

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        if self.crfile is None:
            raise AttributeError(
                "OpInjectCosmicRays cannot run if you don't provide cosmic ray data."
            )
        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            comm = ob.comm.comm_group
            rank = ob.comm.group_rank
            # Make sure detector data output exists
            exists = ob.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )
            sindx = ob.session.uid
            telescope = ob.telescope.uid
            focalplane = ob.telescope.focalplane
            size = ob.detdata[self.det_data][dets[0]].size
            samplerate = focalplane.sample_rate.to_value(u.Hz)

            obstime_seconds = size / samplerate
            n_events_expected = self.eventrate * obstime_seconds
            key1 = self.realization * 4294967296 + telescope * 65536
            counter2 = 0

            for kk, det in enumerate(dets):
                detindx = focalplane[det]["uid"]
                key2 = sindx
                counter1 = detindx

                rngdata = rng.random(
                    size,
                    sampler="gaussian",
                    key=(key1, key2),
                    counter=(counter1, counter2),
                )
                counter2 += size
                filename = self.crfile.replace("detector", f"det{kk}")
                data_dic = self.load_cosmic_ray_data(filename)
                lownoise_params = data_dic["low_noise"]
                var_tot = lownoise_params[1] ** 2
                if not self.include_common_mode:
                    lownoise_hits = lownoise_params[1] * rngdata + lownoise_params[0]
                    tmparray = lownoise_hits
                else:
                    if kk % 2 != 0:  # if kk is odd
                        detid_common = kk - 1
                        kkcol = kk - 1
                    else:  # kk even
                        detid_common = kk
                        kkcol = kk + 1

                    filename_common = self.crfile.replace(
                        "detector", f"det{detid_common}"
                    )
                    data_common = self.load_cosmic_ray_data(filename_common)
                    try:
                        corr_matr = data_common["correlation_matrix"]
                        corr_frac = corr_matr[kk, kkcol]

                    except KeyError:
                        log.warning(
                            "Correlation matrix not provided for common mode, assuming 50% correlation "
                        )
                        corr_frac = 0.5

                    var_corr = corr_frac * data_common["low_noise"][1] ** 2
                    var0 = var_tot - var_corr

                    rngdata_common = rng.random(
                        size,
                        sampler="gaussian",
                        key=(key1, key2),
                        counter=(detid_common, counter2),
                    )
                    counter2 += size
                    cr_common_mode = (
                        np.sqrt(var_corr) * rngdata_common + data_common["low_noise"][0]
                    )

                    lownoise_hits = lownoise_params[1] * rngdata + lownoise_params[0]
                    tmparray = lownoise_hits + cr_common_mode

                if self.inject_direct_hits:
                    glitches_param_distr = data_dic["direct_hits"]
                    fsampl_sims = (data_dic["sampling_rate"][0] * u.Hz).value

                    glitch_seconds = 0.15  # seconds, i.e. ~ 3samples at 19Hz
                    # we approximate the number of samples to the closest integer
                    nsamples_high = int(np.around(glitch_seconds * fsampl_sims))
                    nsamples_low = int(np.around(glitch_seconds * samplerate))
                    # import pdb; pdb.set_trace()
                    # np.random.seed( obsindx//1e3  +detindx//1e3 )
                    n_events = np.random.poisson(n_events_expected)
                    params = self.resample_cosmic_ray_statistics(
                        glitches_param_distr,
                        Nresamples=n_events,
                        key=(key1, key2),
                        counter=(counter1, counter2),
                    )
                    counter2 += n_events
                    # draw n_events uniformly from a continuous distribution
                    # you want the events to happen during one observation
                    # we also make sure that the glitch is injected at most
                    # `glitch_seconds` before the end of the observation ,
                    # otherwise we've problems in downsampling
                    rngunif = rng.random(
                        n_events,
                        sampler="uniform_01",
                        key=(key1, key2),
                        counter=(counter1, counter2),
                    )
                    counter2 += n_events

                    time_glitches = (obstime_seconds - glitch_seconds) * rngunif
                    assert time_glitches.max() < obstime_seconds

                    # otherwise we've problems in downsampling

                    # estimate the timestamps rounding off the events in seconds
                    time_stamp_glitches = np.around(time_glitches * samplerate).astype(
                        np.int64
                    )
                    # we measure the glitch and the bestfit timeconstant in millisec
                    tglitch = np.linspace(0, glitch_seconds * 1e3, nsamples_high)
                    glitch_func = lambda t, C1, C2, tau: C1 + (C2 * np.exp(-t / tau))
                    for i in range(n_events):
                        tmphit = glitch_func(tglitch, *params[i])
                        tmparray[
                            time_stamp_glitches[i] : time_stamp_glitches[i]
                            + nsamples_low
                        ] = signal.resample(tmphit, num=nsamples_low, t=tglitch)[0]
                tmparray = tmparray * self.crdata_units
                ob.detdata[self.det_data][det] += (
                    self.conversion_factor * tmparray
                ).to_value(self.det_data_units)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [
                self.boresight,
            ],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [
                self.det_data,
            ],
        }
        return prov
