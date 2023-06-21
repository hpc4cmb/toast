# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from .. import rng
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Callable, Float, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger
from .operator import Operator
from .sim_tod_noise import sim_noise_timestream


@trait_docs
class GainDrifter(Operator):
    """Operator which injects gain drifts to the signal.

    The drift can be injected into 3 different ways:
    - `linear_drift`: inject a linear drift   with a random slope  for each detector
    - `slow_drift`: inject a drift signal with a `1/f` PSD, simulated up to
    the  frequencies<`cutoff_freq`, in case `cutoff_freq< (1/t_obs)`, `cutoff_freq=1/t_obs`.
    - `thermal_drift`: inject a drift encoding frequencies up to the sampling rate, to simulate
    the thermal fluctuations in the focalplane.
    Both `slow_drift` and `thermal_drift` modes encode the possibility to inject a common mode drifts
    to all the detectors belonging to a group of detectors identified the string `focalplane_group` ( can
    be any string set by the user used to identify the groups in the detector table).
    The amount of common mode contribution is set by setting detector_mismatch to a value `<1`, (with
    0 being the case with only injecting common mode signal).

    """

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key to inject the gain drift"
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    include_common_mode = Bool(
        False, help="If True, inject a common drift to all the local detector group "
    )

    fknee_drift = Quantity(
        20.0 * u.mHz,
        help="fknee of the drift signal",
    )
    cutoff_freq = Quantity(
        0.2 * u.mHz,
        help="cutoff  frequency to simulate a slow  drift (assumed < sampling rate)",
    )
    sigma_drift = Float(
        1e-3,
        help="dimensionless amplitude  of the drift signal, (for `thermal_drift` corresponds to the thermal fluctuation level in K units)",
    )
    alpha_drift = Float(
        1.0,
        help="spectral index  of the drift signal spectrum",
    )

    detector_mismatch = Float(
        1.0,
        help="mismatch between detectors for `thermal_drift` and `slow_drift` ranging from 0 to 1. Default value implies no common mode injected",
    )
    thermal_fluctuation_amplitude = Quantity(
        1 * u.K,
        help="Amplitude of thermal fluctuation for `thermal_drift` in  Kelvin units ",
    )
    focalplane_Tbath = Quantity(
        100 * u.mK,
        help="temperature of the focalplane for `thermal_drift` ",
    )
    responsivity_function = Callable(
        lambda dT: dT,
        help="Responsivity function takes as input  the thermal  fluctuations,`dT` defined as `dT=Tdrift/Tbath + 1 `. Default we assume the identity function ",
    )

    realization = Int(0, help="integer to set a different random seed ")
    component = Int(0, allow_none=False, help="Component index for this simulation")

    drift_mode = Unicode(
        "linear",
        help="a string from [linear_drift, thermal_drift, slow_drift] to set the way the drift is modelled",
    )

    focalplane_group = Unicode(
        "wafer",
        help='focalplane table column to use for grouping detectors: can be any string like "wafer", "pixel"',
    )

    def get_psd(self, f):
        return (
            self.sigma_drift**2
            * (self.fknee_drift.to_value(u.Hz) / f) ** self.alpha_drift
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        """
        Generate gain timestreams.

        This iterates over all observations and detectors, simulates a gain drift across the observation time
        and  multiplies it   to the  signal TOD of the  detectors in each detector pair.


        Args:
            data (toast.Data): The distributed data.
        """
        env = Environment.get()
        log = Logger.get()

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
            fsampl = focalplane.sample_rate.to_value(u.Hz)

            if self.drift_mode == "linear_drift":
                key1 = (
                    self.realization * 4294967296 + telescope * 65536 + self.component
                )
                counter2 = 0

                for det in dets:
                    detindx = focalplane[det]["uid"]
                    key2 = sindx
                    counter1 = detindx

                    rngdata = rng.random(
                        1,
                        sampler="gaussian",
                        key=(key1, key2),
                        counter=(counter1, counter2),
                    )
                    gf = 1 + rngdata[0] * self.sigma_drift
                    gain = (gf - 1) * np.linspace(0, 1, size) + 1

                    ob.detdata[self.det_data][det] *= gain

            elif self.drift_mode == "thermal_drift":
                fmin = fsampl / (4 * size)
                # the factor of 4x the length of the sample vector  is
                # to avoid circular correlations
                freq = np.logspace(np.log10(fmin), np.log10(fsampl / 2.0), 1000)
                psd = self.get_psd(freq)
                det_group = np.unique(focalplane.detector_data[self.focalplane_group])
                thermal_fluct = np.zeros(
                    (len(det_group), ob.n_local_samples), dtype=np.float64
                )
                for iw, w in enumerate(det_group):
                    # simulate a noise-like timestream
                    thermal_fluct[iw][:] = sim_noise_timestream(
                        realization=self.realization,
                        telescope=ob.telescope.uid,
                        component=self.component,
                        sindx=sindx,
                        # we generate the same timestream for the
                        # detectors in the same group
                        detindx=iw,
                        rate=fsampl,
                        firstsamp=ob.local_index_offset,
                        samples=ob.n_local_samples,
                        freq=freq,
                        psd=psd,
                        py=False,
                    )

                for det in dets:
                    detindx = focalplane[det]["uid"]
                    # we inject a detector mismatch in the thermal thermal_fluctuation
                    # only if the mismatch !=0
                    if self.detector_mismatch != 0:
                        key1 = (
                            self.realization * 429496123345
                            + telescope * 6512345
                            + self.component
                        )
                        counter1 = detindx
                        counter2 = 0
                        key2 = sindx
                        rngdata = rng.random(
                            1,
                            sampler="gaussian",
                            key=(key1, key2),
                            counter=(counter1, counter2),
                        )
                        rngdata = 1 + rngdata[0] * self.detector_mismatch
                        thermal_factor = self.thermal_fluctuation_amplitude * rngdata
                    else:
                        thermal_factor = self.thermal_fluctuation_amplitude

                    # identify to which group the detector belongs
                    mask = focalplane[det][self.focalplane_group] == det_group

                    # assign the thermal fluct. simulated for that det. group
                    # making sure that  Tdrift is in the same units as Tbath
                    Tdrift = (thermal_factor * thermal_fluct[mask][0]).to(
                        self.focalplane_Tbath.unit
                    )
                    # we make dT an array of floats (from an array of dimensionless Quantity),
                    # this will avoid unit errors when multiplied to the det_data.

                    dT = (Tdrift / self.focalplane_Tbath + 1).to_value()

                    ob.detdata[self.det_data][det] *= self.responsivity_function(dT)

            elif self.drift_mode == "slow_drift":
                fmin = fsampl / (4 * size)
                # the factor of 4x the length of the sample vector  is
                # to avoid circular correlations
                freq = np.logspace(np.log10(fmin), np.log10(fsampl / 2.0), 1000)
                # making sure that the cut-off  frequency
                # is always above the  observation time scale .
                cutoff = np.max([self.cutoff_freq.to_value(u.Hz), fsampl / size])
                argmin = np.argmin(np.fabs(freq - cutoff))

                psd = np.concatenate(
                    [self.get_psd(freq[:argmin]), np.zeros_like(freq[argmin:])]
                )
                det_group = np.unique(focalplane.detector_data[self.focalplane_group])

                # if the mismatch is maximum (i.e. =1 ) we don't
                # inject the common mode but only an indepedendent slow drift

                if self.detector_mismatch == 1:
                    gain_common = np.zeros_like(det_group, dtype=np.float64)
                else:
                    gain_common = []
                    for iw, w in enumerate(det_group):
                        gain = sim_noise_timestream(
                            realization=self.realization,
                            telescope=ob.telescope.uid,
                            component=self.component,
                            sindx=sindx,
                            detindx=iw,  # drift common to all detectors
                            rate=fsampl,
                            firstsamp=ob.local_index_offset,
                            samples=ob.n_local_samples,
                            freq=freq,
                            psd=psd,
                            py=False,
                        )
                        gain_common.append(np.array(gain))
                        gain.clear()
                        del gain
                gain_common = np.array(gain_common)

                for det in dets:
                    detindx = focalplane[det]["uid"]
                    size = ob.detdata[self.det_data][det].size

                    # simulate a noise-like timestream

                    gain = sim_noise_timestream(
                        realization=self.realization,
                        telescope=ob.telescope.uid,
                        component=self.component,
                        sindx=sindx,
                        detindx=detindx,
                        rate=fsampl,
                        firstsamp=ob.local_index_offset,
                        samples=ob.n_local_samples,
                        freq=freq,
                        psd=psd,
                        py=False,
                    )
                    # identify to which group the detector belongs
                    mask = focalplane[det][self.focalplane_group] == det_group
                    # assign the thermal fluct. simulated for that det. group
                    ob.detdata[self.det_data][det] *= (
                        1
                        + (self.detector_mismatch * gain.array())
                        + (1 - self.detector_mismatch) * gain_common[mask][0]
                    )
                    gain.clear()
                    del gain

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
