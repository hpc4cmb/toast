# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np
from astropy import units as u
from ..timing import function_timer

from ..traits import trait_docs, Int,Float, Unicode, Bool, Quantity

from .operator import Operator

from ..utils import Environment, Logger
from .. import rng
from .sim_tod_noise  import sim_noise_timestream



@trait_docs
class GainDrifter(Operator):
    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data  = Unicode(
        "signal", help="Observation detdata key to inject the gain drift"
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
        1e-3 ,
        help="dimensionless amplitude  of the drift signal",
    )
    alpha_drift = Float(
        1. ,
        help="spectral index  of the drift signal spectrum",
    )
    detector_mismatch = Float(
        1. ,
        help="mismatch between detectors for `thermal_drift` and `slow_drift` ranging from 0 to 1. Default value implies no common mode injected",
    )
    thermal_fluctuation_amplitude = Float(
        1e-2 ,
        help="Amplitude of thermal fluctuation for `thermal_drift`. ",
    )
    realization = Int(0, help="integer to set a different random seed ")
    component = Int(0, allow_none=False, help="Component index for this simulation")

    drift_mode= Unicode(
        "linear", help="a string from [linear_drift, thermal_drift, slow_drift] to set the way the drift is modelled")

    focalplane_group  = Unicode(
        "wafer", help='focalplane table column to use for grouping detectors: can be any string like "wafer", "pixel"'
    )

    def get_psd(self, f ):
        return  self.sigma_drift**2 *(self.fknee_drift.to_value(u.Hz)/f)**self.alpha_drift


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
            comm = ob.comm
            rank= ob.comm_rank
            # Make sure detector data output exists
            ob.detdata.ensure(self.det_data, detectors=dets)
            obsindx = ob.uid
            telescope = ob.telescope.uid
            focalplane = ob.telescope.focalplane
            size = ob.detdata[self.det_data][focalplane.detector_data[0]["name"]].size
            fsampl = focalplane.sample_rate.to_value(u.Hz)

            if self.drift_mode == "linear_drift":
                key1 = self.realization * 4294967296 + telescope * 65536 + self.component
                counter1 = 0
                counter2 = 0

                for det in dets:
                    detindx = focalplane[det]["uid"]
                    key2 = obsindx * 4294967296 + detindx

                    rngdata = rng.random(
                        1,
                        sampler="gaussian",
                        key=(key1, key2),
                        counter=(counter1, counter2),
                    )
                    gf = 1 + rngdata[0] * self.sigma_drift
                    gain = (gf-1) * np.linspace(0,1,size ) + 1

                    ob.detdata[self.det_data][det] *= gain

            elif self.drift_mode == "thermal_drift":
                fmin = fsampl / (4*size)
                #the factor of 4x the length of the sample vector  is
                # to avoid circular correlations
                freq= np.logspace(np.log10(fmin),
                                np.log10(fsampl/2.), 1000 )
                psd = self.get_psd(freq )
                det_group= np.unique(focalplane.detector_data[self.focalplane_group])
                thermal_fluct=[]
                for iw, w in enumerate(det_group):
                    # simulate a noise-like timestream
                    gain  = sim_noise_timestream(
                        realization=self.realization ,
                        telescope=ob.telescope.uid,
                        component=self.component ,
                        obsindx=ob.uid,
                        # we generate the same timestream for the
                        # detectors in the same group
                        detindx=iw ,
                        rate=fsampl,
                        firstsamp=ob.local_index_offset,
                        samples=ob.n_local_samples,
                        freq=freq ,
                        psd=psd ,
                        py=False ,
                    )
                    thermal_fluct.append(gain )
                thermal_fluct= np.array(thermal_fluct)

                for det in dets:
                    detindx = focalplane[det]["uid"]
                    #we inject a detector mismatch in the thermal thermal_fluctuation
                    # only if the mismatch !=0
                    if self.detector_mismatch !=0 :
                        key1 = self.realization * 429496123345 + telescope * 6512345 + self.component
                        counter1 = 0
                        counter2 = 0
                        key2 = obsindx * 12345667296 + detindx
                        rngdata =  rng.random(
                                        1,
                                        sampler="gaussian",
                                        key=(key1, key2),
                                        counter=(counter1, counter2),
                                        )
                        rngdata = (1+ rngdata[0] *self.detector_mismatch)
                        thermal_factor= self.thermal_fluctuation_amplitude*rngdata
                    else :
                        thermal_factor= self.thermal_fluctuation_amplitude

                    #identify to which group the detector belongs
                    mask = focalplane[det][self.focalplane_group] ==det_group
                    #assign the thermal fluct. simulated for that det. group
                    ob.detdata[self.det_data][det] *= (1+ thermal_fluct[mask][0]*thermal_factor)

            elif self.drift_mode == "slow_drift":
                fmin = fsampl / (4*size)
                #the factor of 4x the length of the sample vector  is
                # to avoid circular correlations
                freq= np.logspace(np.log10(fmin),
                                np.log10(fsampl/2.), 1000 )
                # making sure that the cut-off  frequency
                #is always above the  observation time scale .
                cutoff = np.max([self.cutoff_freq.to_value(u.Hz ), fsampl/size  ])
                argmin= np.argmin ( np.fabs( freq-cutoff )  )

                psd =np.concatenate([self.get_psd(freq[:argmin] ),
                                         np.zeros_like(freq[argmin:])] )
                det_group= np.unique(focalplane.detector_data[self.focalplane_group])

                #if the mismatch is maximum (i.e. =1 ) we don't
                # inject the common mode but only an indepedendent slow drift

                if self.detector_mismatch == 1 :
                    gain_common=np.zeros_like(det_group, dtype=np.float_)
                else:
                    gain_common=[]
                    for iw, w in enumerate(det_group):
                        gain_common .append( sim_noise_timestream(
                            realization=self.realization ,
                            telescope=ob.telescope.uid,
                            component=self.component ,
                            obsindx=ob.uid,
                            detindx=iw, # drift common to all detectors
                            rate=fsampl,
                            firstsamp=ob.local_index_offset,
                            samples=ob.n_local_samples,
                            freq=freq ,
                            psd=psd ,
                            py=False ,
                        )
                    )
                gain_common=np.array(gain_common)


                for det in dets:
                    detindx = focalplane[det]["uid"]
                    size= ob.detdata[self.det_data][det].size

                    # simulate a noise-like timestream

                    gain  = sim_noise_timestream(
                        realization=self.realization ,
                        telescope=ob.telescope.uid,
                        component=self.component ,
                        obsindx=ob.uid,
                        detindx=detindx,
                        rate=fsampl,
                        firstsamp=ob.local_index_offset,
                        samples=ob.n_local_samples,
                        freq=freq ,
                        psd=psd ,
                        py=False ,
                    )
                    #identify to which group the detector belongs
                    mask = focalplane[det][self.focalplane_group] ==det_group
                    #assign the thermal fluct. simulated for that det. group
                    ob.detdata[self.det_data][det] *= (1+ (self.detector_mismatch *gain)
                                                        + (1-self.detector_mismatch )* gain_common[mask][0])


        return
