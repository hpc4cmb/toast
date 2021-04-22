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
    realization = Int(0, help="integer to set a different random seed ")
    component = Int(0, allow_none=False, help="Component index for this simulation")

    drift_mode= Unicode(
        "linear", help="a string from [linear_drift, thermal_drift, slow_drift] to set the way the drift is modelled")
    user_data  = Unicode(
        "drift_signal", help="samples  encoding a gaindrift provided by the user"
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
            if self.drift_mode == "linear_drift":
                key1 = self.realization * 4294967296 + telescope * 65536 + self.component
                counter1 = 0
                counter2 = 0

                for det in dets:
                    detindx = focalplane[det]["uid"]
                    size= ob.detdata[self.det_data][det].size

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
                for det in dets:
                    detindx = focalplane[det]["uid"]
                    size= ob.detdata[self.det_data][det].size

                    fsampl = ob.telescope.focalplane.sample_rate.to_value(u.Hz)

                    fmin = fsampl / (4*size)
                    #the factor of 4x the length of the sample vector  is
                    # to avoid circular correlations

                    freq= np.logspace(np.log10(fmin),
                                    np.log10(fsampl/2.), 1000 )

                    psd = self.get_psd(freq )
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
                    ob.detdata[self.det_data][det] *= (1+ gain)

            elif self.drift_mode == "slow_drift":
                for det in dets:
                    detindx = focalplane[det]["uid"]
                    size= ob.detdata[self.det_data][det].size

                    fsampl = ob.telescope.focalplane.sample_rate.to_value(u.Hz)

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
                    if self.include_common_mode :
                        gain_common   = sim_noise_timestream(
                            realization=self.realization ,
                            telescope=ob.telescope.uid,
                            component=self.component ,
                            obsindx=ob.uid,
                            detindx=0, # drift common to all detectors
                            rate=fsampl,
                            firstsamp=ob.local_index_offset,
                            samples=ob.n_local_samples,
                            freq=freq ,
                            psd=psd ,
                            py=False ,
                        )
                    else:
                        gain_common=0. 

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
                    ob.detdata[self.det_data][det] *= (1+ gain+gain_common)


        return
"""
            for kdet , det in enumerate(dets):
                    size= views.detdata[self.signalname][vw][det].size
                    if self.include_common_mode:
                        if rank == 0:
                            np.random.seed(
                                  4294967*self.mc_realization  + vw * 123456 + comm.group
                            )
                            #  simulate common mode gaindrift during this observation per detector,
                            #gain_common =   self.make_gain_samples(size)

                            gain_common =self.make_gain_samples(
                                            size=size,
                                            fsampl=   ob.telescope.focalplane.sample_rate.to_value(u.Hz),
                                            )
                        else:
                            gain_common = np.zeros(size)
                        comm.Bcast([gain_common, MPI.DOUBLE], root=0)

                    else:
                        gain_common =  0


                np.random.seed(
                    self.mc_realization * 4294967
                    + rank * 65536
                    + vw *239399
                    + kdet *12345
                )
                #  simulate gaindrift during this observation per detector,
                #gain = self.make_gain_samples(size)
                gain =self.make_gain_samples(
                        size=size,
                        fsampl=   ob.telescope.focalplane.sample_rate ,
                    )
                gaindrift =( 1+ gain_common) *( 1+  gain )
                views.detdata[self.signalname][vw][det]  *= ( gaindrift )
                """
