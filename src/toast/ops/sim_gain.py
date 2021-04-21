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

@trait_docs
class GainDrifter(Operator):
    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    signalname  = Unicode(
        "signal", help="Observation detdata key to inject the gain drift"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    include_common_mode = Bool(
        False, help="If True, inject a common drift to all the local detector group "
    )


    fknee_drift = Quantity(
        20.0 * u.mHz,
        help="fknee of the drift signal",
    )
    downsampled_freq = Quantity(
            200.0 * u.mHz,
            help="sampling frequency to simulate the drift (assumed < sampling rate)",
        )
    sigma_drift = Float(
        1e-3 ,
        help="dimensionless amplitude  of the drift signal",
    )
    alpha_drift = Float(
        1. ,
        help="spectral index  of the drift signal spectrum",
    )
    mc_realization = Int(0, help="integer to set a different random seed ")


    def draw_noise(self, freq,alpha , fknee, sigmag, deltanu):
        """
        Draw an amplitude for frequency nu assuming frequency spacing dnu.
        Assume Gaussian distributed noise with variance
         (Pink(freq)**2 )  / (2 dnu).

        Args:
            freq: frequency in Hz
        """

        pink = ( fknee  / np.absolute(freq)) ** self.alpha_drift * self.sigma_drift
        total = np.sqrt(pink ** 2)
        return np.random.normal(scale=np.sqrt(total ** 2 / (2.0 * deltanu )))

    @function_timer
    def make_gain_samples(self, size, fsampl ):
        """
        Generate gain drift samples given an input PSD.
        Drifts are simulated up to a frequency cut off, hard coded to 0.5mHz, and sampled
        with a lower sampling rate. Once they are simulated,  they are high sampled
        to the  rate of time samples.
        """
        fd= self.downsampled_freq.to_value(u.Hz ) ,
        fsampl=   fsampl.to_value(u.Hz),
        fknee= self.fknee_drift.to_value(u.Hz ),

        dt = 1 / fd
        obstime = size / fsampl
        # We artificially set the cut off frequency to 0.5 mHz
        # to get rid of the unphysical fluctuations above this threshold
        fcutoff = 0.5e-3

        N = np.int_(obstime * fd)
        t0 = 0.0

        if N % 2 == 0:
            N += 1

        dnu = 1.0 / (obstime)

        nu_array = np.fft.fftshift(np.fft.fftfreq(n=N, d=dt))

        mask_cut = np.ma.masked_less_equal(abs(nu_array), fcutoff).mask
        mask_neg = np.ma.masked_less((nu_array), 0).mask
        mask = np.logical_and(mask_cut, mask_neg)
        noise_spectrum = np.zeros(N, dtype="complex")
        re = draw_drift(nu_array[mask], fknee, self.sigma_drift, self.alpha_drift,dnu )
        im = draw_drift(nu_array[mask], fknee, self.sigma_drift, self.alpha_drift , dnu )
        k_m = 2.0 * np.pi * np.fft.fftfreq(len(nu_array), d=dt)
        noise_spectrum[mask] = (1.0 / np.sqrt(2.0)) * (re + 1j * im)
        noise_spectrum[slice((N - 1) // 2 + 1, N, 1)] = np.conj(
            np.flip(noise_spectrum[slice(N // 2)])
        )
        timestream = np.fft.ifft(
            np.fft.ifftshift(noise_spectrum) * np.exp(1j * k_m * t0), n=(N)
        ) / (dt)
        t_array = np.linspace(0, obstime, N)

        try:
            assert abs(timestream.imag.max() / timestream.real.max()) < 1e-10
        except AssertionError:
            print(
                "Simulated timestreams are complex(size %d).\nMax imag. part = %g, Max real part= %g"
                % (N, timestream.imag.max(), timestream.real.max())
            )

        t_interp = np.linspace(0, obstime, size)
        interpolated_timestream = np.interp(t_interp, xp=t_array, fp=timestream.real)

        return interpolated_timestream

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
        comm = data.comm
        rank= comm.group_rank
        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Make sure detector data output exists
            ob.detdata.ensure(self.signalname, detectors=dets)

            # Loop over views
            views = ob.view[self.view]

            for vw in range(len(views)):
                # Focalplane for this observation
                focalplane = ob.telescope.focalplane
                for kdet , det in enumerate(focalplane.detectors):
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

        return
