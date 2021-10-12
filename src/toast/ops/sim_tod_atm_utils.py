# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from numpy.core.fromnumeric import size
import traitlets

import numpy as np

from astropy import units as u

import healpy as hp

from ..timing import function_timer, GlobalTimers

from .. import qarray as qa

from ..traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from .operator import Operator

from ..utils import Environment, Logger

from ..observation import default_values as defaults

from ..atm import AtmSim


@trait_docs
class ObserveAtmosphere(Operator):
    """Operator which uses detector pointing to observe a simulated atmosphere slab."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating dipole timestreams",
    )

    quats_azel = Unicode(
        defaults.quats_azel,
        allow_none=True,
        help="Observation detdata key for detector quaternions",
    )

    weights = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for detector Stokes weights",
    )

    weights_mode = Unicode("IQU", help="Stokes weights mode (eg. 'I', 'IQU', 'QU')")

    polarization_fraction = Float(
        0,
        help="Polarization fraction (only Q polarization).",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of valid data in all observations"
    )

    wind_view = Unicode(
        "wind", help="The view of times matching individual simulated atmosphere slabs"
    )

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    sim = Unicode("atmsim", help="The observation key for the list of AtmSim objects")

    absorption = Unicode(
        None, allow_none=True, help="The observation key for the absorption"
    )

    loading = Unicode(None, allow_none=True, help="The observation key for the loading")

    n_bandpass_freqs = Int(
        100,
        help="The number of sampling frequencies used when convolving the bandpass with atmosphere absorption and loading",
    )

    gain = Float(1.0, help="Scaling applied to the simulated TOD")

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        gt = GlobalTimers.get()

        gt.start("ObserveAtmosphere:  total")

        comm = data.comm.comm_group
        group = data.comm.group
        rank = data.comm.group_rank

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            gt.start("ObserveAtmosphere:  per-observation setup")
            # Bandpass-specific unit conversion, relative to 150GHz
            absorption, loading = self._get_absorption_and_loading(ob, dets)

            # Make sure detector data output exists
            ob.detdata.ensure(self.det_data, detectors=dets)

            # Prefix for logging
            log_prefix = f"{group} : {ob.name}"

            # The current wind-driven timespan
            cur_wind = 0

            # Loop over views
            views = ob.view[self.view]

            ngood_tot = 0
            nbad_tot = 0

            gt.stop("ObserveAtmosphere:  per-observation setup")
            for vw in range(len(views)):
                times = views.shared[self.times][vw]

                # Determine the wind interval we are in, and hence which atmosphere
                # simulation to use.  The wind intervals are already guaranteed
                # by the calling code to break on the data view boundaries.
                if len(views) > 1:
                    while (
                        cur_wind < (len(ob.view[self.wind_view]) - 1)
                        and times[0]
                        > ob.view[self.wind_view].shared[self.times][cur_wind][-1]
                    ):
                        cur_wind += 1

                # Get the flags if needed
                sh_flags = None
                if self.shared_flags is not None:
                    sh_flags = (
                        np.array(views.shared[self.shared_flags][vw])
                        & self.shared_flag_mask
                    )

                sim_list = ob[self.sim][cur_wind]

                for det in dets:
                    gt.start("ObserveAtmosphere:  detector setup")
                    flags = None
                    if self.det_flags is not None:
                        flags = (
                            np.array(views.detdata[self.det_flags][vw][det])
                            & self.det_flag_mask
                        )
                        if sh_flags is not None:
                            flags |= sh_flags
                    elif sh_flags is not None:
                        flags = sh_flags

                    good = slice(None, None, None)
                    ngood = len(views.detdata[self.det_data][vw][det])
                    if flags is not None:
                        good = flags == 0
                        ngood = np.sum(good)

                    if ngood == 0:
                        continue
                    ngood_tot += ngood

                    # Detector Az / El quaternions for good samples
                    azel_quat = views.detdata[self.quats_azel][vw][det][good]

                    # Convert Az/El quaternion of the detector back into
                    # angles from the simulation.
                    theta, phi = qa.to_position(azel_quat)

                    # Stokes weights for observing polarized atmosphere
                    if self.weights is None:
                        weights_I = 1
                        weights_Q = 0
                    else:
                        weights = views.detdata[self.weights][vw][det][good]
                        if "I" in self.weights_mode:
                            ind = self.weights_mode.index("I")
                            weights_I = weights[:, ind].copy()
                        else:
                            weights_I = 0
                        if "Q" in self.weights_mode:
                            ind = self.weights_mode.index("Q")
                            weights_Q = weights[:, ind].copy()
                        else:
                            weights_Q = 0

                    # Azimuth is measured in the opposite direction
                    # than longitude
                    az = 2 * np.pi - phi
                    el = np.pi / 2 - theta

                    if np.ptp(az) < np.pi:
                        azmin_det = np.amin(az)
                        azmax_det = np.amax(az)
                    else:
                        # Scanning across the zero azimuth.
                        azmin_det = np.amin(az[az > np.pi]) - 2 * np.pi
                        azmax_det = np.amax(az[az < np.pi])
                    elmin_det = np.amin(el)
                    elmax_det = np.amax(el)

                    tmin_det = times[good][0]
                    tmax_det = times[good][-1]

                    # Integrate detector signal across all slabs at different altitudes

                    atmdata = np.zeros(ngood, dtype=np.float64)

                    gt.stop("ObserveAtmosphere:  detector setup")
                    gt.start("ObserveAtmosphere:  detector AtmSim.observe")
                    for icur, cur_sim in enumerate(sim_list):
                        if cur_sim.tmin > tmin_det or cur_sim.tmax < tmax_det:
                            msg = f"{log_prefix} : {det} "
                            msg += "Detector time: [{:.1f}, {:.1f}], ".format(
                                tmin_det, tmax_det
                            )
                            msg += "is not contained in [{:.1f}, {:.1f}]".format(
                                cur_sim.tmin,
                                cur_sim.tmax,
                            )
                            raise RuntimeError(msg)
                        if (
                            not (
                                cur_sim.azmin <= azmin_det
                                and azmax_det <= cur_sim.azmax
                            )
                            and not (
                                cur_sim.azmin <= azmin_det - 2 * np.pi
                                and azmax_det - 2 * np.pi <= cur_sim.azmax
                            )
                        ) or not (
                            cur_sim.elmin <= elmin_det and elmin_det <= cur_sim.elmax
                        ):
                            msg = f"{log_prefix} : {det} "
                            msg += "Detector Az/El: [{:.5f}, {:.5f}], ".format(
                                azmin_det, azmax_det
                            )
                            msg += "[{:.5f}, {:.5f}] is not contained in ".format(
                                elmin_det, elmax_det
                            )
                            msg += "[{:.5f}, {:.5f}], [{:.5f} {:.5f}]".format(
                                cur_sim.azmin,
                                cur_sim.azmax,
                                cur_sim.elmin,
                                cur_sim.elmax,
                            )
                            raise RuntimeError(msg)

                        err = cur_sim.observe(times[good], az, el, atmdata, -1.0)
                        if err != 0:
                            # Observing failed
                            bad = np.abs(atmdata) < 1e-30
                            nbad = np.sum(bad)
                            msg = f"{log_prefix} : {det} "
                            msg += f"ObserveAtmosphere failed for {nbad} "
                            msg += "({:.2f} %) samples.  det = {}, rank = {}".format(
                                nbad * 100 / ngood, det, rank
                            )
                            log.error(msg)
                            # If any samples failed the simulation, flag them as bad
                            if nbad > 0:
                                atmdata[bad] = 0
                                if self.det_flags is None:
                                    msg = "Some samples failed atmosphere simulation, but "
                                    msg += (
                                        "no det flag field was specified.  Cannot flag "
                                    )
                                    msg += "samples"
                                    log.warning(msg)
                                else:
                                    views.detdata[self.det_flags][vw][det][good][
                                        bad
                                    ] |= self.det_flag_mask
                                    nbad_tot += nbad
                    gt.stop("ObserveAtmosphere:  detector AtmSim.observe")
                    gt.start("ObserveAtmosphere:  detector accumulate")

                    # Calibrate the atmopsheric fluctuations to appropriate bandpass
                    atmdata *= self.gain * absorption[det]

                    # Add the elevation-dependent atmospheric loading
                    atmdata += loading[det] / np.sin(el)

                    # Add polarization.  In our simple model, there is only Q-polarization
                    # and the polarization fraction is constant.
                    pfrac = self.polarization_fraction
                    atmdata *= weights_I + weights_Q * pfrac

                    # Add contribution to output
                    views.detdata[self.det_data][vw][det][good] += atmdata
                    gt.stop("ObserveAtmosphere:  detector accumulate")

            if nbad_tot > 0:
                frac = nbad_tot / (ngood_tot + nbad_tot) * 100
                log.error(
                    "{log_prefix}: Observe atmosphere FAILED on {frac:.2f}% of samples"
                )
        gt.stop("ObserveAtmosphere:  total")

    @function_timer
    def _get_absorption_and_loading(self, obs, dets):
        """Bandpass-specific unit conversion and loading"""

        if obs.telescope.focalplane.bandpass is None:
            raise RuntimeError("Focalplane does not define bandpass")
        bandpass = obs.telescope.focalplane.bandpass

        freq_min, freq_max = bandpass.get_range()
        n_freq = self.n_bandpass_freqs
        freqs = np.linspace(freq_min, freq_max, n_freq)

        absorption = obs[self.absorption]
        loading = obs[self.loading]

        absorption_det = {}
        loading_det = {}
        for det in dets:
            absorption_det[det] = bandpass.convolve(det, freqs, absorption, rj=True)
            loading_det[det] = bandpass.convolve(det, freqs, loading, rj=True)

        return absorption_det, loading_det

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.sim],
            "shared": [
                self.times,
            ],
            "detdata": [
                self.det_data,
                self.quats_azel,
            ],
            "intervals": [
                self.wind_view,
            ],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.weights is not None:
            req["weights"].append(self.weights)
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
            "intervals": list(),
        }
        return prov

    def _accelerators(self):
        return list()
