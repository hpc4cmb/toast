# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from numpy.core.fromnumeric import size
import traitlets

import numpy as np

from astropy import units as u

import healpy as hp

from ..timing import function_timer

from .. import qarray as qa

from ..traits import trait_docs, Int, Unicode, Bool, Quantity, Float, Instance

from .operator import Operator

from ..utils import Environment, Logger

from ..atm import AtmSim


@trait_docs
class ObserveAtmosphere(Operator):
    """Operator which uses detector pointing to observe a simulated atmosphere slab."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode("times", help="Observation shared key for timestamps")

    det_data = Unicode(
        "signal", help="Observation detdata key for accumulating dipole timestreams"
    )

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for detector quaternions",
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

    gain = Float(1.0, help="Scaling applied to the simulated TOD")

    absorption = Float(None, allow_none=True, help="Atmospheric absorption")

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

        comm = data.comm.comm_group
        group = data.comm.group
        rank = data.comm.group_rank

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

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

            for vw in range(len(views)):
                # Determine the wind interval we are in, and hence which atmosphere
                # simulation to use.  The wind intervals are already guaranteed
                # by the calling code to break on the data view boundaries.
                if len(views) > 1:
                    while (
                        cur_wind < (len(ob.view[self.wind_view]) - 1)
                        and views.shared[self.times][vw][0]
                        <= ob.view[self.wind_view].shared[self.times][cur_wind][0]
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
                    azel_quat = views.detdata[self.quats][vw][det][good]

                    # Convert Az/El quaternion of the detector back into
                    # angles from the simulation.
                    theta, phi = qa.to_position(azel_quat)

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

                    # Integrate detector signal across all slabs at different altitudes

                    atmdata = np.zeros(ngood, dtype=np.float64)

                    for icur, cur_sim in enumerate(sim_list):
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
                            # DEBUG begin
                            import pickle

                            with open(
                                f"bad_quats_{rank}_{det}_{cur_wind}_{icur}.pck", "wb"
                            ) as fout:
                                pickle.dump(
                                    [
                                        cur_sim.azmin,
                                        cur_sim.azmax,
                                        cur_sim.elmin,
                                        cur_sim.elmax,
                                        az,
                                        el,
                                        azel_quat,
                                    ],
                                    fout,
                                )
                            # DEBUG end
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

                        err = cur_sim.observe(
                            views.shared[self.times][vw][good], az, el, atmdata, -1.0
                        )
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
                                    ] = 255
                                    nbad_tot += nbad

                    atmdata *= self.gain

                    if self.absorption is not None:
                        # Apply the frequency-dependent absorption-coefficient
                        atmdata *= self.absorption

                    # Add contribution to output
                    views.detdata[self.det_data][vw][det][good] += atmdata

            if comm is not None:
                comm.Barrier()
                ngood_tot = comm.reduce(ngood_tot)
                nbad_tot = comm.reduce(nbad_tot)
            if rank == 0 and nbad_tot > 0:
                log.error(
                    "{}: Observe atmosphere FAILED on {:.2f}% of samples".format(
                        log_prefix, nbad_tot * 100 / ngood_tot
                    )
                )

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
                self.quats,
            ],
            "intervals": [
                self.wind_view,
            ],
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
