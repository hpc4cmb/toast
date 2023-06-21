# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..dipole import dipole
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger, unit_conversion
from .operator import Operator


@trait_docs
class SimDipole(Operator):
    """Operator which generates dipole signal for detectors.

    This uses the detector pointing, the telescope velocity vectors, and the solar
    system motion with respect to the CMB rest frame to compute the observed CMB dipole
    signal.  The dipole timestream is either added (default) or subtracted from the
    specified detector data.

    The telescope velocity and detector quaternions are assumed to be in the same
    coordinate system.

    The "mode" trait determines what components of the telescope motion are included in
    the observed dipole.  Valid options are 'solar' for just the solar system motion,
    'orbital' for just the motion of the telescope with respect to the solarsystem
    barycenter, and 'total' which is the sum of both (and the default).

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating dipole timestreams",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    boresight = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight"
    )

    velocity = Unicode(defaults.velocity, help="Observation shared key for velocity")

    subtract = Bool(
        False, help="If True, subtract the dipole timestream instead of accumulating"
    )

    mode = Unicode("total", help="Valid options are solar, orbital, and total")

    coord = Unicode(
        "E",
        help="Valid options are 'C' (Equatorial), 'E' (Ecliptic), and 'G' (Galactic)",
    )

    solar_speed = Quantity(
        369.0 * u.kilometer / u.second,
        help="Amplitude of the solarsystem barycenter velocity with respect to the CMB",
    )

    solar_gal_lat = Quantity(
        48.26 * u.degree, help="Galactic latitude of direction of solarsystem motion"
    )

    solar_gal_lon = Quantity(
        263.99 * u.degree, help="Galactic longitude of direction of solarsystem motion"
    )

    cmb = Quantity(2.72548 * u.Kelvin, help="CMB monopole value")

    freq = Quantity(0 * u.Hz, help="Optional observing frequency")

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        check = proposal["value"]
        if check not in ["solar", "orbital", "total"]:
            raise traitlets.TraitError(
                "Invalid mode (must be 'solar', 'orbital' or 'total')"
            )
        return check

    @traitlets.validate("coord")
    def _check_coord(self, proposal):
        check = proposal["value"]
        if check is not None:
            if check not in ["E", "C", "G"]:
                raise traitlets.TraitError("coordinate system must be 'E', 'C', or 'G'")
        return check

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

        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        # Compute the solar system velocity in galactic coordinates
        solar_gal_theta = np.deg2rad(90.0 - self.solar_gal_lat.to_value(u.degree))
        solar_gal_phi = np.deg2rad(self.solar_gal_lon.to_value(u.degree))

        solar_speed_kms = self.solar_speed.to_value(u.kilometer / u.second)
        solar_projected = solar_speed_kms * np.sin(solar_gal_theta)

        sol_z = solar_speed_kms * np.cos(solar_gal_theta)
        sol_x = solar_projected * np.cos(solar_gal_phi)
        sol_y = solar_projected * np.sin(solar_gal_phi)
        solar_gal_vel = np.array([sol_x, sol_y, sol_z])

        # Rotate solar system velocity to desired coordinate frame
        solar_vel = None
        if self.coord == "G":
            solar_vel = solar_gal_vel
        else:
            rotmat = hp.rotator.Rotator(coord=["G", self.coord]).mat
            solar_vel = np.ravel(np.dot(rotmat, solar_gal_vel))

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Make sure detector data output exists
            exists = ob.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )

            # Unit conversion from dipole timestream (K) to det data units
            scale = unit_conversion(u.K, ob.detdata[self.det_data].units)

            # Loop over views
            views = ob.view[self.view]

            for vw in range(len(views)):
                # Boresight pointing quaternions
                boresight = views.shared[self.boresight][vw]

                # Set the solar and orbital velocity inputs based on the
                # requested mode.

                sol = None
                vel = None
                if (self.mode == "solar") or (self.mode == "total"):
                    sol = solar_vel
                if (self.mode == "orbital") or (self.mode == "total"):
                    vel = views.shared[self.velocity][vw]

                # Focalplane for this observation
                focalplane = ob.telescope.focalplane

                for det in dets:
                    props = focalplane[det]

                    # Detector quaternion offset from the boresight
                    detquat = props["quat"]

                    # Timestream of detector quaternions
                    quats = qa.mult(boresight, detquat)

                    # Compute the dipole timestream for this view and detector
                    dipole_tod = dipole(
                        quats,
                        vel=vel,
                        solar=sol,
                        cmb=self.cmb,
                        freq=self.freq,
                    )

                    # Add contribution to output
                    if self.subtract:
                        views.detdata[self.det_data][vw][det] -= scale * dipole_tod
                    else:
                        views.detdata[self.det_data][vw][det] += scale * dipole_tod
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
