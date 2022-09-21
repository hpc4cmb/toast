# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import ephem
import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..coordinates import to_DJD
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, List, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, Timer
from .operator import Operator
from .pipeline import Pipeline


@trait_docs
class FlagSSO(Operator):
    """Operator which flags detector data in the vicinity of solar system objects"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector frame",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(defaults.det_mask_sso, help="Bit mask to raise flags with")

    sso_names = List(
        # default_value=["Sun", "Moon"],
        trait=Unicode,
        allow_none=True,
        help="Names of the SSOs, must be recognized by pyEphem",
    )

    sso_radii = List(
        # default_value=[45.0 * u.deg, 5.0 * u.deg],
        trait=Quantity,
        allow_none=True,
        help="Radii around the sources to flag",
    )

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
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

        for trait in "sso_names", "sso_radii":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if len(self.sso_names) != len(self.sso_radii):
            raise RuntimeError("Each SSO must have a radius")

        self.ssos = []
        for sso_name in self.sso_names:
            self.ssos.append(getattr(ephem, sso_name)())
        self.nsso = len(self.ssos)

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            site = obs.telescope.site
            observer = ephem.Observer()
            observer.lon = site.earthloc.lon.to_value(u.radian)
            observer.lat = site.earthloc.lat.to_value(u.radian)
            observer.elevation = site.earthloc.height.to_value(u.meter)
            observer.epoch = ephem.J2000
            observer.temp = 0  # in Celcius
            observer.compute_pressure()

            # Get the observation time span and compute the horizontal
            # position of the SSO
            times = obs.shared[self.times].data
            sso_azs, sso_els = self._get_sso_positions(times, observer)

            self._flag_ssos(data, obs, dets, sso_azs, sso_els)

        return

    @function_timer
    def _get_sso_positions(self, times, observer):
        """
        Calculate the SSO horizontal position
        """
        sso_azs = np.zeros([self.nsso, times.size])
        sso_els = np.zeros([self.nsso, times.size])
        # Only evaluate the position every second and interpolate
        # in between
        n = min(int(times[-1] - times[0]), 2)
        tvec = np.linspace(times[0], times[-1], n)
        for isso, sso in enumerate(self.ssos):
            azvec = np.zeros(n)
            elvec = np.zeros(n)
            for i, t in enumerate(tvec):
                observer.date = to_DJD(t)
                sso.compute(observer)
                azvec[i] = sso.az
                elvec[i] = sso.alt
                # Did the SSO cross zero azimuth?
                if i > 0 and np.abs(azvec[i - 1] - azvec[i]) > np.pi:
                    # yes
                    az1, az2 = azvec[i - 1], azvec[i]
                    # unwind the angle
                    if az1 < az2:
                        az2 -= 2 * np.pi
                    else:
                        az2 += 2 * np.pi
                    # Shift the second to last time stamp
                    tvec[i - 1] = t
                    azvec[i - 1] = az2
                    elvec[i - 1] = sso.alt
            sso_azs[isso] = np.interp(times, tvec, azvec)
            sso_els[isso] = np.interp(times, tvec, elvec)
        return sso_azs, sso_els

    @function_timer
    def _flag_ssos(self, data, obs, dets, sso_azs, sso_els):
        """
        Flag the SSO for each detector in tod
        """

        exists_flags = obs.detdata.ensure(
            self.det_flags, dtype=np.uint8, detectors=dets
        )

        for det in dets:
            try:
                # Use cached detector quaternions
                quats = obs.detdata[self.detector_pointing.quats][det]
            except KeyError:
                # Compute the detector quaternions
                obs_data = data.select(obs_uid=obs.uid)
                self.detector_pointing.apply(obs_data, detectors=[det])
                quats = obs.detdata[self.detector_pointing.quats][det]

            # Convert Az/El quaternion of the detector into angles
            theta, phi, _ = qa.to_iso_angles(quats)

            # Azimuth is measured in the opposite direction
            # than longitude
            az = 2 * np.pi - phi
            el = np.pi / 2 - theta

            flags = obs.detdata[self.det_flags][det]

            cosel = np.cos(el)
            for sso_az, sso_el, sso_radius in zip(sso_azs, sso_els, self.sso_radii):
                radius = sso_radius.to_value(u.radian)
                # Flag samples within search radius.
                if radius > np.radians(15):
                    # Exact formula for cosine of the angular distance
                    rcos = np.sin(el) * np.sin(sso_el) + np.cos(el) * np.cos(
                        sso_el
                    ) * np.cos(az - sso_az)
                    inside = rcos > np.cos(radius)
                else:
                    # This is the planar approximation for squared angular distance
                    x = (az - sso_az) * cosel
                    y = el - sso_el
                    rsquared = x**2 + y**2
                    inside = rsquared < radius**2
                flags[inside] |= self.det_flag_mask

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": [self.view],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_flags],
        }
        return prov
