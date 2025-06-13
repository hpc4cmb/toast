# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import ephem
import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from .. import rng
from ..coordinates import to_DJD
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger
from .interpolate_healpix import InterpolateHealpixMap
from .operator import Operator
from .pipeline import Pipeline


@trait_docs
class SimScanSynchronousSignal(Operator):
    """Operator which generates scan-synchronous signal timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    realization = Int(0, help="The simulation realization index")

    component = Int(663056, help="The simulation component index")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating simulated timestreams",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector frame",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="An operator that produces the Stokes weights in the Az/El frame",
    )

    pol = Bool(False, help="Ground map is polarized")

    nside = Int(128, help="Ground map healpix resolution")

    fwhm = Quantity(10 * u.arcmin, help="Ground map smoothing scale")

    lmax = Int(256, help="Ground map expansion order")

    scale = Quantity(1 * u.mK, help="RMS of the ground signal fluctuations at el=45deg")

    power = Float(
        -1,
        help="Exponential for suppressing ground pickup at higher observing elevation",
    )

    path = Unicode(
        None,
        allow_none=True,
        help="Path to a horizontal Healpix map to sample for the SSS *instead* "
        "of synthesizing Gaussian maps",
    )

    sss_map = "sss_map"

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

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        group = data.comm.group
        comm = data.comm.comm_group

        for trait in ("detector_pointing", "stokes_weights"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        for obs in data.obs:
            dets = obs.select_local_detectors(
                detectors, flagmask=defaults.det_mask_invalid
            )
            log_prefix = f"{group} : {obs.name} : "

            exists = obs.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )

            # The detector data units
            self.units = obs.detdata[self.det_data].units

            site = obs.telescope.site
            weather = site.weather

            key1, key2, counter1, counter2 = self._get_rng_keys(obs)

            log.debug_rank(f"{log_prefix}Simulating SSS", comm=comm)

            self._simulate_sss(obs, key1, key2, counter1, counter2, weather, comm)

            log.debug_rank(f"{log_prefix}Observing SSS", comm=comm)

            self._observe_sss(data, obs, dets)

        return

    @function_timer
    def _get_rng_keys(self, obs):
        """
        The random number generator accepts a key and a counter,
        each made of two 64bit integers.
        Following tod_math.py we set
        key1 = realization * 2^32 + telescope * 2^16 + component
        key2 = sindx * 2^32
        counter1 = hierarchical cone counter
        counter2 = sample in stream
        """
        telescope = obs.telescope.uid
        site = obs.telescope.site.uid
        sindx = obs.session.uid
        key1 = self.realization * 2**32 + telescope * 2**16 + self.component
        key2 = site * 2**16 + sindx
        counter1 = 0
        counter2 = 0
        return key1, key2, counter1, counter2

    @function_timer
    def _simulate_sss(self, obs, key1, key2, counter1, counter2, weather, comm):
        """
        Create a map of the ground signal to observe with all detectors
        """
        # We may already have cached the SSS map
        if self.sss_map in obs.shared and "sss_realization" in obs:
            if obs["sss_realization"] == self.realization:
                return

        # Number of Stokes components
        stokes_components = self.stokes_weights.mode
        nnz = len(stokes_components)

        # Surface temperature is made available but not used yet
        # to scale the SSS
        dtype = np.float32
        if comm is None or comm.rank == 0:
            # Only the root process loads or simulates the map
            temperature = weather.surface_temperature
            if self.path:
                sss_maps = np.atleast_2d(
                    hp.read_map(self.path, np.arange(nnz), dtype=dtype)
                )
            else:
                npix = 12 * self.nside**2
                sss_maps = []
                sss_maps = rng.random(
                    npix * nnz,
                    key=(key1, key2),
                    counter=(counter1, counter2),
                    sampler="gaussian",
                )
                sss_maps = np.reshape(sss_maps, [nnz, -1]).astype(dtype)
                for i, stokes in enumerate(stokes_components):
                    sss_map = sss_maps[i]
                    sss_map = hp.smoothing(
                        sss_map, fwhm=self.fwhm.to_value(u.radian), lmax=self.lmax
                    ).astype(dtype)
                    sss_map /= np.std(sss_map)
                    lon, lat = hp.pix2ang(
                        self.nside, np.arange(npix, dtype=np.int64), lonlat=True
                    )
                    scale = self.scale * (np.abs(lat) / 90 + 0.5) ** self.power
                    # Suppress all polarized componts to 10% of intensity
                    if stokes != "I":
                        scale *= 0.1
                    sss_map *= scale.to_value(self.units)
                    sss_maps[i] = sss_map
            sss_map = np.vstack(sss_maps)
            nmap, npix = sss_map.shape
        else:
            npix = None
            nmap = None
            sss_map = None

        if comm is not None:
            npix = comm.bcast(npix)
            nmap = comm.bcast(nmap)
        self.nside = hp.npix2nside(npix)
        obs.shared.create_group(self.sss_map, shape=(nmap, npix), dtype=dtype)
        obs.shared[self.sss_map].set(sss_map, fromrank=0)
        obs["sss_realization"] = self.realization

        return

    @function_timer
    def _observe_sss(self, data, obs, dets):
        """
        Use healpy bilinear interpolation to observe the ground signal map
        """

        sss_maps = obs.shared[self.sss_map].data

        interpolator = InterpolateHealpixMap(
            maps=[sss_maps],
            det_data=self.det_data,
            detector_pointing=self.detector_pointing,
            stokes_weights=self.stokes_weights,
        )

        obs_data = data.select(obs_uid=obs.uid)
        interpolator.apply(obs_data, detectors=dets)

        return

    def finalize(self, data, **kwargs):
        for obs in data.obs:
            del obs.shared[self.sss_map]
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
            "intervals": list(),
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        return prov
