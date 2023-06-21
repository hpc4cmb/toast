# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels_io_healpix import write_healpix_fits
from ..timing import Timer, function_timer
from ..traits import Bool, Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .copy import Copy
from .delete import Delete
from .mapmaker_utils import BuildNoiseWeighted
from .operator import Operator
from .pipeline import Pipeline
from .pointing import BuildPixelDistribution


class UniformNoise:
    def detector_weight(self, det):
        return 1.0 / (u.K**2)


@trait_docs
class CrossLinking(Operator):
    """Evaluate an ACT-style crosslinking map"""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pixel pointing operator.",
    )

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object should be stored",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional telescope flagging",
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    save_pointing = Bool(False, help="If True, do not clear pixel numbers after use")

    # FIXME: these should be made into traits and also placed in _provides().

    signal = "dummy_signal"
    weights = "crosslinking_weights"
    crosslinking_map = "crosslinking_map"
    noise_model = "uniform_noise_weights"

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _get_weights(self, obs_data, det):
        """Evaluate the special pointing matrix"""

        obs = obs_data.obs[0]
        exists_signal = obs.detdata.ensure(
            self.signal, detectors=[det], create_units=self.det_data_units
        )
        exists_weights = obs.detdata.ensure(
            self.weights, sample_shape=(3,), detectors=[det]
        )

        signal = obs.detdata[self.signal][det]
        signal[:] = 1
        weights = obs.detdata[self.weights][det]
        # Compute the detector quaternions
        self.pixel_pointing.detector_pointing.apply(obs_data, detectors=[det])
        quat = obs.detdata[self.pixel_pointing.detector_pointing.quats][det]
        # measure the scan direction wrt the local meridian for each sample
        theta, phi, _ = qa.to_iso_angles(quat)
        theta = np.pi / 2 - theta
        # scan direction across the reference sample
        dphi = np.roll(phi, -1) - np.roll(phi, 1)
        dtheta = np.roll(theta, -1) - np.roll(theta, 1)
        # except first and last sample
        for dx, x in (dphi, phi), (dtheta, theta):
            dx[0] = x[1] - x[0]
            dx[-1] = x[-1] - x[-2]
        # scale dphi to on-sky
        dphi *= np.cos(theta)
        # Avoid overflows
        tiny = np.abs(dphi) < 1e-30
        if np.any(tiny):
            ang = np.zeros(signal.size)
            ang[tiny] = np.sign(dtheta) * np.sign(dphi) * np.pi / 2
            not_tiny = np.logical_not(tiny)
            ang[not_tiny] = np.arctan(dtheta[not_tiny] / dphi[not_tiny])
        else:
            ang = np.arctan(dtheta / dphi)

        weights[:] = np.vstack(
            [np.ones(signal.size), np.cos(2 * ang), np.sin(2 * ang)]
        ).T

        return

    def _purge_weights(self, obs):
        """Discard special pointing matrix and dummy signal"""
        del obs.detdata[self.signal]
        del obs.detdata[self.weights]
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "pixel_dist":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if data.comm.world_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        # Establish uniform noise weights
        noise_model = UniformNoise()
        for obs in data.obs:
            obs[self.noise_model] = noise_model

        # To accumulate, we need the pixel distribution.

        if self.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                shared_flags=self.shared_flags,
                shared_flag_mask=self.shared_flag_mask,
                save_pointing=self.save_pointing,
            )
            log.info_rank("Caching pixel distribution", comm=data.comm.comm_world)
            pix_dist.apply(data)

        # Accumulation operator

        build_zmap = BuildNoiseWeighted(
            pixel_dist=self.pixel_dist,
            zmap=self.crosslinking_map,
            view=self.pixel_pointing.view,
            pixels=self.pixel_pointing.pixels,
            weights=self.weights,
            noise_model=self.noise_model,
            det_data=self.signal,
            det_data_units=self.det_data_units,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )

        for obs in data.obs:
            obs_data = data.select(obs_uid=obs.uid)
            dets = obs.select_local_detectors(detectors)
            for det in dets:
                # Pointing weights
                self._get_weights(obs_data, det)
                # Pixel numbers
                self.pixel_pointing.apply(obs_data, detectors=[det])
                # Accumulate
                build_zmap.exec(obs_data, detectors=[det])

        build_zmap.finalize(data)

        # Write out the results

        fname = os.path.join(self.output_dir, f"{self.name}.fits")
        write_healpix_fits(
            data[self.crosslinking_map], fname, nest=self.pixel_pointing.nest
        )
        log.info_rank(f"Wrote crosslinking to {fname}", comm=data.comm.comm_world)
        data[self.crosslinking_map].clear()
        del data[self.crosslinking_map]

        for obs in data.obs:
            self._purge_weights(obs)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.detector_pointing.requires()
        return req

    def _provides(self):
        return {}
