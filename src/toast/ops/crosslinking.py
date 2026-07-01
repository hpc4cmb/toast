# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
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
class CrossLinkingWeights(Operator):
    """Helper operator to compute the crosslinking weights.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    weights = Unicode(
        "crosslinking_weights", help="Observation detdata key for output weights"
    )

    temporary_signal = Unicode(
        "crosslinking_temp", help="Observation detdata key for temp signal"
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
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
                "det_mask",
                "quats",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):

        # Compute the detector quaternions
        self.detector_pointing.apply(data, detectors=detectors)
        quats_name = self.detector_pointing.quats

        for obs in data.obs:
            dets = obs.select_local_detectors(
                detectors, flagmask=self.detector_pointing.det_mask
            )
            exists_signal = obs.detdata.ensure(
                self.temporary_signal, detectors=dets, create_units=self.det_data_units
            )
            exists_weights = obs.detdata.ensure(
                self.weights, sample_shape=(3,), detectors=dets
            )

            for det in dets:
                signal = obs.detdata[self.temporary_signal][det]
                signal[:] = 1
                weights = obs.detdata[self.weights][det]
                quat = obs.detdata[quats_name][det]

                # Measure the scan direction wrt the local meridian for each sample
                theta, phi, _ = qa.to_iso_angles(quat)
                theta = np.pi / 2 - theta

                # Scan direction across the reference sample
                dphi = np.roll(phi, -1) - np.roll(phi, 1)
                dtheta = np.roll(theta, -1) - np.roll(theta, 1)

                # Except first and last sample
                for dx, x in (dphi, phi), (dtheta, theta):
                    dx[0] = x[1] - x[0]
                    dx[-1] = x[-1] - x[-2]

                # Scale dphi to on-sky
                dphi *= np.cos(theta)

                # Avoid overflows
                tiny = np.abs(dphi) < 1e-30
                if np.any(tiny):
                    ang = np.zeros(signal.size)
                    ang[tiny] = np.sign(dtheta[tiny]) * np.sign(dphi[tiny]) * np.pi / 2
                    not_tiny = np.logical_not(tiny)
                    ang[not_tiny] = np.arctan(dtheta[not_tiny] / dphi[not_tiny])
                else:
                    ang = np.arctan(dtheta / dphi)

                weights[:] = np.vstack(
                    [np.ones(signal.size), np.cos(2 * ang), np.sin(2 * ang)]
                ).T

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.temporary_signal)
        prov["detdata"].append(self.weights)
        return prov


@trait_docs
class CrossLinking(Operator):
    """Evaluate an ACT-style crosslinking map

    The result is a 3-component map that needs to be processed as
      crosslinking = SQRT(map[1]**2 + map[2]**2) / map[0]
    for the crosslinking statistic.  This is not done in the operator to allow
    adding crosslinking maps together.
    """

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

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
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
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional telescope flagging",
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    crosslinking_map = Unicode(
        "crosslinking_map",
        help="The output Data object for the crosslinking map",
    )

    noise_model = Unicode(
        "uniform_noise_weights",
        help="The output noise model with uniform weights",
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    save_pointing = Bool(False, help="If True, do not clear pixel numbers after use")

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

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
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "pixel_dist":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if data.comm.world_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        # When pipelining our operations below, ensure that detector pointing
        # uses the same mask that the user requested in this operator.
        save_pointing_det_mask = self.pixel_pointing.detector_pointing.det_mask

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

        # Weights
        cross_weights = CrossLinkingWeights(
            detector_pointing=self.pixel_pointing.detector_pointing,
            det_data_units=self.det_data_units,
        )

        # Accumulation operator
        build_zmap = BuildNoiseWeighted(
            pixel_dist=self.pixel_dist,
            zmap=self.crosslinking_map,
            view=self.pixel_pointing.view,
            pixels=self.pixel_pointing.pixels,
            weights=cross_weights.weights,
            noise_model=self.noise_model,
            det_data=cross_weights.temporary_signal,
            det_data_units=self.det_data_units,
            det_mask=self.det_mask,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )

        # Cleanup
        cleanup = Delete(
            detdata=[cross_weights.weights, cross_weights.temporary_signal],
        )

        # Our accumulation pipeline
        accum_pipe = Pipeline(
            detector_sets=["SINGLE"],
            operators=[
                cross_weights,
                self.pixel_pointing,
                build_zmap,
                cleanup,
            ]
        )
        accum_pipe.apply(data, detectors=detectors)

        # Restore detector pointing mask if it was different
        self.pixel_pointing.detector_pointing.det_mask = save_pointing_det_mask

    def _finalize(self, data, **kwargs):
        log = Logger.get()

        # Write out the results
        fname = os.path.join(self.output_dir, f"{self.name}.fits")
        data[self.crosslinking_map].write(fname)
        log.info_rank(f"Wrote crosslinking to {fname}", comm=data.comm.comm_world)

        # Cleanup
        data[self.crosslinking_map].clear()
        del data[self.crosslinking_map]

    def _requires(self):
        req = self.pixel_pointing.detector_pointing.requires()
        return req

    def _provides(self):
        return {}
