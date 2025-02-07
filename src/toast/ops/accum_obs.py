# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np
import traitlets
from astropy import units as u

from ..covariance import covariance_invert
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..pixels_io_healpix import read_healpix_hdf5
from ..pixels_io_wcs import read_wcs_fits
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Unicode, Unit, UseEnum, trait_docs
from ..utils import Logger, unit_conversion
from .operator import Operator
from .pipeline import Pipeline
from .mapmaker_utils import BuildHitMap, BuildInverseCovariance, BuildNoiseWeighted


@trait_docs
class AccumulateObservation(Operator):
    """Operator which accumulates pixel-domain observation quantities.

    This operator computes or loads the hits, inverse pixel covariance, and noise
    weighted map for each observation.  The total quantities can also be accumulated.
    The pixel distribution should have already been computed (for example from a
    sky footprint file).

    This operator uses the load_exec() functionality of the Pipeline operator to
    optionally process one observation at a time, reducing the overall memory
    requirements.

    If caching is enabled, the per-observation quantities are written to disk for
    later use.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="Directory of per-observation directories for reading / writing",
    )

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key containing the PixelDistribution object",
    )

    inverse_covariance = Unicode(
        None,
        allow_none=True,
        help="The Data key where the inverse covariance should be stored",
    )

    hits = Unicode(
        None,
        allow_none=True,
        help="The Data key where the hit map should be stored",
    )

    rcond = Unicode(
        None,
        allow_none=True,
        help="The Data key where the reciprocal condition number should be stored",
    )

    zmap = Unicode(
        None,
        allow_none=True,
        help="The Data key for the accumulated noise weighted map",
    )

    covariance = Unicode(
        None,
        allow_none=True,
        help="The Data key where the covariance should be stored",
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

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional flagging",
    )

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a Stokes weights operator",
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    rcond_threshold = Float(
        1.0e-8, help="Minimum value for inverse condition number cut."
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("sync_type")
    def _check_sync_type(self, proposal):
        check = proposal["value"]
        if check != "allreduce" and check != "alltoallv":
            raise traitlets.TraitError("Invalid communication algorithm")
        return check

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pixels = proposal["value"]
        if pixels is not None:
            if not isinstance(pixels, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pixels.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pixels

    @traitlets.validate("stokes_weights")
    def _check_stokes_weights(self, proposal):
        weights = proposal["value"]
        if weights is not None:
            if not isinstance(weights, Operator):
                raise traitlets.TraitError(
                    "stokes_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["weights", "view"]:
                if not weights.has_trait(trt):
                    msg = f"stokes_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return weights

    # Prefix for Data objects that are per-observation.
    obs_prefix = "accum_obs"

    # File root for objects on disk.
    obs_file_hits = "hits"
    obs_file_invcov = "invcov"
    obs_file_zmap = "zmap"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        self._check_inputs(data)
        accum = self._create_pipeline()

        # for ob in data.obs:
        #     ob_dir = None
        #     if self.cache_dir is not None:
        #         ob_dir = os.path.join(self.cache_dir, ob.name)

        #         if not os.path.exists(ob_dir):
        #             os.makedirs(ob_dir)

        # pipe_out = accum.apply(data, detectors=detectors)

        # # Optionally, store the inverse covariance
        # if self.inverse_covariance is not None:
        #     if self.inverse_covariance in data:
        #         del data[self.inverse_covariance]
        #     data[self.inverse_covariance] = data[self.covariance].duplicate()

        # # Extract the results
        # hits = data[self.hits]
        # cov = data[self.covariance]

        # # Invert the covariance in place
        # rcond = PixelData(cov.distribution, np.float64, n_value=1)
        # covariance_invert(
        #     cov,
        #     self.rcond_threshold,
        #     rcond=rcond,
        #     use_alltoallv=(self.sync_type == "alltoallv"),
        # )

        # rcond_good = rcond.data[:, :, 0] > 0.0
        # rcond_min = 0.0
        # rcond_max = 0.0
        # if np.count_nonzero(rcond_good) > 0:
        #     rcond_min = np.amin(rcond.data[rcond_good, 0])
        #     rcond_max = np.amax(rcond.data[rcond_good, 0])
        # if data.comm.comm_world is not None:
        #     rcond_min = data.comm.comm_world.reduce(rcond_min, root=0, op=MPI.MIN)
        #     rcond_max = data.comm.comm_world.reduce(rcond_max, root=0, op=MPI.MAX)
        # if data.comm.world_rank == 0:
        #     msg = "  Pixel covariance condition number range = "
        #     msg += f"{rcond_min:1.3e} ... {rcond_max:1.3e}"
        #     log.debug(msg)

        # # Store rcond
        # data[self.rcond] = rcond

        return

    def _check_inputs(self, data):
        for trait in "pixel_pointing", "stokes_weights":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        # Set pointing flags
        self.pixel_pointing.detector_pointing.det_mask = self.det_mask
        self.pixel_pointing.detector_pointing.det_flag_mask = self.det_flag_mask
        if hasattr(self.stokes_weights, "detector_pointing"):
            self.stokes_weights.detector_pointing.det_mask = self.det_mask
            self.stokes_weights.detector_pointing.det_flag_mask = self.det_flag_mask

        # Check that the pixel distribution exists
        if self.pixel_dist not in data:
            msg = f"The pixel distribution '{self.pixel_dist}' does not exist in data"
            raise RuntimeError(msg)

        # Check if map domain products exist and are consistent.  Also delete our
        # per-observation products if they exist.
        for map_object in [
            self.hits,
            self.rcond,
            self.zmap,
            self.inverse_covariance,
            self.covariance,
        ]:
            if map_object is None:
                continue
            if map_object in data:
                if data[map_object].distribution == data[self.pixel_dist]:
                    # Distributions are equal, just set to zero
                    data[map_object].reset()
                else:
                    # Inconsistent- delete it so that it will be re-created.
                    del data[map_object]
            obs_map_object = f"{self.obs_prefix}_{map_object}"
            if obs_map_object in data:
                del data[obs_map_object]
        # Special handling of units on the covariance
        if self.covariance is not None and self.covariance in data:
            data[self.covariance].update_units(1.0 / (self.det_data_units**2))

    def _load_products(self, data, ob_dir):
        have_all = True
        # If we are using a WCS projection, there will be a copy in the
        # pixel distribution object.
        use_wcs = hasattr(data[self.pixel_dist], "wcs")

        # Number of map values per pixel
        if self.stokes_weights.mode == "IQU":
            nvalue = 3
        elif self.stokes_weights.mode == "I":
            nvalue = 1
        else:
            raise RuntimeError(
                f"Invalid Stokes weights mode: {self.stokes_weights.mode}"
            )

        for map_object, prefix, nvalue in [
            (self.hits, self.obs_file_hits, 1),
            (self.zmap, self.obs_file_zmap, nvalue),
            (self.inverse_covariance, self.obs_file_invcov, nvalue * (nvalue + 1) / 2),
        ]:
            obs_object = f"{self.obs_prefix}_{map_object}"
            if use_wcs:
                obs_object_path = os.path.join(ob_dir, f"{prefix}.fits")
            else:
                obs_object_path = os.path.join(ob_dir, f"{prefix}.h5")
            if not os.path.exists(obs_object_path):
                have_all = False
                continue
            # Create or zero the per-observation object
            if obs_object not in data:
                data[obs_object] = PixelData(
                    distribution=data[self.pixel_dist],
                    dtype=np.float64,
                    n_value=nvalue,
                )
            else:
                data[obs_object].reset()
            # Load the object into the data
            if use_wcs:
                read_wcs_fits(data[obs_object], obs_object_path)
            else:
                read_healpix_hdf5(data[obs_object], obs_object_path)
        return have_all

    def _create_pipeline(self):
        if self.save_pointing:
            # Process all detectors at once
            accum = Pipeline(detector_sets=["ALL"])
        else:
            # Process one detector at a time.
            accum = Pipeline(detector_sets=["SINGLE"])
        accum.operators = [
            self.pixel_pointing,
            self.stokes_weights,
        ]

        # Hit map operator
        if self.hits is not None:
            accum.operators.append(
                BuildHitMap(
                    pixel_dist=self.pixel_dist,
                    hits=self.hits,
                    view=self.pixel_pointing.view,
                    pixels=self.pixel_pointing.pixels,
                    det_mask=self.det_mask,
                    det_flags=self.det_flags,
                    det_flag_mask=self.det_flag_mask,
                    shared_flags=self.shared_flags,
                    shared_flag_mask=self.shared_flag_mask,
                    sync_type=self.sync_type,
                )
            )

        # Inverse covariance.
        if self.inverse_covariance is not None:
            accum.operators.append(
                BuildInverseCovariance(
                    pixel_dist=self.pixel_dist,
                    inverse_covariance=self.inverse_covariance,
                    view=self.pixel_pointing.view,
                    pixels=self.pixel_pointing.pixels,
                    weights=self.stokes_weights.weights,
                    noise_model=self.noise_model,
                    det_data_units=self.det_data_units,
                    det_mask=self.det_mask,
                    det_flags=self.det_flags,
                    det_flag_mask=self.det_flag_mask,
                    shared_flags=self.shared_flags,
                    shared_flag_mask=self.shared_flag_mask,
                    sync_type=self.sync_type,
                )
            )

        return accum

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req.update(self.stokes_weights.requires())
        req["meta"].append(self.noise_model)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        return req

    def _provides(self):
        prov = {
            "global": [self.pixel_dist, self.hits, self.covariance, self.rcond],
            "shared": list(),
            "detdata": list(),
        }
        if self.save_pointing:
            prov["detdata"].extend([self.pixels, self.weights])
        if self.inverse_covariance is not None:
            prov["global"].append(self.inverse_covariance)
        return prov
