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
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import read_healpix_hdf5, write_healpix_hdf5
from ..pixels_io_wcs import read_wcs_fits, write_wcs_fits
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

    This operator is designed to have its exec() method called on a single observation,
    for example as part of a call to Operator.load_exec().  An exception will be raised
    if applied to a Data object with multiple observations.

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

    overwrite_cache = Bool(
        False,
        help="If True and using a cache, overwrite any inputs found there",
    )

    cache_only = Bool(
        False,
        help="If True, do not accumulate the total products. Useful for pre-caching",
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
        help="If not None, compute the covariance and store it here",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key for the timestream data",
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

    obs_pointing = Bool(
        False,
        help="If True, expand pointing for all detectors at once in an observation",
    )

    save_pointing = Bool(
        False,
        help="If True, leave expanded pointing after processing the observation",
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

    # File root name for objects on disk.
    obs_file_hits = "hits"
    obs_file_invcov = "invcov"
    obs_file_zmap = "zmap"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if len(data.obs) > 1:
            msg = "Each call to exec() should be used on a Data object with no "
            msg += "more than one observation per group."
            log.error_rank(msg, comm=data.comm.comm_group)
            raise RuntimeError(msg)
        elif len(data.obs) == 0:
            # Nothing to do
            return

        # The observation we are working with
        the_obs = data.obs[0]

        # Check inputs for consistency and create any map domain objects that do not
        # yet exist.
        self._check_inputs(data)

        # Compute the cache location for this observation and
        # check if we have all the products we need.
        use_cache, ob_dir = self._check_cache(the_obs)
        all_cached = self._load_obs_products(data, ob_dir)

        if not all_cached:
            # We need to compute the products.

            # The observation pixel distribution
            obs_pixel_dist = f"{self.name}_{self.pixel_dist}"

            # Pipeline to accumulate a single observation
            accum_pipe = self._create_pipeline(data[obs_pixel_dist])

            # Run the pipeline to accumulate the products for this observation,
            # but do not call "finalize" on the pipeline, which syncs data products.
            self._compute_obs_products(data, accum_pipe)

        if not self.cache_only:
            # Accumulate the per-observation data to the global products.  This sums the
            # per process submap data (which has not synced yet).
            self._accumulate(data)

        if use_cache and not all_cached:
            # We are using the cache and we had to compute the per-observation
            # products.  Write these out.  This will first call finalize() on the
            # pipeline to sync the per-observation products before writing.
            self._write_obs_products(data, ob_dir, accum_pipe)

    def _n_stokes(self):
        # Number of map values per pixel
        if self.stokes_weights.mode == "IQU":
            nvalue = 3
        elif self.stokes_weights.mode == "I":
            nvalue = 1
        else:
            raise RuntimeError(
                f"Invalid Stokes weights mode: {self.stokes_weights.mode}"
            )
        return nvalue

    def _check_obs_pixel_dist(self, data):
        pix_dist = data[self.pixel_dist]
        obs_pixel_dist = f"{self.name}_{self.pixel_dist}"
        if obs_pixel_dist in data:
            # Verify that it is equal to the global pixel dist with the exception of the
            # communicator.  This mimics the __eq__ method of the PixelDistribution but
            # with the different communicator check.
            obs_pix_dist = data[obs_pixel_dist]
            dist_equal = True
            if obs_pix_dist.n_pix != pix_dist.n_pix:
                dist_equal = False
            if obs_pix_dist.n_submap != pix_dist.n_submap:
                dist_equal = False
            if obs_pix_dist.n_pix_submap != pix_dist.n_pix_submap:
                dist_equal = False
            if not np.array_equal(obs_pix_dist.local_submaps, pix_dist.local_submaps):
                dist_equal = False
            if obs_pix_dist.comm is None and data.comm.comm_group is not None:
                dist_equal = False
            if obs_pix_dist.comm is not None and data.comm.comm_group is None:
                dist_equal = False
            if obs_pix_dist.comm is not None:
                comp = MPI.Comm.Compare(obs_pix_dist.comm, data.comm.comm_group)
                if comp not in (MPI.IDENT, MPI.CONGRUENT):
                    dist_equal = False
            if not dist_equal:
                # Delete it
                del data[obs_pixel_dist]
        if obs_pixel_dist not in data:
            # Create it
            data[obs_pixel_dist] = PixelDistribution(
                n_pix=pix_dist.n_pix,
                n_submap=pix_dist.n_submap,
                local_submaps=pix_dist.local_submaps,
                comm=data.comm.comm_group,
            )
        return obs_pixel_dist

    def _check_inputs(self, data):
        for trait in "pixel_pointing", "stokes_weights":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if self.covariance is not None and self.inverse_covariance is None:
            msg = "You cannot build the covariance without also specifying"
            msg += " the inverse covariance to accumulate."
            raise RuntimeError(msg)
        if self.covariance is None and self.rcond is not None:
            msg = "The covariance and rcond traits should be either both set"
            msg += " to None, or both set to the data key to use."
            raise RuntimeError(msg)
        if self.covariance is not None and self.rcond is None:
            msg = "The covariance and rcond traits should be either both set"
            msg += " to None, or both set to the data key to use."
            raise RuntimeError(msg)

        # Set pointing flags
        self.pixel_pointing.detector_pointing.det_mask = self.det_mask
        self.pixel_pointing.detector_pointing.det_flag_mask = self.det_flag_mask
        if hasattr(self.stokes_weights, "detector_pointing"):
            self.stokes_weights.detector_pointing.det_mask = self.det_mask
            self.stokes_weights.detector_pointing.det_flag_mask = self.det_flag_mask

        # Check that the global pixel distribution exists
        if self.pixel_dist not in data:
            msg = f"The pixel distribution '{self.pixel_dist}' does not exist in data"
            raise RuntimeError(msg)

        # Ensure that the single observation pixel distribution exists
        obs_pixel_dist = self._check_obs_pixel_dist(data)

        n_value = self._n_stokes()

        for map_object, nval, map_units in [
            (self.hits, 1, u.dimensionless_unscaled),
            (self.zmap, n_value, self.det_data_units),
            (
                self.inverse_covariance,
                n_value * (n_value + 1) / 2,
                1.0 / (self.det_data_units**2),
            ),
        ]:
            if map_object is None:
                continue
            if map_object in data:
                if data[map_object].distribution != data[self.pixel_dist]:
                    # Inconsistent.  Delete and re-create
                    del data[map_object]
            if map_object not in data:
                data[map_object] = PixelData(
                    data[self.pixel_dist],
                    np.float64,
                    n_value=nval,
                    units=map_units,
                )
            obs_object = f"{self.name}_{map_object}"
            if obs_object in data:
                if data[obs_object].distribution != data[obs_pixel_dist]:
                    # Inconsistent.  Delete and re-create
                    del data[obs_object]
            if obs_object not in data:
                data[obs_object] = PixelData(
                    data[obs_pixel_dist],
                    np.float64,
                    n_value=nval,
                    units=map_units,
                )
            else:
                # Zero for use with the current observation.
                data[obs_object].reset()

    def _check_cache(self, obs):
        if self.cache_dir is None:
            ob_dir = None
            use_cache = False
        else:
            ob_dir = os.path.join(self.cache_dir, obs.name)
            use_cache = True
            if obs.comm.group_rank == 0:
                if not os.path.exists(ob_dir):
                    os.makedirs(ob_dir)
                # If we are overwriting the observation products,
                # delete them from disk now.
                if self.overwrite_cache:
                    for map_object, prefix in [
                        (self.hits, self.obs_file_hits),
                        (self.zmap, self.obs_file_zmap),
                        (self.inverse_covariance, self.obs_file_invcov),
                    ]:
                        if map_object is None:
                            continue
                        obs_object = os.path.join(ob_dir, f"{self.name}_{map_object}")
                        if os.path.exists(obs_object):
                            os.remove(obs_object)
            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()
        return use_cache, ob_dir

    def _load_obs_products(self, data, ob_dir):
        """Load the per-observation objects from disk if they exist."""
        have_all = True

        # If we are using a WCS projection, there will be a copy of wcs in the
        # pixel distribution object.
        use_wcs = hasattr(data[self.pixel_dist], "wcs")

        for map_object, prefix in [
            (self.hits, self.obs_file_hits),
            (self.zmap, self.obs_file_zmap),
            (self.inverse_covariance, self.obs_file_invcov),
        ]:
            if map_object is None:
                continue
            obs_object = f"{self.name}_{map_object}"

            if ob_dir is None:
                # Not using cache
                have_all = False
                continue

            if use_wcs:
                obs_object_path = os.path.join(ob_dir, f"{prefix}.fits")
            else:
                obs_object_path = os.path.join(ob_dir, f"{prefix}.h5")

            obs_have_obj = True
            if data.comm.group_rank == 0:
                if not os.path.exists(obs_object_path):
                    obs_have_obj = False
            if data.comm.comm_group is not None:
                obs_have_obj = data.comm.comm_group.bcast(obs_have_obj, root=0)
            if not obs_have_obj:
                have_all = False
                continue

            # Load the object into the data
            if use_wcs:
                read_wcs_fits(data[obs_object], obs_object_path)
            else:
                read_healpix_hdf5(data[obs_object], obs_object_path)
        return have_all

    def _create_pipeline(self, ob_dist):
        if self.obs_pointing:
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
                    pixel_dist=ob_dist,
                    hits=f"{self.name}_{self.hits}",
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
                    pixel_dist=ob_dist,
                    inverse_covariance=f"{self.name}_{self.inverse_covariance}",
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

        # Noise weighted map
        if self.zmap is not None:
            accum.operators.append(
                BuildNoiseWeighted(
                    pixel_dist=ob_dist,
                    zmap=f"{self.name}_{self.zmap}",
                    view=self.pixel_pointing.view,
                    pixels=self.pixel_pointing.pixels,
                    weights=self.stokes_weights.weights,
                    noise_model=self.noise_model,
                    det_data=self.det_data,
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

    def _compute_obs_products(self, data, accum_pipe):
        # The observation we are working with
        the_obs = data.obs[0]

        # We intentionally DO NOT call finalize() / apply() here.  We want to
        # have the raw, per-process hits / noise weighted map / inverse covariance.
        # We can then accumulate these to the global objects.
        accum_pipe.exec(data)

        if self.obs_pointing and not self.save_pointing:
            # We created full detector pointing for this observation, but we are
            # not saving it.  Clean it up now.
            del the_obs.detdata[self.pixel_pointing.pixels]
            del the_obs.detdata[self.stokes_weights.weights]
            del the_obs.detdata[self.pixel_pointing.detector_pointing.quats]

    def _write_obs_products(self, data, ob_dir, accum_pipe):
        if ob_dir is None:
            # Not caching anything
            return

        # Before writing out the data, we sync it.
        accum_pipe.finalize()

        # If we are using a WCS projection, there will be a copy of wcs in the
        # pixel distribution object.
        use_wcs = hasattr(data[self.pixel_dist], "wcs")

        for map_object, prefix in [
            (self.hits, self.obs_file_hits),
            (self.zmap, self.obs_file_zmap),
            (self.inverse_covariance, self.obs_file_invcov),
        ]:
            if map_object is None:
                continue
            obs_object = f"{self.name}_{map_object}"

            if use_wcs:
                obs_object_path = os.path.join(ob_dir, f"{prefix}.fits")
            else:
                obs_object_path = os.path.join(ob_dir, f"{prefix}.h5")

            # Write the object
            if use_wcs:
                write_wcs_fits(data[obs_object], obs_object_path)
            else:
                write_healpix_hdf5(data[obs_object], obs_object_path, nest=True)

    def _accumulate(self, data):
        # The per-observation objects have not yet been synchronized.  So each process
        # can just accumulate their locally hit submaps.

        n_value = self._n_stokes()

        for map_object, nval in [
            (self.hits, 1),
            (self.zmap, n_value),
            (self.inverse_covariance, n_value * (n_value + 1) / 2),
        ]:
            if map_object is None:
                continue
            obs_object = f"{self.name}_{map_object}"
            data[map_object].data[:] += data[obs_object].data

    def _finalize(self, data, **kwargs):
        log = Logger.get()

        if self.cache_only:
            # Nothing to do
            return

        # Sync accumulated products
        for map_object in [
            self.hits,
            self.zmap,
            self.inverse_covariance,
        ]:
            if map_object is None:
                continue
            if self.sync_type == "alltoallv":
                data[map_object].sync_alltoallv()
            else:
                data[map_object].sync_allreduce()

        # Cleanup per-observation objects

        # Invert the total covariance
        if self.covariance is not None:
            # Copy the inverse
            data[self.covariance] = data[self.inverse_covariance].duplicate()
            # Update units
            data[self.covariance].update_units(1.0 / (self.det_data_units**2))
            # Invert in place
            covariance_invert(
                data[self.covariance],
                self.rcond_threshold,
                rcond=data[self.rcond],
                use_alltoallv=(self.sync_type == "alltoallv"),
            )
            rcond = data[self.rcond]
            rcond_good = rcond.data[:, :, 0] > 0.0
            rcond_min = 0.0
            rcond_max = 0.0
            if np.count_nonzero(rcond_good) > 0:
                rcond_min = np.amin(rcond.data[rcond_good, 0])
                rcond_max = np.amax(rcond.data[rcond_good, 0])
            if data.comm.comm_world is not None:
                rcond_min = data.comm.comm_world.reduce(rcond_min, root=0, op=MPI.MIN)
                rcond_max = data.comm.comm_world.reduce(rcond_max, root=0, op=MPI.MAX)
            if data.comm.world_rank == 0:
                msg = "  Pixel covariance condition number range = "
                msg += f"{rcond_min:1.3e} ... {rcond_max:1.3e}"
                log.debug(msg)

    def _requires(self):
        req = self.pixel_pointing.requires()
        req.update(self.stokes_weights.requires())
        req["global"].append(self.pixel_dist)
        req["meta"].append(self.noise_model)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        return req

    def _provides(self):
        prov = {
            "global": list(),
            "shared": list(),
            "detdata": list(),
        }
        for map_obj in [
            self.hits,
            self.zmap,
            self.inverse_covariance,
            self.covariance,
            self.rcond,
        ]:
            if map_obj is not None:
                prov["global"].append(map_obj)
        if self.save_pointing:
            prov["detdata"].extend([self.pixels, self.weights])
        return prov
