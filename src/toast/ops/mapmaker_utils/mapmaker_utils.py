# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ...accelerator import ImplementationType
from ...covariance import covariance_invert
from ...mpi import MPI
from ...observation import default_values as defaults
from ...pixels import PixelData
from ...timing import function_timer
from ...traits import Bool, Float, Instance, Int, Unicode, Unit, UseEnum, trait_docs
from ...utils import Logger, unit_conversion
from ..operator import Operator
from ..pipeline import Pipeline
from ..pointing import BuildPixelDistribution
from .kernels import build_noise_weighted, cov_accum_diag_hits, cov_accum_diag_invnpp


@trait_docs
class BuildHitMap(Operator):
    """Operator which builds a hitmap.

    Given the pointing matrix for each detector, accumulate the hit map.  The PixelData
    object containing the hit map is returned by the finalize() method.

    If any samples have compromised telescope pointing, those pixel indices should
    have already been set to a negative value by the operator that generated the
    pointing matrix.

    Although individual detector flags do not impact the pointing per se, they can be
    used with this operator in order to produce a hit map that is consistent with other
    pixel space products.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        None,
        allow_none=True,
        help="The Data key containing the submap distribution",
    )

    hits = Unicode("hits", help="The Data key for the output hit map")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
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
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    pixels = Unicode(defaults.pixels, help="Observation detdata key for pixel indices")

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("sync_type")
    def _check_sync_type(self, proposal):
        check = proposal["value"]
        if check != "allreduce" and check != "alltoallv":
            raise traitlets.TraitError("Invalid communication algorithm")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        if self.pixel_dist is None:
            raise RuntimeError(
                "You must set the 'pixel_dist' trait before calling exec()"
            )

        if self.pixel_dist not in data:
            msg = "Data does not contain submap distribution '{}'".format(
                self.pixel_dist
            )
            raise RuntimeError(msg)

        dist = data[self.pixel_dist]
        log.verbose_rank(
            f"Building hit map with pixel_distribution {self.pixel_dist}",
            comm=data.comm.comm_world,
        )

        hits = None
        if self.hits in data:
            # We have an existing map from a previous call.  Verify
            # the distribution and nnz.
            if data[self.hits].distribution != dist:
                msg = "Existing hits '{}' has different data distribution".format(
                    self.hits
                )
                log.error(msg)
                raise RuntimeError(msg)
            if data[self.hits].n_value != 1:
                msg = "Existing hits '{}' has {} nnz, not 1".format(
                    self.hits, data[self.hits].n_value
                )
                log.error(msg)
                raise RuntimeError(msg)
            hits = data[self.hits]
        else:
            data[self.hits] = PixelData(dist, np.int64, n_value=1)
            hits = data[self.hits]

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(selection=detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # The pixels and weights view for this observation
            pix = ob.view[self.view].detdata[self.pixels]
            if self.det_flags is not None:
                flgs = ob.view[self.view].detdata[self.det_flags]
            else:
                flgs = [None for x in pix]
            if self.shared_flags is not None:
                shared_flgs = ob.view[self.view].shared[self.shared_flags]
            else:
                shared_flgs = [None for x in pix]

            # Process every data view
            for pview, fview, shared_fview in zip(pix, flgs, shared_flgs):
                for det in dets:
                    # Get local submap and pixels
                    local_sm, local_pix = dist.global_pixel_to_submap(pview[det])

                    # Samples with telescope pointing problems are already flagged in
                    # the pointing operators by setting the pixel numbers to a negative
                    # value.  Here we optionally apply detector flags to the local
                    # pixel numbers to flag more samples.

                    # Apply the flags if needed
                    if self.det_flags is not None:
                        local_pix[(fview[det] & self.det_flag_mask) != 0] = -1
                    if self.shared_flags is not None:
                        local_pix[(shared_fview & self.shared_flag_mask) != 0] = -1

                    cov_accum_diag_hits(
                        dist.n_local_submap,
                        dist.n_pix_submap,
                        1,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        hits.raw,
                        impl=implementation,
                        use_accel=use_accel,
                    )
        return

    def _finalize(self, data, **kwargs):
        if self.hits in data:
            if self.sync_type == "alltoallv":
                data[self.hits].sync_alltoallv()
            else:
                data[self.hits].sync_allreduce()
        return

    def _requires(self):
        req = {
            "global": [self.pixel_dist],
            "shared": list(),
            "detdata": [self.pixels],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"global": [self.hits]}
        return prov

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        # NOTE: the kernels called do not follow the proper pattern yet
        return False


@trait_docs
class BuildInverseCovariance(Operator):
    """Operator which builds a pixel-space diagonal inverse noise covariance.

    Given the pointing matrix and noise model for each detector, accumulate the inverse
    noise covariance:

    .. math::
        N_pp'^{-1} = \\left( P^T N_tt'^{-1} P \\right)

    The PixelData object containing this is returned by the finalize() method.

    If any samples have compromised telescope pointing, those pixel indices should
    have already been set to a negative value by the operator that generated the
    pointing matrix.  Individual detector flags can optionally be applied to
    timesamples when accumulating data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        None,
        allow_none=True,
        help="The Data key containing the submap distribution",
    )

    inverse_covariance = Unicode(
        "inv_covariance", help="The Data key for the output inverse covariance"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    pixels = Unicode("pixels", help="Observation detdata key for pixel indices")

    weights = Unicode("weights", help="Observation detdata key for Stokes weights")

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("sync_type")
    def _check_sync_type(self, proposal):
        check = proposal["value"]
        if check != "allreduce" and check != "alltoallv":
            raise traitlets.TraitError("Invalid communication algorithm")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        if self.pixel_dist is None:
            raise RuntimeError(
                "You must set the 'pixel_dist' trait before calling exec()"
            )

        if self.pixel_dist not in data:
            msg = "Data does not contain submap distribution '{}'".format(
                self.pixel_dist
            )
            raise RuntimeError(msg)

        dist = data[self.pixel_dist]
        log.verbose_rank(
            f"Building inverse covariance with pixel_distribution {self.pixel_dist}",
            comm=data.comm.comm_world,
        )

        invcov_units = 1.0 / (self.det_data_units**2)

        invcov = None
        weight_nnz = None
        cov_nnz = None

        # We will store the lower triangle of the covariance.  This operator requires
        # that all detectors in all observations have the same number of non-zeros
        # in the pointing matrix.

        if self.inverse_covariance in data:
            # We have an existing map from a previous call.  Verify
            # the distribution and units.
            if data[self.inverse_covariance].distribution != dist:
                msg = "Existing inv cov '{}' has different data distribution".format(
                    self.inverse_covariance
                )
                log.error(msg)
                raise RuntimeError(msg)
            if data[self.inverse_covariance].units != invcov_units:
                msg = "Existing inv cov '{}' has different units".format(
                    self.inverse_covariance
                )
                log.error(msg)
                raise RuntimeError(msg)
            invcov = data[self.inverse_covariance]
            cov_nnz = invcov.n_value
            weight_nnz = int((np.sqrt(1 + 8 * cov_nnz) - 1) // 2)
        else:
            # We are creating a new data object
            weight_nnz = 0
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(selection=detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                if self.weights in ob.detdata:
                    if len(ob.detdata[self.weights].detector_shape) == 1:
                        cur_nnz = 1
                    else:
                        cur_nnz = ob.detdata[self.weights].detector_shape[1]
                    weight_nnz = max(weight_nnz, cur_nnz)
                else:
                    raise RuntimeError(
                        f"Stokes weights '{self.weights}' not in obs {ob.name}"
                    )
            if data.comm.comm_world is not None:
                weight_nnz = data.comm.comm_world.allreduce(weight_nnz, op=MPI.MAX)
            cov_nnz = int(weight_nnz * (weight_nnz + 1) // 2)
            data[self.inverse_covariance] = PixelData(
                dist, np.float64, n_value=cov_nnz, units=invcov_units
            )
            invcov = data[self.inverse_covariance]

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(selection=detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that the noise model exists
            if self.noise_model not in ob:
                msg = "Noise model {} does not exist in observation {}".format(
                    self.noise_model, ob.name
                )
                raise RuntimeError(msg)

            noise = ob[self.noise_model]

            # The pixels and weights view for this observation
            pix = ob.view[self.view].detdata[self.pixels]
            wts = ob.view[self.view].detdata[self.weights]
            if self.det_flags is not None:
                flgs = ob.view[self.view].detdata[self.det_flags]
            else:
                flgs = [None for x in wts]
            if self.shared_flags is not None:
                shared_flgs = ob.view[self.view].shared[self.shared_flags]
            else:
                shared_flgs = [None for x in wts]

            # Process every data view
            for pview, wview, fview, shared_fview in zip(pix, wts, flgs, shared_flgs):
                for det in dets:
                    # We require that the pointing matrix has the same number of
                    # non-zero elements for every detector and every observation.
                    # We check that here.

                    check_nnz = None
                    if len(wview.detector_shape) == 1:
                        check_nnz = 1
                    else:
                        check_nnz = wview.detector_shape[1]
                    if check_nnz != weight_nnz:
                        msg = "observation '{}', detector '{}', pointing weights '{}' has {} nnz, not {}".format(
                            ob.name, det, self.weights, check_nnz, weight_nnz
                        )
                        raise RuntimeError(msg)

                    # Get local submap and pixels
                    local_sm, local_pix = dist.global_pixel_to_submap(pview[det])

                    # Get the detector weight from the noise model.
                    detweight = noise.detector_weight(det)

                    # Samples with telescope pointing problems are already flagged in
                    # the pointing operators by setting the pixel numbers to a negative
                    # value.  Here we optionally apply detector flags to the local
                    # pixel numbers to flag more samples.

                    # Apply the flags if needed
                    if self.det_flags is not None:
                        local_pix[(fview[det] & self.det_flag_mask) != 0] = -1
                    if self.shared_flags is not None:
                        local_pix[(shared_fview & self.shared_flag_mask) != 0] = -1

                    # Accumulate
                    cov_accum_diag_invnpp(
                        dist.n_local_submap,
                        dist.n_pix_submap,
                        weight_nnz,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        wview[det].reshape(-1),
                        detweight.to_value(invcov_units),
                        invcov.raw,
                        impl=implementation,
                        use_accel=use_accel,
                    )
        return

    def _finalize(self, data, **kwargs):
        if self.inverse_covariance in data:
            if self.sync_type == "alltoallv":
                data[self.inverse_covariance].sync_alltoallv()
            else:
                data[self.inverse_covariance].sync_allreduce()
        return

    def _requires(self):
        req = {
            "global": [self.pixel_dist],
            "meta": [self.noise_model],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"global": [self.inverse_covariance]}
        return prov

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        # NOTE: the kernels called do not follow the proper pattern yet
        return False


@trait_docs
class BuildNoiseWeighted(Operator):
    """Operator which builds a noise-weighted map.

    Given the pointing matrix and noise model for each detector, accumulate the noise
    weighted map:

    .. math::
        Z_p = P^T N_tt'^{-1} d

    Which is the timestream data waited by the diagonal time domain noise covariance
    and projected into pixel space.  The PixelData object containing this is returned
    by the finalize() method.

    If any samples have compromised telescope pointing, those pixel indices should
    have already been set to a negative value by the operator that generated the
    pointing matrix.  Individual detector flags can optionally be applied to
    timesamples when accumulating data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        None,
        allow_none=True,
        help="The Data key containing the submap distribution",
    )

    zmap = Unicode("zmap", help="The Data key for the output noise weighted map")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key for the timestream data",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    pixels = Unicode("pixels", help="Observation detdata key for pixel indices")

    weights = Unicode("weights", help="Observation detdata key for Stokes weights")

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("sync_type")
    def _check_sync_type(self, proposal):
        check = proposal["value"]
        if check != "allreduce" and check != "alltoallv":
            raise traitlets.TraitError("Invalid communication algorithm")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        if self.pixel_dist is None:
            raise RuntimeError(
                "You must set the 'pixel_dist' trait before calling exec()"
            )

        if self.pixel_dist not in data:
            msg = "Data does not contain submap distribution '{}'".format(
                self.pixel_dist
            )
            raise RuntimeError(msg)

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        dist = data[self.pixel_dist]
        if data.comm.world_rank == 0:
            log.verbose(
                "Building noise weighted map with pixel_distribution {}".format(
                    self.pixel_dist
                )
            )

        detwt_units = 1.0 / (self.det_data_units**2)
        zmap_units = 1.0 / self.det_data_units

        zmap = None
        weight_nnz = None

        # This operator requires that all detectors in all observations have the same
        # number of non-zeros in the pointing matrix.

        if self.zmap in data:
            # We have an existing map from a previous call.  Verify
            # the distribution and units
            if data[self.zmap].distribution != dist:
                msg = "Existing zmap '{}' has different data distribution".format(
                    self.zmap
                )
                log.error(msg)
                raise RuntimeError(msg)
            if data[self.zmap].units != zmap_units:
                msg = f"Existing zmap '{self.zmap}' has different units"
                msg += f" ({data[self.zmap].units}) != {zmap_units}"
                log.error(msg)
                raise RuntimeError(msg)
            zmap = data[self.zmap]
            weight_nnz = zmap.n_value
        else:
            weight_nnz = 0
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(selection=detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                if self.weights in ob.detdata:
                    if len(ob.detdata[self.weights].detector_shape) == 1:
                        weight_nnz = 1
                    else:
                        weight_nnz = ob.detdata[self.weights].detector_shape[1]
                else:
                    raise RuntimeError(
                        f"Stokes weights '{self.weights}' not in obs {ob.name}"
                    )
            if data.comm.comm_world is not None:
                weight_nnz = data.comm.comm_world.allreduce(weight_nnz, op=MPI.MAX)
            data[self.zmap] = PixelData(
                dist, np.float64, n_value=weight_nnz, units=zmap_units
            )
            zmap = data[self.zmap]

        if use_accel:
            if not zmap.accel_exists():
                # Does not yet exist, create it
                log.verbose_rank(
                    f"Operator {self.name} zmap not yet on device, creating",
                    comm=data.comm.comm_group,
                )
                zmap.accel_create(f"{self.name}", zero_out=True)
                zmap.accel_used(True)
            elif not zmap.accel_in_use():
                # Device copy not currently in use
                log.verbose_rank(
                    f"Operator {self.name} zmap:  copy host to device",
                    comm=data.comm.comm_group,
                )
                zmap.accel_update_device()
            else:
                log.verbose_rank(
                    f"Operator {self.name} zmap:  already in use on device",
                    comm=data.comm.comm_group,
                )
        else:
            if zmap.accel_in_use():
                # Device copy in use, but we are running on host.  Update host
                log.verbose_rank(
                    f"Operator {self.name} zmap:  update host from device",
                    comm=data.comm.comm_group,
                )
                zmap.accel_update_host()

        # # DEBUGGING
        # restore_dev = False
        # prefix="HOST"
        # if zmap.accel_in_use():
        #     zmap.accel_update_host()
        #     restore_dev = True
        #     prefix="DEVICE"
        # zmap_min = np.amin(zmap.data)
        # zmap_max = np.amax(zmap.data)
        # print(f"{prefix} {self.name} dets {detectors} starting zmap output:  min={zmap_min}, max={zmap_max}", flush=True)
        # for ism, sm in enumerate(zmap.data):
        #     for ismpix, smpix in enumerate(sm):
        #         if np.count_nonzero(smpix) > 0:
        #             print(f"{prefix} {self.name} ({ism}, {ismpix}) = {smpix}", flush=True)
        # if restore_dev:
        #     zmap.accel_update_device()

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(selection=detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that the noise model exists
            if self.noise_model not in ob:
                msg = "Noise model {} does not exist in observation {}".format(
                    self.noise_model, ob.name
                )
                raise RuntimeError(msg)

            noise = ob[self.noise_model]

            # Scale factor to get timestream data into desired units.
            data_scale = unit_conversion(
                ob.detdata[self.det_data].units, self.det_data_units
            )

            # Detector inverse variance weights
            detweights = np.array(
                [noise.detector_weight(x).to_value(detwt_units) for x in dets],
                dtype=np.float64,
            )

            # Pre-multiply the detector inverse variance weights by the
            # data scaling factor, so that this combination is applied
            # in the compiled kernel below.
            detweights *= data_scale

            pix_indx = ob.detdata[self.pixels].indices(dets)
            weight_indx = ob.detdata[self.weights].indices(dets)
            data_indx = ob.detdata[self.det_data].indices(dets)

            n_weight_dets = ob.detdata[self.weights].data.shape[0]

            if self.det_flags is not None:
                flag_indx = ob.detdata[self.det_flags].indices(dets)
                flag_data = ob.detdata[self.det_flags].data
            else:
                flag_indx = np.array([-1], dtype=np.int32)
                flag_data = np.zeros((1, 1), dtype=np.uint8)

            if self.shared_flags is not None:
                shared_flag_data = ob.shared[self.shared_flags].data
            else:
                shared_flag_data = np.zeros(1, dtype=np.uint8)

            build_noise_weighted(
                zmap.distribution.global_submap_to_local,
                zmap.data,
                pix_indx,
                ob.detdata[self.pixels].data,
                weight_indx,
                ob.detdata[self.weights].data,
                data_indx,
                ob.detdata[self.det_data].data,
                flag_indx,
                flag_data,
                detweights,
                self.det_flag_mask,
                ob.intervals[self.view].data,
                shared_flag_data,
                self.shared_flag_mask,
                impl=implementation,
                use_accel=use_accel,
            )

        # # DEBUGGING
        # restore_dev = False
        # prefix="HOST"
        # if zmap.accel_in_use():
        #     zmap.accel_update_host()
        #     restore_dev = True
        #     prefix="DEVICE"
        # zmap_min = np.amin(zmap.data)
        # zmap_max = np.amax(zmap.data)
        # print(f"{prefix} {self.name} dets {detectors} ending zmap output:  min={zmap_min}, max={zmap_max}", flush=True)
        # for ism, sm in enumerate(zmap.data):
        #     for ismpix, smpix in enumerate(sm):
        #         if np.count_nonzero(smpix) > 0:
        #             print(f"{prefix} {self.name} ({ism}, {ismpix}) = {smpix}", flush=True)
        # if restore_dev:
        #     zmap.accel_update_device()

        return

    def _finalize(self, data, use_accel=None, **kwargs):
        if self.zmap in data:
            log = Logger.get()
            # We have called exec() at least once
            restore_device = False
            if data[self.zmap].accel_in_use():
                log.verbose_rank(
                    f"Operator {self.name} finalize calling zmap update self",
                    comm=data.comm.comm_group,
                )
                restore_device = True
                data[self.zmap].accel_update_host()
            if self.sync_type == "alltoallv":
                data[self.zmap].sync_alltoallv()
            else:
                data[self.zmap].sync_allreduce()

            zmap_good = data[self.zmap].data[:, :, 0] != 0.0
            zmap_min = np.zeros((data[self.zmap].n_value), dtype=np.float64)
            zmap_max = np.zeros((data[self.zmap].n_value), dtype=np.float64)
            if np.count_nonzero(zmap_good) > 0:
                zmap_min[:] = np.amin(data[self.zmap].data[zmap_good, :], axis=0)
                zmap_max[:] = np.amax(data[self.zmap].data[zmap_good, :], axis=0)
            all_zmap_min = np.zeros_like(zmap_min)
            all_zmap_max = np.zeros_like(zmap_max)
            if data.comm.comm_world is not None:
                data.comm.comm_world.Reduce(zmap_min, all_zmap_min, op=MPI.MIN, root=0)
                data.comm.comm_world.Reduce(zmap_max, all_zmap_max, op=MPI.MAX, root=0)
            if data.comm.world_rank == 0:
                msg = f"  Noise-weighted map pixel value range:\n"
                for m in range(data[self.zmap].n_value):
                    msg += f"    map {m} {zmap_min[m]:1.3e} ... {zmap_max[m]:1.3e}"
                log.debug(msg)

            if restore_device:
                log.verbose_rank(
                    f"Operator {self.name} finalize calling zmap update device",
                    comm=data.comm.comm_group,
                )
                data[self.zmap].accel_update_device()
        return

    def _requires(self):
        req = {
            "global": [self.pixel_dist],
            "meta": [self.noise_model],
            "shared": list(),
            "detdata": [self.pixels, self.weights, self.det_data],
            "intervals": list(),
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
            "global": [self.zmap],
        }
        return prov

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        return True


@trait_docs
class CovarianceAndHits(Operator):
    """Operator which builds the pixel-space diagonal noise covariance and hit map.

    Frequently the first step in map making is to determine what pixels on the sky
    have been covered and build the diagonal noise covariance.  During the construction
    of the covariance we can cut pixels that are poorly conditioned.

    This operator runs the pointing operator and builds the PixelDist instance
    describing how submaps are distributed among processes.  It builds the hit map
    and the inverse covariance and then inverts this with a threshold on the condition
    number in each pixel.

    NOTE:  The pixel pointing operator must have the "pixels", "create_dist"
    traits, which will be set by this operator during execution.

    Output PixelData objects are stored in the Data dictionary.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object should be stored",
    )

    covariance = Unicode(
        "covariance",
        help="The Data key where the covariance should be stored",
    )

    inverse_covariance = Unicode(
        None,
        allow_none=True,
        help="The Data key where the inverse covariance should be stored",
    )

    hits = Unicode(
        "hits",
        help="The Data key where the hit map should be stored",
    )

    rcond = Unicode(
        "rcond",
        help="The Data key where the reciprocal condition number should be stored",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
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

    save_pointing = Bool(
        False, help="If True, do not clear detector pointing matrices after use"
    )

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "stokes_weights":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        # Construct the pointing distribution if it does not already exist

        if self.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                shared_flags=self.shared_flags,
                shared_flag_mask=self.shared_flag_mask,
                save_pointing=self.save_pointing,
            )
            pix_dist.apply(data)

        # Check if map domain products exist and are consistent.  The hits
        # and inverse covariance accumulation operators support multiple
        # calls to exec() to accumulate data.  But in this convenience
        # function we are explicitly accumulating in one-shot.  This means
        # that any existing data products must be set to zero.

        if self.hits in data:
            if data[self.hits].distribution == data[self.pixel_dist]:
                # Distributions are equal, just set to zero
                data[self.hits].reset()
            else:
                # Inconsistent- delete it so that it will be re-created.
                del data[self.hits]
        if self.covariance in data:
            if data[self.covariance].distribution == data[self.pixel_dist]:
                # Distribution matches, set to zero and update units
                data[self.covariance].reset()
                invcov_units = 1.0 / (self.det_data_units**2)
                data[self.covariance].update_units(invcov_units)
            else:
                del data[self.covariance]

        # Hit map operator

        build_hits = BuildHitMap(
            pixel_dist=self.pixel_dist,
            hits=self.hits,
            view=self.pixel_pointing.view,
            pixels=self.pixel_pointing.pixels,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )

        # Inverse covariance.  Note that we save the output to our specified
        # "covariance" key, because we are going to invert it in-place.

        build_invcov = BuildInverseCovariance(
            pixel_dist=self.pixel_dist,
            inverse_covariance=self.covariance,
            view=self.pixel_pointing.view,
            pixels=self.pixel_pointing.pixels,
            weights=self.stokes_weights.weights,
            noise_model=self.noise_model,
            det_data_units=self.det_data_units,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )

        # Build a pipeline to expand pointing and accumulate

        accum = None
        if self.save_pointing:
            # Process all detectors at once
            accum = Pipeline(detector_sets=["ALL"])
        else:
            # Process one detector at a time.
            accum = Pipeline(detector_sets=["SINGLE"])
        accum.operators = [
            self.pixel_pointing,
            self.stokes_weights,
            build_hits,
            build_invcov,
        ]

        pipe_out = accum.apply(data, detectors=detectors)

        # Optionally, store the inverse covariance
        if self.inverse_covariance is not None:
            if self.inverse_covariance in data:
                del data[self.inverse_covariance]
            data[self.inverse_covariance] = data[self.covariance].duplicate()

        # Extract the results
        hits = data[self.hits]
        cov = data[self.covariance]

        # Invert the covariance in place
        rcond = PixelData(cov.distribution, np.float64, n_value=1)
        covariance_invert(
            cov,
            self.rcond_threshold,
            rcond=rcond,
            use_alltoallv=(self.sync_type == "alltoallv"),
        )

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
            msg = f"  Pixel covariance condition number range = "
            msg += f"{rcond_min:1.3e} ... {rcond_max:1.3e}"
            log.debug(msg)

        # Store rcond
        data[self.rcond] = rcond

        return

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
