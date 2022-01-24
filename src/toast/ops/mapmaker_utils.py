# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance, Float

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from ..covariance import covariance_invert

from ..observation import default_values as defaults

from .._libtoast import (
    cov_accum_zmap,
    cov_accum_diag_hits,
    cov_accum_diag_invnpp,
)

from .operator import Operator

from .delete import Delete

from .pipeline import Pipeline

from .pointing import BuildPixelDistribution


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
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

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
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

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
                        local_pix[fview[det] & self.det_flag_mask != 0] = -1
                    if self.shared_flags is not None:
                        local_pix[shared_fview & self.shared_flag_mask != 0] = -1

                    cov_accum_diag_hits(
                        dist.n_local_submap,
                        dist.n_pix_submap,
                        1,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        hits.raw,
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
        prov = {"global": [self.hits]}
        return prov


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
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

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
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

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

        invcov = None
        weight_nnz = None
        cov_nnz = None

        # We will store the lower triangle of the covariance.  This operator requires
        # that all detectors in all observations have the same number of non-zeros
        # in the pointing matrix.

        if self.inverse_covariance in data:
            # We have an existing map from a previous call.  Verify
            # the distribution.
            if data[self.inverse_covariance].distribution != dist:
                msg = "Existing inv cov '{}' has different data distribution".format(
                    self.inverse_covariance
                )
                log.error(msg)
                raise RuntimeError(msg)
            invcov = data[self.inverse_covariance]
            cov_nnz = invcov.n_value
            weight_nnz = int((np.sqrt(1 + 8 * cov_nnz) - 1) // 2)
        else:
            try:
                first_detwt = data.obs[0].detdata[self.weights][0]
                if len(first_detwt.shape) == 1:
                    weight_nnz = 1
                else:
                    weight_nnz = first_detwt.shape[1]
            except KeyError:
                weight_nnz = 0
            if data.comm.comm_world is not None:
                weight_nnz = np.amax(data.comm.comm_world.allgather(weight_nnz))
            cov_nnz = int(weight_nnz * (weight_nnz + 1) // 2)
            data[self.inverse_covariance] = PixelData(dist, np.float64, n_value=cov_nnz)
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
                        local_pix[fview[det] & self.det_flag_mask != 0] = -1
                    if self.shared_flags is not None:
                        local_pix[shared_fview & self.shared_flag_mask != 0] = -1

                    # Accumulate
                    cov_accum_diag_invnpp(
                        dist.n_local_submap,
                        dist.n_pix_submap,
                        weight_nnz,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        wview[det].reshape(-1),
                        detweight,
                        invcov.raw,
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
            "global": [self.pixel_dist, self.noise_model],
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

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

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
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

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
            log.debug(
                "Building noise weighted map with pixel_distribution {}".format(
                    self.pixel_dist
                )
            )

        zmap = None
        weight_nnz = None

        # This operator requires that all detectors in all observations have the same
        # number of non-zeros in the pointing matrix.

        if self.zmap in data:
            # We have an existing map from a previous call.  Verify
            # the distribution.
            if data[self.zmap].distribution != dist:
                msg = "Existing zmap '{}' has different data distribution".format(
                    self.zmap
                )
                log.error(msg)
                raise RuntimeError(msg)
            zmap = data[self.zmap]
            weight_nnz = zmap.n_value
        else:
            try:
                first_detwt = data.obs[0].detdata[self.weights][0]
                if len(first_detwt.shape) == 1:
                    weight_nnz = 1
                else:
                    weight_nnz = first_detwt.shape[1]
            except KeyError:
                weight_nnz = 0
            if data.comm.comm_world is not None:
                weight_nnz = np.amax(data.comm.comm_world.allgather(weight_nnz))
            data[self.zmap] = PixelData(dist, np.float64, n_value=weight_nnz)
            zmap = data[self.zmap]

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
            ddat = ob.view[self.view].detdata[self.det_data]

            if self.shared_flags is not None:
                shared_flgs = ob.view[self.view].shared[self.shared_flags]
            else:
                shared_flgs = [None for x in wts]
            if self.det_flags is not None:
                flgs = ob.view[self.view].detdata[self.det_flags]
            else:
                flgs = [None for x in wts]

            # Process every data view
            for pview, wview, dview, fview, shared_fview in zip(
                pix, wts, ddat, flgs, shared_flgs
            ):
                for det in dets:
                    # Data for this detector
                    ddata = dview[det]

                    # We require that the pointing matrix has the same number of
                    # non-zero elements for every detector and every observation.
                    # We check that here.

                    check_nnz = None
                    if len(wview.detector_shape) == 1:
                        check_nnz = 1
                    else:
                        check_nnz = wview.detector_shape[1]
                    if check_nnz != weight_nnz:
                        msg = (
                            f"observation {ob.name}, detector {det}, pointing "
                            f"weights {self.weights} has inconsistent number of values"
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
                        local_pix[fview[det] & self.det_flag_mask != 0] = -1
                    if self.shared_flags is not None:
                        local_pix[shared_fview & self.shared_flag_mask != 0] = -1

                    # Accumulate
                    cov_accum_zmap(
                        dist.n_local_submap,
                        dist.n_pix_submap,
                        zmap.n_value,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        wview[det].reshape(-1),
                        detweight,
                        ddata,
                        zmap.raw,
                    )
        return

    def _finalize(self, data, **kwargs):
        if self.zmap in data:
            # We have called exec() at least once
            if self.sync_type == "alltoallv":
                data[self.zmap].sync_alltoallv()
            else:
                data[self.zmap].sync_allreduce()
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
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

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
