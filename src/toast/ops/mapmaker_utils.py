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

from .._libtoast import (
    cov_accum_zmap,
    cov_accum_diag_hits,
    cov_accum_diag_invnpp,
)

from .operator import Operator

from .delete import Delete

from .pipeline import Pipeline


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

    pixels = Unicode("pixels", help="Observation detdata key for pixel indices")

    sync_type = Unicode(
        "allreduce", help="Communication algorithm: 'allreduce' or 'alltoallv'"
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
        if data.comm.world_rank == 0:
            log.debug(
                "Building hit map with pixel_distribution {}".format(self.pixel_dist)
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
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # The pixels and weights view for this observation
            pix = ob.view[self.view].detdata[self.pixels]
            flgs = [None for x in pix]
            if self.det_flags is not None:
                flgs = ob.view[self.view].detdata[self.det_flags]

            # Process every data view
            for pview, fview in zip(pix, flgs):
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
            "meta": [self.pixel_dist],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
            "intervals": list(),
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"meta": [self.hits]}
        return prov

    def _accelerators(self):
        return list()


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
        "allreduce", help="Communication algorithm: 'allreduce' or 'alltoallv'"
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
        if data.comm.world_rank == 0:
            log.debug(
                "Building inverse covariance with pixel_distribution {}".format(
                    self.pixel_dist
                )
            )

        invcov = None
        weight_nnz = None
        cov_nnz = None

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
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
            flgs = [None for x in wts]
            if self.det_flags is not None:
                flgs = ob.view[self.view].detdata[self.det_flags]

            # Process every data view
            for pview, wview, fview in zip(pix, wts, flgs):
                for det in dets:
                    # We require that the pointing matrix has the same number of
                    # non-zero elements for every detector and every observation.
                    # We check that here, and if this is the first observation and
                    # detector we have worked with we create the PixelData object.
                    if invcov is None:
                        # We will store the lower triangle of the covariance.
                        if len(wview.detector_shape) == 1:
                            weight_nnz = 1
                        else:
                            weight_nnz = wview.detector_shape[1]
                        cov_nnz = weight_nnz * (weight_nnz + 1) // 2
                        if self.inverse_covariance in data:
                            # We have an existing map from a previous call.  Verify
                            # the distribution and nnz.
                            if data[self.inverse_covariance].distribution != dist:
                                msg = "Existing inv cov '{}' has different data distribution".format(
                                    self.inverse_covariance
                                )
                                log.error(msg)
                                raise RuntimeError(msg)
                            if data[self.inverse_covariance].n_value != cov_nnz:
                                msg = "Existing inv cov '{}' has {} nnz, but pointing implies {}".format(
                                    self.inverse_covariance,
                                    data[self.inverse_covariance].n_value,
                                    cov_nnz,
                                )
                                log.error(msg)
                                raise RuntimeError(msg)
                            invcov = data[self.inverse_covariance]
                        else:
                            data[self.inverse_covariance] = PixelData(
                                dist, np.float64, n_value=cov_nnz
                            )
                            invcov = data[self.inverse_covariance]
                    else:
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
            "meta": [self.pixel_dist, self.noise_model],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
            "intervals": list(),
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"meta": [self.inverse_covariance]}
        return prov

    def _accelerators(self):
        return list()


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
        None, allow_none=True, help="Observation detdata key for the timestream data"
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
        "allreduce", help="Communication algorithm: 'allreduce' or 'alltoallv'"
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

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
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
            flgs = [None for x in wts]
            if self.det_flags is not None:
                flgs = ob.view[self.view].detdata[self.det_flags]

            # Process every data view
            for pview, wview, dview, fview in zip(pix, wts, ddat, flgs):
                for det in dets:
                    # Data for this detector
                    ddata = dview[det]

                    # We require that the pointing matrix has the same number of
                    # non-zero elements for every detector and every observation.
                    # We check that here, and if this is the first observation and
                    # detector we have worked with we create the PixelData object
                    # if needed.
                    if zmap is None:
                        if len(wview.detector_shape) == 1:
                            weight_nnz = 1
                        else:
                            weight_nnz = wview.detector_shape[1]
                        if self.zmap in data:
                            # We have an existing map from a previous call.  Verify
                            # the distribution and nnz.
                            if data[self.zmap].distribution != dist:
                                msg = "Existing ZMap '{}' has different data distribution".format(
                                    self.zmap
                                )
                                log.error(msg)
                                raise RuntimeError(msg)
                            if data[self.zmap].n_value != weight_nnz:
                                msg = "Existing ZMap '{}' has {} nnz, but pointing has {}".format(
                                    self.zmap, data[self.zmap].n_value, weight_nnz
                                )
                                log.error(msg)
                                raise RuntimeError(msg)
                            zmap = data[self.zmap]
                        else:
                            data[self.zmap] = PixelData(
                                dist, np.float64, n_value=weight_nnz
                            )
                            zmap = data[self.zmap]
                    else:
                        check_nnz = None
                        if len(wview.detector_shape) == 1:
                            check_nnz = 1
                        else:
                            check_nnz = wview.detector_shape[1]
                        if check_nnz != weight_nnz:
                            msg = "observation {}, detector {}, pointing weights {} has inconsistent number of values".format(
                                ob.name, det, self.weights
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
                    zm = zmap.raw.array()
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
            "meta": [self.pixel_dist, self.noise_model, self.det_data],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
            "intervals": list(),
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"meta": [self.zmap]}
        return prov

    def _accelerators(self):
        return list()


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

    NOTE:  The pointing operator must have the "pixels", "weights", and "create_dist"
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

    hits = Unicode(
        "hits",
        help="The Data key where the hit map should be stored",
    )

    rcond = Unicode(
        "rcond",
        help="The Data key where the inverse condition number should be stored",
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

    pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator",
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    rcond_threshold = Float(
        1.0e-8, help="Minimum value for inverse condition number cut."
    )

    sync_type = Unicode(
        "allreduce", help="Communication algorithm: 'allreduce' or 'alltoallv'"
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

    @traitlets.validate("sync_type")
    def _check_sync_type(self, proposal):
        check = proposal["value"]
        if check != "allreduce" and check != "alltoallv":
            raise traitlets.TraitError("Invalid communication algorithm")
        return check

    @traitlets.validate("pointing")
    def _check_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError("pointing should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["pixels", "weights", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = "pointing operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.pixel_dist is None:
            raise RuntimeError(
                "You must set the 'pixel_dist' trait before calling exec()"
            )

        # Set outputs of the pointing operator

        self.pointing.create_dist = None

        # If we do not have a pixel distribution yet, we must make one pass through
        # the pointing to build this first.

        pointing_done = False

        if self.pixel_dist not in data:
            if detectors is not None:
                msg = "A subset of detectors is specified, but the pixel distribution\n"
                msg += "does not yet exist- and creating this requires all detectors.\n"
                msg += "Either pre-create the pixel distribution with all detectors\n"
                msg += "or run this operator with all detectors."
                raise RuntimeError(msg)

            msg = "Creating pixel distribution '{}' in Data".format(self.pixel_dist)
            if data.comm.world_rank == 0:
                log.debug(msg)

            # Turn on creation of the pixel distribution
            self.pointing.create_dist = self.pixel_dist

            # Compute the pointing matrix

            pixel_dist_pipe = None
            if self.save_pointing:
                # We are keeping the pointing, which means we need to run all detectors
                # at once so they all end up in the detdata for all observations.
                pixel_dist_pipe = Pipeline(detector_sets=["ALL"])
                pointing_done = True
            else:
                # Run one detector a at time and discard.
                pixel_dist_pipe = Pipeline(detector_sets=["SINGLE"])
            pixel_dist_pipe.operators = [
                self.pointing,
            ]
            pipe_out = pixel_dist_pipe.apply(data, detectors=detectors)

            # Turn pixel distribution creation off again
            self.pointing.create_dist = None

        # Hit map operator

        build_hits = BuildHitMap(
            pixel_dist=self.pixel_dist,
            hits=self.hits,
            view=self.pointing.view,
            pixels=self.pointing.pixels,
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
            view=self.pointing.view,
            pixels=self.pointing.pixels,
            weights=self.pointing.weights,
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
            if pointing_done:
                # We already computed the pointing once and saved it.
                accum.operators = [build_hits, build_invcov]
            else:
                accum.operators = [self.pointing, build_hits, build_invcov]
        else:
            # Process one detector at a time.
            accum = Pipeline(detector_sets=["SINGLE"])
            accum.operators = [self.pointing, build_hits, build_invcov]

        pipe_out = accum.apply(data, detectors=detectors)

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
        req = self.pointing.requires()
        req["meta"].append(self.noise_model)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {
            "meta": [self.pixel_dist, self.hits, self.covariance, self.rcond],
            "shared": list(),
            "detdata": list(),
        }
        if self.save_pointing:
            prov["detdata"].extend([self.pixels, self.weights])
        return prov

    def _accelerators(self):
        return list()
