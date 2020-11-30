# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..operator import Operator

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from .._libtoast import (
    cov_accum_zmap,
    cov_accum_diag_hits,
    cov_accum_diag_invnpp,
)

from .clear import Clear


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

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional flagging")

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
        self._hits = None

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

        # On first call, get the pixel distribution and create our distributed hitmap
        if self._hits is None:
            self._hits = PixelData(dist, np.int64, n_value=1)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            for det in dets:
                # Get local submap and pixels
                local_sm, local_pix = dist.global_pixel_to_submap(
                    ob.detdata[self.pixels][det]
                )

                # Samples with telescope pointing problems are already flagged in the
                # the pointing operators by setting the pixel numbers to a negative
                # value.  Here we optionally apply detector flags to the local
                # pixel numbers to flag more samples.

                # Apply the flags if needed
                if self.det_flags is not None:
                    flags = np.array(ob.detdata[self.det_flags])
                    flags &= self.det_flag_mask
                    local_pix[flags != 0] = -1

                cov_accum_diag_hits(
                    dist.n_local_submap,
                    dist.n_pix_submap,
                    1,
                    local_sm.astype(np.int64),
                    local_pix.astype(np.int64),
                    self._hits.raw,
                )

        return

    def _finalize(self, data, **kwargs):
        if self._hits is not None:
            if self.sync_type == "alltoallv":
                self._hits.sync_alltoallv()
            else:
                self._hits.sync_allreduce()
        return self._hits

    def _requires(self):
        req = {
            "meta": [self.pixel_dist],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
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

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional flagging")

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
        self._invcov = None

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

            for det in dets:
                # The pixels and weights for this detector.
                pix = ob.detdata[self.pixels]
                wts = ob.detdata[self.weights]

                # We require that the pointing matrix has the same number of
                # non-zero elements for every detector and every observation.
                # We check that here, and if this is the first observation and
                # detector we have worked with we create the PixelData object.
                if self._invcov is None:
                    # We will store the lower triangle of the covariance.
                    if len(wts.detector_shape) == 1:
                        weight_nnz = 1
                    else:
                        weight_nnz = wts.detector_shape[1]
                    cov_nnz = weight_nnz * (weight_nnz + 1) // 2
                    self._invcov = PixelData(dist, np.float64, n_value=cov_nnz)
                else:
                    check_nnz = None
                    if len(wts.detector_shape) == 1:
                        check_nnz = 1
                    else:
                        check_nnz = wts.detector_shape[1]
                    if check_nnz != weight_nnz:
                        msg = "observation {}, detector {}, pointing weights {} has inconsistent number of values".format(
                            ob.name, det, self.weights
                        )
                        raise RuntimeError(msg)

                # Get local submap and pixels
                local_sm, local_pix = dist.global_pixel_to_submap(pix[det])

                # Get the detector weight from the noise model.
                detweight = noise.detector_weight(det)

                # Samples with telescope pointing problems are already flagged in the
                # the pointing operators by setting the pixel numbers to a negative
                # value.  Here we optionally apply detector flags to the local
                # pixel numbers to flag more samples.

                # Apply the flags if needed
                if self.det_flags is not None:
                    flags = np.array(ob.detdata[self.det_flags])
                    flags &= self.det_flag_mask
                    local_pix[flags != 0] = -1

                # Accumulate
                cov_accum_diag_invnpp(
                    dist.n_local_submap,
                    dist.n_pix_submap,
                    weight_nnz,
                    local_sm.astype(np.int64),
                    local_pix.astype(np.int64),
                    wts[det].reshape(-1),
                    detweight,
                    self._invcov.raw,
                )
        return

    def _finalize(self, data, **kwargs):
        if self._invcov is not None:
            if self.sync_type == "alltoallv":
                self._invcov.sync_alltoallv()
            else:
                self._invcov.sync_allreduce()
        return self._invcov

    def _requires(self):
        req = {
            "meta": [self.pixel_dist, self.noise_model],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
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

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional flagging")

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
        self._zmap = None

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

            for det in dets:
                # The pixels and weights for this detector.
                pix = ob.detdata[self.pixels]
                wts = ob.detdata[self.weights]
                ddata = ob.detdata[self.det_data][det]

                # We require that the pointing matrix has the same number of
                # non-zero elements for every detector and every observation.
                # We check that here, and if this is the first observation and
                # detector we have worked with we create the PixelData object.
                if self._zmap is None:
                    if len(wts.detector_shape) == 1:
                        weight_nnz = 1
                    else:
                        weight_nnz = wts.detector_shape[1]
                    self._zmap = PixelData(dist, np.float64, n_value=weight_nnz)
                else:
                    check_nnz = None
                    if len(wts.detector_shape) == 1:
                        check_nnz = 1
                    else:
                        check_nnz = wts.detector_shape[1]
                    if check_nnz != weight_nnz:
                        msg = "observation {}, detector {}, pointing weights {} has inconsistent number of values".format(
                            ob.name, det, self.weights
                        )
                        raise RuntimeError(msg)

                # Get local submap and pixels
                local_sm, local_pix = dist.global_pixel_to_submap(pix[det])

                # Get the detector weight from the noise model.
                detweight = noise.detector_weight(det)

                # Samples with telescope pointing problems are already flagged in the
                # the pointing operators by setting the pixel numbers to a negative
                # value.  Here we optionally apply detector flags to the local
                # pixel numbers to flag more samples.

                # Apply the flags if needed
                if self.det_flags is not None:
                    flags = np.array(ob.detdata[self.det_flags])
                    flags &= self.det_flag_mask
                    local_pix[flags != 0] = -1

                # Accumulate
                cov_accum_zmap(
                    dist.n_local_submap,
                    dist.n_pix_submap,
                    self._zmap.n_value,
                    local_sm.astype(np.int64),
                    local_pix.astype(np.int64),
                    wts[det].reshape(-1),
                    detweight,
                    ddata,
                    self._zmap.raw,
                )
        return

    def _finalize(self, data, **kwargs):
        if self._zmap is not None:
            if self.sync_type == "alltoallv":
                self._zmap.sync_alltoallv()
            else:
                self._zmap.sync_allreduce()
        return self._zmap

    def _requires(self):
        req = {
            "meta": [self.pixel_dist, self.noise_model, self.det_data],
            "shared": list(),
            "detdata": [self.pixels, self.weights],
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
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

    det_flag_mask = Int(0, help="Bit mask value for optional flagging")

    pointing = Instance(
        klass=None,
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
            if not pntg.has_trait("pixels"):
                raise traitlets.TraitError(
                    "pointing operator should have a 'pixels' trait"
                )
            if not pntg.has_trait("weights"):
                raise traitlets.TraitError(
                    "pointing operator should have a 'weights' trait"
                )
            if not pntg.has_trait("create_dist"):
                raise traitlets.TraitError(
                    "pointing operator should have a 'create_dist' trait"
                )
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._invcov = None

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.pixel_dist is None:
            raise RuntimeError(
                "You must set the 'pixel_dist' trait before calling exec()"
            )

        # Set outputs of the pointing operator

        self.pointing.create_dist = None

        # Set up clearing of the pointing matrices

        clear_pointing = Clear(detdata=[self.pointing.pixels, self.pointing.weights])

        # If we do not have a pixel distribution yet, we must make one pass through
        # the pointing to build this first.

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
                pixel_dist_pipe.operators = [
                    self.pointing,
                ]
            else:
                # Run one detector a at time and discard.
                pixel_dist_pipe = Pipeline(detector_sets=["SINGLE"])
                pixel_dist_pipe.operators = [
                    self.pointing,
                    clear_pointing,
                ]
            pipe_out = pixel_dist_pipe.apply(data, detectors=detectors)

            # Turn pixel distribution creation off again
            self.pointing.create_dist = None

        # Hit map operator

        build_hits = BuildHitMap(
            pixel_dist=self.pixel_dist,
            pixels=self.pointing.pixels,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            sync_type=self.sync_type,
        )

        # Inverse covariance

        build_invcov = BuildInverseCovariance(
            pixel_dist=self.pixel_dist,
            pixels=self.pointing.pixels,
            weights=self.pointing.weights,
            noise_model=self.noise_model,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            sync_type=self.sync_type,
        )

        # Build a pipeline to expand pointing and accumulate

        accum = None
        if self.save_pointing:
            # Process all detectors at once
            accum = Pipeline(detector_sets=["ALL"])
            accum.operators = [self.pointing, build_hits, build_invcov]
        else:
            # Process one detector at a time and clear pointing after each one.
            accum = Pipeline(detector_sets=["SINGLE"])
            accum.operators = [self.pointing, build_hits, build_invcov, clear_pointing]

        pipe_out = accum.apply(data, detectors=detectors)

        # Extract the results
        hits = pipe_out[1]
        cov = pipe_out[2]

        # Invert the covariance
        rcond = PixelData(cov.distribution, np.float64, n_value=1)
        covariance_invert(
            cov,
            self.rcond_threshold,
            rcond=rcond,
            use_alltoallv=(self.sync_type == "alltoallv"),
        )

        # Store products
        data[self.hits] = hits
        data[self.covariance] = cov
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
