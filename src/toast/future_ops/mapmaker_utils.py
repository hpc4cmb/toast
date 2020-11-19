# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool

from ..operator import Operator

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from .._libtoast import (
    cov_accum_zmap,
    cov_accum_diag_hits,
    cov_accum_diag_invnpp,
)


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

    weights = Unicode("weights", help="Observation detdata key for Stokes weights")

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
    def _check_flag_mask(self, proposal):
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

        # On first call, get the pixel distribution and create our distributed hitmap
        if self._hits is None:
            self._hits = PixelData(dist, np.int32, n_value=1)

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
                    dist.n_submap,
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
    def _check_flag_mask(self, proposal):
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
                    weight_nnz = len(wts.detector_shape)
                    cov_nnz = weight_nnz * (weight_nnz + 1) // 2
                    self._invcov = PixelData(dist, np.float64, n_value=cov_nnz)
                else:
                    if len(wts.detector_shape) != weight_nnz:
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
                    dist.n_submap,
                    dist.n_pix_submap,
                    weight_nnz,
                    local_sm.astype(np.int64),
                    local_pix.astype(np.int64),
                    wts.reshape(-1),
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
    def _check_flag_mask(self, proposal):
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

        dist = data[self.pixel_dist]

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
                    self._zmap = PixelData(
                        dist, np.float64, n_value=len(wts.detector_shape)
                    )
                else:
                    if len(wts.detector_shape) != self._zmap.n_value:
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
                    dist.n_submap,
                    dist.n_pix_submap,
                    self._zmap.n_value,
                    local_sm.astype(np.int64),
                    local_pix.astype(np.int64),
                    wts.reshape(-1),
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
