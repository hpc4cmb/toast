# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger, AlignedF64

from ..traits import trait_docs, Int, Unicode, Bool

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from .._libtoast import scan_map_float64, scan_map_float32

from .operator import Operator


@trait_docs
class ScanMap(Operator):
    """Operator which uses the pointing matrix to scan timestream values from a map.

    The map must be a PixelData instance with either float32 or float64 values.  The
    values can either be accumulated or subtracted from the input timestream, and the
    input timestream can be optionally zeroed out beforehand.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    pixels = Unicode("pixels", help="Observation detdata key for pixel indices")

    weights = Unicode("weights", help="Observation detdata key for Stokes weights")

    map_key = Unicode(
        None,
        allow_none=True,
        help="The Data key where the map is located",
    )

    subtract = Bool(
        False, help="If True, subtract the map timestream instead of accumulating"
    )

    zero = Bool(False, help="If True, zero the data before accumulating / subtracting")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check that the map is set
        if self.map_key is None:
            raise RuntimeError("You must set the map_key trait before calling exec()")
        if self.map_key not in data:
            msg = "The map_key '{}' does not exist in the data".format(self.map_key)
            raise RuntimeError(msg)

        map_data = data[self.map_key]
        if not isinstance(map_data, PixelData):
            raise RuntimeError("The map to scan must be a PixelData instance")
        map_dist = map_data.distribution

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Sanity check the number of non-zeros between the map and the
            # pointing matrix
            check_nnz = 1
            if len(ob.detdata[self.weights].detector_shape) > 1:
                check_nnz = ob.detdata[self.weights].detector_shape[-1]
            if map_data.n_value != check_nnz:
                msg = "Detector data '{}' in observation '{}' has {} nnz instead of {} in the map".format(
                    self.weights, ob.name, check_nnz, map_data.n_value
                )
                log.error(msg)
                raise RuntimeError(msg)

            # Temporary array, re-used for all detectors
            maptod_raw = AlignedF64.zeros(ob.n_local_samples)
            maptod = maptod_raw.array()

            # If our output detector data does not yet exist, create it
            if self.det_data not in ob:
                ob.detdata.create(self.det_data, dtype=np.float64, detectors=dets)

            for det in dets:
                # The pixels, weights, and data.
                pix = ob.detdata[self.pixels][det]
                wts = ob.detdata[self.weights][det]
                ddata = ob.detdata[self.det_data][det]

                # Get local submap and pixels
                local_sm, local_pix = map_dist.global_pixel_to_submap(pix)

                # We support projecting from either float64 or float32 maps.

                maptod[:] = 0.0

                if map_data.dtype.char == "d":
                    scan_map_float64(
                        map_data.distribution.n_pix_submap,
                        map_data.n_value,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        map_data.raw,
                        wts.astype(np.float64).reshape(-1),
                        maptod,
                    )
                elif map_data.dtype.char == "f":
                    scan_map_float32(
                        map_data.distribution.n_pix_submap,
                        map_data.n_value,
                        local_sm.astype(np.int64),
                        local_pix.astype(np.int64),
                        map_data.raw,
                        wts.astype(np.float64).reshape(-1),
                        maptod,
                    )
                else:
                    raise RuntimeError(
                        "Projection supports only float32 and float64 binned maps"
                    )

                # zero-out if needed
                if self.zero:
                    ddata[:] = 0.0

                # Add or subtract.  Note that the map scanned timestream will have
                # zeros anywhere that the pointing is bad, but those samples (and
                # any other detector flags) should be handled at other steps of the
                # processing.
                if self.subtract:
                    ddata -= maptod
                else:
                    ddata += maptod

            del maptod
            maptod_raw.clear()
            del maptod_raw

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [map_key],
            "shared": list(),
            "detdata": [self.pixels, self.weights, self.det_data],
        }
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()


@trait_docs
class ScanMask(Operator):
    """Operator which uses the pointing matrix to set timestream flags from a mask.

    The mask must be a PixelData instance with an integer data type.  The data for each
    pixel is bitwise-and combined with the mask_bits to form a result.  for each
    detector sample crossing a pixel with a non-zero result, the detector flag is
    bitwise-or'd with the specified value.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to set"
    )

    det_flags_value = Int(
        1, help="The detector flag value to set where the mask result is non-zero"
    )

    pixels = Unicode("pixels", help="Observation detdata key for pixel indices")

    mask_key = Unicode(
        None,
        allow_none=True,
        help="The Data key where the mask is located",
    )

    mask_bits = Int(
        255, help="The number to bitwise-and with each mask value to form the result"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_flags is None:
            raise RuntimeError("You must set the det_flags trait before calling exec()")

        # Check that the mask is set
        if self.mask_key is None:
            raise RuntimeError("You must set the mask_key trait before calling exec()")
        if self.mask_key not in data:
            msg = "The mask_key '{}' does not exist in the data".format(self.mask_key)
            raise RuntimeError(msg)

        mask_data = data[self.mask_key]
        if not isinstance(mask_data, PixelData):
            raise RuntimeError("The mask to scan must be a PixelData instance")
        mask_dist = mask_data.distribution

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # If our output detector data does not yet exist, create it with a default
            # width of one byte per sample.
            if self.det_flags not in ob:
                ob.detdata.create(self.det_flags, dtype=np.uint8, detectors=dets)

            for det in dets:
                # The pixels and flags.
                pix = ob.detdata[self.pixels][det]
                dflags = ob.detdata[self.det_flags][det]

                # Get local submap and pixels
                local_sm, local_pix = mask_dist.global_pixel_to_submap(pix)

                # We could move this to compiled code if it is too slow...
                masked = mask_data[local_sm, local_pix, 0] & self.mask_bits
                dflags[masked != 0] |= self.det_flags_value

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [mask_key],
            "shared": list(),
            "detdata": [self.pixels, self.det_flags],
        }
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        return prov

    def _accelerators(self):
        return list()
