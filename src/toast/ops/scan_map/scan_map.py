# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ...accelerator import ImplementationType
from ...observation import default_values as defaults
from ...pixels import PixelData, PixelDistribution
from ...timing import function_timer
from ...traits import Bool, Int, Unicode, Unit, UseEnum, trait_docs
from ...utils import AlignedF64, Logger, unit_conversion
from ..operator import Operator
from .kernels import ImplementationType, scan_map


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
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode(defaults.pixels, help="Observation detdata key for pixel indices")

    weights = Unicode(
        defaults.weights,
        allow_none=True,
        help="Observation detdata key for Stokes weights",
    )

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
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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

            if self.weights is not None:
                # Sanity check the number of non-zeros between the map and the
                # pointing matrix
                check_nnz = 1
                if len(ob.detdata[self.weights].detector_shape) > 1:
                    check_nnz = ob.detdata[self.weights].detector_shape[-1]
                if map_data.n_value != check_nnz:
                    msg = (
                        f"Detector data '{self.weights}' in observation '{ob.name}' "
                        f"has {check_nnz} nnz instead of {map_data.n_value} in the map"
                    )
                    log.error(msg)
                    raise RuntimeError(msg)

            # If our output detector data does not yet exist, create it
            _ = ob.detdata.ensure(
                self.det_data,
                detectors=dets,
                create_units=self.det_data_units,
                accel=use_accel,
            )

            intervals = ob.intervals[self.view].data
            det_data = ob.detdata[self.det_data].data
            det_data_indx = ob.detdata[self.det_data].indices(dets)
            pixels = ob.detdata[self.pixels].data
            pixels_indx = ob.detdata[self.pixels].indices(dets)
            data_scale = unit_conversion(
                map_data.units, ob.detdata[self.det_data].units
            )
            if self.weights is None:
                # Use empty arrays, rather than None, so that we can pass that more
                # easily to compiled kernels that expect a buffer.
                weights = np.array([], dtype=np.float64)
                weight_indx = np.array([], dtype=np.int32)
            else:
                weights = ob.detdata[self.weights].data
                weight_indx = ob.detdata[self.weights].indices(dets)

            scan_map(
                map_dist.global_submap_to_local,
                map_dist.n_pix_submap,
                map_data.data,
                det_data,
                det_data_indx,
                pixels,
                pixels_indx,
                weights,
                weight_indx,
                intervals,
                data_scale,
                bool(self.zero),
                bool(self.subtract),
                False,
                impl=implementation,
                use_accel=use_accel,
            )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "global": [self.map_key],
            "meta": list(),
            "shared": list(),
            "detdata": [self.pixels, self.det_data],
            "intervals": list(),
        }
        if self.weights is not None:
            req["detdata"].append(self.weights)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        return {"detdata": [self.det_data]}

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
class ScanMask(Operator):
    """Operator which uses the pointing matrix to set timestream flags from a mask.

    The mask must be a PixelData instance with an integer data type.  The data for each
    pixel is bitwise-and combined with the mask_bits to form a result.  For each
    detector sample crossing a pixel with a non-zero result, the detector flag is
    bitwise-or'd with the specified value.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flags_value = Int(
        1, help="The detector flag value to set where the mask result is non-zero"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
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
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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
            if self.det_flags not in ob.detdata:
                ob.detdata.create(self.det_flags, dtype=np.uint8, detectors=dets)

            views = ob.view[self.view]
            for ivw, vw in enumerate(views):
                for det in dets:
                    # The pixels and flags.
                    pix = views.detdata[self.pixels][ivw][det]
                    dflags = views.detdata[self.det_flags][ivw][det]

                    # Get local submap and pixels
                    local_sm, local_pix = mask_dist.global_pixel_to_submap(pix)

                    # We could move this to compiled code if it is too slow...
                    masked = (mask_data[local_sm, local_pix, 0] & self.mask_bits) != 0
                    dflags[masked] |= self.det_flags_value

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "global": [self.mask_key],
            "shared": list(),
            "detdata": [self.pixels, self.det_flags],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": [self.det_flags]}
        return prov

    def _supports_accel(self):
        return False


@trait_docs
class ScanScale(Operator):
    """Operator which uses the pointing matrix to apply pixel weights to timestreams.

    The map must be a PixelData instance with either float32 or float64 values and
    one value per pixel.  The timestream samples are multiplied by their corresponding
    pixel values.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode(defaults.pixels, help="Observation detdata key for pixel indices")

    weights = Unicode(
        defaults.weights, help="Observation detdata key for Stokes weights"
    )

    map_key = Unicode(
        None,
        allow_none=True,
        help="The Data key where the weight map is located",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

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
        if map_data.n_value != 1:
            raise RuntimeError("The map to scan must have one value per pixel")
        map_dist = map_data.distribution

        if map_data.units != u.dimensionless_unscaled:
            log.warning("Map for scaling is not unitless.  Ignoring units.")

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            if self.det_data not in ob.detdata:
                msg = "detector data '{}' does not exist in observation {}".format(
                    self.det_data, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)

            intervals = ob.intervals[self.view].data
            det_data = ob.detdata[self.det_data].data
            det_data_indx = ob.detdata[self.det_data].indices(dets)
            pixels = ob.detdata[self.pixels].data
            pixels_indx = ob.detdata[self.pixels].indices(dets)
            if self.weights is None:
                # Use empty arrays, rather than None, so that we can pass that more
                # easily to compiled kernels that expect a buffer.
                weights = np.array([], dtype=np.float64)
                weight_indx = np.array([], dtype=np.int32)
            else:
                weights = ob.detdata[self.weights].data
                weight_indx = ob.detdata[self.weights].indices(dets)
            scan_map(
                map_dist.global_submap_to_local,
                map_dist.n_pix_submap,
                map_data.data,
                det_data,
                det_data_indx,
                pixels,
                pixels_indx,
                weights,
                weight_indx,
                intervals,
                1.0,
                False,
                False,
                True,
                impl=implementation,
                use_accel=use_accel,
            )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "global": [self.map_key],
            "shared": list(),
            "detdata": [self.pixels, self.weights, self.det_data],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": [self.det_data]}
        return prov

    def _supports_accel(self):
        return True
