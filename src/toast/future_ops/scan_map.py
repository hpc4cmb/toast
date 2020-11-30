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

from .._libtoast import scan_map_float64, scan_map_float32


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

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Temporary array, re-used for all detectors
            maptod_raw = AlignedF64.zeros(ob.n_local_samples)
            maptod = maptod_raw.array()

            for det in dets:
                # The pixels, weights, and data.
                pix = ob.detdata[self.pixels][det]
                wts = ob.detdata[self.weights][det]
                ddata = ob.detdata[self.det_data][det]

                # Get local submap and pixels
                local_sm, local_pix = dist.global_pixel_to_submap(pix)

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
