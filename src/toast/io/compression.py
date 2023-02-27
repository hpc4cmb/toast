# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import gzip

import numpy as np
from astropy import units as u

from ..observation_data import DetectorData
from ..timing import function_timer
from ..utils import AlignedU8, Logger
from .compression_flac import compress_detdata_flac, decompress_detdata_flac


@function_timer
def compress_detdata(detdata, comp_params=None):
    """Compress a DetectorData object.

    The comp_params dictionary should include at least a key "type" set to one
    of the supported compression types:  "none", "gzip", or "flac".  For each
    compression type, additional parameters may be specified:

    gzip:
        "level": 0-9 (9 is used if not specified)

    flac:
        "level": 0-8 (5 is the default)
        "quanta":  The floating point amount that should correspond to one
            integer value in the compressed data.  If not specified, the
            full amplitude of the input data (less the mean) will correspond
            to the range of 32bit integer values.  This can be an array, a
            scalar, or None.

    The compressed bytes are returned in an ndarray of type uint8.  The byte ranges
    are a tuple for each detector specifying the (start byte, end byte).

    Args:
        detdata (DetectorData):  The object to compress.
        comp_params (dict):  Dictionary of compression parameters.

    Returns:
        (tuple): The (compressed bytes, byte ranges, updated params)

    """
    log = Logger.get()

    if comp_params is None:
        comp_params = {"type": "none"}
    if "type" not in comp_params:
        raise ValueError("comp_params must contain the 'type' key")

    comp_params["detectors"] = detdata.detectors
    n_det = len(detdata.detectors)
    comp_params["det_shape"] = detdata.detector_shape
    comp_params["dtype"] = detdata.dtype
    comp_params["units"] = detdata.units

    msg = f"Compressing detector data {detdata} with {comp_params}"
    log.verbose(msg)

    det_stride = 1
    for ds in detdata.detector_shape:
        det_stride *= ds

    if comp_params["type"] == "none":
        det_stride = 1
        for ds in detdata.detector_shape:
            det_stride *= ds
        det_bytes = det_stride * detdata.dtype.itemsize
        comp_ranges = [(x * det_bytes, (x + 1) * det_bytes) for x in range(n_det)]
        comp_bytes = np.array(detdata.flatdata.view(dtype=np.uint8), dtype=np.uint8)
        return (
            comp_bytes,
            comp_ranges,
            comp_params,
        )

    elif comp_params["type"] == "gzip":
        comp_bytes = AlignedU8()
        off = 0
        if "level" in comp_params:
            comp_level = comp_params["level"]
        else:
            comp_level = 9
            comp_params["level"] = comp_level
        start_bytes = list()
        for idet, det in enumerate(detdata.detectors):
            start_bytes.append(off)
            dbytes = gzip.compress(
                detdata[det, :].tobytes(), compresslevel=comp_level, mtime=0
            )
            nb = len(dbytes)
            comp_bytes.resize(off + nb)
            comp_bytes[off : off + nb] = dbytes
            off += nb
        comp_ranges = list()
        last = None
        for idet in range(n_det):
            if idet == n_det - 1:
                cur = (last[1], len(comp_bytes))
            else:
                cur = (start_bytes[idet], start_bytes[idet + 1])
            comp_ranges.append(cur)
            last = cur
        return (comp_bytes.array(), comp_ranges, comp_params)

    elif comp_params["type"] == "flac":
        ftypes = [np.dtype(np.float32), np.dtype(np.float64)]
        itypes = [np.dtype(np.int32), np.dtype(np.int64)]

        # Check parameters
        if "level" in comp_params:
            comp_level = comp_params["level"]
        else:
            comp_level = 5
            comp_params["level"] = comp_level
        if "quanta" in comp_params:
            quanta = comp_params["quanta"]
            if detdata.dtype in itypes:
                raise RuntimeError(
                    "Cannot specify quanta for FLAC compression of integers"
                )
        else:
            quanta = None
            comp_params["quanta"] = quanta

        if detdata.dtype in ftypes:
            (
                comp_bytes,
                comp_ranges,
                comp_params["data_offsets"],
                comp_params["data_gains"],
            ) = compress_detdata_flac(detdata, level=comp_level, quanta=quanta)
        elif detdata.dtype == np.dtype(np.int64):
            (
                comp_bytes,
                comp_ranges,
                comp_params["data_offsets"],
            ) = compress_detdata_flac(detdata, level=comp_level)
        elif detdata.dtype == np.dtype(np.int32):
            (
                comp_bytes,
                comp_ranges,
            ) = compress_detdata_flac(detdata, level=comp_level)
        else:
            msg = f"FLAC Compression of type '{detdata.dtype}' is not supported"
            raise RuntimeError(msg)

        return (comp_bytes, comp_ranges, comp_params)
    else:
        msg = f"Compression type \"{comp_params['type']}\" is not supported"
        raise NotImplementedError(msg)


@function_timer
def decompress_detdata(comp_bytes, comp_ranges, comp_params, detdata=None):
    """Decompress bytes and construct a DetectorData object

    If detdata is not None, its properties are checked and the data is decompressed
    into this container.  Otherwise a new DetectorData object is created and returned.

    Args:
        comp_bytes (array):  The stream of compressed bytes.
        comp_ranges (array):  The starting byte offset for each detector.
        comp_params (dict):  Compression parameters.
        detdata (DetectorData):  (Optional) starting instance to fill.

    Returns:
        (DetectorData):  Object containing the decompressed data.

    """
    log = Logger.get()
    for prop in ["type", "det_shape", "dtype", "units"]:
        if prop not in comp_params:
            msg = f"key '{prop}' not found in comp_params"
            raise ValueError(msg)

    if detdata is not None:
        detectors = detdata.detectors
    elif "detectors" in comp_params:
        detectors = comp_params["detectors"]
    else:
        raise ValueError(
            "Must either provide detdata or a detectors list in comp_params"
        )

    n_det = len(detectors)
    detector_shape = comp_params["det_shape"]
    dtype = comp_params["dtype"]
    units = comp_params["units"]

    msg = f"Decompressing detector data with {comp_params}, {len(comp_bytes)} bytes"
    msg += f", offsets = {comp_ranges}"
    log.verbose(msg)

    if detdata is None:
        # Create a new object
        detdata = DetectorData(
            detectors,
            detector_shape,
            dtype,
            units=units,
        )
    else:
        # We are populating an existing data object.  Verify consistent
        # properties.
        if detectors != detdata.detectors:
            msg = f"Input detdata container has different detectors "
            msg += f"({detdata.detectors}) than compressed data "
            msg += f"({detectors})"
            raise RuntimeError(msg)
        if detector_shape != detdata.detector_shape:
            msg = f"Input detdata container has different det shape "
            msg += f"({detdata.detector_shape}) than compressed data "
            msg += f"({detector_shape})"
            raise RuntimeError(msg)
        if dtype != detdata.dtype:
            msg = f"Input detdata container has different dtype "
            msg += f"({detdata.dtype}) than compressed data "
            msg += f"({dtype})"
            raise RuntimeError(msg)
        if units != detdata.units:
            msg = f"Input detdata container has different units "
            msg += f"({detdata.units}) than compressed data "
            msg += f"({units})"
            raise RuntimeError(msg)

    det_stride = 1
    for ds in detector_shape:
        det_stride *= ds

    if comp_params["type"] == "none":
        n_elem = n_det * det_stride
        view = np.ndarray(
            n_elem,
            dtype=dtype,
            buffer=comp_bytes,
        )
        detdata[:] = view.reshape((n_det,) + detector_shape)

    elif comp_params["type"] == "gzip":
        for idet, det in enumerate(detectors):
            slc = slice(comp_ranges[idet][0], comp_ranges[idet][1], 1)
            dbytes = gzip.decompress(comp_bytes[slc])
            view = np.ndarray(
                det_stride,
                dtype=dtype,
                buffer=dbytes,
            ).reshape(detector_shape)
            detdata[det, :] = view

    elif comp_params["type"] == "flac":
        data_offsets = None
        if "data_offsets" in comp_params:
            data_offsets = comp_params["data_offsets"]
        dslices = list()

        data_gains = None
        if "data_gains" in comp_params:
            data_gains = comp_params["data_gains"]
        decompress_detdata_flac(
            detdata,
            comp_bytes,
            comp_ranges,
            det_offsets=data_offsets,
            det_gains=data_gains,
        )

    else:
        msg = f"Compression type \"{comp_params['type']}\" is not supported"
        raise NotImplementedError(msg)

    return detdata
