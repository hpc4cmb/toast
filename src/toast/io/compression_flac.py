# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import time

import numpy as np

from .._libtoast import (
    compress_flac,
    compress_flac_2D,
    decompress_flac,
    decompress_flac_2D,
    have_flac_support,
)
from ..utils import AlignedU8, Logger, dtype_to_aligned


def float2int(data, quanta=None):
    """Convert floating point data to integers.

    This function subtracts the mean and rescales data before rounding to 32bit
    integer values.

    Args:
        data (array):  The floating point data.
        quanta (float):  The floating point quantity corresponding to one integer
            resolution amount in the output.

    Returns:
        (tuple):  The (integer data, offset, gain)

    """
    dmin = np.amin(data)
    dmax = np.amax(data)
    offset = 0.5 * (dmin + dmax)
    amp = 1.01 * max(np.abs(dmin - offset), np.abs(dmax - offset))
    if quanta is None:
        # Use the full bit range of int32 FLAC.  Actually we lose one bit
        # Due to internal FLAC implementation.
        max_flac = np.iinfo(np.int32).max // 2
        quanta = amp / max_flac
    if quanta == 0:
        # This can happen if fed a vector of all zeros
        quanta = 1.0
    gain = 1.0 / quanta

    return (
        np.array(np.around(gain * (data - offset)), dtype=np.int32),
        offset,
        gain,
    )


def int64to32(data):
    """Convert 64bit integers to 32bit.

    This finds the 64bit integer mean and subtracts it.  It then checks that the value
    will fit in 32bit integers.  If you want to treat the integer values as floating
    point data, use float2int instead.

    Args:
        data (array):  The 64bit integer data.

    Returns:
        (tuple):  The (integer data, offset)

    """
    if data.dtype != np.dtype(np.int64):
        raise ValueError("Only int64 data is supported")

    dmin = np.amin(data)
    dmax = np.amax(data)
    offset = int(round(0.5 * (dmin + dmax)))

    temp = np.array(data - offset, dtype=np.int64)

    # FLAC uses an extra bit...
    flac_max = np.iinfo(np.int32).max // 2

    bad = np.logical_or(temp > flac_max, temp < -flac_max)
    n_bad = np.count_nonzero(bad)
    if n_bad > 0:
        msg = f"64bit integers minus {offset} has {n_bad} values outside 32bit range"
        raise RuntimeError(msg)

    return (
        temp.astype(np.int32),
        offset,
    )


def int2float(idata, offset, gain):
    """Restore floating point data from integers.

    The gain and offset are applied and the resulting float32 data is returned.

    Args:
        idata (array):  The 32bit integer data.
        offset (float):  The offset used in the original conversion.
        gain (float):  The gain used in the original conversion.

    Returns:
        (array):  The restored float32 data.

    """
    if len(idata.shape) > 1:
        raise ValueError("Only works with flat packed arrays")
    coeff = 1.0 / gain
    return np.array((idata * coeff) + offset, dtype=np.float32)


def compress_detdata_flac(detdata, level=5, quanta=None):
    """Compress a 2D DetectorData array into FLAC bytes.

    The input data is converted to 32bit integers.  The "quanta" value is used
    for floating point data conversion and represents the floating point increment
    for a single integer value.  If quanta is None, each detector timestream is
    scaled independently based on its data range.  If quanta is a scalar, all
    detectors are scaled with the same value.  If quanta is an array, it specifies
    the scaling independently for each detector.

    The following rules specify the data conversion that is performed depending on
    the input type:

        int32:  No conversion.

        int64:  Subtract the integer closest to the mean, then truncate to lower
            32 bits, and check that the higher bits were zero.

        float32:  Subtract the mean and scale data based on the quanta value (see
            above).  Then round to nearest 32bit integer.

        float64:  Subtract the mean and scale data based on the quanta value (see
            above).  Then round to nearest 32bit integer.

    After conversion to 32bit integers, each detector's data is separately compressed
    into a sequence of FLAC bytes, which is appended to the total.  The offset in
    bytes for each detector is recorded.

    Args:
        detdata (DetectorData):  The input detector data.
        level (int):  Compression level
        quanta (array):  For floating point data, the increment of each integer.

    Returns:
        (tuple):  The (compressed bytes, byte ranges,
            [detector value offset, detector value gain])

    """
    if not have_flac_support():
        raise RuntimeError("TOAST was not compiled with libFLAC support")

    if quanta is not None:
        try:
            nq = len(quanta)
            # This is a sequence
            if nq != len(detdata.detectors):
                raise ValueError(
                    "If not a scalar, quanta must have a value for each detector"
                )
            dquanta = quanta
        except TypeError:
            # This is a scalar, applied to all detectors
            dquanta = quanta * np.ones(len(detdata.detectors), dtype=np.float64)
    else:
        dquanta = [None for x in detdata.detectors]

    comp_bytes = AlignedU8()

    if detdata.dtype == np.dtype(np.int32):
        pass
    elif detdata.dtype == np.dtype(np.int64):
        data_ioffsets = np.zeros(len(detdata.detectors), dtype=np.int64)
    elif detdata.dtype == np.dtype(np.float32) or detdata.dtype == np.dtype(np.float64):
        data_offsets = np.zeros(len(detdata.detectors), dtype=np.float64)
        data_gains = np.ones(len(detdata.detectors), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported data type '{detdata.dtype}'")

    start_bytes = list()
    for idet, det in enumerate(detdata.detectors):
        cur = comp_bytes.size()
        start_bytes.append(cur)

        if detdata.dtype == np.dtype(np.int32):
            intdata = detdata[idet].reshape(-1)
        elif detdata.dtype == np.dtype(np.int64):
            intdata, ioff = int64to32(detdata[idet, :].reshape(-1))
            data_ioffsets[idet] = ioff
        else:
            intdata, foff, fgain = float2int(
                detdata[idet, :].reshape(-1), quanta=dquanta[idet]
            )
            data_offsets[idet] = foff
            data_gains[idet] = fgain

        dbytes = compress_flac(intdata, level)
        ext = len(dbytes)
        comp_bytes.resize(cur + ext)
        comp_bytes[cur : cur + ext] = dbytes

    comp_ranges = list()
    for idet in range(len(detdata.detectors)):
        if idet == len(detdata.detectors) - 1:
            comp_ranges.append((start_bytes[idet], comp_bytes.size()))
        else:
            comp_ranges.append((start_bytes[idet], start_bytes[idet + 1]))

    if detdata.dtype == np.dtype(np.int32):
        return (comp_bytes.array(), comp_ranges)
    elif detdata.dtype == np.dtype(np.int64):
        return (comp_bytes.array(), comp_ranges, data_ioffsets)
    else:
        return (comp_bytes.array(), comp_ranges, data_offsets, data_gains)


def decompress_detdata_flac(
    detdata, flacbytes, byte_ranges, det_offsets=None, det_gains=None
):
    """Decompress FLAC bytes into a DetectorData array.

    Given an existing DetectorData object, decompress individual detector data from
    the FLAC byte stream given the starting byte for each detector and optionally
    the offset and gain factor to apply to convert the native 32bit integers into
    the output type.

    Args:
        detdata (DetectorData):  The object to fill with decompressed data.
        flacbytes (array):  Compressed FLAC bytes
        byte_ranges (list):  The byte range for each detector.
        det_offsets (array):  The offset to apply to each detector during type
            conversion.
        det_gains (array):  The scale factor to apply to each detector during
            type conversion.

    Returns:
        None

    """
    if not have_flac_support():
        raise RuntimeError("TOAST was not compiled with libFLAC support")

    # Since we may have discontiguous slices into the bytestream, we decompress
    # all data types one detector at a time.

    for idet, det in enumerate(detdata.detectors):
        slc = slice(byte_ranges[idet][0], byte_ranges[idet][1], 1)
        idata = decompress_flac(flacbytes[slc])
        if detdata.dtype == np.dtype(np.int32):
            # Just copy it into place
            detdata[idet] = idata.array().reshape(detdata.detector_shape)
        elif detdata.dtype == np.dtype(np.int64):
            detdata[idet] = idata.array().reshape(detdata.detector_shape)
            if det_offsets is not None:
                detdata[idet] += det_offsets[idet]
        elif detdata.dtype == np.dtype(np.float32) or detdata.dtype == np.dtype(
            np.float64
        ):
            if det_offsets is None:
                doff = 0.0
            else:
                doff = det_offsets[idet]
            if det_gains is None:
                dgain = 1.0
            else:
                dgain = det_gains[idet]
            detdata[idet] = int2float(idata.array(), doff, dgain).reshape(
                detdata.detector_shape
            )
        else:
            raise ValueError(f"Unsupported data type '{detdata.dtype}'")
