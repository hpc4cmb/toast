# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from ..utils import Environment, Logger

try:
    # First see if we are using spt3g bundled with so3g
    import so3g
    from spt3g import core as c3g

    available = True
except ImportError:
    # Try plain spt3g import
    try:
        from spt3g import core as c3g

        available = True
    except ImportError:
        available = False


def from_g3_scalar_type(val, unit=None):
    if unit is not None:
        return u.Quantity(val * from_g3_unit(unit))
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return float(val)
    check = str(val)
    if check == "NONE":
        return None
    else:
        return check


def to_g3_scalar_type(val):
    if isinstance(val, u.Quantity):
        gunit, scale = to_g3_unit(val.unit)
        return c3g.G3Double(val.value * scale), gunit
    if val is None:
        return c3g.G3String("NONE"), None
    if isinstance(val, bool):
        return c3g.G3Bool(val), None
    if isinstance(val, int):
        return c3g.G3Int(val), None
    if isinstance(val, float):
        return c3g.G3Double(val), None
    try:
        dc = val.dtype.char
        if dc == "f" or dc == "d":
            return c3g.G3Double(val), None
        else:
            return c3g.G3Int(val), None
    except Exception:
        # This is not a numpy type
        if isinstance(val, str):
            return c3g.G3String(val), None
        else:
            raise RuntimeError(f"Cannot convert '{val}'")


def from_g3_array_type(ar):
    if isinstance(ar, c3g.G3VectorUnsignedChar):
        return np.dtype(np.uint8)
    elif isinstance(ar, c3g.G3VectorInt):
        return np.dtype(np.int32)
    elif isinstance(ar, c3g.G3VectorTime):
        return np.dtype(np.int64)
    else:
        return np.dtype(np.float64)


def to_g3_array_type(dt):
    if dt.char == "B":
        return c3g.G3VectorUnsignedChar
    elif dt.char == "i":
        return c3g.G3VectorInt
    elif dt.char == "l":
        return c3g.G3VectorInt
    else:
        return c3g.G3VectorDouble


def to_g3_map_array_type(dt):
    # FIXME:  We should really request more types upstream...
    if dt.char == "f" or dt.char == "d":
        return c3g.G3MapVectorDouble
    else:
        return c3g.G3MapVectorInt


# FIXME:  For now, we treat "Counts" and "None" as equivalent.  We could
# consider doing something smarter, but that would require other information.


def to_g3_unit(aunit):
    """Convert astropy unit to G3 timestream unit.

    We convert our input units to SI base units (no milli-, micro-, etc prefix).
    We also return the scale factor needed to transform to this unit.

    Args:
        aunit (astropy.unit):  The input unit.

    Returns:
        (tuple):  The G3TimestreamUnit, scale factor.

    """
    if aunit == u.dimensionless_unscaled:
        return (c3g.G3TimestreamUnits.Counts, 1.0)
    else:
        scale = 1.0 * aunit
        # Try to convert the units to the various types of quantities
        try:
            scale = scale.to(u.volt)
            return (c3g.G3TimestreamUnits.Voltage, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.watt)
            return (c3g.G3TimestreamUnits.Power, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.ohm)
            return (c3g.G3TimestreamUnits.Resistance, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.ampere)
            return (c3g.G3TimestreamUnits.Current, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.meter)
            return (c3g.G3TimestreamUnits.Distance, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.pascal)
            return (c3g.G3TimestreamUnits.Pressure, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.radian)
            return (c3g.G3TimestreamUnits.Angle, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.Jy)
            return (c3g.G3TimestreamUnits.FluxDensity, scale.value)
        except Exception:
            pass
        try:
            scale = scale.to(u.Kelvin)
            return (c3g.G3TimestreamUnits.Tcmb, scale.value)
        except Exception:
            pass
        raise RuntimeError(f"Cannot convert unit '{aunit}' into G3TimestreamUnit value")


def from_g3_unit(gunit):
    """Convert G3 timestream unit to astropy.

    This function assumes that the quantities are in SI units.

    Args:
        gunit (G3TimestreamUnit):  The input units.

    Returns:
        (astropy.unit):  The astropy unit.

    """
    if gunit == c3g.G3TimestreamUnits.Counts:
        return u.dimensionless_unscaled
    elif gunit == c3g.G3TimestreamUnits.Voltage:
        return u.volt
    elif gunit == c3g.G3TimestreamUnits.Power:
        return u.watt
    elif gunit == c3g.G3TimestreamUnits.Resistance:
        return u.ohm
    elif gunit == c3g.G3TimestreamUnits.Current:
        return u.ampere
    elif gunit == c3g.G3TimestreamUnits.Distance:
        return u.meter
    elif gunit == c3g.G3TimestreamUnits.Pressure:
        return u.pascal
    elif gunit == c3g.G3TimestreamUnits.Angle:
        return u.radian
    elif gunit == c3g.G3TimestreamUnits.FluxDensity:
        return u.Jy
    elif gunit == c3g.G3TimestreamUnits.Tcmb:
        return u.Kelvin
    else:
        return u.dimensionless_unscaled


def compress_timestream(ts, params, rmstarget=2**10, rmsmode="white"):
    """Use FLAC compression to compress a timestream.

    `ts` is a G3Timestream.  Returns a new G3Timestream for same samples as ts, but
    with data scaled and translated with gain and offset, rounded, and with FLAC
    compression enabled.

    Args:
        ts (G3Timestream) :  Input signal
        params (None, bool or dict) :  If None, False or an empty dict,
             no compression or casting to integers.  If True or
             non-empty dictionary, enable compression.  Expected fields
             in the dictionary ('rmstarget', 'gain', 'offset', 'rmsmode')
             allow overriding defaults.
        rmstarget (float) :  Scale the iput signal to have this RMS.
            Should be much smaller then the 24-bit integer range:
            [-2 ** 23 : 2 ** 23] = [-8,388,608 : 8,388,608].
            The gain will be reduced if the scaled signal does
            not fit within the range of allowed values.
        rmsmode (string) : "white" or "full", determines how the
            signal RMS is measured.
    Returns:
        new_ts (G3Timestream) :  Scaled and translated timestream
            with the FLAC compression enabled
        gain (float) :  The applied gain
        offset (float) :  The removed offset

    """
    if params is None or not params:
        return ts, 1, 0
    gain = None
    offset = None
    if isinstance(params, dict):
        if "rmsmode" in params:
            rmsmode = params["rmsmode"]
        if "rmstarget" in params:
            rmstarget = params["rmstarget"]
        if "gain" in params:
            gain = params["gain"]
        if "offset" in params:
            offset = params["offset"]
    v = np.array(ts)
    vmin = np.amin(v)
    vmax = np.amax(v)
    if offset is None:
        offset = 0.5 * (vmin + vmax)
        amp = vmax - offset
    else:
        amp = np.max(np.abs(vmin - offset), np.abs(vmax - offset))
    if gain is None:
        if rmsmode == "white":
            rms = np.std(np.diff(v)) / np.sqrt(2)
        elif rmsmode == "full":
            rms = np.std(v)
        else:
            raise RuntimeError("Unrecognized RMS mode = '{}'".format(rmsmode))
        if rms == 0:
            gain = 1
        else:
            gain = rmstarget / rms
        # If the data have extreme outliers, we have to reduce the gain
        # to fit the 24-bit signed integer range
        while amp * gain >= 2**23:
            gain *= 0.5
    elif amp * gain >= 2**23:
        raise RuntimeError("The specified gain and offset saturate the band.")
    v = np.round((v - offset) * gain)
    new_ts = c3g.G3Timestream(v)
    new_ts.units = c3g.G3TimestreamUnits.Counts
    new_ts.SetFLACCompression(True)
    new_ts.start = ts.start
    new_ts.stop = ts.stop
    return new_ts, gain, offset


def decompress_timestream(ts, gain, offset, units):
    """Decompress a FLAC encoded timestream.

    `ts` is a G3Timestream.  Returns a new G3Timestream for same samples as ts, but
    with data decompressed using the gain and offset.

    Args:
        ts (G3Timestream) :  Input signal
        gain (float):  The gain that was applied during compression
        offset (float):  The offset that was applied during compression
        units (G3TimestreamUnits):  The units of the output timestream
    Returns:
        new_ts (G3Timestream) :  Decompressed timestream.

    """
    v = np.array(ts)
    inv_gain = 1.0 / gain
    v *= inv_gain
    v += offset
    new_ts = c3g.G3Timestream(v)
    new_ts.units = units
    new_ts.SetFLACCompression(False)
    new_ts.start = ts.start
    new_ts.stop = ts.stop
    return new_ts


def from_g3_time(input):
    """Convert a G3Time scalar or vector into seconds.

    Args:
        input (G3Time or G3VectorTime):  The input.

    Returns:
        (array):  Array of float64 seconds
    """
    scale = 1 / 1e8
    if isinstance(input, c3g.G3Time):
        return scale * input.time
    else:
        return np.array([scale * x.time for x in input], dtype=np.float64)


def to_g3_time(input):
    """Convert float64 seconds into G3Time values.

    Args:
        input (float or array):  The input.

    Returns:
        (G3Time or G3TimeVector):  Output G3Time values
    """
    if np.isscalar(input):
        return c3g.G3Time(input * 1e8)
    else:
        # This seems to be the only way to construct a G3VectorTime- using a list
        # of G3Time values.
        return c3g.G3VectorTime([c3g.G3Time(x * 1e8) for x in input])


def from_g3_quats(input):
    """Convert Spt3G quaternions into TOAST format.

    Spt3G uses boost-format storage order, while TOAST uses scipy storage order.

    Args:
        input (G3VectorQuat):  The input.

    Returns:
        (array):  The toast.qarray compatible data.

    """
    if isinstance(input, c3g.G3VectorQuat):
        return np.array([np.array([x.b, x.c, x.d, x.a]) for x in input])
    else:
        # Assume it is a scalar
        return np.array([input.b, input.c, input.d, input.a])


def to_g3_quats(input):
    """Convert TOAST quaternions into Spt3G format.

    Spt3G uses boost-format storage order, while TOAST uses scipy storage order.

    Args:
        input (array):  The input.

    Returns:
        (G3VectorQuat):  The Spt3G data.

    """
    if len(input.shape) == 2:
        # Array of values
        return c3g.G3VectorQuat([c3g.quat(x[3], x[0], x[1], x[2]) for x in input])
    else:
        # One value
        return c3g.quat(input[3], input[0], input[1], input[2])


# Callable that just builds a list of frames
class frame_collector(object):
    def __init__(self):
        self.frames = list()

    def __call__(self, frame):
        if frame is not None and frame.type != c3g.G3FrameType.EndProcessing:
            self.frames.append(frame)
        return


# Callable that just emits pre-created frames
class frame_emitter(object):
    def __init__(self, frames=list()):
        self._frames = frames
        self._counter = 0
        self._n_frame = len(frames)
        self._done = False

    def __call__(self, frame):
        if self._done:
            return list()
        if self._counter < self._n_frame:
            self._counter += 1
            return self._frames[self._counter - 1]
        else:
            self._done = True
            return c3g.G3Frame(c3g.G3FrameType.EndProcessing)
