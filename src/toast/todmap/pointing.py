# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import healpy as hp

from ..qarray import rotate

from ..utils import Environment

from ..healpix import HealpixPixels

from ..op import Operator

from ..timing import function_timer

from .._libtoast import pointing_matrix_healpix


class OpPointingHpix(Operator):
    """
    Operator which generates I/Q/U healpix pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.  An optional dictionary of calibration factors may
    be specified.  Additional options include specifying a constant cross-polar
    response (eps) and a rotating, perfect half-wave plate.  The timestream
    model is then (see Jones, et al, 2006):

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a} + U \\sin{2a}\\right]\\right]

    Or, if a HWP is included in the response with time varying angle "w", then
    the total response is:

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a+4w} + U \\sin{2a+4w}\\right]\\right]

    Args:
        pixels (str): write pixels to the cache with name <pixels>_<detector>.
            If the named cache objects do not exist, then they are created.
        weights (str): write pixel weights to the cache with name
            <weights>_<detector>.  If the named cache objects do not exist,
            then they are created.
        nside (int): NSIDE resolution for Healpix NEST ordered intensity map.
        nest (bool): if True, use NESTED ordering.
        mode (string): either "I" or "IQU"
        cal (dict): dictionary of calibration values per detector. A None
            value means a value of 1.0 for all detectors.
        epsilon (dict): dictionary of cross-polar response per detector. A
            None value means epsilon is zero for all detectors.
        common_flag_name (str): the optional name of a cache object to use for
            the common flags.
        common_flag_mask (byte): the bitmask to use when flagging the pointing
            matrix using the common flags.
        apply_flags (bool): whether to read the TOD common flags, bitwise OR
            with the common_flag_mask, and then flag the pointing matrix.
        single_precision (bool):  Return the pixel numbers and pointing
             weights in single precision.  Default=False.
        nside_submap (int):  Size of a submap is 12 * nside_submap ** 2
    """

    def __init__(
        self,
        pixels="pixels",
        weights="weights",
        nside=64,
        nest=False,
        mode="I",
        cal=None,
        epsilon=None,
        common_flag_name=None,
        common_flag_mask=255,
        apply_flags=False,
        keep_quats=False,
        single_precision=False,
        nside_submap=16,
    ):
        self._pixels = pixels
        self._weights = weights
        self._nside = nside
        self._nest = nest
        self._mode = mode
        self._cal = cal
        self._epsilon = epsilon
        self._common_flag_mask = common_flag_mask
        self._apply_flags = apply_flags
        self._common_flag_name = common_flag_name
        self._keep_quats = keep_quats
        self._single_precision = single_precision
        self._nside_submap = min(nside, nside_submap)
        self._npix_submap = 12 * self._nside_submap ** 2
        self._nsubmap = (self._nside // self._nside_submap) ** 2
        self._hit_submaps = np.zeros(self._nsubmap, dtype=bool)

        # initialize the healpix pixels object
        self.hpix = HealpixPixels(self._nside)

        if self._mode == "I":
            self._nnz = 1
        elif self._mode == "IQU":
            self._nnz = 3
        else:
            raise RuntimeError("Unsupported mode")

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    @property
    def nside(self):
        """(int): the HEALPix NSIDE value used."""
        return self._nside

    @property
    def nest(self):
        """(bool): if True, the pointing is NESTED ordering."""
        return self._nest

    @property
    def mode(self):
        """(str): the pointing mode "I", "IQU", etc."""
        return self._mode

    @property
    def local_submaps(self):
        """(list): Indices of locally hit submaps"""
        if self._single_precision:
            dtype = np.int32
        else:
            dtype = np.int64
        return np.arange(self._nsubmap, dtype=dtype)[self._hit_submaps]

    @function_timer
    def exec(self, data):
        """Create pixels and weights.

        This iterates over all observations and detectors, and creates
        the pixel and weight arrays representing the pointing matrix.
        This data is stored in the TOD cache.

        Args:
            data (toast.Data): The distributed data.

        """
        env = Environment.get()
        tod_buffer_length = env.tod_buffer_length()

        for obs in data.obs:
            tod = obs["tod"]

            # compute effective sample rate

            times = tod.local_times()
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt
            del times

            offset, nsamp = tod.local_samples

            hwpang = None
            try:
                hwpang = tod.local_hwp_angle()
            except:
                hwpang = None

            # read the common flags and apply bitmask

            common = None
            if self._apply_flags:
                common = tod.local_common_flags(self._common_flag_name)
                common = common & self._common_flag_mask
            else:
                common = np.zeros(nsamp, dtype=np.uint8)

            for det in tod.local_dets:
                eps = 0.0
                if self._epsilon is not None:
                    eps = self._epsilon[det]

                cal = 1.0
                if self._cal is not None:
                    cal = self._cal[det]

                # Create cache objects and use that memory directly

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)

                pixelsref = None
                weightsref = None

                if tod.cache.exists(pixelsname):
                    pixelsref = tod.cache.reference(pixelsname)
                else:
                    pixelsref = tod.cache.create(pixelsname, np.int64, (nsamp,))

                if tod.cache.exists(weightsname):
                    weightsref = tod.cache.reference(weightsname)
                else:
                    weightsref = tod.cache.create(
                        weightsname, np.float64, (nsamp, self._nnz)
                    )

                pdata = None
                if self._keep_quats:
                    # We are keeping the detector quaternions, so cache
                    # them now for the full sample range.
                    pdata = tod.local_pointing(det)

                buf_off = 0
                buf_n = tod_buffer_length
                while buf_off < nsamp:
                    if buf_off + buf_n > nsamp:
                        buf_n = nsamp - buf_off
                    bslice = slice(buf_off, buf_off + buf_n)

                    detp = None
                    if pdata is None:
                        # Read and discard
                        detp = tod.read_pntg(detector=det, local_start=buf_off, n=buf_n)
                    else:
                        # Use cached version
                        detp = pdata[bslice, :]

                    hslice = None
                    if hwpang is not None:
                        hslice = hwpang[bslice].reshape(-1)
                    fslice = common[bslice].reshape(-1)

                    pointing_matrix_healpix(
                        self.hpix,
                        self._nest,
                        eps,
                        cal,
                        self._mode,
                        detp.reshape(-1),
                        hslice,
                        fslice,
                        pixelsref[bslice].reshape(-1),
                        weightsref[bslice, :].reshape(-1),
                    )
                    buf_off += buf_n

                if self._single_precision:
                    pixels = pixelsref.astype(np.int32)
                    del pixelsref
                    pixelsref = tod.cache.put(pixelsname, pixels, replace=True)
                    del pixels
                    weights = weightsref.astype(np.float32)
                    del weightsref
                    weightsref = tod.cache.put(weightsname, weights, replace=True)
                    del weights

                self._hit_submaps[pixelsref // self._npix_submap] = True

                del pixelsref
                del weightsref
                del pdata

            del common

        # Store the local submaps in the data object under the same name
        # as the pixel numbers

        if self._single_precision:
            dtype = np.int32
        else:
            dtype = np.int64

        local_submaps = np.arange(self._nsubmap, dtype=dtype)[self._hit_submaps]
        submap_name = "{}_local_submaps".format(self._pixels)
        data[submap_name] = local_submaps
        npix_submap_name = "{}_npix_submap".format(self._pixels)
        data[npix_submap_name] = self._npix_submap
        nsubmap_name = "{}_nsubmap".format(self._pixels)
        data[nsubmap_name] = self._nsubmap
        npix_name = "{}_npix".format(self._pixels)
        data[npix_name] = 12 * self._nside ** 2

        return


class OpMuellerPointingHpix(Operator):
    """Operator which generates I/Q/U healpix pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.  An optional dictionary of calibration factors may
    be specified.  Additional options include specifying a constant cross-polar
    response (eps) and a set of HWP Mueller matrix parameters.  The timestream
    model is then (see Jones, et al, 2006):

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a} + U \\sin{2a}\\right]\\right]

    Or, if a HWP is included in the response with time varying angle "w", then
    the total response is:

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a+4w} + U \\sin{2a+4w}\\right]\\right]

    Args:
        pixels (str): write pixels to the cache with name <pixels>_<detector>.
            If the named cache objects do not exist, then they are created.
        weights (str): write pixel weights to the cache with name
            <weights>_<detector>.  If the named cache objects do not exist,
            then they are created.
        nside (int): NSIDE resolution for Healpix NEST ordered intensity map.
        nest (bool): if True, use NESTED ordering.
        mode (string): either "I" or "IQU"
        cal (dict): dictionary of calibration values per detector. A None
            value means a value of 1.0 for all detectors.
        epsilon (dict): dictionary of cross-polar response per detector. A
            None value means epsilon is zero for all detectors.
        common_flag_name (str): the optional name of a cache object to use for
            the common flags.
        common_flag_mask (byte): the bitmask to use when flagging the pointing
            matrix using the common flags.
        apply_flags (bool): whether to read the TOD common flags, bitwise OR
            with the common_flag_mask, and then flag the pointing matrix.
        single_precision (bool):  Return the pixel numbers and pointing
            weights in single precision.  Default=False.
        nside_submap (int):  Size of a submap is 12 * nside_submap ** 2
        hwp_parameters_set (tuple) : tuple of  parameters describing
            the Mueller Matrix [T, c,rho,s ] as defined in Bryan et al. 2010 :

            .. math::
                M_{HWP}=
                \begin{pmatrix}
                    T & \rho  & 0 & 0 \\
                    \rho & T & 0 &  0 \\
                      0  & 0  & c & -s  \\
                      0  & 0  & s & c  \\
                \end{pmatrix}

            default assumes an ideal HWP i.e. T=1,c=-1,rho=0,s=0.

    """

    def __init__(
        self,
        pixels="pixels",
        weights="weights",
        nside=64,
        nest=False,
        mode="I",
        cal=None,
        epsilon=None,
        common_flag_name=None,
        common_flag_mask=255,
        apply_flags=False,
        keep_quats=False,  # this should be set to True to work in L480
        single_precision=False,
        nside_submap=16,
        hwp_parameters_set=[1, -1, 0, 0],
    ):
        self._pixels = pixels
        self._weights = weights
        self._nside = nside
        self._nest = nest
        self._mode = mode
        self._cal = cal
        self._epsilon = epsilon
        self._common_flag_mask = common_flag_mask
        self._apply_flags = apply_flags
        self._common_flag_name = common_flag_name
        self._keep_quats = keep_quats
        self._single_precision = single_precision
        self._nside_submap = min(nside, nside_submap)
        self._npix_submap = 12 * self._nside_submap ** 2
        self._nsubmap = (self._nside // self._nside_submap) ** 2
        self._hit_submaps = np.zeros(self._nsubmap, dtype=bool)
        self._hwp_parameters_set = hwp_parameters_set
        # initialize the healpix pixels object
        self.hpix = HealpixPixels(self._nside)

        if self._mode == "I":
            self._nnz = 1
        elif self._mode == "IQU":
            self._nnz = 3
        else:
            raise RuntimeError("Unsupported mode")

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    @property
    def nside(self):
        """(int): the HEALPix NSIDE value used."""
        return self._nside

    @property
    def nest(self):
        """(bool): if True, the pointing is NESTED ordering."""
        return self._nest

    @property
    def mode(self):
        """(str): the pointing mode "I", "IQU", etc."""
        return self._mode

    @property
    def local_submaps(self):
        """(list): Indices of locally hit submaps"""
        if self._single_precision:
            dtype = np.int32
        else:
            dtype = np.int64
        return np.arange(self._nsubmap, dtype=dtype)[self._hit_submaps]

    @function_timer
    def exec(self, data):
        """Create pixels and weights.

        This iterates over all observations and detectors, and creates
        the pixel and weight arrays representing the pointing matrix.
        This data is stored in the TOD cache.

        Args:
            data (toast.Data): The distributed data.

        """
        env = Environment.get()
        tod_buffer_length = env.tod_buffer_length()

        for obs in data.obs:
            tod = obs["tod"]

            # compute effective sample rate

            times = tod.local_times()
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt
            del times

            offset, nsamp = tod.local_samples

            hwpang = None
            try:
                hwpang = tod.local_hwp_angle()
            except:
                hwpang = None
                raise RuntimeError("Can't run this operator if hwpang is None ")

            # read the common flags and apply bitmask

            common = None
            if self._apply_flags:
                common = tod.local_common_flags(self._common_flag_name)
                common = common & self._common_flag_mask
            else:
                common = np.zeros(nsamp, dtype=np.uint8)

            for det in tod.local_dets:
                eps = 0.0
                if self._epsilon is not None:
                    eps = self._epsilon[det]

                cal = 1.0
                if self._cal is not None:
                    cal = self._cal[det]

                # Create cache objects and use that memory directly

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)

                pixelsref = None
                weightsref = None

                if tod.cache.exists(pixelsname):
                    pixelsref = tod.cache.reference(pixelsname)
                else:
                    pixelsref = tod.cache.create(pixelsname, np.int64, (nsamp,))

                if tod.cache.exists(weightsname):
                    weightsref = tod.cache.reference(weightsname)
                else:
                    weightsref = tod.cache.create(
                        weightsname, np.float64, (nsamp, self._nnz)
                    )

                pdata = None
                if self._keep_quats:
                    # We are keeping the detector quaternions, so cache
                    # them now for the full sample range.
                    pdata = tod.local_pointing(det)

                xaxis = np.array([1.0, 0.0, 0.0])
                zaxis = np.array([0.0, 0.0, 1.0])
                nullquat = np.array([0.0, 0.0, 0.0, 1.0])
                eta = (1.0 - eps) / (1.0 + eps)

                pdata[common] = nullquat

                dir = rotate(pdata, zaxis)
                pixelsref = hp.vec2pix(
                    nside=self._nside, x=dir[:, 0], y=dir[:, 1], z=dir[:, 2], nest=True
                )

                pixelsref[common] = -1
                # import pdb
                # pdb.set_trace()

                T = self._hwp_parameters_set[0]
                c = self._hwp_parameters_set[1]
                rho = self._hwp_parameters_set[2]
                s = self._hwp_parameters_set[3]
                cos2hwp = np.cos(2 * hwpang)

                if self._mode == "I":
                    weightsref = cal * (T + eta * rho * cos2hwp)
                elif self._mode == "IQU":
                    orient = rotate(pdata, xaxis)

                    by = orient[:, 0] * dir[:, 1] - orient[:, 1] * dir[:, 0]
                    bx = (
                        orient[:, 0] * (-dir[:, 2] * dir[:, 0])
                        + orient[:, 1] * (-dir[:, 2] * dir[:, 1])
                        + orient[:, 2] * (dir[:, 0] * dir[:, 0] + dir[:, 1] * dir[:, 1])
                    )

                    detang = np.arctan2(by, bx)

                    sindetang = np.sin(2 * detang)
                    cosdetang = np.cos(2 * detang)

                    ang4 = 2 * detang + 4 * hwpang
                    ang2 = 2 * detang + 2 * hwpang

                    sin4 = np.sin(ang4)
                    sin2 = np.sin(ang2)
                    cos4 = np.cos(ang4)
                    cos2 = np.cos(ang2)

                    weightsref[:, 0] = cal * (T + (eta * rho * cos2hwp))
                    weightsref[:, 1] = cal * (
                        (rho * cos2)
                        + eta * (T + c) / 2.0 * cosdetang
                        + eta * (T - c) / 2.0 * cos4
                    )
                    weightsref[:, 2] = cal * (
                        (rho * sin2)
                        + eta * (T + c) / 2.0 * sindetang
                        + eta * (T - c) / 2.0 * sin4
                    )
                else:
                    raise RuntimeError("Unknown healpix pointing matrix mode")

                if self._single_precision:
                    pixels = pixelsref.astype(np.int32)
                    del pixelsref
                    pixelsref = tod.cache.put(pixelsname, pixels, replace=True)
                    del pixels
                    weights = weightsref.astype(np.float32)
                    del weightsref
                    weightsref = tod.cache.put(weightsname, weights, replace=True)
                    del weights

                self._hit_submaps[pixelsref // self._npix_submap] = True

                del pixelsref
                del weightsref
                del pdata

            del common

        # Store the local submaps in the data object under the same name
        # as the pixel numbers

        if self._single_precision:
            dtype = np.int32
        else:
            dtype = np.int64

        local_submaps = np.arange(self._nsubmap, dtype=dtype)[self._hit_submaps]
        submap_name = "{}_local_submaps".format(self._pixels)
        data[submap_name] = local_submaps
        npix_submap_name = "{}_npix_submap".format(self._pixels)
        data[npix_submap_name] = self._npix_submap
        nsubmap_name = "{}_nsubmap".format(self._pixels)
        data[nsubmap_name] = self._nsubmap
        npix_name = "{}_npix".format(self._pixels)
        data[npix_name] = 12 * self._nside ** 2

        return
