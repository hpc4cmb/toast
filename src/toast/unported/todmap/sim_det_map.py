# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np

from .. import qarray as qa
from .._libtoast import scan_map_float32, scan_map_float64
from ..map import DistPixels
from ..mpi import MPI
from ..operator import Operator
from ..timing import GlobalTimers, function_timer
from ..utils import Environment, Logger


class OpSimGradient(Operator):
    """Generate a fake sky signal as a gradient between the poles.

    This passes through each observation and creates a fake signal timestream
    based on the cartesian Z coordinate of the HEALPix pixel containing the
    detector pointing.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        nside (int): the HEALPix NSIDE value to use.
        min (float): the minimum value to use at the South Pole.
        max (float): the maximum value to use at the North Pole.
        nest (bool): whether to use NESTED ordering.
    """

    def __init__(
        self,
        out="grad",
        nside=512,
        min=-100.0,
        max=100.0,
        nest=False,
        flag_mask=255,
        common_flag_mask=255,
        keep_quats=False,
    ):
        # Call the parent class constructor
        super().__init__()
        self._nside = nside
        self._out = out
        self._min = min
        self._max = max
        self._nest = nest
        self._flag_mask = flag_mask
        self._common_flag_mask = common_flag_mask
        self._keep_quats = keep_quats

    @function_timer
    def exec(self, data):
        """Create the gradient timestreams.

        This pixelizes each detector's pointing and then assigns a
        timestream value based on the cartesian Z coordinate of the pixel
        center.

        Args:
            data (toast.Data): The distributed data.

        """
        zaxis = np.array([0, 0, 1], dtype=np.float64)
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        range = self._max - self._min

        for obs in data.obs:
            tod = obs["tod"]

            offset, nsamp = tod.local_samples

            common = tod.local_common_flags() & self._common_flag_mask

            for det in tod.local_dets:
                flags = tod.local_flags(det) & self._flag_mask
                totflags = flags | common
                del flags

                pdata = tod.local_pointing(det).copy()
                pdata[totflags != 0, :] = nullquat

                dir = qa.rotate(pdata, zaxis)

                pixels = hp.vec2pix(
                    self._nside, dir[:, 0], dir[:, 1], dir[:, 2], nest=self._nest
                )
                x, y, z = hp.pix2vec(self._nside, pixels, nest=self._nest)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                z[totflags != 0] = 0.0

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (nsamp,))
                ref = tod.cache.reference(cachename)
                ref[:] += z
                del ref

                if not self._keep_quats:
                    cachename = "quat_{}".format(det)
                    tod.cache.destroy(cachename)

            del common
        return

    def sigmap(self):
        """(array): Return the underlying signal map (full map on all processes)."""
        range = self._max - self._min
        pix = np.arange(0, 12 * self._nside * self._nside, dtype=np.int64)
        x, y, z = hp.pix2vec(self._nside, pix, nest=self._nest)
        z += 1.0
        z *= 0.5
        z *= range
        z += self._min
        return z


class OpSimScan(Operator):
    """Operator which generates sky signal by scanning from a map.

    The signal to use should already be in a distributed pixel structure,
    and local pointing should already exist.

    Args:
        input_map (DistPixels or string):  Path to the map to load and
            sample.  If tag {detector} is encountered, it will be replaced
            with the actual detector name.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        nnz (int):  Number of non-zero weights per sample
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        mc (int):  Monte Carlo index used in synthezing file names

    """

    def __init__(
        self,
        distmap=None,
        input_map=None,
        pixels="pixels",
        weights="weights",
        nnz=3,
        out="scan",
        dets=None,
        mc=None,
    ):
        # Call the parent class constructor
        super().__init__()
        if input_map is None:
            raise RuntimeError("OpSimScan requires an input map")
        if distmap is not None:
            warnings.warn(
                "`distmap` is deprecated, please use `input_map`", DeprecationWarning
            )
            self._input_map = distmap
        else:
            self._input_map = input_map
        self._pixels = pixels
        self._weights = weights
        self._nnz = nnz
        self._out = out
        self._dets = dets
        self._mc = mc

    @function_timer
    def exec(self, data):
        """Create the timestreams by scanning from the map.

        This loops over all observations and detectors and uses the pointing
        matrix to project the distributed map into a timestream.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            None

        """
        log = Logger.get()

        if isinstance(self._input_map, DistPixels):
            input_map = self._input_map
        elif "{detector}" not in self._input_map:
            fname = self._input_map.format(mc=self._mc)
            if data.comm is None or data.comm.world_rank == 0:
                log.info("Scanning {}".format(fname))
                if not os.path.isfile(fname):
                    raise RuntimeError("Input map not found: {}".format(fname))
            input_map = DistPixels(
                data, nnz=self._nnz, dtype=np.float32, pixels=self._pixels
            )
            input_map.read_healpix_fits(fname)
        else:
            input_map = None

        for obs in data.obs:
            tod = obs["tod"]

            if self._dets is None:
                dets = tod.local_dets
            else:
                dets = self._dets

            for det in dets:
                if input_map is None:
                    if MPI is None:
                        comm = None
                    else:
                        comm = MPI.COMM_SELF
                    filename = self._input_map.format(detector=det, mc=self._mc)
                    if not os.path.isfile(filename):
                        raise RuntimeError("Input map not found: {}".format(filename))
                    detector_map = DistPixels(
                        data,
                        comm=comm,
                        nnz=self._nnz,
                        dtype=np.float32,
                        pixels=self._pixels,
                    )
                    detector_map.read_healpix_fits(filename)
                else:
                    detector_map = input_map

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                nsamp, nnz = weights.shape

                gt = GlobalTimers.get()
                gt.start("OpSimScan.exec.global_to_local")
                sm, lpix = detector_map.global_to_local(pixels)
                gt.stop("OpSimScan.exec.global_to_local")

                maptod = np.zeros(nsamp)
                maptype = np.dtype(detector_map.dtype)
                gt.start("OpSimScan.exec.scan_map")
                if maptype.char == "d":
                    scan_map_float64(
                        detector_map.npix_submap,
                        nnz,
                        sm.astype(np.int64),
                        lpix.astype(np.int64),
                        detector_map.flatdata,
                        weights.astype(np.float64).reshape(-1),
                        maptod,
                    )
                elif maptype.char == "f":
                    scan_map_float32(
                        detector_map.npix_submap,
                        nnz,
                        sm.astype(np.int64),
                        lpix.astype(np.int64),
                        detector_map.flatdata,
                        weights.astype(np.float64).reshape(-1),
                        maptod,
                    )
                else:
                    raise RuntimeError(
                        "Scanning from a map only supports float32 and float64 maps"
                    )
                gt.stop("OpSimScan.exec.scan_map")

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (nsamp,))
                ref = tod.cache.reference(cachename)
                ref[:] += maptod

                del ref
                del pixels
                del weights

        return
