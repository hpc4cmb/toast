# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import numpy as np

from ..dist import Comm, Data
from ..op import Operator
from .pixels import DistPixels

from .. import ctoast as ctoast
import timemory

class OpAccumDiag(Operator):
    """
    Operator which accumulates the diagonal covariance and noise weighted map.

    This operator requires that the local pointing matrix has already been
    computed.  Each process has local pieces of the map products.  This
    operator can optionally accumulate any combination of the hit map,
    the white noise weighted map, and the diagonal inverse pixel covariance.

    NOTE:  The input DistPixels objects (noise weighted map, hit map, and
    inverse covariance) are purposefully NOT set to zero at the start.  This
    allows accumulating multiple instances of the distributed data.  This might
    be needed (for example) if all desired timestream data does not fit into
    memory.  You should manually clear the pixel domain objects before
    accumulation if desired.

    Args:
        zmap (DistPixels):  (optional) the noise weighted map to accumulate.
        hits (DistPixels):  (optional) the hits to accumulate.
        invnpp (DistPixels):  (optional) the diagonal covariance matrix.
        detweights (dictionary): individual noise weights to use for each
            detector.
        name (str): the name of the cache object (<name>_<detector>) to
            use for the detector timestream.  If None, use the TOD.
        flag_name (str): the name of the cache object (<flag_name>_<detector>) to
            use for the detector flags.  If None, use the TOD.
        flag_mask (int): the integer bit mask (0-255) that should be
            used with the detector flags in a bitwise AND.
        common_flag_name (str): the name of the cache object
            (<common_flag_name>_<detector>) to use for the common flags.
            If None, use the TOD.
        common_flag_mask (int): the integer bit mask (0-255) that should be
            used with the common flags in a bitwise AND.
        apply_flags (bool): whether to apply flags to the pixel numbers.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        nside (int): NSIDE resolution for Healpix NEST ordered intensity map.
        nest (bool): if True, use NESTED ordering.
        mode (string): either "I" or "IQU"
        cal (dict): dictionary of calibration values per detector. A None
            value means a value of 1.0 for all detectors.
        epsilon (dict): dictionary of cross-polar response per detector. A
            None value means epsilon is zero for all detectors.
        hwprpm: if None, a constantly rotating HWP is not included.  Otherwise
            it is the rate (in RPM) of constant rotation.
        hwpstep: if None, then a stepped HWP is not included.  Otherwise, this
            is the step in degrees.
        hwpsteptime: The time in minutes between HWP steps.
    """

    def __init__(
            self, zmap=None, hits=None, invnpp=None, detweights=None,
            name=None, flag_name=None, flag_mask=255, common_flag_name=None,
            common_flag_mask=255, pixels='pixels', weights='weights',
            apply_flags=True):

        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._apply_flags = apply_flags

        self._name = name
        self._pixels = pixels
        self._weights = weights
        self._detweights = detweights

        # Ensure that the 3 different DistPixel objects have the same number
        # of pixels.

        self._nsub = None
        self._subsize = None
        self._nnz = None

        self._do_z = False
        self._do_hits = False
        self._do_invn = False

        self._zmap = zmap
        self._hits = hits
        self._invnpp = invnpp

        self._globloc = None

        if zmap is not None:
            self._do_z = True
            self._nsub = zmap.nsubmap
            self._subsize = zmap.submap
            self._nnz = zmap.nnz
            if self._globloc is None:
                self._globloc = self._zmap

        if hits is not None:
            self._do_hits = True
            if hits.nnz != 1:
                raise RuntimeError("Hit map should always have NNZ == 1")
            if self._nsub is None:
                self._nsub = hits.nsubmap
                self._subsize = hits.submap
            else:
                if self._nsub != hits.nsubmap:
                    raise RuntimeError("All pixel domain objects must have the same submap size.")
                if self._subsize != hits.submap:
                    raise RuntimeError("All pixel domain objects must have the same submap size.")
            if self._globloc is None:
                self._globloc = self._hits

        if invnpp is not None:
            self._do_invn = True
            block = invnpp.nnz
            blocknnz = int( ( (np.sqrt(8 * block) - 1) / 2 ) + 0.5 )
            if self._nsub is None:
                self._nsub = invnpp.nsubmap
                self._subsize = invnpp.submap
                self._nnz = blocknnz
            else:
                if self._nsub != invnpp.nsubmap:
                    raise RuntimeError("All pixel domain objects must have the same submap size.")
                if self._subsize != invnpp.submap:
                    raise RuntimeError("All pixel domain objects must have the same submap size.")
                if self._nnz is None:
                    self._nnz = blocknnz
                elif self._nnz != blocknnz:
                    raise RuntimeError("All pixel domain objects must have the same submap size.")
            if self._globloc is None:
                self._globloc = self._invnpp

        if self._nnz is None:
            # this means we only have a hit map
            self._nnz = 1

        if self._do_invn and (not self._do_hits):
            raise RuntimeError("When accumulating the diagonal pixel covariance, you must also accumulate the hit map")

        if self._do_z and (self._do_hits != self._do_invn):
            raise RuntimeError("When accumulating the noise weighted map, you must accumulate either both the hits and covariance or neither.")

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    def exec(self, data):
        """
        Iterate over all observations and detectors and accumulate.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timemory.auto_timer(type(self).__name__)
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        for obs in data.obs:
            tod = obs['tod']

            nsamp = tod.local_samples[1]

            commonflags = None
            if self._apply_flags:
                commonflags = tod.local_common_flags(
                    self._common_flag_name).copy()
                commonflags &= self._common_flag_mask

            for det in tod.local_dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                cachename = None
                signal = None

                if self._do_z:
                    signal = tod.local_signal(det, self._name)

                # get flags

                if self._apply_flags:
                    detflags = tod.local_flags(det, self._flag_name)
                    flags = np.logical_or((detflags & self._flag_mask) != 0,
                                          commonflags != 0)

                    del detflags

                    # Don't change the cached pixel numbers
                    pixels = pixels.copy()
                    pixels[flags] = -1

                # local pointing

                sm, lpix = self._globloc.global_to_local(pixels)

                detweight = 1.0

                if self._detweights is not None:
                    if det not in self._detweights.keys():
                        raise RuntimeError("no detector weights found for {}"
                                           "".format(det))
                    detweight = self._detweights[det]
                    if detweight == 0:
                        continue

                # Now call the correct accumulation operator depending
                # on which input pixel objects were given.

                if self._do_invn and self._do_z:

                    ctoast.cov_accumulate_diagonal(
                        self._nsub, self._subsize, self._nnz, nsamp, sm, lpix,
                        weights, detweight, signal, self._zmap.data,
                        self._hits.data, self._invnpp.data)

                elif self._do_invn:

                    ctoast.cov_accumulate_diagonal_invnpp(
                        self._nsub, self._subsize, self._nnz, nsamp, sm, lpix,
                        weights, detweight, self._hits.data, self._invnpp.data)

                elif self._do_z:

                    ctoast.cov_accumulate_zmap(self._nsub, self._subsize,
                        self._nnz, nsamp, sm, lpix, weights, detweight, signal,
                        self._zmap.data)

                elif self._do_hits:

                    ctoast.cov_accumulate_diagonal_hits(self._nsub,
                        self._subsize, self._nnz, nsamp, sm, lpix, self._hits.data)

                # print("det {}:".format(det))
                # if self._zmap is not None:
                #     print(self._zmap.data)
                # if self._hits is not None:
                #     print(self._hits.data)
                # if self._invnpp is not None:
                #     print(self._invnpp.data)

        return


def covariance_invert(npp, threshold, rcond=None):
    """
    Invert a diagonal noise covariance.

    This does an inversion of the covariance.  The threshold is
    applied to the condition number of each block of the matrix.  Pixels
    failing the cut are set to zero.

    Args:
        npp (DistPixels): The distributed covariance.
        threshold (float): The condition number threshold to apply.
        rcond (DistPixels): (Optional) The distributed inverse condition number map to fill.
    """
    autotimer = timemory.auto_timer(timemory.FILE(use_dirname = True))
    mapnnz = int( ( (np.sqrt(8 * npp.nnz) - 1) / 2 ) + 0.5 )

    if rcond is not None:
        if rcond.size != npp.size:
            raise RuntimeError("covariance matrix and condition number map must have same number of pixels")
        if rcond.submap != npp.submap:
            raise RuntimeError("covariance matrix and condition number map must have same submap size")
        if rcond.nnz != 1:
            raise RuntimeError("condition number map should have NNZ = 1")
        do_rcond = 1

        ctoast.cov_eigendecompose_diagonal(npp.nsubmap, npp.submap, mapnnz,
            npp.data, rcond.data, threshold, 1, 1)

    else:
        temp = np.zeros(1, dtype=np.float64)
        ctoast.cov_eigendecompose_diagonal(npp.nsubmap, npp.submap, mapnnz,
            npp.data, temp, threshold, 1, 0)
    return


def covariance_multiply(npp1, npp2):
    """
    Multiply two diagonal noise covariances.

    This does an in-place multiplication of the covariance.
    The data values of the first covariance (npp1) are replaced with
    the result.

    Args:
        npp1 (3D array): The first distributed covariance.
        npp2 (3D array): The second distributed covariance.
    """
    autotimer = timemory.auto_timer(timemory.FILE(use_dirname = True))
    mapnnz = int( ( (np.sqrt(8 * npp1.nnz) - 1) / 2 ) + 0.5 )

    if npp1.size != npp2.size:
        raise RuntimeError("covariance matrices must have same number of pixels")
    if npp1.submap != npp2.submap:
        raise RuntimeError("covariance matrices must have same submap size")
    if npp1.nnz != npp2.nnz:
        raise RuntimeError("covariance matrices must have same NNZ values")

    ctoast.cov_multiply_diagonal(npp1.nsubmap, npp1.submap, mapnnz,
        npp1.data, npp2.data)
    return


def covariance_apply(npp, m):
    """
    Multiply a map by a diagonal noise covariance.

    This does an in-place multiplication of the covariance and a
    map.  The results are returned in place of the input map.

    Args:
        npp (DistPixels): The distributed covariance.
        m (DistPixels): The distributed map.
    """
    autotimer = timemory.auto_timer(timemory.FILE(use_dirname = True))
    mapnnz = int( ( (np.sqrt(8 * npp.nnz) - 1) / 2 ) + 0.5 )

    if m.size != npp.size:
        raise RuntimeError("covariance matrix and map must have same number of pixels")
    if m.submap != npp.submap:
        raise RuntimeError("covariance matrix and map must have same submap size")
    if m.nnz != mapnnz:
        raise RuntimeError("covariance matrix and map have incompatible NNZ values")

    ctoast.cov_apply_diagonal(npp.nsubmap, npp.submap, mapnnz, npp.data, m.data)
    return


def covariance_rcond(npp):
    """
    Compute the inverse condition number map.

    This computes the inverse condition number map of the supplied
    covariance matrix.

    Args:
        npp (DistPixels): The distributed covariance.

    Returns:
        rcond (DistPixels): The distributed inverse condition number map.
    """
    autotimer = timemory.auto_timer(timemory.FILE(use_dirname = True))
    mapnnz = int( ( (np.sqrt(8 * npp.nnz) - 1) / 2 ) + 0.5 )

    rcond = DistPixels(comm=npp.comm, size=npp.size, nnz=1, dtype=np.float64,
        submap=npp.submap, local=npp.local, nest=npp.nested)

    threshold = np.finfo(np.float64).eps

    ctoast.cov_eigendecompose_diagonal(npp.nsubmap, npp.submap, mapnnz,
        npp.data, rcond.data, threshold, 0, 1)

    return rcond
