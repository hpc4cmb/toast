# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import scipy.constants as constants
import numpy as np

from ..timing import function_timer, GlobalTimers

from ..op import Operator

from .. import qarray as qa

from .._libtoast import (
    cov_accum_diag,
    cov_accum_zmap,
    cov_accum_diag_hits,
    cov_accum_diag_invnpp,
    scan_map_float64,
    scan_map_float32,
    apply_flags_to_pixels,
)

from ..map import DistPixels


class OpAccumDiag(Operator):
    """Operator which accumulates the diagonal covariance and noise weighted map.

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
        self,
        zmap=None,
        hits=None,
        invnpp=None,
        detweights=None,
        name=None,
        flag_name=None,
        flag_mask=255,
        common_flag_name=None,
        common_flag_mask=255,
        pixels="pixels",
        weights="weights",
        apply_flags=True,
    ):

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
            self._subsize = zmap.npix_submap
            self._nnz = zmap.nnz
            if self._globloc is None:
                self._globloc = self._zmap

        if hits is not None:
            self._do_hits = True
            if hits.nnz != 1:
                raise RuntimeError("Hit map should always have NNZ == 1")
            if self._nsub is None:
                self._nsub = hits.nsubmap
                self._subsize = hits.npix_submap
            else:
                if self._nsub != hits.nsubmap:
                    raise RuntimeError(
                        "All pixel domain objects must have the same number "
                        "of local submaps."
                    )
                if self._subsize != hits.npix_submap:
                    raise RuntimeError(
                        "All pixel domain objects must have the same submap size."
                    )
            if self._globloc is None:
                self._globloc = self._hits

        if invnpp is not None:
            self._do_invn = True
            block = invnpp.nnz
            blocknnz = int(((np.sqrt(8 * block) - 1) / 2) + 0.5)
            if self._nsub is None:
                self._nsub = invnpp.nsubmap
                self._subsize = invnpp.npix_submap
                self._nnz = blocknnz
            else:
                if self._nsub != invnpp.nsubmap:
                    raise RuntimeError(
                        "All pixel domain objects must have the same submap size."
                    )
                if self._subsize != invnpp.npix_submap:
                    raise RuntimeError(
                        "All pixel domain objects must have the same submap size."
                    )
                if self._nnz is None:
                    self._nnz = blocknnz
                elif self._nnz != blocknnz:
                    raise RuntimeError(
                        "All pixel domain objects must have the same submap size."
                    )
            if self._globloc is None:
                self._globloc = self._invnpp

        if self._nnz is None:
            # this means we only have a hit map
            self._nnz = 1

        if self._do_invn and (not self._do_hits):
            raise RuntimeError(
                "When accumulating the diagonal pixel covariance, you must "
                "also accumulate the hit map"
            )

        if self._do_z and (self._do_hits != self._do_invn):
            raise RuntimeError(
                "When accumulating the noise weighted map, you must accumulate "
                "either both the hits and covariance or neither."
            )

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    @function_timer
    def exec(self, data):
        """Iterate over all observations and detectors and accumulate.

        Args:
            data (toast.Data): The distributed data.
        """
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank
        gt = GlobalTimers.get()

        for obs in data.obs:
            tod = obs["tod"]

            nsamp = tod.local_samples[1]

            commonflags = None
            if self._apply_flags:
                commonflags = tod.local_common_flags(self._common_flag_name).copy()

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
                    gt.start("OpAccumDiag.exec.apply_flags")
                    # Don't change the cached pixel numbers
                    pixels = pixels.copy()
                    detflags = tod.local_flags(det, self._flag_name)
                    apply_flags_to_pixels(
                        commonflags,
                        self._common_flag_mask,
                        detflags,
                        self._flag_mask,
                        pixels,
                    )
                    gt.stop("OpAccumDiag.exec.apply_flags")

                # local pointing

                gt.start("OpAccumDiag.exec.global_to_local")
                sm, lpix = self._globloc.global_to_local(pixels)
                gt.stop("OpAccumDiag.exec.global_to_local")

                detweight = 1.0

                if self._detweights is not None:
                    if det not in self._detweights.keys():
                        raise RuntimeError(
                            "no detector weights found for {}".format(det)
                        )
                    detweight = self._detweights[det]
                    if detweight == 0:
                        continue

                # Now call the correct accumulation operator depending
                # on which input pixel objects were given.

                if self._do_invn and self._do_z:
                    invnpp = self._invnpp.flatdata
                    if invnpp is None:
                        invnpp = np.empty(shape=0, dtype=np.float64)
                    zmap = self._zmap.flatdata
                    if zmap is None:
                        zmap = np.empty(shape=0, dtype=np.float64)
                    hits = self._hits.flatdata
                    if hits is None:
                        hits = np.empty(shape=0, dtype=np.int64)
                    cov_accum_diag(
                        self._nsub,
                        self._subsize,
                        self._nnz,
                        sm,
                        lpix,
                        weights.reshape(-1),
                        detweight,
                        signal,
                        invnpp,
                        hits,
                        zmap,
                    )

                elif self._do_invn:
                    invnpp = self._invnpp.flatdata
                    if invnpp is None:
                        invnpp = np.empty(shape=0, dtype=np.float64)
                    hits = self._hits.flatdata
                    if hits is None:
                        hits = np.empty(shape=0, dtype=np.int64)
                    cov_accum_diag_invnpp(
                        self._nsub,
                        self._subsize,
                        self._nnz,
                        sm,
                        lpix,
                        weights.reshape(-1),
                        detweight,
                        invnpp,
                        hits,
                    )

                elif self._do_z:
                    zmap = self._zmap.flatdata
                    if zmap is None:
                        zmap = np.empty(shape=0, dtype=np.float64)
                    cov_accum_zmap(
                        self._nsub,
                        self._subsize,
                        self._nnz,
                        sm,
                        lpix,
                        weights.reshape(-1),
                        detweight,
                        signal,
                        zmap,
                    )

                elif self._do_hits:
                    hits = self._hits.flatdata
                    if hits is None:
                        hits = np.empty(shape=0, dtype=np.int64)
                    cov_accum_diag_hits(
                        self._nsub, self._subsize, self._nnz, sm, lpix, hits
                    )

                # print("det {}:".format(det))
                # if self._zmap is not None:
                #     print(self._zmap.data)
                # if self._hits is not None:
                #     print(self._hits.data)
                # if self._invnpp is not None:
                #     print(self._invnpp.data)

        return


class OpScanScale(Operator):
    """Operator which scales sky signal by weights from a map.

    Local pixels should already exist.

    Args:
        distmap (DistPixels): the distributed map domain data.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        name (str): scale data in cache with name <name>_<detector>.

    """

    def __init__(self, distmap=None, pixels="pixels", name=None):
        # Call the parent class constructor
        super().__init__()
        self.map = distmap
        self.pixels = pixels
        self.name = name

    def exec(self, data):
        """Scale signal by scanning weights from the map.

        This loops over all observations and detectors and uses the pointing
        matrix to project the distributed map into a timestream.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            None

        """
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                # get the pixels and weights from the cache
                pixelsname = "{}_{}".format(self.pixels, det)
                pixels = tod.cache.reference(pixelsname)
                sm, lpix = self.map.global_to_local(pixels)
                ref = tod.local_signal(det, self.name)
                weighted = np.zeros(pixels.size)

                maptype = np.dtype(self.map.dtype)
                if maptype.char == "d":
                    scan_map = scan_map_float64
                elif maptype.char == "f":
                    scan_map = scan_map_float32
                else:
                    raise RuntimeError(
                        "Scanning weights from a map only supports float32 and float64 maps"
                    )
                # We pass the signal to be scaled in place of the pointing weights
                # The returned TOD is already TOD x weigths
                scan_map(
                    self.map.npix_submap,
                    1,
                    sm,
                    lpix,
                    self.map.flatdata,
                    ref.astype(np.float64),
                    weighted,
                )
                ref[:] = weighted
                del ref
                del pixels
        return


class OpScanMask(Operator):
    """Operator which scans a mask and sets the TOD flags accordingly

    Local pixels should already exist.

    Args:
        distmap (DistPixels): the distributed mask.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        name (str): scale data in cache with name <name>_<detector>.
    """

    def __init__(self, distmap=None, pixels="pixels", flags=None, flagmask=1):
        # Call the parent class constructor
        super().__init__()
        self.map = distmap
        self.pixels = pixels
        self.flags = flags
        self.flagmask = flagmask

    def exec(self, data):
        """Scan mask values from a map and update the detector flags.
        Positive (non-zero) mask values indicate usable sky pixels.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            None
        """
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                # get the pixels and weights from the cache
                pixelsname = "{}_{}".format(self.pixels, det)
                pixels = tod.cache.reference(pixelsname)
                nsamp = pixels.size
                sm, lpix = self.map.global_to_local(pixels)

                maptype = np.dtype(self.map.dtype)
                if maptype.char == "d":
                    scan_map = scan_map_float64
                elif maptype.char == "f":
                    scan_map = scan_map_float32
                else:
                    raise RuntimeError(
                        "Scanning weights from a map only supports float32 and float64 maps"
                    )
                # We pass the signal to be scaled in place of the pointing weights
                # The returned TOD is already TOD x weigths
                weights = np.ones(nsamp, dtype=np.float64)
                masktod = np.zeros(nsamp, dtype=np.float64)
                scan_map(
                    self.map.npix_submap,
                    1,
                    sm,
                    lpix,
                    self.map.flatdata,
                    weights,
                    masktod,
                )
                flags = tod.local_flags(det, self.flags)
                flags[masktod < 0.5] |= self.flagmask
        return


def array_dot(u, v):
    """Dot product of each row of two 2D arrays"""
    return np.sum(u * v, axis=1).reshape((-1, 1))


@function_timer
def dipole(pntg, vel=None, solar=None, cmb=2.72548, freq=0):
    """Compute a dipole timestream.

    This uses detector pointing, telescope velocity and the solar system
    motion to compute the observed dipole.  It is assumed that the detector
    pointing, the telescope velocity vectors, and the solar system velocity
    vector are all in the same coordinate system.

    Args:
        pntg (array): the 2D array of quaternions of detector pointing.
        vel (array): 2D array of velocity vectors relative to the solar
            system barycenter.  if None, return only the solar system dipole.
            Units are km/s
        solar (array): a 3 element vector containing the solar system velocity
            vector relative to the CMB rest frame.  Units are km/s.
        cmb (float): CMB monopole in Kelvin.  Default value from Fixsen
            2009 (see arXiv:0911.1955)
        freq (float): optional observing frequency in Hz (NOT GHz).

    Returns:
        (array):  detector dipole timestream.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    nsamp = pntg.shape[0]

    inv_light = 1.0e3 / constants.speed_of_light

    if (vel is not None) and (solar is not None):
        # relativistic addition of velocities

        solar_speed = np.sqrt(np.sum(solar * solar, axis=0))

        vpar = (array_dot(vel, solar) / solar_speed ** 2) * solar
        vperp = vel - vpar

        vdot = 1.0 / (1.0 + array_dot(solar, vel) * inv_light ** 2)
        invgamma = np.sqrt(1.0 - (solar_speed * inv_light) ** 2)

        vpar += solar
        vperp *= invgamma

        v = vdot * (vpar + vperp)
    elif solar is not None:
        v = np.tile(solar, nsamp).reshape((-1, 3))
    elif vel is not None:
        v = vel.copy()

    speed = np.sqrt(array_dot(v, v))
    v /= speed

    beta = inv_light * speed.flatten()

    direct = qa.rotate(pntg, zaxis)

    dipoletod = None
    if freq == 0:
        inv_gamma = np.sqrt(1.0 - beta ** 2)
        num = 1.0 - beta * np.sum(v * direct, axis=1)
        dipoletod = cmb * (inv_gamma / num - 1.0)
    else:
        # Use frequency for quadrupole correction
        fx = constants.h * freq / (constants.k * cmb)
        fcor = (fx / 2) * (np.exp(fx) + 1) / (np.exp(fx) - 1)
        bt = beta * np.sum(v * direct, axis=1)
        dipoletod = cmb * (bt + fcor * bt ** 2)

    return dipoletod
