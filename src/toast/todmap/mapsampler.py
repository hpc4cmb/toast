# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import warnings

import numpy as np

from scipy.constants import arcmin

import healpy as hp

from ..mpi import use_mpi, MPIShared

from ..timing import function_timer

from .._libtoast import fast_scanning_float32

DTYPE = np.float32


@function_timer
def plug_holes(m, verbose=False, in_place=True, nest=False):
    """Use simple downgrading to derive estimates of the missing pixel values
    """
    nbad_start = np.sum(np.isclose(m, hp.UNSEEN))

    if nbad_start == m.size:
        if verbose:
            print("plug_holes: All map pixels are empty. Cannot plug holes", flush=True)
        return

    if nbad_start == 0:
        return

    nside = hp.get_nside(m)
    npix = m.size
    if nest:
        mnest = m.copy()
    else:
        mnest = hp.reorder(m, r2n=True)

    lowres = mnest
    nside_lowres = nside
    bad = np.isclose(mnest, hp.UNSEEN)
    while np.any(bad) and nside_lowres > 1:
        nside_lowres //= 2
        lowres = hp.ud_grade(lowres, nside_lowres, order_in="NESTED")
        hires = hp.ud_grade(lowres, nside, order_in="NESTED")
        bad = np.isclose(mnest, hp.UNSEEN)
        mnest[bad] = hires[bad]

    nbad_end = np.sum(bad)

    if nbad_end != 0:
        mn = np.mean(mnest[np.logical_not(bad)])
        mnest[bad] = mn

    if not in_place:
        m = m.copy()
    if nest:
        m[:] = mnest
    else:
        m[:] = hp.reorder(mnest, n2r=True)

    if verbose and nbad_start != 0:
        print(
            "plug_holes: Filled {} missing pixels ({:.2f}%), lowest "
            "resolution was Nside={}.".format(
                nbad_start, (100.0 * nbad_start) // npix, nside_lowres
            )
        )
    return m


class MapSampler:
    """
    MapSampler objects store maps in the node shared memory and allow
    bilinear interpolation of the maps into TOD.
    """

    @function_timer
    def __init__(
        self,
        map_path,
        pol=False,
        pol_fwhm=None,
        no_temperature=False,
        dtype=None,
        verbose=False,
        nside=None,
        comm=None,
        cache=None,
        preloaded_map=None,
        buflen=1000000,
        nest=False,
    ):
        """
        Instantiate the map sampler object, load a healpix
        map in a file located at map_path

        if pol==True, reads I,Q,U maps from extensions 0, 1, 2
        """

        if not pol and no_temperature:
            raise RuntimeError("You cannot have pol=False, " "no_temperature=True")

        self.path = map_path
        self.pol = pol
        self.pol_fwhm = pol_fwhm
        self._map = None
        self._map_Q = None
        self._map_U = None
        self.nest = nest
        if nest:
            self.order = "NESTED"
        else:
            self.order = "RING"
        self.buflen = buflen
        # Output data type, internal is always DTYPE
        if dtype is not None:
            warnings.warn("MapSampler no longer supports dtype", DeprecationWarning)

        # Use healpy to load the map into memory.

        if comm is None:
            self.comm = None
            self.rank = 0
            self.ntask = 1
        else:
            self.comm = comm
            self.rank = comm.Get_rank()
            self.ntask = comm.Get_size()

        self.shmem = self.ntask > 1
        self.pol = pol

        if self.rank == 0:
            if self.pol:
                if preloaded_map is not None:
                    if no_temperature:
                        (self._map_Q, self._map_U) = np.array(
                            preloaded_map, dtype=DTYPE
                        )
                    else:
                        (self._map, self._map_Q, self._map_U) = np.array(
                            preloaded_map, dtype=DTYPE
                        )
                else:
                    if no_temperature:
                        self._map_Q, self._map_U = hp.read_map(
                            self.path,
                            field=[1, 2],
                            dtype=DTYPE,
                            verbose=verbose,
                            memmmap=True,
                            nest=self.nest,
                        )
                    else:
                        try:
                            self._map, self._map_Q, self._map_U = hp.read_map(
                                self.path,
                                field=[0, 1, 2],
                                dtype=DTYPE,
                                verbose=verbose,
                                memmap=True,
                                nest=self.nest,
                            )
                        except IndexError:
                            print(
                                "WARNING: {} is not polarized".format(self.path),
                                flush=True,
                            )
                            self.pol = False
                            self._map = hp.read_map(
                                self.path,
                                dtype=DTYPE,
                                verbose=verbose,
                                memmap=True,
                                nest=self.nest,
                            )

                if nside is not None:
                    if not no_temperature:
                        self._map = hp.ud_grade(
                            self._map,
                            nside,
                            dtype=DTYPE,
                            order_in=self.order,
                            order_out=self.order,
                        )
                    if self.pol:
                        self._map_Q = hp.ud_grade(
                            self._map_Q,
                            nside,
                            dtype=DTYPE,
                            order_in=self.order,
                            order_out=self.order,
                        )
                        self._map_U = hp.ud_grade(
                            self._map_U,
                            nside,
                            dtype=DTYPE,
                            order_in=self.order,
                            order_out=self.order,
                        )

                if self.pol_fwhm is not None:
                    if not no_temperature:
                        plug_holes(self._map, verbose=verbose, nest=self.nest)
                    if self.pol:
                        plug_holes(self._map_Q, verbose=verbose, nest=self.nest)
                        plug_holes(self._map_U, verbose=verbose, nest=self.nest)
            else:
                if preloaded_map is not None:
                    self._map = np.array(preloaded_map, dtype=DTYPE)
                else:
                    self._map = hp.read_map(
                        map_path,
                        field=[0],
                        dtype=DTYPE,
                        verbose=verbose,
                        memmap=True,
                        nest=self.nest,
                    )
                if nside is not None:
                    self._map = hp.ud_grade(
                        self._map,
                        nside,
                        dtype=DTYPE,
                        order_in=self.order,
                        order_out=self.order,
                    )
                plug_holes(self._map, verbose=verbose, nest=self.nest)

        if self.ntask > 1:
            self.pol = comm.bcast(self.pol, root=0)
            npix = 0
            if self.rank == 0:
                if self.pol:
                    npix = len(self._map_Q)
                else:
                    npix = len(self._map)
            npix = comm.bcast(npix, root=0)
            if self.shmem:
                shared = MPIShared((npix,), np.dtype(DTYPE), comm)
                if not no_temperature:
                    shared.set(self._map, (0,), fromrank=0)
                    self._map = shared
                if self.pol:
                    shared_Q = MPIShared((npix,), np.dtype(DTYPE), comm)
                    shared_Q.set(self._map_Q, (0,), fromrank=0)
                    self._map_Q = shared_Q
                    shared_U = MPIShared((npix,), np.dtype(DTYPE), comm)
                    shared_U.set(self._map_U, (0,), fromrank=0)
                    self._map_U = shared_U
            else:
                if self.rank != 0:
                    if not no_temperature:
                        self._map = np.zeros(npix, dtype=DTYPE)
                    if self.pol:
                        self._map_Q = np.zeros(npix, dtype=DTYPE)
                        self._map_U = np.zeros(npix, dtype=DTYPE)

                if not no_temperature:
                    comm.Bcast(self._map, root=0)
                if self.pol:
                    comm.Bcast(self._map_Q, root=0)
                    comm.Bcast(self._map_U, root=0)

        if self.pol:
            self.npix = len(self._map_Q[:])
        else:
            self.npix = len(self._map[:])
        self.nside = hp.npix2nside(self.npix)

        self.cache = cache
        self.instance = 0
        if self.cache is not None and not self.shmem:
            # Increase the instance counter until we find an unused
            # instance.  If the user did not want to store duplicates,
            # they would not have created two identical mapsampler
            # objects.
            while self.cache.exists(self._cachename("I")):
                self.instance += 1
            if not no_temperature:
                self._map = self.cache.put(self._cachename("I"), self._map)
            if self.pol:
                self._map_Q = self.cache.put(self._cachename("Q"), self._map_Q)
                self._map_U = self.cache.put(self._cachename("U"), self._map_U)

        if self.pol_fwhm is not None:
            self.smooth(self.pol_fwhm, pol_only=True)
        return

    @function_timer
    def smooth(self, fwhm, lmax=None, pol_only=False):
        """Smooth the map with a Gaussian kernel.
        """
        if self.rank == 0:
            if pol_only:
                print(
                    "Smoothing the polarization to {} arcmin".format(fwhm), flush=True
                )
            else:
                print("Smoothing the map to {} arcmin".format(fwhm), flush=True)

        if lmax is None:
            lmax = min(np.int(fwhm / 60 * 512), 2 * self.nside)

        # If the map is in node-shared memory, only the root process on each
        # node does the smoothing.
        if not self.shmem or self._map.nodecomm.rank == 0:
            if self.pol:
                m = np.vstack([self._map[:], self._map_Q[:], self._map_U[:]])
            else:
                m = self._map[:]
            if self.nest:
                m = hp.reorder(m, n2r=True)
            smap = hp.smoothing(m, fwhm=fwhm * arcmin, lmax=lmax, verbose=False)
            del m
            if self.nest:
                smap = hp.reorder(smap, r2n=True)
        else:
            # Convenience dummy variable
            smap = np.zeros([3, 12])

        if not pol_only:
            if self.shmem:
                self._map.set(smap[0].astype(DTYPE), (0,), fromrank=0)
            else:
                self._map[:] = smap[0]

        if self.pol:
            if self.shmem:
                self._map_Q.set(smap[1].astype(DTYPE), (0,), fromrank=0)
                self._map_U.set(smap[2].astype(DTYPE), (0,), fromrank=0)
            else:
                self._map_Q[:] = smap[1]
                self._map_U[:] = smap[2]

        self.pol_fwhm = fwhm
        return

    def _cachename(self, stokes):
        """
        Construct a cache name string for the selected Stokes map
        """
        return "{}_ns{:04}_{}_{:04}".format(
            self.path, self.nside, stokes, self.instance
        )

    @function_timer
    def __del__(self):
        """
        Explicitly free memory taken up in the cache.
        """
        if self.cache is not None:
            if self._map is not None:
                del self._map
                if not self.shmem:
                    self.cache.destroy(self._cachename("I"))
            if self.pol:
                del self._map_Q
                del self._map_U
                if not self.shmem:
                    self.cache.destroy(self._cachename("Q"))
                    self.cache.destroy(self._cachename("U"))
        return

    @function_timer
    def __iadd__(self, other):
        """Accumulate provided Mapsampler object with this one.
        """
        if self.shmem:
            # One process does the manipulation on each node
            self._map._nodecomm.Barrier()
            if self._map._noderank == 0:
                self._map._data[:] += other._map[:]
            if self.pol and other.pol:
                if self._map_Q._noderank == (1 % self._map_Q._nodeprocs):
                    self._map_Q._data[:] += other._map_Q[:]
                if self._map_U._noderank == (2 % self._map_U._nodeprocs):
                    self._map_U._data[:] += other._map_U[:]
            self._map._nodecomm.Barrier()
        else:
            self._map += other._map
            if self.pol and other.pol:
                self._map_Q += other._map_Q
                self._map_U += other._map_U
        return self

    @function_timer
    def __isub__(self, other):
        """Subtract provided Mapsampler object from this one.
        """
        if self.shmem:
            # One process does the manipulation on each node
            self._map._nodecomm.Barrier()
            if self._map._noderank == 0:
                self._map._data[:] -= other._map[:]
            if self.pol and other.pol:
                if self._map_Q._noderank == (1 % self._map_Q._nodeprocs):
                    self._map_Q._data[:] -= other._map_Q[:]
                if self._map_U._noderank == (2 % self._map_U._nodeprocs):
                    self._map_U._data[:] -= other._map_U[:]
            self._map._nodecomm.Barrier()
        else:
            self._map -= other._map
            if self.pol and other.pol:
                self._map_Q -= other._map_Q
                self._map_U -= other._map_U
        return self

    @function_timer
    def __imul__(self, other):
        """Scale the maps in this MapSampler object
        """
        if self.shmem:
            # One process does the manipulation on each node
            self._map._nodecomm.Barrier()
            if self._map._noderank == 0:
                self._map._data[:] *= other
            if self.pol:
                if self._map_Q._noderank == (1 % self._map_Q._nodeprocs):
                    self._map_Q._data[:] *= other
                if self._map_U._noderank == (2 % self._map_U._nodeprocs):
                    self._map_U._data[:] *= other
            self._map._nodecomm.Barrier()
        else:
            self._map *= other
            if self.pol:
                self._map_Q *= other
                self._map_U *= other
        return self

    @function_timer
    def __itruediv__(self, other):
        """ Divide the maps in this MapSampler object
        """
        if self.shmem:
            self._map._nodecomm.Barrier()
            if self._map._noderank == 0:
                self._map._data[:] /= other
            if self.pol:
                if self._map_Q._noderank == (1 % self._map_Q._nodeprocs):
                    self._map_Q._data[:] /= other
                if self._map_U._noderank == (2 % self._map_U._nodeprocs):
                    self._map_U._data[:] /= other
            self._map._nodecomm.Barrier()
        else:
            self._map /= other
            if self.pol:
                self._map_Q /= other
                self._map_U /= other
        return self

    @function_timer
    def at(self, theta, phi, interp_pix=None, interp_weights=None):
        """
        Use healpy bilinear interpolation to interpolate the
        map.  User must make sure that coordinate system used
        for theta and phi matches the map coordinate system.
        """
        if self._map is None:
            raise RuntimeError("No temperature map to sample")

        n = len(theta)
        stepsize = self.buflen
        signal = np.zeros(n, dtype=np.float32)

        # DEBUG begin
        if np.any(theta < 0) or np.any(theta > np.pi):
            raise RuntimeError("bad theta")
        if np.any(phi < 0) or np.any(phi > 2 * np.pi):
            raise RuntimeError("bad phi")
        # DEBUG end

        for istart in range(0, n, stepsize):
            istop = min(istart + stepsize, n)
            ind = slice(istart, istop)
            if interp_pix is None or interp_weights is None:
                p, w = hp.get_interp_weights(
                    self.nside, theta[ind], phi[ind], nest=self.nest
                )
            else:
                p = np.ascontiguousarray(interp_pix[:, ind])
                w = np.ascontiguousarray(interp_weights[:, ind])
            buf = np.zeros(istop - istart, dtype=np.float64)
            fast_scanning_float32(buf, p, w, self._map[:])
            signal[ind] = buf
        return signal

    @function_timer
    def atpol(
        self,
        theta,
        phi,
        IQUweight,
        onlypol=False,
        interp_pix=None,
        interp_weights=None,
        pol=True,
        pol_deriv=False,
    ):
        """
        Use healpy bilinear interpolation to interpolate the
        map.  User must make sure that coordinate system used
        for theta and phi matches the map coordinate system.
        IQUweight is an array of shape (nsamp,3) returned by the
        pointing library that gives the weights of the I,Q, and U maps.

        Args:
            pol_deriv(bool):  Return the polarization angle derivative
                of the signal instead of the actual signal.

        """
        if onlypol and not self.pol:
            return None

        if not self.pol or not pol:
            return self.at(
                theta, phi, interp_pix=interp_pix, interp_weights=interp_weights
            )

        if np.shape(IQUweight)[1] != 3:
            raise RuntimeError(
                "Cannot sample polarized map with only " "intensity weights"
            )

        n = len(theta)
        stepsize = self.buflen
        signal = np.zeros(n, dtype=np.float32)

        for istart in range(0, n, stepsize):
            istop = min(istart + stepsize, n)
            ind = slice(istart, istop)

            if interp_pix is None or interp_weights is None:
                p, w = hp.get_interp_weights(
                    self.nside, theta[ind], phi[ind], nest=self.nest
                )
            else:
                p = np.ascontiguousarray(interp_pix[:, ind])
                w = np.ascontiguousarray(interp_weights[:, ind])

            weights = np.ascontiguousarray(IQUweight[ind].T)

            buf = np.zeros(istop - istart, dtype=np.float64)
            fast_scanning_float32(buf, p, w, self._map_Q[:])
            if pol_deriv:
                signal[ind] = -2 * weights[2] * buf
            else:
                signal[ind] = weights[1] * buf

            buf[:] = 0
            fast_scanning_float32(buf, p, w, self._map_U[:])
            if pol_deriv:
                signal[ind] += 2 * weights[1] * buf
            else:
                signal[ind] += weights[2] * buf

            if not onlypol:
                if self._map is None:
                    raise RuntimeError("No temperature map to sample")
                buf[:] = 0
                fast_scanning_float32(buf, p, w, self._map[:])
                signal[ind] += weights[0] * buf

        return signal
