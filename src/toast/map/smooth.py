# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import use_mpi

import numpy as np

import healpy as hp

from ..timing import function_timer

libsharp = None
available = False
if use_mpi:
    try:
        import libsharp

        available = True
    except ImportError:
        libsharp = None
        available = False


class LibSharpSmooth(object):
    """Class that smooths a distributed map with a beam

    Apply a beam window function to a distributed map, it transforms
    an I or IQU distributed map to a_lm with libsharp, multiplies by
    beam(ell) and transforms back to pixel space.

    The beam can be specified either as a Full Width at Half Maximum
    of a gaussian beam in degrees or with a custom beam factor as a
    function of ell to be multiplied to the a_lm

    Parameters
    ----------
        comm: mpi4py communicator
            MPI communicator object
        signal_map : str
            the name of the cache object (<signal_map>_<detector>)
            containing the input map.
        lmax : int
            maximum ell value of the spherical harmonics transform
        grid : libsharp healpix_grid
            libsharp healpix grid object
        fwhm_deg : float
            Full Width Half Max in degrees of the gaussian beam
        beam : 2D np.ndarray
            1 column (I only) or 3 columns (different beam for
            polarization) beam as a function of ell
    """

    @function_timer
    def __init__(self, comm=None, lmax=None, grid=None, fwhm_deg=None, beam=None):
        self.comm = comm
        self.lmax = lmax
        self.grid = grid

        # distribute alms
        local_m_indices = np.arange(
            self.comm.rank, lmax + 1, self.comm.size, dtype=np.int32
        )

        self.order = libsharp.packed_real_order(lmax, ms=local_m_indices)

        if (fwhm_deg is not None) and (beam is not None):
            raise Exception(
                "OpSmooth error, specify either fwhm_deg or beam, " "not both"
            )

        if (fwhm_deg is None) and (beam is None):
            raise Exception("OpSmooth error, specify fwhm_deg or beam")

        if fwhm_deg is not None:
            self.beam = hp.gauss_beam(fwhm=np.radians(fwhm_deg), lmax=lmax, pol=True)
        else:
            self.beam = beam

    @function_timer
    def exec(self, data):
        """Process a set of maps.

        This takes a set of maps (I or IQU), convolves them with the beam, and
        returns the result.

        Args:
            data (list): The maps.

        Returns:
            (array):  The beam convolved maps.

        """

        if libsharp is None:
            raise RuntimeError("libsharp not available")
        has_pol = len(data) > 1

        alm_sharp_I = libsharp.analysis(
            self.grid,
            self.order,
            np.ascontiguousarray(data[0].reshape((1, 1, -1))),
            spin=0,
            comm=self.comm,
        )
        self.order.almxfl(alm_sharp_I, np.ascontiguousarray(self.beam[:, 0:1]))
        out = libsharp.synthesis(
            self.grid, self.order, alm_sharp_I, spin=0, comm=self.comm
        )[0]

        if has_pol:
            alm_sharp_P = libsharp.analysis(
                self.grid,
                self.order,
                np.ascontiguousarray(data[1:3, :].reshape((1, 2, -1))),
                spin=2,
                comm=self.comm,
            )

            self.order.almxfl(alm_sharp_P, np.ascontiguousarray(self.beam[:, (1, 2)]))

            signal_map_P = libsharp.synthesis(
                self.grid, self.order, alm_sharp_P, spin=2, comm=self.comm
            )[0]
            out = np.vstack((out, signal_map_P))
        return out
