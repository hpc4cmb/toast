# Copyright (c) 2017-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

try:
    import libsharp
    available = True
except ModuleNotFoundError:
    libsharp = None
    available = False

import timemory


class LibSharpSmooth():
    """
    Operator that smooths a distributed map with a beam

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

    def __init__(self, comm=None, signal_map="signal_map",
                 lmax=None, grid=None, fwhm_deg=None, beam=None,
                 out="smoothed_signal_map"):
        autotimer = timemory.auto_timer(type(self).__name__)
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self.comm = comm
        self.signal_map = signal_map
        self.lmax = lmax
        self.out = out
        self.grid = grid

        # distribute alms
        local_m_indices = np.arange(self.comm.rank, lmax + 1, self.comm.size,
                                    dtype=np.int32)

        self.order = libsharp.packed_real_order(lmax, ms=local_m_indices)

        if (fwhm_deg is not None) and (beam is not None):
            raise Exception("OpSmooth error, specify either fwhm_deg or beam, "
                            "not both")

        if (fwhm_deg is None) and (beam is None):
            raise Exception("OpSmooth error, specify fwhm_deg or beam")

        if fwhm_deg is not None:
            self.beam = hp.gauss_beam(fwhm=np.radians(fwhm_deg), lmax=lmax,
                                      pol=True)
        else:
            self.beam = beam

    def exec(self, data):
        """
        Create the timestreams...

        This loops over all observations and detectors and uses the pointing
        matrix to ...

        Args:
            data (toast.Data): The distributed data.
        """

        if libsharp is None:
            raise RuntimeError('libsharp not available')
        autotimer = timemory.auto_timer(type(self).__name__)
        has_pol = len(data[self.signal_map]) > 1

        alm_sharp_I = libsharp.analysis(
            self.grid, self.order,
            np.ascontiguousarray(data[self.signal_map][0].reshape((1, 1, -1))),
            spin=0, comm=self.comm)
        self.order.almxfl(alm_sharp_I, np.ascontiguousarray(self.beam[:, 0:1]))
        out = libsharp.synthesis(self.grid, self.order, alm_sharp_I, spin=0,
                                 comm=self.comm)[0]

        if has_pol:
            alm_sharp_P = libsharp.analysis(
                self.grid, self.order,
                np.ascontiguousarray(
                    data[self.signal_map][1:3, :].reshape((1, 2, -1))),
                spin=2, comm=self.comm)

            self.order.almxfl(alm_sharp_P,
                              np.ascontiguousarray(self.beam[:, (1, 2)]))

            signal_map_P = libsharp.synthesis(
                self.grid, self.order, alm_sharp_P, spin=2, comm=self.comm)[0]
            out = np.vstack((
                        out,
                        signal_map_P))
        data[self.out] = out
        assert data[self.signal_map].shape == data[self.out].shape
