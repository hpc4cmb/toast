# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np
import healpy as hp
import libsharp

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_map import *
from ..map.pixels import *


def expand_pix(startpix, ringpix, local_npix):
    """Turn first pixel index and number of pixel in full array of pixels

    to be optimized with cython or numba
    """
    local_pix = np.empty(local_npix, dtype=np.int64)
    i = 0
    for start, num in zip(startpix, ringpix):
        local_pix[i:i+num] = np.arange(start, start+num)
        i += num
    return local_pix

def distribute_rings(nside, rank, n_mpi_processes):
    """Create a libsharp map distribution based on rings

    Build a libsharp grid object to distribute a HEALPix map
    balancing North and South distribution of rings to achieve
    the best performance on Harmonic Transforms

    Returns the grid object and the pixel indices array in RING ordering

    Parameters
    ---------

    nside : int
        HEALPix NSIDE parameter of the distributed map
    rank, n_mpi_processes, ints
        rank of the current MPI process and total number of processes

    Returns
    -------
    grid : libsharp.healpix_grid
        libsharp object that includes metadata about HEALPix distributed rings
    local_pix : np.ndarray
        integer array of local pixel indices in the current MPI process in RING
        ordering
    """
    nrings = 4 * nside - 1  # four missing pixels

    # ring indices are 1-based
    ring_indices_emisphere = np.arange(2*nside, dtype=np.int32) + 1

    local_ring_indices = ring_indices_emisphere[rank::n_mpi_processes]

    # to improve performance, simmetric rings north/south need to be in the same rank
    # therefore we use symmetry to create the full ring indexing

    if local_ring_indices[-1] == 2 * nside:
        # has equator ring
        local_ring_indices = np.concatenate(
          [local_ring_indices[:-1],
           nrings - local_ring_indices[::-1] + 1]
        )
    else:
        # does not have equator ring
        local_ring_indices = np.concatenate(
          [local_ring_indices,
           nrings - local_ring_indices[::-1] + 1]
        )

    grid = libsharp.healpix_grid(nside, rings=local_ring_indices)

    # returns start index of the ring and number of pixels
    startpix, ringpix, _, _, _ = hp.ringinfo(nside, local_ring_indices.astype(np.int64))

    local_npix = grid.local_size()
    local_pix = expand_pix(startpix, ringpix, local_npix)
    return grid, local_pix

class OpSmoothTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

        # Note: self.comm is set by the test infrastructure
        self.worldsize = self.comm.size
        if (self.worldsize >= 2):
            self.groupsize = int( self.worldsize / 2 )
            self.ngroup = 2
        else:
            self.groupsize = 1
            self.ngroup = 1
        self.toastcomm = Comm(world=self.comm, groupsize=self.groupsize)
        self.data = {} #FIXME once Data supports map, use Data: Data(self.toastcomm)

        # create the same noise input map on all processes
        self.nside = 32
        np.random.seed(100)
        npix = hp.nside2npix(self.nside)
        self.input_map = np.random.normal(size=(3, npix))
        self.fwhm_deg = 10
        self.lmax = self.nside

        # distribute longitudinal rings
        # this will be performed by a dedicated operator before
        # calling the OpSmooth operator

        self.grid, local_pix = distribute_rings(self.nside, self.comm.rank, self.worldsize)
        self.data["signal_map"] = self.input_map[:, local_pix]

    def tearDown(self):
        del self.data


    def test_smooth(self):
        start = MPI.Wtime()


        # construct the PySM operator.  Pass in information needed by PySM...
        op = OpSmooth(comm=self.comm, signal_map="signal_map",
                lmax=self.lmax, grid=self.grid, fwhm_deg=self.fwhm_deg, beam=None)
        op.exec(self.data)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("test_opsmooth took {:.3f} s".format(elapsed))

        output_map = np.zeros(self.input_map.shape, dtype=np.float64) if self.comm.rank == 0 else None
        self.comm.Reduce(self.data["smoothed_signal_map"], output_map, root=0, op=MPI.SUM)

        if self.comm.rank == 0:
            hp_smoothed = hp.smoothing(self.input_map, fwhm=np.radians(self.fwhm_deg), lmax=self.lmax)
            np.testing.assert_array_almost_equal(hp_smoothed, output_map, decimal=2)
            print("Std of difference between libsharp and healpy", (hp_smoothed-output_map).std())
