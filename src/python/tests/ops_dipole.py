# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os

import numpy as np
import numpy.testing as nt

import healpy as hp

from ..tod.tod import *
from ..tod.pointing import *
from ..tod.noise import *
from ..tod.sim_noise import *
from ..tod.sim_det_noise import *
from ..tod.sim_det_dipole import *
from ..tod.sim_tod import *
from ..tod.tod_math import *
from ..map import *


class OpSimDipoleTest(MPITestCase):

    def setUp(self):
        self.outdir = "toast_test_output"
        if self.comm.rank == 0:
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        self.mapdir = os.path.join(self.outdir, "dipole")
        if self.comm.rank == 0:
            if not os.path.isdir(self.mapdir):
                os.mkdir(self.mapdir)

        # Note: self.comm is set by the test infrastructure

        self.toastcomm = Comm(world=self.comm)
        self.data = Data(self.toastcomm)

        self.detnames = ['bore']
        self.dets = {
            'bore' : np.array([0.0, 0.0, 1.0, 0.0])
            }

        self.nside = 64
        self.npix = 12 * self.nside**2

        self.subnside = 16
        if self.subnside > self.nside:
            self.subnside = self.nside
        self.subnpix = 12 * self.subnside * self.subnside

        self.totsamp = self.npix

        self.rate = 1.0

        self.tod = TODHpixSpiral(
            self.toastcomm.comm_group, 
            self.dets, 
            self.totsamp, 
            firsttime=0.0, 
            rate=self.rate, 
            nside=self.nside)

        ob = {}
        ob['name'] = 'test'
        ob['id'] = 0
        ob['tod'] = self.tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = None

        self.data.obs.append(ob)

        self.solar_speed = 369.0
        gal_theta = np.deg2rad(90.0 - 48.05)
        gal_phi = np.deg2rad(264.31)
        z = self.solar_speed * np.cos(gal_theta)
        x = self.solar_speed * np.sin(gal_theta) * np.cos(gal_phi)
        y = self.solar_speed * np.sin(gal_theta) * np.sin(gal_phi)
        self.solar_vel = np.array([x, y, z])
        self.solar_quat = qa.from_vectors(np.array([0.0, 0.0, 1.0]), self.solar_vel)

        self.dip_check = 0.00335673

        self.dip_max_pix = hp.ang2pix(self.nside, gal_theta, gal_phi, nest=False)
        self.dip_min_pix = hp.ang2pix(self.nside, (np.pi - gal_theta), (np.pi + gal_phi), nest=False)

    def tearDown(self):
        del self.data


    def test_dipole_func(self):
        start = MPI.Wtime()

        # Verify that we get the right magnitude if we are pointed at the velocity
        # maximum.

        dtod = dipole(self.solar_quat.reshape((1, 4)), solar=self.solar_vel)
        nt.assert_almost_equal(dtod, self.dip_check * np.ones_like(dtod))

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("dipole function test took {:.3f} s".format(elapsed))


    def test_dipole_func_total(self):
        start = MPI.Wtime()

        # Verify that we get the right magnitude if we are pointed at the velocity
        # maximum.

        quat = np.array(
        [[ 0.5213338 , 0.47771442,-0.5213338 , 0.47771442],
         [ 0.52143458, 0.47770023,-0.52123302, 0.4777286 ],
         [ 0.52153535, 0.47768602,-0.52113222, 0.47774277],
         [ 0.52163611, 0.4776718 ,-0.52103142, 0.47775692],
         [ 0.52173686, 0.47765757,-0.52093061, 0.47777106]])

        v_sat = np.array(
        [[  1.82378638e-15,  2.97846918e+01,  0.00000000e+00],
         [ -1.48252084e-07,  2.97846918e+01,  0.00000000e+00],
         [ -2.96504176e-07,  2.97846918e+01,  0.00000000e+00],
         [ -4.44756262e-07,  2.97846918e+01,  0.00000000e+00],
         [ -5.93008348e-07,  2.97846918e+01,  0.00000000e+00]])

        v_sol = np.array([ -25.7213418,  -244.31203375,  275.33805175])

        dtod = dipole(quat, vel=v_sat, solar=v_sol, cmb=2.725)
        # computed with github.com/zonca/dipole
        expected = np.array([0.00196249,  0.00196203,  0.00196157,  0.00196111,  0.00196065])
        nt.assert_allclose(dtod, expected, rtol=1e-5)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("dipole function test took {:.3f} s".format(elapsed))



    def test_sim(self):
        start = MPI.Wtime()

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.nside, nest=False, mode='I')
        pointing.exec(self.data)

        # generate timestreams
        op = OpSimDipole(mode='solar', coord='G')
        op.exec(self.data)

        # make a binned map

        # get locally hit pixels
        lc = OpLocalPixels()
        localpix = lc.exec(self.data)

        # find the locally hit submaps.
        localsm = np.unique(np.floor_divide(localpix, self.subnpix))

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        invnpp = DistPixels(comm=self.toastcomm.comm_group, size=self.npix, nnz=1, dtype=np.float64, submap=self.subnpix, local=localsm, nest=False)
        invnpp.data.fill(0.0)

        zmap = DistPixels(comm=self.toastcomm.comm_group, size=self.npix, nnz=1, dtype=np.float64, submap=self.subnpix, local=localsm, nest=False)
        zmap.data.fill(0.0)

        hits = DistPixels(comm=self.toastcomm.comm_group, size=self.npix, nnz=1, dtype=np.int64, submap=self.subnpix, local=localsm, nest=False)
        hits.data.fill(0)

        # accumulate the inverse covariance and noise weighted map.  
        # Use detector weights based on the analytic NET.

        tod = self.data.obs[0]['tod']

        detweights = {}
        for d in tod.local_dets:
            detweights[d] = 1.0

        build_invnpp = OpAccumDiag(detweights=detweights, invnpp=invnpp, hits=hits, zmap=zmap, name="dipole")
        build_invnpp.exec(self.data)

        invnpp.allreduce()
        hits.allreduce()
        zmap.allreduce()

        hits.write_healpix_fits(os.path.join(self.mapdir, "hits.fits"))
        invnpp.write_healpix_fits(os.path.join(self.mapdir, "invnpp.fits"))
        zmap.write_healpix_fits(os.path.join(self.mapdir, "zmap.fits"))

        # invert it
        covariance_invert(invnpp, 1.0e-3)

        invnpp.write_healpix_fits(os.path.join(self.mapdir, "npp.fits"))

        # compute the binned map, N_pp x Z

        covariance_apply(invnpp, zmap)
        zmap.write_healpix_fits(os.path.join(self.mapdir, "binned.fits"))

        if self.comm.rank == 0:
            import matplotlib.pyplot as plt

            mapfile = os.path.join(self.mapdir, 'hits.fits')
            data = hp.read_map(mapfile, nest=False)
            nt.assert_almost_equal(data, np.ones_like(data))

            outfile = "{}.png".format(mapfile)
            hp.mollview(data, xsize=1600, nest=False)
            plt.savefig(outfile)
            plt.close()

            mapfile = os.path.join(self.mapdir, 'binned.fits')
            data = hp.read_map(mapfile, nest=False)

            # verify that the extrema are in the correct location
            # and have the correct value.
            
            minmap = np.min(data)
            maxmap = np.max(data)
            nt.assert_almost_equal(maxmap, self.dip_check, decimal=5)
            nt.assert_almost_equal(minmap, -self.dip_check, decimal=5)

            minloc = np.argmin(data)
            maxloc = np.argmax(data)
            nt.assert_equal(minloc, self.dip_min_pix)
            nt.assert_equal(maxloc, self.dip_max_pix)

            outfile = "{}.png".format(mapfile)
            hp.mollview(data, xsize=1600, nest=False)
            plt.savefig(outfile)
            plt.close()

        #self.assertTrue(False)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("sim dipole test took {:.3f} s".format(elapsed))

