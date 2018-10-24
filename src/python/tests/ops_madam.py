# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

import sys
import os
import shutil

import healpy as hp
from ..tod.tod import *
from ..tod.pointing import *
from ..tod.sim_tod import *
from ..tod.sim_det_noise import *
from ..tod.sim_det_map import *
from ..map.madam import *

from ._helpers import (create_outdir, create_distdata, boresight_focalplane,
    uniform_chunks)


class OpMadamTest(MPITestCase):

    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

        # One observation per group
        self.data = create_distdata(self.comm, obs_per_group=1)

        self.ndet = self.data.comm.group_size
        self.rate = 50.0

        # Create detectors with defaults
        dnames, dquat, depsilon, drate, dnet, dfmin, dfknee, dalpha = \
            boresight_focalplane(self.ndet, samplerate=self.rate)

        # Samples per observation
        self.totsamp = 3 * 49152

        # Pixelization
        self.sim_nside = 64
        self.map_nside = 64

        # Populate the observations

        tod = TODHpixSpiral(
            self.data.comm.comm_group,
            dquat,
            self.totsamp,
            detranks=self.data.comm.group_size,
            rate=self.rate,
            nside=self.sim_nside
        )

        self.data.obs[0]["tod"] = tod


    def test_madam_gradient(self):
        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        # Write outputs to a test-specific directory
        mapdir = os.path.join(self.outdir, "grad")
        if self.comm.rank == 0:
            if os.path.isdir(mapdir):
                shutil.rmtree(mapdir)
            os.makedirs(mapdir)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(mapdir,"out_test_madam_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        pars = {}
        pars[ "kfirst" ] = "F"
        pars[ "base_first" ] = 1.0
        pars[ "fsample" ] = self.rate
        pars[ "nside_map" ] = self.map_nside
        pars[ "nside_cross" ] = self.map_nside
        pars[ "nside_submap" ] = min(8, self.map_nside)
        pars[ "write_map" ] = "F"
        pars[ "write_binmap" ] = "T"
        pars[ "write_matrix" ] = "F"
        pars[ "write_wcov" ] = "F"
        pars[ "write_hits" ] = "T"
        pars[ "kfilter" ] = "F"
        pars[ "path_output" ] = mapdir
        pars[ "info" ] = 0

        madam = OpMadam(params=pars, name="grad")
        if madam.available:
            # Run Madam twice on the same data and ensure the result
            # does not change
            madam.exec(self.data)

            m0 = None
            if self.comm.rank == 0:
                m0 = hp.read_map(os.path.join(mapdir,"madam_bmap.fits"))

            madam.exec(self.data)

            m1 = None
            failed = False
            if self.comm.rank == 0:
                m1 = hp.read_map(os.path.join(mapdir, "madam_bmap_001.fits"))
                if not np.allclose(m0, m1):
                    print(\
                        "Madam did not produce the same map from the "
                        "same data.")
                    failed = True
            failed = self.comm.bcast(failed, root=0)
            self.assertFalse(failed)
        else:
            print("libmadam not available, skipping tests")


    def test_madam_output(self):
        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside, nest=True)
        grad.exec(self.data)

        # make a simple pointing matrix
        pointing = OpPointingHpix(nside=self.map_nside, nest=True)
        pointing.exec(self.data)

        # Write outputs to a test-specific directory
        mapdir = os.path.join(self.outdir, "out")
        if self.comm.rank == 0:
            if os.path.isdir(mapdir):
                shutil.rmtree(mapdir)
            os.makedirs(mapdir)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(mapdir,"out_test_madam_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        pars = {}
        pars[ "kfirst" ] = "T"
        pars[ "iter_max" ] = 100
        pars[ "base_first" ] = 5.0
        pars[ "fsample" ] = self.rate
        pars[ "nside_map" ] = self.map_nside
        pars[ "nside_cross" ] = self.map_nside
        pars[ "nside_submap" ] = min(8, self.map_nside)
        pars[ "write_map" ] = "F"
        pars[ "write_binmap" ] = "T"
        pars[ "write_matrix" ] = "F"
        pars[ "write_wcov" ] = "F"
        pars[ "write_hits" ] = "T"
        pars[ "kfilter" ] = "F"
        pars[ "path_output" ] = mapdir
        pars[ "info" ] = 0

        madam = OpMadam(params=pars, name="grad", name_out="destriped")

        if madam.available:
            tod = self.data.obs[0]["tod"]

            rms1 = None
            for det in tod.local_dets:
                ref_in = tod.cache.reference("grad_"+det)
                rms0 = np.std(ref_in)
                ref_in[ref_in.size//2:] += 1e6 # Add an offset
                rms1 = np.std(ref_in)
                del ref_in

            madam.exec(self.data)

            for det in tod.local_dets:
                ref_out = tod.cache.reference("destriped_"+det)
                rms2 = np.std(ref_out)
                del ref_out
                if rms1 < 0.9*rms2:
                    raise Exception("Destriped TOD does not have lower RMS")
        else:
            print("libmadam not available, skipping tests")
