# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

from ..dist import *

import numpy as np

import sys
import os


class DataTest(MPITestCase):

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
        self.toastcomm = Comm(self.comm, groupsize=self.groupsize)
        self.ntask = 24
        self.sizes1 = [
            29218,
            430879,
            43684,
            430338,
            36289,
            437553,
            37461,
            436200,
            41249,
            432593,
            42467,
            431195,
            35387,
            438274,
            36740,
            436741,
            40663,
            432999,
            42015,
            431285,
            35297,
            438004,
            37010,
            436291,
            41114,
            432186,
            42828,
            430293,
            36243,
            436697,
            38318,
            434802,
            42602,
            430338,
            44676,
            428264,
            38273,
            434306,
            40708,
            432051,
            45308,
            427452,
            36695,
            435884,
            41520,
            430879,
            44090,
            428309,
            38273,
            434126,
            40843,
            431375
        ]
        self.totsamp1 = np.sum(self.sizes1)
        
        self.sizes2 = [ (int(3600*169.7)) for i in range(8640) ]
        self.totsamp2 = np.sum(self.sizes2)


    def test_construction(self):
        start = MPI.Wtime()

        dist_uni1 = distribute_uniform(self.totsamp1, self.ntask)
        # with open("test_uni_{}".format(self.comm.rank), 'w') as f:
        #     for d in dist_uni:
        #         f.write("uniform:  {} {}\n".format(d[0], d[1]))
        n1 = np.sum(np.array(dist_uni1)[:,1])
        assert(n1 == self.totsamp1)

        n = self.totsamp1
        breaks = [n//2+1000, n//4-1000000, n//2+1000, (3*n)//4]
        dist_uni2 = distribute_uniform(self.totsamp1, self.ntask, 
            breaks=breaks)

        n2 = np.sum(np.array(dist_uni2)[:,1])
        assert(n2 == self.totsamp1)

        for offset, nsamp in dist_uni2:
            for brk in breaks:
                if brk > offset and brk < offset+nsamp:
                    raise Exception(
                    'Uniform data distribution did not honor the breaks')

        dist_disc1 = distribute_discrete(self.sizes1, self.ntask)
        # with open("test_disc_{}".format(self.comm.rank), 'w') as f:
        #     for d in dist_disc:
        #         f.write("discrete:  {} {}\n".format(d[0], d[1]))

        n = np.sum(np.array(dist_disc1)[:,1])
        assert(n == len(self.sizes1))

        n = len(self.sizes1)
        breaks = [n//2, n//4, n//2, (3*n)//4]
        dist_disc2 = distribute_discrete(self.sizes1, self.ntask, 
            breaks=breaks)

        n = np.sum(np.array(dist_disc2)[:,1])
        assert(n == len(self.sizes1))

        for offset, nchunk in dist_disc2:
            for brk in breaks:
                if brk > offset and brk < offset+nchunk:
                    raise Exception(
                    'Discrete data distribution did not honor the breaks')

        self.assertEqual(self.toastcomm.ngroups, self.ngroup)
        self.assertEqual(self.toastcomm.group_size, self.groupsize)
        
        self.data = Data(self.toastcomm)

        handle = None
        if self.comm.rank == 0:
            handle = open(os.path.join(self.outdir,"out_test_construct_info"), "w")
        self.data.info(handle)
        if self.comm.rank == 0:
            handle.close()

        dist_disc3 = distribute_discrete(self.sizes2, 384)

        if self.comm.rank == 0:
            with open(os.path.join(self.outdir,"dist_discrete_8640x384.txt"), "w") as f:
                indx = 0
                for d in dist_disc3:
                    f.write("{:04d} = ({}, {})\n".format(indx, d[0], d[1]))
                    indx += 1

        stop = MPI.Wtime()
        elapsed = stop - start
        #print('Proc {}:  test took {:.4f} s'.format( MPI.COMM_WORLD.rank, elapsed ))

