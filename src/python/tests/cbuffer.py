# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from ..mpi import MPI
from .mpi import MPITestCase

from ..cbuffer import ToastBuffer

import numpy as np

import sys

import gc


class CbufferTest(MPITestCase):

    def setUp(self):
        self.nsamp = 100
        self.types = {
            'float64': np.float64,
            'float32': np.float32,
            'int64': np.int64,
            'uint64': np.uint64,
            'int32': np.int32,
            'uint32': np.uint32,
            'int16': np.int16,
            'uint16': np.uint16,
            'int8': np.int8,
            'uint8': np.uint8
        }

    def tearDown(self):
        pass


    def test_create(self):
        start = MPI.Wtime()

        data = {}

        for k, v in self.types.items():
            data[k] = ToastBuffer(size=self.nsamp, type=k)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("ToastBuffer create test took {:.3f} s".format(elapsed))


    def test_buffer(self):
        start = MPI.Wtime()

        data = {}

        for k, v in self.types.items():
            raw = ToastBuffer(size=self.nsamp, type=k)
            data[k] = np.asarray(raw)
            for i in range(self.nsamp):
                data[k][i] = i
            print(data[k])

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("ToastBuffer buffer test took {:.3f} s".format(elapsed))


    def test_refcount(self):
        start = MPI.Wtime()

        data = {}

        for k, v in self.types.items():
            raw = ToastBuffer(size=self.nsamp, type=k)
            data[k] = np.asarray(raw)
            print("buffer {} has numpy type {}".format(k, data[k].dtype))
            for i in range(self.nsamp):
                data[k][i] = i
            print("    buffer {} has refcount {}".format(k, sys.getrefcount(raw)))
            print("    buffer {} has refcount {}".format(k, sys.getrefcount(data[k])))
            gc.collect()
            print(gc.get_referrers(raw))
            del raw
        
        del data

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("ToastBuffer refcount test took {:.3f} s".format(elapsed))

