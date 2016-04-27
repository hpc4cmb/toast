# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

if 'PYTOAST_NOMPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

from toast.mpirunner import MPITestCase

from toast.tod.cache import *


class CacheTest(MPITestCase):

    def setUp(self):
        self.nsamp = 1000000
        self.cache = Cache()
        self.pycache = Cache(pymem=True)
        self.types = {
            'f64': np.float64,
            'f32': np.float32,
            'i64': np.int64,
            'u64': np.uint64,
            'i32': np.int32,
            'u32': np.uint32,
            'i16': np.int16,
            'u16': np.uint16,
            'i8': np.int8,
            'u8': np.uint8
        }

    def tearDown(self):
        del self.cache
        del self.pycache


    def test_create(self):
        start = MPI.Wtime()

        for k, v in self.types.items():
            self.cache.create('test-{}'.format(k), v, (self.nsamp,4))

        for k, v in self.types.items():
            data = self.cache.reference('test-{}'.format(k))
            data[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1,4)

        for k, v in self.types.items():
            data = self.cache.reference('test-{}'.format(k))
            print(data)

        for k, v in self.types.items():
            self.cache.destroy('test-{}'.format(k))

        for k, v in self.types.items():
            self.pycache.create('test-{}'.format(k), v, (self.nsamp,4))

        for k, v in self.types.items():
            data = self.pycache.reference('test-{}'.format(k))
            data[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1,4)

        for k, v in self.types.items():
            data = self.pycache.reference('test-{}'.format(k))
            print(data)

        for k, v in self.types.items():
            self.pycache.destroy('test-{}'.format(k))

        #self.assertTrue(False)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache create test took {:.3f} s".format(elapsed))

