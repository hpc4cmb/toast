# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
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
        data = {}
        for k, v in self.types.items():
            data[k] = ToastBuffer(self.nsamp, k)
        del data
        return


    def test_buffer(self):
        data = {}
        for k, v in self.types.items():
            raw = ToastBuffer(self.nsamp, k)
            data[k] = np.asarray(raw)
            for i in range(self.nsamp):
                data[k][i] = i
            #print(data[k])
        del data
        return


    def test_refcount(self):
        data = {}
        for k, v in self.types.items():
            raw = ToastBuffer(self.nsamp, k)
            gc.collect()
            #print("    raw buffer {} has refcount {}".format(k, sys.getrefcount(raw)))
            #print(gc.get_referrers(raw))

            data[k] = np.asarray(raw)
            #print("buffer {} has numpy type {}".format(k, data[k].dtype))
            for i in range(self.nsamp):
                data[k][i] = i
            gc.collect()
            #print("    now raw buffer {} has refcount {}".format(k, sys.getrefcount(raw)))
            #print(gc.get_referrers(raw))

            del raw

        for k, v in self.types.items():
            gc.collect()
            self.assertTrue(sys.getrefcount(data[k]) <= 2)
            #print("    dict buffer {} has refcount {}".format(k, sys.getrefcount(data[k])))
            #print(gc.get_referrers(data[k]))

        del data
        return
