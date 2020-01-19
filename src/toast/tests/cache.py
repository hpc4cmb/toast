# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys

import numpy as np

import ctypes

from .mpi import MPITestCase

from ..cache import Cache

from ..utils import AlignedF64, memreport


class CacheTest(MPITestCase):
    def setUp(self):
        self.nsamp = 1000
        self.membytes = 100000000
        self.memnbuf = 100
        self.cache = Cache(pymem=False)
        self.pycache = Cache(pymem=True)
        self.types = {
            "f64": np.float64,
            "f32": np.float32,
            "i64": np.int64,
            "u64": np.uint64,
            "i32": np.int32,
            "u32": np.uint32,
            "i16": np.int16,
            "u16": np.uint16,
            "i8": np.int8,
            "u8": np.uint8,
        }

    def tearDown(self):
        del self.cache
        del self.pycache

    def test_refcount(self):
        buf = AlignedF64.zeros(self.nsamp)
        # print("REFCNT: ", sys.getrefcount(buf), flush=True)
        # wbuf1 = buf.weak()
        # print("REFCNT 1: ", sys.getrefcount(buf), flush=True)
        # wbuf2 = buf.weak()
        # print("REFCNT 2: ", sys.getrefcount(buf), flush=True)

        return

    def test_create(self):
        for k, v in self.types.items():
            ref = self.cache.create("test-{}".format(k), v, (self.nsamp, 4))
            del ref

        for k, v in self.types.items():
            data = self.cache.reference("test-{}".format(k))
            data[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1, 4)
            del data

        for k, v in self.types.items():
            ex = self.cache.exists("test-{}".format(k))
            self.assertTrue(ex)

        for k, v in self.types.items():
            self.cache.destroy("test-{}".format(k))

        self.cache.clear()

        for k, v in self.types.items():
            ref = self.pycache.create("test-{}".format(k), v, (self.nsamp, 4))
            del ref

        for k, v in self.types.items():
            data = self.pycache.reference("test-{}".format(k))
            data[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1, 4)
            del data

        for k, v in self.types.items():
            ex = self.pycache.exists("test-{}".format(k))
            self.assertTrue(ex)

        for k, v in self.types.items():
            self.pycache.destroy("test-{}".format(k))

        self.pycache.clear()
        return

    def test_put(self):
        # Populate some buffers
        for k, v in self.types.items():
            pname = "test-{}".format(k)
            pdata = np.ones((self.nsamp, 4), v)
            ref = self.cache.put(pname, pdata)
            del ref

        # Test putting the same buffer- it should just return a new array
        # wrapping the same underlying memory.
        for k, v in self.types.items():
            pname = "test-{}".format(k)
            pdata = np.ones((self.nsamp, 4), v)
            ref = self.cache.reference(pname)
            copyref = self.cache.put(pname, ref)
            ref_pnt = ref.ctypes.data_as(ctypes.c_void_p).value
            copyref_pnt = copyref.ctypes.data_as(ctypes.c_void_p).value
            self.assertTrue(ref_pnt == copyref_pnt)
            del ref
            del copyref

        # Test replacement of buffers with the same name
        for k, v in self.types.items():
            pname = "test-{}".format(k)
            pdata = np.ones((self.nsamp, 8), v)
            try:
                # This should raise- same name as existing buffer, but
                # replace == False
                ref = self.cache.put(pname, pdata)
                del ref
                self.assertTrue(False)
            except RuntimeError:
                # Success!
                pass
            # This should work
            ref = self.cache.put(pname, pdata, replace=True)
            del ref

        self.cache.clear()

        # Populate some buffers
        for k, v in self.types.items():
            pname = "test-{}".format(k)
            pdata = np.ones((self.nsamp, 4), v)
            ref = self.pycache.put(pname, pdata)
            del ref

        # Test putting the same buffer- it should just return a new array
        # wrapping the same underlying memory.
        for k, v in self.types.items():
            pname = "test-{}".format(k)
            pdata = np.ones((self.nsamp, 4), v)
            ref = self.pycache.reference(pname)
            copyref = self.pycache.put(pname, ref)
            ref_pnt = ref.ctypes.data_as(ctypes.c_void_p).value
            copyref_pnt = copyref.ctypes.data_as(ctypes.c_void_p).value
            self.assertTrue(ref_pnt == copyref_pnt)
            del ref
            del copyref

        # Test replacement of buffers with the same name
        for k, v in self.types.items():
            pname = "test-{}".format(k)
            pdata = np.ones((self.nsamp, 8), v)
            try:
                # This should raise- same name as existing buffer, but
                # replace == False
                ref = self.pycache.put(pname, pdata)
                del ref
                self.assertTrue(False)
            except RuntimeError:
                # Success!
                pass
            # This should work
            ref = self.pycache.put(pname, pdata, replace=True)
            del ref

        self.pycache.clear()
        return

    def test_create_none(self):
        try:
            ref = self.cache.create(None, np.float, (1, 10))
            del ref
            ref = self.pycache.create(None, np.float, (1, 10))
            del ref
            raise RuntimeError("Creating object with None key succeeded")
        except ValueError:
            pass

        self.cache.clear()
        self.pycache.clear()
        return

    def test_put_none(self):
        try:
            ref = self.cache.put(None, np.float, np.arange(10))
            del ref
            ref = self.pycache.put(None, np.float, np.arange(10))
            del ref
            raise RuntimeError("Putting an object with None key succeeded")
        except ValueError:
            pass

        self.cache.clear()
        self.pycache.clear()
        return

    def test_clear(self):
        for k, v in self.types.items():
            ref = self.cache.create("test-{}".format(k), v, (self.nsamp, 4))
            del ref
            ref = self.pycache.create("test-{}".format(k), v, (self.nsamp, 4))
            del ref
        self.cache.clear()
        self.pycache.clear()
        return

    def test_alias(self):
        ref = self.cache.put("test", np.arange(10))
        del ref

        self.cache.add_alias("test-alias", "test")
        self.cache.add_alias("test-alias-2", "test")

        data = self.cache.reference("test-alias")
        del data

        self.cache.destroy("test-alias")

        data = self.cache.reference("test-alias-2")
        del data

        self.cache.destroy("test")

        ex = self.cache.exists("test-alias-2")
        self.assertFalse(ex)

        self.cache.clear()

        ref = self.pycache.put("test", np.arange(10))
        del ref

        self.pycache.add_alias("test-alias", "test")
        self.pycache.add_alias("test-alias-2", "test")

        data = self.pycache.reference("test-alias")
        del data

        self.pycache.destroy("test-alias")

        data = self.pycache.reference("test-alias-2")
        del data

        self.pycache.destroy("test")

        ex = self.pycache.exists("test-alias-2")
        self.assertFalse(ex)

        self.pycache.clear()
        return

    def test_memfree(self):
        memreport(comm=self.comm, msg="Before large buffer creation")
        memcache = Cache(pymem=False)
        ref = memcache.create("test-big", np.uint8, (self.membytes,))
        del ref

        memreport(comm=self.comm, msg="After large buffer creation")

        mem = memcache.report(silent=True)
        self.assertEqual(mem, self.membytes)
        print("Cache now has {} bytes".format(mem), flush=True)

        memcache.clear()

        memreport(comm=self.comm, msg="After cache clear")
        mem = memcache.report(silent=True)
        self.assertEqual(mem, 0)
        print("Cache now has {} bytes".format(mem), flush=True)

        smallbytes = self.membytes // self.memnbuf
        for i in range(self.memnbuf):
            name = "test-small_{}".format(i)
            ref = memcache.create(name, np.uint8, (smallbytes,))
            del ref

        memreport(
            comm=self.comm,
            msg="After creation of {} small buffers".format(self.memnbuf),
        )
        mem = memcache.report(silent=True)
        self.assertEqual(mem, self.memnbuf * smallbytes)
        print("Cache now has {} bytes".format(mem), flush=True)

        memcache.clear()
        memreport(comm=self.comm, msg="After cache clear")
        mem = memcache.report(silent=True)
        self.assertEqual(mem, 0)
        print("Cache now has {} bytes".format(mem), flush=True)
