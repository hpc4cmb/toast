# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI
from .mpi import MPITestCase

from ..cache import *

import sys
import warnings

import timemory

class CacheTest(MPITestCase):

    def setUp(self):
        self.nsamp = 1000
        self.cache = Cache(pymem=False)
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
            ref = self.cache.create('test-{}'.format(k), v, (self.nsamp,4))
            del ref

        for k, v in self.types.items():
            data = self.cache.reference('test-{}'.format(k))
            data[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1,4)
            del data

        for k, v in self.types.items():
            ex = self.cache.exists('test-{}'.format(k))
            self.assertTrue(ex)

        for k, v in self.types.items():
            self.cache.destroy('test-{}'.format(k))

        self.cache.clear()

        for k, v in self.types.items():
            ref = self.pycache.create('test-{}'.format(k), v, (self.nsamp,4))
            del ref

        for k, v in self.types.items():
            data = self.pycache.reference('test-{}'.format(k))
            data[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1,4)
            del data

        for k, v in self.types.items():
            ex = self.pycache.exists('test-{}'.format(k))
            self.assertTrue(ex)

        for k, v in self.types.items():
            self.pycache.destroy('test-{}'.format(k))

        self.cache.clear()

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache create test took {:.3f} s".format(elapsed))


    def test_create_none(self):
        start = MPI.Wtime()

        try:
            ref = self.cache.create(None, np.float, (1,10))
            raise RuntimeError('Creating object with None key succeeded')
        except ValueError:
            pass

        self.cache.clear()

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns(
            "cache create none test took {:.3f} s".format(elapsed))


    def test_put_none(self):
        start = MPI.Wtime()

        try:
            ref = self.cache.put(None, np.float, np.arange(10))
            raise RuntimeError('Putting an object with None key succeeded')
        except ValueError:
            pass

        self.cache.clear()

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache put none test took {:.3f} s".format(elapsed))


    def test_clear(self):
        start = MPI.Wtime()

        for k, v in self.types.items():
            ref = self.cache.create('test-{}'.format(k), v, (self.nsamp,4))
            del ref

        warnings.filterwarnings('error')
        self.cache.clear('.*')
        self.cache.clear('.*')
        warnings.resetwarnings()

        self.cache.clear()

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache clear test took {:.3f} s".format(elapsed))


    def test_alias(self):
        start = MPI.Wtime()

        ref = self.cache.put('test', np.arange(10))
        del ref

        self.cache.add_alias('test-alias', 'test')
        self.cache.add_alias('test-alias-2', 'test')

        data = self.cache.reference('test-alias')
        del data

        self.cache.destroy('test-alias')

        data = self.cache.reference('test-alias-2')
        del data

        self.cache.destroy('test')

        ex = self.cache.exists('test-alias-2')
        self.assertFalse(ex)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache alias test took {:.3f} s".format(elapsed))


    def test_fscache(self):
        start = MPI.Wtime()

        FsBuffer.temporary_directory = '/tmp'
        self.cache.clear()
        self.cache._pymem = False

        compare = {}
        # create and store
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            # create with use_disk=True --> auto_disk_array
            ref = self.cache.create(name, v, (self.nsamp,4), use_disk=True)
            # check that ref is auto_disk_array
            self.assertTrue(isinstance(ref, auto_disk_array))
            compare[name] = self.cache.reference(name)
            self.assertTrue(self.cache.auto_reference_count(name) == 2)
            del ref

        # various checks
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            # the increment array
            incr = np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1,4)

            # check the only Cache._auto_refs count is only holding onto one in compare
            self.assertTrue(self.cache.auto_reference_count(name) == 1)
            # get a reference
            data = self.cache.reference(name)

            # check the Cache object updated Cache._auto_refs count
            self.assertTrue(self.cache.auto_reference_count(name) == 2)

            # check the types
            self.assertTrue(isinstance(data, auto_disk_array))
            self.assertTrue(isinstance(compare[name], auto_disk_array))
            # modify the 'data' object obtained from Cache.reference
            data[:] += incr

            # check the Cache object didn't update Cache._auto_refs count
            self.assertTrue(self.cache.auto_reference_count(name) == 2)

            # check the modifications applied to compare[name] also
            self.assertTrue(np.array_equal(compare[name], data))
            del data
            compare[name][:] += incr

            # check the modifications are synced between to compare[name] and Cache._refs
            self.assertTrue(np.array_equal(compare[name], self.cache.reference(name)))
            # check the Cache object updated Cache._auto_refs count
            self.assertTrue(self.cache.auto_reference_count(name) == 1)

        # delete the dictionary of auto_disk_arrays (should put into FsBuffer objects)
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            del compare[name]

        # check that auto_disk_array put data back into FsBuffers at deletion
        for k, v in self.types.items():
            print('')
            name = 'test-{}'.format(k)
            self.assertTrue(self.cache.exists(name, use_disk=True))
            self.assertFalse(self.cache.exists(name, use_disk=False))

        # similar to clear but don't delete FsBuffer objects
        self.cache.free()

        # check the data still exists after Cache.free()
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            self.assertTrue(self.cache.exists(name, use_disk=True))
            self.assertFalse(self.cache.exists(name, use_disk=False))

        # call destroy but don't delete FsBuffer objects
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            self.cache.destroy(name, remove_disk=False)

        # check the data still exists after Cache.destroy(name, remove_disk=False)
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            self.assertTrue(self.cache.exists(name, use_disk=True))
            self.assertFalse(self.cache.exists(name, use_disk=False))

        # actually delete the data
        self.cache.clear()

        # check the cache really has been cleared
        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            self.assertFalse(self.cache.exists(name, use_disk=True))
            self.assertFalse(self.cache.exists(name, use_disk=False))

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache fscache test took {:.3f} s".format(elapsed))


    def test_fscache_copy(self):
        global toast_cache_tmpdir

        start = MPI.Wtime()

        toast_cache_tmpdir = '/tmp'
        self.cache._pymem = False

        def get_clones():
            clones = {}
            for k, v in self.types.items():
                name = 'test-{}'.format(k)
                ref = self.cache.create(name, v, (self.nsamp,4), use_disk=True)
                # check the create produces an auto_disk_array
                self.assertTrue(isinstance(ref, auto_disk_array))
                # check the Cache._auto_refs count
                self.assertTrue(self.cache.auto_reference_count(name) == 1)
                # get a copy
                dup = self.cache.copy(name)
                # check the copy is not be an auto_disk_array
                self.assertFalse(isinstance(dup, auto_disk_array))
                # store the np.ndarray
                clones[name] = dup
                # delete our auto_disk_array
                del ref
                # modify the np.ndarray
                dup[:] += np.repeat(np.arange(self.nsamp, dtype=v), 4).reshape(-1,4)
            return clones

        clones = get_clones()

        for k, v in self.types.items():
            name = 'test-{}'.format(k)
            print('Checking for "{}"...'.format(name))
            # check the del of auto_disk_array produced a FsBuffer object
            self.assertTrue(self.cache.exists(name, use_disk=True))
            self.assertFalse(self.cache.exists(name, use_disk=False))
            # get the reference
            ref = self.cache.reference(name)
            # check out modification to the copy did not affect the reference
            self.assertFalse(np.array_equal(clones[name], ref))
            self.cache.destroy(name)

        clones.clear()

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache fscache copy test took {:.3f} s".format(elapsed))


    def test_fscache_size(self):
        start = MPI.Wtime()

        _size = 5000
        _name = 'test-array'
        _key = { 0 : 'initial-memory-usage',        # 0
                 1 : 'regular-reference',           # 1
                 2 : 'move-to-disk',                # 2
                 3 : 'load-from-disk',              # 3
                 4 : 'clear-cache',                 # 4
                 5 : 'auto-disk-array-init',        # 5
                 6 : 'auto-disk-array-del',         # 6
                 7 : 'final',                       # 7
                }
        _rss = { _key[0] : timemory.rss_usage(),    # 0
                 _key[1] : timemory.rss_usage(),    # 1
                 _key[2] : timemory.rss_usage(),    # 2
                 _key[3] : timemory.rss_usage(),    # 3
                 _key[4] : timemory.rss_usage(),    # 4
                 _key[5] : timemory.rss_usage(),    # 5
                 _key[6] : timemory.rss_usage(),    # 6
                 _key[7] : timemory.rss_usage(),    # 7
                }
        _base = timemory.rss_usage()

        # clear the cache
        self.cache.clear()
        self.cache._pymem = False

        _base.record()

        #
        # --> 0. record the initial RSS usage
        #
        _rss['initial-memory-usage'].record()

        # create a non-disk reference
        _ref = self.cache.create(_name, np.float64, [_size, _size], use_disk=False)
        del _ref
        #
        # --> 1. record the size of cache with standard reference
        #
        _rss['regular-reference'].record()

        # move to disk
        self.cache.move_to_disk(_name)
        #
        # --> 2. record the size of cache with FsBuffer reference
        #
        _rss['move-to-disk'].record()

        # load from disk
        self.cache.load_from_disk(_name)
        #
        # --> 3. record the size of the cache with a loaded FsBuffer reference
        #
        _rss['load-from-disk'].record()

        # clear the cache, RSS should
        self.cache.clear()
        #
        # --> 4. record the size of the cleared cache
        #
        _rss['clear-cache'].record()

        # create a auto_disk_array reference
        _ref = self.cache.create(_name, np.float64, [_size, _size], use_disk=True)
        #
        # --> 5. record the size with auto_disk_array (ref)
        #
        _rss['auto-disk-array-init'].record()

        # delete the auto_disk_array, this should move to FsBuffer
        del _ref
        #
        # --> 6. record the size with auto_disk_array (ref) --> FsBuffer
        #
        _rss['auto-disk-array-del'].record()

        # clear the cache
        self.cache.clear()
        #
        # --> 7. record the final size
        #
        _rss['final'].record()


        # calculate the relative difference
        def relative_difference(_lhs, _rhs):
            _sum = _lhs + _rhs
            # sum to less than KB == negligible
            if _sum.current() < 0.001:
                return 0.0
            return abs(_lhs.current() - _rhs.current()) / (0.5 * _sum.current())

        print('\nTotal memory summary:\n')
        for key, val in _rss.items():
            print('\t{:20} --> {}'.format(key, val))

        # subtract out the baseline
        for key, val in _rss.items():
            val -= _base

        print('\nRelative (to initial) memory summary:\n')
        for key, val in _rss.items():
            print('\t{:20} --> {}'.format(key, val))

        epsilon = 1.0e-7 # ~ 10 bytes
        #
        # --> Tests:
        #
        # regular reference should be less than FsBuffer references
        self.assertFalse(_rss[_key[1]].current() < _rss[_key[2]].current())
        self.assertFalse(_rss[_key[1]].current() < _rss[_key[6]].current())
        self.assertFalse(_rss[_key[1]].current() < _rss[_key[2]].current())

        # check approximately same size
        self.assertTrue(relative_difference(_rss[_key[1]], _rss[_key[3]]) < epsilon)
        self.assertTrue(relative_difference(_rss[_key[2]], _rss[_key[6]]) < epsilon)
        self.assertTrue(relative_difference(_rss[_key[3]], _rss[_key[5]]) < epsilon)
        self.assertTrue(relative_difference(_rss[_key[4]], _rss[_key[7]]) < epsilon)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("cache fscache size test took {:.3f} s".format(elapsed))
