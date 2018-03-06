# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re
import sys
import numpy as np

import warnings
import gc

from .cbuffer import ToastBuffer

import os
from tempfile import NamedTemporaryFile as named_temp_file
import timemory
from .mpi import MPI
import traceback

# ---------------------------------------------------------------------------- #
# traceback
def print_traceback():
    print('')
    lines = traceback.format_stack(limit=7)
    lines = [ "{}> {}".format(MPI.COMM_WORLD.rank, x) for x in lines ]
    lines = ''.join(lines)
    print("{}".format(lines))


# ---------------------------------------------------------------------------- #
# global values
toast_cache_tmpdir = None


# ---------------------------------------------------------------------------- #
def get_cache_verbosity():
    _verbose = 0
    try:
        _verbose = int(os.environ['TOAST_CACHE_VERBOSE'])
    except:
        pass
    return _verbose


# ---------------------------------------------------------------------------- #
def get_fscache_default_behavior():
    _enable = 0
    try:
        _enable = int(os.environ['TOAST_USE_FSCACHE'])
    except:
        pass
    return True if _enable > 0 else False


# ---------------------------------------------------------------------------- #
def numpy2toast(nptype):
    comptype = np.dtype(nptype)
    tbtype = None
    if comptype == np.dtype(np.float64):
        tbtype = "float64"
    elif comptype == np.dtype(np.float32):
        tbtype = "float32"
    elif comptype == np.dtype(np.int64):
        tbtype = "int64"
    elif comptype == np.dtype(np.uint64):
        tbtype = "uint64"
    elif comptype == np.dtype(np.int32):
        tbtype = "int32"
    elif comptype == np.dtype(np.uint32):
        tbtype = "uint32"
    elif comptype == np.dtype(np.int16):
        tbtype = "int16"
    elif comptype == np.dtype(np.uint16):
        tbtype = "uint16"
    elif comptype == np.dtype(np.int8):
        tbtype = "int8"
    elif comptype == np.dtype(np.uint8):
        tbtype = "uint8"
    else:
        raise RuntimeError("incompatible numpy type {}".format(comptype))
    return tbtype


# ---------------------------------------------------------------------------- #
def toast2numpy(tbtype):
    nptype = None
    if tbtype == "float64":
        nptype = np.dtype(np.float64)
    elif tbtype == "float32":
        nptype = np.dtype(np.float32)
    elif tbtype == "int64":
        nptype = np.dtype(np.int64)
    elif tbtype == "uint64":
        nptype = np.dtype(np.uint64)
    elif tbtype == "int32":
        nptype = np.dtype(np.int32)
    elif tbtype == "uint32":
        nptype = np.dtype(np.uint32)
    elif tbtype == "int16":
        nptype = np.dtype(np.int16)
    elif tbtype == "uint16":
        nptype = np.dtype(np.uint16)
    elif tbtype == "int8":
        nptype = np.dtype(np.int8)
    elif tbtype == "uint8":
        nptype = np.dtype(np.uint8)
    else:
        raise RuntimeError("incompatible ToastBuffer type {}".format(tbtype))
    return nptype


# ---------------------------------------------------------------------------- #
def get_buffer_directory():
    """
    The location of where the FsBuffer cache files are located
    """
    global toast_cache_tmpdir

    # if set explicitly or set during first call of this function
    if toast_cache_tmpdir is None:
        toast_cache_tmpdir = '/tmp'
        try:
            toast_cache_tmpdir = os.environ['TOAST_TMPDIR']
        except:
            pass

    # ensure directory exists
    if not os.path.exists(toast_cache_tmpdir):
        os.makedirs(toast_cache_tmpdir)

    # return toast_cache_tmpdir
    return toast_cache_tmpdir


# ---------------------------------------------------------------------------- #
class auto_disk_array(np.ndarray):
    """
    A special wrapper class around an np.ndarray that handles synchronization
    between FsBuffer storage and memory
    """
    # ------------------------------------------------------------------------ #
    def __new__(cls, _input, _cache, _name, _incr=0):
        # if input is FsBuffer object: load it
        # else: use it
        input_array = None
        if isinstance(_input, FsBuffer):
            # if FsBuffer object, load from memory
            input_array = _input.load()
            # ensure no FsBuffer object and reference object by deleting
            # from Cache object
            del _cache._fsbuffers[_name]

        elif isinstance(_input, auto_disk_array):
            # read message below
            raise RuntimeError("Internal coding error! auto_disk_array (name " +
                "== '{}') should not be created from another ".format(_name) +
                "auto_disk_array instance. This very likely means that the " +
                "auto_disk_array instance is stored somewhere in the Cache " +
                "object -- which means garbage collection will not happen " +
                "--> defeating the purpose of this class");

        else:
            # should be np.ndarray here
            if not isinstance(_input, np.ndarray):
                # this is a problem
                raise RuntimeError("Trying to create an auto_disk_array from " +
                    "{} which is not one of: [ {}, {}, {} ]".format(
                    type(_input).__name__, "FsBuffer", "auto_disk_array", "np.ndarray"))
            else:
                # it is np.ndarray, it's good
                input_array = _input

        # make sure it is not None
        assert(input_array is not None)

        # add to Cache._refs
        if _name not in _cache._refs.keys():
            input_array = _cache.create(_name, input_array.dtype, input_array.shape)

        # input_array is an already formed np.ndarray instance
        # --> cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        obj._cache = _cache
        obj._name = _name
        obj._incr = _incr
        obj._base = input_array

        # notify of creation
        if get_cache_verbosity() > 0:
            _b = 4
            print('--> Creating auto_disk_array[{}] (#{}) [data={}] from {}@{}:{}'.format(_name,
                _cache.auto_reference_count(_name)+1, obj.copy(),
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

        # insert into Cache._auto_refs if not already there
        if not _name in _cache._auto_refs.keys():
            _cache._auto_refs[_name] = 0

        # increment the Cache._auto_refs
        _cache._auto_refs[_name] += 1

        # don't let it exist in FsBuffer dictionary simulateously
        if _name in _cache._fsbuffers.keys():
            del _cache._fsbuffers[_name]

        # Finally, we must return the newly created object:
        return obj


    # ------------------------------------------------------------------------ #
    def __array_finalize__(self, obj):
        """
        Similar to __init__ but required by np.ndarray for proper sub-classing.
        """
        if obj is None:
            return
        self._cache = getattr(obj, '_cache', None)
        self._name = getattr(obj, '_name', None)
        self._incr = getattr(obj, '_incr', 1)
        self._base = getattr(obj, '_base', obj)


        if get_cache_verbosity() > 0:
            _b = 2
            if isinstance(obj, auto_disk_array) and self.base is not None:
                print('--> Finalizing auto_disk_array with "{}" [data={}] from {}@{}:{}'.format(
                    type(obj).__name__, obj.copy(),
                    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b) ))

            elif isinstance(obj, np.ndarray) and self.base is not None:
                print('--> Finalizing auto_disk_array with "{}" [data={}] from {}@{}:{}'.format(
                    type(obj).__name__, obj.copy(),
                    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b) ))
            #
            elif isinstance(obj, np.ndarray) and not isinstance(obj, auto_disk_array) :
                print('--> Finalizing auto_disk_array with "{}" [data={}] from {}@{}:{}'.format(
                    type(obj).__name__, obj.copy(),
                    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b) ))
            else:
                print('--> Finalizing auto_disk_array with "{}" [data={}] from {}@{}:{}'.format(
                    type(obj).__name__, 'unknown',
                    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b) ))

        # increment the Cache._auto_refs
        if self._cache is not None and self._name is not None:
            # notify of creation
            if get_cache_verbosity() > 0:
                _n = self._cache.auto_reference_count(self._name) + self._incr
                _b = 2
                print('--> Finalizing auto_disk_array[{}] (#{}) [zero={}] from {}@{}:{}'.format(
                    self._name, _n, 0,
                    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))
                #print('--> {} [{}] (#{} - +{}) from {}@{}:{}'.format(timemory.FUNC(),
                #    self._name, _n, self._incr,
                #    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

            if self._incr > 0:
                # insert into Cache._auto_refs if not already there
                if not self._name in self._cache._auto_refs.keys():
                    self._cache._auto_refs[self._name] = 0

                self._cache._auto_refs[self._name] += self._incr

            # don't let it exist in FsBuffer dictionary simulateously
            if (self._name in self._cache._fsbuffers.keys() and
                self._cache.auto_reference_count(self._name) > 0):
                del self._cache._fsbuffers[self._name]
                if get_cache_verbosity() > 0:
                    print('--> Deleted FsBuffer for auto_disk_array[{}] (#{}) [zero={}]'.format(
                        self._name, self._cache.auto_reference_count(self._name),
                        0 ))


    # ------------------------------------------------------------------------ #
    def __del__(self):
        """
        This destructor will check if the auto_ref object should delete the
        reference and put the array into a FsBuffer object or just do nothing
        """
        # if _cache._auto_refs no longer references these, ignore
        if self._name in self._cache._auto_refs.keys():
            if get_cache_verbosity() > 0:
                print('Deleting auto_disk_array[{}] (#{}) [zero={}]'.format(self._name,
                    self._cache.auto_reference_count(self._name), 0 ))

            if self._cache._auto_refs[self._name] > 0:
                self._cache._auto_refs[self._name] -= 1

            if self._cache._auto_refs[self._name] < 0:
                raise RuntimeError('Cache object ["{}"]'.format(self) +
                    ' in auto_disk_array has an unexpected _auto_ref count: ' +
                    '{}'.format(self._cache.auto_reference_count(self._name)))

            if self._cache.auto_reference_count(self._name) == 0:

                if self._name in self._cache._auto_refs.keys():
                    # remove from the Cache._auto_refs
                    del self._cache._auto_refs[self._name]

                if self.base is not None:
                    # tell Cache object to move to disk
                    self._cache.move_to_disk(self._name)


    # ------------------------------------------------------------------------ #
    def __str__(self):
        self._base = self.base
        return '{}(name="{}", incr={}, dtype={}, shape={}) = {}'.format(type(self).__name__,
            self._name, self._incr, self.dtype, self.shape, self.base.copy())


    # ------------------------------------------------------------------------ #
    def __array__(self, idtype):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__array__({}) from {}@{}:{}'.format(self._name,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

        if idtype == self.dtype:
            return self
        else:
            return self.base.copy()

    # ------------------------------------------------------------------------ #
    def __array_wrap__(self, out_arr, context=None):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__array_wrap__({}) from {}@{}:{}'.format(self._name,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

        # then just call the parent
        obj = self.base.__array_wrap__(out_arr, context)
        return auto_disk_array(np.asarray(obj).view(np.ndarray), self._cache, self._name)


    # ------------------------------------------------------------------------ #
    def copy(self):
        return self.base.copy()


    # ------------------------------------------------------------------------ #
    def __call__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__call__({})'.format(self))
        return super(auto_disk_array, self).__call__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __setitem__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__setitem__({}) from {}@{}:{}'.format(self,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))
        self._incr = 1
        self.base.__setitem__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __repr__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__repr__({}) from {}@{}:{}'.format(self,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))
        return super(auto_disk_array, self).__repr__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __getitem__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__getitem__({}) from {}@{}:{}'.format(self,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

        if self.base is None:
            return None

        self._incr = 1
        return self.base.__getitem__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    # ensure the returned type is a auto_disk_array instance
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html (v1.14)
    #
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        class _array_info(object):

            def __init__(self, _arr):
                self._cache = _arr._cache
                self._name = _arr._name
                self._dtype = _arr.dtype
                self._shape = _arr.shape

        if get_cache_verbosity() > 0:
            print('ufunc: "{}", method: "{}"'.format(ufunc.__name__, method))
            print_traceback()
            _b = 2
            print('__array_ufunc__[{}] (# {}) from {}@{}:{}'.format(self._name,
                self._cache.auto_reference_count(self._name),
                timemory.FUNC(_b),
                timemory.FILE(_b+1),
                timemory.LINE(_b)))

        args = []
        in_no = -1
        in_info = []
        try:
            for i, _input in enumerate(inputs):
                if isinstance(_input, auto_disk_array):
                    if in_no < 0:
                        in_no += 1
                    args.append(_input.view(np.ndarray))
                    in_info.append(_array_info(_input))
                else:
                    args.append(_input)

                if get_cache_verbosity() > 0:
                    if isinstance(_input, auto_disk_array):
                        print('Input {} : {}(dtype={}, shape={}) [zero={}]'.format(
                            i, type(_input).__name__, _input.dtype, _input.shape, 0 ))
                    elif isinstance(_input, np.ndarray):
                        print('Input {} : {}(dtype={}, shape={}) [zero={}]'.format(
                            i, type(_input).__name__, _input.dtype, _input.shape, 0 ))
                    else:
                        print('Input {} : {} == {}'.format(i, type(_input).__name__), _input)
        except:
            pass

        outputs = kwargs.pop('out', None)
        out_no = -1
        out_info = []
        if outputs:
            out_args = []
            for j, _output in enumerate(outputs):
                if isinstance(_output, auto_disk_array):
                    if out_no < 0:
                        out_no += 1
                    out_args.append(_output.view(np.ndarray))
                    out_info.append(_array_info(_output))
                else:
                    out_args.append(_output)

                if get_cache_verbosity() > 0:
                    if isinstance(_output, auto_disk_array):
                        print('Output {} : {}(dtype={}, shape={}) [zero={}]'.format(
                            j, type(_output).__name__, _output.dtype, _output.shape, 0 ))
                    elif isinstance(_output, np.ndarray):
                        print('Output {} : {}(dtype={}, shape={}) [zero={}]'.format(
                            j, type(_output).__name__, _output.dtype, _output.shape, 0 ))
                    else:
                        print('Output {} : {} == {}'.format(j, type(_output).__name__), _output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        _info = None
        if not in_no < 0:
            _info = in_info[in_no]

        if not out_no < 0:
            _info = out_info[out_no]

        results = super(auto_disk_array, self).__array_ufunc__(ufunc, method,
                                                               *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], auto_disk_array):
                inputs[0]._cache = _info._cache
                inputs[0]._name = _info._name
                inputs[0]._incr = 0
            return

        if ufunc.nout == 1:
            results = (results,)

        if results:
            _result = results[0]

            if (isinstance(_result, np.ndarray) and _result.dtype == _info._dtype):
                    results = tuple((#auto_disk_array(result, _info._cache, _info._name)
                                     np.asarray(result).view(auto_disk_array)
                                     if output is None else output)
                                     for result, output in zip(results, outputs))

            elif isinstance(_result, np.ndarray):
                results = tuple((np.asarray(result)
                                if output is None else output)
                                for result, output in zip(results, outputs))

            _result = results[0]

            if isinstance(_result, auto_disk_array):
                _result._cache = _info._cache
                _result._name = _info._name
                _result._incr = 1

            if get_cache_verbosity() > 0:
                if isinstance(_result, auto_disk_array):
                    print('Result : {}(dtype={}, shape={}) [zero={}]'.format(
                        type(_result).__name__, _result.dtype, _result.shape, 0 ))
                elif isinstance(_result, np.ndarray):
                    print('Result : {}(dtype={}, shape={}) [zero={}]'.format(
                        type(_result).__name__, _result.dtype, _result.shape, 0 ))
                else:
                    print('Result : {} is {}'.format(type(_result).__name__, _result))
        else:
            if get_cache_verbosity() > 0:
                print('Result : None')

        return results[0] if len(results) == 1 else results


    """
    # ------------------------------------------------------------------------ #
    def __array_interface__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_interface__({})'.format(self))
        return super(auto_disk_array, self).__array_interface__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __array_prepare__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_prepare__({})'.format(self))
        return super(auto_disk_array, self).__array_prepare__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __array_priority__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_priority__({})'.format(self))
        return super(auto_disk_array, self).__array_priority__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __array_struct__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_struct__({})'.format(self))
        return super(auto_disk_array, self).__array_struct__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __copy__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__copy__({})'.format(self))
        _base = super(auto_disk_array, self).__copy__(*args, **kwargs)
        return auto_disk_array(_base, self._cache, '{}{}'.format(self._name, '+'))


    # ------------------------------------------------------------------------ #
    def __deepcopy__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__deepcopy__({})'.format(self))
        return self.base.__deepcopy__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __index__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__index__({})'.format(self))
        return self.base.__index__(*args, **kwargs)


    # ------------------------------------------------------------------------ #
    def __iter__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__iter__({})'.format(self))
        return super(auto_disk_array, self).__iter__(*args, **kwargs)
    """


# ---------------------------------------------------------------------------- #
class FsBuffer(object):
    """
    File system buffer (used @ NERSC specifically for BurstBuffer) but
    can also use a temporary folder defined in TOAST_TMPDIR environment
    variable or with toast.cache.toast_tmpdir. The latter overrides the former
    """
    def __init__(self, obj):
        """
        Create a random temporary file for the FsBuffer object then
        write to disk.

        NOTE: calling function in Cache responsible for deleting from Cache._refs
        """
        self._buffer_file = named_temp_file(prefix='toast-cache-',
                                            dir=get_buffer_directory(),
                                            suffix='.cache',
                                            delete=False)
        self._buffer_type = obj.dtype
        self._buffer_shape = obj.shape
        self.unload(obj)


    def __del__(self):
        """
        Close the FsBuffer file and remove from OS
        """
        self.close()
        os.remove(self._buffer_file.name)


    def __str__(self):
        _name = self._buffer_file.name
        _name = _name.replace(get_buffer_directory(), '${TOAST_TMPDIR}')
        return 'FsBuffer(file={}, type={}, shape={})'.format(_name,
                                                             self._buffer_type,
                                                             self._buffer_shape)


    def open(self, fmode='w+b'):
        """
        Open the FsBuffer file
        """
        self._buffer_file = open(self._buffer_file.name, fmode)


    def close(self):
        """
        Close the FsBuffer file
        """
        self._buffer_file.close()


    def load(self):
        """
        Load the np.ndarray object into memory, i.e. read from file-system cache
        """
        if get_cache_verbosity() > 0:
            print('Load from buffer "{}"...'.format(self._buffer_file.name))

        try:
            obj = np.fromfile(self._buffer_file.name, dtype=self._buffer_type)

        except Exception as e:
            raise RuntimeError('{}\nFile {} invalid (ndarray type: {})'.format(
                e, self._buffer_file.name, self._buffer_type))

        if get_cache_verbosity() > 0:
            print('Load from buffer "{}" self (type: {}, shape: {}) [data={}]...'.format(
            self._buffer_file.name, self._buffer_type, self._buffer_shape,
            obj.copy().reshape(self._buffer_shape) ))

        return obj.reshape(self._buffer_shape)


    def unload(self, obj):
        """
        Unload the np.ndarray object from memory, i.e. put into file-system cache
        """

        if get_cache_verbosity():
            print('Unloading {} from buffer "{}" obj (type: {}, shape: {}) [data={}]...'.format(
            type(obj).__name__, self._buffer_file.name, obj.dtype, obj.shape,
            obj.copy() ))

        self._buffer_type = obj.dtype
        self._buffer_shape = obj.shape
        self.open()
        obj.tofile(self._buffer_file.name)
        self.close()


    def update(self, obj):
        """
        Alias for unload. Used for overwriting with changes
        """
        self.unload(obj)


# ---------------------------------------------------------------------------- #
class Cache(object):
    """
    Timestream data cache with explicit memory management.

    Args:
        pymem (bool): if True, use python memory rather than external
            allocations in C.  Only used for testing.
    """

    def __init__(self, pymem=False, use_fscache=None):
        self._pymem = pymem
        self._use_fscache = use_fscache
        self._refs = {}
        self._aliases = {}
        # file-system buffer objects
        self._fsbuffers = {}
        # counter for auto_disk_array. Used to detect whether _refs object
        # is a typical np.ndarray or an auto_disk_array object
        self._auto_refs = {}


    def __del__(self):
        # free all buffers at destruction time
        self._aliases.clear()
        if not self._pymem:
            keylist = list(self._refs.keys())
            for k in keylist:
                self.check_ref_count(k)
                del self._refs[k]
            for k in list(self._fsbuffers.keys()):
                del self._fsbuffers[k]
        self._refs.clear()
        self._fsbuffers.clear()
        self._auto_refs.clear()


    def set_use_disk(self, name, use_disk=True):
        """
        Enable or disable the use of file-system cache.
        """
        if use_disk:
            # enable FsBuffer storage
            if name in self._auto_refs.keys():
                # already set to using disk
                pass
            elif name in self._refs.keys():
                # if currently in Cache._refs, create FsBuffer object
                self._fsbuffers[name] = FsBuffer(self._refs[name])
                # destroy reference in lieu of FsBuffer storage
                self.destroy(name, remove_disk=False)
            elif name in self._aliases.keys():
                ref_name = self._aliases[name]
                if (not ref_name in self._auto_refs.keys() and
                    not ref_name in self._fsbuffers.keys()):
                    # if currently in Cache._refs, create FsBuffer object
                    self._fsbuffers[ref_name] = FsBuffer(self._refs[ref_name])
                    # destroy reference in lieu of FsBuffer storage
                    self.destroy(ref_name, remove_disk=False)
            else:
                if not name in self._fsbuffers.keys():
                    # not already an fsbuffer object
                    raise RuntimeError("Cache object named '{}' does not exist".format(name))
        elif not use_disk:
            # disable FsBuffer storage
            ref_name = name
            if name in self._aliases.keys() and not name in self._refs.keys():
                ref_name = self._aliases[name]

            # disable FsBuffer storage
            if ref_name in self._auto_refs.keys():
                return auto_disk_array(self._refs[ref_name], self, ref_name, 1)
            elif ref_name in self._fsbuffers.keys():
                if not ref_name in self._refs.keys():
                    tmp = self._fsbuffers[ref_name].load()
                    del self._fsbuffers[ref_name]
                    ref = self.put(ref_name, tmp, replace=True)
                    del tmp
                    return ref
                else:
                    # we should not have an object in _fsbuffers and _refs
                    raise RuntimeError("Internal logic error! Cache object named {} " + \
                        "exists in FsBuffer list and reference list".format(name))
            else:
                # doesn't exist
                raise RuntimeError("Cache object named {} does not exist".format(name))
        return None


    def move_to_disk(self, name):
        """
        Move an existing reference to file-system cache. Alias for
        Cache.set_use_disk(name, use_disk=True)

        Args:
            name (str): the name of the buffer.
        """
        self.set_use_disk(name, use_disk=True)


    def load_from_disk(self, name):
        """
        Load an existing FsBuffer object from the file-system cache. Alias for
        Cache.set_use_disk(name, use_disk=False)

        Args:
            name (str): the name of the buffer.
        """
        return self.set_use_disk(name, use_disk=False)


    def check_ref_count(self, key):
        """
        Check the reference count of a key in self._refs

        Args:
            name (str): the name to assign to the buffer.
        """
        # do garbage collection
        gc.collect()
        # if the object is None or is garbage-collected
        if self._refs[key] is None or gc.is_tracked(self._refs[key]):
            if self._refs[key] is not None:
                print('Object: {} is garbage collected'.format(type(self._refs[key]).__name__))
            return
        # if NOT garbage-collected, check the reference count
        _n = sys.getrefcount(self._refs[key])
        # if we have more than one reference (in Python, 2 == 1)
        if _n > 2:
            print_traceback()
            try:
                referrers = gc.get_referrers(self._refs[key])
                print("reference count for '{}' is {}".format(key, _n))
                for i in range(0, len(referrers)):
                    _ref = referrers[i]
                    print("  --> referrer #{} for '{}' is {} (line={})".format(i, key, _ref,
                                                                    _ref.f_code))
            except:
                print ('error with "gc.get_referrers(self._refs[{}])"'.format(key))
                #pass

            if get_cache_verbosity() > 1:
                print('Cache: {}'.format(self))

            msg = ("Cache object {} has external references [n={}] ".format(key, _n) +
                   "and will not be freed.")
            warnings.warn(msg, RuntimeWarning)


    def clear(self, pattern=None, remove_disk=True):
        """
        Clear one or more buffers.

        Args:
            pattern (str): a regular expression to match against the buffer
                names when determining what should be cleared.  If None,
                then all buffers are cleared.
        """
        if pattern is None:
            # free all buffers
            self._aliases.clear()
            _keep = {}
            if not self._pymem:
                keylist = list(self._refs.keys())
                for k in keylist:
                    if not remove_disk:
                        if not k in self._auto_refs.keys():
                            self.check_ref_count(k)
                            del self._refs[k]
                        else:
                            _keep[k] = self._refs[k]
                if remove_disk:
                    for k in list(self._fsbuffers.keys()):
                        del self._fsbuffers[k]
            self._refs.clear()
            self._refs = _keep
            if remove_disk:
                self._fsbuffers.clear()
                self._auto_refs.clear()
        else:
            pat = re.compile(pattern)
            names = []
            for n, r in self._refs.items():
                mat = pat.match(n)
                if mat is not None:
                    names.append(n)
                del r
            if remove_disk:
                for n, r in self._fsbuffers.items():
                    mat = pat.match(n)
                    if mat is not None:
                        names.append(n)
                    del r
            for n in names:
                self.destroy(n, remove_disk)
        return


    def free(self, pattern=None):
        """
        Call clear on the buffers but don't delete disk copies

        Args:
            pattern (str): a regular expression to match against the buffer
                names when determining what should be cleared.  If None,
                then all buffers are cleared.
        """
        self.clear(pattern, remove_disk=False)


    def create(self, name, type, shape, use_disk=None):
        """
        Create a named data buffer of the given type and shape.

        Args:
            name (str): the name to assign to the buffer.
            type (numpy.dtype): one of the supported numpy types.
            shape (tuple): a tuple containing the shape of the buffer.
            use_disk (bool): explicitly enable/disable using file-system cache
              with auto_disk_array. Default is 'None' and if == None then
              fallback on default behavior
        """

        # self.get_fscache_behavior cannot be default parameter
        if use_disk is None:
            use_disk = self.get_fscache_behavior()

        if name is None:
            raise ValueError('Cache name cannot be None')

        if self.exists(name):
            raise RuntimeError("Data buffer or alias {} already exists".format(name))

        if self._pymem:
            self._refs[name] = np.zeros(shape, dtype=type)
        else:
            flatsize = 1
            for s in range(len(shape)):
                if shape[s] <= 0:
                    raise RuntimeError("Cache object must have non-zero sizes in all dimensions")
                flatsize *= shape[s]
            self._refs[name] = np.asarray( ToastBuffer(int(flatsize),
                numpy2toast(type)) ).reshape(shape)

        ret = self._refs[name]

        if use_disk:
            return auto_disk_array(self._refs[name], self, name)

        return ret


    def put(self, name, data, replace=False, use_disk=None):
        """
        Create a named data buffer to hold the provided data.
        If replace is True, existing buffer of the same name is first
        destroyed. If replace is True and the name is an alias, it is
        promoted to a new data buffer.

        Args:
            name (str): the name to assign to the buffer.
            data (numpy.ndarray): Numpy array
            replace (bool): Overwrite any existing keys
        """

        # self.get_fscache_behavior cannot be default parameter
        if use_disk is None:
            use_disk = self.get_fscache_behavior()

        if name is None:
            raise ValueError('Cache name cannot be None')

        if self.exists(name) and replace:
            ref = self.reference(name)
            if data is ref:
                return ref
            else:
                ref[:] = data.copy()
                del ref
            # Destroy the existing cache object but first make a copy
            # of the supplied data in case it is a view of a subset
            # of the cache data.
            mydata = data.copy()
            self.destroy(name)
        else:
            mydata = data

        ref = self.create(name, mydata.dtype, mydata.shape, use_disk=use_disk)
        ref[:] = mydata

        return ref


    def add_alias(self, alias, name):
        """
        Add an alias to a name that already exists in the cache.

        Args:
            alias (str): alias to create
            name (str): an existing key in the cache
        """

        if alias is None or name is None:
            raise ValueError('Cache name or alias cannot be None')

        if (name not in self._refs.keys() and
            name not in self._fsbuffers.keys()):
            raise RuntimeError("Data buffer {} does not exist for alias {}".format(name, alias))

        if alias in self._refs.keys():
            raise RuntimeError("Proposed alias {} would shadow existing buffer.".format(alias))

        self._aliases[alias] = name


    def destroy(self, name, remove_disk=True):
        """
        Deallocate the specified buffer.

        Only call this if all numpy arrays that reference the memory
        are out of use.

        Args:
            name (str): the name of the buffer or alias to destroy.
        """

        if name in self._aliases.keys():
            # Alias is a soft link. Do not remove the buffer
            del self._aliases[name]
            return

        if name not in self._refs.keys() and name not in self._fsbuffers.keys():
            raise RuntimeError("Data buffer {} does not exist".format(name))

        # Remove aliases to the buffer
        aliases_to_remove = []
        for key, value in self._aliases.items():
            if value == name and remove_disk:
                aliases_to_remove.append( key )
        for key in aliases_to_remove:
            del self._aliases[key]

        # check reference count
        if remove_disk and not self._pymem and name in self._refs.keys():
            self.check_ref_count(name)

        # Remove actual buffer
        if name in self._refs.keys():
            del self._refs[name]

        # remove disk reference
        if remove_disk:
            if name in self._fsbuffers.keys():
                del self._fsbuffers[name]
            if name in self._auto_refs.keys():
                del self._auto_refs[name]

        return


    def exists(self, name, return_ref=False, use_disk=True):
        """
        Check whether a buffer exists.

        Args:
            name (str): the name of the buffer to search for.
            return_ref (bool): return a reference if exists
            use_disk (bool): include checking the FsBuffer

        Returns:
            (array): a numpy array wrapping the raw data buffer or None if it does not exist.
        """
        # Do the existence check first, to avoid creating extra
        # references if we are not returning a reference.
        check = False
        if name in self._auto_refs.keys():
            check = True
        elif name in self._refs.keys():
            check = True
        elif name in self._aliases.keys():
            check = True
        elif use_disk and name in self._fsbuffers.keys():
            check = True

        if not return_ref:
            return check
        else:
            if not check:
                return None
            else:
                ref = None
                if name in self._auto_refs.keys():
                    # if it is in auto_refs, then we have another auto_ref
                    # elsewhere, so create another auto_disk_array from
                    # the base np.ndarray  in self._refs[name]
                    ref = auto_disk_array(self._refs[name], self, name)
                elif name in self._refs.keys():
                    # if already loaded (held by another auto_disk_array that
                    # has not been garbage-collected)
                    ref = self._refs[name]
                elif name in self._aliases.keys():
                    if self.get_fscache_behavior() or use_disk:
                        # if the aliased array is in FsBuffer:
                        #   load and delete FsBuffer object, the alias call will
                        #   hold a reference until garbage-collected
                        # else:
                        #   return reference
                        ref_name = self._aliases[name]
                        if ref_name in self._auto_refs.keys():
                            ref = auto_disk_array(self._refs[ref_name], self, ref_name)
                        elif ref_name in self._fsbuffers.keys():
                            tmp = self._fsbuffers[ref_name].load()
                            del self._fsbuffers[ref_name]
                            ref = auto_disk_array(tmp, self, ref_name, 1)
                        else:
                            ref = self._refs[ref_name]
                    else:
                        ref = self._refs[self._aliases[name]]
                elif use_disk and name in self._fsbuffers.keys():
                    # load a np.ndarray from the FsBuffer, let auto_disk_array
                    # handle putting into _refs and deleting from _fsbuffers
                    tmp = self._fsbuffers[name].load()
                    del self._fsbuffers[name]
                    ref = auto_disk_array(tmp, self, name)
                return ref


    def reference(self, name):
        """
        Return a numpy array pointing to the buffer.

        The returned array will wrap a pointer to the raw buffer, but will
        not claim ownership.  When the numpy array is garbage collected, it
        will NOT attempt to free the memory (you must manually use the
        destroy method).

        Args:
            name (str): the name of the buffer to return.

        Returns:
            (array): a numpy array wrapping the raw data buffer.
        """
        ref = self.exists(name, return_ref=True)
        if ref is None:
            raise RuntimeError("Data buffer (nor alias) {} does not exist".format(name))
        return ref


    def copy(self, name):
        """
        Return a copy of numpy array pointing to the buffer.

        Args:
            name (str): the name of the buffer to return.

        Returns:
            (array): a numpy array wrapping the raw data buffer.
        """
        ref = self.exists(name, return_ref=True)
        if ref is None:
            raise RuntimeError("Data buffer (nor alias) {} does not exist".format(name))
        # if we have an auto_disk_array instance from exists, then
        # return a copy of the base np.ndarray
        if isinstance(ref, auto_disk_array):
            return ref.base.copy()
        return ref.copy()


    def auto_reference_count(self, name):
        """
        Returns the number of auto_disk_arrays that exist for a specific buffer

        Args:
            name (str): the name of the buffer to return.

        Returns:
            (int): number of auto_disk_array objects using buffer
        """
        if name in self._auto_refs.keys():
            return self._auto_refs[name]
        return 0


    def keys(self):
        """
        Return a list of all the keys in the cache.

        Args:

        Returns:
            (list): List of key strings.
        """
        _list = list(self._refs.keys())
        _list.extend(list(self._fsbuffers.keys()))
        return _list


    def aliases(self):
        """
        Return a dictionary of all the aliases to keys in the cache.

        Args:

        Returns:
            (dict): Dictionary of aliases.
        """

        return self._aliases.copy()


    def __str__(self):
        _str = '[ '
        _err = ''
        try:
            # try printing aliases
            _err = 'aliases'
            if len(self._aliases) > 0:
                for _key, _val in self._aliases.items():
                    _str += " " if len(_str) > 2 else ""
                    _str += '[alias: {} = {}]'.format(_key, _val)
            # try printing auto references
            _err = 'auto-refs'
            if len(self._auto_refs) > 0:
                for _key in self._auto_refs.keys():
                    _str += " " if len(_str) > 2 else ""
                    _str += '[auto-ref: {}]'.format(_key)
            # try printing fsbuffer objects
            _err = 'fsbuffers'
            if len(self._fsbuffers) > 0:
                for _key, _val in self._fsbuffers.items():
                    _str += " " if len(_str) > 2 else ""
                    _str += '[fsbuffer: {} = {}]'.format(_key, _val)
            # try printing references
            _err = 'refs'
            if len(self._refs) > 0:
                for _key, _val in self._refs.items():
                    _str += " " if len(_str) > 2 else ""
                    _str += '[ref: {}]'.format(_key)
        except Exception as e:
            print ('Exception when processing "{}": {}'.format(_err, e))

        return _str + ']'


    def report(self, silent=False):
        """
        Report memory usage.

        Args:
            silent (bool):  Count and return the memory without printing.

        Returns:
            (int):  Amount of allocated memory in bytes
        """

        if not silent:
            print('Cache memory usage:')

        tot = 0
        for key in self.keys():
            ref = self.reference(key)
            sz = ref.nbytes
            del ref
            tot += sz
            if not silent:
                print(' - {:25} {:5.2f} MB'.format(key, sz/2**20))

        if not silent:
            print(' {:27} {:5.2f} MB'.format('TOTAL', tot/2**20))

        return tot


    #
    def get_fscache_behavior(self):
        """
        Use global 'cache.get_fscache_default_behavior()' as the file-system
        cache behavior unless specified by 'self._use_fscache', which is
        enabled/disabled by setting use_fscache={True,False} at Cache object
        initialization or using Cache.set_fscache_behavior(bool)
        """
        if self._use_fscache is not None:
            return self._use_fscache
        else:
            return get_fscache_default_behavior()


    def set_fscache_behavior(self, use_fscache):
        """
        Set file-system cache default behavior when invoking Cache.create(...)
        and Cache.put(...).
        If not set, global 'toast.cache.get_fscache_default_behavior()' will
        define the default file-system cache behavior

        Args:
            use_fscache (bool): Set file-system cache default behavior when
                invoking Cache.create(...) and Cache.put(...).
        """
        self._use_fscache = use_fscache
