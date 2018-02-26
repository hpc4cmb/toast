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
    #
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

        if get_cache_verbosity() > 0:
            _b = 2
            print('Creating auto_disk_array[{}] (#{}) from {}@{}:{}'.format(_name,
                _cache.auto_reference_count(_name)+1,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

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

        # notify of creation
        if get_cache_verbosity() > 0:
            _b = 4
            print('--> Cache name: {} from {}@{}:{}'.format(_name,
                                                            timemory.FUNC(_b),
                                                            timemory.FILE(_b+1),
                                                            timemory.LINE(_b)))

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


    #
    def __array_finalize__(self, obj):
        """
        Similar to __init__ but required by np.ndarray for proper sub-classing.
        """
        if obj is None:
            return
        self._cache = getattr(obj, '_cache', None)
        self._name = getattr(obj, '_name', None)
        self._incr = getattr(obj, '_incr', 0)


        # increment the Cache._auto_refs
        if self._cache is not None and self._name is not None:
            # notify of creation
            if get_cache_verbosity() > 0:
                _n = self._cache.auto_reference_count(self._name) + self._incr
                _b = 2
                print('--> {} [{}] (#{} - +{}) from {}@{}:{}'.format(timemory.FUNC(),
                    self._name, _n, self._incr,
                    timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

            if self._incr > 0:
                # insert into Cache._auto_refs if not already there
                if not self._name in self._cache._auto_refs.keys():
                    self._cache._auto_refs[self._name] = 0

                self._cache._auto_refs[self._name] += self._incr

            # don't let it exist in FsBuffer dictionary simulateously
            if (self._name in self._cache._fsbuffers.keys() and
                self._cache.auto_reference_count(self._name) > 0):
                del self._cache._fsbuffers[self._name]


    #
    def __del__(self):
        """
        This destructor will check if the auto_ref object should delete the
        reference and put the array into a FsBuffer object or just do nothing
        """
        # if _cache._auto_refs no longer references these, ignore
        if self._name in self._cache._auto_refs.keys():
            if get_cache_verbosity() > 0:
                print('Deleting auto_disk_array[{}] (#{})'.format(self._name,
                    self._cache.auto_reference_count(self._name)))

            if self._cache._auto_refs[self._name] > 0:
                self._cache._auto_refs[self._name] -= 1

            if self._cache._auto_refs[self._name] < 0:
                raise RuntimeError('Cache object ["{}"]'.format(self._name) +
                    ' in auto_disk_array has an unexpected _auto_ref count: ' +
                    '{}'.format(self._cache.auto_reference_count(self._name)))

            if self._cache.auto_reference_count(self._name) == 0:
                if get_cache_verbosity() > 0:
                    print ('Deleting auto_disk_array({})...'.format(self._name))

                if self.base is not None:
                    # unload to FsBuffer object
                    self._cache._fsbuffers[self._name] = FsBuffer(self.base)

                    # remove from the Cache._refs
                    self._cache.destroy(self._name, remove_disk=False)

                if self._name in self._cache._auto_refs.keys():
                    # remove from the Cache._auto_refs
                    del self._cache._auto_refs[self._name]


    #
    def __array_wrap__(self, out_arr, context=None):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__call__({}) from {}@{}:{}'.format(self._name,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))

        # then just call the parent
        obj = super(auto_disk_array, self).__array_wrap__(self, out_arr, context)
        return auto_disk_array(np.asarray(obj).view(np.ndarray), self._cache, self._name)


    #
    def __call__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__call__({})'.format(self._name))
        return super(auto_disk_array, self).__call__(*args, **kwargs)


    #
    def __setitem__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__setitem__({}) from {}@{}:{}'.format(self._name,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))
        self._incr = 1
        return super(auto_disk_array, self).__setitem__(*args, **kwargs)


    """
    #
    def __array_interface__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_interface__({})'.format(self._name))
        return super(auto_disk_array, self).__array_interface__(*args, **kwargs)


    #
    def __array_prepare__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_prepare__({})'.format(self._name))
        return super(auto_disk_array, self).__array_prepare__(*args, **kwargs)


    #
    def __array_priority__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_priority__({})'.format(self._name))
        return super(auto_disk_array, self).__array_priority__(*args, **kwargs)


    #
    def __array_struct__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__array_struct__({})'.format(self._name))
        return super(auto_disk_array, self).__array_struct__(*args, **kwargs)


    #
    def __copy__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__copy__({})'.format(self._name))
        return super(auto_disk_array, self).__copy__(*args, **kwargs)


    #
    def __deepcopy__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__deepcopy__({})'.format(self._name))
        return super(auto_disk_array, self).__deepcopy__(*args, **kwargs)

    #
    def __index__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__index__({})'.format(self._name))
        return super(auto_disk_array, self).__index__(*args, **kwargs)


    #
    def __iter__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            print('__iter__({})'.format(self._name))
        return super(auto_disk_array, self).__iter__(*args, **kwargs)
    """

    #
    def __repr__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__repr__({}) from {}@{}:{}'.format(self._name,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))
        return super(auto_disk_array, self).__repr__(*args, **kwargs)


    #
    def __getitem__(self, *args, **kwargs):
        if get_cache_verbosity() > 0:
            _b = 2
            print('__getitem__({}) from {}@{}:{}'.format(self._name,
                timemory.FUNC(_b), timemory.FILE(_b+1), timemory.LINE(_b)))
        return super(auto_disk_array, self).__getitem__(*args, **kwargs)


    # ensure the returned type is a auto_disk_array instance
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html (v1.14)
    #
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if get_cache_verbosity() > 0:
            _b = 2
            print('__array_ufunc__[{}] (# {}) from {}@{}:{}'.format(self._name,
                self._cache.auto_reference_count(self._name),
                timemory.FUNC(_b),
                timemory.FILE(_b+1),
                timemory.LINE(_b)))

        args = []
        in_no = -1
        in_caches = []
        in_names = []
        try:
            for i, input_ in enumerate(inputs):
                if isinstance(input_, auto_disk_array):
                    if in_no < 0:
                        in_no += 1
                    args.append(input_.view(np.ndarray))
                    in_caches.append(input_._cache)
                    in_names.append(input_._name)
                else:
                    args.append(input_)
        except:
            pass

        outputs = kwargs.pop('out', None)
        out_no = -1
        out_caches = []
        out_names = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, auto_disk_array):
                    if out_no < 0:
                        out_no += 1
                    out_args.append(output.view(np.ndarray))
                    out_caches.append(output._cache)
                    out_names.append(output._name)
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        _cache = None
        _name = None
        if not in_no < 0:
            _cache = in_caches[in_no]
            _name = in_caches[in_no]

        if not out_no < 0:
            _cache = out_caches[out_no]
            _name = out_caches[out_no]

        results = super(auto_disk_array, self).__array_ufunc__(ufunc, method,
                                                               *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], auto_disk_array):
                inputs[0]._cache = _cache
                inputs[0]._name = _name
                inputs[0]._incr = 0
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(auto_disk_array)
                         #auto_disk_array(np.asarray(result).view(np.ndarray), _cache, _name, 1)
                         if output is None else output)
                         for result, output in zip(results, outputs))

        if results and isinstance(results[0], auto_disk_array):
            results[0]._cache = _cache
            results[0]._name = _name
            results[0]._incr = 1

        return results[0] if len(results) == 1 else results


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
        self.unload(obj)


    def __del__(self):
        """
        Close the FsBuffer file and remove from OS
        """
        self.close()
        os.remove(self._buffer_file.name)


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
        return obj.reshape(self._buffer_shape)


    def unload(self, obj):
        """
        Unload the np.ndarray object from memory, i.e. put into file-system cache
        """

        if get_cache_verbosity():
            print('Unloading {} to buffer "{}"...'.format(type(obj).__name__,
                                                          self._buffer_file.name))

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

    def __init__(self, pymem=False):
        self._pymem = pymem
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
            else:
                if not name in self._fsbuffers.keys():
                    # not already an fsbuffer object
                    raise RuntimeError("Cache object named {} " + \
                        "does not exist".format(name))
        elif not use_disk:
            # disable FsBuffer storage
            if name in self._auto_refs.keys():
                # we should not have an object in _fsbuffers and _refs
                raise RuntimeError("Logic error! Cache object named {} " + \
                    "already has disk-usage enabled".format(name))
            elif name in self._fsbuffers.keys():
                if not name in self._refs.keys():
                    tmp = self._fsbuffers[name].load()
                    del self._fsbuffers[name]
                    ref = self.put(name, tmp, replace=True)
                    del tmp
                    del ref
                else:
                    # we should not have an object in _fsbuffers and _refs
                    raise RuntimeError("Internal logic error! Cache object named {} " + \
                        "exists in FsBuffer list and reference list".format(name))
            else:
                # doesn't exist
                raise RuntimeError("Cache object named {} does not exist".format(name))


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
        self.set_use_disk(name, use_disk=False)


    def check_ref_count(self, key):
        """
        Check the reference count of a key in self._refs

        Args:
            name (str): the name to assign to the buffer.
        """
        #referrers = gc.get_referrers(self._refs[key])
        #print("clear {} referrers for {} are: ".format(len(referrers), k), referrers)
        #print("clear refcount for {} is ".format(k), sys.getrefcount(self._refs[k]) )
        if sys.getrefcount(self._refs[key]) > 2:
            warnings.warn("Cache object {} has external references and will not be freed.".format(key),
                RuntimeWarning)


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


    def create(self, name, type, shape, use_disk=get_fscache_default_behavior()):
        """
        Create a named data buffer of the given type and shape.

        Args:
            name (str): the name to assign to the buffer.
            type (numpy.dtype): one of the supported numpy types.
            shape (tuple): a tuple containing the shape of the buffer.
        """

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


    def put(self, name, data, replace=False):
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

        if name is None:
            raise ValueError('Cache name cannot be None')

        if self.exists(name) and replace:
            ref = self.reference(name)
            if data is ref:
                return ref
            else:
                del ref
            # Destroy the existing cache object but first make a copy
            # of the supplied data in case it is a view of a subset
            # of the cache data.
            mydata = data.copy()
            self.destroy(name)
        else:
            mydata = data

        ref = self.create(name, mydata.dtype, mydata.shape, use_disk=False)
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

        if name not in self._refs.keys():
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
            if value == name:
                aliases_to_remove.append( key )
        for key in aliases_to_remove:
            del self._aliases[key]

        # check reference count
        if not self._pymem and name in self._refs.keys():
            self.check_ref_count(name)

        # Remove actual buffer
        if name in self._refs.keys():
            del self._refs[name]
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
                    ref = self._refs[name]
                elif name in self._aliases.keys():
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
