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
import tempfile


def get_buffer_directory():
    _dir = None
    try:
        _dir = os.environ['TOAST_TMPDIR']
    except:
        _dir = '/tmp'
    # ensure directory exists
    if not os.path.exists(_dir) and _dir != '':
        os.makedirs(_dir)
    return _dir


def get_use_fsbuffer():
    """
    Check environment for TOAST_NO_FSBUFFER=[0-9]. If TOAST_NO_FSBUFFER > 0,
    disable using FsBuffer
    """
    _fsbuffer_disable = 0
    try:
        _fsbuffer_disable = int(os.environ['TOAST_NO_FSBUFFER'])
    except:
        _fsbuffer_disable = 0
    if _fsbuffer_disable == 0:
        return True
    return False


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


class FsBuffer(object):
    """
    File system buffer (used @ NERSC specifically for BurstBuffer)
    """

    temporary_directory = get_buffer_directory()


    def __init__(self, obj, mode='TRANSIENT'):
        self._buffer_file = tempfile.NamedTemporaryFile(prefix='toast-cache-',
            dir=FsBuffer.temporary_directory, suffix='.cache', delete=False)
        self._buffer_mode = mode
        self.unload(obj)


    def __del__(self):
        self.close()
        if self._buffer_mode == 'TRANSIENT':
            os.remove(self._buffer_file.name)


    def open(self, fmode='w+b'):
        self._buffer_file = open(self._buffer_file.name, fmode)


    def close(self):
        self._buffer_file.close()


    def load(self):
        #print('Load from buffer "{}"...{}'.format(self._buffer_file.name, obj))
        try:
            obj = np.fromfile(self._buffer_file.name, dtype=self._buffer_type)
        except:
            print('File {} invalid (ndarray type: {})'.format(self._buffer_file.name,
                self._buffer_type))
            raise
        return obj.reshape(self._buffer_shape)


    def update(self, obj):
        self.unload(obj)


    def unload(self, obj):
        #print('Unloading to buffer "{}"...{}'.format(self._buffer_file.name, obj))
        self._buffer_type = obj.dtype
        self._buffer_shape = obj.shape
        self.open()
        obj.tofile(self._buffer_file.name)
        self.close()


    def free(self):
        self._buffer_obj = None


class Cache(object):
    """
    Timestream data cache with explicit memory management.

    Args:
        pymem (bool): if True, use python memory rather than external
            allocations in C.  Only used for testing.
    """

    use_fsbuffer = get_use_fsbuffer()

    def __init__(self, pymem=False):
        self._pymem = pymem
        self._refs = {}
        self._aliases = {}
        if Cache.use_fsbuffer:
            self._buffers = {}


    def __del__(self):
        # free all buffers at destruction time
        self._aliases.clear()
        if not self._pymem:
            keylist = list(self._refs.keys())
            for k in keylist:
                #gc.collect()
                #referrers = gc.get_referrers(self._refs[k])
                #print("__del__ {} referrers for {} are: ".format(len(referrers), k), referrers)
                #print("__del__ refcount for {} is ".format(k), sys.getrefcount(self._refs[k]) )
                if sys.getrefcount(self._refs[k]) > 2:
                    warnings.warn("Cache object {} has external references and will not be freed.".format(k), RuntimeWarning)
                del self._refs[k]
                if Cache.use_fsbuffer:
                    del self._buffers[k]
        self._refs.clear()
        if Cache.use_fsbuffer:
            self._buffers.clear()


    def clear(self, pattern=None, remove_fsbuffer=True):
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
            if not self._pymem:
                keylist = list(self._refs.keys())
                for k in keylist:
                    #gc.collect()
                    #referrers = gc.get_referrers(self._refs[k])
                    #print("clear {} referrers for {} are: ".format(len(referrers), k), referrers)
                    #print("clear refcount for {} is ".format(k), sys.getrefcount(self._refs[k]) )
                    if sys.getrefcount(self._refs[k]) > 2:
                        warnings.warn("Cache object {} has external references and will not be freed.".format(k), RuntimeWarning)
                    del self._refs[k]
                    if Cache.use_fsbuffer and remove_fsbuffer:
                        del self._buffers[k]
            self._refs.clear()
            if Cache.use_fsbuffer and remove_fsbuffer:
                self._buffers.clear()
        else:
            pat = re.compile(pattern)
            names = []
            for n, r in self._refs.items():
                mat = pat.match(n)
                if mat is not None:
                    names.append(n)
                del r
            for n in names:
                self.destroy(n, remove_fsbuffer=remove_fsbuffer)
        return


    def free(self, pattern=None):
        """
        Clear one or more buffers but keep the fsbuffer.

        Args:
            pattern (str): a regular expression to match against the buffer
                names when determining what should be cleared.  If None,
                then all buffers are cleared.
        """
        # only call clear with remove_fsbuffer=False if using FsBuffer
        if Cache.use_fsbuffer:
            self.clear(pattern, remove_fsbuffer=False)
        # if not using FsBuffer then calling clear would actually remove
        return


    def create(self, name, type, shape, use_fsbuffer=True):
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
            if Cache.use_fsbuffer and use_fsbuffer:
                self._buffers[name] = FsBuffer(self._refs[name])
        else:
            flatsize = 1
            for s in range(len(shape)):
                if shape[s] <= 0:
                    raise RuntimeError("Cache object must have non-zero sizes in all dimensions")
                flatsize *= shape[s]
            self._refs[name] = np.asarray( ToastBuffer(int(flatsize), numpy2toast(type)) ).reshape(shape)
            if Cache.use_fsbuffer and use_fsbuffer:
                self._buffers[name] = FsBuffer(self._refs[name])

        return self._refs[name]


    def put(self, name, data, replace=False, use_fsbuffer=True):
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
            self.destroy(name, remove_fsbuffer=True)
        else:
            mydata = data

        # never use_fsbuffer bc over-ridden below
        ref = self.create(name, mydata.dtype, mydata.shape, use_fsbuffer=False)
        ref[:] = mydata
        if Cache.use_fsbuffer and use_fsbuffer:
            if name in self._buffers.keys():
                del self._buffers[name]
            self._buffers[name] = FsBuffer(ref)

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


    def destroy(self, name, remove_fsbuffer=True):
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

        if name not in self._refs.keys():
            if name in self._buffers.keys() and remove_fsbuffer:
                del self._buffers[name]
                return
            else:
                print('Reference Keys: {}'.format(self._refs.keys()))
                print('Buffer Keys: {}'.format(self._buffers.keys()))
                raise RuntimeError("Data buffer {} does not exist".format(name))

        # Remove aliases to the buffer
        if Cache.use_fsbuffer and remove_fsbuffer:
            aliases_to_remove = []
            for key, value in self._aliases.items():
                if value == name:
                    aliases_to_remove.append( key )
            for key in aliases_to_remove:
                del self._aliases[key]

        # Remove actual buffer
        if not self._pymem:
            # print("destroy referents for {} are ".format(name), gc.get_referents(self._refs[name]))
            # print("destroy referrers for {} are ".format(name), gc.get_referrers(self._refs[name]))
            # print("destroy refcount for {} is ".format(name), sys.getrefcount(self._refs[name]) )
            if sys.getrefcount(self._refs[name]) > 2:
                warnings.warn("Cache object {} has external references and will not be freed.".format(name), RuntimeWarning)
        if Cache.use_fsbuffer and not remove_fsbuffer:
            self._buffers[name].update(self._refs[name])
        del self._refs[name]
        if Cache.use_fsbuffer and remove_fsbuffer:
            del self._buffers[name]

        return


    def exists(self, name, return_ref=False):
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
        if name in self._refs.keys():
            check = True
        elif name in self._aliases.keys():
            check = True
        elif Cache.use_fsbuffer and name in self._buffers.keys():
            check = True

        if not return_ref:
            return check
        else:
            if not check and not name in self._buffers.keys():
                return None
            elif not check and Cache.use_fsbuffer and name in self._buffers.keys():
                return self._buffers[name].load()
            else:
                ref = None
                if name in self._refs.keys():
                    ref = self._refs[name]
                    if Cache.use_fsbuffer and ref is None:
                        ref = self._buffers[name].load()
                elif name in self._aliases.keys():
                    ref = self._refs[self._aliases[name]]
                elif Cache.use_fsbuffer and name in self._buffers.keys():
                    ref = self._buffers[name].load()
                return ref


    def reference(self, name, free_alloc=False):
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
        if Cache.use_fsbuffer and free_alloc and name in self._refs.keys():
            self.destroy(name, remove_fsbuffer=False)
        return ref


    def clone(self, name):
        """
        Return a clone (copy) of a numpy array pointing to the buffer.

        The returned array will wrap a pointer to the raw buffer, but will
        not claim ownership.  When the numpy array is garbage collected, it
        will NOT attempt to free the memory (you must manually use the
        destroy method).

        Args:
            name (str): the name of the buffer to return.

        Returns:
            (array): a numpy array wrapping the raw data buffer.
        """
        if Cache.use_fsbuffer and name in self._buffers.keys():
            # always a copy
            return self._buffers[name].load()
        ref = self.exists(name, return_ref=True)
        if ref is None:
            raise RuntimeError("Data buffer (nor alias) {} does not exist".format(name))
        return ref.copy()


    def keys(self):
        """
        Return a list of all the keys in the cache.

        Args:

        Returns:
            (list): List of key strings.
        """

        return list(self._refs.keys())


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
