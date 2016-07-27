# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import re
import numpy as np

from ._cache import _alloc, _free


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


    def __del__(self):
        # free all buffers at destruction time
        if not self._pymem:
            for n, r in self._refs.items():
                _free(r)
        self._refs.clear()
        self._aliases.clear()


    def clear(self, pattern=None):
        """
        Clear one or more buffers.

        Args:
            pattern (str): a regular expression to match against the buffer
                names when determining what should be cleared.  If None,
                then all buffers are cleared.
        """
        if pattern is None:
            # free all buffers
            if not self._pymem:
                for n, r in self._refs.items():
                    _free(r)
            self._refs.clear()
            self._aliases.clear()
        else:
            pat = re.compile(pattern)
            names = []
            for n, r in self._refs.items():
                mat = pat.match(n)
                if mat is not None:
                    names.append(n)
            for n in names:
                self.destroy(n)
        return


    def create(self, name, type, shape):
        """
        Create a named data buffer of the given type and shape.

        Args:
            name (str): the name to assign to the buffer.
            type (numpy.dtype): one of the supported numpy types.
            shape (tuple): a tuple containing the shape of the buffer.
        """

        if self.exists(name):
            raise RuntimeError("Data buffer or alias {} already exists".format(name))
        
        if self._pymem:
            self._refs[name] = np.zeros(shape, dtype=type)
        else:
            dims = np.asarray(shape, dtype=np.uint64)
            self._refs[name] = _alloc(dims, type).reshape(shape)
            
        return self._refs[name]


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

        if self.exists(name) and replace:
            ref = self.reference(name)
            if data is ref:
                return ref
            # Destroy the existing cache object but first make a copy
            # of the supplied data in case it is a view of a subset
            # of the cache data.
            mydata = data.copy()
            self.destroy(name)
        else:
            mydata = data

        ref = self.create(name, mydata.dtype, mydata.shape)
        ref[:] = mydata
        
        return ref


    def add_alias(self, alias, name):
        """
        Add an alias to a name that already exists in the cache.

        Args:
            alias (str): alias to create
            name (str): an existing key in the cache
        """

        if name not in self._refs.keys():
            raise RuntimeError("Data buffer {} does not exist for alias {}".format(name, alias))

        if alias in self._refs.keys():
            raise RuntimeError("Proposed alias {} would shadow existing buffer.".format(alias))

        self._aliases[alias] = name


    def destroy(self, name):
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
            raise RuntimeError("Data buffer {} does not exist".format(name))

        # Remove aliases to the buffer
        aliases_to_remove = []
        for key, value in self._aliases.items():
            if value == name:
                aliases_to_remove.append( key )
        for key in aliases_to_remove:
            del self._aliases[key]

        # Remove actual buffer
        if not self._pymem:
            _free(self._refs[name])
        del self._refs[name]
        return


    def exists(self, name, return_ref=False):
        """
        Check whether a buffer exists.

        Args:
            name (str): the name of the buffer to search for.

        Returns:
            (array): a numpy array wrapping the raw data buffer or None if it does not exist.
        """
        ref = None
        if name in self._refs.keys():
            ref = self._refs[name]
        elif name in self._aliases.keys():
            ref = self._refs[self._aliases[name]]

        if return_ref:
            return ref
        else:
            return ref is not None


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

    
    def keys(self):
        """
        Return a list of all the keys in the cache.

        Args:

        Returns:
            (list): List of key strings.
        """

        return self._refs.keys()


    def aliases(self):
        """
        Return a dictionary of all the aliases to keys in the cache.

        Args:

        Returns:
            (dict): Dictionary of aliases.
        """

        return self._aliases.copy()


    def report(self):
        """
        Report memory usage.
        """

        print('Cache memory usage:')
        
        tot = 0
        for key in self.keys():
            ref = self.reference(key)
            sz = ref.nbytes
            tot += sz
            print(' - {:25} {:5.2f} MB'.format(key,sz/2**20))
                  
        print(' {:27} {:5.2f} MB'.format('TOTAL',tot/2**20))
            
