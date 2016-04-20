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

    """

    def __init__(self):
        self._refs = {}


    def __del__(self):
        # free all buffers at destruction time
        for n, r in self._refs.items():
            _free(r)
        self._refs.clear()


    def clear(self, pattern=None):
        if pattern is None:
            # free all buffers
            for n, r in self._refs.items():
                _free(r)
            self._refs.clear()
        else:
            pat = re.compile(pattern)
            names = []
            for n, r in self._refs.items():
                mat = pat.match(n)
                if mat is not None:
                    names.append(n)
            for n in names:
                _free(self._refs[n])
                del self._refs[n]
        return


    def create(self, name, type, shape):
        """
        Create a named data buffer of the give type and shape.
        """
        dims = np.asarray(shape, dtype=np.uint64)
        self._refs[name] = _alloc(dims, type).reshape(shape)
        return


    def destroy(self, name):
        """
        Deallocate the specified buffer.  Only call this if all numpy arrays
        that reference the memory are out of use.
        """
        if name not in self._refs.keys():
            raise RuntimeError("Data buffer {} does not exist".format(name))
        _free(self._refs[name])
        del self._refs[name]
        return


    def exists(self, name):
        result = (name in self._refs.keys())
        return result


    def reference(self, name):
        """
        Return a numpy array that contains a reference to the specified data
        buffer.
        """
        if name not in self._refs.keys():
            raise RuntimeError("Data buffer {} does not exist".format(name))
        return self._refs[name]

