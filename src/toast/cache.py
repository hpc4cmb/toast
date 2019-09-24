# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re
import ctypes
import numpy as np

from .utils import (
    Logger,
    AlignedI8,
    AlignedU8,
    AlignedI16,
    AlignedU16,
    AlignedI32,
    AlignedU32,
    AlignedI64,
    AlignedU64,
    AlignedF32,
    AlignedF64,
)


class Cache(object):
    """Data cache with explicit memory management.

    This class acts as a dictionary of named arrays.  Each array may be
    multi-dimensional.

    Args:
        pymem (bool): if True, use python memory rather than external
            allocations in C.  Only used for testing.
    """

    def __init__(self, pymem=False):
        self._pymem = pymem
        self._buffers = dict()
        self._dtypes = dict()
        self._shapes = dict()
        self._aliases = dict()

    def clear(self, pattern=None):
        """Clear one or more buffers.

        Args:
            pattern (str): a regular expression to match against the buffer
                names when determining what should be cleared.  If None,
                then all buffers are cleared.

        Returns:
            None

        """
        if pattern is None:
            # free all buffers
            self._aliases.clear()
            self._buffers.clear()
            self._dtypes.clear()
            self._shapes.clear()
        else:
            pat = re.compile(pattern)
            names = list(self._buffers.keys())
            matching = list()
            for n in names:
                mat = pat.match(n)
                if mat is not None:
                    matching.append(n)
            for n in matching:
                self.destroy(n)
        return

    def create(self, name, type, shape):
        """Create a named data buffer of the given type and shape.

        Args:
            name (str): the name to assign to the buffer.
            type (numpy.dtype): one of the supported numpy types.
            shape (tuple): a tuple containing the shape of the buffer.

        Returns:
            (array): a reference to the allocated array.

        """
        log = Logger.get()
        if name is None:
            raise ValueError("Cache name cannot be None")
        if type is None:
            raise ValueError("Cache type cannot be None")
        if shape is None:
            raise ValueError("Cache shape cannot be None")
        if self.exists(name):
            raise RuntimeError("Data buffer or alias {} already exists".format(name))
        ttype = np.dtype(type)
        flatshape = 1
        for dim in shape:
            flatshape *= dim
        if self._pymem:
            self._buffers[name] = np.zeros(flatshape, dtype=ttype)
        else:
            if ttype.char == "b":
                self._buffers[name] = AlignedI8.zeros(flatshape)
            elif ttype.char == "B":
                self._buffers[name] = AlignedU8.zeros(flatshape)
            elif ttype.char == "h":
                self._buffers[name] = AlignedI16.zeros(flatshape)
            elif ttype.char == "H":
                self._buffers[name] = AlignedU16.zeros(flatshape)
            elif ttype.char == "i":
                self._buffers[name] = AlignedI32.zeros(flatshape)
            elif ttype.char == "I":
                self._buffers[name] = AlignedU32.zeros(flatshape)
            elif (ttype.char == "q") or (ttype.char == "l"):
                self._buffers[name] = AlignedI64.zeros(flatshape)
            elif (ttype.char == "Q") or (ttype.char == "L"):
                self._buffers[name] = AlignedU64.zeros(flatshape)
            elif ttype.char == "f":
                self._buffers[name] = AlignedF32.zeros(flatshape)
            elif ttype.char == "d":
                self._buffers[name] = AlignedF64.zeros(flatshape)
            else:
                msg = "Unsupported data typecode '{}'".format(ttype.char)
                log.error(msg)
                raise ValueError(msg)
        self._dtypes[name] = ttype
        self._shapes[name] = shape
        if self._pymem:
            return self._buffers[name].reshape(self._shapes[name])
        else:
            return self._buffers[name].array().reshape(self._shapes[name])
        return self._buffers[name]

    def put(self, name, data, replace=False):
        """Create a named data buffer to hold the provided data.

        If replace is True, existing buffer of the same name is first
        destroyed. If replace is True and the name is an alias, it is
        promoted to a new data buffer.

        Args:
            name (str): the name to assign to the buffer.
            data (numpy.ndarray): Numpy array
            replace (bool): Overwrite any existing keys

        Returns:
            (array): a numpy array wrapping the raw data buffer.

        """
        if name is None:
            raise ValueError("Cache name cannot be None")
        indata = data
        if self.exists(name):
            # This buffer already exists. Is the input data buffer actually
            # the same memory as the buffer already stored?  If so, just
            # return a new reference.
            realname = name
            if name in self._aliases:
                realname = self._aliases[name]
            addr = None
            if self._pymem:
                p_ref = self._buffers[realname].ctypes.data_as(ctypes.c_void_p).value
            else:
                p_ref = self._buffers[realname].address()

            p_data = data.ctypes.data_as(ctypes.c_void_p).value
            # print("p_ref = {}, p_data = {}".format(p_ref, p_data), flush=True)
            if (
                (p_ref == p_data)
                and (self._shapes[realname] == data.shape)
                and (self._dtypes[realname] == data.dtype)
            ):
                return self.reference(realname)
            if not replace:
                raise RuntimeError(
                    "Cache buffer named {} already exists "
                    "and replace is False.".format(name)
                )
            # At this point we have an existing memory buffer or alias with
            # the same name, and which is not identical to the input.  If this
            # is an alias, just delete it.
            if name in self._aliases:
                del self._aliases[name]
            else:
                # This existing data is not an alias.  However, the input
                # might be a view into this existing memory.  Before deleting
                # the existing data, we copy the input just in case.
                indata = np.array(data)
                self.destroy(name)

        # Now create the new buffer and copy in the data.
        ref = self.create(name, indata.dtype, indata.shape)
        np.copyto(ref, indata)
        return ref

    def add_alias(self, alias, name):
        """Add an alias to a name that already exists in the cache.

        Args:
            alias (str): alias to create
            name (str): an existing key in the cache

        Returns:
            None

        """
        if alias is None or name is None:
            raise ValueError("Cache name or alias cannot be None")
        names = list(self._buffers.keys())
        if name not in names:
            raise RuntimeError(
                "Data buffer {} does not exist for alias {}".format(name, alias)
            )
        if alias in names:
            raise RuntimeError(
                "Proposed alias {} would shadow existing buffer.".format(alias)
            )
        self._aliases[alias] = name
        return

    def destroy(self, name):
        """Deallocate the specified buffer.

        Only call this if all numpy arrays that reference the memory
        are out of use.  If the specified name is an alias, then the alias
        is simply deleted.  If the specified name is an actual buffer, then
        all aliases pointing to that buffer are also deleted.

        Args:
            name (str): the name of the buffer or alias to destroy.

        Returns:
            None

        """
        if name in self._aliases.keys():
            # Name is an alias. Do not remove the buffer
            del self._aliases[name]
            return
        names = list(self._buffers.keys())
        if name not in names:
            raise RuntimeError("Data buffer {} does not exist".format(name))

        # Remove aliases to the buffer
        aliases_to_remove = []
        for key, value in self._aliases.items():
            if value == name:
                aliases_to_remove.append(key)
        for key in aliases_to_remove:
            del self._aliases[key]

        # Forcibly resize this buffer to length zero
        if not self._pymem:
            self._buffers[name].clear()

        # Remove actual buffer
        del self._buffers[name]
        del self._dtypes[name]
        del self._shapes[name]
        return

    def exists(self, name):
        """Check whether a buffer exists.

        Args:
            name (str): the name of the buffer to search for.

        Returns:
            (bool):  True if a buffer or alias exists with the given name.

        """
        if name in self._aliases:
            # We have an alias with this name, so it exists.
            return True
        names = list(self._buffers.keys())
        if name in names:
            return True
        return False

    def reference(self, name):
        """Return a numpy array pointing to the buffer.

        The returned array will wrap a pointer to the raw buffer, but will
        not claim ownership.  When the numpy array is garbage collected, it
        will NOT attempt to free the memory (you must manually use the
        destroy method).

        Args:
            name (str): the name of the buffer to return.

        Returns:
            (array): a numpy array wrapping the raw data buffer.

        """
        # First check that it exists
        if not self.exists(name):
            raise RuntimeError("Data buffer (nor alias) {} does not exist".format(name))
        realname = name
        if name in self._aliases:
            # This is an alias
            realname = self._aliases[name]
        if self._pymem:
            return self._buffers[realname].reshape(self._shapes[realname])
        else:
            return self._buffers[realname].array().reshape(self._shapes[realname])

    def keys(self):
        """Return a list of all the keys in the cache.

        Returns:
            (list): List of key strings.

        """
        return sorted(list(self._buffers.keys()))

    def aliases(self):
        """Return a dictionary of all the aliases to keys in the cache.

        Returns:
            (dict): Dictionary of aliases.

        """
        return self._aliases.copy()

    def report(self, silent=False):
        """Report memory usage.

        Args:
            silent (bool):  Count and return the memory without printing.

        Returns:
            (int):  Amount of allocated memory in bytes

        """
        log = Logger.get()
        if not silent:
            log.info("Cache memory usage:")
        tot = 0
        for key in self.keys():
            ref = self.reference(key)
            sz = ref.nbytes
            del ref
            tot += sz
            if not silent:
                log.info(" - {:25} {:5.2f} MB".format(key, sz / 2 ** 20))
        if not silent:
            log.info(" {:27} {:5.2f} MB".format("TOTAL", tot / 2 ** 20))
        return tot
