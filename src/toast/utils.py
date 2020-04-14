# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import gc

from functools import wraps

import numpy as np

from ._libtoast import Environment, Timer, GlobalTimers, Logger

from ._libtoast import (
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

from ._libtoast import vsin, vcos, vsincos, vatan2, vsqrt, vrsqrt, vexp, vlog

from ._libtoast import (
    vfast_sin,
    vfast_cos,
    vfast_sincos,
    vfast_atan2,
    vfast_sqrt,
    vfast_rsqrt,
    vfast_exp,
    vfast_log,
    vfast_erfinv,
)


def astropy_override():
    """Override astropy locations.

    The astropy package assumes that software is able to download
    arbitrary data files at any time and to put those in a user's home
    directory.  Both of these assumptions may be bad on a distributed system.
    Astropy also forcibly loads a config file from the home directory which
    may have a performance impact when done concurrently from thousands of
    processes.

    This function checks for the "TOAST_ASTROPY_HOME" environment variable.
    If set it gets the astropy.config.set_temp_config and set_temp_cache
    decorators and returns them.  Otherwise it returns empty decorators.
    These decorators can be used when calling external functions that
    use astropy in order to force them to relocate astropy files to an
    alternate location, like a fast scratch disk or a ramdisk.

    Returns:
        (tuple):  the config and cache decorators.

    """
    if "TOAST_ASTROPY_HOME" in os.environ:
        from astropy.config import set_temp_config, set_temp_cache
        return (
            set_temp_config(os.environ["TOAST_ASTROPY_HOME"]),
            set_temp_cache(os.environ["TOAST_ASTROPY_HOME"])
        )
    else:
        @wraps(f)
        def config_dec(*args, **kwargs):
            result = f(*args, **kwargs)
            return result

        @wraps(f)
        def cache_dec(*args, **kwargs):
            result = f(*args, **kwargs)
            return result

        return (config_dec, cache_dec)



numba_threading_layer = "NA"


def set_numba_threading():
    """Set the numba threading layer.

    For parallel numba jit blocks, the backend threading layer is selected at runtime
    based on an order set inside the numba package.  We would like to change the
    order of selection to prefer one of the thread-based backends (omp or tbb).  We also
    set the maximum number of threads used by numba to be the same as the number of
    threads used by TOAST.  Since TOAST does not use numba, it means that there will
    be a consistent maximum number of threads in use at all times and no
    oversubscription.

    Args:
        None

    Returns:
        None

    """
    global numba_threading_layer
    # Get the number of threads used by TOAST at runtime.
    env = Environment.get()
    log = Logger.get()
    toastthreads = env.max_threads()

    # Allow user-override of numba threads
    if "NUMBA_NUM_THREADS" in os.environ:
        if int(os.environ["NUMBA_NUM_THREADS"]) < toastthreads:
            toastthreads = int(os.environ["NUMBA_NUM_THREADS"])

    rank = 0
    if env.use_mpi():
        from .mpi import MPI

        rank = MPI.COMM_WORLD.rank

    threading = "default"
    try:
        from numba.npyufunc import omppool

        threading = "omp"
        if rank == 0:
            log.debug("Numba has OpenMP threading support")
    except ImportError:
        # no OpenMP support
        if rank == 0:
            log.debug("Numba does not support OpenMP")
        try:
            from numba.npyufunc import tbbpool

            threading = "tbb"
            if rank == 0:
                log.debug("Numba has TBB threading support")
        except ImportError:
            # no TBB
            if rank == 0:
                log.debug("Numba does not support TBB")
    try:
        from numba import vectorize, config, threading_layer

        # Set threading layer and number of threads.  Note that this still
        # does not always work.  The conf structure is repopulated from the
        # environment on every compilation if any of the NUMBA_* variables
        # have changed.
        config.THREADING_LAYER = threading
        config.NUMBA_DEFAULT_NUM_THREADS = toastthreads
        config.NUMBA_NUM_THREADS = toastthreads
        os.environ["NUMBA_THREADING_LAYER"] = threading
        os.environ["NUMBA_DEFAULT_NUM_THREADS"] = "{:d}".format(toastthreads)
        os.environ["NUMBA_NUM_THREADS"] = "{:d}".format(toastthreads)

        # In order to get numba to actually select a threading layer, we must
        # trigger compilation of a parallel function.
        @vectorize("float64(float64)", target="parallel")
        def force_thread_launch(x):
            return x + 1

        force_thread_launch(np.zeros(1))

        # Log the layer that was selected
        numba_threading_layer = threading_layer()
        if rank == 0:
            log.debug("Numba threading layer set to {}".format(numba_threading_layer))
            log.debug(
                "Numba max threads now forced to {}".format(config.NUMBA_NUM_THREADS)
            )
    except ImportError:
        # Numba not available at all
        if rank == 0:
            log.debug("Cannot import numba- ignoring threading layer.")


try:
    import psutil

    def memreport(msg="", comm=None):
        """ Gather and report the amount of allocated, free and swapped system memory
        """
        if psutil is None:
            return
        vmem = psutil.virtual_memory()._asdict()
        gc.collect()
        vmem2 = psutil.virtual_memory()._asdict()
        memstr = "Memory usage {}\n".format(msg)
        for key, value in vmem.items():
            value2 = vmem2[key]
            if comm is None:
                vlist = [value]
                vlist2 = [value2]
            else:
                vlist = comm.gather(value)
                vlist2 = comm.gather(value2)
            if comm is None or comm.rank == 0:
                vlist = np.array(vlist, dtype=np.float64)
                vlist2 = np.array(vlist2, dtype=np.float64)
                if key != "percent":
                    # From bytes to better units
                    if np.amax(vlist) < 2 ** 20:
                        vlist /= 2 ** 10
                        vlist2 /= 2 ** 10
                        unit = "kB"
                    elif np.amax(vlist) < 2 ** 30:
                        vlist /= 2 ** 20
                        vlist2 /= 2 ** 20
                        unit = "MB"
                    else:
                        vlist /= 2 ** 30
                        vlist2 /= 2 ** 30
                        unit = "GB"
                else:
                    unit = "% "
                if comm is None or comm.size == 1:
                    memstr += "{:>12} : {:8.3f} {}\n".format(key, vlist[0], unit)
                    if np.abs(vlist2[0] - vlist[0]) / vlist[0] > 1e-3:
                        memstr += "{:>12} : {:8.3f} {} (after GC)\n".format(
                            key, vlist2[0], unit
                        )
                else:
                    med1 = np.median(vlist)
                    memstr += (
                        "{:>12} : {:8.3f} {}  < {:8.3f} +- {:8.3f} {}  "
                        "< {:8.3f} {}\n".format(
                            key,
                            np.amin(vlist),
                            unit,
                            med1,
                            np.std(vlist),
                            unit,
                            np.amax(vlist),
                            unit,
                        )
                    )
                    med2 = np.median(vlist2)
                    if np.abs(med2 - med1) / med1 > 1e-3:
                        memstr += (
                            "{:>12} : {:8.3f} {}  < {:8.3f} +- {:8.3f} {}  "
                            "< {:8.3f} {} (after GC)\n".format(
                                key,
                                np.amin(vlist2),
                                unit,
                                med2,
                                np.std(vlist2),
                                unit,
                                np.amax(vlist2),
                                unit,
                            )
                        )
        if comm is None or comm.rank == 0:
            print(memstr, flush=True)
        if comm is not None:
            comm.Barrier()
        return


except ImportError:

    def memreport(msg="", comm=None):
        return


def object_ndim(x):
    """Get the number of dimension of an object.

    Scalars return 0.  Numpy arrays return their actual ndim value.
    Objects that support the buffer protocol return the ndim value of the
    corresponding memoryview.  Nested lists are traversed to compute the
    effective number of dimensions.

    Args:
        x (object): some stuff.

    Returns:
        (int): the number of dimensions of the stuff.

    """
    try:
        nd = x.ndim
        return nd
    except AttributeError:
        # Not a numpy array...
        try:
            view = memoryview(x)
            nd = view.ndim
            return nd
        except TypeError:
            # Does not support buffer protocol...
            try:
                lg = len(x)
                # It's a list!
                nd = 1
                cur = x[0]
                try:
                    lg = len(cur)
                    nd += 1
                    cur = cur[0]
                    try:
                        lg = len(cur)
                        nd += 1
                        # Using lists of more than 3 dimensions (rather than
                        # a numpy array) is kind of crazy...
                    except TypeError:
                        pass
                except TypeError:
                    pass
                return nd
            except TypeError:
                # Must be a scalar.
                return 0


def ensure_buffer_i64(data):
    """Return a flattened object that supports the buffer protocol.

    Numpy arrays and objects supporting the buffer protocol are flattened and
    copied to contiguous memory if needed.  Scalars and lists are converted
    to numpy arrays.

    Args:
        data (object):  A scalar or iterable.

    Returns:
        (array):  The input unmodified or converted to an array.

    """
    return np.ascontiguousarray(data, dtype=np.int64).flatten()


def ensure_buffer_f64(data):
    """Return a flattened object that supports the buffer protocol.

    Numpy arrays and objects supporting the buffer protocol are flattened and
    copied to contiguous memory if needed.  Scalars and lists are converted
    to numpy arrays.

    Args:
        data (object):  A scalar or iterable.

    Returns:
        (array):  The input unmodified or converted to an array.

    """
    return np.ascontiguousarray(data, dtype=np.float64).flatten()
    # if isinstance(data, np.ndarray):
    #     print("ensure: found numpy array, shape=", data.shape, flush=True)
    #     return np.ascontiguousarray(data.flatten(), dtype=np.float64)
    # try:
    #     view = memoryview(data)
    #     if view.ndim == 0:
    #         # A single element...
    #         print("ensure: converting scalar buffer", flush=True)
    #         return np.array([view], dtype=np.float64)
    #     elif (not view.c_contiguous) or (view.ndim != 1):
    #         print("ensure: found non-contiguous memory view shape=", view.shape, flush=True)
    #         return np.ascontiguousarray(view, dtype=np.float64)
    #     else:
    #         print("ensure: returning original data shape=", view.shape, flush=True)
    #         return data
    # except TypeError:
    #     # Does not support buffer protocol
    #     print("ensure: converting non-buffer object ", data, flush=True)
    #     return np.ascontiguousarray(data, dtype=np.float64)
