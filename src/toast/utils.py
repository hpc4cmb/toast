# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import gc

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
    # Get the number of threads used by TOAST at runtime.
    env = Environment.get()
    toastthreads = env.max_threads()

    log = Logger.get()
    threading = "default"
    try:
        from numba.npyufunc import omppool

        threading = "omp"
    except ImportError:
        # no OpenMP support
        log.debug("Numba: No OpenMP support")
        try:
            from numba.npyufunc import tbbpool

            threading = "tbb"
        except ImportError:
            # no TBB
            log.debug("Numba: No TBB support")
            pass
    try:
        from numba import vectorize, threading_layer

        # Set threading layer and number of threads by way of
        # the environment.
        os.environ["NUMBA_THREADING_LAYER"] = threading
        os.environ["NUMBA_NUM_THREADS"] = "{:d}".format(toastthreads)
        log.debug("Attempting to set numba threading layer to '{}'".format(threading))

        # In order to get numba to actually select a threading layer, we must
        # trigger compilation of a parallel function.
        @vectorize("float64(float64)", target="parallel")
        def force_thread_launch(x):
            return x + 1

        force_thread_launch(np.zeros(1))

        # Log the layer that was selected
        layer = threading_layer()
        log.info("Numba threading layer set to:  {}".format(layer))
        log.info("Numba configured to use {} threads".format(toastthreads))

    except ImportError:
        # Numba not available at all
        log.info("Numba not available:  skipping threading layer selection")

try:
    import psutil

    def memreport(comm=None, msg=""):
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

    def memreport(comm=None, msg=""):
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
