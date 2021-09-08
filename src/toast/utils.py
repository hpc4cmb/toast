# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import gc
import hashlib
import warnings

import importlib

import datetime

import numpy as np

from ._libtoast import Environment, Timer, GlobalTimers

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

from .mpi import MPI, use_mpi

from ._libtoast import Logger


def _create_log_rank(level):
    def log_rank(self, msg, comm=None, rank=0, timer=None):
        """Log a message on one process with optional timing.

        A common use case when logging from high-level code is to print a message on
        only a single process, and also to include some timing information.

        If a timer is specified, then it must be specified on all processes.  In this
        case the method is collective, and a barrier() is used to ensure accurate
        timing results.

        Args:
            msg (str):  The log message (only used on specified rank).
            comm (MPI.Comm):  The communicator, or None.
            rank (int):  The rank of the process that should do the logging.
            timer (Timer):  The optional timer.

        """
        log = Logger.get()
        my_rank = 0
        if comm is not None:
            my_rank = comm.rank

        if comm is not None and timer is not None:
            comm.barrier()

        if timer is not None:
            timer.stop()

        if my_rank == rank:
            if timer is not None:
                msg = f"{msg} {timer.seconds():0.2f} s"
            if level == "VERBOSE":
                log.verbose(msg)
            elif level == "DEBUG":
                log.debug(msg)
            elif level == "INFO":
                log.info(msg)
            elif level == "WARNING":
                log.warning(msg)
            elif level == "ERROR":
                log.error(msg)
            elif level == "CRITICAL":
                log.critical(msg)
            else:
                raise RuntimeError(f"invalid log level {level}")

        if timer is not None:
            timer.clear()
            timer.start()

    return log_rank


# Add rank logging for all log levels
Logger.verbose_rank = _create_log_rank("VERBOSE")
Logger.debug_rank = _create_log_rank("DEBUG")
Logger.info_rank = _create_log_rank("INFO")
Logger.warning_rank = _create_log_rank("WARNING")
Logger.error_rank = _create_log_rank("ERROR")
Logger.critical_rank = _create_log_rank("CRITICAL")


# This function sets the numba threading layer to (hopefully) be compatible with TOAST.
# The TOAST threading concurrency is used to attempt to set the numba threading.  We
# try to use the OpenMP backend for numba and then TBB.  The "workqueue" backend (which
# is process based).  May not be compatible with all systems, so we use that as a
# last resort.  This function should be called by any operators that use numba.

numba_threading_layer = None


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
    if numba_threading_layer is not None:
        # Already set.
        return

    # Get the number of threads used by TOAST at runtime.
    env = Environment.get()
    log = Logger.get()
    toastthreads = env.max_threads()
    print("max toast threads = ", toastthreads, flush=True)

    rank = 0
    if use_mpi:
        rank = MPI.COMM_WORLD.rank

    threading = "default"
    have_numba_omp = False
    try:
        # New style package layout
        from numba.np.ufunc import omppool

        have_numba_omp = True
        if rank == 0:
            log.debug("Numba has OpenMP threading support")
    except ImportError:
        try:
            # Old style
            from numba.npyufunc import omppool

            have_numba_omp = True
            if rank == 0:
                log.debug("Numba has OpenMP threading support")
        except ImportError:
            # no OpenMP support
            if rank == 0:
                log.debug("Numba does not support OpenMP")
    have_numba_tbb = False
    try:
        # New style package layout
        from numba.np.ufunc import tbbpool

        have_numba_tbb = True
        if rank == 0:
            log.debug("Numba has TBB threading support")
    except ImportError:
        try:
            # Old style
            from numba.npyufunc import tbbpool

            have_numba_tbb = True
            if rank == 0:
                log.debug("Numba has TBB threading support")
        except ImportError:
            # no TBB
            if rank == 0:
                log.debug("Numba does not support TBB")

    # Prefer OMP backend
    if have_numba_omp:
        threading = "omp"
    elif have_numba_tbb:
        threading = "tbb"

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


def astropy_control(max_future=None, offline=False, node_local=False):
    """This function attempts to trigger any astropy downloads.

    The astropy package will automatically download external data files on demand.
    This can be a problem if multiple processes on a distributed system are using the
    same astropy installation.  This function will trigger IERS downloads in a
    controlled way on one process (or one per node if astropy is node-local).

    When requesting Alt / Az coordinate system transforms, times outside of the IERS
    data range will produce a warning on every process.  If the max_future time is set,
    we check if that will trigger a warning, print the warning once, and then disable
    it to avoid spewing the warning on every process of a parallel job.

    Args:
        None

    Returns:
        None

    """
    from astropy.utils import iers

    log = Logger.get()

    rank = 0
    if use_mpi:
        rank = MPI.COMM_WORLD.rank

    # If auto_download is False, the bundled IERS-B table is always used, even if a
    # previously downloaded IERS-A table exists.
    if offline:
        iers.conf.auto_download = False
        log.warning(
            "Disabling downloaded astropy IERS- coordinate transforms will be less accurate"
        )
    else:
        iers.conf.auto_download = True

    # Check future time range

    now_time = datetime.datetime.now(tz=datetime.timezone.utc)
    if max_future is not None:
        if max_future > now_time + datetime.timedelta(days=365):
            msg = f"Maximum future time {max_future} is more than one year in future.\n"
            msg += f"Coordinate transforms using IERS will be less accurate."
            if rank == 0:
                log.warning(msg)

    # Disable future warnings
    # log.info("Disabling astropy IERSRangeError warnings")
    # warnings.filterwarnings("ignore", category=iers.IERSRangeError, module=iers)

    if node_local:
        # Every node has a local astropy installation (i.e. it is not installed to a
        # network / shared filesystem).  In this case one process per node does the
        # trigger.
        pass
    else:
        # Only one global process triggers
        pass


try:
    import psutil

    def memreport(msg="", comm=None, silent=False):
        """Gather and report the amount of allocated, free and swapped system memory"""
        if psutil is None:
            return
        vmem = psutil.virtual_memory()._asdict()
        gc.collect()
        vmem2 = psutil.virtual_memory()._asdict()
        memstr = None
        if comm is None or comm.rank == 0:
            memstr = "Memory usage {}\n".format(msg)
        for key, value in vmem.items():
            value2 = vmem2[key]
            vlist = None
            vlist2 = None
            if comm is None:
                vlist = [value]
                vlist2 = [value2]
            else:
                vlist = comm.gather(value, root=0)
                vlist2 = comm.gather(value2, root=0)
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
            if not silent:
                print(memstr, flush=True)
        if comm is not None:
            comm.Barrier()
        return memstr


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


def name_UID(name):
    """Return a unique integer for a specified name string."""
    bdet = name.encode("utf-8")
    dhash = hashlib.md5()
    dhash.update(bdet)
    bdet = dhash.digest()
    uid = None
    try:
        ind = int.from_bytes(bdet, byteorder="little")
        uid = int(ind & 0xFFFFFFFF)
    except:
        raise RuntimeError(
            "Cannot convert detector name {} to a unique integer-\
            maybe it is too long?".format(
                name
            )
        )
    return uid


def rate_from_times(timestamps, mean=False):
    """Compute effective sample rate in Hz from timestamps.

    There are many cases when we want to apply algorithms that require a fixed
    sample rate.  We want to compute that from timestamps while also checking for
    any outliers that could compromise the results.

    By default this function uses the median delta_t, under the assumption that
    variations in the timing is due to small numerical / bit noise effects.  For larger
    variations using the mean may be more appropriate (set mean=True).

    This returns the sample rate and also the statistics of the time deltas between
    samples.

    Args:
        timestamps (array):  The array of timestamps.

    Returns:
        (tuple):  The (rate, dt, dt_min, dt_max, dt_std) values.

    """
    tdiff = np.diff(timestamps)
    dt_min = np.min(tdiff)
    dt_max = np.max(tdiff)
    dt_std = np.std(tdiff)
    dt = None
    if mean:
        dt = np.mean(np.diff(timestamps))
    else:
        dt = np.median(np.diff(timestamps))
    return (1.0 / dt, dt, dt_min, dt_max, dt_std)


def dtype_to_aligned(dt):
    """For a numpy dtype, return the equivalent internal Aligned storage class.

    Args:
        dt (dtype):  The numpy dtype.

    Returns:
        (tuple):  The (storage class, item size).

    """
    log = Logger.get()
    itemsize = None
    storage_class = None
    ttype = np.dtype(dt)
    if ttype.char == "b":
        storage_class = AlignedI8
        itemsize = 1
    elif ttype.char == "B":
        storage_class = AlignedU8
        itemsize = 1
    elif ttype.char == "h":
        storage_class = AlignedI16
        itemsize = 2
    elif ttype.char == "H":
        storage_class = AlignedU16
        itemsize = 2
    elif ttype.char == "i":
        storage_class = AlignedI32
        itemsize = 4
    elif ttype.char == "I":
        storage_class = AlignedU32
        itemsize = 4
    elif (ttype.char == "q") or (ttype.char == "l"):
        storage_class = AlignedI64
        itemsize = 8
    elif (ttype.char == "Q") or (ttype.char == "L"):
        storage_class = AlignedU64
        itemsize = 8
    elif ttype.char == "f":
        storage_class = AlignedF32
        itemsize = 4
    elif ttype.char == "d":
        storage_class = AlignedF64
        itemsize = 8
    elif ttype.char == "F":
        raise NotImplementedError("No support yet for complex numbers")
    elif ttype.char == "D":
        raise NotImplementedError("No support yet for complex numbers")
    else:
        msg = "Unsupported data typecode '{}'".format(ttype.char)
        log.error(msg)
        raise ValueError(msg)
    return (storage_class, itemsize)


def array_dot(u, v):
    """Dot product of each row of two 2D arrays"""
    return np.sum(u * v, axis=1).reshape((-1, 1))


def object_fullname(o):
    """Return the fully qualified name of an object."""
    module = o.__module__
    if module is None or module == str.__module__:
        return o.__qualname__
    return "{}.{}".format(module, o.__qualname__)


def import_from_name(name):
    """Import a class from its full name."""
    cls_parts = name.split(".")
    cls_name = cls_parts.pop()
    cls_mod_name = ".".join(cls_parts)
    cls = None
    try:
        cls_mod = importlib.import_module(cls_mod_name)
        cls = getattr(cls_mod, cls_name)
    except:
        msg = f"Cannot import class '{cls_name}' from module '{cls_mod_name}'"
        raise RuntimeError(msg)
    return cls
