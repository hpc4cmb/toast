import os
import sys
from collections import abc, defaultdict
from functools import wraps

from ..accelerator import accel_data_present
from ..utils import Logger

# ---------------------------------------------------------------------------
# RECORDING CLASS

use_debug_assert = ("TOAST_LOGLEVEL" in os.environ) and (
    os.environ["TOAST_LOGLEVEL"] in ["DEBUG", "VERBOSE"]
)
"""
Assert is used only if `TOAST_LOGLEVEL` is set to `DEBUG`.
"""


def bytes_of_data(data):
    """
    Returns the size of an input in bytes.
    """
    if hasattr(data, "nbytes"):
        return data.nbytes
    else:
        return sys.getsizeof(data)


def is_buffer(data):
    """
    Returns true if the data is a buffer type.
    The function is needed as `omp_accel_present` will break on scalar types
    """
    return hasattr(data, "__len__") or isinstance(data, abc.Sequence)


class DataMovementRecord:
    """
    data structure used to track data movement to and from the GPU for a particular function
    """

    def __init__(self):
        self.nb_calls = 0
        self.input_bytes = 0
        self.input_to_gpu_bytes = 0

    def add(self, args, kwargs, display=False):
        """
        Adds data to the record, returns True if at least one of the inputs was on GPU
        """
        self.nb_calls += 1
        some_gpu_input = False
        # args
        for i, input in enumerate(args):
            bytes = bytes_of_data(input)
            gpu_data = is_buffer(input) and accel_data_present(input)
            self.input_bytes += bytes
            if not gpu_data:
                self.input_to_gpu_bytes += bytes
            if display:
                print(f"input[{i}] size:{bytes} GPU:{gpu_data}")
            some_gpu_input = some_gpu_input or gpu_data
        # kwargs
        for name, input in kwargs.items():
            bytes = bytes_of_data(input)
            gpu_data = is_buffer(input) and accel_data_present(input)
            self.input_bytes += bytes
            if not gpu_data:
                self.input_to_gpu_bytes += bytes
            if display:
                print(f"input[{name}] size:{bytes} GPU:{gpu_data}")
            some_gpu_input = some_gpu_input or gpu_data
        return some_gpu_input


class DataMovementTracker:
    """
    data structure used to track data movement to and from the GPU for several functions
    """

    def __init__(self):
        # each function gets a record
        self.records = defaultdict(lambda: DataMovementRecord())

    def add(self, functionname, args, kwargs, display=False):
        """
        keep track of the total size of the inputs
        and the size of the inputs already on GPU
        this is useful to monitor data movement on a per function basis
        """
        # records the data size and movement
        if display:
            print(f"DATA MOVEMENT ({functionname}):")
        some_gpu_input = self.records[functionname].add(args, kwargs, display)
        # send a warning in case of suspicious data movement
        use_accel = kwargs.get("use_accel", False)
        if use_accel and (not some_gpu_input):
            msg = (
                f"function '{functionname}' has NO input on GPU despite use_accel=True!"
            )
            log = Logger.get()
            log.warning(msg)
        elif some_gpu_input and (not use_accel):
            msg = (
                f"function '{functionname}' has inputs on GPU despite use_accel=False!"
            )
            log = Logger.get()
            log.warning(msg)

    def __str__(self):
        """
        produces a CSV representation of the DataMovementTracker
        with one line per function and using ',' as the separator
        """
        if not use_debug_assert:
            return "DataMovementTracker: no monitoring unless TOAST_LOGLEVEL is set to DEBUG or VERBOSE."
        result = "DataMovementTracker:\n-----\n"
        result += "function, nb calls, input bytes, input bytes moved to GPU\n"
        for function_name, record in sorted(self.records.items()):
            result += f"{function_name}, {record.nb_calls}, {record.input_bytes}, {record.input_to_gpu_bytes}\n"
        result += "-----"
        return result


# ----------------------------------------------------------------------------------
# OPERATIONS

dataMovementTracker = DataMovementTracker()
"""global variable used to track data movement to and from the GPU"""


def function_datamovementtracker(f):
    """
    Wraps a function to keep track of movement to and from the GPU
    NOTE: the recording is only done if we are in DEBUG or VERBOSE mode
    """

    @wraps(f)
    def f_result(*args, **kwargs):
        dataMovementTracker.add(f.__name__, args, kwargs)
        return f(*args, **kwargs)

    return f_result if use_debug_assert else f


def display_datamovement():
    """Displays all recorded data movement so far"""
    if use_debug_assert:
        log = Logger.get()
        log.debug(str(dataMovementTracker))
