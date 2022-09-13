from collections import defaultdict
import os
from .mutableArray import MutableJaxArray
import jax.numpy as jnp
from ....utils import Logger

use_debug_assert = ("TOAST_LOGLEVEL" in os.environ) and (os.environ["TOAST_LOGLEVEL"] in ["DEBUG", "VERBOSE"])
"""
Assert is used only if `TOAST_LOGLEVEL` is set to `DEBUG`.
"""

#----------------------------------------------------------------------------------------
# ASSERT

def assert_data_localization(function_name, use_accel, inputs, outputs):
    """
    Checks that the data position (GPU|CPU) is consistent with the `use_accel` flag.
    This function will send a warning in case of inconsisency if `TOAST_LOGLEVEL` is set to `DEBUG`.
    """
    if use_debug_assert:
        # checks if the data is on GPU
        gpu_input = any(isinstance(x, MutableJaxArray) or isinstance(x, jnp.ndarray) for x in inputs)
        gpu_output = any(isinstance(x, MutableJaxArray) or isinstance(x, jnp.ndarray) for x in outputs)

        if use_accel:
            # checks that at least some inputs the GPU
            if not gpu_input:
                if gpu_output:
                    msg = f"function '{function_name}' has NO input on GPU (only some output data) but is running with use_accel=True"
                else:
                    msg = f"function '{function_name}' has NO input on GPU but is running with use_accel=True"
                log = Logger.get()
                log.warning(msg)
                #raise RuntimeError("GPU localisation error")
        else:
            # checks that no data is on the GPU
            if gpu_input or gpu_output:
                msg = f"function '{function_name}' has an input on GPU but is running with use_accel=False"
                log = Logger.get()
                log.warning(msg)
                #raise RuntimeError("GPU localisation error")

#----------------------------------------------------------------------------------------
# TRACKER

class DataMovement:
    """data structure used to track data movement to and from the GPU for a particular function"""

    def __init__(self):
        self.nb_calls = 0
        self.input_bytes = 0
        self.input_to_gpu_bytes = 0
        self.output_bytes = 0
        self.output_from_gpu_bytes = 0
    
    def add(self, inputs, outputs):
        self.nb_calls += 1
        # inputs
        for input in inputs:
            bytes = input.nbytes
            self.input_bytes += bytes 
            if not isinstance(input, jnp.ndarray):
                self.input_to_gpu_bytes += bytes
        # outputs
        for output in outputs:
            bytes = output.nbytes
            self.output_bytes += bytes 
            if not isinstance(output, MutableJaxArray):
                self.output_from_gpu_bytes += bytes

class DataMovementTracker:
    """data structure used to track data movement to and from the GPU for several function"""

    def __init__(self):
        # each function gets a pair of record: CPU,GPU
        self.records = defaultdict(lambda: (DataMovement(), DataMovement()))

    def add(self, functionname, use_accel, inputs, outputs):
        """adds information on a function call if we are in DEBUG or VERBOSE mode"""
        if use_debug_assert: 
            record_cpu, record_gpu = self.records[functionname]
            if use_accel:
                record_gpu.add(inputs, outputs)
            else:
                record_cpu.add(inputs, outputs)

    def __str__(self):
        """
        produces a CSV representation of the DataMovementTracker
        with one line per function and using ',' as the separator
        the first rows are dedicated to the CPU pipelines
        the last rows are dedicated to the GPU pipelines
        """
        result = "DataMovementTracker:\n-----\n"
        result += "function, "
        result += "nb calls cpu, input bytes cpu, input bytes moved to GPU cpu, output bytes cpu, output bytes moved from GPU cpu, "
        result += "nb calls gpu, input bytes gpu, input bytes moved to GPU gpu, output bytes gpu, output bytes moved from GPU gpu\n"
        for function_name, (record_cpu, record_gpu) in sorted(self.records.items()):
            columns_cpu = f"{record_cpu.nb_calls}, {record_cpu.input_bytes}, {record_cpu.input_to_gpu_bytes}, {record_cpu.output_bytes}, {record_cpu.output_from_gpu_bytes}"
            columns_gpu = f"{record_gpu.nb_calls}, {record_gpu.input_bytes}, {record_gpu.input_to_gpu_bytes}, {record_gpu.output_bytes}, {record_gpu.output_from_gpu_bytes}"
            result += f"{function_name}, {columns_cpu}, {columns_gpu}\n"
        result += "-----"
        return result
    
    # TODO integrate assert functionalities in this object?

dataMovementTracker = DataMovementTracker()
"""global variable used to track data movement to and from the GPU"""
