
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import ImplementationType, select_implementation, get_compile_time
from ..._libtoast import template_offset_add_to_signal as template_offset_add_to_signal_compiled
from ..._libtoast import template_offset_project_signal as template_offset_project_signal_compiled

# -------------------------------------------------------------------------------------------------
# JAX

def template_offset_add_to_signal_jitted(step_length, amplitudes, data):
    print(f"DEBUG: jit-compiling 'template_offset_add_to_signal' step_length:{step_length} nb_amp:{amplitudes.size} nb_data:{data.size}")

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_amplitudes = amplitudes.size

    # All but the last amplitude have step_length samples.
    data_first = data[:(nb_amplitudes - 1) * step_length]
    data_first = jnp.reshape(data_first, newshape=(-1,step_length))
    data_first = data_first + amplitudes[:-1, jnp.newaxis]
    #data_first[:] += amplitudes[:-1, jnp.newaxis]
    # TODO could we express this to be more likely to have an inplace modification?
    data = data.at[:(nb_amplitudes - 1) * step_length].set(data_first.ravel())

    # Now handle the final amplitude.
    #data_last = data[(nb_amplitudes - 1) * step_length:]
    #data_last[:] += amplitudes[-1]
    data = data.at[(nb_amplitudes - 1) * step_length:].add(amplitudes[-1])

    return data

# JIT compiles the code
template_offset_add_to_signal_jitted = jax.jit(template_offset_add_to_signal_jitted, static_argnames=['step_length'])

def template_offset_add_to_signal_jax(step_length, amplitudes, data):
    """
    Accumulate offset amplitudes to timestream data.

    Each amplitude value is accumulated to `step_length` number of samples.  The
    final offset will be at least this many samples, but may be more if the step
    size does not evenly divide into the number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amplitudes (array):  The float64 amplitude values.
        data (array):  The float64 timestream values to accumulate.

    Returns:
        None.
    """
    data[:] = template_offset_add_to_signal_jitted(step_length, amplitudes, data)

def template_offset_project_signal_jitted(step_length, data, amplitudes):
    print(f"DEBUG: jit-compiling 'template_offset_project_signal' step_length:{step_length} nb_amp:{amplitudes.size} nb_data:{data.size}")

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_amplitudes = amplitudes.size
    data_first = data[:(nb_amplitudes - 1) * step_length]
    data_first = jnp.reshape(data_first, newshape=(-1,step_length))
    data_last = data[(nb_amplitudes - 1) * step_length:]
        
    # All but the last amplitude have step_length samples.
    #amplitudes[:-1] += np.sum(data_first, axis=1)
    amplitudes = amplitudes.at[:-1].add(np.sum(data_first, axis=1))

    # Now handle the final amplitude.
    #amplitudes[-1] += np.sum(data_last)
    amplitudes = amplitudes.at[-1].add(np.sum(data_last))

    return amplitudes

# JIT compiles the code
template_offset_project_signal_jitted = jax.jit(template_offset_project_signal_jitted, static_argnames=['step_length'])

def template_offset_project_signal_jax(step_length, data, amplitudes):
    """
    Accumulate timestream data into offset amplitudes.

    Chunks of `step_length` number of samples are accumulated into the offset
    amplitudes.  If step_length does not evenly divide into the total number of
    samples, the final amplitude will be extended to include the remainder.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        data (array):  The float64 timestream values.
        amplitudes (array):  The float64 amplitude values.

    Returns:
        None.
    """
    amplitudes[:] = template_offset_project_signal_jitted(step_length, data, amplitudes)

# -------------------------------------------------------------------------------------------------
# NUMPY

def template_offset_add_to_signal_numpy(step_length, amplitudes, data):
    """
    Accumulate offset amplitudes to timestream data.

    Each amplitude value is accumulated to `step_length` number of samples.  The
    final offset will be at least this many samples, but may be more if the step
    size does not evenly divide into the number of samples.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        amplitudes (array):  The float64 amplitude values.
        data (array):  The float64 timestream values to accumulate.

    Returns:
        None.
    """
    #print(f"DEBUG: template_offset_add_to_signal_numpy step_length:{step_length} nb_amp:{amplitudes.size} nb_data:{data.size}")

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_amplitudes = amplitudes.size
    data_first = data[:(nb_amplitudes - 1) * step_length]
    data_first = np.reshape(data_first, newshape=(-1,step_length))
    data_last = data[(nb_amplitudes - 1) * step_length:]

    # All but the last amplitude have step_length samples.
    data_first[:] += amplitudes[:-1, np.newaxis]

    # Now handle the final amplitude.
    data_last[:] += amplitudes[-1]

def template_offset_project_signal_numpy(step_length, data, amplitudes):
    """
    Accumulate timestream data into offset amplitudes.

    Chunks of `step_length` number of samples are accumulated into the offset
    amplitudes.  If step_length does not evenly divide into the total number of
    samples, the final amplitude will be extended to include the remainder.

    Args:
        step_length (int64):  The minimum number of samples for each offset.
        data (array):  The float64 timestream values.
        amplitudes (array):  The float64 amplitude values.

    Returns:
        None.
    """
    #print(f"DEBUG: template_offset_project_signal_numpy step_length:{step_length} nb_amp:{amplitudes.size} nb_data:{data.size}")

    # split data to separate the final amplitude from the rest
    # as it is the only one that does not have step_length samples
    nb_amplitudes = amplitudes.size
    data_first = data[:(nb_amplitudes - 1) * step_length]
    data_first = np.reshape(data_first, newshape=(-1,step_length))
    data_last = data[(nb_amplitudes - 1) * step_length:]
        
    # All but the last amplitude have step_length samples.
    amplitudes[:-1] += np.sum(data_first, axis=1)

    # Now handle the final amplitude.
    amplitudes[-1] += np.sum(data_last)

# -------------------------------------------------------------------------------------------------
# C++

"""
void toast::template_offset_add_to_signal(int64_t step_length, int64_t n_amp,
                                          double * amplitudes,
                                          int64_t n_data, double * data) 
{
    // All but the last amplitude have the same number of samples.
    for (int64_t i = 0; i < n_amp - 1; ++i) 
    {
        int64_t doff = i * step_length;
        for (int64_t j = 0; j < step_length; ++j) 
        {
            data[doff + j] += amplitudes[i];
        }
    }

    // Now handle the final amplitude.
    for (int64_t j = (n_amp - 1) * step_length; j < n_data; ++j) 
    {
        data[j] += amplitudes[n_amp - 1];
    }
}

void toast::template_offset_project_signal(int64_t step_length, int64_t n_data,
                                           double * data, int64_t n_amp,
                                           double * amplitudes) 
{
    // All but the last amplitude have the same number of samples.
    for (int64_t i = 0; i < n_amp - 1; ++i) 
    {
        int64_t doff = i * step_length;
        for (int64_t j = 0; j < step_length; ++j) 
        {
            amplitudes[i] += data[doff + j];
        }
    }

    // Now handle the final amplitude.
    for (int64_t j = (n_amp - 1) * step_length; j < n_data; ++j) 
    {
        amplitudes[n_amp - 1] += data[j];
    }
}
"""

# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
template_offset_add_to_signal = select_implementation(template_offset_add_to_signal_compiled, 
                                                      template_offset_add_to_signal_numpy, 
                                                      template_offset_add_to_signal_jax, 
                                                      default_implementationType=ImplementationType.JAX)
template_offset_project_signal = select_implementation(template_offset_project_signal_compiled, 
                                                       template_offset_project_signal_numpy, 
                                                       template_offset_project_signal_jax, 
                                                       default_implementationType=ImplementationType.JAX)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
#template_offset_add_to_signal = get_compile_time(template_offset_add_to_signal)
#template_offset_project_signal = get_compile_time(template_offset_project_signal)

# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'

# to bench:
# TODO check bench
# use scanmap config and check PixelsHealpix._exec field in timing.csv
