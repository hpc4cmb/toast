
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
    # problem size
    n_amp = amplitudes.size
    #print(f"DEBUG: template_offset_add_to_signal_numpy step_length:{step_length} n_amp:{n_amp} n_data:{data.size}")
    
    # All but the last amplitude have the same number of samples.
    data_firsts = data[:(n_amp - 1) * step_length]
    data_firsts = np.reshape(data_firsts, newshape=(-1,step_length))
    data_firsts[:] += amplitudes[:(n_amp-1), np.newaxis]

    # Now handle the final amplitude.
    # needed as data's size is not a multiple of step_length
    final_data = data[(n_amp - 1) * step_length:]
    final_data[:] += amplitudes[n_amp - 1]

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
    # problem size
    n_amp = amplitudes.size
    #print(f"DEBUG: template_offset_project_signal_numpy step_length:{step_length} n_amp:{n_amp} n_data:{data.size}")
    
    # All but the last amplitude have the same number of samples.
    data_firsts = data[:(n_amp - 1) * step_length]
    data_firsts = np.reshape(data_firsts, newshape=(-1,step_length))
    amplitudes[:(n_amp-1)] += np.sum(data_firsts, axis=1)

    # Now handle the final amplitude.
    # needed as data's size is not a multiple of step_length
    final_data = data[(n_amp - 1) * step_length:]
    amplitudes[n_amp - 1] += np.sum(final_data)

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
                                                      template_offset_add_to_signal_compiled, 
                                                      default_implementationType=ImplementationType.NUMPY)
template_offset_project_signal = select_implementation(template_offset_project_signal_compiled, 
                                                       template_offset_project_signal_numpy, 
                                                       template_offset_project_signal_compiled, 
                                                       default_implementationType=ImplementationType.NUMPY)

# TODO we extract the compile time at this level to encompas the call and data movement to/from GPU
#template_offset_add_to_signal = get_compile_time(template_offset_add_to_signal)
#template_offset_project_signal = get_compile_time(template_offset_project_signal)

# To test:
# python -c 'import toast.tests; toast.tests.run("template_offset"); toast.tests.run("ops_mapmaker_solve"); toast.tests.run("ops_mapmaker")'

# to bench:
# TODO check bench
# use scanmap config and check PixelsHealpix._exec field in timing.csv
