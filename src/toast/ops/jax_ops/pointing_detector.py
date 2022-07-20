# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from .utils import assert_data_localization, select_implementation, ImplementationType, MutableJaxArray
from .qarray import mult_one_one_numpy as qa_mult_numpy, mult_one_one_jax as qa_mult_jax
from ..._libtoast import pointing_detector as pointing_detector_compiled
from ..._libtoast import Logger

#-------------------------------------------------------------------------------------------------
# JAX

def pointing_detector_inner_jax(focalplane, boresight, flag, mask):
    """
    Process a single detector and a single sample inside an interval.

    Args:
        focalplane (array, double): size 4
        boresight (array, double): size 4
        flag (uint8)
        mask (uint8)

    Returns:
        quats (array, double): size 4
    """ 
    quats = jnp.where((flag & mask) == 0, # if
                      qa_mult_jax(boresight, focalplane), # then
                      focalplane) # else
    return quats

# maps over samples and detectors
pointing_detector_inner_jax = jax_xmap(pointing_detector_inner_jax, 
                                       in_axes=[['detectors',...], # focalplane
                                                ['samples',...], # boresight
                                                ['samples'], # samples
                                                [...]], # mask
                                       out_axes=['detectors','samples',...])

def pointing_detector_interval_jax(focalplane, boresight, shared_flags, shared_flag_mask):
    """
    Process a full interval at once.
    NOTE: this function was added for debugging purposes, one could replace it with pointing_detector_inner_jax

    Args:
        focalplane (array, double): size n_det*4
        boresight (array, double): size n_samp_interval*4
        shared_flags (array, uint8): size n_samp_interval
        shared_flag_mask (uint8)

    Returns:
        quats (array, double): size n_det*n_samp_interval*4
    """
    print(f"DEBUG: jit-compiling 'pointing_detector_interval_jax' with n_det:{focalplane.shape[0]} n_samp_interval:{shared_flags.size} mask:{shared_flag_mask}")
    return pointing_detector_inner_jax(focalplane, boresight, shared_flags, shared_flag_mask)

# jit compiling
pointing_detector_interval_jax = jax.jit(pointing_detector_interval_jax, static_argnames='shared_flag_mask')

def pointing_detector_jax(focalplane, boresight, quat_index, quats, intervals, shared_flags, shared_flag_mask, use_accel):
    """
    Args:
        focalplane (array, double): size n_det*4
        boresight (array, double): size n_samp*4
        quat_index (array, int): size n_det
        quats (array, double): size ???*n_samp*4
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        use_accel (bool): should we use the accelerator

    Returns:
        None (the result is put in quats).
    """
    # make sure the data is where we expect it
    # TODO assert_data_localization('pointing_detector', use_accel, [focalplane, boresight, shared_flags], [quats])

    # moves focalplane to GPU once for all loop iterations
    focalplane = jnp.array(focalplane)

    # we loop over intervals
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        # extract interval slices
        boresight_interval = boresight[interval_start:interval_end, :]
        shared_flags_interval = shared_flags[interval_start:interval_end]
        # process the interval then updates quats in place
        new_quats_interval = pointing_detector_interval_jax(focalplane, boresight_interval, shared_flags_interval, shared_flag_mask)
        quats[quat_index, interval_start:interval_end, :] = new_quats_interval

#-------------------------------------------------------------------------------------------------
# NUMPY

def pointing_detector_inner_numpy(flag, boresight, focalplane, mask):
    """
    Process a single detector and a single sample inside an interval.

    Args:
        flag (uint8)
        boresight (array, double): size 4
        focalplane (array, double): size 4
        mask (uint8)

    Returns:
        quats (array, double): size 4
    """ 
    if ((flag & mask) == 0):
        quats = qa_mult_numpy(boresight, focalplane)
    else:
        quats = focalplane

def pointing_detector_numpy(focalplane, boresight, quat_index, quats, intervals, shared_flags, shared_flag_mask, use_accel):
    """
    Args:
        focalplane (array, double): size n_det*4
        boresight (array, double): size n_samp*4
        quat_index (array, int): size n_det
        quats (array, double): size ???*n_samp*4
        intervals (array, Interval): The intervals to modify (size n_view)
        shared_flags (array, uint8): size n_samp
        shared_flag_mask (uint8)
        use_accel (bool): should weuse the accelerator

    Returns:
        None (the result is put in quats).
    """
    # input sizes
    n_det = quat_index.size    
    print(f"DEBUG: running 'pointing_detector_numpy' with n_view:{intervals.size} n_det:{n_det} n_samp:{shared_flags.size}")

    # iterates on all detectors and all intervals
    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval['first']
            interval_end = interval['last']+1
            for isamp in range(interval_start,interval_end):
                q_index = quat_index[idet]
                quats[q_index,isamp,:] = pointing_detector_inner_numpy(shared_flags[isamp],
                                                                       boresight[isamp,:],
                                                                       focalplane[idet,:],
                                                                       shared_flag_mask)

    return quats

#-------------------------------------------------------------------------------------------------
# C++

"""
void pointing_detector_inner(
    int32_t const * q_index,
    uint8_t const * flags,
    double const * boresight,
    double const * fp,
    double * quats,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    uint8_t mask
) 
{
    int32_t qidx = q_index[idet];
    double temp_bore[4];
    if ((flags[isamp] & mask) == 0) 
    {
        temp_bore[0] = boresight[4 * isamp];
        temp_bore[1] = boresight[4 * isamp + 1];
        temp_bore[2] = boresight[4 * isamp + 2];
        temp_bore[3] = boresight[4 * isamp + 3];
    } 
    else 
    {
        temp_bore[0] = 0.0;
        temp_bore[1] = 0.0;
        temp_bore[2] = 0.0;
        temp_bore[3] = 1.0;
    }
    qa_mult(temp_bore, &(fp[4 * idet]), &(quats[(qidx * 4 * n_samp) + 4 * isamp]));
}

void pointing_detector(
    py::buffer focalplane,
    py::buffer boresight,
    py::buffer quat_index,
    py::buffer quats,
    py::buffer intervals,
    py::buffer shared_flags,
    uint8_t shared_flag_mask) 
{
    // This is used to return the actual shape of each buffer
    std::vector <int64_t> temp_shape(3);

    int32_t * raw_quat_index = extract_buffer <int32_t> (quat_index, "quat_index", 1, temp_shape, {-1});
    int64_t n_det = temp_shape[0];

    double * raw_focalplane = extract_buffer <double> (focalplane, "focalplane", 2, temp_shape, {n_det, 4});

    double * raw_boresight = extract_buffer <double> (boresight, "boresight", 2, temp_shape, {-1, 4});
    int64_t n_samp = temp_shape[0];

    double * raw_quats = extract_buffer <double> (quats, "quats", 3, temp_shape, {-1, n_samp, 4});

    Interval * raw_intervals = extract_buffer <Interval> (intervals, "intervals", 1, temp_shape, {-1});
    int64_t n_view = temp_shape[0];

    uint8_t * raw_flags = extract_buffer <uint8_t> (shared_flags, "flags", 1, temp_shape, {n_samp});

    double * dev_boresight = raw_boresight;
    double * dev_quats = raw_quats;
    Interval * dev_intervals = raw_intervals;
    uint8_t * dev_flags = raw_flags;
    
    for (int64_t idet = 0; idet < n_det; idet++) 
    {
        for (int64_t iview = 0; iview < n_view; iview++) 
        {
            #pragma omp parallel for
            for (int64_t isamp = dev_intervals[iview]['first']; isamp <= dev_intervals[iview]['last']; isamp++)
            {
                pointing_detector_inner(
                    raw_quat_index,
                    dev_flags,
                    dev_boresight,
                    raw_focalplane,
                    dev_quats,
                    isamp,
                    n_samp,
                    idet,
                    shared_flag_mask);
            }
        }
    }
}
"""

#-------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
pointing_detector = select_implementation(pointing_detector_compiled, 
                                          pointing_detector_numpy, 
                                          pointing_detector_jax)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix")'

# to bench:
# use scanmap config and check PixelsHealpix._exec (TODO check) field in timing.csv