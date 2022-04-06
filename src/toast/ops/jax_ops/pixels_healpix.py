
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

from .utils import select_implementation, ImplementationType, math_qarray as qarray, math_healpix as healpix

# -------------------------------------------------------------------------------------------------
# JAX

def pixels_healpix_single_sample_jax(hpix, quats, nest):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        hpix (HPIX_JAX): Healpix projection object.
        quats (array, float64): Detector quaternion (size 4).
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        pixels (array, int64): The detector pixel indices to store the result.
    """
    # constants
    zaxis = jnp.array([0.0, 0.0, 1.0])

    # initialize dir
    dir = qarray.rotate_one_one_jax(quats, zaxis)

    # pixel computation
    (phi, region, z, rtz) = healpix.vec2zphi_jax(dir)
    if (nest):
        pixel = healpix.zphi2nest_jax(hpix, phi, region, z, rtz)
    else:
        pixel = healpix.zphi2ring_jax(hpix, phi, region, z, rtz)

    return pixel

# vmap over samples
pixels_healpix_single_detector_jax = jax.vmap(pixels_healpix_single_sample_jax, in_axes=(None,0,None), out_axes=0)
# vmap over detectors
pixels_healpix_interval_jax = jax.vmap(pixels_healpix_single_detector_jax, in_axes=(None,0,None), out_axes=0)

def pixels_healpix_unjitted_jax(hpix, quats, nest):
    """
    Process a full interval at once.
    NOTE: this function was added for debugging purposes, one could replace it with `pixels_healpix_interval_jax`

    Args:
        hpix (HPIX_JAX): Healpix projection object.
        quats (array, float64): The flat-packed array of detector quaternions (size n_det*n_samp_interval*4).
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        pixels (array, int64): The detector pixel indices to store the result (size n_det*n_samp_interval).
    """
    # display sizes
    print(f"DEBUG: jit compiling 'pixels_healpix_interval_jax' with n_side:{hpix.nside} n_det:{quats.shape[0]} n_samp_interval:{quats.shape[1]} nest:{nest}")
    # does the computation
    return pixels_healpix_interval_jax(hpix, quats, nest)

# jit compiling
pixels_healpix_jitted_jax = jax.jit(pixels_healpix_unjitted_jax, static_argnames=['hpix, nest'])


def pixels_healpix_jax(quat_index, quats, pixel_index, pixels, intervals, hit_submaps, n_pix_submap, nside, nest):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size ???*n_samp).
        intervals (array, float64): size n_view
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (array, float64):  
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        None (results are stored in pixels and hit_submaps).
    """
    # initialize hpix for all computations
    hpix = healpix.HPIX_JAX(nside)

    # loop on the intervals
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        # extract interval slices
        quats_interval = quats[quat_index, interval_start:interval_end, :]
        pixels_interval = pixels[pixel_index, interval_start:interval_end]
        # does the computation and stores the result in pixels
        pixels_interval[:] = pixels_healpix_jitted_jax(hpix, quats_interval, nest)
        # modifies hit_submap in place
        submap_interval = pixels_interval // n_pix_submap
        hit_submaps[submap_interval] = 1

# -------------------------------------------------------------------------------------------------
# NUMPY

def pixels_healpix_inner_numpy(hpix, quats, nest):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        hpix (HPIX_NUMPY): Healpix projection object.
        quats (array, float64): Detector quaternion (size 4).
        hsub (array, uint8): The pointing flags (size ???).
        intervals (array, float64): size n_view
        n_pix_submap (array, float64):  
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        pixels (array, int64): The detector pixel indices to store the result.
    """
    # constants
    zaxis = np.array([0.0, 0.0, 1.0])

    # initialize dir
    dir = qarray.rotate_one_one_numpy(quats, zaxis)

    # pixel computation
    (phi, region, z, rtz) = healpix.vec2zphi_numpy(dir)
    if (nest):
        pixel = healpix.zphi2nest_numpy(hpix, phi, region, z, rtz)
    else:
        pixel = healpix.zphi2ring_numpy(hpix, phi, region, z, rtz)

    return pixel

def pixels_healpix_numpy(quat_index, quats, pixel_index, pixels, intervals, hit_submaps, n_pix_submap, nside, nest):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size ???*n_samp).
        intervals (array, float64): size n_view
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (array, float64):  
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        None (results are stored in pixels and hit_submaps).
    """
    # problem size
    n_det = quat_index.size
    print(f"DEBUG: running 'pixels_healpix_numpy' with n_view:{intervals.size} n_det:{n_det} n_samp:{pixels.shape[1]} nest:{nest}")

    hpix = healpix.HPIX_NUMPY(nside)

    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval['first']
            interval_end = interval['last']
            for isamp in range(interval_start,interval_end+1):
                # computes pixel value and saves it
                p_index = pixel_index[idet]
                q_index = quat_index[idet]
                pixel = pixels_healpix_inner_numpy(hpix, quats[q_index,isamp,:], nest)
                pixels[p_index,isamp] = pixel
                # modifies submap in place
                sub_map = pixel // n_pix_submap
                hit_submaps[sub_map] = 1

# -------------------------------------------------------------------------------------------------
# C++

"""
void pixels_healpix_inner(
    hpix & hp,
    int32_t const * quat_index,
    int32_t const * pixel_index,
    double const * quats,
    uint8_t * hsub,
    int64_t * pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    bool nest) 
{
    const double zaxis[3] = {0.0, 0.0, 1.0};
    int32_t p_indx = pixel_index[idet];
    int32_t q_indx = quat_index[idet];
    double dir[3];
    double z;
    double rtz;
    double phi;
    int region;
    size_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;
    size_t poff = p_indx * n_samp + isamp;
    int64_t sub_map;

    qa_rotate(&(quats[qoff]), zaxis, dir);
    hpix_vec2zphi(&hp, dir, &phi, &region, &z, &rtz);

    if (nest)
    {
        hpix_zphi2nest(&hp, phi, region, z, rtz, &(pixels[poff]));
    }
    else
    {
        hpix_zphi2ring(&hp, phi, region, z, rtz, &(pixels[poff]));
    }
    
    sub_map = (int64_t)(pixels[poff] / n_pix_submap);
    hsub[sub_map] = 1;
}

void pixels_healpix(
            py::buffer quat_index,
            py::buffer quats,
            py::buffer pixel_index,
            py::buffer pixels,
            py::buffer intervals,
            py::buffer hit_submaps,
            int64_t n_pix_submap,
            int64_t nside,
            bool nest)
{
    // This is used to return the actual shape of each buffer
    std::vector <int64_t> temp_shape(3);

    int32_t * raw_quat_index = extract_buffer <int32_t> (quat_index, "quat_index", 1, temp_shape, {-1});
    int64_t n_det = temp_shape[0];

    int32_t * raw_pixel_index = extract_buffer <int32_t> (pixel_index, "pixel_index", 1, temp_shape, {n_det});

    int64_t * raw_pixels = extract_buffer <int64_t> (pixels, "pixels", 2, temp_shape, {-1, -1});
    int64_t n_samp = temp_shape[1];

    double * raw_quats = extract_buffer <double> (quats, "quats", 3, temp_shape, {-1, n_samp, 4});

    Interval * raw_intervals = extract_buffer <Interval> (intervals, "intervals", 1, temp_shape, {-1});
    int64_t n_view = temp_shape[0];

    uint8_t * raw_hsub = extract_buffer <uint8_t> (hit_submaps, "hit_submaps", 1, temp_shape, {-1});

    int64_t * dev_pixels = raw_pixels;
    double * dev_quats = raw_quats;
    Interval * dev_intervals = raw_intervals;

    hpix hp;
    hpix_init(&hp, nside);

    for (int64_t idet = 0; idet < n_det; idet++) 
    {
        for (int64_t iview = 0; iview < n_view; iview++) 
        {
            #pragma omp parallel for
            for (int64_t isamp = dev_intervals[iview]['first']; isamp <= dev_intervals[iview]['last']; isamp++)
            {
                pixels_healpix_inner(
                    hp,
                    raw_quat_index,
                    raw_pixel_index,
                    dev_quats,
                    raw_hsub,
                    dev_pixels,
                    n_pix_submap,
                    isamp,
                    n_samp,
                    idet,
                    nest);
            }
        }
    }
}
"""

# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
pixels_healpix = select_implementation(pixels_healpix_numpy,
                                       pixels_healpix_numpy,
                                       pixels_healpix_jax,
                                       default_implementationType=ImplementationType.NUMPY)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix")'

# to bench:
# use scanmap config and check PixelsHealpix._exec field in timing.csv