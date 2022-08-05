
# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap as jax_xmap

from .utils import assert_data_localization, select_implementation, ImplementationType, math_qarray as qarray, math_healpix as healpix, optional_put_device
from .utils.mutableArray import MutableJaxArray, reorder_by_index_jitted
from ..._libtoast import pixels_healpix as pixels_healpix_compiled

# -------------------------------------------------------------------------------------------------
# JAX

def pixels_healpix_inner_jax(hpix, quats, nest):
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

# maps over samples and detectors
#pixels_healpix_inner_jax = jax_xmap(pixels_healpix_inner_jax, 
#                                    in_axes=[[...], # hpix
#                                             ['detectors','samples',...], # quats
#                                             [...]], # nest
#                                    out_axes=['detectors','samples'])
# TODO xmap is commented out for now due to a [bug with static argnum](https://github.com/google/jax/issues/10741)
pixels_healpix_inner_jax = jax.vmap(pixels_healpix_inner_jax, in_axes=[None,0,None], out_axes=0) # loop on samples
pixels_healpix_inner_jax = jax.vmap(pixels_healpix_inner_jax, in_axes=[None,0,None], out_axes=0) # loop on detectors

def pixels_healpix_interval_jax(hpix, quats, flags, flag_mask, hit_submaps, n_pix_submap, nest):
    """
    Process a full interval at once.
    NOTE: this function was added for debugging purposes, one could replace it with `pixels_healpix_inner_jax`

    Args:
        hpix (HPIX_JAX): Healpix projection object.
        quats (array, float64): The flat-packed array of detector quaternions (size n_det*n_samp_interval*4).
        flags (optional array, bool): size n_samp_interval (could be None)
        flag_mask (uint8): integer used to select flags (not necesarely a boolean)
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (array, float64):  
        nest (bool): If True, then use NESTED ordering, else RING.

    Returns:
        pixels (array, int64): The detector pixel indices to store the result (size n_det*n_samp_interval).
    """
    # display sizes
    print(f"DEBUG: jit-compiling 'pixels_healpix_interval_jax' with n_side:{hpix.nside} n_det:{quats.shape[0]} n_samp_interval:{quats.shape[1]} n_pix_submap:{n_pix_submap} hit_submaps:{hit_submaps.size} flag_mask:{flag_mask} nest:{nest}")
    
    # computes the pixels and submap
    pixels = pixels_healpix_inner_jax(hpix, quats, nest)
    sub_map = pixels // n_pix_submap

    # applies the flags
    if flags is not None:
        is_flagged = (flags & flag_mask) != 0
        pixels = jnp.where(is_flagged, -1, pixels)
        new_hit_submap = jnp.where(is_flagged, hit_submaps[sub_map], 1)
    else:
        new_hit_submap = 1

    # updates hit_submaps
    hit_submaps = hit_submaps.at[sub_map].set(new_hit_submap)
    return (pixels, hit_submaps)

# jit compiling
pixels_healpix_interval_jax = jax.jit(pixels_healpix_interval_jax, 
                                      static_argnames=['hpix', 'flag_mask', 'n_pix_submap', 'nest'],
                                      donate_argnums=[4])

def pixels_healpix_jax(quat_index, quats, flags, flag_mask, pixel_index, pixels, intervals, hit_submaps, n_pix_submap, nside, nest, use_accel):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        flags (array, uint8): size n_samp (or you shouldn't use flags)
        flag_mask (uint8): integer used to select flags (not necesarely a boolean)
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size n_det*n_samp).
        intervals (array, float64): size n_view
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (int):  
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.
        use_accel (bool): should we use the accelerator

    Returns:
        None (results are stored in pixels and hit_submaps).
    """
    # make sure the data is where we expect it
    assert_data_localization('pixels_healpix', use_accel, [quats, flags, hit_submaps], [pixels, hit_submaps])

    # initialize hpix for all computations
    hpix = healpix.HPIX_JAX.init(nside)

    # should we use flags?
    n_samp = pixels.shape[1]
    use_flags = (flag_mask != 0) and (flags.size == n_samp)

    # moves hit_submaps to GPU once for all loop iterations
    hit_submaps_gpu = optional_put_device(hit_submaps)

    # does the indexing once and for all
    if isinstance(quats, MutableJaxArray):
        quats = quats.data[quat_index,:,:]
    else:
        quats = quats[quat_index,:,:]

    # loop on the intervals
    for interval in intervals:
        interval_start = interval['first']
        interval_end = interval['last']+1
        # extract interval slices
        quats_interval = quats[:, interval_start:interval_end, :]
        flags_interval = flags[interval_start:interval_end] if use_flags else None
        # does the computation and updates pixels and hit_submaps in place
        new_pixels_interval, hit_submaps_gpu = pixels_healpix_interval_jax(hpix, quats_interval, flags_interval, flag_mask, hit_submaps_gpu, n_pix_submap, nest)
        pixels[:, interval_start:interval_end] = new_pixels_interval # NOTE: we ignore pixel_index and will shuffle once later

    # reshuffles outputs according to pixel_index
    if isinstance(pixels, MutableJaxArray):
        # NOTE: one could omit this line but it might lead to avoidable data copying
        pixels.data = reorder_by_index_jitted(pixels.data, pixel_index)
    else:
        pixels[:,:] = reorder_by_index_jitted(pixels[:,:], pixel_index)

    # goe back to CPU
    hit_submaps[:] = hit_submaps_gpu

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

def pixels_healpix_numpy(quat_index, quats, flags, flag_mask, pixel_index, pixels, intervals, hit_submaps, n_pix_submap, nside, nest, use_accel):
    """
    Compute the healpix pixel indices for the detectors.

    Args:
        quat_index (array, int): size n_det
        quats (array, float64): The flat-packed array of detector quaternions (size ???*n_samp*4).
        flags (array, uint8): size n_samp (or you shouldn't use flags)
        flag_mask (uint8)
        pixel_index (array, int): size n_det
        pixels (array, int64): The detector pixel indices to store the result (size ???*n_samp).
        intervals (array, float64): size n_view
        hit_submaps (array, uint8): The pointing flags (size ???).
        n_pix_submap (array, float64):  
        nside (int): Used to build the healpix projection object.
        nest (bool): If True, then use NESTED ordering, else RING.
        use_accel (bool): should we use the accelerator

    Returns:
        None (results are stored in pixels and hit_submaps).
    """
    # problem size
    n_det = quat_index.size
    n_samp = pixels.shape[1]
    print(f"DEBUG: running 'pixels_healpix_numpy' with n_view:{intervals.size} flag_mask:{flag_mask} n_det:{n_det} n_samp:{n_samp} nest:{nest}")

    # constants
    hpix = healpix.HPIX_NUMPY(nside)
    use_flags = (flag_mask != 0) and (flags.size == n_samp)

    for idet in range(n_det):
        for interval in intervals:
            interval_start = interval['first']
            interval_end = interval['last']+1
            for isamp in range(interval_start,interval_end):
                p_index = pixel_index[idet]
                q_index = quat_index[idet]
                is_flagged = (flags[isamp] & flag_mask) != 0
                if use_flags and is_flagged:
                    # masked pixel
                    pixels[p_index,isamp] = -1
                else:
                    # computes pixel value and saves it
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
    uint8_t const * flags,
    uint8_t * hsub,
    int64_t * pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    uint8_t mask,
    bool use_flags,
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

    if (use_flags && ((flags[isamp] & mask) != 0)) 
    {
        pixels[poff] = -1;
    } 
    else 
    {
        sub_map = (int64_t)(pixels[poff] / n_pix_submap);
        hsub[sub_map] = 1;
    }
}

void pixels_healpix(
            py::buffer quat_index,
            py::buffer quats,
            py::buffer shared_flags,
            uint8_t shared_flag_mask,
            py::buffer pixel_index,
            py::buffer pixels,
            py::buffer intervals,
            py::buffer hit_submaps,
            int64_t n_pix_submap,
            int64_t nside,
            bool nest,
            bool use_accel)
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

    // Optionally use flags
    bool use_flags = true;
    uint8_t * raw_flags = extract_buffer <uint8_t> (shared_flags, "flags", 1, temp_shape, {-1});
    if (temp_shape[0] != n_samp) 
    {
        raw_flags = (uint8_t *)omgr.null;
        use_flags = false;
    }

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
                    raw_flags,
                    raw_hsub,
                    dev_pixels,
                    n_pix_submap,
                    isamp,
                    n_samp,
                    idet,
                    shared_flag_mask,
                    use_flags,
                    nest);
            }
        }
    }
}
"""

# -------------------------------------------------------------------------------------------------
# IMPLEMENTATION SWITCH

# lets us play with the various implementations
pixels_healpix = select_implementation(pixels_healpix_compiled,
                                       pixels_healpix_numpy,
                                       pixels_healpix_jax)

# To test:
# python -c 'import toast.tests; toast.tests.run("ops_pointing_healpix"); toast.tests.run("ops_sim_ground");  toast.tests.run("ops_sim_satellite");'

# to bench:
# use scanmap config and check PixelsHealpix._exec field in timing.csv
