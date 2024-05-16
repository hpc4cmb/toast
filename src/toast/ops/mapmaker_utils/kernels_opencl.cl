// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// https://github.com/KhronosGroup/OpenCL-Docs/blob/main/extensions/cl_ext_float_atomics.asciidoc

// https://stackoverflow.com/questions/73838432/looking-for-examples-for-atomic-fetch-add-for-float32-in-opencl-3-0

// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// #if __OPENCL_C_VERSION__ >= CL_VERSION_3_0
//     #pragma OPENCL EXTENSION cl_ext_float_atomics : enable
//     #pragma OPENCL EXTENSION cl_ext_double_atomics : enable
//     #define atomic_add_double(a,b) atomic_fetch_add((volatile atomic_float *)(a),(b))

//     atomic_fetch_add(volatile __global A *object, M operand)
// #else

//   inline float atomic_add_double(volatile __global float* address, const float value) {
//     float old = value, orig;
//     while ((old = atomic_xchg(address, (orig = atomic_xchg(address, 0.0f)) + old)) != 0.0f);
//     return orig;
//   }
// #endif

// Based on:
// https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/

void __attribute__((always_inline)) atomic_add_float(
    volatile global float* addr, const float val
) {
    union {
        uint u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg(
            (volatile global uint*)addr, expected.u32, next.u32
        );
    } while(current.u32 != expected.u32);
}

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
void __attribute__((always_inline)) atomic_add_double(
    volatile global double* addr, const double val
) {
    union {
        ulong  u64;
        double f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        expected.f64 = current.f64;
        next.f64 = expected.f64 + val;
        current.u64 = atom_cmpxchg(
            (volatile global ulong*)addr, expected.u64, next.u64
        );
    } while(current.u64 != expected.u64);
}
#endif



// Kernels

__kernel void build_noise_weighted(
    int n_det,
    long n_sample,
    long first_sample,
    __global int const * pixels_index,
    __global long const * pixels,
    __global int const * weights_index,
    __global double const * weights,
    __global int const * det_data_index,
    __global double const * det_data,
    __global int const * det_flags_index,
    __global unsigned char const * det_flags,
    __global unsigned char const * shared_flags,
    __global double * zmap,
    __global long const * global2local,
    __global double const * det_scale,
    long nnz,
    long npix_submap,
    unsigned char det_flag_mask,
    unsigned char shared_flag_mask,
    unsigned char use_shared_flags,
    unsigned char use_det_flags
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    int d_indx = det_data_index[idet];
    int p_indx = pixels_index[idet];
    int w_indx = weights_index[idet];
    int f_indx = det_flags_index[idet];

    size_t woff = (w_indx * nnz * n_sample) + nnz * isamp;
    size_t poff = p_indx * n_sample + isamp;
    size_t doff = d_indx * n_sample + isamp;
    size_t foff = f_indx * n_sample + isamp;

    long global_submap;
    long local_submap_pix;
    long local_submap;
    long local_pix;
    long zoff;

    double scaled_data;
    double map_val;

    unsigned char det_check = 0;
    if (use_det_flags) {
        det_check = det_flags[foff] & det_flag_mask;
    }
    unsigned char shared_check = 0;
    if (use_shared_flags) {
        shared_check = shared_flags[isamp] & shared_flag_mask;
    }

    if (
        (pixels[poff] >= 0) &&
        (det_check == 0) &&
        (shared_check == 0)
    ) {
        global_submap = (long)(pixels[poff] / npix_submap);
        local_submap_pix = pixels[poff] - global_submap * npix_submap;
        local_submap = global2local[global_submap];
        local_pix = local_submap * npix_submap + local_submap_pix;
        zoff = nnz * local_pix;

        scaled_data = det_scale[d_indx] * det_data[doff];
        for (long i = 0; i < nnz; i++) {
            atomic_add_double(&(zmap[zoff + i]), scaled_data * weights[woff + i]);
        }
    }

    return;
}



