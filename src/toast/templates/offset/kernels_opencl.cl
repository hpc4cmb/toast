// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.


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

__kernel void offset_add_to_signal(
    long n_sample,
    long first_sample,
    long step_length,
    long amp_offset,
    __global double const * amplitudes,
    __global unsigned char const * amplitude_flags,
    int det_data_index,
    __global double * det_data,
    unsigned char use_amp_flags
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    size_t doff = det_data_index * n_sample + isamp;
    long amp = amp_offset + (long)(isamp / step_length);

    unsigned char amp_check = 0;
    if (use_amp_flags) {
        amp_check = amplitude_flags[amp];
    }

    if (amp_check == 0) {
        det_data[doff] += amplitudes[amp];
    }
    return;
}


__kernel void offset_project_signal(
    long n_sample,
    long first_sample,
    int det_data_index,
    __global double const * det_data,
    int det_flag_index,
    __global unsigned char const * det_flags,
    unsigned char flag_mask,
    long step_length,
    long amp_offset,
    __global double * amplitudes,
    __global unsigned char const * amplitude_flags,
    unsigned char use_det_flags,
    unsigned char use_amp_flags
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    size_t doff = det_data_index * n_sample + isamp;
    long amp = amp_offset + (long)(isamp / step_length);

    unsigned char det_check = 0;
    if (use_det_flags) {
        det_check = det_flags[doff] & flag_mask;
    }
    unsigned char amp_check = 0;
    if (use_amp_flags) {
        amp_check = amplitude_flags[amp];
    }

    if ((det_check == 0) && (amp_check == 0)) {
        atomic_add_double(&(amplitudes[amp]), det_data[doff]);
    }
    return;
}

__kernel void offset_apply_diag_precond(
    __global double const * amplitudes_in,
    __global double * amplitudes_out,
    __global double const * offset_var,
    __global unsigned char const * amplitude_flags,
    unsigned char use_amp_flags
) {
    int iamp = get_global_id(0);

    unsigned char amp_check = 0;
    if (use_amp_flags) {
        amp_check = amplitude_flags[iamp];
    }
    if (amp_check == 0) {
        amplitudes_out[iamp] = offset_var[iamp] * amplitudes_in[iamp];
    }
    return;
}
