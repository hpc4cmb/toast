// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

void pointing_detector_qa_mult(double const * p, double const * q, double * r) {
    r[0] = p[0] * q[3] + p[1] * q[2] -
           p[2] * q[1] + p[3] * q[0];
    r[1] = -p[0] * q[2] + p[1] * q[3] +
           p[2] * q[0] + p[3] * q[1];
    r[2] = p[0] * q[1] - p[1] * q[0] +
           p[2] * q[3] + p[3] * q[2];
    r[3] = -p[0] * q[0] - p[1] * q[1] -
           p[2] * q[2] + p[3] * q[3];
    return;
}

__kernel void pointing_detector(
    int n_det,
    long n_sample,
    long first_sample,
    __global double const * focalplane,
    __global double const * boresight,
    __global int const * quat_index,
    __global double * quats,
    __global unsigned char const * shared_flags,
    unsigned char shared_flag_mask,
    unsigned char use_flags
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    int qidx = quat_index[idet];

    // Copy to private variables in order to pass to subroutines.
    double temp_bore[4];
    double temp_fp[4];
    double temp_quat[4];

    unsigned char check = 0;
    if (use_flags != 0) {
        check = shared_flags[isamp] & shared_flag_mask;
    }

    if (check == 0) {
        temp_bore[0] = boresight[4 * isamp];
        temp_bore[1] = boresight[4 * isamp + 1];
        temp_bore[2] = boresight[4 * isamp + 2];
        temp_bore[3] = boresight[4 * isamp + 3];
    } else {
        temp_bore[0] = 0.0;
        temp_bore[1] = 0.0;
        temp_bore[2] = 0.0;
        temp_bore[3] = 1.0;
    }
    temp_fp[0] = focalplane[4 * idet];
    temp_fp[1] = focalplane[4 * idet + 1];
    temp_fp[2] = focalplane[4 * idet + 2];
    temp_fp[3] = focalplane[4 * idet + 3];

    pointing_detector_qa_mult(temp_bore, temp_fp, temp_quat);

    quats[(qidx * 4 * n_sample) + 4 * isamp] = temp_quat[0];
    quats[(qidx * 4 * n_sample) + 4 * isamp + 1] = temp_quat[1];
    quats[(qidx * 4 * n_sample) + 4 * isamp + 2] = temp_quat[2];
    quats[(qidx * 4 * n_sample) + 4 * isamp + 3] = temp_quat[3];
}

