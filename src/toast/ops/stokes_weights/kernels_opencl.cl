// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

void stokes_weights_qa_rotate(
    double const * q_in,
    double const * v_in,
    double * v_out
) {
    // The input quaternion has already been normalized on the host.

    double xw = q_in[3] * q_in[0];
    double yw = q_in[3] * q_in[1];
    double zw = q_in[3] * q_in[2];
    double x2 = -q_in[0] * q_in[0];
    double xy = q_in[0] * q_in[1];
    double xz = q_in[0] * q_in[2];
    double y2 = -q_in[1] * q_in[1];
    double yz = q_in[1] * q_in[2];
    double z2 = -q_in[2] * q_in[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                    (yw + xz) * v_in[2]) + v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) + v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) + v_in[2];

    return;
}

void stokes_weights_alpha(
    double const * quats,
    double * alpha
) {
    const double xaxis[3] = {1.0, 0.0, 0.0};
    const double zaxis[3] = {0.0, 0.0, 1.0};
    double vd[3];
    double vo[3];

    stokes_weights_qa_rotate(quats, zaxis, vd);
    stokes_weights_qa_rotate(quats, xaxis, vo);

    double ang_xy = atan2(vd[1], vd[0]);
    double vm_x = vd[2] * cos(ang_xy);
    double vm_y = vd[2] * sin(ang_xy);
    double vm_z = - sqrt(1.0 - vd[2] * vd[2]);

    double alpha_y = (
        vd[0] * (vm_y * vo[2] - vm_z * vo[1]) - vd[1] * (vm_x * vo[2] - vm_z * vo[0]) +
        vd[2] * (vm_x * vo[1] - vm_y * vo[0])
    );
    double alpha_x = (vm_x * vo[0] + vm_y * vo[1] + vm_z * vo[2]);

    (*alpha) = atan2(alpha_y, alpha_x);
    return;
}


// Kernels

__kernel void stokes_weights_IQU(
    int n_det,
    long n_sample,
    long first_sample,
    __global int const * quat_index,
    __global double const * quats,
    __global int const * weight_index,
    __global double * weights,
    __global double const * epsilon,
    __global double const * gamma,
    __global double const * cal,
    double U_sign,
    unsigned char IAU
) {
    // NOTE:  Flags are not needed here, since the quaternions
    // have already had bad samples converted to null rotations.

    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = get_global_id(1);

    double eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet]);
    int q_indx = quat_index[idet];
    int w_indx = weight_index[idet];
    long qoff = (q_indx * 4 * n_sample) + 4 * isamp;

    // Copy to private variable in order to pass to subroutines.
    double temp_quat[4];
    temp_quat[0] = quats[qoff];
    temp_quat[1] = quats[qoff + 1];
    temp_quat[2] = quats[qoff + 2];
    temp_quat[3] = quats[qoff + 3];

    double alpha;
    stokes_weights_alpha(temp_quat, &alpha);

    alpha *= 2.0;
    double cang = cos(alpha);
    double sang = sin(alpha);

    long woff = (w_indx * 3 * n_sample) + 3 * isamp;
    weights[woff] = cal[idet];
    weights[woff + 1] = cang * eta * cal[idet];
    weights[woff + 2] = sang * eta * cal[idet] * U_sign;

    return;
}

__kernel void stokes_weights_IQU_hwp(
    int n_det,
    long n_sample,
    long first_sample,
    __global int const * quat_index,
    __global double const * quats,
    __global int const * weight_index,
    __global double * weights,
    __global double const * hwp,
    __global double const * epsilon,
    __global double const * gamma,
    __global double const * cal,
    double U_sign,
    unsigned char IAU
) {
    // NOTE:  Flags are not needed here, since the quaternions
    // have already had bad samples converted to null rotations.

    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    double eta = (1.0 - epsilon[idet]) / (1.0 + epsilon[idet]);
    int q_indx = quat_index[idet];
    int w_indx = weight_index[idet];
    long qoff = (q_indx * 4 * n_sample) + 4 * isamp;

    // Copy to private variable in order to pass to subroutines.
    double temp_quat[4];
    temp_quat[0] = quats[qoff];
    temp_quat[1] = quats[qoff + 1];
    temp_quat[2] = quats[qoff + 2];
    temp_quat[3] = quats[qoff + 3];

    double alpha;
    stokes_weights_alpha(temp_quat, &alpha);

    double ang = 2.0 * (2.0 * (gamma[idet] - hwp[isamp]) - alpha);
    double cang = cos(ang);
    double sang = sin(ang);

    long woff = (w_indx * 3 * n_sample) + 3 * isamp;
    weights[woff] = cal[idet];
    weights[woff + 1] = cang * eta * cal[idet];
    weights[woff + 2] = -sang * eta * cal[idet] * U_sign;

    return;
}

__kernel void stokes_weights_I(
    int n_det,
    long n_sample,
    long first_sample,
    __global int const * weight_index,
    __global double * weights,
    __global double const * cal
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    int w_indx = weight_index[idet];

    long woff = (w_indx * n_sample) + isamp;
    weights[woff] = cal[idet];

    return;
}
