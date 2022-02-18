// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>


void qa_normalize_inplace(size_t n, double * q) {
    for (size_t i = 0; i < n; ++i) {
        size_t off = 4 * i;
        double norm = 0.0;
        for (size_t j = 0; j < 4; ++j) {
            norm += q[off + j] * q[off + j];
        }
        norm = 1.0 / sqrt(norm);
        for (size_t j = 0; j < 4; ++j) {
            q[off + j] *= norm;
        }
    }
    return;
}

void qa_rotate(double const * q_in, double const * v_in,
               double * v_out) {
    // The input quaternion has already been normalized on the host.

    double xw =  q_in[3] * q_in[0];
    double yw =  q_in[3] * q_in[1];
    double zw =  q_in[3] * q_in[2];
    double x2 = -q_in[0] * q_in[0];
    double xy =  q_in[0] * q_in[1];
    double xz =  q_in[0] * q_in[2];
    double y2 = -q_in[1] * q_in[1];
    double yz =  q_in[1] * q_in[2];
    double z2 = -q_in[2] * q_in[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                    (yw + xz) * v_in[2]) + v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) + v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) + v_in[2];

    return;
}

void qa_mult(double const * p, double const * q, double * r) {
    r[0] =  p[0] * q[3] + p[1] * q[2] -
           p[2] * q[1] + p[3] * q[0];
    r[1] = -p[0] * q[2] + p[1] * q[3] +
           p[2] * q[0] + p[3] * q[1];
    r[2] =  p[0] * q[1] - p[1] * q[0] +
           p[2] * q[3] + p[3] * q[2];
    r[3] = -p[0] * q[0] - p[1] * q[1] -
           p[2] * q[2] + p[3] * q[3];
    return;
}
