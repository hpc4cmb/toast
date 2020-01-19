
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.


#include <toast/sys_utils.hpp>
#include <toast/math_sf.hpp>
#include <toast/math_qarray.hpp>

#include <cmath>
#include <vector>


// Dot product of lists of arrays.

void toast::qa_list_dot(size_t n, size_t m, size_t d, double const * a,
                        double const * b, double * dotprod) {
    if (toast::is_aligned(a) && toast::is_aligned(b) &&
        toast::is_aligned(dotprod)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            dotprod[i] = 0.0;
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                dotprod[i] += a[off + j] * b[off + j];
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            dotprod[i] = 0.0;
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                dotprod[i] += a[off + j] * b[off + j];
            }
        }
    }
    return;
}

// Inverse of a quaternion array.

void toast::qa_inv(size_t n, double * q) {
    if (toast::is_aligned(q)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                q[4 * i + j] *= -1;
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                q[4 * i + j] *= -1;
            }
        }
    }
    return;
}

// Norm of quaternion array

#pragma omp declare simd
void toast::qa_amplitude_one(size_t d, double const * v,
                             double * norm) {
    double dotprod = 0.0;
    for (size_t i = 0; i < d; ++i) {
        dotprod += v[i] * v[i];
    }
    (*norm) = ::sqrt(dotprod);
    return;
}

void toast::qa_amplitude(size_t n, size_t m, size_t d, double const * v,
                         double * norm) {
    toast::AlignedVector <double> temp(n);

    toast::qa_list_dot(n, m, d, v, v, temp.data());

    toast::vsqrt(n, temp.data(), norm);

    return;
}

// Normalize quaternion array.

#pragma omp declare simd
void toast::qa_normalize_one(size_t d, double const * q_in,
                             double * q_out) {
    double norm = 0.0;
    for (size_t i = 0; i < d; ++i) {
        norm += q_in[i] * q_in[i];
    }
    norm = 1.0 / ::sqrt(norm);
    for (size_t i = 0; i < d; ++i) {
        q_out[i] = q_in[i] * norm;
    }
    return;
}

void toast::qa_normalize(size_t n, size_t m, size_t d,
                         double const * q_in, double * q_out) {
    toast::AlignedVector <double> norm(n);

    toast::qa_amplitude(n, m, d, q_in, norm.data());

    if (toast::is_aligned(q_in) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                q_out[off + j] = q_in[off + j] / norm[i];
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                q_out[off + j] = q_in[off + j] / norm[i];
            }
        }
    }

    return;
}

// Normalize quaternion array in place.

#pragma omp declare simd
void toast::qa_normalize_inplace_one(size_t d, double * q) {
    double norm = 0.0;
    for (size_t i = 0; i < d; ++i) {
        norm += q[i] * q[i];
    }
    norm = 1.0 / ::sqrt(norm);
    for (size_t i = 0; i < d; ++i) {
        q[i] *= norm;
    }
    return;
}

void toast::qa_normalize_inplace(size_t n, size_t m, size_t d, double * q) {
    toast::AlignedVector <double> norm(n);

    toast::qa_amplitude(n, m, d, q, norm.data());

    if (toast::is_aligned(q)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                q[m * i + j] /= norm[i];
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                q[m * i + j] /= norm[i];
            }
        }
    }

    return;
}

// Rotate an array of vectors by an array of quaternions.

#pragma omp declare simd
void toast::qa_rotate_one_one(double const * q, double const * v_in,
                              double * v_out) {
    double norm = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        norm += q[i] * q[i];
    }
    norm = 1.0 / ::sqrt(norm);
    double q_unit[4];
    for (size_t i = 0; i < 4; ++i) {
        q_unit[i] = q[i] * norm;
    }

    double xw =  q_unit[3] * q_unit[0];
    double yw =  q_unit[3] * q_unit[1];
    double zw =  q_unit[3] * q_unit[2];
    double x2 = -q_unit[0] * q_unit[0];
    double xy =  q_unit[0] * q_unit[1];
    double xz =  q_unit[0] * q_unit[2];
    double y2 = -q_unit[1] * q_unit[1];
    double yz =  q_unit[1] * q_unit[2];
    double z2 = -q_unit[2] * q_unit[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                    (yw + xz) * v_in[2]) + v_in[0];
    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) + v_in[1];
    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) + v_in[2];

    return;
}

void toast::qa_rotate_many_one(size_t nq, double const * q,
                               double const * v_in, double * v_out) {
    toast::AlignedVector <double> q_unit(4 * nq);

    toast::qa_normalize(nq, 4, 4, q, q_unit.data());

    if (toast::is_aligned(v_in) && toast::is_aligned(v_out)) {
        #pragma omp simd
        for (size_t i = 0; i < nq; ++i) {
            size_t vfout = 3 * i;
            size_t qf = 4 * i;
            double xw =  q_unit[qf + 3] * q_unit[qf + 0];
            double yw =  q_unit[qf + 3] * q_unit[qf + 1];
            double zw =  q_unit[qf + 3] * q_unit[qf + 2];
            double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
            double xy =  q_unit[qf + 0] * q_unit[qf + 1];
            double xz =  q_unit[qf + 0] * q_unit[qf + 2];
            double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
            double yz =  q_unit[qf + 1] * q_unit[qf + 2];
            double z2 = -q_unit[qf + 2] * q_unit[qf + 2];

            v_out[vfout + 0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                                    (yw + xz) * v_in[2]) + v_in[0];

            v_out[vfout + 1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                                    (yz - xw) * v_in[2]) + v_in[1];

            v_out[vfout + 2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                                    (x2 + y2) * v_in[2]) + v_in[2];
        }
    } else {
        for (size_t i = 0; i < nq; ++i) {
            size_t vfout = 3 * i;
            size_t qf = 4 * i;
            double xw =  q_unit[qf + 3] * q_unit[qf + 0];
            double yw =  q_unit[qf + 3] * q_unit[qf + 1];
            double zw =  q_unit[qf + 3] * q_unit[qf + 2];
            double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
            double xy =  q_unit[qf + 0] * q_unit[qf + 1];
            double xz =  q_unit[qf + 0] * q_unit[qf + 2];
            double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
            double yz =  q_unit[qf + 1] * q_unit[qf + 2];
            double z2 = -q_unit[qf + 2] * q_unit[qf + 2];

            v_out[vfout + 0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                                    (yw + xz) * v_in[2]) + v_in[0];

            v_out[vfout + 1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                                    (yz - xw) * v_in[2]) + v_in[1];

            v_out[vfout + 2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                                    (x2 + y2) * v_in[2]) + v_in[2];
        }
    }

    return;
}

void toast::qa_rotate_one_many(double const * q, size_t nv,
                               double const * v_in, double * v_out) {
    double norm = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        norm += q[i] * q[i];
    }
    norm = 1.0 / ::sqrt(norm);
    double q_unit[4];
    for (size_t i = 0; i < 4; ++i) {
        q_unit[i] = q[i] * norm;
    }

    double xw =  q_unit[3] * q_unit[0];
    double yw =  q_unit[3] * q_unit[1];
    double zw =  q_unit[3] * q_unit[2];
    double x2 = -q_unit[0] * q_unit[0];
    double xy =  q_unit[0] * q_unit[1];
    double xz =  q_unit[0] * q_unit[2];
    double y2 = -q_unit[1] * q_unit[1];
    double yz =  q_unit[1] * q_unit[2];
    double z2 = -q_unit[2] * q_unit[2];

    if (toast::is_aligned(v_in) && toast::is_aligned(v_out)) {
        #pragma omp simd
        for (size_t i = 0; i < nv; ++i) {
            size_t vf = 3 * i;
            v_out[vf + 0] = 2 *
                            ((y2 + z2) * v_in[vf + 0] + (xy - zw) *
                             v_in[vf + 1] + (yw + xz) *
                             v_in[vf + 2]) + v_in[vf + 0];
            v_out[vf + 1] = 2 *
                            ((zw + xy) * v_in[vf + 0] + (x2 + z2) *
                             v_in[vf + 1] + (yz - xw) *
                             v_in[vf + 2]) + v_in[vf + 1];
            v_out[vf + 2] = 2 *
                            ((xz - yw) * v_in[vf + 0] + (xw + yz) *
                             v_in[vf + 1] + (x2 + y2) *
                             v_in[vf + 2]) + v_in[vf + 2];
        }
    } else {
        for (size_t i = 0; i < nv; ++i) {
            size_t vf = 3 * i;
            v_out[vf + 0] = 2 *
                            ((y2 + z2) * v_in[vf + 0] + (xy - zw) *
                             v_in[vf + 1] + (yw + xz) *
                             v_in[vf + 2]) + v_in[vf + 0];
            v_out[vf + 1] = 2 *
                            ((zw + xy) * v_in[vf + 0] + (x2 + z2) *
                             v_in[vf + 1] + (yz - xw) *
                             v_in[vf + 2]) + v_in[vf + 1];
            v_out[vf + 2] = 2 *
                            ((xz - yw) * v_in[vf + 0] + (xw + yz) *
                             v_in[vf + 1] + (x2 + y2) *
                             v_in[vf + 2]) + v_in[vf + 2];
        }
    }

    return;
}

void toast::qa_rotate_many_many(size_t n, double const * q,
                                double const * v_in, double * v_out) {
    toast::AlignedVector <double> q_unit(4 * n);

    toast::qa_normalize(n, 4, 4, q, q_unit.data());

    if (toast::is_aligned(v_in) && toast::is_aligned(v_out)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t vf = 3 * i;
            size_t qf = 4 * i;
            double xw =  q_unit[qf + 3] * q_unit[qf + 0];
            double yw =  q_unit[qf + 3] * q_unit[qf + 1];
            double zw =  q_unit[qf + 3] * q_unit[qf + 2];
            double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
            double xy =  q_unit[qf + 0] * q_unit[qf + 1];
            double xz =  q_unit[qf + 0] * q_unit[qf + 2];
            double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
            double yz =  q_unit[qf + 1] * q_unit[qf + 2];
            double z2 = -q_unit[qf + 2] * q_unit[qf + 2];
            v_out[vf + 0] = 2 * ((y2 + z2) * v_in[vf + 0] +
                                 (xy - zw) * v_in[vf + 1] + (yw + xz) *
                                 v_in[vf + 2])
                            + v_in[vf + 0];
            v_out[vf + 1] = 2 * ((zw + xy) * v_in[vf + 0] +
                                 (x2 + z2) * v_in[vf + 1] + (yz - xw) *
                                 v_in[vf + 2])
                            + v_in[vf + 1];
            v_out[vf + 2] = 2 * ((xz - yw) * v_in[vf + 0] +
                                 (xw + yz) * v_in[vf + 1] + (x2 + y2) *
                                 v_in[vf + 2])
                            + v_in[vf + 2];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t vf = 3 * i;
            size_t qf = 4 * i;
            double xw =  q_unit[qf + 3] * q_unit[qf + 0];
            double yw =  q_unit[qf + 3] * q_unit[qf + 1];
            double zw =  q_unit[qf + 3] * q_unit[qf + 2];
            double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
            double xy =  q_unit[qf + 0] * q_unit[qf + 1];
            double xz =  q_unit[qf + 0] * q_unit[qf + 2];
            double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
            double yz =  q_unit[qf + 1] * q_unit[qf + 2];
            double z2 = -q_unit[qf + 2] * q_unit[qf + 2];
            v_out[vf + 0] = 2 * ((y2 + z2) * v_in[vf + 0] +
                                 (xy - zw) * v_in[vf + 1] + (yw + xz) *
                                 v_in[vf + 2])
                            + v_in[vf + 0];
            v_out[vf + 1] = 2 * ((zw + xy) * v_in[vf + 0] +
                                 (x2 + z2) * v_in[vf + 1] + (yz - xw) *
                                 v_in[vf + 2])
                            + v_in[vf + 1];
            v_out[vf + 2] = 2 * ((xz - yw) * v_in[vf + 0] +
                                 (xw + yz) * v_in[vf + 1] + (x2 + y2) *
                                 v_in[vf + 2])
                            + v_in[vf + 2];
        }
    }

    return;
}

void toast::qa_rotate(size_t nq, double const * q, size_t nv,
                      double const * v_in, double * v_out) {
    if ((nq == 1) && (nv == 1)) {
        toast::qa_rotate_one_one(q, v_in, v_out);
    } else if (nq == 1) {
        toast::qa_rotate_one_many(q, nv, v_in, v_out);
    } else if (nv == 1) {
        toast::qa_rotate_many_one(nq, q, v_in, v_out);
    } else if (nq == nv) {
        toast::qa_rotate_many_many(nq, q, v_in, v_out);
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("incompatible quaternion and vector array dimensions");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    return;
}

// Multiply arrays of quaternions.

#pragma omp declare simd
void toast::qa_mult_one_one(double const * p, double const * q,
                            double * r) {
    r[0] =  p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
    r[1] = -p[0] * q[2] + p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
    r[2] =  p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2];
    r[3] = -p[0] * q[0] - p[1] * q[1] - p[2] * q[2] + p[3] * q[3];

    return;
}

void toast::qa_mult_one_many(double const * p, size_t nq,
                             double const * q, double * r) {
    if (toast::is_aligned(p) && toast::is_aligned(q) && toast::is_aligned(r)) {
        #pragma omp simd
        for (size_t i = 0; i < nq; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[0] * q[f + 3] + p[1] * q[f + 2] - p[2] * q[f + 1] +
                       p[3] * q[f + 0];
            r[f + 1] = -p[0] * q[f + 2] + p[1] * q[f + 3] + p[2] * q[f + 0] +
                       p[3] * q[f + 1];
            r[f + 2] =  p[0] * q[f + 1] - p[1] * q[f + 0] + p[2] * q[f + 3] +
                       p[3] * q[f + 2];
            r[f + 3] = -p[0] * q[f + 0] - p[1] * q[f + 1] - p[2] * q[f + 2] +
                       p[3] * q[f + 3];
        }
    } else {
        for (size_t i = 0; i < nq; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[0] * q[f + 3] + p[1] * q[f + 2] - p[2] * q[f + 1] +
                       p[3] * q[f + 0];
            r[f + 1] = -p[0] * q[f + 2] + p[1] * q[f + 3] + p[2] * q[f + 0] +
                       p[3] * q[f + 1];
            r[f + 2] =  p[0] * q[f + 1] - p[1] * q[f + 0] + p[2] * q[f + 3] +
                       p[3] * q[f + 2];
            r[f + 3] = -p[0] * q[f + 0] - p[1] * q[f + 1] - p[2] * q[f + 2] +
                       p[3] * q[f + 3];
        }
    }

    return;
}

void toast::qa_mult_many_one(size_t np, double const * p,
                             double const * q, double * r) {
    if (toast::is_aligned(p) && toast::is_aligned(q) && toast::is_aligned(r)) {
        #pragma omp simd
        for (size_t i = 0; i < np; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[f + 0] * q[3] + p[f + 1] * q[2] -
                       p[f + 2] * q[1] + p[f + 3] * q[0];
            r[f + 1] = -p[f + 0] * q[2] + p[f + 1] * q[3] +
                       p[f + 2] * q[0] + p[f + 3] * q[1];
            r[f + 2] =  p[f + 0] * q[1] - p[f + 1] * q[0] +
                       p[f + 2] * q[3] + p[f + 3] * q[2];
            r[f + 3] = -p[f + 0] * q[0] - p[f + 1] * q[1] -
                       p[f + 2] * q[2] + p[f + 3] * q[3];
        }
    } else {
        for (size_t i = 0; i < np; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[f + 0] * q[3] + p[f + 1] * q[2] -
                       p[f + 2] * q[1] + p[f + 3] * q[0];
            r[f + 1] = -p[f + 0] * q[2] + p[f + 1] * q[3] +
                       p[f + 2] * q[0] + p[f + 3] * q[1];
            r[f + 2] =  p[f + 0] * q[1] - p[f + 1] * q[0] +
                       p[f + 2] * q[3] + p[f + 3] * q[2];
            r[f + 3] = -p[f + 0] * q[0] - p[f + 1] * q[1] -
                       p[f + 2] * q[2] + p[f + 3] * q[3];
        }
    }

    return;
}

void toast::qa_mult_many_many(size_t np, double const * p, size_t nq,
                              double const * q, double * r) {
    if (toast::is_aligned(p) && toast::is_aligned(q) && toast::is_aligned(r)) {
        #pragma omp simd
        for (size_t i = 0; i < np; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[f + 0] * q[f + 3] + p[f + 1] * q[f + 2] -
                       p[f + 2] * q[f + 1] + p[f + 3] * q[f + 0];
            r[f + 1] = -p[f + 0] * q[f + 2] + p[f + 1] * q[f + 3] +
                       p[f + 2] * q[f + 0] + p[f + 3] * q[f + 1];
            r[f + 2] =  p[f + 0] * q[f + 1] - p[f + 1] * q[f + 0] +
                       p[f + 2] * q[f + 3] + p[f + 3] * q[f + 2];
            r[f + 3] = -p[f + 0] * q[f + 0] - p[f + 1] * q[f + 1] -
                       p[f + 2] * q[f + 2] + p[f + 3] * q[f + 3];
        }
    } else {
        for (size_t i = 0; i < np; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[f + 0] * q[f + 3] + p[f + 1] * q[f + 2] -
                       p[f + 2] * q[f + 1] + p[f + 3] * q[f + 0];
            r[f + 1] = -p[f + 0] * q[f + 2] + p[f + 1] * q[f + 3] +
                       p[f + 2] * q[f + 0] + p[f + 3] * q[f + 1];
            r[f + 2] =  p[f + 0] * q[f + 1] - p[f + 1] * q[f + 0] +
                       p[f + 2] * q[f + 3] + p[f + 3] * q[f + 2];
            r[f + 3] = -p[f + 0] * q[f + 0] - p[f + 1] * q[f + 1] -
                       p[f + 2] * q[f + 2] + p[f + 3] * q[f + 3];
        }
    }

    return;
}

void toast::qa_mult(size_t np, double const * p, size_t nq,
                    double const * q, double * r) {
    if ((np == 1) && (nq == 1)) {
        toast::qa_mult_one_one(p, q, r);
    } else if (np == 1) {
        toast::qa_mult_one_many(p, nq, q, r);
    } else if (nq == 1) {
        toast::qa_mult_many_one(np, p, q, r);
    } else if (np == nq) {
        toast::qa_mult_many_many(np, p, nq, q, r);
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("incompatible quaternion array dimensions");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    return;
}

// Spherical interpolation of quaternion array from time to targettime.

void toast::qa_slerp(size_t n_time, size_t n_targettime,
                     double const * time, double const * targettime,
                     double const * q_in, double * q_interp) {
    #pragma \
    omp parallel default(none) shared(n_time, n_targettime, time, targettime, q_in, q_interp)
    {
        double diff;
        double frac;
        double costheta;
        double const * qlow;
        double const * qhigh;
        double theta;
        double invsintheta;
        double norm;
        double * q;
        double ratio1;
        double ratio2;

        size_t off = 0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_targettime; ++i) {
            // scroll forward to the correct time sample
            while ((off + 1 < n_time) && (time[off + 1] < targettime[i])) {
                ++off;
            }
            diff = time[off + 1] - time[off];
            frac = (targettime[i] - time[off]) / diff;

            qlow = &(q_in[4 * off]);
            qhigh = &(q_in[4 * (off + 1)]);
            q = &(q_interp[4 * i]);

            costheta = qlow[0] * qhigh[0] + qlow[1] * qhigh[1] + qlow[2] *
                       qhigh[2] + qlow[3] * qhigh[3];

            if (::fabs(costheta - 1.0) < 1.0e-10) {
                q[0] = qlow[0];
                q[1] = qlow[1];
                q[2] = qlow[2];
                q[3] = qlow[3];
            } else {
                theta = ::acos(costheta);
                invsintheta = 1.0 / ::sqrt(1.0 - costheta * costheta);
                ratio1 = ::sin((1.0 - frac) * theta) * invsintheta;
                ratio2 = ::sin(frac * theta) * invsintheta;
                q[0] = ratio1 * qlow[0] + ratio2 * qhigh[0];
                q[1] = ratio1 * qlow[1] + ratio2 * qhigh[1];
                q[2] = ratio1 * qlow[2] + ratio2 * qhigh[2];
                q[3] = ratio1 * qlow[3] + ratio2 * qhigh[3];
            }

            norm = 1.0 / ::sqrt(
                q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            q[0] *= norm;
            q[1] *= norm;
            q[2] *= norm;
            q[3] *= norm;
        }
    }

    return;
}

// Exponential of a quaternion array.

void toast::qa_exp(size_t n, double const * q_in, double * q_out) {
    toast::AlignedVector <double> normv(n);

    toast::qa_amplitude(n, 4, 3, q_in, normv.data());

    if (toast::is_aligned(q_in) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 4 * i;
            double exp_q_w = ::exp(q_in[off + 3]);
            q_out[off + 3] = exp_q_w * ::cos(normv[i]);
            exp_q_w /= normv[i];
            exp_q_w *= ::sin(normv[i]);
            q_out[off] = exp_q_w * q_in[off];
            q_out[off + 1] = exp_q_w * q_in[off + 1];
            q_out[off + 2] = exp_q_w * q_in[off + 2];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t off = 4 * i;
            double exp_q_w = ::exp(q_in[off + 3]);
            q_out[off + 3] = exp_q_w * ::cos(normv[i]);
            exp_q_w /= normv[i];
            exp_q_w *= ::sin(normv[i]);
            q_out[off] = exp_q_w * q_in[off];
            q_out[off + 1] = exp_q_w * q_in[off + 1];
            q_out[off + 2] = exp_q_w * q_in[off + 2];
        }
    }

    return;
}

// Natural logarithm of a quaternion array.

void toast::qa_ln(size_t n, double const * q_in, double * q_out) {
    toast::AlignedVector <double> normq(n);

    toast::qa_amplitude(n, 4, 4, q_in, normq.data());

    toast::qa_normalize(n, 4, 3, q_in, q_out);

    if (toast::is_aligned(q_in) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 4 * i;
            q_out[off + 3] = ::log(normq[i]);
            double tmp = ::acos(q_in[off + 3] / normq[i]);
            q_out[off] *= tmp;
            q_out[off + 1] *= tmp;
            q_out[off + 2] *= tmp;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t off = 4 * i;
            q_out[off + 3] = ::log(normq[i]);
            double tmp = ::acos(q_in[off + 3] / normq[i]);
            q_out[off] *= tmp;
            q_out[off + 1] *= tmp;
            q_out[off + 2] *= tmp;
        }
    }

    return;
}

// Real power of quaternion array

void toast::qa_pow(size_t nq, size_t np, double const * p, double const * q_in,
                   double * q_out) {
    toast::AlignedVector <double> q_tmp(4 * nq);

    toast::qa_ln(nq, q_in, q_tmp.data());

    if (np == 1) {
        #pragma omp simd
        for (size_t i = 0; i < nq; ++i) {
            size_t off = 4 * i;
            q_tmp[off] *= p[0];
            q_tmp[off + 1] *= p[0];
            q_tmp[off + 2] *= p[0];
            q_tmp[off + 3] *= p[0];
        }
    } else if (np == nq) {
        if (toast::is_aligned(p)) {
            #pragma omp simd
            for (size_t i = 0; i < nq; ++i) {
                size_t off = 4 * i;
                q_tmp[off] *= p[i];
                q_tmp[off + 1] *= p[i];
                q_tmp[off + 2] *= p[i];
                q_tmp[off + 3] *= p[i];
            }
        } else {
            for (size_t i = 0; i < nq; ++i) {
                size_t off = 4 * i;
                q_tmp[off] *= p[i];
                q_tmp[off + 1] *= p[i];
                q_tmp[off + 2] *= p[i];
                q_tmp[off + 3] *= p[i];
            }
        }
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg(
            "Length of power exponent must be one or equal to\
                         the number of quaternions");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::qa_exp(nq, q_tmp.data(), q_out);

    return;
}

// Creates rotation quaternions of angles (in [rad]) around axes [already
// normalized]
// axis is an n by 3 array, angle is a n-array, q_out is a n by 4 array

void toast::qa_from_axisangle_one_one(double const * axis,
                                      double angle, double * q_out) {
    double half = 0.5 * angle;
    double sin_a = ::sin(half);
    q_out[0] = axis[0] * sin_a;
    q_out[1] = axis[1] * sin_a;
    q_out[2] = axis[2] * sin_a;
    q_out[3] = ::cos(half);
    return;
}

void toast::qa_from_axisangle_one_many(size_t nang, double const * axis,
                                       double const * angle, double * q_out) {
    toast::AlignedVector <double> a(nang);
    if (toast::is_aligned(angle)) {
        #pragma omp simd
        for (size_t i = 0; i < nang; ++i) {
            a[i] = 0.5 * angle[i];
        }
    } else {
        for (size_t i = 0; i < nang; ++i) {
            a[i] = 0.5 * angle[i];
        }
    }

    toast::AlignedVector <double> sin_a(nang);
    toast::AlignedVector <double> cos_a(nang);

    toast::vsincos(nang, a.data(), sin_a.data(), cos_a.data());

    if (toast::is_aligned(axis) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < nang; ++i) {
            size_t off = 4 * i;
            q_out[off] = axis[0] * sin_a[i];
            q_out[off + 1] = axis[1] * sin_a[i];
            q_out[off + 2] = axis[2] * sin_a[i];
            q_out[off + 3] = cos_a[i];
        }
    } else {
        for (size_t i = 0; i < nang; ++i) {
            size_t off = 4 * i;
            q_out[off] = axis[0] * sin_a[i];
            q_out[off + 1] = axis[1] * sin_a[i];
            q_out[off + 2] = axis[2] * sin_a[i];
            q_out[off + 3] = cos_a[i];
        }
    }
    return;
}

void toast::qa_from_axisangle_many_one(size_t naxis, double const * axis,
                                       double angle, double * q_out) {
    double sin_a = ::sin(0.5 * angle);
    double cos_a = ::cos(0.5 * angle);

    if (toast::is_aligned(axis) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < naxis; ++i) {
            size_t off = 4 * i;
            size_t voff = 3 * i;
            q_out[off] = axis[voff] * sin_a;
            q_out[off + 1] = axis[voff + 1] * sin_a;
            q_out[off + 2] = axis[voff + 2] * sin_a;
            q_out[off + 3] = cos_a;
        }
    } else {
        for (size_t i = 0; i < naxis; ++i) {
            size_t off = 4 * i;
            size_t voff = 3 * i;
            q_out[off] = axis[voff] * sin_a;
            q_out[off + 1] = axis[voff + 1] * sin_a;
            q_out[off + 2] = axis[voff + 2] * sin_a;
            q_out[off + 3] = cos_a;
        }
    }
    return;
}

void toast::qa_from_axisangle_many_many(size_t n, double const * axis,
                                        double const * angle, double * q_out) {
    toast::AlignedVector <double> a(n);

    if (toast::is_aligned(angle)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            a[i] = 0.5 * angle[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            a[i] = 0.5 * angle[i];
        }
    }

    toast::AlignedVector <double> sin_a(n);
    toast::AlignedVector <double> cos_a(n);

    toast::vsincos(n, a.data(), sin_a.data(), cos_a.data());

    if (toast::is_aligned(axis) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 4 * i;
            size_t voff = 3 * i;
            q_out[off] = axis[voff] * sin_a[i];
            q_out[off + 1] = axis[voff + 1] * sin_a[i];
            q_out[off + 2] = axis[voff + 2] * sin_a[i];
            q_out[off + 3] = cos_a[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t off = 4 * i;
            size_t voff = 3 * i;
            q_out[off] = axis[voff] * sin_a[i];
            q_out[off + 1] = axis[voff + 1] * sin_a[i];
            q_out[off + 2] = axis[voff + 2] * sin_a[i];
            q_out[off + 3] = cos_a[i];
        }
    }
    return;
}

void toast::qa_from_axisangle(size_t naxis, double const * axis, size_t nang,
                              double const * angle, double * q_out) {
    if ((naxis == 1) && (nang == 1)) {
        toast::qa_from_axisangle_one_one(axis, angle[0], q_out);
    } else if (naxis == 1) {
        toast::qa_from_axisangle_one_many(nang, axis, angle, q_out);
    } else if (nang == 1) {
        toast::qa_from_axisangle_many_one(naxis, axis, angle[0], q_out);
    } else if (naxis == nang) {
        toast::qa_from_axisangle_many_many(naxis, axis, angle, q_out);
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg(
            "number of axes and angles must either match or\
                        one of them must have a single element");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return;
}

// Returns the axis and angle of rotation of a quaternion.

void toast::qa_to_axisangle_one(double const * q, double * axis,
                                double * angle) {
    double tacos = ::acos(q[3]);

    (*angle) = 2.0 * tacos;

    if ((*angle) < 1e-10) {
        axis[0] = 0.0;
        axis[1] = 0.0;
        axis[2] = 0.0;
    } else {
        double tmp = 1.0 / ::sin(tacos);
        axis[0] = q[0] * tmp;
        axis[1] = q[1] * tmp;
        axis[2] = q[2] * tmp;
    }

    return;
}

void toast::qa_to_axisangle(size_t n, double const * q, double * axis,
                            double * angle) {
    if (toast::is_aligned(q) && toast::is_aligned(axis) &&
        toast::is_aligned(angle)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t qf = 4 * i;
            size_t vf = 3 * i;
            angle[i] = 2.0 * ::acos(q[qf + 3]);
            if (angle[i] < 1e-10) {
                axis[vf + 0] = 0.0;
                axis[vf + 1] = 0.0;
                axis[vf + 2] = 0.0;
            } else {
                double tmp = 1.0 / ::sin(0.5 * angle[i]);
                axis[vf + 0] = q[qf + 0] * tmp;
                axis[vf + 1] = q[qf + 1] * tmp;
                axis[vf + 2] = q[qf + 2] * tmp;
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t qf = 4 * i;
            size_t vf = 3 * i;
            angle[i] = 2.0 * ::acos(q[qf + 3]);
            if (angle[i] < 1e-10) {
                axis[vf + 0] = 0.0;
                axis[vf + 1] = 0.0;
                axis[vf + 2] = 0.0;
            } else {
                double tmp = 1.0 / ::sin(0.5 * angle[i]);
                axis[vf + 0] = q[qf + 0] * tmp;
                axis[vf + 1] = q[qf + 1] * tmp;
                axis[vf + 2] = q[qf + 2] * tmp;
            }
        }
    }
    return;
}

// Creates the rotation matrix corresponding to a quaternion.

void toast::qa_to_rotmat(double const * q, double * rotmat) {
    double xx = q[0] * q[0];
    double xy = q[0] * q[1];
    double xz = q[0] * q[2];
    double xw = q[0] * q[3];
    double yy = q[1] * q[1];
    double yz = q[1] * q[2];
    double yw = q[1] * q[3];
    double zz = q[2] * q[2];
    double zw = q[2] * q[3];

    rotmat[0] = 1 - 2 * (yy + zz);
    rotmat[1] =     2 * (xy - zw);
    rotmat[2] =     2 * (xz + yw);

    rotmat[3] =     2 * (xy + zw);
    rotmat[4] = 1 - 2 * (xx + zz);
    rotmat[5] =     2 * (yz - xw);

    rotmat[6] =     2 * (xz - yw);
    rotmat[7] =     2 * (yz + xw);
    rotmat[8] = 1 - 2 * (xx + yy);
    return;
}

// Creates the quaternion from a rotation matrix.

void toast::qa_from_rotmat(const double * rotmat, double * q) {
    double tr = rotmat[0] + rotmat[4] + rotmat[8];
    double S;
    double invS;
    if (tr > 0) {
        S = ::sqrt(tr + 1.0) * 2.0; /* S=4*qw */
        invS = 1.0 / S;
        q[0] = (rotmat[7] - rotmat[5]) * invS;
        q[1] = (rotmat[2] - rotmat[6]) * invS;
        q[2] = (rotmat[3] - rotmat[1]) * invS;
        q[3] = 0.25 * S;
    } else if ((rotmat[0] > rotmat[4]) && (rotmat[0] > rotmat[8])) {
        S = ::sqrt(1.0 + rotmat[0] - rotmat[4] - rotmat[8]) * 2.0; /* S=4*qx */
        invS = 1.0 / S;
        q[0] = 0.25 * S;
        q[1] = (rotmat[1] + rotmat[3]) * invS;
        q[2] = (rotmat[2] + rotmat[6]) * invS;
        q[3] = (rotmat[7] - rotmat[5]) * invS;
    } else if (rotmat[4] > rotmat[8]) {
        S = ::sqrt(1.0 + rotmat[4] - rotmat[0] - rotmat[8]) * 2.0; /* S=4*qy */
        invS = 1.0 / S;
        q[0] = (rotmat[1] + rotmat[3]) * invS;
        q[1] = 0.25 * S;
        q[2] = (rotmat[5] + rotmat[7]) * invS;
        q[3] = (rotmat[2] - rotmat[6]) * invS;
    } else {
        S = ::sqrt(1.0 + rotmat[8] - rotmat[0] - rotmat[4]) * 2.0; /* S=4*qz */
        invS = 1.0 / S;
        q[0] = (rotmat[2] + rotmat[6]) * invS;
        q[1] = (rotmat[5] + rotmat[7]) * invS;
        q[2] = 0.25 * S;
        q[3] = (rotmat[3] - rotmat[1]) * invS;
    }
    return;
}

// Creates the quaternion from normalized vectors.

void toast::qa_from_vectors(size_t n, double const * vec1,
                            double const * vec2, double * q) {
    if (toast::is_aligned(vec1) && toast::is_aligned(vec2) &&
        toast::is_aligned(q)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t vf = 3 * i;
            size_t qf = 4 * i;
            double dotprod = vec1[vf] * vec2[vf] + vec1[vf + 1] *
                             vec2[vf + 1] +
                             vec1[vf + 2] * vec2[vf + 2];
            double vec1prod = vec1[vf] * vec1[vf] + vec1[vf + 1] *
                              vec1[vf + 1] +
                              vec1[vf + 2] * vec1[vf + 2];
            double vec2prod = vec2[vf] * vec2[vf] + vec2[vf + 1] *
                              vec2[vf + 1] +
                              vec2[vf + 2] * vec2[vf + 2];

            // shortcut for coincident vectors
            if (::fabs(dotprod - 1.0) < 1.0e-12) {
                q[qf] = 0.0;
                q[qf + 1] = 0.0;
                q[qf + 2] = 0.0;
                q[qf + 3] = 1.0;
            } else {
                q[qf] = vec1[vf + 1] * vec2[vf + 2] - vec1[vf + 2] *
                        vec2[vf + 1];
                q[qf + 1] = vec1[vf + 2] * vec2[vf] - vec1[vf] * vec2[vf + 2];
                q[qf + 2] = vec1[vf] * vec2[vf + 1] - vec1[vf + 1] * vec2[vf];
                q[qf + 3] = ::sqrt(vec1prod * vec2prod) + dotprod;
                toast::qa_normalize_inplace_one(4, &(q[qf]));
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t vf = 3 * i;
            size_t qf = 4 * i;
            double dotprod = vec1[vf] * vec2[vf] + vec1[vf + 1] *
                             vec2[vf + 1] +
                             vec1[vf + 2] * vec2[vf + 2];
            double vec1prod = vec1[vf] * vec1[vf] + vec1[vf + 1] *
                              vec1[vf + 1] +
                              vec1[vf + 2] * vec1[vf + 2];
            double vec2prod = vec2[vf] * vec2[vf] + vec2[vf + 1] *
                              vec2[vf + 1] +
                              vec2[vf + 2] * vec2[vf + 2];

            // shortcut for coincident vectors
            if (::fabs(dotprod - 1.0) < 1.0e-12) {
                q[qf] = 0.0;
                q[qf + 1] = 0.0;
                q[qf + 2] = 0.0;
                q[qf + 3] = 1.0;
            } else {
                q[qf] = vec1[vf + 1] * vec2[vf + 2] - vec1[vf + 2] *
                        vec2[vf + 1];
                q[qf + 1] = vec1[vf + 2] * vec2[vf] - vec1[vf] * vec2[vf + 2];
                q[qf + 2] = vec1[vf] * vec2[vf + 1] - vec1[vf + 1] * vec2[vf];
                q[qf + 3] = ::sqrt(vec1prod * vec2prod) + dotprod;
                toast::qa_normalize_inplace_one(4, &(q[qf]));
            }
        }
    }

    return;
}

// Create quaternions from latitude, longitude, and position angles

void toast::qa_from_angles(size_t n, double const * theta,
                           double const * phi, double * const pa,
                           double * quat, bool IAU) {
    if (toast::is_aligned(theta) && toast::is_aligned(phi) &&
        toast::is_aligned(pa) && toast::is_aligned(quat)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t qf = 4 * i;
            double qR[4];
            double qD[4];
            double qP[4];
            double qtemp[4];

            // phi rotation around z-axis

            double angR = 0.5 * (phi[i] + toast::PI_2);

            qR[0] = 0.0;
            qR[1] = 0.0;
            qR[2] = ::sin(angR);
            qR[3] = ::cos(angR);

            // theta rotation around x-axis

            double angD = 0.5 * theta[i];

            qD[0] = ::sin(angD);
            qD[1] = 0.0;
            qD[2] = 0.0;
            qD[3] = ::cos(angD);

            // position angle rotation about z-axis

            double angP = toast::PI_2;
            if (IAU) {
                angP -= pa[i];
            } else {
                angP += pa[i];
            }
            angP *= 0.5;

            qP[0] = 0.0;
            qP[1] = 0.0;
            qP[2] = ::sin(angP);
            qP[3] = ::cos(angP);

            toast::qa_mult_one_one(qD, qP, qtemp);
            toast::qa_mult_one_one(qR, qtemp, &(quat[qf]));
            toast::qa_normalize_inplace_one(4, &(quat[qf]));
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t qf = 4 * i;
            double qR[4];
            double qD[4];
            double qP[4];
            double qtemp[4];

            // phi rotation around z-axis

            double angR = 0.5 * (phi[i] + toast::PI_2);

            qR[0] = 0.0;
            qR[1] = 0.0;
            qR[2] = ::sin(angR);
            qR[3] = ::cos(angR);

            // theta rotation around x-axis

            double angD = 0.5 * theta[i];

            qD[0] = ::sin(angD);
            qD[1] = 0.0;
            qD[2] = 0.0;
            qD[3] = ::cos(angD);

            // position angle rotation about z-axis

            double angP = toast::PI_2;
            if (IAU) {
                angP -= pa[i];
            } else {
                angP += pa[i];
            }
            angP *= 0.5;

            qP[0] = 0.0;
            qP[1] = 0.0;
            qP[2] = ::sin(angP);
            qP[3] = ::cos(angP);

            toast::qa_mult_one_one(qD, qP, qtemp);
            toast::qa_mult_one_one(qR, qtemp, &(quat[qf]));
            toast::qa_normalize_inplace_one(4, &(quat[qf]));
        }
    }

    return;
}

// Convert quaternions to latitude, longitude, and position angle

void toast::qa_to_angles(size_t n, double const * quat, double * theta,
                         double * phi, double * pa, bool IAU) {
    if (toast::is_aligned(theta) && toast::is_aligned(phi) &&
        toast::is_aligned(pa) && toast::is_aligned(quat)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            double const xaxis[3] = {1.0, 0.0, 0.0};
            double const zaxis[3] = {0.0, 0.0, 1.0};
            size_t qf = 4 * i;
            double dir[3];
            double orient[3];
            double qtemp[4];

            toast::qa_normalize_one(4, &(quat[qf]), qtemp);
            toast::qa_rotate_one_one(qtemp, zaxis, dir);
            toast::qa_rotate_one_one(qtemp, xaxis, orient);

            theta[i] = toast::PI_2 - ::asin(dir[2]);
            phi[i] = ::atan2(dir[1], dir[0]);

            if (phi[i] < 0.0) {
                phi[i] += toast::TWOPI;
            }

            pa[i] = ::atan2(orient[0] * dir[1] - orient[1] * dir[0],
                            -(orient[0] * dir[2] * dir[0])
                            - (orient[1] * dir[2] * dir[1])
                            + (orient[2] *
                               (dir[0] * dir[0] + dir[1] * dir[1])));

            if (IAU) {
                pa[i] = -pa[i];
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double const xaxis[3] = {1.0, 0.0, 0.0};
            double const zaxis[3] = {0.0, 0.0, 1.0};
            size_t qf = 4 * i;
            double dir[3];
            double orient[3];
            double qtemp[4];

            toast::qa_normalize_one(4, &(quat[qf]), qtemp);
            toast::qa_rotate_one_one(qtemp, zaxis, dir);
            toast::qa_rotate_one_one(qtemp, xaxis, orient);

            theta[i] = toast::PI_2 - ::asin(dir[2]);
            phi[i] = ::atan2(dir[1], dir[0]);

            if (phi[i] < 0.0) {
                phi[i] += toast::TWOPI;
            }

            pa[i] = ::atan2(orient[0] * dir[1] - orient[1] * dir[0],
                            -(orient[0] * dir[2] * dir[0])
                            - (orient[1] * dir[2] * dir[1])
                            + (orient[2] *
                               (dir[0] * dir[0] + dir[1] * dir[1])));

            if (IAU) {
                pa[i] = -pa[i];
            }
        }
    }

    // Quaternions describing a rotation about the Z-axis alone
    // require a special treatment

    for (size_t i = 0; i < n; ++i) {
        size_t qf = 4 * i;
        if ((::fabs(quat[qf]) < 1e-10) && (::fabs(quat[qf + 1]) < 1e-10)) {
            pa[i] = ::atan2(2 * quat[qf + 2] * quat[qf + 3],
                            1 - 2 * quat[qf + 2] * quat[qf + 2]);
        }
    }

    return;
}

// Convert quaternions to latitude and longitude

void toast::qa_from_position(size_t n, double const * theta,
                             double const * phi, double * quat) {
    if (toast::is_aligned(theta) && toast::is_aligned(phi)
        && toast::is_aligned(quat)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t qf = 4 * i;
            double qR[4];
            double qD[4];

            // phi rotation around z-axis

            double angR = 0.5 * (phi[i] + toast::PI_2);

            qR[0] = 0.0;
            qR[1] = 0.0;
            qR[2] = ::sin(angR);
            qR[3] = ::cos(angR);

            // theta rotation around x-axis

            double angD = 0.5 * theta[i];

            qD[0] = ::sin(angD);
            qD[1] = 0.0;
            qD[2] = 0.0;
            qD[3] = ::cos(angD);

            toast::qa_mult_one_one(qR, qD, &(quat[qf]));
            toast::qa_normalize_inplace_one(4, &(quat[qf]));
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t qf = 4 * i;
            double qR[4];
            double qD[4];
            double qtemp[4];

            // phi rotation around z-axis

            double angR = 0.5 * (phi[i] + toast::PI_2);

            qR[0] = 0.0;
            qR[1] = 0.0;
            qR[2] = ::sin(angR);
            qR[3] = ::cos(angR);

            // theta rotation around x-axis

            double angD = 0.5 * theta[i];

            qD[0] = ::sin(angD);
            qD[1] = 0.0;
            qD[2] = 0.0;
            qD[3] = ::cos(angD);

            toast::qa_mult_one_one(qR, qD, &(quat[qf]));
            toast::qa_normalize_inplace_one(4, &(quat[qf]));
        }
    }

    return;
}

void toast::qa_to_position(size_t n, double const * quat, double * theta,
                           double * phi) {
    if (toast::is_aligned(theta) && toast::is_aligned(phi) &&
        toast::is_aligned(quat)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            double const zaxis[3] = {0.0, 0.0, 1.0};
            size_t qf = 4 * i;
            double dir[3];
            double qtemp[4];

            double norm = 0.0;
            norm += quat[qf] * quat[qf];
            norm += quat[qf + 1] * quat[qf + 1];
            norm += quat[qf + 2] * quat[qf + 2];
            norm += quat[qf + 3] * quat[qf + 3];
            norm = 1.0 / ::sqrt(norm);
            qtemp[0] = quat[qf] * norm;
            qtemp[1] = quat[qf + 1] * norm;
            qtemp[2] = quat[qf + 2] * norm;
            qtemp[3] = quat[qf + 3] * norm;

            toast::qa_rotate_one_one(qtemp, zaxis, dir);

            theta[i] = toast::PI_2 - ::asin(dir[2]);
            phi[i] = ::atan2(dir[1], dir[0]);

            if (phi[i] < 0.0) {
                phi[i] += toast::TWOPI;
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double const zaxis[3] = {0.0, 0.0, 1.0};
            size_t qf = 4 * i;
            double dir[3];
            double qtemp[4];

            double norm = 0.0;
            norm += quat[qf] * quat[qf];
            norm += quat[qf + 1] * quat[qf + 1];
            norm += quat[qf + 2] * quat[qf + 2];
            norm += quat[qf + 3] * quat[qf + 3];
            norm = 1.0 / ::sqrt(norm);
            qtemp[0] = quat[qf] * norm;
            qtemp[1] = quat[qf + 1] * norm;
            qtemp[2] = quat[qf + 2] * norm;
            qtemp[3] = quat[qf + 3] * norm;

            toast::qa_rotate_one_one(qtemp, zaxis, dir);

            theta[i] = toast::PI_2 - ::asin(dir[2]);
            phi[i] = ::atan2(dir[1], dir[0]);

            if (phi[i] < 0.0) {
                phi[i] += toast::TWOPI;
            }
        }
    }

    return;
}
