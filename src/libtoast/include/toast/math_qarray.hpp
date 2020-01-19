
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_QARRAY_HPP
#define TOAST_MATH_QARRAY_HPP


namespace toast {
void qa_list_dot(size_t n, size_t m, size_t d, double const * a,
                 double const * b, double * dotprod);

void qa_inv(size_t n, double * q);

void qa_amplitude_one(size_t d, double const * v, double * norm);

void qa_amplitude(size_t n, size_t m, size_t d, double const * v,
                  double * norm);

void qa_normalize_one(size_t d, double const * q_in, double * q_out);

void qa_normalize(size_t n, size_t m, size_t d, double const * q_in,
                  double * q_out);

void qa_normalize_inplace_one(size_t d, double * q);

void qa_normalize_inplace(size_t n, size_t m, size_t d, double * q);

void qa_rotate_one_one(double const * q, double const * v_in,
                       double * v_out);

void qa_rotate_many_one(size_t nq, double const * q, double const * v_in,
                        double * v_out);

void qa_rotate_one_many(double const * q, size_t nv, double const * v_in,
                        double * v_out);

void qa_rotate_many_many(size_t n, double const * q, double const * v_in,
                         double * v_out);

void qa_rotate(size_t nq, double const * q, size_t nv, double const * v_in,
               double * v_out);

void qa_mult_one_one(double const * p, double const * q, double * r);

void qa_mult_one_many(double const * p, size_t nq, double const * q,
                      double * r);

void qa_mult_many_one(size_t np, double const * p, double const * q,
                      double * r);

void qa_mult_many_many(size_t np, double const * p, size_t nq,
                       double const * q, double * r);

void qa_mult(size_t np, double const * p, size_t nq, double const * q,
             double * r);

void qa_slerp(size_t n_time, size_t n_targettime, double const * time,
              double const * targettime, double const * q_in,
              double * q_interp);

void qa_exp(size_t n, double const * q_in, double * q_out);

void qa_ln(size_t n, double const * q_in, double * q_out);

void qa_pow(size_t nq, size_t np, double const * p, double const * q_in,
            double * q_out);

void qa_from_axisangle_one_one(double const * axis, double angle,
                               double * q_out);

void qa_from_axisangle_one_many(size_t nang, double const * axis,
                                double const * angle, double * q_out);

void qa_from_axisangle_many_one(size_t naxis, double const * axis,
                                double angle, double * q_out);

void qa_from_axisangle_many_many(size_t n, double const * axis,
                                 double const * angle, double * q_out);

void qa_from_axisangle(size_t naxis, double const * axis, size_t nang,
                       double const * angle, double * q_out);

void qa_to_axisangle_one(double const * q, double * axis, double * angle);

void qa_to_axisangle(size_t n, double const * q, double * axis,
                     double * angle);

void qa_to_rotmat(double const * q, double * rotmat);

void qa_from_rotmat(const double * rotmat, double * q);

void qa_from_vectors(size_t n, double const * vec1, double const * vec2,
                     double * q);

void qa_from_angles(size_t n, double const * theta, double const * phi,
                    double * const pa, double * quat, bool IAU = false);

void qa_to_angles(size_t n, double const * quat, double * theta,
                  double * phi, double * pa, bool IAU = false);

void qa_from_position(size_t n, double const * theta, double const * phi,
                      double * quat);

void qa_to_position(size_t n, double const * quat, double * theta,
                    double * phi);
}

#endif // ifndef TOAST_QARRAY_HPP
