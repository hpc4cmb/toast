/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_QARRAY_HPP
#define TOAST_QARRAY_HPP


namespace toast { namespace qarray {

    void list_dot ( size_t n, size_t m, size_t d, double const * a,
        double const * b, double * dotprod );

    void inv ( size_t n, double * q );

    void amplitude ( size_t n, size_t m, size_t d, double const * v,
        double * norm );

    void normalize ( size_t n, size_t m, size_t d, double const * q_in,
        double * q_out );

    void normalize_inplace ( size_t n, size_t m, size_t d, double * q );

    void rotate ( size_t nq, double const * q, size_t nv, double const * v_in,
        double * v_out );

    void mult ( size_t np, double const * p, size_t nq, double const * q, double * r );

    void slerp ( size_t n_time, size_t n_targettime, double const * time,
        double const * targettime, double const * q_in, double * q_interp );

    void exp ( size_t n, double const * q_in, double * q_out );

    void ln ( size_t n, double const * q_in, double * q_out );

    void pow ( size_t n, double const * p, double const * q_in,
        double * q_out );

    void from_axisangle ( size_t n, double const * axis, double const * angle,
        double * q_out );

    void to_axisangle ( size_t n, double const * q, double * axis,
        double * angle );

    void to_rotmat ( double const * q, double * rotmat );

    void from_rotmat ( const double * rotmat, double * q );

    void from_vectors ( size_t n, double const * vec1, double const * vec2,
        double * q );

    void from_angles ( size_t n, double const * theta, double const * phi,
        double * const pa, double * quat, bool IAU = false );

    void to_angles ( size_t n, double const * quat, double * theta,
        double * phi, double * pa, bool IAU = false );

    void to_position ( size_t n, double const * quat, double * theta,
        double * phi );

} }

#endif
