/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_COV_HPP
#define TOAST_COV_HPP


namespace toast { namespace cov {

    void accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * zdata, int64_t * hits, double * invnpp, int64_t nsamp, double const * signal,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, double scale );

    void eigendecompose_covariance ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data, double * cond, double threshold, int32_t do_invert, int32_t do_rcond );

    void multiply_covariance ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data1, double const * data2 );

    void apply_covariance ( int64_t nsub, int64_t subsize, int64_t nnz,
    double const * mat, double * vec );


} }

#endif

