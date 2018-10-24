/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_COV_HPP
#define TOAST_COV_HPP


namespace toast { namespace cov {

    void accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
        int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
        double scale, double const * signal, double * zdata, int64_t * hits, double * invnpp );

    void accumulate_diagonal_hits ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
        int64_t const * indx_submap, int64_t const * indx_pix, int64_t * hits );

    void accumulate_diagonal_invnpp ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
        int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
        double scale, int64_t * hits, double * invnpp );

    void accumulate_zmap ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
        int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
        double scale, double const * signal, double * zdata );

    void eigendecompose_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data, double * cond, double threshold, int32_t do_invert, int32_t do_rcond );

    void multiply_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data1, double const * data2 );

    void apply_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double const * mat, double * vec );


} }

#endif

