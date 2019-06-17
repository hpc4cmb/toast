
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MAP_COV_HPP
#define TOAST_MAP_COV_HPP


namespace toast {

void cov_accum_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                    int64_t nsamp,
                    int64_t const * indx_submap, int64_t const * indx_pix,
                    double const * weights,
                    double scale, double const * signal, double * zdata,
                    int64_t * hits, double * invnpp);

void cov_accum_diag_hits(int64_t nsub, int64_t subsize, int64_t nnz,
                         int64_t nsamp,
                         int64_t const * indx_submap,
                         int64_t const * indx_pix, int64_t * hits);

void cov_accum_diag_invnpp(int64_t nsub, int64_t subsize, int64_t nnz,
                           int64_t nsamp,
                           int64_t const * indx_submap,
                           int64_t const * indx_pix,
                           double const * weights,
                           double scale, int64_t * hits, double * invnpp);

void cov_accum_zmap(int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
                    int64_t const * indx_submap, int64_t const * indx_pix,
                    double const * weights,
                    double scale, double const * signal, double * zdata);

void cov_eigendecompose_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                             double * data, double * cond, double threshold,
                             bool invert);

void cov_mult_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                   double * data1, double const * data2);

void cov_apply_diag(int64_t nsub, int64_t subsize, int64_t nnz,
                    double const * mat, double * vec);

}

#endif // ifndef TOAST_MAP_COV_HPP
