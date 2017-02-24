/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_map_internal.hpp>

#include <cstring>

#ifdef _OPENMP
#  include <omp.h>
#endif


void toast::cov::accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * zdata, int64_t * hits, double * invnpp, int64_t nsamp, double const * signal,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, double scale ) {

    bool do_z = ( zdata != NULL );
    bool do_hits = ( hits != NULL );
    bool do_invn = ( invnpp != NULL );

    // This duplicates code in order to keep conditionals out of the loop.

    int64_t i, j, k;
    int64_t px;
    int64_t off;
    double zsig;

    if ( do_z && do_hits && do_invn ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                hits[px] += 1;
                zsig = scale * signal[i];
                off = 0;
                for ( j = 0; j < nnz; ++j ) {
                    zdata[px + j] += zsig * weights[i * nnz + j];
                    for ( k = j; k < nnz; ++k ) {
                        invnpp[px + off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
                        off += 1;
                    }
                }
            }
        }

    } else if ( do_z && do_hits ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                hits[px] += 1;
                zsig = scale * signal[i];
                for ( j = 0; j < nnz; ++j ) {
                    zdata[px + j] += zsig * weights[i * nnz + j];
                }
            }
        }

    } else if ( do_hits && do_invn ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                hits[px] += 1;
                off = 0;
                for ( j = 0; j < nnz; ++j ) {
                    for ( k = j; k < nnz; ++k ) {
                        invnpp[px + off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
                        off += 1;
                    }
                }
            }
        }

    } else if ( do_z && do_invn ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                zsig = scale * signal[i];
                off = 0;
                for ( j = 0; j < nnz; ++j ) {
                    zdata[px + j] += zsig * weights[i * nnz + j];
                    for ( k = j; k < nnz; ++k ) {
                        invnpp[px + off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
                        off += 1;
                    }
                }
            }
        }

    } else if ( do_z ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                zsig = scale * signal[i];
                for ( j = 0; j < nnz; ++j ) {
                    zdata[px + j] += zsig * weights[i * nnz + j];
                }
            }
        }

    } else if ( do_hits ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                hits[px] += 1;
            }
        }

    } else if ( do_invn ) {

        #pragma omp parallel for default(shared) private(i, j, k, px, off, zsig) schedule(static)
        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                px = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                off = 0;
                for ( j = 0; j < nnz; ++j ) {
                    for ( k = j; k < nnz; ++k ) {
                        invnpp[px + off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
                        off += 1;
                    }
                }
            }
        }

    }

    return;
}


void toast::cov::eigendecompose_covariance ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data, double * cond, double threshold, int32_t do_invert, int32_t do_rcond ) {

    if ( ( ! do_invert ) && ( ! do_rcond ) ) {
        return;
    }

    int64_t i, j, k;
    int64_t px;

    if ( nnz == 1 ) {
        // shortcut for NNZ == 1
        
        if ( ! do_invert ) {

            for ( i = 0; i < nsub; ++i ) {
                for ( j = 0; j < subsize; ++j ) {
                    px = (i * subsize * nnz) + (j * nnz);
                    cond[px] = 1.0;
                }
            }

        } else if ( ! do_rcond ) {

            for ( i = 0; i < nsub; ++i ) {
                for ( j = 0; j < subsize; ++j ) {
                    px = (i * subsize * nnz) + (j * nnz);
                    if ( data[px] != 0 ) {
                        data[px] = 1.0 / data[px];
                    }
                }
            }

        } else {

            for ( i = 0; i < nsub; ++i ) {
                for ( j = 0; j < subsize; ++j ) {
                    px = (i * subsize * nnz) + (j * nnz);
                    cond[px] = 1.0;
                    if ( data[px] != 0 ) {
                        data[px] = 1.0 / data[px];
                    }
                }
            }

        }

    } else {
    
        // We assume a large value here, since the work space needed
        // will still be small.
        int NB = 256;
    
        int lwork = NB * 2 + (int)nnz;
        int fnnz = (int)nnz;
        
        double fzero = 0.0;
        double fone = 1.0;
    
        char jobz_vec = 'V';
        char jobz_val = 'N';
        char uplo = 'L';
        char transN = 'N';
        char transT = 'T';

        // Even if the actual BLAS/LAPACK library is threaded, these are very
        // small matrices.  So instead we divide up the map data across threads
        // and each thread does some large number of small eigenvalue problems.

        #pragma omp parallel default(shared) private(i, j, k, px)
        {
            // thread-private variables

            int64_t m;
            int64_t off;

            double emin;
            double emax;
            double rcond;
        
            int info;

            double * fdata = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );
            
            double * ftemp = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );
            
            double * finv = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );
            
            double * evals = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );
            
            double * work = static_cast < double * > ( toast::mem::aligned_alloc ( 
                lwork * sizeof(double), toast::mem::SIMD_ALIGN ) );

            // Here we "unroll" the loop over submaps and pixels within each submap.
            // This allows us to distribute the total pixels across all threads.

            #pragma omp for schedule(static)
            for ( i = 0; i < (nsub * subsize); ++i ) {
                
                px = i * nnz;
                    
                // copy to fortran buffer
                off = 0;
                ::memset ( fdata, 0, nnz*nnz*sizeof(double) );
                for ( k = 0; k < nnz; ++k ) {
                    for ( m = k; m < nnz; ++m ) {
                        fdata[k*nnz + m] = data[px + off];
                        off += 1;
                    }
                }

                // eigendecompose
                if ( ! do_invert ) {
                    toast::lapack::syev(&jobz_val, &uplo, &fnnz, fdata, &fnnz, evals, work, &lwork, &info);
                } else {
                    toast::lapack::syev(&jobz_vec, &uplo, &fnnz, fdata, &fnnz, evals, work, &lwork, &info);
                }

                rcond = 0.0;

                if ( info == 0 ) {
                    
                    // it worked, compute condition number
                    emin = 1.0e100;
                    emax = 0.0;
                    for ( k = 0; k < nnz; ++k ) {
                        if ( evals[k] < emin ) {
                            emin = evals[k];
                        }
                        if ( evals[k] > emax ) {
                            emax = evals[k];
                        }
                    }
                    if ( emax > 0.0 ) {
                        rcond = emin / emax;
                    }

                    // compare to threshold
                    if ( rcond >= threshold ) {
                        if ( do_invert ) {
                            for ( k = 0; k < nnz; ++k ) {
                                evals[k] = 1.0 / evals[k];
                                for ( m = 0; m < nnz; ++m ) {
                                    ftemp[k*nnz + m] = evals[k] * fdata[k*nnz + m];
                                }
                            }
                            toast::lapack::gemm(&transN, &transT, &fnnz, &fnnz, &fnnz, 
                                &fone, ftemp, &fnnz, fdata, &fnnz, &fzero, finv, &fnnz);
                                
                            off = 0;
                            for ( k = 0; k < nnz; ++k ) {
                                for ( m = k; m < nnz; ++m ) {
                                    data[px + off] = finv[k*nnz + m];
                                    off += 1;
                                }
                            }
                        }
                    } else {
                        // reject this pixel
                        rcond = 0.0;
                        info = 1;
                    }

                }
                
                if ( do_invert ) {
                    if ( info != 0 ) {
                        off = 0;
                        for ( k = 0; k < nnz; ++k ) {
                            for ( m = k; m < nnz; ++m ) {
                                data[px + off] = 0.0;
                                off += 1;
                            }
                        }
                    }
                }

                if ( do_rcond ) {
                    cond[px] = rcond;
                }

            }

            toast::mem::aligned_free ( fdata );
            toast::mem::aligned_free ( ftemp );
            toast::mem::aligned_free ( finv );
            toast::mem::aligned_free ( evals );
            toast::mem::aligned_free ( work );

        }

    }

    return;
}


void toast::cov::multiply_covariance ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data1, double const * data2 ) {

    int64_t i, j, k;
    int64_t px;

    if ( nnz == 1 ) {
        // shortcut for NNZ == 1

        for ( i = 0; i < nsub; ++i ) {
            for ( j = 0; j < subsize; ++j ) {
                px = (i * subsize * nnz) + (j * nnz);
                data1[px] *= data2[px];
            }
        }

    } else {

        int fnnz = (int)nnz;
        
        double fzero = 0.0;
        double fone = 1.0;
    
        char uplo = 'L';
        char side = 'L';

        // Even if the actual BLAS/LAPACK library is threaded, these are very
        // small matrices.  So instead we divide up the map data across threads
        // and each thread does some large number of small eigenvalue problems.

        #pragma omp parallel default(shared) private(i, j, k, px)
        {
            // thread-private variables

            int64_t m;
            int64_t off;

            double * fdata1 = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );

            double * fdata2 = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );

            double * fdata3 = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );

            // Here we "unroll" the loop over submaps and pixels within each submap.
            // This allows us to distribute the total pixels across all threads.

            #pragma omp for schedule(static)
            for ( i = 0; i < (nsub * subsize); ++i ) {
                
                px = i * nnz;
                    
                // copy to fortran buffer

                ::memset ( fdata1, 0, nnz*nnz*sizeof(double) );
                ::memset ( fdata2, 0, nnz*nnz*sizeof(double) );
                ::memset ( fdata3, 0, nnz*nnz*sizeof(double) );
                
                off = 0;
                for ( k = 0; k < nnz; ++k ) {
                    for ( m = k; m < nnz; ++m ) {
                        fdata1[k*nnz + m] = data1[px + off];
                        fdata2[k*nnz + m] = data2[px + off];
                        if ( k != m ) {
                            // Second argument to dsymm must be full
                            fdata2[m*nnz + k] = data2[px + off];
                        }
                        off += 1;
                    }
                }

                toast::lapack::symm(&side, &uplo, &fnnz, &fnnz, &fone, fdata1, &fnnz, fdata2, &fnnz, &fzero, fdata3, &fnnz);

                off = 0;
                for ( k = 0; k < nnz; ++k ) {
                    for ( m = k; m < nnz; ++m ) {
                        data1[px + off] = fdata3[k*nnz + m];
                        off += 1;
                    }
                }

            }

            toast::mem::aligned_free ( fdata1 );
            toast::mem::aligned_free ( fdata2 );
            toast::mem::aligned_free ( fdata3 );

        }

    }

    return;
}


void toast::cov::apply_covariance ( int64_t nsub, int64_t subsize, int64_t nnz,
    double const * mat, double * vec ) {

    int64_t i, j, k;
    int64_t px;

    if ( nnz == 1 ) {
        // shortcut for NNZ == 1

        for ( i = 0; i < nsub; ++i ) {
            for ( j = 0; j < subsize; ++j ) {
                px = (i * subsize * nnz) + (j * nnz);
                vec[px] *= mat[px];
            }
        }

    } else {

        // We do this manually now, but could use dsymv if needed...
        // Since this is just multiply / add operations, the overhead of threading
        // is likely more than the savings.

        int64_t m;
        int64_t off;

        double * temp = static_cast < double * > ( toast::mem::aligned_alloc ( 
                nnz * sizeof(double), toast::mem::SIMD_ALIGN ) );

        for ( i = 0; i < nsub; ++i ) {
            for ( j = 0; j < subsize; ++j ) {
                px = (i * subsize * nnz) + (j * nnz);

                ::memset(temp, 0, nnz * sizeof(double));

                off = 0;
                for ( k = 0; k < nnz; ++k ) {
                    for ( m = k; m < nnz; ++m ) {
                        temp[k] += mat[px + off] * vec[px + m];
                        if ( m != k ) {
                            temp[m] += mat[px + off] * vec[px + k];
                        }
                        off++;
                    }
                }

                for ( k = 0; k < nnz; ++k ) {
                    vec[px + k] = temp[k];
                }
            }
        }

        toast::mem::aligned_free ( temp );

    }

    return;
}


