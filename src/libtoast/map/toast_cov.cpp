/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_map_internal.hpp>

#include <cstring>
#include <iostream>

#ifdef _OPENMP
#  include <omp.h>
#endif


// void toast::cov::accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
//     int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
//     double scale, double const * signal, double * zdata, int64_t * hits, double * invnpp ) {

//     #pragma omp parallel default(shared)
//     {

//         int64_t i, j, k;
//         int64_t block = (int64_t)(nnz * (nnz+1) / 2);
//         int64_t zpx;
//         int64_t hpx;
//         int64_t ipx;
//         int64_t off;
//         double zsig;

//         double tzdata[nnz];
//         double tinvnpp[block];

//         #pragma omp for schedule(static)
//         for ( i = 0; i < nsamp; ++i ) {
//             if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
//                 zpx = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
//                 hpx = (indx_submap[i] * subsize) + indx_pix[i];
//                 ipx = (indx_submap[i] * subsize * block) + (indx_pix[i] * block);
//                 zsig = scale * signal[i];
                
//                 memset ( tzdata, 0, nnz*sizeof(double) );
//                 memset ( tinvnpp, 0, block*sizeof(double) );

//                 off = 0;
//                 for ( j = 0; j < nnz; ++j ) {
//                     tzdata[j] += zsig * weights[i * nnz + j];
//                     for ( k = j; k < nnz; ++k ) {
//                         tinvnpp[off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
//                         off += 1;
//                     }
//                 }

//                 #pragma omp critical ( accumdiag )
//                 {
//                     hits[hpx] += 1;
//                     for ( j = 0; j < nnz; ++j ) {
//                         zdata[zpx + j] += tzdata[j];
//                     }
//                     for ( j = 0; j < block; ++j ) {
//                         invnpp[ipx + j] += tinvnpp[j];
//                     }
//                 }
//             }
//         }
//     }

//     return;
// }


void toast::cov::accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
    double scale, double const * signal, double * zdata, int64_t * hits, double * invnpp ) {

    #pragma omp parallel default(shared)
    {

        int64_t i, j, k;
        int64_t block = (int64_t)(nnz * (nnz+1) / 2);
        int64_t zpx;
        int64_t hpx;
        int64_t ipx;
        int64_t off;

        int threads = 1;
        int trank = 0;

        #ifdef _OPENMP
        threads = omp_get_num_threads();
        trank = omp_get_thread_num();
        #endif

        int tpix;

        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                hpx = (indx_submap[i] * subsize) + indx_pix[i];
                tpix = hpx % threads;
                if ( tpix == trank ) {
                    zpx = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
                    ipx = (indx_submap[i] * subsize * block) + (indx_pix[i] * block);

                    off = 0;
                    for ( j = 0; j < nnz; ++j ) {
                        zdata[zpx + j] += scale * signal[i] * weights[i * nnz + j];
                        for ( k = j; k < nnz; ++k ) {
                            invnpp[ipx + off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
                            off += 1;
                        }
                    }

                    hits[hpx] += 1;
                }
            }
        }
    }

    return;
}


// void toast::cov::accumulate_diagonal_hits ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
//     int64_t const * indx_submap, int64_t const * indx_pix, int64_t * hits ) {

//     #pragma omp parallel default(shared)
//     {

//         int64_t i, j, k;
//         int64_t hpx;

//         #pragma omp for schedule(static)
//         for ( i = 0; i < nsamp; ++i ) {
//             if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
//                 hpx = (indx_submap[i] * subsize) + indx_pix[i];
//                 #pragma omp critical ( accumdiaghits )
//                 {
//                     hits[hpx] += 1;
//                 }
//             }
//         }

//     }

//     return;
// }


void toast::cov::accumulate_diagonal_hits ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
    int64_t const * indx_submap, int64_t const * indx_pix, int64_t * hits ) {

    #pragma omp parallel default(shared)
    {

        int64_t i, j, k;
        int64_t hpx;

        int threads = 1;
        int trank = 0;

        #ifdef _OPENMP
        threads = omp_get_num_threads();
        trank = omp_get_thread_num();
        #endif

        int tpix;

        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                hpx = (indx_submap[i] * subsize) + indx_pix[i];
                tpix = hpx % threads;
                if ( tpix == trank ) {
                    hits[hpx] += 1;
                }    
            }
        }

    }

    return;
}


// void toast::cov::accumulate_diagonal_invnpp ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
//     int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
//     double scale, int64_t * hits, double * invnpp ) {

//     #pragma omp parallel default(shared)
//     {

//         int64_t i, j, k;
//         int64_t block = (int64_t)(nnz * (nnz+1) / 2);
//         int64_t hpx;
//         int64_t ipx;
//         int64_t off;

//         double tinvnpp[block];

//         #pragma omp for schedule(static)
//         for ( i = 0; i < nsamp; ++i ) {
//             if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
//                 hpx = (indx_submap[i] * subsize) + indx_pix[i];
//                 ipx = (indx_submap[i] * subsize * block) + (indx_pix[i] * block);

//                 memset ( tinvnpp, 0, block*sizeof(double) );

//                 off = 0;
//                 for ( j = 0; j < nnz; ++j ) {
//                     for ( k = j; k < nnz; ++k ) {
//                         tinvnpp[off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
//                         off += 1;
//                     }
//                 }

//                 #pragma omp critical ( accumdiaginvnpp )
//                 {
//                     hits[hpx] += 1;
//                     for ( j = 0; j < block; ++j ) {
//                         invnpp[ipx + j] += tinvnpp[j];
//                     }
//                 }
//             }
//         }
//     }

//     return;
// }


void toast::cov::accumulate_diagonal_invnpp ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
    double scale, int64_t * hits, double * invnpp ) {

    #pragma omp parallel default(shared)
    {

        int64_t i, j, k;
        int64_t block = (int64_t)(nnz * (nnz+1) / 2);
        int64_t hpx;
        int64_t ipx;
        int64_t off;

        int threads = 1;
        int trank = 0;

        #ifdef _OPENMP
        threads = omp_get_num_threads();
        trank = omp_get_thread_num();
        #endif

        int tpix;

        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                hpx = (indx_submap[i] * subsize) + indx_pix[i];
                tpix = hpx % threads;
                if ( tpix == trank ) {
                    ipx = (indx_submap[i] * subsize * block) + (indx_pix[i] * block);

                    off = 0;
                    for ( j = 0; j < nnz; ++j ) {
                        for ( k = j; k < nnz; ++k ) {
                            invnpp[ipx + off] += scale * weights[i * nnz + j] * weights[i * nnz + k];
                            off += 1;
                        }
                    }

                    hits[hpx] += 1;
                }
            }
        }
    }

    return;
}


// void toast::cov::accumulate_zmap ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
//     int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
//     double scale, double const * signal, double * zdata ) {

//     #pragma omp parallel default(shared)
//     {
//         int64_t i, j, k;
//         int64_t zpx;
//         double zsig;

//         double tzdata[nnz];

//         #pragma omp for schedule(static)
//         for ( i = 0; i < nsamp; ++i ) {
//             if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
//                 zpx = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);
//                 zsig = scale * signal[i];

//                 memset ( tzdata, 0, nnz*sizeof(double) );

//                 for ( j = 0; j < nnz; ++j ) {
//                     tzdata[j] += zsig * weights[i * nnz + j];
//                 }

//                 #pragma omp critical ( accumdiagzmap )
//                 {
//                     for ( j = 0; j < nnz; ++j ) {
//                         zdata[zpx + j] += tzdata[j];
//                     }
//                 }
//             }
//         }
//     }

//     return;
// }

#include "toast_util_internal.hpp"

void toast::cov::accumulate_zmap ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp, 
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights, 
    double scale, double const * signal, double * zdata ) {

    toast::util::auto_timer auto_timer("[cxx] accumulate_zmap_direct");

#if defined(FASTER)
    // > [cxx] accumulate_zmap_direct
    // :   1.474 wall,   2.930 user +   0.020 system =   2.950 CPU [seconds] (200.1%)
    // (total # of laps: 32)
    #pragma omp parallel default(shared)
    {
        int threads = 1;
        int trank = 0;

    #ifdef _OPENMP
        threads = omp_get_num_threads();
        trank = omp_get_thread_num();
    #endif

        for (int64_t i = 0; i < nsamp; ++i )
        {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) )
            {
                int64_t hpx = (indx_submap[i] * subsize) + indx_pix[i];
                int64_t tpix = hpx % threads;
                if ( tpix == trank )
                {
                    int64_t zpx = (indx_submap[i] * subsize * nnz)
                                  + (indx_pix[i] * nnz);

                    for (int64_t j = 0; j < nnz; ++j )
                    {
                        if(zpx + j > subsize*nnz*nsub)
                            zdata[zpx + j] += scale * signal[i] * weights[i * nnz + j];
                    }
                }
            }
        }
    }
#else
    // > [cxx] ctoast_cov_accumulate_zmap
    // : 118.837 wall,   4.820 user +   4.650 system =   9.470 CPU [seconds] (  8.0%)
    // (total # of laps: 32)
    #pragma omp parallel default(shared)
    {
        int64_t i, j, k;
        int64_t hpx;
        int64_t zpx;

        int threads = 1;
        int trank = 0;

        #ifdef _OPENMP
        threads = omp_get_num_threads();
        trank = omp_get_thread_num();
        #endif

        int tpix;

        for ( i = 0; i < nsamp; ++i ) {
            if ( ( indx_submap[i] >= 0 ) && ( indx_pix[i] >= 0 ) ) {
                hpx = (indx_submap[i] * subsize) + indx_pix[i];
                tpix = hpx % threads;
                if ( tpix == trank ) {
                    zpx = (indx_submap[i] * subsize * nnz) + (indx_pix[i] * nnz);

                    for ( j = 0; j < nnz; ++j ) {
                        zdata[zpx + j] += scale * signal[i] * weights[i * nnz + j];
                    }
                }
            }
        }
    }
#endif

    return;
}


void toast::cov::eigendecompose_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data, double * cond, double threshold, int32_t do_invert, int32_t do_rcond ) {

    if ( ( do_invert == 0 ) && ( do_rcond == 0 ) ) {
        return;
    }

    int64_t i, j, k;
    int64_t block = (int64_t)(nnz * (nnz+1) / 2);
    int64_t dpx;

    if ( nnz == 1 ) {
        // shortcut for NNZ == 1
        
        if ( do_invert == 0 ) {

            for ( i = 0; i < nsub; ++i ) {
                for ( j = 0; j < subsize; ++j ) {
                    cond[i * subsize + j] = 1.0;
                }
            }

        } else if ( do_rcond == 0 ) {

            for ( i = 0; i < nsub; ++i ) {
                for ( j = 0; j < subsize; ++j ) {
                    dpx = (i * subsize) + j;
                    if ( data[dpx] != 0 ) {
                        data[dpx] = 1.0 / data[dpx];
                    }
                }
            }

        } else {

            for ( i = 0; i < nsub; ++i ) {
                for ( j = 0; j < subsize; ++j ) {
                    dpx = (i * subsize) + j;
                    cond[dpx] = 1.0;
                    if ( data[dpx] != 0 ) {
                        data[dpx] = 1.0 / data[dpx];
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

        #pragma omp parallel default(shared) private(i, j, k, dpx)
        {
            // thread-private variables

            int64_t m;
            int64_t off;

            double emin;
            double emax;
            double rcond;
        
            int info;

            toast::mem::simd_array<double> fdata(nnz*nnz);
            toast::mem::simd_array<double> ftemp(nnz*nnz);
            toast::mem::simd_array<double> finv(nnz*nnz);
            toast::mem::simd_array<double> evals(nnz);
            toast::mem::simd_array<double> work(lwork);

            // Here we "unroll" the loop over submaps and pixels within each submap.
            // This allows us to distribute the total pixels across all threads.

            #pragma omp for schedule(static)
            for ( i = 0; i < (nsub * subsize); ++i ) {

                dpx = i * block;
                    
                // copy to fortran buffer
                off = 0;
                ::memset ( fdata, 0, nnz*nnz*sizeof(double) );
                for ( k = 0; k < nnz; ++k ) {
                    for ( m = k; m < nnz; ++m ) {
                        fdata[k*nnz + m] = data[dpx + off];
                        off += 1;
                    }
                }

                // eigendecompose
                if ( do_invert == 0 ) {
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
                        if ( do_invert != 0 ) {
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
                                    data[dpx + off] = finv[k*nnz + m];
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
                
                if ( do_invert != 0 ) {
                    if ( info != 0 ) {
                        off = 0;
                        for ( k = 0; k < nnz; ++k ) {
                            for ( m = k; m < nnz; ++m ) {
                                data[dpx + off] = 0.0;
                                off += 1;
                            }
                        }
                    }
                }

                if ( do_rcond != 0 ) {
                    cond[i] = rcond;
                }

            }
        }

    }

    return;
}


void toast::cov::multiply_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double * data1, double const * data2 ) {

    int64_t i, j, k;
    int64_t block = (int64_t)(nnz * (nnz+1) / 2);
    int64_t px;

    if ( nnz == 1 ) {
        // shortcut for NNZ == 1

        for ( i = 0; i < nsub; ++i ) {
            for ( j = 0; j < subsize; ++j ) {
                px = (i * subsize * block) + (j * block);
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

            toast::mem::simd_array<double> fdata1(nnz*nnz);
            toast::mem::simd_array<double> fdata2(nnz*nnz);
            toast::mem::simd_array<double> fdata3(nnz*nnz);

            // Here we "unroll" the loop over submaps and pixels within each submap.
            // This allows us to distribute the total pixels across all threads.

            #pragma omp for schedule(static)
            for ( i = 0; i < (nsub * subsize); ++i ) {
                
                px = i * block;
                    
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
        }

    }

    return;
}


void toast::cov::apply_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double const * mat, double * vec ) {

    int64_t i, j, k;
    int64_t block = (int64_t)(nnz * (nnz+1) / 2);
    int64_t mpx;
    int64_t vpx;

    if ( nnz == 1 ) {
        // shortcut for NNZ == 1

        for ( i = 0; i < nsub; ++i ) {
            for ( j = 0; j < subsize; ++j ) {
                mpx = (i * subsize * block) + (j * block);
                vpx = (i * subsize * nnz) + (j * nnz);
                vec[vpx] *= mat[mpx];
            }
        }

    } else {

        // We do this manually now, but could use dsymv if needed...
        // Since this is just multiply / add operations, the overhead of threading
        // is likely more than the savings.

        int64_t m;
        int64_t off;

        toast::mem::simd_array<double> temp(nnz);

        for ( i = 0; i < nsub; ++i ) {
            for ( j = 0; j < subsize; ++j ) {
                mpx = (i * subsize * block) + (j * block);
                vpx = (i * subsize * nnz) + (j * nnz);

                ::memset(temp, 0, nnz * sizeof(double));

                off = 0;
                for ( k = 0; k < nnz; ++k ) {
                    for ( m = k; m < nnz; ++m ) {
                        temp[k] += mat[mpx + off] * vec[vpx + m];
                        if ( m != k ) {
                            temp[m] += mat[mpx + off] * vec[vpx + k];
                        }
                        off++;
                    }
                }

                for ( k = 0; k < nnz; ++k ) {
                    vec[vpx + k] = temp[k];
                }
            }
        }
    }

    return;
}


