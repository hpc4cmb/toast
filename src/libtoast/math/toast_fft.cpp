/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#ifdef HAVE_MKL
#  include <mkl_dfti.h>
#else
#  ifdef HAVE_FFTW
#    include <fftw3.h>
#  endif
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif


#ifdef HAVE_FFTW

class r1d_fftw : public toast::fft::r1d {
    
    public :
        
        r1d_fftw ( int64_t length, int64_t n, toast::fft::plan_type type, toast::fft::direction dir, double scale ) : toast::fft::r1d ( length, n, type, dir, scale ) {

            int threads = 1;

            // enable threads

            #ifdef HAVE_FFTW_THREADS
            #  ifdef _OPENMP

            threads = omp_get_max_threads();

            #  endif

            fftw_plan_with_nthreads(threads);

            #endif

            // allocate memory

            data_.resize ( n * 2 * length );

            // create vector views and raw pointers

            traw_ = static_cast < double * > ( & data_[0] );
            fraw_ = static_cast < double * > ( & data_[n * length] );

            tview_.clear();
            fview_.clear();

            for ( int64_t i = 0; i < n; ++i ) {
                tview_.push_back ( & data_[i * length] );
                fview_.push_back ( & data_[(n * length) + i * length] );
            }

            // create plan

            int ilength = static_cast < int > ( length );
            int iN = static_cast < int > ( n );

            unsigned flags = 0;
            double * rawin;
            double * rawout;

            fftw_r2r_kind kind;

            if ( dir == toast::fft::direction::forward ) {
                rawin = traw_;
                rawout = fraw_;
                kind = FFTW_R2HC;
            } else {
                rawin = fraw_;
                rawout = traw_;
                kind = FFTW_HC2R;
            }

            flags = flags | FFTW_DESTROY_INPUT;

            if ( type == toast::fft::plan_type::best ) {
                flags = flags | FFTW_MEASURE;
            } else {
                flags = flags | FFTW_ESTIMATE;
            }

            plan_ = fftw_plan_many_r2r ( 1, &ilength, iN, rawin, &ilength, 1, ilength, rawout, &ilength, 1, ilength, &kind, flags);


        }

        ~r1d_fftw ( ) {
            fftw_destroy_plan ( static_cast < fftw_plan > ( plan_ ) );
            tview_.clear();
            fview_.clear();
            data_.clear();
        }

        void exec ( ) {

            fftw_execute ( plan_ );

            double * rawout;
            double norm;

            if ( dir_ == toast::fft::direction::forward ) {
                rawout = fraw_;
                norm = scale_;
            } else {
                rawout = traw_;
                norm = scale_ / static_cast < double > ( length_ );
            }

            int64_t len = mult_ * length_;

            for ( int64_t i = 0; i < len; ++i ) {
                rawout[i] *= norm;
            }

            return;
        }

        std::vector < double * > tdata ( ) {
            return tview_;
        }
        
        std::vector < double * > fdata ( ) {
            return fview_;
        }

    private :
        fftw_plan plan_;
        toast::fft::fft_data data_;
        double * traw_;
        double * fraw_;
        std::vector < double * > tview_;
        std::vector < double * > fview_;

};

#endif


#ifdef HAVE_MKL

class r1d_mkl : public toast::fft::r1d {

    public :
        
        r1d_mkl ( int64_t length, int64_t n, toast::fft::plan_type type, toast::fft::direction dir, double scale ) : toast::fft::r1d ( length, n, type, dir, scale ) {

            // allocate memory

            data_.resize ( 2 * n * length );

            // create vector views and raw pointers

            traw_ = static_cast < double * > ( & data_[0] );
            fraw_ = static_cast < double * > ( & data_[n * length] );

            tview_.clear();
            fview_.clear();

            for ( int64_t i = 0; i < n; ++i ) {
                tview_.push_back ( & data_[i * length] );
                fview_.push_back ( & data_[(n * length) + i * length] );
            }

            // create plan

            MKL_LONG status = DftiCreateDescriptor ( &descriptor_, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) length );

            status = DftiSetValue ( descriptor_, DFTI_NUMBER_OF_TRANSFORMS, n );

            status = DftiSetValue ( descriptor_, DFTI_INPUT_DISTANCE, length );

            status = DftiSetValue ( descriptor_, DFTI_OUTPUT_DISTANCE, length );

            status = DftiSetValue ( descriptor_, DFTI_PLACEMENT, DFTI_INPLACE );

            status = DftiSetValue ( descriptor_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_REAL );

            status = DftiSetValue ( descriptor_, DFTI_PACKED_FORMAT, DFTI_PERM_FORMAT );

            if ( dir_ == toast::fft::direction::forward ) {
                status = DftiSetValue ( descriptor_, DFTI_FORWARD_SCALE, scale );
                status = DftiSetValue ( descriptor_, DFTI_BACKWARD_SCALE, 1.0 );
            } else {
                status = DftiSetValue ( descriptor_, DFTI_FORWARD_SCALE, 1.0 );
                status = DftiSetValue ( descriptor_, DFTI_BACKWARD_SCALE, scale / (double)length );
            }

            status = DftiCommitDescriptor ( descriptor_ );

            if ( status != 0 ) {
                std::ostringstream o;
                o << "failed to create mkl FFT plan, status = " << status << std::endl;
                o << "Message: " << DftiErrorMessage ( status ) ;
                TOAST_THROW( o.str().c_str() );
            }

        }

        ~r1d_mkl ( ) {
            MKL_LONG status = DftiFreeDescriptor ( &descriptor_ );
        }

        void exec ( ) {
            MKL_LONG status = 0;

            if ( dir_ == toast::fft::direction::forward ) {
                status = DftiComputeForward ( descriptor_, traw_ );
                cce2hc ( mult_, length_, traw_, fraw_ );
            } else {
                hc2cce ( mult_, length_, fraw_, traw_ );
                status = DftiComputeBackward ( descriptor_, traw_ );
            }

            if ( status != 0 ) {
                std::ostringstream o;
                o << "failed to execute MKL transform, status = " << status;
                TOAST_THROW( o.str().c_str() );
            }

            return;
        }

        std::vector < double * > tdata ( ) {
            return tview_;
        }

        std::vector < double * > fdata ( ) {
            return fview_;
        }

    private :

        void cce2hc ( int64_t n, int64_t len, double * cce, double * hc ) {
            // "permutation" format is
            // N even : R_0, R_n/2, R_1, I_1, ..., R_n/2-1, I_n/2-1
            // N odd : R_0, R_1, I_1, ..., R_n/2, I_n/2

            int64_t offset = 0;
            int64_t half = (int64_t)( len / 2 );
            int64_t even = 0;

            if ( len % 2 == 0 ) {
                even = 1;
            }

            int64_t i, j;
            int64_t t;

            for ( i = 0; i < n; ++i ) {
                //cerr << "cce2hc: vec " << i << ", offset " << offset << endl;
                hc[ offset ] = cce[ offset ];
                //cerr << "cce2hc:  set hc[" << 0 << "] = " << cce[ offset ] << endl;

                if ( even ) {
                    hc[ offset + half ] = cce[ offset + even ];
                    //cerr << "cce2hc:  set hc[" << half << "] = " << cce[ offset + even ] << endl;
                }

                for ( j = 1; j < half; ++j ) {
                    t = 2 * j;
                    //cerr << "cce2hc:  hc/cce (" << j << "," << len-j << ")/(" << even+t-1 << "," << even+t << ") set to Re/Im [" << cce [ offset + even + t - 1 ] << ", " << cce [ offset + even + t ] << " ]" << endl; 
                    hc[ offset + j ] = cce [ offset + even + t - 1 ];
                    hc[ offset + len - j ] = cce [ offset + even + t ];
                }
                offset += len;
            }

            return;
        }

        void hc2cce ( int64_t n, int64_t len, double * hc, double * cce ) {
            int64_t offset = 0;
            int64_t half = (int64_t)( len / 2 );
            int64_t even = 0;

            if ( len % 2 == 0 ) {
                even = 1;
            }

            int64_t i, j;
            int64_t t;

            for ( i = 0; i < n; ++i ) {
                cce[ offset ] = hc[ offset ];
                if ( even ) {
                    cce[ offset + even ] = hc[ offset + half ];
                }

                for ( j = 1; j < half; ++j ) {
                    t = 2 * j;
                    cce [ offset + even + t - 1 ] = hc[ offset + j ];
                    cce [ offset + even + t ] = hc[ offset + len - j ];
                }
                offset += len;
            }

            return;
        }

        DFTI_DESCRIPTOR_HANDLE descriptor_;
        toast::fft::fft_data data_;
        double * traw_;
        double * fraw_;
        std::vector < double * > tview_;
        std::vector < double * > fview_;

};

#endif



// Public 1D plan class

toast::fft::r1d::r1d ( int64_t length, int64_t n, plan_type type, direction dir, double scale ) {
    type_ = type;
    dir_ = dir;
    length_ = length;
    mult_ = n;
    scale_ = scale;
}


toast::fft::r1d * toast::fft::r1d::create ( int64_t length, int64_t n, plan_type type, direction dir, double scale ) {

#ifdef HAVE_MKL

    return new r1d_mkl ( length, n, type, dir, scale );

#else
#  ifdef HAVE_FFTW

    return new r1d_fftw ( length, n, type, dir, scale );

#  else

    TOAST_THROW("FFTs require MKL or FFTW");

#  endif
#endif
  
    return NULL;
}


// Persistant storage of 1D plans for a fixed size

toast::fft::r1d_plan_store::~r1d_plan_store ( ) {
}


void toast::fft::r1d_plan_store::clear ( ) {
    fplans_.clear();
    rplans_.clear();
    return;  
}


toast::fft::r1d_plan_store & toast::fft::r1d_plan_store::get ( ) {
    static toast::fft::r1d_plan_store instance;
    return instance;
}


void toast::fft::r1d_plan_store::cache ( int64_t len, int64_t n ) {

    int nthreads = 1;

    #ifdef _OPENMP  
    nthreads = omp_get_max_threads();
    #endif

    std::pair < int64_t, int64_t > key ( len, n );

    for ( int i = 0; i < nthreads; ++i ) {

        std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > & frank_plan = fplans_[ i ];
        std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > & rrank_plan = rplans_[ i ];

        std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > :: iterator fit = frank_plan.find ( key );
        if ( fit == frank_plan.end() ) {
            // allocate plan and add to store
            frank_plan[ key ] = toast::fft::r1d_p ( toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast, toast::fft::direction::forward, 1.0 ) );
        }

        std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > :: iterator rit = rrank_plan.find ( key );
        if ( rit == rrank_plan.end() ) {
            // allocate plan and add to store
            rrank_plan[ key ] = toast::fft::r1d_p ( toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast, toast::fft::direction::backward, 1.0 ) );
        }

    }

    return;
}


toast::fft::r1d_p toast::fft::r1d_plan_store::forward ( int64_t len, int64_t n ) {

    int rank = 0;

    #ifdef _OPENMP  
    rank = omp_get_thread_num();
    #endif

    std::pair < int64_t, int64_t > key ( len, n );

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > & rank_plan = fplans_[ rank ];

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > :: iterator it = rank_plan.find ( key );

    if ( it == rank_plan.end() ) {
        if ( rank != 0 ) {
            TOAST_THROW( "attempting to allocate fft plan within a threaded region!" );
        }
        // allocate plan and add to store
        rank_plan[ key ] = toast::fft::r1d_p ( toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast, toast::fft::direction::forward, 1.0 ) );
    }

    return rank_plan[ key ];
}


toast::fft::r1d_p toast::fft::r1d_plan_store::backward ( int64_t len, int64_t n ) {

    int rank = 0;

    #ifdef _OPENMP  
    rank = omp_get_thread_num();
    #endif

    std::pair < int64_t, int64_t > key ( len, n );

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > & rank_plan = rplans_[ rank ];

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > :: iterator it = rank_plan.find ( key );

    if ( it == rank_plan.end() ) {
        if ( rank != 0 ) {
            TOAST_THROW( "attempting to allocate fft plan within a threaded region!" );
        }
        // allocate plan and add to store
        rank_plan[ key ] = toast::fft::r1d_p ( toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast, toast::fft::direction::backward, 1.0 ) );
    }

    return rank_plan[ key ];  
}





