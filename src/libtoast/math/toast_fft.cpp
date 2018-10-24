/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>
#include <toast_util_internal.hpp>

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

#include <sstream>
#include <cstdio>
#include <cstring>


// In all cases, the memory buffer used for these FFTs is allocated as a single
// block.  The first half of the block is for the real space data and the 
// second half of the buffer is for the complex Fourier space data.  The data
// in each half is further split into buffers for each of the inputs and 
// outputs.


#ifdef HAVE_FFTW

class r1d_fftw : public toast::fft::r1d {
    
    public :
        
        r1d_fftw ( int64_t length, int64_t n, toast::fft::plan_type type, 
            toast::fft::direction dir, double scale ) : 
            toast::fft::r1d ( length, n, type, dir, scale ) {


            int threads = 1;

            // enable threads
        #ifdef HAVE_FFTW_THREADS
        #   ifdef _OPENMP
            threads = omp_get_max_threads();
        #   endif
            fftw_plan_with_nthreads(threads);
        #endif

            // allocate memory

            data_.resize ( n_ * 2 * length_ );
            std::fill ( data_.begin(), data_.end(), 0 );

            // create vector views and raw pointers

            traw_ = static_cast < double * > ( & data_[0] );
            fraw_ = static_cast < double * > ( & data_[n_ * length_] );

            tview_.clear();
            fview_.clear();

            for ( int64_t i = 0; i < n_; ++i ) {
                tview_.push_back ( & data_[i * length_] );
                fview_.push_back ( & data_[(n_ + i) * length_] );
            }

            // create plan

            int ilength = static_cast < int > ( length_ );
            int iN = static_cast < int > ( n_ );

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

            plan_ = fftw_plan_many_r2r ( 1, &ilength, iN, rawin, &ilength, 
                1, ilength, rawout, &ilength, 1, ilength, &kind, flags);

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

            int64_t len = n_ * length_;

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
        
        r1d_mkl ( int64_t length, int64_t n, toast::fft::plan_type type, 
            toast::fft::direction dir, double scale ) : 
            toast::fft::r1d ( length, n, type, dir, scale ) {

            // Allocate memory.

            // Verify that datatype sizes are as expected.
            if ( sizeof(MKL_Complex16) != 2 * sizeof(double) ) {
                std::ostringstream o;
                o << "MKL_Complex16 is not the size of 2 doubles, check MKL API";
                TOAST_THROW( o.str().c_str() );
            }

            buflength_ = 2 * (length_ / 2 + 1);

            data_.resize ( 2 * n_ * buflength_ );

            // create vector views and raw pointers

            traw_ = static_cast < double * > ( & data_[0] );
            fraw_ = static_cast < double * > ( & data_[n_ * buflength_] );

            tview_.clear();
            fview_.clear();

            for ( int64_t i = 0; i < n_; ++i ) {
                tview_.push_back ( & data_[i * buflength_] );
                fview_.push_back ( & data_[(n_ + i) * buflength_] );
            }

            // Create plan.

            descriptor_ = 0;

            // For 1D transforms, the documentation implies that we just pass
            // the single number, rather than a one-element array.
            MKL_LONG status = DftiCreateDescriptor ( &descriptor_, 
                DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) length_ );
            check_status ( stderr, status );

            status = DftiSetValue ( descriptor_, DFTI_PLACEMENT, 
                DFTI_NOT_INPLACE );
            check_status ( stderr, status );

            // DFTI_COMPLEX_COMPLEX is not the default packing, but is
            // recommended in the documentation as the best choice.
            status = DftiSetValue ( descriptor_, 
                DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX );
            check_status ( stderr, status );

            // ---- Not needed for DFTI_COMPLEX_COMPLEX
            // status = DftiSetValue ( descriptor_, DFTI_PACKED_FORMAT, 
            //     DFTI_CCE_FORMAT );
            // check_status ( stderr, status );

            status = DftiSetValue ( descriptor_, DFTI_NUMBER_OF_TRANSFORMS, 
                n_ );
            check_status ( stderr, status );

            // From the docs...
            //
            // "The configuration parameters DFTI_INPUT_DISTANCE and
            // DFTI_OUTPUT_DISTANCE define the distance within input and
            // output data, and not within the forward-domain and 
            // backward-domain data."
            //
            // We also set the scaling here to mimic the normalization of FFTW.

            if ( dir_ == toast::fft::direction::forward ) {

                status = DftiSetValue ( descriptor_, DFTI_INPUT_DISTANCE, 
                    (MKL_LONG)buflength_ );
                check_status ( stderr, status );

                status = DftiSetValue ( descriptor_, DFTI_OUTPUT_DISTANCE, 
                    (MKL_LONG)(buflength_ / 2) );
                check_status ( stderr, status );

                status = DftiSetValue ( descriptor_, DFTI_FORWARD_SCALE, 
                    scale_ );
                check_status ( stderr, status );

                status = DftiSetValue ( descriptor_, DFTI_BACKWARD_SCALE, 
                    1.0 );
                check_status ( stderr, status );

            } else {

                status = DftiSetValue ( descriptor_, DFTI_OUTPUT_DISTANCE, 
                    (MKL_LONG)buflength_ );
                check_status ( stderr, status );

                status = DftiSetValue ( descriptor_, DFTI_INPUT_DISTANCE, 
                    (MKL_LONG)(buflength_ / 2) );
                check_status ( stderr, status );

                status = DftiSetValue ( descriptor_, DFTI_FORWARD_SCALE, 1.0 );
                check_status ( stderr, status );

                status = DftiSetValue ( descriptor_, DFTI_BACKWARD_SCALE,
                    scale_ / (double)length_ );
                check_status ( stderr, status );
            }

            status = DftiCommitDescriptor ( descriptor_ );
            check_status ( stderr, status );

            if ( status != 0 ) {
                std::ostringstream o;
                o << "failed to create mkl FFT plan, status = " << status 
                    << std::endl;
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
                status = DftiComputeForward ( descriptor_, traw_, 
                    (MKL_Complex16*)fraw_ );
                cce2hc();
            } else {
                hc2cce();
                status = DftiComputeBackward ( descriptor_, (MKL_Complex16*)fraw_, traw_ );
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

        void check_status ( FILE * fp, MKL_LONG status ) {
            if ( status != 0 ) {
                fprintf ( fp, "MKL DFTI error = %s\n", 
                    DftiErrorMessage(status) );
            }
            return;
        }

        void cce2hc ( ) {
            // CCE packed format is a vector of complex real / imaginary pairs
            // from 0 to Nyquist (0 to N/2 + 1).  We use the real space buffer
            // as workspace for the shuffling.

            int64_t half = (int64_t)( length_ / 2 );
            bool even = false;

            if ( length_ % 2 == 0 ) {
                even = true;
            }

            int64_t offcce;

            for ( int64_t i = 0; i < n_; ++i ) {

                // copy the first element.
                tview_[i][0] = fview_[i][0];
                
                if ( even ) {
                    // copy in the real part of the last element of the
                    // CCE data, which has N/2+1 complex element pairs.
                    // This element is located at 2 * half == length_.
                    tview_[i][half] = fview_[i][length_];
                }

                for ( int64_t j = 1; j < half; ++j ) {
                    offcce = 2 * j;
                    tview_[i][j] = fview_[i][offcce];
                    tview_[i][length_ - j] = fview_[i][offcce + 1];
                }

                tview_[i][length_] = 0.0;
                tview_[i][length_ + 1] = 0.0;

                memcpy ( (void*)fview_[i], (void*)tview_[i], buflength_*sizeof(double) );

            }

            memset ( (void*)traw_, 0, n_ * buflength_ * sizeof(double) );

            return;
        }

        void hc2cce ( ) {
            // CCE packed format is a vector of complex real / imaginary pairs
            // from 0 to Nyquist (0 to N/2 + 1).  We use the real space buffer
            // as workspace for the shuffling.

            int64_t half = (int64_t)( length_ / 2 );
            bool even = false;

            if ( length_ % 2 == 0 ) {
                even = true;
            }

            int64_t offcce;

            for ( int64_t i = 0; i < n_; ++i ) {

                // copy the first element.
                tview_[i][0] = fview_[i][0];
                tview_[i][1] = 0.0;
                
                if ( even ) {
                    tview_[i][length_] = fview_[i][half];
                    tview_[i][length_ + 1] = 0.0;
                }

                for ( int64_t j = 1; j < half; ++j ) {
                    offcce = 2 * j;
                    tview_[i][offcce] = fview_[i][j];
                    tview_[i][offcce + 1] = fview_[i][length_ - j];
                }

                memcpy ( (void*)fview_[i], (void*)tview_[i], buflength_*sizeof(double) );

            }

            memset ( (void*)traw_, 0, n_ * buflength_ * sizeof(double) );

            return;
        }

        DFTI_DESCRIPTOR_HANDLE descriptor_;
        toast::fft::fft_data data_;
        double * traw_;
        double * fraw_;
        std::vector < double * > tview_;
        std::vector < double * > fview_;
        int64_t buflength_;

};

#endif



// Public 1D plan class

toast::fft::r1d::r1d ( int64_t length, int64_t n, plan_type type, 
    direction dir, double scale ) {
    type_ = type;
    dir_ = dir;
    length_ = length;
    n_ = n;
    scale_ = scale;
}

int64_t toast::fft::r1d::length ( ) {
    return length_;
}

int64_t toast::fft::r1d::count ( ) {
    return n_;
}

toast::fft::r1d * toast::fft::r1d::create ( int64_t length, int64_t n, 
    plan_type type, direction dir, double scale ) {

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

    std::pair < int64_t, int64_t > key ( len, n );

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > 
        :: iterator fit = fplans_.find ( key );
    if ( fit == fplans_.end() ) {
        // allocate plan and add to store
        fplans_[ key ] = toast::fft::r1d_p ( 
            toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast,
            toast::fft::direction::forward, 1.0 ) );
    }

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > 
        :: iterator rit = rplans_.find ( key );
    if ( rit == rplans_.end() ) {
        // allocate plan and add to store
        rplans_[ key ] = toast::fft::r1d_p ( 
            toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast,
            toast::fft::direction::backward, 1.0 ) );
    }

    return;
}


toast::fft::r1d_p toast::fft::r1d_plan_store::forward ( int64_t len, 
    int64_t n ) {

    std::pair < int64_t, int64_t > key ( len, n );

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > 
        :: iterator it = fplans_.find ( key );

    if ( it == fplans_.end() ) {
        // allocate plan and add to store
        fplans_[ key ] = toast::fft::r1d_p ( 
            toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast, 
            toast::fft::direction::forward, 1.0 ) );
    }

    return fplans_[ key ];
}


toast::fft::r1d_p toast::fft::r1d_plan_store::backward ( int64_t len, 
    int64_t n ) {

    std::pair < int64_t, int64_t > key ( len, n );

    std::map < std::pair < int64_t, int64_t >, toast::fft::r1d_p > 
        :: iterator it = rplans_.find ( key );

    if ( it == rplans_.end() ) {
        // allocate plan and add to store
        rplans_[ key ] = toast::fft::r1d_p ( 
            toast::fft::r1d::create ( len, n, toast::fft::plan_type::fast, 
            toast::fft::direction::backward, 1.0 ) );
    }

    return rplans_[ key ];  
}





