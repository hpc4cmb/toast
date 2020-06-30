
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_fft_fftw.hpp>
#include <toast/math_fft_mkl.hpp>

#include <cmath>
#include <vector>


// In all cases, the memory buffer used for these FFTs is allocated as a single
// block.  The first half of the block is for the real space data and the
// second half of the buffer is for the complex Fourier space data.  The data
// in each half is further split into buffers for each of the inputs and
// outputs.

#ifdef HAVE_FFTW

toast::FFTPlanReal1DFFTW::FFTPlanReal1DFFTW(
    int64_t length, int64_t n, toast::fft_plan_type type,
    toast::fft_direction dir, double scale) :
    toast::FFTPlanReal1D(length, n, type, dir, scale) {
    int threads = 1;

    // enable threads
    # ifdef HAVE_FFTW_THREADS
    auto env = toast::Environment::get();
    threads = env.max_threads();
    fftw_plan_with_nthreads(threads);
    # endif // ifdef HAVE_FFTW_THREADS

    // allocate memory

    data_.resize(n_ * 2 * length_);
    std::fill(data_.begin(), data_.end(), 0);

    // create vector views and raw pointers

    traw_ = static_cast <double *> (&data_[0]);
    fraw_ = static_cast <double *> (&data_[n_ * length_]);

    tview_.clear();
    fview_.clear();

    for (int64_t i = 0; i < n_; ++i) {
        tview_.push_back(&data_[i * length_]);
        fview_.push_back(&data_[(n_ + i) * length_]);
    }

    // create plan

    int ilength = static_cast <int> (length_);
    int iN = static_cast <int> (n_);

    unsigned flags = 0;
    double * rawin;
    double * rawout;

    fftw_r2r_kind kind;

    if (dir == toast::fft_direction::forward) {
        rawin = traw_;
        rawout = fraw_;
        kind = FFTW_R2HC;
    } else {
        rawin = fraw_;
        rawout = traw_;
        kind = FFTW_HC2R;
    }

    flags = flags | FFTW_DESTROY_INPUT;

    if (type == toast::fft_plan_type::best) {
        flags = flags | FFTW_MEASURE;
    } else {
        flags = flags | FFTW_ESTIMATE;
    }

    plan_ = fftw_plan_many_r2r(1, &ilength, iN, rawin, &ilength,
                               1, ilength, rawout, &ilength, 1,
                               ilength, &kind, flags);
    if (plan_ == NULL) {
        // This can occur, for example, if MKL is masquerading as FFTW.
        std::string msg = "fftw_plan_many_r2r returned plan=NULL unexpectedly; MKL linking issue?";
        throw std::runtime_error(msg.c_str());
    }
}

toast::FFTPlanReal1DFFTW::~FFTPlanReal1DFFTW() {
    fftw_destroy_plan(static_cast <fftw_plan> (plan_));
    tview_.clear();
    fview_.clear();
    data_.clear();
}

void toast::FFTPlanReal1DFFTW::exec() {
    fftw_execute(plan_);

    double * rawout;
    double norm;

    if (dir_ == toast::fft_direction::forward) {
        rawout = fraw_;
        norm = scale_;
    } else {
        rawout = traw_;
        norm = scale_ / static_cast <double> (length_);
    }

    int64_t len = n_ * length_;

    for (int64_t i = 0; i < len; ++i) {
        rawout[i] *= norm;
    }

    return;
}

double * toast::FFTPlanReal1DFFTW::tdata(int64_t indx) {
    if ((indx < 0) || (indx >= n_)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return tview_[indx];
}

double * toast::FFTPlanReal1DFFTW::fdata(int64_t indx) {
    if ((indx < 0) || (indx >= n_)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return fview_[indx];
}

#endif // ifdef HAVE_FFTW


#ifdef HAVE_MKL

toast::FFTPlanReal1DMKL::FFTPlanReal1DMKL(
    int64_t length, int64_t n, toast::fft_plan_type type,
    toast::fft_direction dir, double scale) :
    toast::FFTPlanReal1D(length, n, type, dir, scale) {
    // Allocate memory.

    // Verify that datatype sizes are as expected.
    if (sizeof(MKL_Complex16) != 2 * sizeof(double)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "MKL_Complex16 is not the size of 2 doubles, "
          << "check MKL API";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    buflength_ = 2 * (length_ / 2 + 1);

    data_.resize(2 * n_ * buflength_);

    // create vector views and raw pointers

    traw_ = static_cast <double *> (&data_[0]);
    fraw_ = static_cast <double *> (&data_[n_ * buflength_]);

    tview_.clear();
    fview_.clear();

    for (int64_t i = 0; i < n_; ++i) {
        tview_.push_back(&data_[i * buflength_]);
        fview_.push_back(&data_[(n_ + i) * buflength_]);
    }

    // Create plan.

    descriptor_ = 0;

    // For 1D transforms, the documentation implies that we just pass
    // the single number, rather than a one-element array.
    MKL_LONG status = DftiCreateDescriptor(&descriptor_,
                                           DFTI_DOUBLE, DFTI_REAL, 1,
                                           (MKL_LONG)length_);
    check_status(stderr, status);

    status = DftiSetValue(descriptor_, DFTI_PLACEMENT,
                          DFTI_NOT_INPLACE);
    check_status(stderr, status);

    // DFTI_COMPLEX_COMPLEX is not the default packing, but is
    // recommended in the documentation as the best choice.
    status = DftiSetValue(descriptor_,
                          DFTI_CONJUGATE_EVEN_STORAGE,
                          DFTI_COMPLEX_COMPLEX);
    check_status(stderr, status);

    // ---- Not needed for DFTI_COMPLEX_COMPLEX
    // status = DftiSetValue ( descriptor_, DFTI_PACKED_FORMAT,
    //     DFTI_CCE_FORMAT );
    // check_status ( stderr, status );

    status = DftiSetValue(descriptor_, DFTI_NUMBER_OF_TRANSFORMS,
                          n_);
    check_status(stderr, status);

    // From the docs...
    //
    // "The configuration parameters DFTI_INPUT_DISTANCE and
    // DFTI_OUTPUT_DISTANCE define the distance within input and
    // output data, and not within the forward-domain and
    // backward-domain data."
    //
    // We also set the scaling here to mimic the normalization of FFTW.

    if (dir_ == toast::fft_direction::forward) {
        status = DftiSetValue(descriptor_, DFTI_INPUT_DISTANCE,
                              (MKL_LONG)buflength_);
        check_status(stderr, status);

        status = DftiSetValue(descriptor_, DFTI_OUTPUT_DISTANCE,
                              (MKL_LONG)(buflength_ / 2));
        check_status(stderr, status);

        status = DftiSetValue(descriptor_, DFTI_FORWARD_SCALE,
                              scale_);
        check_status(stderr, status);

        status = DftiSetValue(descriptor_, DFTI_BACKWARD_SCALE,
                              1.0);
        check_status(stderr, status);
    } else {
        status = DftiSetValue(descriptor_, DFTI_OUTPUT_DISTANCE,
                              (MKL_LONG)buflength_);
        check_status(stderr, status);

        status = DftiSetValue(descriptor_, DFTI_INPUT_DISTANCE,
                              (MKL_LONG)(buflength_ / 2));
        check_status(stderr, status);

        status = DftiSetValue(descriptor_, DFTI_FORWARD_SCALE, 1.0);
        check_status(stderr, status);

        status = DftiSetValue(descriptor_, DFTI_BACKWARD_SCALE,
                              scale_ / (double)length_);
        check_status(stderr, status);
    }

    status = DftiCommitDescriptor(descriptor_);
    check_status(stderr, status);

    if (status != 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "failed to create mkl FFT plan, status = "
          << status << std::endl
          << "Message: " << DftiErrorMessage(status);
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }
}

toast::FFTPlanReal1DMKL::~FFTPlanReal1DMKL() {
    MKL_LONG status = DftiFreeDescriptor(&descriptor_);
}

void toast::FFTPlanReal1DMKL::exec() {
    MKL_LONG status = 0;

    if (dir_ == toast::fft_direction::forward) {
        status = DftiComputeForward(descriptor_, traw_,
                                    (MKL_Complex16 *)fraw_);
        cce2hc();
    } else {
        hc2cce();
        status = DftiComputeBackward(descriptor_,
                                     (MKL_Complex16 *)fraw_, traw_);
    }

    if (status != 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "failed to execute MKL transform, status = " << status;
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    return;
}

double * toast::FFTPlanReal1DMKL::tdata(int64_t indx) {
    if ((indx < 0) || (indx >= n_)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return tview_[indx];
}

double * toast::FFTPlanReal1DMKL::fdata(int64_t indx) {
    if ((indx < 0) || (indx >= n_)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return fview_[indx];
}

void toast::FFTPlanReal1DMKL::check_status(FILE * fp, MKL_LONG status) {
    if (status != 0) {
        fprintf(fp, "MKL DFTI error = %s\n",
                DftiErrorMessage(status));
    }
    return;
}

void toast::FFTPlanReal1DMKL::cce2hc() {
    // CCE packed format is a vector of complex real / imaginary pairs
    // from 0 to Nyquist (0 to N/2 + 1).  We use the real space buffer
    // as workspace for the shuffling.

    int64_t half = (int64_t)(length_ / 2);
    bool even = false;

    if (length_ % 2 == 0) {
        even = true;
    }

    int64_t offcce;

    for (int64_t i = 0; i < n_; ++i) {
        // copy the first element.
        tview_[i][0] = fview_[i][0];

        if (even) {
            // copy in the real part of the last element of the
            // CCE data, which has N/2+1 complex element pairs.
            // This element is located at 2 * half == length_.
            tview_[i][half] = fview_[i][length_];
        }

        for (int64_t j = 1; j < half; ++j) {
            offcce = 2 * j;
            tview_[i][j] = fview_[i][offcce];
            tview_[i][length_ - j] = fview_[i][offcce + 1];
        }

        tview_[i][length_] = 0.0;
        tview_[i][length_ + 1] = 0.0;

        memcpy((void *)fview_[i], (void *)tview_[i],
               buflength_ * sizeof(double));
    }

    memset((void *)traw_, 0, n_ * buflength_ * sizeof(double));

    return;
}

void toast::FFTPlanReal1DMKL::hc2cce() {
    // CCE packed format is a vector of complex real / imaginary pairs
    // from 0 to Nyquist (0 to N/2 + 1).  We use the real space buffer
    // as workspace for the shuffling.

    int64_t half = (int64_t)(length_ / 2);
    bool even = false;

    if (length_ % 2 == 0) {
        even = true;
    }

    int64_t offcce;

    for (int64_t i = 0; i < n_; ++i) {
        // copy the first element.
        tview_[i][0] = fview_[i][0];
        tview_[i][1] = 0.0;

        if (even) {
            tview_[i][length_] = fview_[i][half];
            tview_[i][length_ + 1] = 0.0;
        }

        for (int64_t j = 1; j < half; ++j) {
            offcce = 2 * j;
            tview_[i][offcce] = fview_[i][j];
            tview_[i][offcce + 1] = fview_[i][length_ - j];
        }

        memcpy((void *)fview_[i], (void *)tview_[i],
               buflength_ * sizeof(double));
    }

    memset((void *)traw_, 0, n_ * buflength_ * sizeof(double));

    return;
}

#endif // ifdef HAVE_MKL


// Public 1D plan class

toast::FFTPlanReal1D::FFTPlanReal1D(int64_t length, int64_t n,
                                    fft_plan_type type,
                                    fft_direction dir, double scale) {
    type_ = type;
    dir_ = dir;
    length_ = length;
    n_ = n;
    scale_ = scale;
}

int64_t toast::FFTPlanReal1D::length() {
    return length_;
}

int64_t toast::FFTPlanReal1D::count() {
    return n_;
}

toast::FFTPlanReal1D * toast::FFTPlanReal1D::create(int64_t length, int64_t n,
                                                    fft_plan_type type,
                                                    fft_direction dir,
                                                    double scale) {
    #ifdef HAVE_MKL

    return new FFTPlanReal1DMKL(length, n, type, dir, scale);

    #else // ifdef HAVE_MKL
    # ifdef HAVE_FFTW

    return new FFTPlanReal1DFFTW(length, n, type, dir, scale);

    # else // ifdef HAVE_FFTW
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg = "FFTs require MKL or FFTW";
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    # endif // ifdef HAVE_FFTW
    #endif  // ifdef HAVE_MKL

    return NULL;
}

// Persistant storage of 1D plans for a fixed size

toast::FFTPlanReal1DStore::~FFTPlanReal1DStore() {}

void toast::FFTPlanReal1DStore::clear() {
    fplans_.clear();
    rplans_.clear();
    return;
}

toast::FFTPlanReal1DStore & toast::FFTPlanReal1DStore::get() {
    static toast::FFTPlanReal1DStore instance;
    return instance;
}

void toast::FFTPlanReal1DStore::cache(int64_t len, int64_t n) {
    std::pair <int64_t, int64_t> key(len, n);

    std::map <std::pair <int64_t, int64_t>, toast::FFTPlanReal1D::pshr>
    ::iterator fit = fplans_.find(key);
    if (fit == fplans_.end()) {
        // allocate plan and add to store
        fplans_[key] = toast::FFTPlanReal1D::pshr(
            toast::FFTPlanReal1D::create(len, n, toast::fft_plan_type::fast,
                                         toast::fft_direction::forward, 1.0));
    }

    std::map <std::pair <int64_t, int64_t>, toast::FFTPlanReal1D::pshr>
    ::iterator rit = rplans_.find(key);
    if (rit == rplans_.end()) {
        // allocate plan and add to store
        rplans_[key] = toast::FFTPlanReal1D::pshr(
            toast::FFTPlanReal1D::create(len, n, toast::fft_plan_type::fast,
                                         toast::fft_direction::backward, 1.0));
    }

    return;
}

toast::FFTPlanReal1D::pshr toast::FFTPlanReal1DStore::forward(int64_t len,
                                                              int64_t n) {
    std::pair <int64_t, int64_t> key(len, n);

    std::map <std::pair <int64_t, int64_t>, toast::FFTPlanReal1D::pshr>
    ::iterator it = fplans_.find(key);

    if (it == fplans_.end()) {
        // allocate plan and add to store
        fplans_[key] = toast::FFTPlanReal1D::pshr(
            toast::FFTPlanReal1D::create(len, n, toast::fft_plan_type::fast,
                                         toast::fft_direction::forward, 1.0));
    }

    return fplans_[key];
}

toast::FFTPlanReal1D::pshr toast::FFTPlanReal1DStore::backward(int64_t len,
                                                               int64_t n) {
    std::pair <int64_t, int64_t> key(len, n);

    std::map <std::pair <int64_t, int64_t>, toast::FFTPlanReal1D::pshr>
    ::iterator it = rplans_.find(key);

    if (it == rplans_.end()) {
        // allocate plan and add to store
        rplans_[key] = toast::FFTPlanReal1D::pshr(
            toast::FFTPlanReal1D::create(len, n, toast::fft_plan_type::fast,
                                         toast::fft_direction::backward, 1.0));
    }

    return rplans_[key];
}
