
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_fft_mkl.hpp>

#include <cstring>
#include <cmath>
#include <vector>


// The memory buffer used for these FFTs is allocated as a single
// block.  The first half of the block is for the real space data and the
// second half of the buffer is for the complex Fourier space data.  The data
// in each half is further split into buffers for each of the inputs and
// outputs.

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
        std::string msg("MKL_Complex16 is not the size of 2 doubles, check MKL API");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
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
        fprintf(fp, "MKL DFTI error = %s\n", DftiErrorMessage(status));
    }
}

// CCE packed format is a vector of complex real / imaginary pairs
// from 0 to Nyquist (0 to N/2 + 1).  We use the real space buffer
// as workspace for the shuffling.
void toast::FFTPlanReal1DMKL::cce2hc() {
    const int64_t half = length_ / 2;
    const bool even = (length_ % 2 == 0);

    for (int64_t i = 0; i < n_; i++)
    {
        // copy the first element.
        tview_[i][0] = fview_[i][0];

        if (even)
        {
            // copy in the real part of the last element of the
            // CCE data, which has N/2+1 complex element pairs.
            // This element is located at 2 * half == length_.
            tview_[i][half] = fview_[i][length_];
        }

        for (int64_t j = 1; j < half; j++)
        {
            const int64_t offcce = 2 * j;
            tview_[i][j] = fview_[i][offcce];
            tview_[i][length_ - j] = fview_[i][offcce + 1];
        }

        tview_[i][length_] = 0.0;
        tview_[i][length_ + 1] = 0.0;

        memcpy((void *)fview_[i], (void *)tview_[i], buflength_ * sizeof(double));
    }

    memset((void *)traw_, 0, n_ * buflength_ * sizeof(double));
}

// CCE packed format is a vector of complex real / imaginary pairs
// from 0 to Nyquist (0 to N/2 + 1).  We use the real space buffer
// as workspace for the shuffling.
void toast::FFTPlanReal1DMKL::hc2cce() {
    const int64_t half = length_ / 2;
    const bool even = (length_ % 2 == 0);

    for (int64_t i = 0; i < n_; i++)
    {
        // copy the first element.
        tview_[i][0] = fview_[i][0];
        tview_[i][1] = 0.0;

        if (even)
        {
            tview_[i][length_] = fview_[i][half];
            tview_[i][length_ + 1] = 0.0;
        }

        for (int64_t j = 1; j < half; j++)
        {
            const int64_t offcce = 2 * j;
            tview_[i][offcce] = fview_[i][j];
            tview_[i][offcce + 1] = fview_[i][length_ - j];
        }

        memcpy((void *)fview_[i], (void *)tview_[i], buflength_ * sizeof(double));
    }

    memset((void *)traw_, 0, n_ * buflength_ * sizeof(double));
}

#endif // ifdef HAVE_MKL
