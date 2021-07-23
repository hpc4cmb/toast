
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_fft_cufft.hpp>

#include <cstring>
#include <cmath>
#include <vector>

// TODO
//  move data to GPU
//  decide on which data should be on gpu by default
//  take into account need for halfcomplex format

#ifdef HAVE_CUDALIBS

toast::FFTPlanReal1DCUFFT::FFTPlanReal1DCUFFT(
    int64_t length, int64_t n, toast::fft_plan_type type,
    toast::fft_direction dir, double scale) :
    toast::FFTPlanReal1D(length, n, type, dir, scale) {

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

    flags = flags | FFTW_DESTROY_INPUT;

    if (type == toast::fft_plan_type::best) {
        flags = flags | FFTW_MEASURE;
    } else {
        flags = flags | FFTW_ESTIMATE;
    }

    // TODO doc on halfcomplex
    //  http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
    if (dir == toast::fft_direction::forward) {
        rawin = traw_;
        rawout = fraw_;
        // TODO we should store output in halfcomplex-format
        //  it is expecting a fftw_complex
        plan_ = fftw_plan_many_dft_r2c(1, &ilength, iN,
                                       rawin, &ilength, 1, ilength,
                                       rawout, &ilength, 1, ilength,
                                       flags);
    } else {
        rawin = fraw_;
        rawout = traw_;
        // TODO we should store output in halfcomplex-format
        //  it is expecting a fftw_complex
        plan_ = fftw_plan_many_dft_c2r(1, &ilength, iN,
                                       rawin, &ilength, 1, ilength,
                                       rawout, &ilength, 1, ilength,
                                       flags);
    }

    if (plan_ == NULL) {
        // This can occur, for example, if MKL is masquerading as FFTW.
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "fftw_plan_many_r2r returned plan=NULL unexpectedly; MKL linking issue?";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
}

toast::FFTPlanReal1DCUFFT::~FFTPlanReal1DCUFFT() {
    fftw_destroy_plan(plan_);
    tview_.clear();
    fview_.clear();
    data_.clear();
}

void toast::FFTPlanReal1DCUFFT::exec() {
    // TODO reoder data
    //  send data to GPU
    fftw_execute(plan_);
    // TODO get data from GPU
    //  deorder it

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

double * toast::FFTPlanReal1DCUFFT::tdata(int64_t indx) {
    if ((indx < 0) || (indx >= n_)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return tview_[indx];
}

double * toast::FFTPlanReal1DCUFFT::fdata(int64_t indx) {
    if ((indx < 0) || (indx >= n_)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return fview_[indx];
}

#endif // ifdef HAVE_CUDALIBS
