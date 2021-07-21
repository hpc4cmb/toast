
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_fft_fftw.hpp>
#include <toast/math_fft_mkl.hpp>
#include <toast/math_fft_cufft.hpp>

#include <cstring>
#include <cmath>

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
#ifdef HAVE_CUDALIBS
    return new FFTPlanReal1DCUFFT(length, n, type, dir, scale);
#elif HAVE_MKL
    return new FFTPlanReal1DMKL(length, n, type, dir, scale);
#elif HAVE_FFTW
    return new FFTPlanReal1DFFTW(length, n, type, dir, scale);
#else
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg = "FFTs require MKL, FFTW or CUFFT";
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
#endif
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
