
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_fft_fftw.hpp>

#include <cstring>
#include <cmath>
#include <vector>

// The memory buffer used for these FFTs is allocated as a single
// block.  The first half of the block is for the real space data and the
// second half of the buffer is for the complex Fourier space data.  The data
// in each half is further split into buffers for each of the inputs and
// outputs.

// #ifdef HAVE_FFTW
// the not HAVE_CUDALIBS is needed to deal with conflict in symbol names between both
// libs
#if defined(HAVE_FFTW) && !defined(HAVE_CUDALIBS)

toast::FFTPlanReal1DFFTW::FFTPlanReal1DFFTW(
    int64_t length, int64_t n, toast::fft_plan_type type,
    toast::fft_direction dir, double scale) : toast::FFTPlanReal1D(length, n, type, dir,
                                                                   scale)
{
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

    for (int64_t i = 0; i < n_; ++i)
    {
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

    if (dir == toast::fft_direction::forward)
    {
        rawin = traw_;
        rawout = fraw_;
        kind = FFTW_R2HC;
    }
    else
    {
        rawin = fraw_;
        rawout = traw_;
        kind = FFTW_HC2R;
    }

    flags = flags | FFTW_DESTROY_INPUT;

    if (type == toast::fft_plan_type::best)
    {
        flags = flags | FFTW_MEASURE;
    }
    else
    {
        flags = flags | FFTW_ESTIMATE;
    }

    plan_ = fftw_plan_many_r2r(1, &ilength, iN, rawin, &ilength,
                               1, ilength, rawout, &ilength, 1,
                               ilength, &kind, flags);
    if (plan_ == NULL)
    {
        // This can occur, for example, if MKL is masquerading as FFTW.
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg =
            "fftw_plan_many_r2r returned plan=NULL unexpectedly; MKL linking issue?";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
}

toast::FFTPlanReal1DFFTW::~FFTPlanReal1DFFTW()
{
    fftw_destroy_plan(static_cast <fftw_plan> (plan_));
    tview_.clear();
    fview_.clear();
    data_.clear();
}

void toast::FFTPlanReal1DFFTW::exec()
{
    fftw_execute(plan_);

    double * rawout;
    double norm;

    if (dir_ == toast::fft_direction::forward)
    {
        rawout = fraw_;
        norm = scale_;
    }
    else
    {
        rawout = traw_;
        norm = scale_ / static_cast <double> (length_);
    }

    int64_t len = n_ * length_;

    for (int64_t i = 0; i < len; ++i)
    {
        rawout[i] *= norm;
    }

    return;
}

double * toast::FFTPlanReal1DFFTW::tdata(int64_t indx)
{
    if ((indx < 0) || (indx >= n_))
    {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return tview_[indx];
}

double * toast::FFTPlanReal1DFFTW::fdata(int64_t indx)
{
    if ((indx < 0) || (indx >= n_))
    {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg = "batch index out of range";
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return fview_[indx];
}

#endif // ifdef HAVE_FFTW
