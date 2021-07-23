
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

// TODO doc on halfcomplex
//  http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html

#ifdef HAVE_CUDALIBS

toast::FFTPlanReal1DCUFFT::FFTPlanReal1DCUFFT(
    int64_t length, int64_t n, toast::fft_plan_type type,
    toast::fft_direction dir, double scale) :
    toast::FFTPlanReal1D(length, n, type, dir, scale) {

    // allocate CPU memory
    data_.resize(n_ * 2 * length_);
    std::fill(data_.begin(), data_.end(), 0);

    // creates raw pointers for input and output
    // TODO might not be useful anymore
    traw_ = static_cast <double *> (&data_[0]);
    fraw_ = static_cast <double *> (&data_[n_ * length_]);

    // creates vector views that will be used by user to send and receive data
    tview_.clear();
    fview_.clear();
    for (int64_t i = 0; i < n_; ++i)
    {
        tview_.push_back(&data_[i * length_]);
        fview_.push_back(&data_[(n_ + i) * length_]);
    }

    // creates a plan
    int ilength = static_cast <int> (length_);
    int iN = static_cast <int> (n_);
    cufftType fft_type = (dir == toast::fft_direction::forward) ? CUFFT_R2C : CUFFT_C2R;
    cufftResult errorCodePlan = cufftPlanMany(&plan_, /*rank*/ 1, /*n*/ &ilength,
                  /*inembed*/ &ilength, /*istride*/ 1, /*idist*/ ilength,
                  /*onembed*/ &ilength, /*ostride*/ 1, /*odist*/ ilength,
                  fft_type, /*batch*/ iN);
    checkCufftErrorCode(errorCodePlan, "FFTPlanReal1DCUFFT::cufftPlanMany");
}

toast::FFTPlanReal1DCUFFT::~FFTPlanReal1DCUFFT() {
    cufftDestroy(plan_);
    tview_.clear();
    fview_.clear();
    data_.clear();
    // TODO deallocate GPU memory or allocate/deallocate in exec?
}

void toast::FFTPlanReal1DCUFFT::exec() {
    // TODO reoder input data

    if (dir_ == toast::fft_direction::forward) // R2C
    {
        // TODO get input data from CPU
        cufftReal* idata = GPU_memory_pool.alloc<cufftReal>(n_ * length_); //traw_
        cufftComplex* odata = GPU_memory_pool.alloc<cufftComplex>(n_ * length_); //fraw_
        // execute plan
        cufftResult errorCodeExec = cufftExecR2C(plan_, idata, odata);
        checkCufftErrorCode(errorCodeExec, "FFTPlanReal1DCUFFT::cufftExecR2C");
        cudaError statusSync = cudaDeviceSynchronize();
        checkCudaErrorCode(statusSync, "FFTPlanReal1DCUFFT::cudaDeviceSynchronize");
        // TODO send output data to CPU
        GPU_memory_pool.free(idata);
        GPU_memory_pool.free(odata);
    }
    else // C2R
    {
        // TODO get input data from CPU
        cufftComplex* idata = GPU_memory_pool.alloc<cufftComplex>(n_ * length_); //fraw_
        cufftReal* odata = GPU_memory_pool.alloc<cufftReal>(n_ * length_); //traw_
        // execute plan
        cufftResult errorCodeExec = cufftExecC2R(plan_, idata, odata);
        checkCufftErrorCode(errorCodeExec, "FFTPlanReal1DCUFFT::cufftExecC2R");
        cudaError statusSync = cudaDeviceSynchronize();
        checkCudaErrorCode(statusSync, "FFTPlanReal1DCUFFT::cudaDeviceSynchronize");
        // TODO send output data to CPU
        GPU_memory_pool.free(idata);
        GPU_memory_pool.free(odata);
    }

    // TODO reorder output data

    // gets parameters to normalize output
    double * rawout;
    double norm;
    if (dir_ == toast::fft_direction::forward) {
        rawout = fraw_;
        norm = scale_;
    } else {
        rawout = traw_;
        norm = scale_ / static_cast <double> (length_);
    }

    // normalize output
    int64_t len = n_ * length_;
    for (int64_t i = 0; i < len; ++i) {
        rawout[i] *= norm;
    }
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
