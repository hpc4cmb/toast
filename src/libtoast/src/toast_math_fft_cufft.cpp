
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_fft_cufft.hpp>

#include <cstring>
#include <cmath>

#ifdef HAVE_CUDALIBS

toast::FFTPlanReal1DCUFFT::FFTPlanReal1DCUFFT(
    int64_t length, int64_t n, toast::fft_plan_type type,
    toast::fft_direction dir, double scale) :
    toast::FFTPlanReal1D(length, n, type, dir, scale) {

    // Verifies that datatype sizes are as expected.
    if (sizeof(cufftDoubleComplex) != 2 * sizeof(double))
    {
        // this would kill the complexToHalfcomplex and halfcomplexToComplex functions
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("cufftDoubleComplex is not the size of 2 doubles, check CUFFT API");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    // size needed to store the complex numbers representing the halfcomplex
    // taking into account the addition of a few, previously-implicit, zeroes
    buflength_ = 2 * (1 + (length_ / 2));

    // allocate CPU memory
    data_.resize(2 * n_ * buflength_);
    std::fill(data_.begin(), data_.end(), 0.0);

    // creates raw pointers for input and output
    traw_ = static_cast <double *> (&data_[0]);
    fraw_ = static_cast <double *> (&data_[n_ * buflength_]);

    // creates vector views that will be used by user to send and receive data
    // TODO we could parallelize this loop but is it useful?
    tview_.clear();
    fview_.clear();
    for (int64_t batchId = 0; batchId < n_; batchId++)
    {
        tview_.push_back(&traw_[batchId * buflength_]);
        fview_.push_back(&fraw_[batchId * buflength_]);
    }

    // creates a plan
    int ilength = static_cast <int> (length_);
    int iN = static_cast <int> (n_);
    cufftType fft_type = (dir == toast::fft_direction::forward) ? CUFFT_D2Z : CUFFT_Z2D;
    cufftResult errorCodePlan = cufftPlanMany(&plan_, /*rank*/ 1, /*n*/ &ilength,
                                              /*inembed*/ NULL, /*istride*/ 1, /*idist*/ ilength,
                                              /*onembed*/ NULL, /*ostride*/ 1, /*odist*/ ilength,
                                              fft_type, /*batch*/ iN);
    checkCufftErrorCode(errorCodePlan, "FFTPlanReal1DCUFFT::cufftPlanMany");
}

toast::FFTPlanReal1DCUFFT::~FFTPlanReal1DCUFFT() {
    cufftDestroy(plan_);
}

void toast::FFTPlanReal1DCUFFT::exec() {
    // number of elements to be manipulated
    const int64_t nb_elements_real = n_ * length_;
    const int64_t nb_elements_complex = n_ * (1 + (length_ / 2));

    // actual execution of the FFT
    if (dir_ == toast::fft_direction::forward) // R2C
    {
        // get input data from CPU
        cufftDoubleReal* idata = GPU_memory_pool.toDevice(traw_, nb_elements_real);
        cufftDoubleComplex* odata = GPU_memory_pool.alloc<cufftDoubleComplex>(nb_elements_complex);
        // execute plan
        const cufftResult errorCodeExec = cufftExecD2Z(plan_, idata, odata);
        checkCufftErrorCode(errorCodeExec, "FFTPlanReal1DCUFFT::cufftExecR2C");
        const cudaError statusSync = cudaDeviceSynchronize();
        checkCudaErrorCode(statusSync, "FFTPlanReal1DCUFFT::cudaDeviceSynchronize");
        // send output data to CPU
        GPU_memory_pool.free(idata);
        GPU_memory_pool.fromDevice((cufftDoubleComplex*)(traw_), odata, nb_elements_complex);
        // reorder data from rcrc... (stored in traw_) to rr...cc (stored in fraw_)
        complexToHalfcomplex();
    }
    else // C2R
    {
        // reorder data from rr...cc (stored in fraw_) to rcrc... (stored in traw_)
        halfcomplexToComplex();
        // get input data from CPU
        cufftDoubleComplex* idata = GPU_memory_pool.toDevice((cufftDoubleComplex*)(traw_), nb_elements_complex);
        cufftDoubleReal* odata = GPU_memory_pool.alloc<cufftDoubleReal>(nb_elements_real);
        // execute plan
        const cufftResult errorCodeExec = cufftExecZ2D(plan_, idata, odata);
        checkCufftErrorCode(errorCodeExec, "FFTPlanReal1DCUFFT::cufftExecC2R");
        const cudaError statusSync = cudaDeviceSynchronize();
        checkCudaErrorCode(statusSync, "FFTPlanReal1DCUFFT::cudaDeviceSynchronize");
        // send output data to CPU
        GPU_memory_pool.free(idata);
        GPU_memory_pool.fromDevice(traw_, odata, nb_elements_real);
    }

    // gets parameters to rescale output
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

    // normalize output
    // TODO we could parallelize this loop
    for (int64_t i = 0; i < n_ * buflength_; i++)
    {
        rawout[i] *= norm;
    }
}

// moves CCE packed data in tview_ / traw_ to HC packed data in fview_ / fraw_
// CCE packed format is a vector of complex real / imaginary pairs from 0 to Nyquist (0 to N/2 + 1).
// see: https://accserv.lepp.cornell.edu/svn/packages/fftw/doc/html/The-Halfcomplex_002dformat-DFT.html
void toast::FFTPlanReal1DCUFFT::complexToHalfcomplex()
{
    const int64_t half = length_ / 2;
    const bool is_even = (length_ % 2) == 0;

    // TODO we could parallelize this loop if realistic test cases have enough batch elements
    // iterates on all batches one after the other
    for (int64_t batchId = 0; batchId < n_; batchId++)
    {
        // 0th value
        fview_[batchId][0] = tview_[batchId][0]; // real
        // imag is zero by convention and thus not encoded

        // all intermediate values
        for (int64_t i = 1; i < half; i++)
        {
            fview_[batchId][i] = tview_[batchId][2 * i]; // real
            fview_[batchId][length_ - i] = tview_[batchId][2 * i + 1]; // imag
        }

        // n/2th value
        if (is_even)
        {
            fview_[batchId][half] = tview_[batchId][length_]; // real
            // imag is zero by convention and thus not encoded
        }

        // sets additional, buffer, values to 0
        fview_[batchId][length_] = 0.0;
        fview_[batchId][length_ + 1] = 0.0;
    }
}

// moves HC packed data in fview_ / fraw_ to CCE packed data in tview_ / traw_
// CCE packed format is a vector of complex real / imaginary pairs from 0 to Nyquist (0 to N/2 + 1).
// see: https://accserv.lepp.cornell.edu/svn/packages/fftw/doc/html/The-Halfcomplex_002dformat-DFT.html
void toast::FFTPlanReal1DCUFFT::halfcomplexToComplex() {
    const int64_t half = length_ / 2;
    const bool is_even = (length_ % 2) == 0;

    // TODO we could parallelize this loop if realistic test cases have enough batch elements
    // iterates on all batches one after the other
    for (int64_t batchId = 0; batchId < n_; batchId++)
    {
        // 0th value
        tview_[batchId][0] = fview_[batchId][0]; // real
        tview_[batchId][1] = 0.0; // imag is 0 by convention

        // all intermediate values
        for (int64_t i = 1; i < half; i++)
        {
            tview_[batchId][2 * i] = fview_[batchId][i]; // real
            tview_[batchId][2 * i + 1] = fview_[batchId][length_ - i]; // imag
        }

        // n/2th value
        if (is_even)
        {
            tview_[batchId][length_] = fview_[batchId][half]; // real
            tview_[batchId][length_ + 1] = 0.0; // imag is 0 by convention
        }
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
