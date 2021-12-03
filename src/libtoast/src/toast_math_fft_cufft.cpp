
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
    if (sizeof(cufftDoubleComplex) != 2 * sizeof(double)) {
        // this would kill the complexToHalfcomplex and halfcomplexToComplex functions
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg(
            "cufftDoubleComplex is not the size of 2 doubles, check CUFFT API");
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
    tview_.clear();
    fview_.clear();
    for (int64_t batchId = 0; batchId < n_; batchId++) {
        tview_.push_back(&traw_[batchId * buflength_]);
        fview_.push_back(&fraw_[batchId * buflength_]);
    }

    // creates a plan

    // 1D FFTs
    const int rank = 1;

    // Size of the Fourier transform
    int ilength = length_;

    // Input/output size with pitch (ignored for 1D transforms)
    int inembed = 0, onembed = 0;

    // Distance between two successive input/output elements
    const int istride = 1, ostride = 1;

    // Distance between input batches
    const int idist =
        (dir == toast::fft_direction::forward) ? buflength_ : (buflength_ / 2);

    // Distance between output batches
    const int odist =
        (dir == toast::fft_direction::forward) ? (buflength_ / 2) : buflength_;

    const cufftType fft_type =
        (dir == toast::fft_direction::forward) ? CUFFT_D2Z : CUFFT_Z2D;

    // Number of batched executions
    const int batch = n_;
    cufftResult errorCodePlan = cufftPlanMany(&plan_, rank, &ilength,
                                              &inembed, istride, idist,
                                              &onembed, ostride, odist,
                                              fft_type, batch);
    checkCufftErrorCode(errorCodePlan, "FFTPlanReal1DCUFFT::cufftPlanMany");
}

toast::FFTPlanReal1DCUFFT::~FFTPlanReal1DCUFFT() {
    cufftDestroy(plan_);
}

void toast::FFTPlanReal1DCUFFT::exec() {
    // number of elements to be moved around
    const int64_t nb_elements_real = n_ * buflength_;
    const int64_t nb_elements_complex = n_ * (buflength_ / 2);

    // actual execution of the FFT

    // R2C, real input in traw_ and complex output in fraw_
    if (dir_ == toast::fft_direction::forward) {
        // get input data from CPU
        cufftDoubleReal * idata = GPU_memory_pool.toDevice(traw_, nb_elements_real);
        cufftDoubleComplex * odata = GPU_memory_pool.alloc <cufftDoubleComplex> (
            nb_elements_complex);

        // execute plan
        const cufftResult errorCodeExec = cufftExecD2Z(plan_, idata, odata);
        checkCufftErrorCode(errorCodeExec, "FFTPlanReal1DCUFFT::cufftExecR2C");
        const cudaError statusSync = cudaDeviceSynchronize();
        checkCudaErrorCode(statusSync, "FFTPlanReal1DCUFFT::cudaDeviceSynchronize");

        // send output data to CPU
        GPU_memory_pool.free(idata);
        GPU_memory_pool.fromDevice((cufftDoubleComplex *)(traw_), odata,
                                   nb_elements_complex);

        // reorder data from rcrc... (stored in traw_) to rr...cc (stored in fraw_)
        complexToHalfcomplex(length_, n_, tview_.data(), fview_.data());
    } else { // C2R, complex input in fraw_ and real output in traw_
        // reorder data from rr...cc (stored in fraw_) to rcrc... (stored in traw_)
        halfcomplexToComplex(length_, n_, fview_.data(), tview_.data());

        // get input data from CPU
        cufftDoubleComplex * idata =
            GPU_memory_pool.toDevice((cufftDoubleComplex *)(traw_),
                                     nb_elements_complex);
        cufftDoubleReal * odata = GPU_memory_pool.alloc <cufftDoubleReal> (
            nb_elements_real);

        // execute plan
        const cufftResult errorCodeExec = cufftExecZ2D(plan_, idata, odata);
        checkCufftErrorCode(errorCodeExec, "FFTPlanReal1DCUFFT::cufftExecC2R");
        const cudaError statusSync = cudaDeviceSynchronize();
        checkCudaErrorCode(statusSync, "FFTPlanReal1DCUFFT::cudaDeviceSynchronize");

        // send output data to CPU
        GPU_memory_pool.free(idata);
        GPU_memory_pool.fromDevice(traw_, odata, nb_elements_real);
    }

    // normalize output
    double * output = (dir_ == toast::fft_direction::forward) ? fraw_ : traw_;
    const double scaling =
        (dir_ == toast::fft_direction::forward) ? scale_ : (scale_ / length_);
    if (scaling != 1.0) { // usually 1.0 in the forward case
        # pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_ * buflength_; i++) {
            output[i] *= scaling;
        }
    }
}

// reorder `nbBatch` arrays of size `length`
// from a traditional complex representation (rcrc... stored in `batchedComplexInputs`)
// to half-complex (rr...cc stored in `batchedHalfcomplexOutputs`)
// for more information on the half-complex format, see:
// https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
void toast::FFTPlanReal1DCUFFT::complexToHalfcomplex(const int64_t length,
                                                     const int64_t nbBatch,
                                                     double * batchedComplexInputs[],
                                                     double * batchedHalfcomplexOutputs[])
{
    const int64_t half = length / 2;
    const bool is_even = (length % 2) == 0;

    // iterates on all batches one after the other
    for (int64_t batchId = 0; batchId < nbBatch; batchId++) {
        // 0th value

        // real
        batchedHalfcomplexOutputs[batchId][0] = batchedComplexInputs[batchId][0];

        // imag is zero by convention and thus not encoded

        // all intermediate values
        # pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < half; i++) {
            batchedHalfcomplexOutputs[batchId][i] =
                batchedComplexInputs[batchId][2 * i];     // real
            batchedHalfcomplexOutputs[batchId][length -
                                               i] =
                batchedComplexInputs[batchId][2 * i + 1]; // imag
        }

        // n/2th value
        if (is_even) {
            batchedHalfcomplexOutputs[batchId][half] =
                batchedComplexInputs[batchId][length]; // real
            // imag is zero by convention and thus not encoded
        }
    }
}

// reorder `nbBatch` arrays of size `length`
// from half-complex (rr...cc stored in `batchedHalfcomplexInputs`)
// to a traditional complex representation (rcrc... stored in `batchedHComplexOutputs`)
// for more information on the half-complex format, see:
// https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
void toast::FFTPlanReal1DCUFFT::halfcomplexToComplex(const int64_t length,
                                                     const int64_t nbBatch,
                                                     double * batchedHalfcomplexInputs[],
                                                     double * batchedHComplexOutputs[])
{
    const int64_t half = length / 2;
    const bool is_even = (length % 2) == 0;

    // iterates on all batches one after the other
    for (int64_t batchId = 0; batchId < nbBatch; batchId++) {
        // 0th value

        // real
        batchedHComplexOutputs[batchId][0] = batchedHalfcomplexInputs[batchId][0];

        // imag is 0 by convention
        batchedHComplexOutputs[batchId][1] = 0.0;

        // all intermediate values
        # pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < half; i++) {
            // real
            batchedHComplexOutputs[batchId][2 * i] = \
                batchedHalfcomplexInputs[batchId][i];

            // imag
            batchedHComplexOutputs[batchId][2 * i + 1] = \
                batchedHalfcomplexInputs[batchId][length - i];
        }

        // n/2th value
        if (is_even) {
            // real
            batchedHComplexOutputs[batchId][length] =
                batchedHalfcomplexInputs[batchId][half];

            // imag is 0 by convention
            batchedHComplexOutputs[batchId][length + 1] = 0.0;
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
