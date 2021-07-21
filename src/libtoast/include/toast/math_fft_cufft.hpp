
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_FFT_CUFFT_HPP
#define TOAST_MATH_FFT_CUFFT_HPP

#include <vector>

#ifdef HAVE_CUDALIBS

# include <cufftw.h>

namespace toast {
class FFTPlanReal1DCUFFT : public toast::FFTPlanReal1D {
    public:

        FFTPlanReal1DCUFFT(int64_t length, int64_t n, toast::fft_plan_type type,
                          toast::fft_direction dir, double scale);

        ~FFTPlanReal1DCUFFT();

        void exec();

        double * tdata(int64_t indx);
        double * fdata(int64_t indx);

    private:

        // TODO keep data on GPU?
        fftw_plan plan_;
        toast::AlignedVector <double> data_;
        double * traw_;
        double * fraw_;
        std::vector <double *> tview_;
        std::vector <double *> fview_;
};
}

#endif // ifdef HAVE_CUDALIBS

#endif // ifndef TOAST_MATH_FFT_CUFFT_HPP
