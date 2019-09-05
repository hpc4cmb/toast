
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_FFT_FFTW_HPP
#define TOAST_MATH_FFT_FFTW_HPP

#include <vector>

#ifdef HAVE_FFTW

# include <fftw3.h>

namespace toast {
class FFTPlanReal1DFFTW : public toast::FFTPlanReal1D {
    public:

        FFTPlanReal1DFFTW(int64_t length, int64_t n, toast::fft_plan_type type,
                          toast::fft_direction dir, double scale);

        ~FFTPlanReal1DFFTW();

        void exec();

        double * tdata(int64_t indx);
        double * fdata(int64_t indx);

    private:

        fftw_plan plan_;
        toast::AlignedVector <double> data_;
        double * traw_;
        double * fraw_;
        std::vector <double *> tview_;
        std::vector <double *> fview_;
};
}

#endif // ifdef HAVE_FFTW

#endif // ifndef TOAST_MATH_FFT_FFTW_HPP
