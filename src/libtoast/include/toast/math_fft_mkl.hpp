
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_FFT_MKL_HPP
#define TOAST_MATH_FFT_MKL_HPP

#include <vector>


#ifdef HAVE_MKL

# include <mkl_dfti.h>

namespace toast {
class FFTPlanReal1DMKL : public toast::FFTPlanReal1D {
    public:

        FFTPlanReal1DMKL(int64_t length, int64_t n, toast::fft_plan_type type,
                         toast::fft_direction dir, double scale);

        ~FFTPlanReal1DMKL();

        void exec();

        double * tdata(int64_t indx);

        double * fdata(int64_t indx);

    private:

        void check_status(FILE * fp, MKL_LONG status);

        void cce2hc();

        void hc2cce();

        DFTI_DESCRIPTOR_HANDLE descriptor_;
        toast::AlignedVector <double> data_;
        double * traw_;
        double * fraw_;
        std::vector <double *> tview_;
        std::vector <double *> fview_;
        int64_t buflength_;
};
}

#endif // ifdef HAVE_MKL

#endif // ifndef TOAST_MATH_FFT_MKL_HPP
