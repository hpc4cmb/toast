
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_FFT_HPP
#define TOAST_FFT_HPP

#include <vector>


namespace toast {

enum class fft_plan_type {
    fast,
    best
};

enum class fft_direction {
    forward,
    backward
};


// This uses aligned memory allocation
typedef std::vector <double, toast::simd_allocator <double> > fft_data;

class fft_r1d {
    public:

        typedef std::shared_ptr <fft_r1d> pshr;

        static fft_r1d * create(int64_t length, int64_t n, fft_plan_type type,
                                fft_direction dir, double scale);

        virtual ~fft_r1d() {}

        virtual void exec() {
            return;
        }

        virtual std::vector <double *> tdata() {
            return std::vector <double *> ();
        }

        virtual std::vector <double *> fdata() {
            return std::vector <double *> ();
        }

        int64_t length();

        int64_t count();

    protected:

        fft_r1d(int64_t length, int64_t n, fft_plan_type type,
                fft_direction dir, double scale);

        int64_t length_;
        int64_t n_;
        double scale_;
        fft_plan_type type_;
        fft_direction dir_;
};


// R1D FFT plan store

class fft_r1d_plan_store {
    public:

        ~fft_r1d_plan_store();
        static fft_r1d_plan_store & get();
        void cache(int64_t len, int64_t n = 1);
        fft_r1d::pshr forward(int64_t len, int64_t n = 1);
        fft_r1d::pshr backward(int64_t len, int64_t n = 1);
        void clear();

    private:

        fft_r1d_plan_store() {}

        std::map <std::pair <int64_t, int64_t>, fft_r1d::pshr> fplans_;
        std::map <std::pair <int64_t, int64_t>, fft_r1d::pshr> rplans_;
};

}

#endif // ifndef TOAST_RNG_HPP
