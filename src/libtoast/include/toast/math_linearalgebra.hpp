
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_LINEARALGEBRA_HPP
#define TOAST_MATH_LINEARALGEBRA_HPP

#include "gpu_helpers.hpp"

// TODO
//  - batch operations would be much faster where possible

namespace toast {
    // encapsulates construction and destruction of GPU linear algebra handles
    // WARNING: this class is *not* threadsafe
    class LinearAlgebra {
    public:
        // handles creation and destruction of gpu handles
        LinearAlgebra();
        ~LinearAlgebra();

        // insures that the class is never copied
        LinearAlgebra(LinearAlgebra const&) = delete;
        void operator=(LinearAlgebra const&) = delete;

        void gemm(char * TRANSA, char * TRANSB, int * M, int * N, int * K,
                  double * ALPHA, double * A, int * LDA, double * B, int * LDB,
                  double * BETA, double * C, int * LDC) const;

        int syev_buffersize(char * JOBZ, char * UPLO, int * N, double * A,
                            int * LDA, double * W) const;

        void syev(char * JOBZ, char * UPLO, int * N, double * A, int * LDA,
                  double * W, double * WORK, int * LWORK, int * INFO) const;

        void symm(char * SIDE, char * UPLO, int * M, int * N, double * ALPHA,
                  double * A, int * LDA, double * B, int * LDB, double * BETA,
                  double * C, int * LDC) const;

        void syrk(char * UPLO, char * TRANS, int * N, int * K, double * ALPHA,
                  double * A, int * LDA, double * BETA, double * C, int * LDC) const;

        void dgelss(int * M, int * N, int * NRHS, double * A, int * LDA,
                    double * B, int * LDB, double * S, double * RCOND,
                    int * RANK, double * WORK, int * LWORK, int * INFO) const;
    private:
    #ifdef HAVE_CUDALIBS
        cublasHandle_t handleBlas = NULL;
        cusolverDnHandle_t handleSolver = NULL;
        int* gpu_allocated_integer = NULL;
    #endif
    };
}

#endif // ifndef TOAST_LAPACK_HPP