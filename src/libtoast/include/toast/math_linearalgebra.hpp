
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_LINEARALGEBRA_HPP
#define TOAST_MATH_LINEARALGEBRA_HPP

#include "gpu_helpers.hpp"

namespace toast {
    // encapsulates construction and destruction of GPU linear algebra handles
    // WARNING: this class is *not* threadsafe and one should create one LinearAlgebra object per thread
    class LinearAlgebra {
    public:
        // handles creation and destruction of gpu handles
        LinearAlgebra();
        ~LinearAlgebra();

        // insures that the class is never copied
        LinearAlgebra(LinearAlgebra const&) = delete;
        void operator=(LinearAlgebra const&) = delete;

        void gemm(char TRANSA, char TRANSB, int M, int N, int K,
                  double ALPHA, double * A, int LDA, double * B, int LDB,
                  double BETA, double * C, int LDC) const;

        void gemm_batched(char TRANSA, char TRANSB, int M, int N, int K,
                          double ALPHA, double * A_batch, int LDA, double * B_batch, int LDB,
                          double BETA, double * C_batch, int LDC, const int batchCount) const;

        int syev_buffersize(char JOBZ, char UPLO, int N, double * A,
                            int LDA, double * W) const;

        void syev(char JOBZ, char UPLO, int N, double * A, int LDA,
                  double * W, double * WORK, int LWORK, int * INFO);

        int syev_batched_buffersize(char JOBZ, char UPLO, int N, double * A_batch,
                                    int LDA, double * W_batch, const int batchCount) const;

        void syev_batched(char JOBZ, char UPLO, int N, double * A_batched,
                          int LDA, double * W_batched, double * WORK, int LWORK,
                          int * INFO, const int batchCount);

        void symm(char SIDE, char UPLO, int M, int N, double ALPHA,
                  double * A, int LDA, double * B, int LDB, double BETA,
                  double * C, int LDC) const;

        void syrk(char UPLO, char TRANS, int N, int K, double ALPHA,
                  double * A, int LDA, double BETA, double * C, int LDC) const;

        void gelss(int M, int N, int NRHS, double * A, int LDA,
                    double * B, int LDB, double * S, double RCOND,
                    int RANK, double * WORK, int LWORK, int * INFO) const;
    private:
    #ifdef HAVE_CUDALIBS
        cublasHandle_t handleBlas = NULL;
        cusolverDnHandle_t handleSolver = NULL;
        syevjInfo_t jacobiParameters = NULL;
    #endif
    };
}

#endif // TOAST_MATH_LINEARALGEBRA_HPP
