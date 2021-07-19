
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_LINEARALGEBRA_HPP
#define TOAST_MATH_LINEARALGEBRA_HPP

#include "gpu_helpers.hpp"

namespace toast {
    namespace LinearAlgebra {
        void gemm(char TRANSA, char TRANSB, int M, int N, int K,
                  double ALPHA, double * A, int LDA, double * B, int LDB,
                  double BETA, double * C, int LDC);

        void gemm_batched(char TRANSA, char TRANSB, int M, int N, int K,
                          double ALPHA, double * A_batch, int LDA, double * B_batch, int LDB,
                          double BETA, double * C_batch, int LDC, int batchCount);

        void syev_batched(char JOBZ, char UPLO, int N, double * A_batched,
                          int LDA, double * W_batched, int * INFO, int batchCount);

        void symm(char SIDE, char UPLO, int M, int N, double ALPHA,
                  double * A, int LDA, double * B, int LDB, double BETA,
                  double * C, int LDC);

        void symm_batched(char SIDE, char UPLO, int M, int N, double ALPHA,
                          double * A_batch, int LDA, double * B_batch, int LDB, double BETA,
                          double * C_batch, int LDC, int batchCount);

        void syrk(char UPLO, char TRANS, int N, int K, double ALPHA,
                  double * A, int LDA, double BETA, double * C, int LDC);

        void syrk_batched(char UPLO, char TRANS, int N, int K, double ALPHA,
                          double * A_batch, int LDA, double BETA, double * C_batch, int LDC, int batchCount);

        void gels(int M, int N, int NRHS, double * A, int LDA,
                  double * B, int LDB, int * INFO);

        void gelss(int M, int N, int NRHS, double * A, int LDA,
                   double * B, int LDB, double * S, double RCOND,
                   int * RANK, double * WORK, int LWORK, int * INFO);
    }
}

#endif // TOAST_MATH_LINEARALGEBRA_HPP
