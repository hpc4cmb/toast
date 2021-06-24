
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_LAPACK_HPP
#define TOAST_MATH_LAPACK_HPP

#ifdef HAVE_CUDALIBS
#include <cublas_v2.h>
#include <cusolverDn.h>

void checkCudaErrorCode(const cudaError errorCode);
void checkCublasErrorCode(const cublasStatus_t errorCode);
void checkCusolverErrorCode(const cusolverStatus_t errorCode);
#endif

// TODO:
//  - rename file
//  - delete copy methods to insure handle is handled properly
//  - put gpu helper function in dedicated file

namespace toast {
    class LinearAlgebra {
    public:
        LinearAlgebra()
        {
            #ifdef HAVE_CUDALIBS
            // creates cublas handle
            cublasStatus_t errorCodeHandle = cublasCreate(&handleBlas);
            checkCublasErrorCode(errorCodeHandle);
            // creates cusolver handle
            cusolverStatus_t statusHandle = cusolverDnCreate(&handleSolver);
            checkCusolverErrorCode(statusHandle);
            #endif
        };

        ~LinearAlgebra()
        {
            #ifdef HAVE_CUDALIBS
            // free cublas handle
            cublasDestroy(handleBlas);
            // free cusolver handle
            cusolverDnDestroy(handleSolver);
            #endif
        };

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
    #endif
    };
}

#endif // ifndef TOAST_LAPACK_HPP
