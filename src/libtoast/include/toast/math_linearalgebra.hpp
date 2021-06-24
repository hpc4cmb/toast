
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_LINEARALGEBRA_HPP
#define TOAST_MATH_LINEARALGEBRA_HPP

#ifdef HAVE_CUDALIBS
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>

void checkCudaErrorCode(const cudaError errorCode);
void checkCublasErrorCode(const cublasStatus_t errorCode);
void checkCusolverErrorCode(const cusolverStatus_t errorCode);
#endif

// TODO:
//  - put gpu helper function in dedicated file

namespace toast {
    // encapsulates construction and destruction of GPU linear algebra handles
    // WARNING: this class is *not* threadsafe
    class LinearAlgebra {
    public:
        LinearAlgebra()
        {
            #ifdef HAVE_CUDALIBS
            // creates cublas handle
            cublasStatus_t statusHandleBlas = cublasCreate(&handleBlas);
            checkCublasErrorCode(statusHandleBlas);
            // creates cusolver handle
            cusolverStatus_t statusHandleCusolver = cusolverDnCreate(&handleSolver);
            checkCusolverErrorCode(statusHandleCusolver);
            // allocates an integer on GPU to use it as an output parameter
            cudaError statusAlloc = cudaMallocManaged((void**)&gpu_allocated_integer, sizeof(int));
            checkCudaErrorCode(statusAlloc);
            #endif
        };

        // insures that the class is never copied
        LinearAlgebra(LinearAlgebra const&) = delete;
        void operator=(LinearAlgebra const&) = delete;

        ~LinearAlgebra()
        {
            #ifdef HAVE_CUDALIBS
            // free cublas handle
            cublasDestroy(handleBlas);
            // free cusolver handle
            cusolverDnDestroy(handleSolver);
            // release integer allocation
            cudaFree(gpu_allocated_integer);
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
        int* gpu_allocated_integer = NULL;
    #endif
    };
}

#endif // ifndef TOAST_LAPACK_HPP
