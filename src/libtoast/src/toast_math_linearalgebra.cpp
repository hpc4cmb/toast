
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_linearalgebra.hpp>

// Define macros for lapack name mangling

#if defined LAPACK_NAMES_LOWER
# define LAPACK_FUNC(lname, uname) lname
#elif defined LAPACK_NAMES_UPPER
# define LAPACK_FUNC(lname, uname) uname
#elif defined LAPACK_NAMES_UBACK
# define LAPACK_FUNC(lname, uname) lname ## _
#elif defined LAPACK_NAMES_UFRONT
# define LAPACK_FUNC(lname, uname) _ ## lname
#else // if defined LAPACK_NAMES_LOWER
# define LAPACK_FUNC(lname, uname) lname
#endif // if defined LAPACK_NAMES_LOWER

toast::LinearAlgebra::LinearAlgebra()
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
}

toast::LinearAlgebra::~LinearAlgebra()
{
#ifdef HAVE_CUDALIBS
    // free cublas handle
    cublasDestroy(handleBlas);
    // free cusolver handle
    cusolverDnDestroy(handleSolver);
    // release integer allocation
    cudaFree(gpu_allocated_integer);
#endif
}

#define dgemm LAPACK_FUNC(dgemm, DGEMM)

extern "C" void dgemm(char * TRANSA, char * TRANSB, int * M, int * N, int * K,
                      double * ALPHA, double * A, int * LDA, double * B,
                      int * LDB, double * BETA, double * C, int * LDC);

void toast::LinearAlgebra::gemm(char * TRANSA, char * TRANSB, int * M, int * N,
                        int * K, double * ALPHA, double * A, int * LDA,
                        double * B, int * LDB, double * BETA, double * C,
                        int * LDC) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasOperation_t transA_cuda = (*TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_cuda = (*TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // compute gemm
    cublasStatus_t errorCodeOp = cublasDgemm(handleBlas, transA_cuda, transB_cuda, *M, *N, *K, ALPHA, A, *LDA, B, *LDB, BETA, C, *LDC);
    checkCublasErrorCode(errorCodeOp);
    #elif HAVE_LAPACK
    dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dsyev LAPACK_FUNC(dsyev, DSYEV)

extern "C" void dsyev(char * JOBZ, char * UPLO, int * N, double * A, int * LDA,
                      double * W, double * WORK, int * LWORK, int * INFO);

// computes LWORK, the size (in number of elements) of WORK, the workspace used during the computation of syev
int toast::LinearAlgebra::syev_buffersize(char * JOBZ, char * UPLO, int * N, double * A,
                                  int * LDA, double * W) const {
    // We assume a large value here, since the work space needed will still be small.
    int NB = 256;
    int LWORK = NB * 2 + (*N);

    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (*JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (*UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // computes buffersize
    cusolverStatus_t statusBuffer = cusolverDnDsyevd_bufferSize(handleSolver, jobz_cuda, uplo_cuda, *N, A, *LDA, W, &LWORK);
    checkCusolverErrorCode(statusBuffer);
    #endif

    return LWORK;
}

// LWORK, the size of WORK in number of elements, should have been computed with lapack_syev_buffersize
void toast::LinearAlgebra::syev(char * JOBZ, char * UPLO, int * N, double * A,
                        int * LDA, double * W, double * WORK, int * LWORK,
                        int * INFO) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (*JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (*UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int* INFO_cuda = gpu_allocated_integer;
    // compute spectrum
    cusolverStatus_t statusSolver = cusolverDnDsyevd(handleSolver, jobz_cuda, uplo_cuda, *N, A, *LDA, W, WORK, *LWORK, INFO_cuda);
    checkCusolverErrorCode(statusSolver);
    // gets info back to CPU
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    *INFO = *INFO_cuda;
    #elif HAVE_LAPACK
    dsyev(JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dsymm LAPACK_FUNC(dsymm, DSYMM)

extern "C" void dsymm(char * SIDE, char * UPLO, int * M, int * N,
                      double * ALPHA, double * A, int * LDA, double * B,
                      int * LDB, double * BETA, double * C, int * LDC);

void toast::LinearAlgebra::symm(char * SIDE, char * UPLO, int * M, int * N,
                        double * ALPHA, double * A, int * LDA, double * B,
                        int * LDB, double * BETA, double * C, int * LDC) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasSideMode_t side_cuda = (*SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo_cuda = (*UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // compute gemm
    cublasStatus_t errorCodeOp = cublasDsymm(handleBlas, side_cuda, uplo_cuda, *M, *N, ALPHA, A, *LDA, B, *LDB, BETA, C, *LDC);
    checkCublasErrorCode(errorCodeOp);
    #elif HAVE_LAPACK
    dsymm(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dsyrk LAPACK_FUNC(dsyrk, DSYRK)

extern "C" void dsyrk(char * UPLO, char * TRANS, int * N, int * K,
                      double * ALPHA, double * A, int * LDA, double * BETA,
                      double * C, int * LDC);

void toast::LinearAlgebra::syrk(char * UPLO, char * TRANS, int * N, int * K,
                        double * ALPHA, double * A, int * LDA, double * BETA,
                        double * C, int * LDC) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasFillMode_t uplo_cuda = (*UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans_cuda = (*TRANS == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // compute gemm
    cublasStatus_t errorCodeOp = cublasDsyrk(handleBlas, uplo_cuda, trans_cuda, *N, *K, ALPHA, A, *LDA, BETA, C, *LDC);
    checkCublasErrorCode(errorCodeOp);
    #elif HAVE_LAPACK
    dsyrk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dgelss LAPACK_FUNC(dgelss, DGELSS)

extern "C" void dgelss(int * M, int * N, int * NRHS, double * A, int * LDA,
                       double * B, int * LDB, double * S, double * RCOND,
                       int * RANK, double * WORK, int * LWORK, int * INFO);

void toast::LinearAlgebra::dgelss(int * M, int * N, int * NRHS, double * A, int * LDA,
                          double * B, int * LDB, double * S, double * RCOND,
                          int * RANK, double * WORK, int * LWORK, int * INFO) const {
    // NOTE: there is no GPU dgelss implementation at the moment (there are dgels implementations however)
    #ifdef HAVE_LAPACK
    dgelss(M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}
