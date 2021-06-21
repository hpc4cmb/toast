
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_lapack.hpp>

#ifdef HAVE_CUDALIBS
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

// TODO error out if status is not CUBLAS_STATUS_SUCCESS
//  https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t
//  a helper function would be nice
// TODO handle is costly to create and delete, put it inside singleton
// TODO batch operations would be much faster
// TODO are the side, uplo, trans encoded in capital (L) or not (l)?

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


#define dgemm LAPACK_FUNC(dgemm, DGEMM)

extern "C" void dgemm(char * TRANSA, char * TRANSB, int * M, int * N, int * K,
                      double * ALPHA, double * A, int * LDA, double * B,
                      int * LDB, double * BETA, double * C, int * LDC);

void toast::lapack_gemm(char * TRANSA, char * TRANSB, int * M, int * N,
                        int * K, double * ALPHA, double * A, int * LDA,
                        double * B, int * LDB, double * BETA, double * C,
                        int * LDC) {
    #ifdef HAVE_CUDALIBS
    // make cublas handle
    cublasHandle_t handle;
    cublasStatus_t errorCodeHandle = cublasCreate(&handle);
    // prepare inputs
    cublasOperation_t transA_cuda = (*TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_cuda = (*TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // compute gemm
    cublasStatus_t errorCodeOp = cublasDgemm(handle, transA_cuda, transB_cuda, *M, *N, *K, ALPHA, A, *LDA, B, *LDB, BETA, C, *LDC);
    // free handle
    cublasDestroy(handle);
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

#define dgemv LAPACK_FUNC(dgemv, DGEMV)

extern "C" void dgemv(char * TRANS, int * M, int * N, double * ALPHA,
                      double * A, int * LDA, double * X, int * INCX,
                      double * BETA, double * Y, int * INCY);

void toast::lapack_gemv(char * TRANS, int * M, int * N, double * ALPHA,
                        double * A, int * LDA, double * X, int * INCX,
                        double * BETA, double * Y, int * INCY) {
    #ifdef HAVE_LAPACK
    dgemv(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
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

void toast::lapack_syev(char * JOBZ, char * UPLO, int * N, double * A,
                        int * LDA, double * W, double * WORK, int * LWORK,
                        int * INFO) {
    #ifdef HAVE_CUDALIBS
    // make cusolver handle
    cusolverDnHandle_t handle;
    cusolverStatus_t statusHandle = cusolverDnCreate(&handle);
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (*JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (*UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // query working space
    cusolverStatus_t statusBuffer = cusolverDnDsyevd_bufferSize(handle, jobz_cuda, uplo_cuda, *N, A, *LDA, W, LWORK);
    // compute spectrum
    cusolverStatus_t statusSolver = cusolverDnDsyevd(handle, jobz_cuda, uplo_cuda, *N, A, *LDA, W, WORK, *LWORK, INFO);
    // free handle
    cusolverDnDestroy(handle);
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

#define dsymv LAPACK_FUNC(dsymv, DSYMV)

extern "C" void dsymv(char * UPLO, int * N, double * ALPHA, double * A,
                      int * LDA, double * X, int * INCX, double * BETA,
                      double * Y, int * INCY);

void toast::lapack_symv(char * UPLO, int * N, double * ALPHA, double * A,
                        int * LDA, double * X, int * INCX, double * BETA,
                        double * Y, int * INCY) {
    #ifdef HAVE_LAPACK
    dsymv(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dtrmv LAPACK_FUNC(dtrmv, DTRMV)

extern "C" void dtrmv(char * UPLO, char * TRANS, char * DIAG, int * N,
                      double * A, int * LDA, double * X, int * INCX);

void toast::lapack_trmv(char * UPLO, char * TRANS, char * DIAG, int * N,
                        double * A, int * LDA, double * X, int * INCX) {
    #ifdef HAVE_LAPACK
    dtrmv(UPLO, TRANS, DIAG, N, A, LDA, X, INCX);
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

void toast::lapack_symm(char * SIDE, char * UPLO, int * M, int * N,
                        double * ALPHA, double * A, int * LDA, double * B,
                        int * LDB, double * BETA, double * C, int * LDC) {
    #ifdef HAVE_CUDALIBS
    // make cublas handle
    cublasHandle_t handle;
    cublasStatus_t errorCodeHandle = cublasCreate(&handle);
    // prepare inputs
    cublasSideMode_t side_cuda = (*SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo_cuda = (*UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // compute gemm
    cublasStatus_t errorCodeOp = cublasDsymm(handle, side_cuda, uplo_cuda, *M, *N, ALPHA, A, *LDA, B, *LDB, BETA, C, *LDC);
    // free handle
    cublasDestroy(handle);
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

void toast::lapack_syrk(char * UPLO, char * TRANS, int * N, int * K,
                        double * ALPHA, double * A, int * LDA, double * BETA,
                        double * C, int * LDC) {
    #ifdef HAVE_CUDALIBS
    // TODO
    dsyrk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC);
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

#define dpotrf LAPACK_FUNC(dpotrf, DPOTRF)

extern "C" void dpotrf(char * UPLO, int * N, double * A, int * LDA,
                       int * INFO);

void toast::lapack_potrf(char * UPLO, int * N, double * A, int * LDA,
                         int * INFO) {
    #ifdef HAVE_LAPACK
    dpotrf(UPLO, N, A, LDA, INFO);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dpocon LAPACK_FUNC(dpocon, DPOCON)

extern "C" void dpocon(char * UPLO, int * N, double * A, int * LDA,
                       double * ANORM, double * RCOND, double * WORK,
                       int * IWORK, int * INFO);

void toast::lapack_pocon(char * UPLO, int * N, double * A, int * LDA,
                         double * ANORM, double * RCOND, double * WORK,
                         int * IWORK, int * INFO) {
    #ifdef HAVE_LAPACK
    dpocon(UPLO, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
    return;
}

#define dpotri LAPACK_FUNC(dpotri, DPOTRI)

extern "C" void dpotri(char * UPLOW, int * N, double * A, int * LDA,
                       int * INFO);

void toast::lapack_potri(char * UPLO, int * N, double * A, int * LDA,
                         int * INFO) {
    #ifdef HAVE_LAPACK
    dpotri(UPLO, N, A, LDA, INFO);
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

void toast::lapack_dgelss(int * M, int * N, int * NRHS, double * A, int * LDA,
                          double * B, int * LDB, double * S, double * RCOND,
                          int * RANK, double * WORK, int * LWORK, int * INFO) {
    #ifdef HAVE_CUDALIBS
    // TODO
    dgelss(M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO);
    #elif HAVE_LAPACK
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
