
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_lapack.hpp>

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
    #ifdef HAVE_LAPACK
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
    #ifdef HAVE_LAPACK
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
    #ifdef HAVE_LAPACK
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
    #ifdef HAVE_LAPACK
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
