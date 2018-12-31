
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_LAPACK_HPP
#define TOAST_MATH_LAPACK_HPP


namespace toast {

void lapack_gemm(char * TRANSA, char * TRANSB, int * M, int * N, int * K,
                 double * ALPHA, double * A, int * LDA, double * B, int * LDB,
                 double * BETA, double * C, int * LDC);

void lapack_gemv(char * TRANS, int * M, int * N, double * ALPHA, double * A,
                 int * LDA, double * X, int * INCX, double * BETA, double * Y,
                 int * INCY);

void lapack_syev(char * JOBZ, char * UPLO, int * N, double * A, int * LDA,
                 double * W, double * WORK, int * LWORK, int * INFO);

void lapack_symv(char * UPLO, int * N, double * ALPHA, double * A, int * LDA,
                 double * X, int * INCX, double * BETA, double * Y,
                 int * INCY);

void lapack_trmv(char * UPLO, char * TRANS, char * DIAG, int * N, double * A,
                 int * LDA, double * X, int * INCX);

void lapack_symm(char * SIDE, char * UPLO, int * M, int * N, double * ALPHA,
                 double * A, int * LDA, double * B, int * LDB, double * BETA,
                 double * C, int * LDC);

void lapack_syrk(char * UPLO, char * TRANS, int * N, int * K, double * ALPHA,
                 double * A, int * LDA, double * BETA, double * C, int * LDC);

void lapack_potrf(char * UPLO, int * N, double * A, int * LDA, int * INFO);

void lapack_pocon(char * UPLO, int * N, double * A, int * LDA, double * ANORM,
                  double * RCOND, double * WORK, int * IWORK, int * INFO);

void lapack_potri(char * UPLO, int * N, double * A, int * LDA, int * INFO);

}

#endif // ifndef TOAST_LAPACK_HPP
