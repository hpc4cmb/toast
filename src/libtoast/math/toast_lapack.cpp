/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>


#define dgemm F77_FUNC( dgemm )

extern "C" void dgemm ( char * TRANSA, char * TRANSB, int * M, int * N, int * K, double * ALPHA, double * A, int * LDA, double * B, int * LDB, double * BETA, double * C, int * LDC );

void toast::lapack::gemm ( char * TRANSA, char * TRANSB, int * M, int * N, int * K, double * ALPHA, double * A, int * LDA, double * B, int * LDB, double * BETA, double * C, int * LDC ) {
  dgemm ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC );
  return;
}


#define dgemv F77_FUNC( dgemv )

extern "C" void dgemv ( char * TRANS, int * M, int * N, double * ALPHA, double * A, int * LDA, double * X, int * INCX, double * BETA, double * Y, int * INCY );

void toast::lapack::gemv ( char * TRANS, int * M, int * N, double * ALPHA, double * A, int * LDA, double * X, int * INCX, double * BETA, double * Y, int * INCY ) {
  dgemv ( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY );
  return;
}


#define dsyev F77_FUNC( dsyev )

extern "C" void dsyev ( char * JOBZ, char * UPLO, int * N, double * A, int * LDA, double * W, double * WORK, int * LWORK, int * INFO );

void toast::lapack::syev ( char * JOBZ, char * UPLO, int * N, double * A, int * LDA, double * W, double * WORK, int * LWORK, int * INFO ) {
  dsyev ( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO );
  return;
}


#define dsymv F77_FUNC( dsymv )

extern "C" void dsymv ( char * UPLO, int * N, double * ALPHA, double * A, int * LDA, double * X, int * INCX, double * BETA, double * Y, int * INCY );

void toast::lapack::symv ( char * UPLO, int * N, double * ALPHA, double * A, int * LDA, double * X, int * INCX, double * BETA, double * Y, int * INCY ) {
  dsymv ( UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY );
  return;
}


#define dpotrf F77_FUNC( dpotrf )

extern "C" void dpotrf ( char * UPLO, int * N, double * A, int * LDA, int * INFO );

void toast::lapack::potrf ( char * UPLO, int * N, double * A, int * LDA, int * INFO ) {
  dpotrf ( UPLO, N, A, LDA, INFO );
  return;
}


#define dpocon F77_FUNC( dpocon )

extern "C" void dpocon ( char * UPLO, int * N, double * A, int * LDA, double * ANORM, double * RCOND, double * WORK, int * IWORK, int * INFO );

void toast::lapack::pocon ( char * UPLO, int * N, double * A, int * LDA, double * ANORM, double * RCOND, double * WORK, int * IWORK, int * INFO ) {
  dpocon ( UPLO, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO );
  return;
}


#define dpotri F77_FUNC( dpotri )

extern "C" void dpotri ( char * UPLOW, int * N, double * A, int * LDA, int * INFO );

void toast::lapack::potri ( char * UPLO, int * N, double * A, int * LDA, int * INFO ) {
  dpotri ( UPLO, N, A, LDA, INFO );
  return;
}





