
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

// NOTE: one could use one handle per gpu if several are available
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
    cudaError statusAlloc = GPU_memory_pool.malloc((void**)&gpu_allocated_integer, sizeof(int));
    checkCudaErrorCode(statusAlloc);
    // gets jacobi parameters for batched syev
    cusolverStatus_t statusJacobiParams = cusolverDnCreateSyevjInfo(&jacobiParameters);
    checkCusolverErrorCode(statusJacobiParams);
    // gets id of the GPU device being used
    cudaGetDevice(&gpuId);
#endif
}

toast::LinearAlgebra::~LinearAlgebra()
{
#ifdef HAVE_CUDALIBS
    // free cublas handle
    cublasDestroy(handleBlas);
    // free cusolver handle
    cusolverDnDestroy(handleSolver);
    // destroys jacobi parameters for batched syev
    cusolverStatus_t statusJacobiParams = cusolverDnDestroySyevjInfo(jacobiParameters);
    checkCusolverErrorCode(statusJacobiParams);
    // release integer allocation
    GPU_memory_pool.free(gpu_allocated_integer);
#endif
}

#define wrapped_dgemm LAPACK_FUNC(dgemm, DGEMM)

extern "C" void wrapped_dgemm(char * TRANSA, char * TRANSB, int * M, int * N, int * K,
                              double * ALPHA, double * A, int * LDA, double * B,
                              int * LDB, double * BETA, double * C, int * LDC);

void toast::LinearAlgebra::gemm(char TRANSA, char TRANSB, int M, int N,
                                int K, double ALPHA, double * A, int LDA,
                                double * B, int LDB, double BETA, double * C,
                                int LDC) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasOperation_t transA_cuda = (TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_cuda = (TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // prefetch data to GPU (optional)
    cudaMemPrefetchAsync(A, M * K * sizeof(double), gpuId);
    cudaMemPrefetchAsync(B, K * N * sizeof(double), gpuId);
    cudaMemPrefetchAsync(C, M * N * sizeof(double), gpuId);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDgemm(handleBlas, transA_cuda, transB_cuda, M, N, K, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC);
    checkCublasErrorCode(errorCodeOp);
    #elif HAVE_LAPACK
    wrapped_dgemm(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
}

void toast::LinearAlgebra::gemm_batched(char TRANSA, char TRANSB, int M, int N, int K,
                                        double ALPHA, double * A_batch[], int LDA, double * B_batch[], int LDB,
                                        double BETA, double * C_batch[], int LDC, const int batchCount) const {
#ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasOperation_t transA_cuda = (TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_cuda = (TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // compute batched blas operation
    cublasStatus_t errorCodeOp = cublasDgemmBatched(handleBlas, transA_cuda, transB_cuda, M, N, K, &ALPHA, A_batch, LDA, B_batch, LDB, &BETA, C_batch, LDC, batchCount);
    checkCublasErrorCode(errorCodeOp);
#elif HAVE_LAPACK
    // use naive opemMP paralellism
    #pragma omp parallel for
    for(unsigned int b=0; b<batchCount; b++)
    {
        double * A = A_batch[b];
        double * B = B_batch[b];
        double * C = C_batch[b];
        wrapped_dgemm(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    }
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
#endif // ifdef HAVE_LAPACK
}

#define wrapped_dsyev LAPACK_FUNC(dsyev, DSYEV)

extern "C" void wrapped_dsyev(char * JOBZ, char * UPLO, int * N, double * A, int * LDA,
                              double * W, double * WORK, int * LWORK, int * INFO);

// computes LWORK, the size (in number of elements) of WORK, the workspace used during the computation of syev
int toast::LinearAlgebra::syev_buffersize(char JOBZ, char UPLO, int N, double * A,
                                          int LDA, double * W) const {
    // We assume a large value here, since the work space needed will still be small.
    int NB = 256;
    int LWORK = NB * 2 + N;

    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // computes buffersize
    cusolverStatus_t statusBuffer = cusolverDnDsyevd_bufferSize(handleSolver, jobz_cuda, uplo_cuda, N, A, LDA, W, &LWORK);
    checkCusolverErrorCode(statusBuffer);
    #endif

    return LWORK;
}

// LWORK, the size of WORK in number of elements, should have been computed with lapack_syev_buffersize
void toast::LinearAlgebra::syev(char JOBZ, char UPLO, int N, double * A,
                                int LDA, double * W, double * WORK, int LWORK,
                                int * INFO) {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int* INFO_cuda = gpu_allocated_integer;
    // prefetch data to GPU (optional)
    cudaMemPrefetchAsync(A, N * LDA * sizeof(double), gpuId);
    cudaMemPrefetchAsync(W, N * sizeof(double), gpuId);
    cudaMemPrefetchAsync(WORK, LWORK * sizeof(double), gpuId);
    // compute cusolver operation
    cusolverStatus_t statusSolver = cusolverDnDsyevd(handleSolver, jobz_cuda, uplo_cuda, N, A, LDA, W, WORK, LWORK, INFO_cuda);
    checkCusolverErrorCode(statusSolver);
    // gets info back to CPU
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    *INFO = *INFO_cuda;
    #elif HAVE_LAPACK
    wrapped_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, INFO);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
}

// we assume that buffers will be threadlocal
int toast::LinearAlgebra::syev_batched_buffersize(char JOBZ, char UPLO, int N, double * A_batch,
                                                  int LDA, double * W_batch, const int batchCount) const {
    // We assume a large value here, since the work space needed will still be small.
    int NB = 256;
    int LWORK = batchCount * (NB * 2 + N);

#ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // computes buffersize
    cusolverStatus_t statusBuffer = cusolverDnDsyevjBatched_bufferSize(handleSolver, jobz_cuda, uplo_cuda, N, A_batch, LDA, W_batch, &LWORK, jacobiParameters, batchCount);
    checkCusolverErrorCode(statusBuffer);
#endif

    return LWORK;
}

// matrices are expected to be in continuous memory in A_batched (one every N*LDA elements)
// LWORK, the size of WORK in number of elements, should have been computed with lapack_syev_buffersize
void toast::LinearAlgebra::syev_batched(char JOBZ, char UPLO, int N, double * A_batch,
                                        int LDA, double * W_batch, double * WORK, int LWORK,
                                        int * INFO, const int batchCount) {
#ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_cuda = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_cuda = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int* INFO_cuda = gpu_allocated_integer;
    // prefetch data to GPU (optional)
    cudaMemPrefetchAsync(A_batch, batchCount * N * LDA * sizeof(double), gpuId);
    cudaMemPrefetchAsync(W_batch, batchCount * N * sizeof(double), gpuId);
    cudaMemPrefetchAsync(WORK, LWORK * sizeof(double), gpuId);
    // compute cusolver operation
    cusolverStatus_t statusSolver = cusolverDnDsyevjBatched(handleSolver, jobz_cuda, uplo_cuda, N, A_batch, LDA, W_batch, WORK, LWORK, INFO_cuda, jacobiParameters, batchCount);
    checkCusolverErrorCode(statusSolver);
    // gets info back to CPU
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    *INFO = *INFO_cuda;
#elif HAVE_LAPACK
    // use naive opemMP paralellism
    #pragma omp parallel for
    for(unsigned int b=0; b<batchCount; b++)
    {
        double * A = &A_batch[b * N * LDA];
        double * W = &W_batch[b * N * LDA];
        int LWORKb = LWORK/batchCount;
        double * WORKb = &WORK[b*LWORKb];
        int INFOb = 0;
        wrapped_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, WORKb, &LWORKb, &INFOb);
        if (INFOb != 0) *INFO = INFOb;
    }
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
#endif // ifdef HAVE_LAPACK
}

#define wrapped_dsymm LAPACK_FUNC(dsymm, DSYMM)

extern "C" void wrapped_dsymm(char * SIDE, char * UPLO, int * M, int * N,
                              double * ALPHA, double * A, int * LDA, double * B,
                              int * LDB, double * BETA, double * C, int * LDC);

void toast::LinearAlgebra::symm(char SIDE, char UPLO, int M, int N,
                                double ALPHA, double * A, int LDA, double * B,
                                int LDB, double BETA, double * C, int LDC) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasSideMode_t side_cuda = (SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo_cuda = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // prefetch data to GPU (optional)
    cudaMemPrefetchAsync(A, LDA * ( (SIDE == 'L') ? M : N ) * sizeof(double), gpuId);
    cudaMemPrefetchAsync(B, LDB * N * sizeof(double), gpuId);
    cudaMemPrefetchAsync(C, LDC * N * sizeof(double), gpuId);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDsymm(handleBlas, side_cuda, uplo_cuda, M, N, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC);
    checkCublasErrorCode(errorCodeOp);
    #elif HAVE_LAPACK
    wrapped_dsymm(&SIDE, &UPLO, &M, &N, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
}

#define wrapped_dsyrk LAPACK_FUNC(dsyrk, DSYRK)

extern "C" void wrapped_dsyrk(char * UPLO, char * TRANS, int * N, int * K,
                              double * ALPHA, double * A, int * LDA, double * BETA,
                              double * C, int * LDC);

void toast::LinearAlgebra::syrk(char UPLO, char TRANS, int N, int K,
                                double ALPHA, double * A, int LDA, double BETA,
                                double * C, int LDC) const {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasFillMode_t uplo_cuda = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans_cuda = (TRANS == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // prefetch data to GPU (optional)
    cudaMemPrefetchAsync(A, LDA * ( (TRANS == 'T') ? N : K ) * sizeof(double), gpuId);
    cudaMemPrefetchAsync(C, LDC * N * sizeof(double), gpuId);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDsyrk(handleBlas, uplo_cuda, trans_cuda, N, K, &ALPHA, A, LDA, &BETA, C, LDC);
    checkCublasErrorCode(errorCodeOp);
    #elif HAVE_LAPACK
    wrapped_dsyrk(&UPLO, &TRANS, &N, &K, &ALPHA, A, &LDA, &BETA, C, &LDC);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
}

#define wrapped_dgelss LAPACK_FUNC(dgelss, DGELSS)

extern "C" void wrapped_dgelss(int * M, int * N, int * NRHS, double * A, int * LDA,
                               double * B, int * LDB, double * S, double * RCOND,
                               int * RANK, double * WORK, int * LWORK, int * INFO);

void toast::LinearAlgebra::gelss(int M, int N, int NRHS, double * A, int LDA,
                                 double * B, int LDB, double * S, double RCOND,
                                 int RANK, double * WORK, int LWORK, int * INFO) const {
    // NOTE: there is no GPU dgelss implementation at the moment (there are dgels implementations however)
    #ifdef HAVE_LAPACK
    wrapped_dgelss(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, WORK, &LWORK, INFO);
    #else // ifdef HAVE_LAPACK
    auto here = TOAST_HERE();
    auto log = toast::Logger::get();
    std::string msg("TOAST was not compiled with BLAS/LAPACK support.");
    log.error(msg.c_str(), here);
    throw std::runtime_error(msg.c_str());
    #endif // ifdef HAVE_LAPACK
}
