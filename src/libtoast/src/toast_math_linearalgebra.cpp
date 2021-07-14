
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
    cublasOperation_t transA_gpu = (TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_gpu = (TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, M * K);
    double* B_gpu = GPU_memory_pool.toDevice(B, K * N);
    double* C_gpu = GPU_memory_pool.toDevice(C, M * N);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDgemm(handleBlas, transA_gpu, transB_gpu, M, N, K, &ALPHA, A_gpu, LDA, B_gpu, LDB, &BETA, C_gpu, LDC);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets data back from GPU
    GPU_memory_pool.fromDevice(A, A_gpu, M * K);
    GPU_memory_pool.fromDevice(B, B_gpu, K * N);
    GPU_memory_pool.fromDevice(C, C_gpu, M * N);
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
    cublasOperation_t transA_gpu = (TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_gpu = (TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // TODO send data to GPU
    // compute batched blas operation
    cublasStatus_t errorCodeOp = cublasDgemmBatched(handleBlas, transA_gpu, transB_gpu, M, N, K, &ALPHA, A_batch, LDA, B_batch, LDB, &BETA, C_batch, LDC, batchCount);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // TODO get data back from GPU
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
    cusolverEigMode_t jobz_gpu = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // computes buffersize
    cusolverStatus_t statusBuffer = cusolverDnDsyevd_bufferSize(handleSolver, jobz_gpu, uplo_gpu, N, /*A=*/NULL, LDA, /*W=*/NULL, &LWORK);
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
    cusolverEigMode_t jobz_gpu = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int* INFO_gpu = gpu_allocated_integer;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, N * LDA);
    double* W_gpu = GPU_memory_pool.toDevice(W, N);
    // allocates workspace
    void* WORK_gpu = NULL;
    cudaError statusWorkAlloc = GPU_memory_pool.malloc(&WORK_gpu, LWORK * sizeof(double));
    checkCudaErrorCode(statusWorkAlloc, "syev (WORK malloc)");
    // compute cusolver operation
    cusolverStatus_t statusSolver = cusolverDnDsyevd(handleSolver, jobz_gpu, uplo_gpu, N, A_gpu, LDA, W_gpu, static_cast<double*>(WORK_gpu), LWORK, INFO_gpu);
    checkCusolverErrorCode(statusSolver);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets info back to CPU
    GPU_memory_pool.fromDevice(A, A_gpu, N * LDA);
    GPU_memory_pool.fromDevice(W, W_gpu, N);
    GPU_memory_pool.free(WORK_gpu);
    const cudaError errorCodeMemcpy = cudaMemcpy(INFO, INFO_gpu, sizeof(int), cudaMemcpyDeviceToHost); // *INFO = *INFO_gpu
    checkCudaErrorCode(errorCodeMemcpy, "syev (INFO memcpy)");
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
    cusolverEigMode_t jobz_gpu = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // computes buffersize
    cusolverStatus_t statusBuffer = cusolverDnDsyevjBatched_bufferSize(handleSolver, jobz_gpu, uplo_gpu, N, /*A_batch=*/NULL, LDA, /*W_batch=*/NULL, &LWORK, jacobiParameters, batchCount);
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
    cusolverEigMode_t jobz_gpu = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int* INFO_gpu = gpu_allocated_integer;
    // send data to GPU
    double* A_batch_gpu = GPU_memory_pool.toDevice(A_batch, batchCount * N * LDA);
    double* W_batch_gpu = GPU_memory_pool.toDevice(W_batch, batchCount * N);
    // allocates workspace
    void* WORK_gpu = NULL;
    cudaError statusWorkAlloc = GPU_memory_pool.malloc(&WORK_gpu, LWORK * sizeof(double));
    checkCudaErrorCode(statusWorkAlloc, "syev_batched (WORK malloc)");
    // compute cusolver operation
    cusolverStatus_t statusSolver = cusolverDnDsyevjBatched(handleSolver, jobz_gpu, uplo_gpu, N, A_batch_gpu, LDA, W_batch_gpu, static_cast<double*>(WORK_gpu), LWORK, INFO_gpu, jacobiParameters, batchCount);
    checkCusolverErrorCode(statusSolver);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets info back to CPU
    GPU_memory_pool.fromDevice(A_batch, A_batch_gpu, batchCount * N * LDA);
    GPU_memory_pool.fromDevice(W_batch, W_batch_gpu, batchCount * N);
    GPU_memory_pool.free(WORK_gpu);
    const cudaError errorCodeMemcpy = cudaMemcpy(INFO, INFO_gpu, sizeof(int), cudaMemcpyDeviceToHost); // *INFO = *INFO_gpu
    checkCudaErrorCode(errorCodeMemcpy, "syev_batched (INFO memcpy)");
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
    cublasSideMode_t side_gpu = (SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, LDA * ( (SIDE == 'L') ? M : N ));
    double* B_gpu = GPU_memory_pool.toDevice(B, LDB * N);
    double* C_gpu = GPU_memory_pool.toDevice(C, LDC * N);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDsymm(handleBlas, side_gpu, uplo_gpu, M, N, &ALPHA, A_gpu, LDA, B_gpu, LDB, &BETA, C_gpu, LDC);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets data back from GPU
    GPU_memory_pool.fromDevice(A, A_gpu, LDA * ( (SIDE == 'L') ? M : N ));
    GPU_memory_pool.fromDevice(B, B_gpu, LDB * N);
    GPU_memory_pool.fromDevice(C, C_gpu, LDC * N);
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
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans_gpu = (TRANS == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, LDA * ( (TRANS == 'T') ? N : K ));
    double* C_gpu = GPU_memory_pool.toDevice(C, LDC * N);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDsyrk(handleBlas, uplo_gpu, trans_gpu, N, K, &ALPHA, A_gpu, LDA, &BETA, C_gpu, LDC);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets data back from GPU
    GPU_memory_pool.fromDevice(A, A_gpu, LDA * ( (TRANS == 'T') ? N : K ));
    GPU_memory_pool.fromDevice(C, C_gpu, LDC * N);
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
