
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

#define wrapped_dgemm LAPACK_FUNC(dgemm, DGEMM)

extern "C" void wrapped_dgemm(char * TRANSA, char * TRANSB, int * M, int * N, int * K,
                              double * ALPHA, double * A, int * LDA, double * B,
                              int * LDB, double * BETA, double * C, int * LDC);

void toast::LinearAlgebra::gemm(char TRANSA, char TRANSB, int M, int N,
                                int K, double ALPHA, double * A, int LDA,
                                double * B, int LDB, double BETA, double * C,
                                int LDC) {
#ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasOperation_t transA_gpu = (TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_gpu = (TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, LDA * ((TRANSA == 'N') ? K : M));
    double* B_gpu = GPU_memory_pool.toDevice(B, LDB * ((TRANSB == 'N') ? N : K));
    double* C_gpu = (BETA == 0.) ? GPU_memory_pool.alloc<double>(LDC * N)
                                 : GPU_memory_pool.toDevice(C, LDC * N);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDgemm(GPU_memory_pool.handleBlas, transA_gpu, transB_gpu, M, N, K, &ALPHA, A_gpu, LDA, B_gpu, LDB, &BETA, C_gpu, LDC);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets result back from GPU
    // and frees input memory
    GPU_memory_pool.free(A);
    GPU_memory_pool.free(B);
    GPU_memory_pool.fromDevice(C, C_gpu, LDC * N);
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

// matrices are expected to be in continuous memory in A_batched (one every N*LDA elements), B_batch and C_batch
void toast::LinearAlgebra::gemm_batched(char TRANSA, char TRANSB, int M, int N, int K,
                                        double ALPHA, double * A_batch, int LDA, double * B_batch, int LDB,
                                        double BETA, double * C_batch, int LDC, const int batchCount) {
    // size of the various matrices
    size_t A_size = LDA * ((TRANSA == 'N') ? K : M);
    size_t B_size = LDB * ((TRANSB == 'N') ? N : K);
    size_t C_size = LDC * N;
#ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasOperation_t transA_gpu = (TRANSA == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_gpu = (TRANSB == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // send data to GPU
    double* A_batch_gpu = GPU_memory_pool.toDevice(A_batch, batchCount * A_size);
    double* B_batch_gpu = GPU_memory_pool.toDevice(B_batch, batchCount * B_size);
    double* C_batch_gpu = (BETA == 0.) ? GPU_memory_pool.alloc<double>(batchCount * C_size)
                                       : GPU_memory_pool.toDevice(C_batch, batchCount * C_size);
    // gets pointers to each batch matrix
    toast::AlignedVector <double*> A_ptrs(batchCount);
    toast::AlignedVector <double*> B_ptrs(batchCount);
    toast::AlignedVector <double*> C_ptrs(batchCount);
    #pragma omp parallel for schedule(static)
    for(int64_t batchid = 0; batchid < batchCount; batchid++)
    {
        // using GPU adresses
        A_ptrs[batchid] = A_batch_gpu + batchid * A_size;
        B_ptrs[batchid] = B_batch_gpu + batchid * B_size;
        C_ptrs[batchid] = C_batch_gpu + batchid * C_size;
    }
    double ** A_ptrs_gpu = GPU_memory_pool.toDevice(A_ptrs.data(), batchCount);
    double ** B_ptrs_gpu = GPU_memory_pool.toDevice(B_ptrs.data(), batchCount);
    double ** C_ptrs_gpu = GPU_memory_pool.toDevice(C_ptrs.data(), batchCount);
    // compute batched blas operation
    cublasStatus_t errorCodeOp = cublasDgemmBatched(GPU_memory_pool.handleBlas, transA_gpu, transB_gpu, M, N, K, &ALPHA, A_ptrs_gpu, LDA, B_ptrs_gpu, LDB, &BETA, C_ptrs_gpu, LDC, batchCount);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // deletes pointers to each batch matrices
    // no need for a copy back to CPU
    GPU_memory_pool.free(A_ptrs_gpu);
    GPU_memory_pool.free(B_ptrs_gpu);
    GPU_memory_pool.free(C_ptrs_gpu);
    // gets result back from GPU
    // and frees niput n=memory
    GPU_memory_pool.fromDevice(C_batch, C_batch_gpu, batchCount * C_size);
    GPU_memory_pool.free(A_batch_gpu);
    GPU_memory_pool.free(B_batch_gpu);
#elif HAVE_LAPACK
    // use naive opemMP paralellism
    #pragma omp parallel for schedule(static)
    for(unsigned int b=0; b<batchCount; b++)
    {
        double * A = &A_batch[b * A_size];
        double * B = &B_batch[b * B_size];
        double * C = &C_batch[b * C_size];
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

// matrices are expected to be in continuous memory in A_batched (one every N*LDA elements)
// LWORK, the size of WORK in number of elements, should have been computed with lapack_syev_buffersize
void toast::LinearAlgebra::syev_batched(char JOBZ, char UPLO, int N, double * A_batch,
                                        int LDA, double * W_batch, int * INFO, const int batchCount) {
#ifdef HAVE_CUDALIBS
    // prepare inputs
    cusolverEigMode_t jobz_gpu = (JOBZ == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    int* INFO_gpu = GPU_memory_pool.alloc<int>(1);
    // send data to GPU
    double* A_batch_gpu = GPU_memory_pool.toDevice(A_batch, batchCount * N * LDA);
    double* W_batch_gpu = GPU_memory_pool.alloc<double>(batchCount * N); // output, no need to send
    // computes workspace size
    int LWORK = 0;
    cusolverStatus_t statusBuffer = cusolverDnDsyevjBatched_bufferSize(GPU_memory_pool.handleSolver, jobz_gpu, uplo_gpu, N, A_batch_gpu, LDA, W_batch_gpu, &LWORK, GPU_memory_pool.jacobiParameters, batchCount);
    checkCusolverErrorCode(statusBuffer);
    // allocates workspace
    double* WORK_gpu = GPU_memory_pool.alloc<double>(LWORK);
    // compute cusolver operation
    cusolverStatus_t statusSolver = cusolverDnDsyevjBatched(GPU_memory_pool.handleSolver, jobz_gpu, uplo_gpu, N, A_batch_gpu, LDA, W_batch_gpu, WORK_gpu, LWORK, INFO_gpu, GPU_memory_pool.jacobiParameters, batchCount);
    checkCusolverErrorCode(statusSolver);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets info back to CPU
    GPU_memory_pool.fromDevice(W_batch, W_batch_gpu, batchCount * N);
    GPU_memory_pool.free(WORK_gpu);
    GPU_memory_pool.fromDevice(INFO, INFO_gpu, 1);
    // copies only if the eigenvectors have been stored in A
    if(JOBZ == 'V') {
        GPU_memory_pool.fromDevice(A_batch, A_batch_gpu, batchCount * N * LDA);
    }
    else {
        GPU_memory_pool.free(A_batch_gpu);
    }
#elif HAVE_LAPACK
    // workspace size
    // We assume a large value here, since the work space needed will still be small.
    int NB = 256;
    int LWORK = NB * 2 + N;
    #pragma omp parallel
    {
        // threadlocal
        toast::AlignedVector <double> WORK(LWORK);
        // use naive opemMP paralellism
        #pragma omp for schedule(static)
        for(unsigned int b=0; b<batchCount; b++)
        {
            // gets batch element
            double * A = &A_batch[b * N * LDA];
            double * W = &W_batch[b * N];
            // runs syev
            int INFOb = 0;
            wrapped_dsyev(&JOBZ, &UPLO, &N, A, &LDA, W, WORK.data(), &LWORK, &INFOb);
            // checks output status
            // race condition is okay, in the end we just want to know if it is 0
            if (INFOb != 0) *INFO = INFOb;
        }
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
                                int LDB, double BETA, double * C, int LDC) {
    #ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasSideMode_t side_gpu = (SIDE == 'L') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, LDA * ( (SIDE == 'L') ? M : N ));
    double* B_gpu = GPU_memory_pool.toDevice(B, LDB * N);
    double* C_gpu = (BETA == 0.) ? GPU_memory_pool.alloc<double>(LDC * N)
                                 : GPU_memory_pool.toDevice(C, LDC * N);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDsymm(GPU_memory_pool.handleBlas, side_gpu, uplo_gpu, M, N, &ALPHA, A_gpu, LDA, B_gpu, LDB, &BETA, C_gpu, LDC);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets data back from GPU
    // frees input memory that does not need to go back
    GPU_memory_pool.free(A);
    GPU_memory_pool.free(B);
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

void toast::LinearAlgebra::symm_batched(char SIDE, char UPLO, int M, int N, double ALPHA,
                                        double * A_batch, int LDA, double * B_batch, int LDB, double BETA,
                                        double * C_batch, int LDC, int batchCount) {
#ifdef HAVE_CUDALIBS
    char TRANSA = 'N';
    char TRANSB = 'N';
    int K = (SIDE == 'R') ? N : M;
    // fills A to make it truly symmetrical
    #pragma omp parallel for schedule(static)
    for(unsigned int b=0; b<batchCount; b++)
    {
        double * A = &A_batch[b * LDA * K];
        for(int r = 0; r < K; r++)
        {
            for(int c = 0; c < r; c++)
            {
                if(UPLO == 'U')
                {
                    // fill lower part of A
                    A[r + LDA*c] = A[c + LDA*r];
                }
                else
                {
                    // fill upper part of A
                    A[c + LDA*r] = A[r + LDA*c];
                }
            }
        }
    }
    // take side into account
    if(SIDE == 'R')
    {
        std::swap(A_batch, B_batch);
        std::swap(LDA, LDB);
    }
    // the GPU version calls gemm as it can be batched on GPU and not a true syrk
    toast::LinearAlgebra::gemm_batched(TRANSA, TRANSB, M, N, K, ALPHA, A_batch, LDA, B_batch, LDB, BETA, C_batch, LDC, batchCount);
#elif HAVE_LAPACK
    // use naive opemMP paralellism
    #pragma omp parallel for schedule(static)
    for(unsigned int b=0; b<batchCount; b++)
    {
        double * A = &A_batch[b * LDA * ((SIDE == 'L') ? M : N)];
        double * B = &B_batch[b * LDB * N];
        double * C = &C_batch[b * LDC * N];
        wrapped_dsymm(&SIDE, &UPLO, &M, &N, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    }
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
                                double * C, int LDC) {
#ifdef HAVE_CUDALIBS
    // prepare inputs
    cublasFillMode_t uplo_gpu = (UPLO == 'L') ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans_gpu = (TRANS == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
    // send data to GPU
    double* A_gpu = GPU_memory_pool.toDevice(A, LDA * ( (TRANS == 'T') ? N : K ));
    double* C_gpu = (BETA == 0.) ? GPU_memory_pool.alloc<double>(LDC * N)
                                 : GPU_memory_pool.toDevice(C, LDC * N);
    // compute blas operation
    cublasStatus_t errorCodeOp = cublasDsyrk(GPU_memory_pool.handleBlas, uplo_gpu, trans_gpu, N, K, &ALPHA, A_gpu, LDA, &BETA, C_gpu, LDC);
    checkCublasErrorCode(errorCodeOp);
    cudaError statusSync = cudaDeviceSynchronize();
    checkCudaErrorCode(statusSync);
    // gets data back from GPU
    // frees input memory
    GPU_memory_pool.free(A);
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

// the batch element are supposed continuous in memory
void toast::LinearAlgebra::syrk_batched(char UPLO, char TRANS, int N, int K, double ALPHA,
                                        double * A_batched, int LDA, double BETA, double * C_batched, int LDC,
                                        int batchCount) {
#ifdef HAVE_CUDALIBS
    // the GPU version calls gemm as it can be batched on GPU and not a true syrk
    char TRANSA = (TRANS == 'N') ? 'N' : 'T';
    char TRANSB = (TRANS == 'N') ? 'T' : 'N';
    toast::LinearAlgebra::gemm_batched(TRANSA, TRANSB, N, N, K, ALPHA, A_batched, LDA, A_batched, LDA, BETA, C_batched, LDC, batchCount);
#elif HAVE_LAPACK
    // use naive opemMP paralellism
    #pragma omp parallel for schedule(static)
    for(unsigned int b=0; b<batchCount; b++)
    {
        double * A = &A_batched[b * LDA * ((TRANS == 'T') ? N : K)];
        double * C = &C_batched[b * LDC * N];
        wrapped_dsyrk(&UPLO, &TRANS, &N, &K, &ALPHA, A, &LDA, &BETA, C, &LDC);
    }
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

// NOTE: there is no GPU dgelss implementation at the moment (there are dgels implementations however)
void toast::LinearAlgebra::gelss(int M, int N, int NRHS, double * A, int LDA,
                                 double * B, int LDB, double * S, double RCOND,
                                 int RANK, double * WORK, int LWORK, int * INFO) {
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
