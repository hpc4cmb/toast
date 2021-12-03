
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/gpu_helpers.hpp>
#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>

#include <cstring>

#ifdef HAVE_CUDALIBS

// ------------------------------------------------------------------------------------
// ERROR CODE CHECKING

// displays an error message if the cuda runtime computation did not end in success
void checkCudaErrorCode(const cudaError errorCode, const std::string &functionName)
{
    if (errorCode != cudaSuccess)
    {
        auto log = toast::Logger::get();
        std::string msg = "The CUDA Runtime threw a '" +
                          std::string(cudaGetErrorString(errorCode)) + "' error code";
        if (functionName != "unknown")
        {
            msg += " in function '" + functionName + "'.";
        }
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

// turns a cublas error code into human readable text
std::string cublasGetErrorString(const cublasStatus_t errorCode)
{
    switch (errorCode)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "unknown";
}

// displays an error message if the cublas computation did not end in sucess
void checkCublasErrorCode(const cublasStatus_t errorCode,
                          const std::string &functionName)
{
    if (errorCode != CUBLAS_STATUS_SUCCESS)
    {
        auto log = toast::Logger::get();
        std::string msg = "CUBLAS threw a '" + cublasGetErrorString(errorCode) +
                          "' error code";
        if (functionName != "unknown")
        {
            msg += " in function '" + functionName + "'.";
        }
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

// turns a cusolver error code into human readable text
std::string cusolverGetErrorString(
    const cusolverStatus_t errorCode)
{
    switch (errorCode)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";

    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";

    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";

    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";

    case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";

    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";

    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED";

    case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";

    case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";

    case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
        return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";

    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
        return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";

    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
        return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";

    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
        return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";

    case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";

    case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
        return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";

    case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
        return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";

    case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
        return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";

    case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
        return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";

    case CUSOLVER_STATUS_INVALID_WORKSPACE:
        return "CUSOLVER_STATUS_INVALID_WORKSPACE";
    }

    return "unknown";
}

// displays an error message if the cusolver computation did not end in sucess
void checkCusolverErrorCode(const cusolverStatus_t errorCode,
                            const std::string &functionName)
{
    if (errorCode != CUSOLVER_STATUS_SUCCESS)
    {
        auto log = toast::Logger::get();
        std::string msg = "CUSOLVER threw a '" + cusolverGetErrorString(errorCode) +
                          "' error code";
        if (functionName != "unknown")
        {
            msg += " in function '" + functionName + "'.";
        }
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

// turns a cufft error code into human readable text
std::string cufftGetErrorString(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";

    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    }

    return "unknown";
}

// displays an error message if the cufft computation did not end in sucess
void checkCufftErrorCode(const cufftResult errorCode,
                         const std::string &functionName)
{
    if (errorCode != CUFFT_SUCCESS)
    {
        auto log = toast::Logger::get();
        std::string msg = "CUFFT threw a '" + cufftGetErrorString(errorCode) +
                          "' error code";
        if (functionName != "unknown")
        {
            msg += " in function '" + functionName + "'.";
        }
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

// ------------------------------------------------------------------------------------
// MEMORY POOL

// constant used to make sure allocations are aligned
const int ALIGNEMENT_SIZE = 512;

// converts a number of gigabytes into a number of bytes
size_t GigagbytesToBytes(const size_t nbGB)
{
    return nbGB * 1073741824l;
}

// returns a given fraction of the free (currently unused) GPU memory, in bytes
size_t FractionOfFreeGPUMemory(const double fraction)
{
    // computes the GPU memory available
    size_t free_byte;
    size_t total_byte;
    const cudaError errorCodeMemGetInfo = cudaMemGetInfo(&free_byte, &total_byte);
    checkCudaErrorCode(errorCodeMemGetInfo, "FractionOfGPUMemory::cudaMemGetInfo");

    // computes the portion that we want to reserve
    size_t result = fraction * free_byte;

    // makes sure the end of the reservation is a multiple of the alignement
    if (result % ALIGNEMENT_SIZE != 0)
    {
        result += ALIGNEMENT_SIZE - (result % ALIGNEMENT_SIZE);
    }
    return result;
}

// creating a new `GPU_memory_block_t`
// `cpu_ptr` is optional but can be passed to keep trace of the origin of the data
// when allocating during a cpu-2-gpu copy operation
GPU_memory_block_t::GPU_memory_block_t(void *gpu_ptr, size_t size_bytes_arg,
                                       void *cpu_ptr_arg)
{
    // stores size of the allocation, in bytes
    size_bytes = size_bytes_arg;

    // align size with ALIGNEMENT_SIZE
    if (size_bytes % ALIGNEMENT_SIZE != 0)
    {
        size_bytes += ALIGNEMENT_SIZE - (size_bytes % ALIGNEMENT_SIZE);
    }

    // defines the start and end gpu pointers
    start = gpu_ptr;
    end = static_cast<char *>(start) + size_bytes;

    // the block starts allocated (it might be freed later)
    isFree = false;

    // stores cpu pointer (or nullptr)
    cpu_ptr = cpu_ptr_arg;
}

// constructor, does the initial allocation
GPU_memory_pool::GPU_memory_pool() : blocks()
{

    // Get the requested fraction of per-process memory from the environment.
    // If the user does not specify this, use something conservative that is
    // likely to work.
    double fraction = 0.9;
    char *envval = ::getenv("CUDA_MEMPOOL_FRACTION");
    if (envval != NULL)
    {
        try
        {
            fraction = ::atof(envval);
        }
        catch (...)
        {
            fraction = 0.9;
        }
    }

    // Reduce this by the number of processes sharing a device
    auto &env = toast::Environment::get();
    int n_acc;
    int n_proc_per_dev;
    int my_dev;
    env.get_acc(&n_acc, &n_proc_per_dev, &my_dev);
    if (n_proc_per_dev > 1)
    {
        fraction /= (double)n_proc_per_dev;
    }

    // Get the number of bytes for this fraction
    available_memory_bytes = FractionOfFreeGPUMemory(fraction);

    // allocates the memory
    const cudaError errorCode = cudaMalloc(&start, available_memory_bytes);

    // checks the error code and try to give an informative message in case of insufficient memory
    if (errorCode == cudaErrorMemoryAllocation)
    {
        // gets memory information
        size_t free_byte;
        size_t total_byte;
        const cudaError errorCodeMemGetInfo = cudaMemGetInfo(&free_byte, &total_byte);
        checkCudaErrorCode(errorCodeMemGetInfo, "GPU_memory_pool_t::GPU_memory_pool_t::cudaMemGetInfo");
        // converts it from bytes to MB
        const double requested_mb = double(available_memory_bytes) / 1024.0 / 1024.0;
        const double free_mb = double(free_byte) / 1024.0 / 1024.0;
        const double total_mb = double(total_byte) / 1024.0 / 1024.0;
        const double used_mb = double(total_mb - free_mb);
        // displays error message
        auto log = toast::Logger::get();
        std::string msg = "GPU_memory_pool: Unable to pre-allocate " +
                          std::to_string(requested_mb) + "MB (" + std::to_string(free_mb) + "MB available, " +
                          std::to_string(used_mb) + "MB used, " +
                          std::to_string(total_mb) + "MB total on this GPU).";
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
    else
    {
        checkCudaErrorCode(errorCode, "GPU memory pre-allocation");
    }

    // first block to mark the starting point
    GPU_memory_block_t initialBlock(start, 0);
    blocks.push_back(initialBlock);

    // creates cublas handle
    cublasStatus_t statusHandleBlas = cublasCreate(&handleBlas);
    checkCublasErrorCode(statusHandleBlas);

    // creates cusolver handle
    cusolverStatus_t statusHandleCusolver = cusolverDnCreate(&handleSolver);
    checkCusolverErrorCode(statusHandleCusolver);

    // gets jacobi parameters for batched syev
    cusolverStatus_t statusJacobiParams = cusolverDnCreateSyevjInfo(&jacobiParameters);
    checkCusolverErrorCode(statusJacobiParams);
}

// destructor, insures that the pre-allocation is released
GPU_memory_pool::~GPU_memory_pool()
{
    const cudaError errorCode = cudaFree(start);
    checkCudaErrorCode(errorCode, "GPU memory de-allocation");

    // free cublas handle
    cublasDestroy(handleBlas);

    // free cusolver handle
    cusolverDnDestroy(handleSolver);

    // destroys jacobi parameters for batched syev
    cusolverStatus_t statusJacobiParams = cusolverDnDestroySyevjInfo(jacobiParameters);
    checkCusolverErrorCode(statusJacobiParams);
}

GPU_memory_pool &GPU_memory_pool::get()
{
    static GPU_memory_pool instance;

    return instance;
}

// allocates memory, starting from the end of the latest allocated block
// `cpu_ptr` is optional but can be passed to keep trace of the origin of the data
// when allocating during a cpu-2-gpu copy operation
cudaError GPU_memory_pool::malloc(void **gpu_ptr, size_t size_bytes,
                                  void *cpu_ptr)
{
    // insure two threads cannot interfere
    const std::lock_guard<std::mutex> lock(alloc_mutex);

    // builds a block starting at the end of the existing blocks
    *gpu_ptr = blocks.back().end;
    const GPU_memory_block_t memoryBlock(*gpu_ptr, size_bytes, cpu_ptr);

    // errors out if the allocation goes beyond the preallocated memory
    const size_t usedMemory = static_cast<char *>(memoryBlock.end) -
                              static_cast<char *>(start);
    if (usedMemory > available_memory_bytes)
    {
        std::cerr << "INSUFICIENT GPU MEMORY PREALOCATION"
                  << " GPU memory that would be taken after this allocation:" << usedMemory
                  << " (number of bytes requested by this allocation:" << size_bytes << ")"
                  << " total GPU-memory preallocated:" << available_memory_bytes << std::endl;
        *gpu_ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }

    // stores the block and returns
    blocks.push_back(memoryBlock);
    return cudaSuccess;
}

// frees memory by releasing the block
void GPU_memory_pool::free(void *gpu_ptr)
{
    // insure two threads cannot interfere
    const std::lock_guard<std::mutex> lock(alloc_mutex);

    // gets the index of gpu_ptr in the blocks vector, starting from the end
    int i = blocks.size() - 1;
    while ((blocks[i].start != gpu_ptr) and (i >= 0))
    {
        i--;
    }

    // errors-out if we cannot find `gpu_ptr`
    if (i < 0)
    {
        auto log = toast::Logger::get();
        std::string msg =
            "GPU_memory_pool::free: either `gpu_ptr` does not map to GPU memory allocated with this GPU_memory_pool or you have already freed this memory.";
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }

    // frees gpu_ptr
    blocks[i].isFree = true;

    // if gpu_ptr was the last elements, frees a maximum of blocks
    while (blocks.back().isFree)
    {
        blocks.pop_back();
    }
}

#endif // ifdef HAVE_CUDALIBS
