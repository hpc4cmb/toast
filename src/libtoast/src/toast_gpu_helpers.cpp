
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/gpu_helpers.hpp>
#include <toast/sys_utils.hpp>

#ifdef HAVE_CUDALIBS

//---------------------------------------------------------------------------------------
// ERROR CODE CHECKING

// displays an error message if the computation did not end in success
void checkCudaErrorCode(const cudaError errorCode, const std::string& functionName)
{
    if (errorCode != cudaSuccess)
    {
        auto log = toast::Logger::get();
        std::string msg = "CUDA threw a '" + std::string(cudaGetErrorString(errorCode)) + "' error code";
        if (functionName != "unknown") msg += " in function '" + functionName + "'.";
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

// turns an error code into human readable text
std::string cublasGetErrorString(const cublasStatus_t errorCode)
{
    switch(errorCode)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "unknown";
}

// displays an error message if the computation did not end in sucess
void checkCublasErrorCode(const cublasStatus_t errorCode, const std::string& functionName)
{
    if(errorCode != CUBLAS_STATUS_SUCCESS)
    {
        auto log = toast::Logger::get();
        std::string msg = "CUBLAS threw a '" + cublasGetErrorString(errorCode) + "' error code";
        if (functionName != "unknown") msg += " in function '" + functionName + "'.";
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

// turns an error code into human readable text
std::string cusolverGetErrorString(const cusolverStatus_t errorCode)
{
    switch(errorCode)
    {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
        case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
        case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED: return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER: return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
        case CUSOLVER_STATUS_IRS_INTERNAL_ERROR: return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_IRS_NOT_SUPPORTED: return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_IRS_OUT_OF_RANGE: return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
        case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES: return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED: return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED: return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
        case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR: return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
        case CUSOLVER_STATUS_INVALID_WORKSPACE: return "CUSOLVER_STATUS_INVALID_WORKSPACE";
    }

    return "unknown";
}

// displays an error message if the computation did not end in sucess
void checkCusolverErrorCode(const cusolverStatus_t errorCode, const std::string& functionName)
{
    if(errorCode != CUSOLVER_STATUS_SUCCESS)
    {
        auto log = toast::Logger::get();
        std::string msg = "CUSOLVER threw a '" + cusolverGetErrorString(errorCode) + "' error code";
        if (functionName != "unknown") msg += " in function '" + functionName + "'.";
        log.error(msg.c_str());
        throw std::runtime_error(msg.c_str());
    }
}

//---------------------------------------------------------------------------------------
// MEMORY POOL

// initialize the storage variables
GPU_memory_pool_t::GPU_memory_pool_t(): pool(), size_of_ptr() {}

// destructor, insures that all the allocation are freed
GPU_memory_pool_t::~GPU_memory_pool_t()
{
    free_all();
}

// frees all the memory allocated by the pool
// does NOT purge the sizes of allocations currently in use from size_of_ptr
void GPU_memory_pool_t::free_all()
{
    //std::cerr << "free all: " << pool.size() << " allocations" << std::endl;
    for(void* ptr : pool)
    {
        size_of_ptr.erase(ptr);
        cudaFree(ptr);
    }
    pool.clear();
}

// tries to find the memory allocation of size closest (but above or equal) to size in the memory pool
// returns NULL if nothing was found
void * GPU_memory_pool_t::find(size_t size)
{
    void * result = NULL;
    int index_result = -1;
    int lower_memory_overhead = INT32_MAX;

    // finds the closest fit in the gpu memory pool
    for(unsigned int i = 0; i < pool.size(); i++)
    {
        void * ptr = pool[i];
        const size_t sizePtr = size_of_ptr[ptr];
        const int memory_overhead = static_cast<int>(sizePtr) - size;
        // keeps the allocation if it is larger then needed and lower than the previous best
        if( (memory_overhead >= 0) and (memory_overhead < lower_memory_overhead) )
        {
            lower_memory_overhead = memory_overhead;
            result = ptr;
            index_result = i;
        }
    }

    // removes the result from the pool
    if(index_result != -1)
    {
        pool.erase(pool.begin() + index_result);
    }

    return result;
}

// recycles memory from the pool or, if necessary, allocates new memory
cudaError GPU_memory_pool_t::malloc(void** output_ptr, size_t size)
{
    cudaError errorCode = cudaSuccess;
    *output_ptr = find(size);

    // if we did not find an allocation large enough in the pool
    if(*output_ptr == NULL)
    {
        //std::cerr << "actual malloc: " << size << std::endl;
        // tries doing the allocation from scratch
        // allocates with CUDA to get unified memory that can be accessed from CPU and GPU transparently
        // garantees that the memory will be "suitably aligned for any kind of variable"
        errorCode = cudaMallocManaged(output_ptr, size);

        // if it fails, try after a purge of the pool
        if (errorCode != cudaSuccess)
        {
            free_all();
            errorCode = cudaMallocManaged(output_ptr, size);
        }

        size_of_ptr[*output_ptr] = size;
    }

    return errorCode;
}

// frees memory
// by storing it in the pool for later reuse
// its size is still in the pool
void GPU_memory_pool_t::free(void* ptr)
{
    pool.push_back(ptr);
}

// global variable
thread_local GPU_memory_pool_t GPU_memory_pool = GPU_memory_pool_t();

#endif
