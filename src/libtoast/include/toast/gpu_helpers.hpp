
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_GPU_HELPER_H
#define TOAST_GPU_HELPER_H
#ifdef HAVE_CUDALIBS

#include <string>
#include <unordered_map>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>

void checkCudaErrorCode(const cudaError errorCode, const std::string& functionName = "unknown");
void checkCublasErrorCode(const cublasStatus_t errorCode, const std::string& functionName = "unknown");
void checkCusolverErrorCode(const cusolverStatus_t errorCode, const std::string& functionName = "unknown");

class GPU_memory_block_t
{
public:
    void* start;
    void* end;
    bool isFree;

    GPU_memory_block_t(void* ptr, size_t size);
};

// used to recycle GPU allocation
// this class is NOT threadsafe
// one should use one instance per thread and not have a thread free memory it did not allocate
class GPU_memory_pool_t
{
private:
    // starting point of the current allocation
    void* start;
    // what is the total memory allocated here
    size_t available_memory;
    // memory blocks that have been allocated at some point
    std::vector<GPU_memory_block_t> blocks;
public:
    GPU_memory_pool_t();
    ~GPU_memory_pool_t();
    cudaError malloc(void** output_ptr, size_t size);
    void free(void* ptr);

    // allocates memory for the given number of elements and returns a pointer to the allocated memory
    // this function is slightly higher level than malloc
    template<typename T>
    T* alloc(size_t size)
    {
        T* output_ptr = NULL;
        cudaError errorCode = this->malloc((void**)&output_ptr, size*sizeof(T));
        checkCudaErrorCode(errorCode, "GPU_memory_pool_t::alloc");
        return output_ptr;
    }

    // allocates gpu memory and returns a pointer to the memory after having copied the data there
    template<typename T>
    T* toDevice(T* data, size_t nbElements)
    {
        // memory allocation
        void* data_gpu = NULL;
        const cudaError errorCodeMalloc = this->malloc(&data_gpu, nbElements*sizeof(T));
        checkCudaErrorCode(errorCodeMalloc, "GPU_memory_pool_t::toDevice (malloc)");
        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(data_gpu, data, nbElements*sizeof(T), cudaMemcpyHostToDevice);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::toDevice (memcpy)");
        return static_cast<T*>(data_gpu);
    }

    // gets data back from GPU and deallocates gpu memory
    template<typename T>
    void fromDevice(T* data_cpu, T* data_gpu, size_t nbElements)
    {
        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(data_cpu, data_gpu, nbElements*sizeof(T), cudaMemcpyDeviceToHost);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::fromDevice (memcpy)");
        // deallocation
        this->free(data_gpu);
    }
};

// global variable (one instance per thread) containing the pool
extern thread_local GPU_memory_pool_t GPU_memory_pool;

#endif //HAVE_CUDALIBS
#endif //TOAST_GPU_HELPER_H
