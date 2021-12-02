
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_GPU_HELPER_H
#define TOAST_GPU_HELPER_H
#ifdef HAVE_CUDALIBS

#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

// checks on various type of cuda error codes
void checkCudaErrorCode(const cudaError errorCode, const std::string &functionName = "unknown");
void checkCublasErrorCode(const cublasStatus_t errorCode, const std::string &functionName = "unknown");
void checkCusolverErrorCode(const cusolverStatus_t errorCode, const std::string &functionName = "unknown");
void checkCufftErrorCode(const cufftResult errorCode, const std::string &functionName = "unknown");

// functions to get a number of bbytes for an allocation
size_t GigagbytesToBytes(const size_t nbGB = 4);
size_t FractionOfGPUMemory(const double fraction = 0.9);

// represents a block of memory that has been allocated
class GPU_memory_block_t
{
public:
    // pointers to the start and end of the gpu allocation represented by the block
    void *start;
    void *end;
    // the size of the allocation in bytes
    size_t size_bytes;
    // whether that block has been freed (meaning it could be garbage collected later)
    bool isFree;
    // cpu pointer that was used as a source for the allocation, might be nullptr if the data didn't come from cpu
    void *cpu_ptr;

    // creating a new `GPU_memory_block_t`
    // `cpu_ptr` is an, optional, pointer to the cpu memory that was moved into the block
    GPU_memory_block_t(void *ptr, size_t size_bytes, void *cpu_ptr = nullptr);
};

// used to recycle GPU allocation
class GPU_memory_pool_t
{
private:
    // used to make class threadsafe
    std::mutex alloc_mutex;
    // starting point of the current allocation
    void *start;
    // what is the total memory allocated here, in bytes
    size_t available_memory_bytes;
    // memory blocks that have been allocated but cannot be freed yet
    std::vector<GPU_memory_block_t> blocks;

public:
    // reused handles for linear algebra (as they are very expensive to create)
    cublasHandle_t handleBlas = NULL;
    cusolverDnHandle_t handleSolver = NULL;
    syevjInfo_t jacobiParameters = NULL;

    GPU_memory_pool_t(size_t bytesPreallocated);
    ~GPU_memory_pool_t();
    cudaError malloc(void **output_ptr, size_t size_bytes, void *cpu_ptr = nullptr);
    void free(void *gpu_ptr);

    // allocates memory for the given number of elements and returns a pointer to the allocated memory
    // this function is slightly higher level than malloc as it is typed and takes a number of elements rather than a number of bytes
    template <typename T>
    T *alloc(size_t nb_elements)
    {
        T *output_ptr = NULL;
        const cudaError errorCode = this->malloc((void **)&output_ptr, nb_elements * sizeof(T));
        checkCudaErrorCode(errorCode, "GPU_memory_pool_t::alloc");
        return output_ptr;
    }

    // allocates gpu memory and returns a pointer to the memory after having copied the data there
    // stores the cpu_ptr for a potential later transfer back
    template <typename T>
    T *toDevice(T *data, size_t nb_elements)
    {
        // memory allocation
        void *data_gpu = NULL;
        const cudaError errorCodeMalloc = this->malloc(&data_gpu, nb_elements * sizeof(T), data);
        checkCudaErrorCode(errorCodeMalloc, "GPU_memory_pool_t::toDevice (malloc)");
        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(data_gpu, data, nb_elements * sizeof(T), cudaMemcpyHostToDevice);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::toDevice (memcpy)");
        return static_cast<T *>(data_gpu);
    }

    // gets the given number of elements back from GPU
    // put them in the given cpu memory
    // deallocates gpu memory
    template <typename T>
    void fromDevice(T *data_cpu, T *data_gpu, size_t nb_elements)
    {
        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(data_cpu, data_gpu, nb_elements * sizeof(T), cudaMemcpyDeviceToHost);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::fromDevice (memcpy)");
        // deallocation
        this->free(data_gpu);
    }

    // gets data back from GPU and into the pointer whence it came from
    // deallocates gpu memory
    template <typename T>
    void fromDevice(T *data_cpu)
    {
        // gets index of cpu_ptr in blocks, starting from the end
        int i = blocks.size() - 1;
        while (blocks[i].cpu_ptr != data_cpu)
        {
            i--;
        }
        // extract the gpu_ptr and the size of the allocation in bytes
        T *data_gpu = blocks[i].start;
        const size_t size = blocks[i].size_bytes;

        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(data_cpu, data_gpu, size, cudaMemcpyDeviceToHost);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::fromDevice (memcpy)");
        // deallocation
        this->free(data_gpu);
    }
};

// global variable (one instance per thread) containing the pool
extern GPU_memory_pool_t GPU_memory_pool;

#endif // HAVE_CUDALIBS
#endif // TOAST_GPU_HELPER_H
