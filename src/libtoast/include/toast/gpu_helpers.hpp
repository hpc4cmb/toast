
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_GPU_HELPER_H
#define TOAST_GPU_HELPER_H
#ifdef HAVE_CUDALIBS

#include <string>
#include <vector>
#include <mutex>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

// checks for various type of cuda error codes
void checkCudaErrorCode(const cudaError errorCode, const std::string &functionName = "unknown");
void checkCublasErrorCode(const cublasStatus_t errorCode, const std::string &functionName = "unknown");
void checkCusolverErrorCode(const cusolverStatus_t errorCode, const std::string &functionName = "unknown");
void checkCufftErrorCode(const cufftResult errorCode, const std::string &functionName = "unknown");

// functions to get a number of bytes for an allocation
size_t GigagbytesToBytes(const size_t nbGB = 4);
size_t FractionOfGPUMemory(const double fraction = 0.9);

// represents a block of memory that has been allocated
class GPU_memory_block_t
{
public:
    // pointers to the start and end of the gpu allocation represented by the block
    void *start;
    void *end;
    // the size of the allocation, in bytes
    // note that this might be lower than the distance between start and end due to padding introduced for alignement purposes
    size_t size_bytes;
    // whether that block has been freed (meaning it could be garbage collected later)
    bool isFree;
    // pointer to the cpu data that was put in the block
    // optional (might be nullptr) if the block was not allocated in a cpu-2-gpu copy operation
    void *cpu_ptr;

    // creating a new `GPU_memory_block_t`
    // `cpu_ptr` is optional but can be passed to keep trace of the origin of the data
    // when allocating during a cpu-2-gpu copy operation
    GPU_memory_block_t(void *ptr, size_t size_bytes, void *cpu_ptr = nullptr);
};

// allocates a slab of memory and then recycles allocations
class GPU_memory_pool_t
{
private:
    // used to make class threadsafe
    std::mutex alloc_mutex;
    // starting point of the gpu pre-allocation
    void *start;
    // what is the total memory allocated by the pool, in bytes
    size_t available_memory_bytes;
    // memory blocks that have been allocated but cannot be freed yet
    std::vector<GPU_memory_block_t> blocks;

public:
    // handles for linear algebra, to be re-used (as they are expensive to create)
    cublasHandle_t handleBlas = NULL;
    cusolverDnHandle_t handleSolver = NULL;
    syevjInfo_t jacobiParameters = NULL;

    // allocates a number of bytes on the gpu
    GPU_memory_pool_t(size_t bytesPreallocated);

    // frees the pool
    ~GPU_memory_pool_t();

    // gets a pointer to a number of preallocated bytes
    // `cpu_ptr` is optional but can be passed to keep trace of the origin of the data
    // when allocating during a cpu-2-gpu copy operation
    cudaError malloc(void **gpu_ptr, size_t size_bytes, void *cpu_ptr = nullptr);

    // free the memory pointed to
    void free(void *gpu_ptr);

    // allocates memory for the given number of elements and returns a pointer to the allocated memory
    // this function is slightly higher level than malloc as:
    // - it is typed,
    // - it takes a number of elements rather than a number of bytes
    // - it returns a pointer rather than taking it as an input
    template <typename T>
    T *alloc(size_t nb_elements)
    {
        T *output_ptr = NULL;
        const cudaError errorCode = this->malloc((void **)&output_ptr, nb_elements * sizeof(T));
        checkCudaErrorCode(errorCode, "GPU_memory_pool_t::alloc");
        return output_ptr;
    }

    // allocates gpu memory for the given number of elements
    // returns a pointer to the memory after having copied the cpu data there
    // stores the `cpu_ptr` for a potential transfer back when freeing with `fromDevice`
    template <typename T>
    T *toDevice(T *cpu_ptr, size_t nb_elements)
    {
        // memory allocation
        void *gpu_ptr = NULL;
        const cudaError errorCodeMalloc = this->malloc(&gpu_ptr, nb_elements * sizeof(T), cpu_ptr);
        checkCudaErrorCode(errorCodeMalloc, "GPU_memory_pool_t::toDevice (malloc)");
        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(gpu_ptr, cpu_ptr, nb_elements * sizeof(T), cudaMemcpyHostToDevice);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::toDevice (memcpy)");
        return static_cast<T *>(gpu_ptr);
    }

    // gets the given number of elements back from GPU to the given CPU location
    // frees the gpu memory
    template <typename T>
    void fromDevice(T *cpu_ptr, T *gpu_ptr, size_t nb_elements)
    {
        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(cpu_ptr, gpu_ptr, nb_elements * sizeof(T), cudaMemcpyDeviceToHost);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::fromDevice (memcpy)");
        // deallocation
        this->free(gpu_ptr);
    }

    // gets the given number of elements back from GPU to the given CPU location
    // frees the gpu memory
    // WARNING: this function assumes that `toDevice` was called with `cpu_ptr`
    //          it will use this assumption to identify the gpu pointer and the size of the allocation
    template <typename T>
    void fromDevice(T *cpu_ptr)
    {
        // gets the index of cpu_ptr in the blocks vector, starting from the end
        int i = blocks.size() - 1;
        while ((blocks[i].cpu_ptr != cpu_ptr) and (i >= 0))
        {
            i--;
        }
        // errors-out if `cpu_ptr` was not use in a `toDevice` call
        // meaning that we cannot find the associated block of gpu memory
        if (i < 0)
        {
            auto log = toast::Logger::get();
            std::string msg = "GPU_memory_pool_t::fromDevice(T*): either `cpu_ptr` was never used to send memory to the gpu with the `toDevice` function or you have already freed the associated GPU memory.";
            log.error(msg.c_str());
            throw std::runtime_error(msg.c_str());
        }
        // extract the gpu_ptr and the size of the allocation in bytes
        T *data_gpu = blocks[i].start;
        const size_t size = blocks[i].size_bytes;

        // data transfer
        const cudaError errorCodeMemcpy = cudaMemcpy(cpu_ptr, data_gpu, size, cudaMemcpyDeviceToHost);
        checkCudaErrorCode(errorCodeMemcpy, "GPU_memory_pool_t::fromDevice(T *cpu_ptr) (memcpy)");
        // deallocation
        this->free(data_gpu);
    }
};

// global variable containing the TOAST GPU memory pool
extern GPU_memory_pool_t GPU_memory_pool;

#endif // HAVE_CUDALIBS
#endif // TOAST_GPU_HELPER_H
