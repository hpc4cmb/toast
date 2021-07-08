
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
};

// global variable (one instance per thread) containing the pool
extern thread_local GPU_memory_pool_t GPU_memory_pool;

#endif //HAVE_CUDALIBS
#endif //TOAST_GPU_HELPER_H
