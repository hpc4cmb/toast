
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_GPU_HELPER_H
#define TOAST_GPU_HELPER_H
#ifdef HAVE_CUDALIBS

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>

void checkCudaErrorCode(const cudaError errorCode);
void checkCublasErrorCode(const cublasStatus_t errorCode);
void checkCusolverErrorCode(const cusolverStatus_t errorCode);

#endif //HAVE_CUDALIBS
#endif //TOAST_GPU_HELPER_H
