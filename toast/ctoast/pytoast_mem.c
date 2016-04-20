/*
Copyright (c) 2016 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


/* Functions for memory management */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pytoast.h>


/* this should work for all modern systems, including MIC */
#define alignment 64


double * pytoast_mem_aligned_f64(size_t n) {
    int ret;
    size_t bytes = n * sizeof(double);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (double*)mem;
}


float * pytoast_mem_aligned_f32(size_t n) {
    int ret;
    size_t bytes = n * sizeof(float);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (float*)mem;
}


int64_t * pytoast_mem_aligned_i64(size_t n) {
    int ret;
    size_t bytes = n * sizeof(int64_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (int64_t*)mem;
}


uint64_t * pytoast_mem_aligned_u64(size_t n) {
    int ret;
    size_t bytes = n * sizeof(uint64_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (uint64_t*)mem;
}


int32_t * pytoast_mem_aligned_i32(size_t n) {
    int ret;
    size_t bytes = n * sizeof(int32_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (int32_t*)mem;
}


uint32_t * pytoast_mem_aligned_u32(size_t n) {
    int ret;
    size_t bytes = n * sizeof(uint32_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (uint32_t*)mem;
}


int16_t * pytoast_mem_aligned_i16(size_t n) {
    int ret;
    size_t bytes = n * sizeof(int16_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (int16_t*)mem;
}


uint16_t * pytoast_mem_aligned_u16(size_t n) {
    int ret;
    size_t bytes = n * sizeof(uint16_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (uint16_t*)mem;
}


int8_t * pytoast_mem_aligned_i8(size_t n) {
    int ret;
    size_t bytes = n * sizeof(int8_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (int8_t*)mem;
}


uint8_t * pytoast_mem_aligned_u8(size_t n) {
    int ret;
    size_t bytes = n * sizeof(uint8_t);
    void * mem = NULL;
    ret = posix_memalign(&mem, alignment, bytes);
    if (ret != 0) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", bytes);
        return NULL;
    }
    memset(mem, 0, bytes);
    /* fprintf(stderr, "allocated memory at %p\n", mem); */
    return (uint8_t*)mem;
}


void pytoast_mem_aligned_free(void * mem) {
    /* for now, we just call standard free, which is allowed for posix_memalign */
    /* fprintf(stderr, "freeing memory at %p\n", mem); */
    free(mem);
    return;
}



