/*
Copyright (c) 2016 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <stdint.h>


/* memory management */

double * pytoast_mem_aligned_f64(size_t n);

float * pytoast_mem_aligned_f32(size_t n);

int64_t * pytoast_mem_aligned_i64(size_t n);

uint64_t * pytoast_mem_aligned_u64(size_t n);

int32_t * pytoast_mem_aligned_i32(size_t n);

uint32_t * pytoast_mem_aligned_u32(size_t n);

int16_t * pytoast_mem_aligned_i16(size_t n);

uint16_t * pytoast_mem_aligned_u16(size_t n);

int8_t * pytoast_mem_aligned_i8(size_t n);

uint8_t * pytoast_mem_aligned_u8(size_t n);

void pytoast_mem_aligned_free(void * mem);

