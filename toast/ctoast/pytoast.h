/*
Copyright (c) 2016 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef PYTOAST_H
#define PYTOAST_H

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


/* quaternion array operations */

/*
Dot product of a lists of arrays, returns a column array
*/
void pytoast_qarraylist_dot(int n, int m, int d, const double* a, const double* b, double* dotprod);

/*
Inverse of quaternion array q
q is a n by 4 array
*/
void pytoast_qinv(int n, double* q);

/*
Norm of quaternion array list
*/
void pytoast_qamplitude(int n, int m, int d, const double* v, double* l2);

/*
Normalize quaternion array q or array list to unit quaternions
*/
void pytoast_qnorm(int n, int m, int d, const double* q_in, double* q_out);

/*
Normalize quaternion array q or array list to unit quaternions
*/
void pytoast_qnorm_inplace(int n, int m, int d, double* q);

/*
Rotate vector v by n-quaternion array q and returns array with rotate n-vectors
*/
void pytoast_qrotate(int n, const double* v, const double* q_in, double* v_out);

/*
Multiply arrays of quaternions
p, q and r are n by 4 arrays
*/
void pytoast_qmult(int n, const double* p, const double* q, double* r);

/*
Spherical interpolation of q quaternion array from time to targettime
*/
void pytoast_slerp(int n_time, int n_targettime, const double* time, const double* targettime, const double* q_in, double* q_interp);

/*
Exponential of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qexp(int n, const double* q_in, double* q_out);

/*
Natural logarithm of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qln(int n, const double* q_in, double* q_out);

/*
Real power of quaternion array
p is a n-array, q_in and q_out are n by 4 arrays
*/
void pytoast_qpow(int n, const double* p, const double* q_in, double* q_out);

/*
Creates rotation quaternions of angles (in [rad]) around axes [already normalized]
axis is an n by 3 array, angle is a n-array, q_out is a n by 4 array
*/
void pytoast_from_axisangle(int n, const double* axis, const double* angle, double* q_out);

/*
Returns the axes and angles of rotation of a quaternion array.
*/
void pytoast_to_axisangle(int n, const double* q, double* axis, double* angle);

/*
Creates the rotation matrix corresponding to a quaternion
q is a 4 array, rotmat is a 3 by 3 array
*/
void pytoast_to_rotmat(const double* q, double* rotmat);

/*
Creates the quaternion from a rotation matrix
rotmat is a 3 by 3 array, q is a 4-array.
*/
void pytoast_from_rotmat(const double* rotmat, double* q);

/*
Creates the quaternion from two normalized vectors (be careful with colinear vectors)
vec1 and vec2 are 3-arrays, q is a 4-array
*/
void pytoast_from_vectors(const double* vec1, const double* vec2, double* q);


/* counter-based random number generation with Random123 library */

double pytoast_uneg11(uint64_t in);

double pytoast_u01(uint64_t in);

void pytoast_generate_grv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array);

void pytoast_generate_neg11rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array);

void pytoast_generate_01rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array);

void pytoast_generate_uint64rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, uint64_t* rand_array);


#endif
