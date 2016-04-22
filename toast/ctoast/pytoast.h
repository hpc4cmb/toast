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


/*
Dot product of a lists of arrays, returns a column array
Arrays a and b must be n by 4, but only the m first columns will be used for the dot product (n-array)
*/
void pytoast_qarraylist_dot(const int n, const int m, const double* a, const double* b, double* dotprod);

/*
Inverse of quaternion array q
q is a n by 4 array
*/
void pytoast_qinv(const int n, double* q);

/*
Norm of quaternion array list
v must be a n by 4 array, only the first m rows will be considered, l2 is a n-array
*/
void pytoast_qamplitude(const int n, const int m, const double* v, double* l2);

/*
Normalize quaternion array q or array list to unit quaternions
q_in must be a n by 4 arrray, only the first m rows will be considered, results are output to q_out
*/
void pytoast_qnorm(const int n, const int m, const double* q_in, double* q_out);

/*
Normalize quaternion array q or array list to unit quaternions
q must be a n by 4 arrray, only the first m rows will be considered, results are written to q
*/
void pytoast_qnorm_inplace(const int n, const int m, double* q);

/*
Rotate vector v by n-quaternion array q and returns array with rotate n-vectors
v is a 3D-vector and q is a n by 4 array, v_out is a n by 3 array.
*/
void pytoast_qrotate(const int n, const double* v, const double* q_in, double* v_out);

/*
Multiply arrays of quaternions
p, q and r are n by 4 arrays
*/
void pytoast_qmult(const int n, const double* p, const double* q, double* r);

/*
Normalized interpolation of q quaternion array from time to targettime
targettime is a n_time-array, time is a 2-array (start and end time), q_in is 
a 2 by 4 array, q_interp is a n_time by 4 array
*/
void pytoast_nlerp(const int n_time, const double* targettime, const double* time, const double* q_in, double* q_interp);

/*
Spherical interpolation of q quaternion array from time to targettime
targettime is a n_time-array, time is a 2-array (start and end time), q_in 
is a 2 by 4 array, q_interp is a n_time by 4 array
*/
void pytoast_slerp(const int n_time, const double* targettime, const double* time, const double* q_in, double* q_interp);

/*
Compute the time vector used for interpolation
targettime is a n_time-array, time is a 2-array (start and end time), t_matrix is a n_time-array
*/
void pytoast_compute_t(const int n_time, const double* targettime, const double* time, double* t_matrix);

/*
Exponential of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qexp(const int n, const double* q_in, double* q_out);

/*
Natural logarithm of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qln(const int n, const double* q_in, double* q_out);

/*
Real power of quaternion array
p is a n-array, q_in and q_out are n by 4 arrays
*/
void pytoast_qpow(const int n, const double* p, const double* q_in, double* q_out);

/*
Creates rotation quaternions of angles (in [rad]) around axes [already normalized]
axis is an n by 3 array, angle is a n-array, q_out is a n by 4 array
*/
void pytoast_from_axisangle(const int n, const double* axis, const double* angle, double* q_out);

/*
Returns the axis and angle of rotation of a quaternion
q is a 4-array, axis is a 3-array, angle is 1-array
*/
void pytoast_to_axisangle(const double* q, double* axis, double* angle);

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

#endif
