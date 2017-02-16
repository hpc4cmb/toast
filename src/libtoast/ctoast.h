/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#ifndef CTOAST_H
#define CTOAST_H

#include <mpi.h>

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//--------------------------------------
// Global library initialize / finalize
//--------------------------------------

void ctoast_init ( int argc, char *argv[] );
void ctoast_finalize ( );

//--------------------------------------
// Math sub-library
//--------------------------------------

// aligned memory

void * ctoast_mem_aligned_alloc ( size_t size );
void ctoast_mem_aligned_free ( void * data );

// special functions

void ctoast_sf_sin ( int n, double * ang, double * sinout );
void ctoast_sf_cos ( int n, double * ang, double * cosout );
void ctoast_sf_sincos ( int n, double * ang, double * sinout, double * cosout );
void ctoast_sf_atan2 ( int n, double * y, double * x, double * ang );
void ctoast_sf_sqrt ( int n, double * in, double * out );
void ctoast_sf_rsqrt ( int n, double * in, double * out );
void ctoast_sf_exp ( int n, double * in, double * out );
void ctoast_sf_log ( int n, double * in, double * out );

void ctoast_sf_fast_sin ( int n, double * ang, double * sinout );
void ctoast_sf_fast_cos ( int n, double * ang, double * cosout );
void ctoast_sf_fast_sincos ( int n, double * ang, double * sinout, double * cosout );
void ctoast_sf_fast_atan2 ( int n, double * y, double * x, double * ang );
void ctoast_sf_fast_sqrt ( int n, double * in, double * out );
void ctoast_sf_fast_rsqrt ( int n, double * in, double * out );
void ctoast_sf_fast_exp ( int n, double * in, double * out );
void ctoast_sf_fast_log ( int n, double * in, double * out );

// RNG

void ctoast_rng_dist_uint64 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data );

void ctoast_rng_dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

void ctoast_rng_dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

void ctoast_rng_dist_normal ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

// Quaternion array

void ctoast_qarray_list_dot ( size_t n, size_t m, size_t d, double const * a, double const * b, double * dotprod );

void ctoast_qarray_inv ( size_t n, double * q );

void ctoast_qarray_amplitude ( size_t n, size_t m, size_t d, double const * v, double * norm );

void ctoast_qarray_normalize ( size_t n, size_t m, size_t d, double const * q_in, double * q_out );

void ctoast_qarray_normalize_inplace ( size_t n, size_t m, size_t d, double * q );

void ctoast_qarray_rotate ( size_t n, double const * q, double const * v_in, double * v_out );

void ctoast_qarray_mult ( size_t n, double const * p, double const * q, double * r );

void ctoast_qarray_slerp ( size_t n_time, size_t n_targettime, double const * time, double const * targettime, double const * q_in, double * q_interp );

void ctoast_qarray_exp ( size_t n, double const * q_in, double * q_out );

void ctoast_qarray_ln ( size_t n, double const * q_in, double * q_out );

void ctoast_qarray_pow ( size_t n, double const * p, double const * q_in, double * q_out );

void ctoast_qarray_from_axisangle ( size_t n, double const * axis, double const * angle, double * q_out );

void ctoast_qarray_to_axisangle ( size_t n, double const * q, double * axis, double * angle );

void ctoast_qarray_to_rotmat ( double const * q, double * rotmat );

void ctoast_qarray_from_rotmat ( const double * rotmat, double * q );

void ctoast_qarray_from_vectors ( double const * vec1, double const * vec2, double * q );

// FFT

typedef enum {
    PLAN_FAST = 0,
    PLAN_BEST = 1
} ctoast_fft_plan_type;
  
typedef enum {
    FORWARD = 0,
    BACKWARD = 1
} ctoast_fft_direction;

struct ctoast_fft_r1d_;
typedef struct ctoast_fft_r1d_ ctoast_fft_r1d;

ctoast_fft_r1d * ctoast_fft_r1d_alloc ( int64_t length, int64_t n, ctoast_fft_plan_type type, ctoast_fft_direction dir, double scale );
void ctoast_fft_r1d_free ( ctoast_fft_r1d * frd );

double ** ctoast_fft_r1d_tdata ( ctoast_fft_r1d * frd );

double ** ctoast_fft_r1d_fdata ( ctoast_fft_r1d * frd );

void ctoast_fft_r1d_exec ( ctoast_fft_r1d * frd );

struct ctoast_fft_r1d_plan_store_;
typedef struct ctoast_fft_r1d_plan_store_ ctoast_fft_r1d_plan_store;

ctoast_fft_r1d_plan_store * ctoast_fft_r1d_plan_store_get ( );

void ctoast_fft_r1d_plan_store_clear ( ctoast_fft_r1d_plan_store * pstore );

void ctoast_fft_r1d_plan_store_cache ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n );

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_forward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n );

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_backward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n );

// Healpix

struct ctoast_healpix_pixels_;
typedef struct ctoast_healpix_pixels_ ctoast_healpix_pixels;

ctoast_healpix_pixels * ctoast_healpix_pixels_alloc ( int64_t nside );
void ctoast_healpix_pixels_free ( ctoast_healpix_pixels * hpix );

void ctoast_healpix_pixels_reset ( ctoast_healpix_pixels * hpix, int64_t nside );

void ctoast_healpix_pixels_vec2zphi ( ctoast_healpix_pixels * hpix, int64_t n, double * vec, double * phi, int * region, double * z, double * rtz );

void ctoast_healpix_pixels_theta2z ( ctoast_healpix_pixels * hpix, int64_t n, double * theta, int * region, double * z, double * rtz );

void ctoast_healpix_pixels_zphi2nest ( ctoast_healpix_pixels * hpix, int64_t n, double * phi, int * region, double * z, double * rtz, int64_t * pix );

void ctoast_healpix_pixels_zphi2ring ( ctoast_healpix_pixels * hpix, int64_t n, double * phi, int * region, double * z, double * rtz, int64_t * pix );

void ctoast_healpix_pixels_ang2nest ( ctoast_healpix_pixels * hpix, int64_t n, double * theta, double * phi, int64_t * pix );

void ctoast_healpix_pixels_ang2ring ( ctoast_healpix_pixels * hpix, int64_t n, double * theta, double * phi, int64_t * pix );

void ctoast_healpix_pixels_vec2nest ( ctoast_healpix_pixels * hpix, int64_t n, double * vec, int64_t * pix );

void ctoast_healpix_pixels_vec2ring ( ctoast_healpix_pixels * hpix, int64_t n, double * vec, int64_t * pix );

void ctoast_healpix_pixels_ring2nest ( ctoast_healpix_pixels * hpix, int64_t n, int64_t * ringpix, int64_t * nestpix );

void ctoast_healpix_pixels_nest2ring ( ctoast_healpix_pixels * hpix, int64_t n, int64_t * nestpix, int64_t * ringpix );

void ctoast_healpix_pixels_degrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix );

void ctoast_healpix_pixels_degrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix );

void ctoast_healpix_pixels_upgrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix );

void ctoast_healpix_pixels_upgrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix );


//--------------------------------------
// TOD sub-library
//--------------------------------------



//--------------------------------------
// Map functions
//--------------------------------------




//--------------------------------------
// Run test suite
//--------------------------------------

int ctoast_test_runner ( int argc, char *argv[] );


#ifdef __cplusplus
}
#endif

#endif


