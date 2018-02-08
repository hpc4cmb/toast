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
void ctoast_raise_error(int errcode);

//--------------------------------------
// Util sub-library
//--------------------------------------

struct ctoast_timer_;
typedef struct ctoast_timer_ ctoast_timer;

struct ctoast_timing_manager_;
typedef struct ctoast_timing_manger_ ctoast_timing_manager;

ctoast_timer* ctoast_get_simple_timer(char* ckey, char* cfmt);
void ctoast_del_simple_timer(ctoast_timer*);

int ctoast_timers_enabled();
void ctoast_timers_toggle(int32_t val);
ctoast_timer* ctoast_get_timer(char* ckey);
ctoast_timer* ctoast_get_timer_at(int32_t i);
void ctoast_timer_start(ctoast_timer*);
void ctoast_timer_stop(ctoast_timer*);
void ctoast_timer_report(ctoast_timer*);
uint64_t ctoast_get_timer_instance_count();
void ctoast_op_timer_instance_count(int32_t op, int32_t nhash);

ctoast_timing_manager* ctoast_get_timing_manager();
void ctoast_set_timing_output_file(char* cfname);
void ctoast_serialize_timing_manager(char*);
void ctoast_timing_manager_report();
void ctoast_timing_manager_clear();
void ctoast_timing_manager_set_max_depth(int32_t);
int32_t ctoast_timing_manager_max_depth();
size_t ctoast_timing_manager_size();

double ctoast_timer_real_elapsed(ctoast_timer*);
double ctoast_timer_system_elapsed(ctoast_timer*);
double ctoast_timer_user_elapsed(ctoast_timer*);

//--------------------------------------
// Math sub-library
//--------------------------------------

// aligned memory

void * ctoast_mem_aligned_alloc ( size_t size );
void ctoast_mem_aligned_free ( void * data );

// special functions

void ctoast_sf_sin ( int n, double const * ang, double * sinout );
void ctoast_sf_cos ( int n, double const * ang, double * cosout );
void ctoast_sf_sincos ( int n, double const * ang, double * sinout, double * cosout );
void ctoast_sf_atan2 ( int n, double const * y, double const * x, double * ang );
void ctoast_sf_sqrt ( int n, double const * in, double * out );
void ctoast_sf_rsqrt ( int n, double const * in, double * out );
void ctoast_sf_exp ( int n, double const * in, double * out );
void ctoast_sf_log ( int n, double const * in, double * out );

void ctoast_sf_fast_sin ( int n, double const * ang, double * sinout );
void ctoast_sf_fast_cos ( int n, double const * ang, double * cosout );
void ctoast_sf_fast_sincos ( int n, double const * ang, double * sinout, double * cosout );
void ctoast_sf_fast_atan2 ( int n, double const * y, double const * x, double * ang );
void ctoast_sf_fast_sqrt ( int n, double const * in, double * out );
void ctoast_sf_fast_rsqrt ( int n, double const * in, double * out );
void ctoast_sf_fast_exp ( int n, double const * in, double * out );
void ctoast_sf_fast_log ( int n, double const * in, double * out );

// RNG

void ctoast_rng_dist_uint64 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data );

void ctoast_rng_dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

void ctoast_rng_dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

void ctoast_rng_dist_normal ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, double * data );

void ctoast_rng_dist_normal ( size_t n,
                              uint64_t key1,     uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              double * data );

void ctoast_rng_dist_uint64_mt ( size_t blocks,      size_t n,
                                 uint64_t* key1,     uint64_t* key2,
                                 uint64_t* counter1, uint64_t* counter2,
                                 uint64_t* data );

void ctoast_rng_dist_uniform_01_mt ( size_t blocks,      size_t n,
                                     uint64_t* key1,     uint64_t* key2,
                                     uint64_t* counter1, uint64_t* counter2,
                                     double* data );

void ctoast_rng_dist_uniform_11_mt ( size_t blocks,      size_t n,
                                     uint64_t* key1,     uint64_t* key2,
                                     uint64_t* counter1, uint64_t* counter2,
                                     double* data );

void ctoast_rng_dist_normal_mt ( size_t blocks,      size_t n,
                                 uint64_t* key1,     uint64_t* key2,
                                 uint64_t* counter1, uint64_t* counter2,
                                 double* data );

// Quaternion array

void ctoast_qarray_list_dot ( size_t n, size_t m, size_t d, double const * a, double const * b, double * dotprod );

void ctoast_qarray_inv ( size_t n, double * q );

void ctoast_qarray_amplitude ( size_t n, size_t m, size_t d, double const * v, double * norm );

void ctoast_qarray_normalize ( size_t n, size_t m, size_t d, double const * q_in, double * q_out );

void ctoast_qarray_normalize_inplace ( size_t n, size_t m, size_t d, double * q );

void ctoast_qarray_rotate ( size_t np, double const * q, size_t nv, double const * v_in, double * v_out );

void ctoast_qarray_mult ( size_t np, double const * p, size_t nq, double const * q, double * r );

void ctoast_qarray_slerp ( size_t n_time, size_t n_targettime, double const * time, double const * targettime, double const * q_in, double * q_interp );

void ctoast_qarray_exp ( size_t n, double const * q_in, double * q_out );

void ctoast_qarray_ln ( size_t n, double const * q_in, double * q_out );

void ctoast_qarray_pow ( size_t n, double const * p, double const * q_in, double * q_out );

void ctoast_qarray_from_axisangle ( size_t n, double const * axis, double const * angle, double * q_out );

void ctoast_qarray_to_axisangle ( size_t n, double const * q, double * axis, double * angle );

void ctoast_qarray_to_rotmat ( double const * q, double * rotmat );

void ctoast_qarray_from_rotmat ( const double * rotmat, double * q );

void ctoast_qarray_from_vectors ( double const * vec1, double const * vec2, double * q );

void ctoast_qarray_from_angles ( size_t n, double const * theta, double const * phi,
    double * const pa, double * quat, int IAU );

void ctoast_qarray_to_angles ( size_t n, double const * quat, double * theta,
    double * phi, double * pa, int IAU );

void ctoast_qarray_to_position ( size_t n, double const * quat, double * theta,
    double * phi );

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

int64_t ctoast_fft_r1d_length ( ctoast_fft_r1d * frd );
int64_t ctoast_fft_r1d_count ( ctoast_fft_r1d * frd );

double ** ctoast_fft_r1d_tdata ( ctoast_fft_r1d * frd );
void ctoast_fft_r1d_tdata_set ( ctoast_fft_r1d * frd, double ** data );
void ctoast_fft_r1d_tdata_get ( ctoast_fft_r1d * frd, double ** data );

double ** ctoast_fft_r1d_fdata ( ctoast_fft_r1d * frd );
void ctoast_fft_r1d_fdata_set ( ctoast_fft_r1d * frd, double ** data );
void ctoast_fft_r1d_fdata_get ( ctoast_fft_r1d * frd, double ** data );

void ctoast_fft_r1d_exec ( ctoast_fft_r1d * frd );

struct ctoast_fft_r1d_plan_store_;
typedef struct ctoast_fft_r1d_plan_store_ ctoast_fft_r1d_plan_store;

ctoast_fft_r1d_plan_store * ctoast_fft_r1d_plan_store_get ( );

void ctoast_fft_r1d_plan_store_clear ( ctoast_fft_r1d_plan_store * pstore );

void ctoast_fft_r1d_plan_store_cache ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n );

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_forward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n );

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_backward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n );

// Healpix

void ctoast_healpix_ang2vec ( int64_t n, double const * theta, double const * phi, double * vec );

void ctoast_healpix_vec2ang ( int64_t n, double const * vec, double * theta, double * phi );

void ctoast_healpix_vecs2angpa ( int64_t n, double const * vec, double * theta, double * phi, double * pa );

struct ctoast_healpix_pixels_;
typedef struct ctoast_healpix_pixels_ ctoast_healpix_pixels;

ctoast_healpix_pixels * ctoast_healpix_pixels_alloc ( int64_t nside );
void ctoast_healpix_pixels_free ( ctoast_healpix_pixels * hpix );

void ctoast_healpix_pixels_reset ( ctoast_healpix_pixels * hpix, int64_t nside );

void ctoast_healpix_pixels_vec2zphi ( ctoast_healpix_pixels * hpix, int64_t n, double const * vec, double * phi, int * region, double * z, double * rtz );

void ctoast_healpix_pixels_theta2z ( ctoast_healpix_pixels * hpix, int64_t n, double const * theta, int * region, double * z, double * rtz );

void ctoast_healpix_pixels_zphi2nest ( ctoast_healpix_pixels * hpix, int64_t n, double const * phi, int const * region, double const * z, double const * rtz, int64_t * pix );

void ctoast_healpix_pixels_zphi2ring ( ctoast_healpix_pixels * hpix, int64_t n, double const * phi, int const * region, double const * z, double const * rtz, int64_t * pix );

void ctoast_healpix_pixels_ang2nest ( ctoast_healpix_pixels * hpix, int64_t n, double const * theta, double const * phi, int64_t * pix );

void ctoast_healpix_pixels_ang2ring ( ctoast_healpix_pixels * hpix, int64_t n, double const * theta, double const * phi, int64_t * pix );

void ctoast_healpix_pixels_vec2nest ( ctoast_healpix_pixels * hpix, int64_t n, double const * vec, int64_t * pix );

void ctoast_healpix_pixels_vec2ring ( ctoast_healpix_pixels * hpix, int64_t n, double const * vec, int64_t * pix );

void ctoast_healpix_pixels_ring2nest ( ctoast_healpix_pixels * hpix, int64_t n, int64_t const * ringpix, int64_t * nestpix );

void ctoast_healpix_pixels_nest2ring ( ctoast_healpix_pixels * hpix, int64_t n, int64_t const * nestpix, int64_t * ringpix );

void ctoast_healpix_pixels_degrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix );

void ctoast_healpix_pixels_degrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix );

void ctoast_healpix_pixels_upgrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix );

void ctoast_healpix_pixels_upgrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix );

//--------------------------------------
// Operator helpers
//--------------------------------------

char ** ctoast_string_alloc ( size_t nstring, size_t length );

void ctoast_string_free ( size_t nstring, char ** str );

//--------------------------------------
// Atmosphere sub-library
//--------------------------------------

struct ctoast_atm_sim_;
typedef struct ctoast_atm_sim_ ctoast_atm_sim;

ctoast_atm_sim * ctoast_atm_sim_alloc ( double azmin, double azmax,
    double elmin, double elmax, double tmin, double tmax, double lmin_center,
    double lmin_sigma, double lmax_center, double lmax_sigma, double w_center,
    double w_sigma, double wdir_center, double wdir_sigma, double z0_center,
    double z0_sigma, double T0_center, double T0_sigma, double zatm,
    double zmax, double xstep, double ystep, double zstep, long nelem_sim_max,
    int verbosity, MPI_Comm comm, int gangsize, uint64_t key1, uint64_t key2,
    uint64_t counter1, uint64_t counter2, char *cachedir );

int ctoast_atm_sim_free ( ctoast_atm_sim * sim );

int ctoast_atm_sim_simulate( ctoast_atm_sim * sim, int use_cache );

int ctoast_atm_sim_observe( ctoast_atm_sim * sim, double *t, double *az,
    double *el, double *tod, long nsamp, double fixed_r );

double ctoast_atm_get_absorption_coefficient(double altitude,
					     double temperature,
					     double pressure,
					     double pwv,
					     double freq);
int ctoast_atm_get_absorption_coefficient_vec(double altitude,
					       double temperature,
					       double pressure,
					      double pwv,
					      double freqmin, double freqmax,
					      size_t nfreq, double *absorption);
double ctoast_atm_get_atmospheric_loading(double altitude,
					  double temperature,
					  double pressure,
					  double pwv,
					  double freq);
int ctoast_atm_get_atmospheric_loading_vec(double altitude,
					  double temperature,
					  double pressure,
					  double pwv,
					  double freqmin, double freqmax,
					  size_t nfreq, double *loading);


//--------------------------------------
// TOD sub-library
//--------------------------------------

void ctoast_pointing_healpix_matrix ( ctoast_healpix_pixels * hpix, int nest,
    double eps, double cal, char const * mode, size_t n, double const * pdata,
    double const * hwpang, uint8_t const * flags, int64_t * pixels,
    double * weights );

void ctoast_filter_polyfilter ( const long order, double **signals,
    uint8_t *flags, const size_t n, const size_t nsignal, const long *starts,
    const long *stops, const size_t nscan );

void ctoast_sim_map_scan_map32 (
    long *submap, long subnpix, double *weights, size_t nmap, long *subpix,
    float *map, double *tod, size_t nsamp );

void ctoast_sim_map_scan_map64 (
    long *submap, long subnpix, double *weights, size_t nmap, long *subpix,
    double *map, double *tod, size_t nsamp );

void ctoast_sim_noise_sim_noise_timestream (
    const uint64_t realization, const uint64_t telescope,
    const uint64_t component, const uint64_t obsindx, const uint64_t detindx,
    const double rate, const uint64_t firstsamp, const uint64_t samples,
    const uint64_t oversample, const double *freq, const double *psd,
    const uint64_t psdlen, double *noise );


//--------------------------------------
// FOD sub-library
//--------------------------------------

void ctoast_fod_autosums ( int64_t n, double const * x, uint8_t const * good, int64_t lagmax, double * sums, int64_t * hits );

//--------------------------------------
// Map sub-library
//--------------------------------------

void ctoast_cov_accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights,
    double scale, double const * signal, double * zdata, int64_t * hits, double * invnpp );

void ctoast_cov_accumulate_diagonal_hits ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, int64_t * hits );

void ctoast_cov_accumulate_diagonal_invnpp ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights,
    double scale, int64_t * hits, double * invnpp );

void ctoast_cov_accumulate_zmap ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights,
    double scale, double const * signal, double * zdata );

void ctoast_cov_eigendecompose_diagonal ( int64_t nsub, int64_t subsize,
    int64_t nnz, double * data, double * cond, double threshold,
    int32_t do_invert, int32_t do_rcond );

void ctoast_cov_multiply_diagonal ( int64_t nsub, int64_t subsize,
    int64_t nnz, double * data1, double const * data2 );

void ctoast_cov_apply_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double const * mat, double * vec );


//--------------------------------------
// Run test suite
//--------------------------------------

int ctoast_test_runner ( int argc, char *argv[] );


#ifdef __cplusplus
}
#endif

#endif
