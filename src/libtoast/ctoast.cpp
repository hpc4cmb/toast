/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <ctoast.hpp>


//--------------------------------------
// Math sub-library
//--------------------------------------

// special functions

void ctoast_sf_sin ( int n, double * ang, double * sinout ) {
    toast::sf::sin ( n, ang, sinout );
    return;
}

void ctoast_sf_cos ( int n, double * ang, double * cosout ) {
    toast::sf::cos ( n, ang, cosout );
    return;
}

void ctoast_sf_sincos ( int n, double * ang, double * sinout, double * cosout ) {
    toast::sf::sincos ( n, ang, sinout, cosout );
    return;
}

void ctoast_sf_atan2 ( int n, double * y, double * x, double * ang ) {
    toast::sf::atan2 ( n, y, x, ang );
    return;
}

void ctoast_sf_sqrt ( int n, double * in, double * out ) {
    toast::sf::sqrt ( n, in, out );
    return;
}

void ctoast_sf_rsqrt ( int n, double * in, double * out ) {
    toast::sf::rsqrt ( n, in, out );
    return;
}

void ctoast_sf_exp ( int n, double * in, double * out ) {
    toast::sf::exp ( n, in, out );
    return;
}

void ctoast_sf_log ( int n, double * in, double * out ) {
    toast::sf::log ( n, in, out );
    return;
}

void ctoast_sf_fast_sin ( int n, double * ang, double * sinout ) {
    toast::sf::fast_sin ( n, ang, sinout );
    return;
}

void ctoast_sf_fast_cos ( int n, double * ang, double * cosout ) {
    toast::sf::fast_cos ( n, ang, cosout );
    return;
}

void ctoast_sf_fast_sincos ( int n, double * ang, double * sinout, double * cosout ) {
    toast::sf::fast_sincos ( n, ang, sinout, cosout );
    return;
}

void ctoast_sf_fast_atan2 ( int n, double * y, double * x, double * ang ) {
    toast::sf::fast_atan2 ( n, y, x, ang );
    return;
}

void ctoast_sf_fast_sqrt ( int n, double * in, double * out ) {
    toast::sf::fast_sqrt ( n, in, out );
    return;
}

void ctoast_sf_fast_rsqrt ( int n, double * in, double * out ) {
    toast::sf::fast_rsqrt ( n, in, out );
    return;
}

void ctoast_sf_fast_exp ( int n, double * in, double * out ) {
    toast::sf::fast_exp ( n, in, out );
    return;
}

void ctoast_sf_fast_log ( int n, double * in, double * out ) {
    toast::sf::fast_log ( n, in, out );
    return;
}


// RNG

void ctoast_rng_dist_uint64 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data ) {
    toast::rng::dist_uint64 ( n, key1, key2, counter1, counter2, data );
    return;
}

void ctoast_rng_dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data ) {
    toast::rng::dist_uniform_01 ( n, key1, key2, counter1, counter2, data );
    return;
}

void ctoast_rng_dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data ) {
    toast::rng::dist_uniform_11 ( n, key1, key2, counter1, counter2, data );
    return;
}

void ctoast_rng_dist_normal ( size_t n, uint64_t key1, uint64_t key2, uint64_t counter1, uint64_t counter2, uint64_t * data ) {
    toast::rng::dist_normal ( n, key1, key2, counter1, counter2, data );
    return;
}

// Quaternion array

void ctoast_qarray_list_dot ( size_t n, size_t m, size_t d, double const * a, double const * b, double * dotprod ) {
    toast::qarray::list_dot ( n, m, d, a, b, dotprod );
    return;
}

void ctoast_qarray_inv ( size_t n, double * q ) {
    toast::qarray::inv ( n, q );
    return;
}

void ctoast_qarray_amplitude ( size_t n, size_t m, size_t d, double const * v, double * norm ) {
    toast::qarray::amplitude ( n, m, d, v, norm );
    return;
}

void ctoast_qarray_normalize ( size_t n, size_t m, size_t d, double const * q_in, double * q_out ) {
    toast::qarray::normalize ( n, m, d, q_in, q_out );
    return;
}

void ctoast_qarray_normalize_inplace ( size_t n, size_t m, size_t d, double * q ) {
    toast::qarray::normalize_inplace ( n, m, d, q );
    return;
}

void ctoast_qarray_rotate ( size_t n, double const * q, double const * v_in, double * v_out ) {
    toast::qarray::rotate ( n, q, v_in, v_out );
    return;
}

void ctoast_qarray_mult ( size_t n, double const * p, double const * q, double * r ) {
    toast::qarray::mult ( n, p, q, r );
    return;
}

void ctoast_qarray_slerp ( size_t n_time, size_t n_targettime, double const * time, double const * targettime, double const * q_in, double * q_interp ) {
    toast::qarray::slerp ( n_time, n_targettime, time, targettime, q_in, q_interp );
    return;
}

void ctoast_qarray_exp ( size_t n, double const * q_in, double * q_out ) {
    toast::qarray::exp ( n, q_in, q_out );
    return;
}

void ctoast_qarray_ln ( size_t n, double const * q_in, double * q_out ) {
    toast::qarray::ln ( n, q_in, q_out );
    return;
}

void ctoast_qarray_pow ( size_t n, double const * p, double const * q_in, double * q_out ) {
    toast::qarray::pow ( n, p, q_in, q_out );
    return;
}

void ctoast_qarray_from_axisangle ( size_t n, double const * axis, double const * angle, double * q_out ) {
    toast::qarray::from_axisangle ( n, axis, angle, q_out );
    return;
}

void ctoast_qarray_to_axisangle ( size_t n, double const * q, double * axis, double * angle ) {
    toast::qarray::to_axisangle ( n, q, axis, angle );
    return;
}

void ctoast_qarray_to_rotmat ( double const * q, double * rotmat ) {
    toast::qarray::to_rotmat ( q, rotmat );
    return;
}

void ctoast_qarray_from_rotmat ( const double * rotmat, double * q ) {
    toast::qarray::from_rotmat ( rotmat, q );
    return;
}

void ctoast_qarray_from_vectors ( double const * vec1, double const * vec2, double * q ) {
    toast::qarray::from_vectors ( vec1, vec2, q );
    return;
}

// FFT

toast::fft::plan_type ctoast::convert_from_c ( ctoast_fft_plan_type in ) {
    toast::fft::plan_type ret;
    switch ( in ) {
        case PLAN_FAST :
            ret = toast::fft::plan_type::fast;
            break;
        case PLAN_BEST :
            ret = toast::fft::plan_type::best;
            break;
        default :
            TOAST_THROW( "invalid ctoast_fft_plan_type value" );
            break;
    }
    return ret;
}

ctoast_fft_plan_type ctoast::convert_to_c ( toast::fft::plan_type in ) {
    ctoast_fft_plan_type ret;
    switch ( in ) {
        case toast::fft::plan_type::fast :
            ret = PLAN_FAST;
            break;
        case toast::fft::plan_type::best :
            ret = PLAN_BEST;
            break;
        default :
            TOAST_THROW( "invalid toast::fft::plan_type value" );
            break;
    }
    return ret;
}

toast::fft::direction ctoast::convert_from_c ( ctoast_fft_direction in ) {
    toast::fft::direction ret;
    switch ( in ) {
        case FORWARD :
            ret = toast::fft::direction::forward;
            break;
        case BACKWARD :
            ret = toast::fft::direction::backward;
            break;
        default :
            TOAST_THROW( "invalid ctoast_fft_direction value" );
            break;
    }
    return ret;
}

ctoast_fft_direction ctoast::convert_to_c ( toast::fft::direction in ) {
    ctoast_fft_direction ret;
    switch ( in ) {
        case toast::fft::direction::forward :
            ret = FORWARD;
            break;
        case toast::fft::direction::backward :
            ret = BACKWARD;
            break;
        default :
            TOAST_THROW( "invalid toast::fft::direction value" );
            break;
    }
    return ret;
}

ctoast_fft_r1d * ctoast_fft_r1d_alloc ( int64_t length, int64_t n, ctoast_fft_plan_type type, ctoast_fft_direction dir, double scale ) {
    toast::fft::direction tdir = ctoast::convert_from_c ( dir );
    toast::fft::plan_type ttype = ctoast::convert_from_c ( type );
    return reinterpret_cast < ctoast_fft_r1d * > ( toast::fft::r1d::create ( length, n, ttype, tdir, scale ) );
}

void ctoast_fft_r1d_free ( ctoast_fft_r1d * frd ) {
    delete reinterpret_cast < toast::fft::r1d * > ( frd );
    return;
}

double ** ctoast_fft_r1d_tdata ( ctoast_fft_r1d * frd ) {
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    std::vector < double * > tdata = rd->tdata();
    double ** ret = (double**) malloc ( tdata.size() * sizeof(double*) );
    if ( ret == NULL ) {
        TOAST_THROW( "Cannot allocate fft_r1d_tdata buffer" );
    }
    for ( size_t i = 0; i < tdata.size(); ++i ) {
        ret[i] = tdata[i];
    }
    return ret;
}

double ** ctoast_fft_r1d_fdata ( ctoast_fft_r1d * frd ) {
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    std::vector < double * > fdata = rd->fdata();
    double ** ret = (double**) malloc ( fdata.size() * sizeof(double*) );
    if ( ret == NULL ) {
        TOAST_THROW( "Cannot allocate fft_r1d_fdata buffer" );
    }
    for ( size_t i = 0; i < fdata.size(); ++i ) {
        ret[i] = fdata[i];
    }
    return ret;
}

void ctoast_fft_r1d_exec ( ctoast_fft_r1d * frd ) {
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    rd->exec();
    return;
}

ctoast_fft_r1d_plan_store * ctoast_fft_r1d_plan_store_get ( ) {
    toast::fft::r1d_plan_store & st = toast::fft::r1d_plan_store::get();
    return reinterpret_cast < ctoast_fft_r1d_plan_store * > ( &st );
}

void ctoast_fft_r1d_plan_store_clear ( ctoast_fft_r1d_plan_store * pstore ) {
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    st->clear();
    return;
}

void ctoast_fft_r1d_plan_store_cache ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n ) {
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    st->cache ( len, n );
    return;
}

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_forward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n ) {
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    toast::fft::r1d_p rdp = st->forward ( len, n );
    toast::fft::r1d * rd = rdp.get();
    return reinterpret_cast < ctoast_fft_r1d * > ( rd );
}

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_backward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n ) {
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    toast::fft::r1d_p rdp = st->backward ( len, n );
    toast::fft::r1d * rd = rdp.get();
    return reinterpret_cast < ctoast_fft_r1d * > ( rd );
}

// Healpix

ctoast_healpix_pixels * ctoast_healpix_pixels_alloc ( int64_t nside ) {
    return reinterpret_cast < ctoast_healpix_pixels * > ( new toast::healpix::pixels ( nside ) );
}

void ctoast_healpix_pixels_free ( ctoast_healpix_pixels * hpix ) {
    delete reinterpret_cast < toast::healpix::pixels * > ( hpix );
    return;
}

void ctoast_healpix_pixels_reset ( ctoast_healpix_pixels * hpix, int64_t nside ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->reset ( nside );
    return;
}

void ctoast_healpix_pixels_vec2zphi ( ctoast_healpix_pixels * hpix, int64_t n, double * vec, double * phi, int * region, double * z, double * rtz ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->vec2zphi ( n, vec, phi, region, z, rtz );
    return;
}

void ctoast_healpix_pixels_theta2z ( ctoast_healpix_pixels * hpix, int64_t n, double * theta, int * region, double * z, double * rtz ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->theta2z ( n, theta, region, z, rtz );
    return;
}

void ctoast_healpix_pixels_zphi2nest ( ctoast_healpix_pixels * hpix, int64_t n, double * phi, int * region, double * z, double * rtz, int64_t * pix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->zphi2nest ( n, phi, region, z, rtz, pix );
    return;
}

void ctoast_healpix_pixels_zphi2ring ( ctoast_healpix_pixels * hpix, int64_t n, double * phi, int * region, double * z, double * rtz, int64_t * pix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->zphi2ring ( n, phi, region, z, rtz, pix );
    return;
}

void ctoast_healpix_pixels_ang2nest ( ctoast_healpix_pixels * hpix, int64_t n, double * theta, double * phi, int64_t * pix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->ang2nest ( n, theta, phi, pix );
    return;
}

void ctoast_healpix_pixels_ang2ring ( ctoast_healpix_pixels * hpix, int64_t n, double * theta, double * phi, int64_t * pix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->ang2ring ( n, theta, phi, pix );
    return;
}

void ctoast_healpix_pixels_vec2nest ( ctoast_healpix_pixels * hpix, int64_t n, double * vec, int64_t * pix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->vec2nest ( n, vec, pix );
    return;
}

void ctoast_healpix_pixels_vec2ring ( ctoast_healpix_pixels * hpix, int64_t n, double * vec, int64_t * pix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->vec2ring ( n, vec, pix );
    return;
}

void ctoast_healpix_pixels_ring2nest ( ctoast_healpix_pixels * hpix, int64_t n, int64_t * ringpix, int64_t * nestpix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->ring2nest ( n, ringpix, nestpix );
    return;
}

void ctoast_healpix_pixels_nest2ring ( ctoast_healpix_pixels * hpix, int64_t n, int64_t * nestpix, int64_t * ringpix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->nest2ring ( n, nestpix, ringpix );
    return;
}

void ctoast_healpix_pixels_degrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->degrade_ring ( factor, n, inpix, outpix );
    return;
}

void ctoast_healpix_pixels_degrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->degrade_nest ( factor, n, inpix, outpix );
    return;
}

void ctoast_healpix_pixels_upgrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->upgrade_ring ( factor, n, inpix, outpix );
    return;
}

void ctoast_healpix_pixels_upgrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t * inpix, int64_t * outpix ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->upgrade_nest ( factor, n, inpix, outpix );
    return;
}


//--------------------------------------
// TOD sub-library
//--------------------------------------



//--------------------------------------
// Map functions
//--------------------------------------






