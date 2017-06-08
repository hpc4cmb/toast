/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_tod_internal.hpp>

#include <sstream>
#include <iostream>



void toast::pointing::healpix_matrix ( toast::healpix::pixels const & hpix, 
    bool nest, double eps, double cal, std::string const & mode, size_t n,
    double const * pdata, double const * hwpang, uint8_t const * flags,
    int64_t * pixels, double * weights ) {

    double xaxis[3] = { 1.0, 0.0, 0.0 };
    double zaxis[3] = { 0.0, 0.0, 1.0 };
    double nullquat[4] = { 0.0, 0.0, 0.0, 1.0 };

    double eta = (1.0 - eps) / (1.0 + eps);

    double * dir = static_cast < double * > ( toast::mem::aligned_alloc ( 
        3 * n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    double * pin;

    if ( flags == NULL ) {

        // This is safe, since we are only passing the pointer to functions
        // which take a double const * .
        pin = const_cast < double * > ( pdata );
    
    } else {

        pin = static_cast < double * > ( toast::mem::aligned_alloc ( 
            4 * n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        size_t off;
        for ( size_t i = 0; i < n; ++i ) {
            off = 4 * i;
            if ( flags[i] == 0 ) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }

    }

    toast::qarray::rotate ( n, pin, 1, zaxis, dir );

    if ( nest ) {
        hpix.vec2nest ( n, dir, pixels );
    } else {
        hpix.vec2ring ( n, dir, pixels );
    }

    if ( flags != NULL ) {
        for ( size_t i = 0; i < n; ++i ) {
            pixels[i] = ( flags[i] == 0 ) ? pixels[i] : -1;
        }
    }

    if ( mode == "I" ) {

        for ( size_t i = 0; i < n; ++i ) {
            weights[i] = cal;
        }

    } else if ( mode == "IQU" ) {

        double * orient = static_cast < double * > ( toast::mem::aligned_alloc ( 
            3 * n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        double * buf1 = static_cast < double * > ( toast::mem::aligned_alloc ( 
            n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        double * buf2 = static_cast < double * > ( toast::mem::aligned_alloc ( 
            n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        toast::qarray::rotate ( n, pin, 1, xaxis, orient );

        double * bx = buf1;
        double * by = buf2;

        size_t off;
        for ( size_t i = 0; i < n; ++i ) {
            off = 3 * i;
            by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] * dir[off + 0];
            bx[i] = orient[off + 0] * ( -dir[off + 2] * dir[off + 0] ) + 
                orient[off + 1] * ( -dir[off + 2] * dir[off + 1] ) + 
                orient[off + 2] * ( dir[off + 0] * dir[off + 0] + 
                dir[off + 1] * dir[off + 1] );
        }

        toast::mem::aligned_free ( orient );

        double * detang = static_cast < double * > ( toast::mem::aligned_alloc ( 
            n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        toast::sf::fast_atan2 ( n, by, bx, detang );

        if ( hwpang != NULL ) {
            for ( size_t i = 0; i < n; ++i ) {
                detang[i] += 2.0 * hwpang[i];
                detang[i] *= 2.0;
            }
        }

        double * sinout = buf1;
        double * cosout = buf2;

        toast::sf::fast_sincos ( n, detang, sinout, cosout );

        for ( size_t i = 0; i < n; ++i ) {
            off = 3 * i;
            weights[off + 0] = cal;
            weights[off + 1] = cosout[i] * eta * cal;
            weights[off + 2] = sinout[i] * eta * cal;
        }

        toast::mem::aligned_free ( buf1 );
        toast::mem::aligned_free ( buf2 );
        toast::mem::aligned_free ( detang );

    } else {
        std::ostringstream o;
        o << "unknown healpix pointing matrix mode \"" << mode << "\"";
        TOAST_THROW( o.str().c_str() );
    }

    toast::mem::aligned_free ( dir );

    if ( flags != NULL ) {
        toast::mem::aligned_free ( pin );
    }

    return;
}


