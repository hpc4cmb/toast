/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#ifdef HAVE_MKL
#  include <mkl.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <cmath>

#ifdef HAVE_MKL

// These call MKL VM functions with "High Accuracy" mode.

void toast::sf::sin ( int n, double const * ang, double * sinout ) {
    vmdSin ( n, ang, sinout, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::cos ( int n, double const * ang, double * cosout ) {
    vmdCos ( n, ang, cosout, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::sincos ( int n, double const * ang, double * sinout, double * cosout ) {
    vmdSinCos ( n, ang, sinout, cosout, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::atan2 ( int n, double const * y, double const * x, double * ang ) {
    vmdAtan2 ( n, y, x, ang, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::sqrt ( int n, double const * in, double * out ) {
    vmdSqrt ( n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::rsqrt ( int n, double const * in, double * out ) {
    vmdInvSqrt ( n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::exp ( int n, double const * in, double * out ) {
    vmdExp ( n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::log ( int n, double const * in, double * out ) {
    vmdLn ( n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}


// These call MKL VM functions with "Low Accuracy" mode.

void toast::sf::fast_sin ( int n, double const * ang, double * sinout ) {
    vmdSin ( n, ang, sinout, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_cos ( int n, double const * ang, double * cosout ) {
    vmdCos ( n, ang, cosout, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_sincos ( int n, double const * ang, double * sinout, double * cosout ) {
    vmdSinCos ( n, ang, sinout, cosout, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_atan2 ( int n, double const * y, double const * x, double * ang ) {
    vmdAtan2 ( n, y, x, ang, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_sqrt ( int n, double const * in, double * out ) {
    vmdSqrt ( n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_rsqrt ( int n, double const * in, double * out ) {
    vmdInvSqrt ( n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_exp ( int n, double const * in, double * out ) {
    vmdExp ( n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

void toast::sf::fast_log ( int n, double const * in, double * out ) {
    vmdLn ( n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
    return;
}

#else

// These are simply threaded for-loops that call the standard
// math library functions.

void toast::sf::sin ( int n, double const * ang, double * sinout ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, ang, sinout) schedule(static)
    for ( i = 0; i < n; ++i ) {
        sinout[i] = ::sin ( ang[i] );
    }
    return;
}

void toast::sf::cos ( int n, double const * ang, double * cosout ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, ang, cosout) schedule(static)
    for ( i = 0; i < n; ++i ) {
        cosout[i] = ::cos ( ang[i] );
    }
    return;
}

void toast::sf::sincos ( int n, double const * ang, double * sinout, double * cosout ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, ang, sinout, cosout) schedule(static)
    for ( i = 0; i < n; ++i ) {
        sinout[i] = ::sin ( ang[i] );
        cosout[i] = ::cos ( ang[i] );
    }
    return;
}

void toast::sf::atan2 ( int n, double const * y, double const * x, double * ang ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, x, y, ang) schedule(static)
    for ( i = 0; i < n; ++i ) {
        ang[i] = ::atan2 ( y[i], x[i] );
    }
    return;
}

void toast::sf::sqrt ( int n, double const * in, double * out ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, in, out) schedule(static)
    for ( i = 0; i < n; ++i ) {
        out[i] = ::sqrt ( in[i] );
    }
    return;
}

void toast::sf::rsqrt ( int n, double const * in, double * out ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, in, out) schedule(static)
    for ( i = 0; i < n; ++i ) {
        out[i] = 1.0 / ::sqrt ( in[i] );
    }
    return;
}

void toast::sf::exp ( int n, double const * in, double * out ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, in, out) schedule(static)
    for ( i = 0; i < n; ++i ) {
        out[i] = ::exp ( in[i] );
    }
    return;
}

void toast::sf::log ( int n, double const * in, double * out ) {
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, in, out) schedule(static)
    for ( i = 0; i < n; ++i ) {
        out[i] = ::log ( in[i] );
    }
    return;
}


// These use polynomial approximations for some functions.

void toast::sf::fast_sin ( int n, double const * ang, double * sinout ) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double sx;
    double sx2;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    #pragma omp parallel for default(none) private(i, sx, sx2, quot, rem, x, quad) shared(n, ang, sinout) schedule(static)
    for ( i = 0; i < n; i++ ) {
        quot = ang[i] * INV_TWOPI;
        rem = quot - floor ( quot );
        x = rem * TWOPI;
        while ( x < 0.0 ) {
            x += TWOPI;
        }
        quad = static_cast < int > ( x * TWOINVPI );
        switch ( quad ) {
            case 1:
                sx = x - PI_2;
                sx2 = sx * sx;
                sinout[i] = (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
            case 2:
                sx = THREEPI_2 - x;
                sx2 = sx * sx;
                sinout[i] = - (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
            case 3:
                sx = x - THREEPI_2;
                sx2 = sx * sx;
                sinout[i] = - (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
            default:
                sx = PI_2 - x;
                sx2 = sx * sx;
                sinout[i] = (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
        }
    }
    return;
}

void toast::sf::fast_cos ( int n, double const * ang, double * cosout ) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double cx;
    double cx2;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    #pragma omp parallel for default(none) private(i, cx, cx2, quot, rem, x, quad) shared(n, ang, cosout) schedule(static)
    for ( i = 0; i < n; i++ ) {
        quot = ang[i] * INV_TWOPI;
        rem = quot - floor ( quot );
        x = rem * TWOPI;
        while ( x < 0.0 ) {
            x += TWOPI;
        }
        quad = static_cast < int > ( x * TWOINVPI );
        switch ( quad ) {
            case 1:
                cx = PI - x;
                cx2 = cx * cx;
                cosout[i] = - (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                break;
            case 2:
                cx = x - PI;
                cx2 = cx * cx;
                cosout[i] = - (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                break;
            case 3:
                cx = TWOPI - x;
                cx2 = cx * cx;
                cosout[i] = (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                break;
            default:
                cx2 = x * x;
                cosout[i] = (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                break;
        }
    }
    return;
}

void toast::sf::fast_sincos ( int n, double const * ang, double * sinout, double * cosout ) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double sx, cx;
    double sx2, cx2;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    #pragma omp parallel for default(none) private(i, sx, cx, sx2, cx2, quot, rem, x, quad) shared(n, ang, sinout, cosout) schedule(static)
    for ( i = 0; i < n; i++ ) {
        quot = ang[i] * INV_TWOPI;
        rem = quot - floor ( quot );
        x = rem * TWOPI;
        while ( x < 0.0 ) {
            x += TWOPI;
        }
        quad = static_cast < int > ( x * TWOINVPI );
        switch ( quad ) {
            case 1:
                sx = x - PI_2;
                cx = PI - x;
                cx2 = cx * cx;
                sx2 = sx * sx;
                cosout[i] = - (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                sinout[i] = (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
            case 2:
                sx = THREEPI_2 - x;
                cx = x - PI;
                cx2 = cx * cx;
                sx2 = sx * sx;
                cosout[i] = - (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                sinout[i] = - (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
            case 3:
                sx = x - THREEPI_2;
                cx = TWOPI - x;
                cx2 = cx * cx;
                sx2 = sx * sx;
                cosout[i] = (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                sinout[i] = - (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
            default:
                sx = PI_2 - x;
                cx2 = x * x;
                sx2 = sx * sx;
                cosout[i] = (SC1 + cx2 * (SC2 + cx2 * (SC3 + cx2 * (SC4 + cx2 * (SC5 + cx2 * (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
                sinout[i] = (SC1 + sx2 * (SC2 + sx2 * (SC3 + sx2 * (SC4 + sx2 * (SC5 + sx2 * (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
                break;
        }
    }
    return;
}

void toast::sf::fast_atan2 ( int n, double const * y, double const * x, double * ang ) {
    double const ATCHEB1 = 48.70107004404898384;
    double const ATCHEB2 = 49.5326263772254345;
    double const ATCHEB3 = 9.40604244231624;
    double const ATCHEB4 = 48.70107004404996166;
    double const ATCHEB5 = 65.7663163908956299;
    double const ATCHEB6 = 21.587934067020262;
    int i;
    double r2;
    double r;
    int complement;
    int region;
    int sign;

    #pragma omp parallel for default(none) private(i, r, r2, complement, region, sign) shared(n, ang, x, y) schedule(static)
    for ( i = 0; i < n; i++ ) {

        r = y[i] / x[i];

        // reduce range to PI/12

        complement = 0;
        region = 0;
        sign = 0;
        if ( r < 0 ) {
            r = -r;
            sign = 1;
        }
        if ( r > 1.0 ) {
            r = 1.0 / r;
            complement = 1;
        }
        if ( r > TANTWELFTHPI ) {
            r = ( r - TANSIXTHPI ) / ( 1 + TANSIXTHPI * r );
            region = 1;
        }
        r2 = r * r;
        r = (r * (ATCHEB1 + r2 * (ATCHEB2 + r2 * ATCHEB3))) / (ATCHEB4 + r2 * (ATCHEB5 + r2 * (ATCHEB6 + r2)));
        if ( region ) {
            r += SIXTHPI;
        }
        if ( complement ) {
            r = PI_2 - r;
        }
        if ( sign ) {
            r = -r;
        }

        // adjust quadrant

        if ( x[i] > 0.0 ) {
            ang[i] = r;
        } else if ( x[i] < 0.0 ) {
            ang[i] = r + PI;
            if ( ang[i] > PI ) {
                ang[i] -= TWOPI;
            }
        } else if ( y[i] > 0.0 ) {
            ang[i] = PI_2;
        } else if ( y[i] < 0.0 ) {
            ang[i] = -PI_2;
        } else {
            ang[i] = 0.0;
        }
    }
    return;
}

void toast::sf::fast_sqrt ( int n, double const * in, double * out ) {
    toast::sf::sqrt ( n, in, out );
    return;
}

void toast::sf::fast_rsqrt ( int n, double const * in, double * out ) {
    toast::sf::rsqrt ( n, in, out );
    return;
}

void toast::sf::fast_exp ( int n, double const * in, double * out ) {
    toast::sf::exp ( n, in, out );
    return;
}

void toast::sf::fast_log ( int n, double const * in, double * out ) {
    toast::sf::log ( n, in, out );
    return;
}



#endif


