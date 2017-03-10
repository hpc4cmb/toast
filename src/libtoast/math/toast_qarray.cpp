/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>
#include <cmath>

#include <sstream>


// Dot product of lists of arrays.

void toast::qarray::list_dot ( size_t n, size_t m, size_t d, double const * a, double const * b, double * dotprod ) {
    for ( size_t i = 0; i < n; ++i ) {
        dotprod[i] = 0.0;
        for ( size_t j = 0; j < d; ++j ) {
            dotprod[i] += a[m * i + j] * b[m * i + j];
        }
    }
    return;
}


// Inverse of a quaternion array.

void toast::qarray::inv ( size_t n, double * q ) {
    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < 3; ++j ) {
            q[4 * i + j] *= -1;
        }
    }
    return;
}


// Norm of quaternion array

void toast::qarray::amplitude ( size_t n, size_t m, size_t d, double const * v, double * norm ) {
    toast::qarray::list_dot ( n, m, d, v, v, norm );
    for ( size_t i = 0; i < n; ++i ) {
        norm[i] = ::sqrt ( norm[i] );
    }
    return;
}


// Normalize quaternion array.

void toast::qarray::normalize ( size_t n, size_t m, size_t d, double const * q_in, double * q_out ) {

    double * norm = static_cast < double * > ( toast::mem::aligned_alloc ( n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, m, d, q_in, norm );

    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < d; ++j ) {
            q_out[m * i + j] = q_in[m * i + j] / norm[i];
        }
    }

    toast::mem::aligned_free ( norm );
    return;
}


// Normalize quaternion array in place.

void toast::qarray::normalize_inplace ( size_t n, size_t m, size_t d, double * q ) {

    double * norm = static_cast < double * > ( toast::mem::aligned_alloc ( n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, m, d, q, norm );

    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < d; ++j ) {
            q[m * i + j] /= norm[i];
        }
    }

    toast::mem::aligned_free ( norm );
    return;
}


// Rotate an array of vectors by an array of quaternions.

void toast::qarray::rotate ( size_t n, double const * q, double const * v_in, double * v_out ) {

    double * q_unit = static_cast < double * > ( toast::mem::aligned_alloc ( 4 * n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::normalize ( n, 4, 4, q, q_unit );

    size_t vf;
    size_t qf;
    double xw, yw, zw, x2, xy, xz, y2, yz, z2;

    for ( size_t i = 0; i < n; ++i ) {
        vf = 3 * i;
        qf = 4 * i;
        xw =  q_unit[qf + 3] * q_unit[qf + 0];
        yw =  q_unit[qf + 3] * q_unit[qf + 1];
        zw =  q_unit[qf + 3] * q_unit[qf + 2];
        x2 = -q_unit[qf + 0] * q_unit[qf + 0];
        xy =  q_unit[qf + 0] * q_unit[qf + 1];
        xz =  q_unit[qf + 0] * q_unit[qf + 2];
        y2 = -q_unit[qf + 1] * q_unit[qf + 1];
        yz =  q_unit[qf + 1] * q_unit[qf + 2];
        z2 = -q_unit[qf + 2] * q_unit[qf + 2];

        v_out[vf + 0] = 2*( (y2 + z2) * v_in[vf + 0] + (xy - zw) * v_in[vf + 1] + (yw + xz) * v_in[vf + 2] ) + v_in[vf + 0];
        v_out[vf + 1] = 2*( (zw + xy) * v_in[vf + 0] + (x2 + z2) * v_in[vf + 1] + (yz - xw) * v_in[vf + 2] ) + v_in[vf + 1];
        v_out[vf + 2] = 2*( (xz - yw) * v_in[vf + 0] + (xw + yz) * v_in[vf + 1] + (x2 + y2) * v_in[vf + 2] ) + v_in[vf + 2];
    }

    toast::mem::aligned_free ( q_unit );
    return;
}


// Multiply arrays of quaternions.

void toast::qarray::mult ( size_t n, double const * p, double const * q, double * r ) {
    size_t qf;
    for ( size_t i = 0; i < n; ++i ) {
        qf = 4 * i;
        r[qf + 0] =  p[qf + 0] * q[qf + 3] + p[qf + 1] * q[qf + 2] 
            - p[qf + 2] * q[qf + 1] + p[qf + 3] * q[qf + 0];
        r[qf + 1] = -p[qf + 0] * q[qf + 2] + p[qf + 1] * q[qf + 3] 
            + p[qf + 2] * q[qf + 0] + p[qf + 3] * q[qf + 1];
        r[qf + 2] =  p[qf + 0] * q[qf + 1] - p[qf + 1] * q[qf + 0] 
            + p[qf + 2] * q[qf + 3] + p[qf + 3] * q[qf + 2];
        r[qf + 3] = -p[qf + 0] * q[qf + 0] - p[qf + 1] * q[qf + 1] 
            - p[qf + 2] * q[qf + 2] + p[qf + 3] * q[qf + 3];
    }
    return;
}


// Spherical interpolation of quaternion array from time to targettime.

void toast::qarray::slerp ( size_t n_time, size_t n_targettime, double const * time, double const * targettime, double const * q_in, double * q_interp ) {

    double diff;
    double frac;
    double costheta;
    double const * qlow;
    double const * qhigh;
    double theta;
    double invsintheta;
    double norm;
    double * q;
    double ratio1;
    double ratio2;

    size_t off = 0;

    for ( size_t i = 0; i < n_targettime; ++i ) {
        // scroll forward to the correct time sample
        while ( ( off+1 < n_time ) && ( time[off+1] < targettime[i] ) ) {
            ++off;
        }
        diff = time[off+1] - time[off];
        frac = ( targettime[i] - time[off] ) / diff;

        qlow = &( q_in[4*off] );
        qhigh = &( q_in[4*(off + 1)] );
        q = &( q_interp[4*i] );

        costheta = qlow[0] * qhigh[0] + qlow[1] * qhigh[1] + qlow[2] * qhigh[2] + qlow[3] * qhigh[3];
        
        if ( ::fabs ( costheta - 1.0 ) < 1.0e-10 ) {
            q[0] = qlow[0];
            q[1] = qlow[1];
            q[2] = qlow[2];
            q[3] = qlow[3];
        } else {
            theta = ::acos ( costheta );
            invsintheta = 1.0 / ::sqrt ( 1.0 - costheta * costheta );
            ratio1 = ::sin ( ( 1.0 - frac ) * theta ) * invsintheta;
            ratio2 = ::sin ( frac * theta ) * invsintheta;
            q[0] = ratio1 * qlow[0] + ratio2 * qhigh[0];
            q[1] = ratio1 * qlow[1] + ratio2 * qhigh[1];
            q[2] = ratio1 * qlow[2] + ratio2 * qhigh[2];
            q[3] = ratio1 * qlow[3] + ratio2 * qhigh[3];
        }
        
        norm = 1.0 / ::sqrt ( q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] );
        q[0] *= norm;
        q[1] *= norm;
        q[2] *= norm;
        q[3] *= norm;
    }

    return;
}


// Exponential of a quaternion array.

void toast::qarray::exp ( size_t n, double const * q_in, double * q_out ) {

    double * normv = static_cast < double * > ( toast::mem::aligned_alloc ( n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, 4, 3, q_in, normv );
    
    double exp_q_w;

    for ( size_t i = 0; i < n; ++i ) {
        exp_q_w = ::exp ( q_in[4*i + 3] );
        q_out[4*i + 3] = exp_q_w * ::cos ( normv[i] );
        exp_q_w /= normv[i];
        exp_q_w *= ::sin ( normv[i] );
        for ( size_t j = 0; j < 3; ++j ) {
            q_out[4*i + j] = exp_q_w * q_in[4*i + j];
        }
    }

    toast::mem::aligned_free ( normv );
    return;
}


// Natural logarithm of a quaternion array.

void toast::qarray::ln ( size_t n, double const * q_in, double * q_out ) {

    double * normq = static_cast < double * > ( toast::mem::aligned_alloc ( n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, 4, 4, q_in, normq );

    for ( size_t i = 0; i < n; ++i ) {
        q_out[4*i + 3] = ::log ( normq[i] );
    }

    toast::qarray::normalize ( n, 4, 3, q_in, q_out );

    double tmp;
    for ( size_t i = 0; i < n; ++i ) {
        tmp = ::acos ( q_in[4*i + 3] / normq[i] );
        for ( size_t j = 0; j < 3; ++j ) {
            q_out[4*i + j] *= tmp;
        }
    }

    toast::mem::aligned_free ( normq );
    return;
}


// Real power of quaternion array

void toast::qarray::pow ( size_t n, double const * p, double const * q_in, double * q_out ) {
    
    double * q_tmp = static_cast < double * > ( toast::mem::aligned_alloc ( n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::ln ( n, q_in, q_tmp );
    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < 4; ++j ) {
            q_tmp[4*i + j] *= p[i];
        }
    }

    toast::qarray::exp ( n, q_tmp, q_out );

    toast::mem::aligned_free ( q_tmp );
    return;
}


// Creates rotation quaternions of angles (in [rad]) around axes [already normalized]
// axis is an n by 3 array, angle is a n-array, q_out is a n by 4 array

void toast::qarray::from_axisangle ( size_t n, double const * axis, double const * angle, double * q_out ) {
    double sin_a;
    for ( size_t i = 0; i < n; ++i ) {
        sin_a = ::sin ( 0.5 * angle[i] );
        for ( size_t j = 0; j < 3; ++j ) {
            q_out[4*i + j] = axis[3*i + j] * sin_a;
        }
        q_out[4*i + 3] = ::cos ( 0.5 * angle[i] );
    }
    return;
}


//Returns the axis and angle of rotation of a quaternion.

void toast::qarray::to_axisangle ( size_t n, double const * q, double * axis, double * angle ) {

    size_t vf;
    size_t qf;
    double tmp;

    for ( size_t i = 0; i < n; ++i ) {
        qf = 4 * i;
        vf = 3 * i;
        angle[i] = 2.0 * ::acos ( q[qf+3] );
        if ( angle[i] < 1e-10 ) {
            axis[vf+0] = 0;
            axis[vf+1] = 0;
            axis[vf+2] = 0;
        } else {
            tmp = 1.0 / ::sin ( 0.5 * angle[i] );
            axis[vf+0] = q[qf+0] * tmp;
            axis[vf+1] = q[qf+1] * tmp;
            axis[vf+2] = q[qf+2] * tmp;
        }
    }
    return;
}


// Creates the rotation matrix corresponding to a quaternion.

void toast::qarray::to_rotmat ( double const * q, double * rotmat ) {
    double xx = q[0] * q[0];
    double xy = q[0] * q[1];
    double xz = q[0] * q[2];
    double xw = q[0] * q[3];
    double yy = q[1] * q[1];
    double yz = q[1] * q[2];
    double yw = q[1] * q[3];
    double zz = q[2] * q[2];
    double zw = q[2] * q[3];

    rotmat[0] = 1 - 2 * ( yy + zz );
    rotmat[1] =     2 * ( xy - zw );
    rotmat[2] =     2 * ( xz + yw );

    rotmat[3] =     2 * ( xy + zw );
    rotmat[4] = 1 - 2 * ( xx + zz );
    rotmat[5] =     2 * ( yz - xw );

    rotmat[6] =     2 * ( xz - yw );
    rotmat[7] =     2 * ( yz + xw );
    rotmat[8] = 1 - 2 * ( xx + yy );
    return;
}


// Creates the quaternion from a rotation matrix.

void toast::qarray::from_rotmat ( const double * rotmat, double * q ) {
    double tr = rotmat[0] + rotmat[4] + rotmat[8];
    double S;
    double invS;
    if ( tr > 0 ) { 
        S = ::sqrt ( tr + 1.0 ) * 2.0; /* S=4*qw */
        invS = 1.0 / S;
        q[0] = (rotmat[7] - rotmat[5]) * invS;
        q[1] = (rotmat[2] - rotmat[6]) * invS;
        q[2] = (rotmat[3] - rotmat[1]) * invS; 
        q[3] = 0.25 * S;
    } else if ( ( rotmat[0] > rotmat[4] ) && ( rotmat[0] > rotmat[8] ) ) { 
        S = ::sqrt ( 1.0 + rotmat[0] - rotmat[4] - rotmat[8] ) * 2.0; /* S=4*qx */
        invS = 1.0 / S;
        q[0] = 0.25 * S;
        q[1] = (rotmat[1] + rotmat[3]) * invS; 
        q[2] = (rotmat[2] + rotmat[6]) * invS; 
        q[3] = (rotmat[7] - rotmat[5]) * invS;
    } else if ( rotmat[4] > rotmat[8] ) { 
        S = ::sqrt ( 1.0 + rotmat[4] - rotmat[0] - rotmat[8] ) * 2.0; /* S=4*qy */
        invS = 1.0 / S;
        q[0] = (rotmat[1] + rotmat[3]) * invS; 
        q[1] = 0.25 * S;
        q[2] = (rotmat[5] + rotmat[7]) * invS; 
        q[3] = (rotmat[2] - rotmat[6]) * invS;
    } else {
        S = ::sqrt ( 1.0 + rotmat[8] - rotmat[0] - rotmat[4] ) * 2.0; /* S=4*qz */
        invS = 1.0 / S;
        q[0] = (rotmat[2] + rotmat[6]) * invS;
        q[1] = (rotmat[5] + rotmat[7]) * invS;
        q[2] = 0.25 * S;
        q[3] = (rotmat[3] - rotmat[1]) * invS;
    }
    return;
}


// Creates the quaternion from two normalized vectors.

void toast::qarray::from_vectors ( double const * vec1, double const * vec2, double * q ) {
    double dotprod = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    double vec1prod = vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2];
    double vec2prod = vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2];

    // shortcut for coincident vectors
    if ( ::fabs ( dotprod ) < 1.0e-12 ) {
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 1.0;
    } else {
        q[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        q[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        q[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
        q[3] = ::sqrt ( vec1prod * vec2prod ) + dotprod;
        toast::qarray::normalize_inplace ( 1, 4, 4, q );
    }

    return;
}


