/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_math_internal.hpp>

#include <cstring>
#include <cmath>

#include <sstream>
#include <iostream>

#ifdef _OPENMP
#  include <omp.h>
#endif


// Fixed length at which we have enough work to justify using threads.
const static int toast_qarray_ompthresh = 100;


// Dot product of lists of arrays.

void toast::qarray::list_dot ( size_t n, size_t m, size_t d, double const * a, 
    double const * b, double * dotprod ) {

    int nt = 1;
    #ifdef _OPENMP
    nt = omp_get_num_threads();
    #endif
    
    if ( n < toast_qarray_ompthresh * nt ) {
        for ( size_t i = 0; i < n; ++i ) {
            dotprod[i] = 0.0;
            for ( size_t j = 0; j < d; ++j ) {
                dotprod[i] += a[m * i + j] * b[m * i + j];
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            dotprod[i] = 0.0;
            for ( size_t j = 0; j < d; ++j ) {
                dotprod[i] += a[m * i + j] * b[m * i + j];
            }
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
    
    double * temp = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::list_dot ( n, m, d, v, v, temp );

    toast::sf::sqrt ( n, temp, norm );

    toast::mem::aligned_free ( temp );

    return;
}


// Normalize quaternion array.

void toast::qarray::normalize ( size_t n, size_t m, size_t d, double const * q_in, double * q_out ) {

    double * norm = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, m, d, q_in, norm );

    int nt = 1;
    #ifdef _OPENMP
    nt = omp_get_num_threads();
    #endif

    if ( n < toast_qarray_ompthresh * nt ) {
        for ( size_t i = 0; i < n; ++i ) {
            for ( size_t j = 0; j < d; ++j ) {
                q_out[m * i + j] = q_in[m * i + j] / norm[i];
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            for ( size_t j = 0; j < d; ++j ) {
                q_out[m * i + j] = q_in[m * i + j] / norm[i];
            }
        }
    }

    toast::mem::aligned_free ( norm );

    return;
}


// Normalize quaternion array in place.

void toast::qarray::normalize_inplace ( size_t n, size_t m, size_t d, double * q ) {

    double * norm = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, m, d, q, norm );

    int nt = 1;
    #ifdef _OPENMP
    nt = omp_get_num_threads();
    #endif

    if ( n < toast_qarray_ompthresh * nt ) {
        for ( size_t i = 0; i < n; ++i ) {
            for ( size_t j = 0; j < d; ++j ) {
                q[m * i + j] /= norm[i];
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            for ( size_t j = 0; j < d; ++j ) {
                q[m * i + j] /= norm[i];
            }
        }
    }

    toast::mem::aligned_free ( norm );

    return;
}


// Rotate an array of vectors by an array of quaternions.

void toast::qarray::rotate ( size_t nq, double const * q, size_t nv, double const * v_in, double * v_out ) {

    size_t n = nq;
    if ( nv > n ) {
        n = nv;
    }

    double * q_unit = static_cast < double * > ( toast::mem::aligned_alloc ( 
        4 * nq * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::normalize ( nq, 4, 4, q, q_unit );

    size_t i;
    size_t vf;
    size_t vfin;
    size_t vfout;
    size_t qf;
    double xw, yw, zw, x2, xy, xz, y2, yz, z2;

    int nt = 1;
    #ifdef _OPENMP
    nt = omp_get_num_threads();
    #endif

    // this is to avoid branching inside the for loop.
    size_t chv;
    size_t chq;
    if ( nv == 1 ) {
        chv = 0;
    } else {
        chv = 1;
    }
    if ( nq == 1 ) {
        chq = 0;
    } else {
        chq = 1;
    }

    if ( n < toast_qarray_ompthresh * nt ) {
        if ( nq == 1 ) {
            xw =  q_unit[3] * q_unit[0];
            yw =  q_unit[3] * q_unit[1];
            zw =  q_unit[3] * q_unit[2];
            x2 = -q_unit[0] * q_unit[0];
            xy =  q_unit[0] * q_unit[1];
            xz =  q_unit[0] * q_unit[2];
            y2 = -q_unit[1] * q_unit[1];
            yz =  q_unit[1] * q_unit[2];
            z2 = -q_unit[2] * q_unit[2];
            for ( i = 0; i < n; ++i ) {
                vfin = 3 * i * chv;
                vfout = 3 * i;
                v_out[vfout + 0] = 2*( (y2 + z2) * v_in[vfin + 0] + 
                    (xy - zw) * v_in[vfin + 1] + (yw + xz) * v_in[vfin + 2] ) 
                    + v_in[vfin + 0];
                v_out[vfout + 1] = 2*( (zw + xy) * v_in[vfin + 0] + 
                    (x2 + z2) * v_in[vfin + 1] + (yz - xw) * v_in[vfin + 2] ) 
                    + v_in[vfin + 1];
                v_out[vfout + 2] = 2*( (xz - yw) * v_in[vfin + 0] + 
                    (xw + yz) * v_in[vfin + 1] + (x2 + y2) * v_in[vfin + 2] ) 
                    + v_in[vfin + 2];
            }
        } else {
            for ( i = 0; i < n; ++i ) {
                vfin = 3 * i * chv;
                vfout = 3 * i;
                qf = 4 * i * chq;
                xw =  q_unit[qf + 3] * q_unit[qf + 0];
                yw =  q_unit[qf + 3] * q_unit[qf + 1];
                zw =  q_unit[qf + 3] * q_unit[qf + 2];
                x2 = -q_unit[qf + 0] * q_unit[qf + 0];
                xy =  q_unit[qf + 0] * q_unit[qf + 1];
                xz =  q_unit[qf + 0] * q_unit[qf + 2];
                y2 = -q_unit[qf + 1] * q_unit[qf + 1];
                yz =  q_unit[qf + 1] * q_unit[qf + 2];
                z2 = -q_unit[qf + 2] * q_unit[qf + 2];

                v_out[vfout + 0] = 2*( (y2 + z2) * v_in[vfin + 0] + 
                    (xy - zw) * v_in[vfin + 1] + (yw + xz) * v_in[vfin + 2] ) 
                    + v_in[vfin + 0];
                v_out[vfout + 1] = 2*( (zw + xy) * v_in[vfin + 0] + 
                    (x2 + z2) * v_in[vfin + 1] + (yz - xw) * v_in[vfin + 2] ) 
                    + v_in[vfin + 1];
                v_out[vfout + 2] = 2*( (xz - yw) * v_in[vfin + 0] + 
                    (xw + yz) * v_in[vfin + 1] + (x2 + y2) * v_in[vfin + 2] ) 
                    + v_in[vfin + 2];
            }
        }
    } else {
        if ( nq == 1 ) {
            xw =  q_unit[3] * q_unit[0];
            yw =  q_unit[3] * q_unit[1];
            zw =  q_unit[3] * q_unit[2];
            x2 = -q_unit[0] * q_unit[0];
            xy =  q_unit[0] * q_unit[1];
            xz =  q_unit[0] * q_unit[2];
            y2 = -q_unit[1] * q_unit[1];
            yz =  q_unit[1] * q_unit[2];
            z2 = -q_unit[2] * q_unit[2];
            #pragma omp parallel for default(shared) private(i, vfin, vfout) schedule(static)
            for ( i = 0; i < n; ++i ) {
                vfin = 3 * i * chv;
                vfout = 3 * i;
                v_out[vfout + 0] = 2*( (y2 + z2) * v_in[vfin + 0] + 
                    (xy - zw) * v_in[vfin + 1] + (yw + xz) * v_in[vfin + 2] ) 
                    + v_in[vfin + 0];
                v_out[vfout + 1] = 2*( (zw + xy) * v_in[vfin + 0] + 
                    (x2 + z2) * v_in[vfin + 1] + (yz - xw) * v_in[vfin + 2] ) 
                    + v_in[vfin + 1];
                v_out[vfout + 2] = 2*( (xz - yw) * v_in[vfin + 0] + 
                    (xw + yz) * v_in[vfin + 1] + (x2 + y2) * v_in[vfin + 2] ) 
                    + v_in[vfin + 2];
            }

        } else {
            #pragma omp parallel for default(shared) private(i, vfin, vfout, qf, xw, yw, zw, x2, xy, xz, y2, yz, z2) schedule(static)
            for ( i = 0; i < n; ++i ) {
                vfin = 3 * i * chv;
                vfout = 3 * i;
                qf = 4 * i * chq;
                // if ( i % 1000 == 0 ) {
                //     std::ostringstream o;
                //     o << i << " t=" << omp_get_thread_num() << " chv = " << chv << " chq = " << chq << " vfin = " << vfin << " vfout = " << vfout << " qf = " << qf;
                //     std::cout << o.str() << std::endl;
                // }
                xw =  q_unit[qf + 3] * q_unit[qf + 0];
                yw =  q_unit[qf + 3] * q_unit[qf + 1];
                zw =  q_unit[qf + 3] * q_unit[qf + 2];
                x2 = -q_unit[qf + 0] * q_unit[qf + 0];
                xy =  q_unit[qf + 0] * q_unit[qf + 1];
                xz =  q_unit[qf + 0] * q_unit[qf + 2];
                y2 = -q_unit[qf + 1] * q_unit[qf + 1];
                yz =  q_unit[qf + 1] * q_unit[qf + 2];
                z2 = -q_unit[qf + 2] * q_unit[qf + 2];
                v_out[vfout + 0] = 2*( (y2 + z2) * v_in[vfin + 0] + 
                    (xy - zw) * v_in[vfin + 1] + (yw + xz) * v_in[vfin + 2] ) 
                    + v_in[vfin + 0];
                v_out[vfout + 1] = 2*( (zw + xy) * v_in[vfin + 0] + 
                    (x2 + z2) * v_in[vfin + 1] + (yz - xw) * v_in[vfin + 2] ) 
                    + v_in[vfin + 1];
                v_out[vfout + 2] = 2*( (xz - yw) * v_in[vfin + 0] + 
                    (xw + yz) * v_in[vfin + 1] + (x2 + y2) * v_in[vfin + 2] ) 
                    + v_in[vfin + 2];
            }
        }
    }

    toast::mem::aligned_free ( q_unit );

    return;
}


// Multiply arrays of quaternions.

void toast::qarray::mult ( size_t np, double const * p, size_t nq, double const * q, double * r ) {

    int nt = 1;
    #ifdef _OPENMP
    nt = omp_get_num_threads();
    #endif

    size_t n = np;
    if ( nq > n ) {
        n = nq;
    }

    // this is to avoid branching inside the for loop.
    size_t chp;
    size_t chq;
    if ( np == 1 ) {
        chp = 0;
    } else {
        chp = 1;
    }
    if ( nq == 1 ) {
        chq = 0;
    } else {
        chq = 1;
    }

    size_t i;
    size_t pf;
    size_t qf;
    size_t f;

    if ( n < toast_qarray_ompthresh * nt ) {
        for ( size_t i = 0; i < n; ++i ) {
            f = 4 * i;
            pf = 4 * i * chp;
            qf = 4 * i * chq;
            r[f + 0] =  p[pf + 0] * q[qf + 3] + p[pf + 1] * q[qf + 2] 
                - p[pf + 2] * q[qf + 1] + p[pf + 3] * q[qf + 0];
            r[f + 1] = -p[pf + 0] * q[qf + 2] + p[pf + 1] * q[qf + 3] 
                + p[pf + 2] * q[qf + 0] + p[pf + 3] * q[qf + 1];
            r[f + 2] =  p[pf + 0] * q[qf + 1] - p[pf + 1] * q[qf + 0] 
                + p[pf + 2] * q[qf + 3] + p[pf + 3] * q[qf + 2];
            r[f + 3] = -p[pf + 0] * q[qf + 0] - p[pf + 1] * q[qf + 1] 
                - p[pf + 2] * q[qf + 2] + p[pf + 3] * q[qf + 3];
        }
    } else {
        #pragma omp parallel for default(shared) private(i, f, pf, qf) schedule(static)
        for ( i = 0; i < n; ++i ) {
            f = 4 * i;
            pf = 4 * i * chp;
            qf = 4 * i * chq;
            r[f + 0] =  p[pf + 0] * q[qf + 3] + p[pf + 1] * q[qf + 2] 
                - p[pf + 2] * q[qf + 1] + p[pf + 3] * q[qf + 0];
            r[f + 1] = -p[pf + 0] * q[qf + 2] + p[pf + 1] * q[qf + 3] 
                + p[pf + 2] * q[qf + 0] + p[pf + 3] * q[qf + 1];
            r[f + 2] =  p[pf + 0] * q[qf + 1] - p[pf + 1] * q[qf + 0] 
                + p[pf + 2] * q[qf + 3] + p[pf + 3] * q[qf + 2];
            r[f + 3] = -p[pf + 0] * q[qf + 0] - p[pf + 1] * q[qf + 1] 
                - p[pf + 2] * q[qf + 2] + p[pf + 3] * q[qf + 3];
        }
    }

    return;
}


// Spherical interpolation of quaternion array from time to targettime.

void toast::qarray::slerp ( size_t n_time, size_t n_targettime, double const * time, double const * targettime, double const * q_in, double * q_interp ) {

    #pragma omp parallel default(shared)
    {
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

        #pragma omp for schedule(static)
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
    }

    return;
}


// Exponential of a quaternion array.

void toast::qarray::exp ( size_t n, double const * q_in, double * q_out ) {

    double * normv = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, 4, 3, q_in, normv );
    
    double exp_q_w;

    #pragma omp parallel for default(shared) private(exp_q_w) schedule(static)
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

    double * normq = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

    toast::qarray::amplitude ( n, 4, 4, q_in, normq );

    toast::qarray::normalize ( n, 4, 3, q_in, q_out );

    double tmp;

    #pragma omp parallel for default(shared) private(tmp) schedule(static)
    for ( size_t i = 0; i < n; ++i ) {
        q_out[4*i + 3] = ::log ( normq[i] );
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
    
    double * q_tmp = static_cast < double * > ( toast::mem::aligned_alloc ( 
        4 * n * sizeof(double), toast::mem::SIMD_ALIGN ) );

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

    if ( n == 1 ) {

        double sin_a;

        for ( size_t i = 0; i < n; ++i ) {
            sin_a = ::sin ( 0.5 * angle[i] );
            for ( size_t j = 0; j < 3; ++j ) {
                q_out[4*i + j] = axis[3*i + j] * sin_a;
            }
            q_out[4*i + 3] = ::cos ( 0.5 * angle[i] );
        }

    } else {

        double * a = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        for ( size_t i = 0; i < n; ++i ) {
            a[i] = 0.5 * angle[i];
        }

        double * sin_a = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        double * cos_a = static_cast < double * > ( toast::mem::aligned_alloc ( 
        n * sizeof(double), toast::mem::SIMD_ALIGN ) );

        toast::sf::sincos ( n, a, sin_a, cos_a );
    
        for ( size_t i = 0; i < n; ++i ) {
            for ( size_t j = 0; j < 3; ++j ) {
                q_out[4*i + j] = axis[3*i + j] * sin_a[i];
            }
            q_out[4*i + 3] = cos_a[i];
        }
    
        toast::mem::aligned_free ( a );
        toast::mem::aligned_free ( sin_a );
        toast::mem::aligned_free ( cos_a );
    }

    return;
}


// Returns the axis and angle of rotation of a quaternion.

void toast::qarray::to_axisangle ( size_t n, double const * q, double * axis, double * angle ) {

    #pragma omp parallel default(shared)
    {
        size_t vf;
        size_t qf;
        double tmp;

        #pragma omp for schedule(static)
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


// Create quaternions from latitude, longitude, and position angles

void toast::qarray::from_angles ( size_t n, double const * theta, 
    double const * phi, double * const pa, double * quat, bool IAU ) {

    #pragma omp parallel default(shared)
    {
        size_t qf;

        double angR;
        double angD;
        double angP;
        double norm;

        double qR[4];
        double qD[4];
        double qP[4];
        double qtemp[4];

        #pragma omp for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            qf = 4 * i;

            // phi rotation around z-axis

            angR = phi[i] + toast::PI_2;

            qR[0] = 0.0;
            qR[1] = 0.0;
            qR[2] = ::sin ( 0.5 * angR );
            qR[3] = ::cos ( 0.5 * angR );

            // theta rotation around x-axis

            angD = theta[i];

            qD[0] = ::sin ( 0.5 * angD );
            qD[1] = 0.0;
            qD[2] = 0.0;
            qD[3] = ::cos ( 0.5 * angD );

            // position angle rotation about z-axis

            angP = toast::PI_2;
            if ( IAU ) {
                angP -= pa[i];
            } else {
                angP += pa[i];
            }

            qP[0] = 0.0;
            qP[1] = 0.0;
            qP[2] = ::sin ( 0.5 * angP );
            qP[3] = ::cos ( 0.5 * angP );

            toast::qarray::mult ( 1, qD, 1, qP, qtemp );
            toast::qarray::mult ( 1, qR, 1, qtemp, &(quat[qf]) );

            norm = 0.0;
            norm += quat[qf] * quat[qf];
            norm += quat[qf+1] * quat[qf+1];
            norm += quat[qf+2] * quat[qf+2];
            norm += quat[qf+3] * quat[qf+3];
            norm = 1.0 / ::sqrt ( norm );
            quat[qf] *= norm;
            quat[qf+1] *= norm;
            quat[qf+2] *= norm;
            quat[qf+3] *= norm;
        }

    }

    return;
}


// Convert quaternions to latitude, longitude, and position angle

void toast::qarray::to_angles ( size_t n, double const * quat, double * theta, 
    double * phi, double * pa, bool IAU ) {

    double const xaxis[3] = { 1.0, 0.0, 0.0 };
    double const zaxis[3] = { 0.0, 0.0, 1.0 };

    #pragma omp parallel default(shared)
    {
        size_t qf;

        double norm;

        double dir[3];
        double orient[3];
        double qtemp[4];

        #pragma omp for schedule(static)
        for ( size_t i = 0; i < n; ++i ) {
            qf = 4 * i;

            norm = 0.0;
            norm += quat[qf] * quat[qf];
            norm += quat[qf+1] * quat[qf+1];
            norm += quat[qf+2] * quat[qf+2];
            norm += quat[qf+3] * quat[qf+3];
            norm = 1.0 / ::sqrt ( norm );
            qtemp[0] = quat[qf] * norm;
            qtemp[1] = quat[qf+1] * norm;
            qtemp[2] = quat[qf+2] * norm;
            qtemp[3] = quat[qf+3] * norm;

            toast::qarray::rotate ( 1, qtemp, 1, zaxis, dir );
            toast::qarray::rotate ( 1, qtemp, 1, xaxis, orient );

            theta[i] = toast::PI_2 - ::asin ( dir[2] );
            phi[i] = ::atan2 ( dir[1], dir[0] );

            if ( phi[i] < 0.0 ) {
                phi[i] += toast::TWOPI;
            }

            pa[i] = ::atan2 ( orient[0] * dir[1] - orient[1] * dir[0], 
                - ( orient[0] * dir[2] * dir[0] ) 
                - ( orient[1] * dir[2] * dir[1] ) 
                + ( orient[2] * ( dir[0] * dir[0] + dir[1] * dir[1] ) ) );

            if ( IAU ) {
                pa[i] = -pa[i];
            }
        }
    }

    return;
}


