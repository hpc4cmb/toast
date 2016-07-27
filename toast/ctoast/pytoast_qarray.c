/*
    Re-implementation of the quaternion array library

    A n-Quaternion array has 4 columns defined as x y z w and n rows
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <pytoast.h>


/*
Dot product of a lists of arrays, returns a column array.
*/
void pytoast_qarraylist_dot(int n, int m, int d, const double* a, const double* b, double* dotprod) {
    int i, j;
    for (i = 0; i < n; ++i) {
        dotprod[i] = 0.0;
        for (j = 0; j < d; ++j) {
            dotprod[i] += a[m*i + j] * b[m*i + j];
        }
    }
    return;
}

/*
Inverse of quaternion array q
q is a n by 4 array
*/
void pytoast_qinv(int n, double* q) {
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < 3; ++j) {
            q[4*i + j] *= -1;
        }
    }
    return;
}

/*
Norm of quaternion array list
*/
void pytoast_qamplitude(int n, int m, int d, const double* v, double* l2) {
    int i;
    pytoast_qarraylist_dot(n, m, d, v, v, l2);
    for (i = 0; i < n; ++i) {
        l2[i] = sqrt(l2[i]);
    }
    return;
}

/*
Normalize quaternion array q or array list to unit quaternions.
*/
void pytoast_qnorm(int n, int m, int d, const double* q_in, double* q_out) {
    int i, j;
    double* l2 = (double*)malloc(n * sizeof(double));
    if (l2 == NULL) {
        fprintf(stderr, "cannot allocate %d doubles\n", n);
        exit(1);
    }
    pytoast_qamplitude(n, m, d, q_in, l2);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < d; ++j) {
            q_out[m*i + j] = q_in[m*i + j] / l2[i];
        }
    }
    free(l2);
    return;
}

/*
Normalize quaternion array q or array list to unit quaternions
*/
void pytoast_qnorm_inplace(int n, int m, int d, double* q) {
    int i, j;
    double* l2 = (double*)malloc(n * sizeof(double));
    if (l2 == NULL) {
        fprintf(stderr, "cannot allocate %d doubles\n", n);
        exit(1);
    }
    pytoast_qamplitude(n, m, d, q, l2);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < d; ++j) {
            q[m*i + j] /= l2[i];
        }
    }
    free(l2);
    return;
}

/*
Rotate an array of vectors by an array of quaternions and return the
resulting array of vectors.
*/
void pytoast_qrotate(int n, const double* v, const double* q_in, double* v_out) {
    int i;
    double xw,yw,zw,x2,xy,xz,y2,yz,z2;
    int vf;
    int qf;
    double * q_unit;

    /* Allocating temporary unit quaternion array */
    q_unit = (double*)malloc(n * 4 * sizeof(double));
    if (q_unit == NULL) {
        fprintf(stderr, "cannot allocate %d doubles\n", 4*n);
        exit(1);
    }

    pytoast_qnorm(n, 4, 4, q_in, q_unit);

    for (i = 0; i < n; ++i) {
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

        v_out[vf + 0] = 2*( (y2 + z2)*v[vf + 0] + (xy - zw)*v[vf + 1] + (yw + xz)*v[vf + 2] ) + v[vf + 0];
        v_out[vf + 1] = 2*( (zw + xy)*v[vf + 0] + (x2 + z2)*v[vf + 1] + (yz - xw)*v[vf + 2] ) + v[vf + 1];
        v_out[vf + 2] = 2*( (xz - yw)*v[vf + 0] + (xw + yz)*v[vf + 1] + (x2 + y2)*v[vf + 2] ) + v[vf + 2];
    }
    free(q_unit);
    return;
}


/*
Multiply arrays of quaternions
p, q and r are n by 4 arrays
*/
void pytoast_qmult(int n, const double* p, const double* q, double* r) {
    int i;
    int qf;
    for (i = 0; i < n; ++i) {
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


/*
Spherical interpolation of q quaternion array from time to targettime
targettime is a n_time-array, time is a 2-array (start and end time), q_in 
is a 2 by 4 array, q_interp is a n_time by 4 array.
*/
void pytoast_slerp(int n_time, int n_targettime, const double* time, const double* targettime, const double* q_in, double* q_interp) {
    int i;
    /* Allocating temporary arrays */
    int off;
    double diff;
    double frac;
    double costheta;
    double const * qlow;
    double const * qhigh;
    double theta;
    double invsintheta;
    double norm;
    double *q;
    double ratio1;
    double ratio2;

    off = 0;

    for (i = 0; i < n_targettime; ++i) {
        /* scroll forward to the correct time sample */
        while ((off+1 < n_time) && (time[off+1] < targettime[i])) {
            ++off;
        }
        diff = time[off+1] - time[off];
        frac = (targettime[i] - time[off]) / diff;

        qlow = &(q_in[4*off]);
        qhigh = &(q_in[4*(off + 1)]);
        q = &(q_interp[4*i]);

        costheta = qlow[0] * qhigh[0] + qlow[1] * qhigh[1] + qlow[2] * qhigh[2] + qlow[3] * qhigh[3];
        if ( fabs ( costheta - 1.0 ) < 1.0e-10 ) {
            q[0] = qlow[0];
            q[1] = qlow[1];
            q[2] = qlow[2];
            q[3] = qlow[3];
        } else {
            theta = acos(costheta);
            invsintheta = 1.0 / sqrt ( 1.0 - costheta * costheta );
            ratio1 = sin ( ( 1.0 - frac ) * theta ) * invsintheta;
            ratio2 = sin ( frac * theta ) * invsintheta;
            q[0] = ratio1 * qlow[0] + ratio2 * qhigh[0];
            q[1] = ratio1 * qlow[1] + ratio2 * qhigh[1];
            q[2] = ratio1 * qlow[2] + ratio2 * qhigh[2];
            q[3] = ratio1 * qlow[3] + ratio2 * qhigh[3];
        }
        
        norm = 1.0 / sqrt ( q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] );
        q[0] *= norm;
        q[1] *= norm;
        q[2] *= norm;
        q[3] *= norm;
    }

    return;
}


/*
Exponential of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qexp(int n, const double* q_in, double* q_out) {
    int i, j;
    double* normv;
    double exp_q_w;

    normv = (double*)malloc(n * sizeof(double));
    if (normv == NULL) {
        fprintf(stderr, "cannot allocate %d doubles\n", n);
        exit(1);
    }
    pytoast_qamplitude(n, 4, 3, q_in, normv);
    
    for (i = 0; i < n; ++i) {
        exp_q_w = exp(q_in[4*i + 3]);
        q_out[4*i + 3] = exp_q_w * cos(normv[i]);
        /* computing exp(..) * sin(normv) / normv */
        exp_q_w /= normv[i];
        exp_q_w *= sin(normv[i]);
        for (j = 0; j < 3; ++j) {
            q_out[4*i + j] = exp_q_w * q_in[4*i + j];
        }
    }
    free(normv);
    return;
}

/*
Natural logarithm of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qln(int n, const double* q_in, double* q_out) {
    int i, j;
    double* normq;
    double tmp;

    normq = (double*)malloc(n * sizeof(double));
    if (normq == NULL) {
        fprintf(stderr, "cannot allocate %d doubles\n", n);
        exit(1);
    }

    pytoast_qamplitude(n, 4, 4, q_in, normq);
    for (i = 0; i < n; ++i) {
        q_out[4*i + 3] = log(normq[i]);
    }
    pytoast_qnorm(n, 4, 3, q_in, q_out);

    for (i = 0; i < n; ++i) {
        tmp = acos(q_in[4*i + 3]/normq[i]);
        for (j = 0; j < 3; ++j) {
            q_out[4*i + j] *= tmp;
        }
    }
    free(normq);
    return;
}

/*
Real power of quaternion array
p is a n-array, q_in and q_out are n by 4 arrays
*/
void pytoast_qpow(int n, const double* p, const double* q_in, double* q_out) {
    int i, j;
    /* Allocating temporary quaternion array */
    double* q_tmp = (double*)malloc(n * 4 * sizeof(double));
    if (q_tmp == NULL) {
        fprintf(stderr, "cannot allocate %d doubles\n", 4*n);
        exit(1);
    }

    pytoast_qln(n, q_in, q_tmp);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < 4; ++j) {
            q_tmp[4*i + j] *= p[i];
        }
    }
    pytoast_qexp(n, q_tmp, q_out);

    /* Freeing temporary quaternion array */
    free(q_tmp);
    return;
}

/*
Creates rotation quaternions of angles (in [rad]) around axes [already normalized]
axis is an n by 3 array, angle is a n-array, q_out is a n by 4 array
*/
void pytoast_from_axisangle(int n, const double* axis, const double* angle, double* q_out) {
    int i, j;
    double sin_a;
    for (i = 0; i < n; ++i) {
        sin_a=sin(angle[i]/2);
        for (j = 0; j < 3; ++j) {
            q_out[4*i + j] = axis[3*i + j]*sin_a;
        }
        q_out[4*i + 3] = cos(angle[i]/2);
    }
    return;
}

/*
Returns the axis and angle of rotation of a quaternion
q is a 4-array, axis is a 3-array, angle is 1-array
*/
void pytoast_to_axisangle(const double* q, double* axis, double* angle) {
    double tmp;

    angle[0] = 2 * acos(q[3]);
    if (angle[0] < 1e-4) {
        axis[0] = 0;
        axis[1] = 0;
        axis[2] = 0;
    } else {
        tmp = sin(angle[0]/2.);
        axis[0] = q[0]/tmp;
        axis[1] = q[1]/tmp;
        axis[2] = q[2]/tmp;
    }
    return;
}

/*
Creates the rotation matrix corresponding to a quaternion
q is a 4 array, rotmat is a 3 by 3 array
*/
void pytoast_to_rotmat(const double* q, double* rotmat) {
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

/*
Creates the quaternion from a rotation matrix
rotmat is a 3 by 3 array, q is a 4-array.
*/
void pytoast_from_rotmat(const double* rotmat, double* q) {
    double tr = rotmat[0] + rotmat[4] + rotmat[8];
    double S;
    if (tr > 0) { 
        S = sqrt(tr+1.0) * 2; /* S=4*qw */
        q[0] = (rotmat[7] - rotmat[5]) / S;
        q[1] = (rotmat[2] - rotmat[6]) / S; 
        q[2] = (rotmat[3] - rotmat[1]) / S; 
        q[3] = 0.25 * S;
    } else if ((rotmat[0] > rotmat[4]) && (rotmat[0] > rotmat[8])) { 
        S = sqrt(1.0 + rotmat[0] - rotmat[4] - rotmat[8]) * 2; /* S=4*qx */
        q[0] = 0.25 * S;
        q[1] = (rotmat[1] + rotmat[3]) / S; 
        q[2] = (rotmat[2] + rotmat[6]) / S; 
        q[3] = (rotmat[7] - rotmat[5]) / S;
    } else if (rotmat[4] > rotmat[8]) { 
        S = sqrt(1.0 + rotmat[4] - rotmat[0] - rotmat[8]) * 2; /* S=4*qy */
        q[0] = (rotmat[1] + rotmat[3]) / S; 
        q[1] = 0.25 * S;
        q[2] = (rotmat[5] + rotmat[7]) / S; 
        q[3] = (rotmat[2] - rotmat[6]) / S;
    } else {
        S = sqrt(1.0 + rotmat[8] - rotmat[0] - rotmat[4]) * 2; /* S=4*qz */
        q[0] = (rotmat[2] + rotmat[6]) / S;
        q[1] = (rotmat[5] + rotmat[7]) / S;
        q[2] = 0.25 * S;
        q[3] = (rotmat[3] - rotmat[1]) / S;
    }
    return;
}

/*
Creates the quaternion from two normalized vectors (be careful with colinear vectors)
vec1 and vec2 are 3-arrays, q is a 4-array
*/
void pytoast_from_vectors(const double* vec1, const double* vec2, double* q) {
    double dotprod = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    double vec1prod = vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2];
    double vec2prod = vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2];

    /* shortcut for coincident vectors */
    if (fabs(dotprod) < 1.0e-12) {
        q[0] = 0.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 1.0;
    } else {
        q[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
        q[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
        q[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
        q[3] = sqrt(vec1prod * vec2prod) + dotprod;
        pytoast_qnorm_inplace(1, 4, 4, q);
    }

    return;
}


