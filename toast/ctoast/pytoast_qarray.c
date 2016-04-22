/*
    Re-implementation of the quaternion array library

    A n-Quaternion array has 4 columns defined as x y z w and n rows
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <pytoast.h>

int i, j;

/*
Dot product of a lists of arrays, returns a column array.  Arrays a and b 
must be n by 4, but only the m first columns will be used for the dot 
product (n-array)
*/
void pytoast_qarraylist_dot(const int n, const int m, const double* a, const double* b, double* dotprod) {
    for (i = 0; i < n; ++i) {
        dotprod[i]=0.0;
        for (j = 0; j < m; ++j) {
            dotprod[i] += a[4*i + j] * b[4*i + j];
        }
    }
    return;
}

/*
Inverse of quaternion array q
q is a n by 4 array
*/
void pytoast_qinv(const int n, double* q) {
    for (i = 0; i < n; ++i) {
        for (j = 0; j < 3; ++j) {
            q[4*i + j] *= -1;
        }
    }
    return;
}

/*
Norm of quaternion array list
v must be a n by 4 array, only the first m rows will be considered, l2 is a n-array
*/
void pytoast_qamplitude(const int n, const int m, const double* v, double* l2) {
    pytoast_qarraylist_dot(n, m, v, v, l2);
    for (i = 0; i < n; ++i) {
        l2[i] = sqrt(l2[i]);
    }
    return;
}

/*
Normalize quaternion array q or array list to unit quaternions
q_in must be a n by 4 arrray, only the first m rows will be considered, results are output to q_out
*/
void pytoast_qnorm(const int n, const int m, const double* q_in, double* q_out) {
    double* l2 = malloc(n * sizeof(double));
    pytoast_qamplitude(n, m, q_in, l2);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            if (l2[i]<1e-8) {
                /*
                fprintf(stderr, "too small denominator (l2[%d]=%f) at %s:%d\n", i, l2[i], __FILE__, __LINE__);
                fprintf(stderr, "q[%d,:] = [ %f , %f , %f , %f ]\n", i, q_in[4*i], q_in[4*i+1], q_in[4*i+2], q_in[4*i+3]);
                */
            } else {
                q_out[4*i + j] = q_in[4*i + j] / l2[i];
            }
        }
    }
    free(l2);
    return;
}

/*
Normalize quaternion array q or array list to unit quaternions
q must be a n by 4 arrray, only the first m rows will be considered, results are written to q
*/
void pytoast_qnorm_inplace(const int n, const int m, double* q) {
    double* l2 = malloc(n * sizeof(double));
    pytoast_qamplitude(n, m, q, l2);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            q[4*i + j] /= l2[i];
        }
    }
    free(l2);
    return;
}

/*
Rotate vector v by n-quaternion array q and returns array with rotate n-vectors
v is a 3D-vector and q is a n by 4 array, v_out is a n by 3 array.
*/
void pytoast_qrotate(const int n, const double* v, const double* q_in, double* v_out) {
    /* Allocating temporary unit quaternion array */
    double *q_unit = malloc(n * 4 * sizeof(double));
    double xw,yw,zw,x2,xy,xz,y2,yz,z2;

    pytoast_qnorm(n, 4, q_in, q_unit);

    for (i = 0; i < n; ++i) {
        xw =  q_unit[4*i + 3]*q_unit[4*i + 0];
        yw =  q_unit[4*i + 3]*q_unit[4*i + 1];
        zw =  q_unit[4*i + 3]*q_unit[4*i + 2];
        x2 = -q_unit[4*i + 0]*q_unit[4*i + 0];
        xy =  q_unit[4*i + 0]*q_unit[4*i + 1];
        xz =  q_unit[4*i + 0]*q_unit[4*i + 2];
        y2 = -q_unit[4*i + 1]*q_unit[4*i + 1];
        yz =  q_unit[4*i + 1]*q_unit[4*i + 2];
        z2 = -q_unit[4*i + 2]*q_unit[4*i + 2];

        v_out[3*i + 0] = 2*( (y2 + z2)*v[0] + (xy - zw)*v[1] + (yw + xz)*v[2] ) + v[0];
        v_out[3*i + 1] = 2*( (zw + xy)*v[0] + (x2 + z2)*v[1] + (yz - xw)*v[2] ) + v[1];
        v_out[3*i + 2] = 2*( (xz - yw)*v[0] + (xw + yz)*v[1] + (x2 + y2)*v[2] ) + v[2];
    }
    free(q_unit);
    return;
}

/*
Multiply arrays of quaternions
p, q and r are n by 4 arrays
*/
void pytoast_qmult(const int n, const double* p, const double* q, double* r) {
    for (i = 0; i < n; ++i) {
        r[4*i + 0] =  p[4*i + 0] * q[4*i + 3] + p[4*i + 1] * q[4*i + 2] 
            - p[4*i + 2] * q[4*i + 1] + p[4*i + 3] * q[4*i + 0];
        r[4*i + 1] = -p[4*i + 0] * q[4*i + 2] + p[4*i + 1] * q[4*i + 3] 
            + p[4*i + 2] * q[4*i + 0] + p[4*i + 3] * q[4*i + 1];
        r[4*i + 2] =  p[4*i + 0] * q[4*i + 1] - p[4*i + 1] * q[4*i + 0] 
            + p[4*i + 2] * q[4*i + 3] + p[4*i + 3] * q[4*i + 2];
        r[4*i + 3] = -p[4*i + 0] * q[4*i + 0] - p[4*i + 1] * q[4*i + 1] 
            - p[4*i + 2] * q[4*i + 2] + p[4*i + 3] * q[4*i + 3];
    }
    return;
}

/*
Normalized interpolation of q quaternion array from time to targettime.
targettime is a n_time-array, time is a 2-array (start and end time), q_in is
a 2 by 4 array, q_interp is a n_time by 4 array.
*/
void pytoast_nlerp(const int n_time, const double* targettime, const double* time, const double* q_in, double* q_interp) {
    double* t_matrix = malloc(n_time * sizeof(double));

    pytoast_compute_t(n_time, targettime, time, t_matrix);

    for (i = 0; i < n_time; ++i) {
        for (j = 0; j < 4; ++j) {
            q_interp[4*i + j] = q_in[4*0 + j] * (1 - t_matrix[i]) + q_in[4*1 + j] * t_matrix[i];
        }
    }

    pytoast_qnorm_inplace(n_time, 4, q_interp);

    free(t_matrix);
    return;
}

/*
Spherical interpolation of q quaternion array from time to targettime
targettime is a n_time-array, time is a 2-array (start and end time), q_in 
is a 2 by 4 array, q_interp is a n_time by 4 array.
*/
void pytoast_slerp(const int n_time, const double* targettime, const double* time, const double* q_in, double* q_interp) {
    /* Allocating temporary arrays */
    double* t_matrix = malloc(n_time * sizeof(double));
    double* q_tmp1 = malloc(n_time * 4 * sizeof(double));
    double* q_tmp2 = malloc(n_time * 4 * sizeof(double));

    pytoast_compute_t(n_time, targettime, time, t_matrix);

    /*
    Think of removing first and last row from all the arrays 
    (since not interpolated)
    */
    for (i = 0; i < n_time; ++i) {
        for (j = 0; j < 4; ++j) {
            q_tmp1[4*i + j] = q_in[4*1 + j];
        }
    }

    for (i = 0; i < n_time-1; ++i) {
        for (j = 0; j < 4; ++j) {
            q_tmp2[4*i + j] = q_in[4*0 + j];
        }
    }
    for (j = 0; j < 4; ++j) {
        q_tmp2[4*(n_time-1) + j] = q_in[4*1 + j];
    }

    pytoast_qinv(n_time, q_tmp2);
    pytoast_qmult(n_time, q_tmp1, q_tmp2, q_interp);
    pytoast_qpow(n_time, t_matrix, q_interp, q_tmp1);
    pytoast_qinv(n_time, q_tmp2); /* q_tmp2 back to where it was at first */
    pytoast_qmult(n_time, q_tmp1, q_tmp2, q_interp);

    /* First and last rows reinit */
    for (j = 0; j < 4; ++j) {
        q_interp[4*0 + j] = q_in[4*0 + j];
        q_interp[4*(n_time-1) + j] = q_in[4*1 + j];
    }
    free(q_tmp2);
    free(q_tmp1);
    free(t_matrix);
    return;
}

/*
Compute the time vector used for interpolation
targettime is a n_time-array, time is a 2-array (start and end time), t_matrix is a n_time-array
*/
void pytoast_compute_t(const int n_time, const double* targettime, const double* time, double* t_matrix) {
    double t_span = time[1] - time[0];
    for (i = 0; i < n_time; ++i) {
        t_matrix[i] = (targettime[i] - time[0])/t_span;
        /*
        Consistency checks
        if (t_matrix[i]<0 || t_matrix[i]>1) {
           fprintf(stderr, "targettime inconsistent (extrapolation) at %s:%d\n", __FILE__, __LINE__);
           return;
        }
        */
    }
    return;
}

/*
Exponential of a quaternion array
q_in and q_out are n by 4 arrays
*/
void pytoast_qexp(const int n, const double* q_in, double* q_out) {
    double* normv = malloc(n * sizeof(double));
    pytoast_qamplitude(n, 3, q_in, normv);
    double exp_q_w;
    for (i = 0; i < n; ++i) {
        exp_q_w = exp(q_in[4*i + 3]);
        q_out[4*i + 3] = exp_q_w * cos(normv[i]);
        // computing exp(..) * sin(normv) / normv
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
void pytoast_qln(const int n, const double* q_in, double* q_out) {
    double* normq = malloc(n * sizeof(double));
    pytoast_qamplitude(n, 4, q_in, normq);
    for (i = 0; i < n; ++i) {
        q_out[4*i + 3] = log(normq[i]);
    }
    pytoast_qnorm(n, 3, q_in, q_out);
    double tmp;
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
void pytoast_qpow(const int n, const double* p, const double* q_in, double* q_out) {
    /* Allocating temporary quaternion array */
    double* q_tmp = malloc(n * 4 * sizeof(double));

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
void pytoast_from_axisangle(const int n, const double* axis, const double* angle, double* q_out) {
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
    angle[0] = 2 * acos(q[3]);
    if (angle[0] < 1e-4) {
        axis[0] = 0;
        axis[1] = 0;
        axis[2] = 0;
    } else {
        double tmp = sin(angle[0]/2.);
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
        S = sqrt(tr+1.0) * 2; // S=4*qw 
        q[0] = (rotmat[7] - rotmat[5]) / S;
        q[1] = (rotmat[2] - rotmat[6]) / S; 
        q[2] = (rotmat[3] - rotmat[1]) / S; 
        q[3] = 0.25 * S;
    } else if ((rotmat[0] > rotmat[4]) && (rotmat[0] > rotmat[8])) { 
        S = sqrt(1.0 + rotmat[0] - rotmat[4] - rotmat[8]) * 2; // S=4*qx 
        q[0] = 0.25 * S;
        q[1] = (rotmat[1] + rotmat[3]) / S; 
        q[2] = (rotmat[2] + rotmat[6]) / S; 
        q[3] = (rotmat[7] - rotmat[5]) / S;
    } else if (rotmat[4] > rotmat[8]) { 
        S = sqrt(1.0 + rotmat[4] - rotmat[0] - rotmat[8]) * 2; // S=4*qy
        q[0] = (rotmat[1] + rotmat[3]) / S; 
        q[1] = 0.25 * S;
        q[2] = (rotmat[5] + rotmat[7]) / S; 
        q[3] = (rotmat[2] - rotmat[6]) / S;
    } else {
        S = sqrt(1.0 + rotmat[8] - rotmat[0] - rotmat[4]) * 2; // S=4*qz
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
    /*
    if (dotprod < -0.999999) {
        
    } else if (dotprod > 0.999999) {
        
    } else {
        
    }
    */
    q[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    q[1] = vec1[0]*vec2[2] - vec1[2]*vec2[0];
    q[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
    q[3] = sqrt(vec1prod * vec2prod) + dotprod;
    pytoast_qnorm_inplace(1,4,q);

    return;
}


