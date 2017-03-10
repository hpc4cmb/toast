/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const double qarrayTest::q1[] = { 0.50487417,  0.61426059,  0.60118994,  0.07972857 };
const double qarrayTest::q1inv[] = { -0.50487417,  -0.61426059,  -0.60118994,  0.07972857 };
const double qarrayTest::q2[] = { 0.43561544,  0.33647027,  0.40417115,  0.73052901 };
const double qarrayTest::qtonormalize[] = { 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0 };
const double qarrayTest::qnormalized[] = { 0.18257419, 0.36514837, 0.54772256, 0.73029674, 0.27216553, 0.40824829, 0.54433105, 0.68041382 };
const double qarrayTest::vec[] = { 0.57734543, 0.30271255, 0.75831218 };
const double qarrayTest::vec2[] = { 0.57734543, 8.30271255, 5.75831218, 1.57734543, 3.30271255, 0.75831218 };
const double qarrayTest::qeasy[] = { 0.3, 0.3, 0.1, 0.9, 0.3, 0.3, 0.1, 0.9 };
const double qarrayTest::mult_result[] = { 0.44954009, 0.53339352, 0.37370443, -0.61135101 };
const double qarrayTest::rot_by_q1[] = { 0.4176698, 0.84203849, 0.34135482 };
const double qarrayTest::rot_by_q2[] = { 0.8077876, 0.3227185, 0.49328689 };


void qarrayTest::SetUp () {
    return;
}


TEST_F( qarrayTest, arraylist_dot1 ) {
    double check;
    double result;
    double pone[3];

    check = 0.0;
    for ( size_t i = 0; i < 3; ++i ) {
        pone[i] = vec[i] + 1.0;
        check += vec[i] * pone[i];
    }

    qarray::list_dot ( 1, 3, 3, vec, pone, &result );

    EXPECT_DOUBLE_EQ( check, result );
}


TEST_F( qarrayTest, arraylist_dot2 ) {
    double check[2];
    double result[2];
    double pone[6];

    for ( size_t i = 0; i < 2; ++i ) {
        check[i] = 0.0;
        for ( size_t j = 0; j < 3; ++j ) {
            pone[3*i+j] = vec2[3*i+j] + 1.0;
            check[i] += vec2[3*i+j] * pone[3*i+j];
        }
    }

    qarray::list_dot ( 2, 3, 3, vec2, pone, result );

    for ( size_t i = 0; i < 2; ++i ) {
        EXPECT_DOUBLE_EQ( check[i], result[i] );
    }
}


TEST_F( qarrayTest, inv ) {
    double result[4];

    for ( size_t i = 0; i < 4; ++i ) {
        result[i] = q1[i];
    }

    qarray::inv ( 1, result );

    for ( size_t i = 0; i < 4; ++i ) {
        EXPECT_FLOAT_EQ( q1inv[i], result[i] );
    }
}


TEST_F( qarrayTest, norm ) {
    double result[4];

    qarray::normalize ( 1, 4, 4, qtonormalize, result );

    for ( size_t i = 0; i < 4; ++i ) {
        EXPECT_FLOAT_EQ( qnormalized[i], result[i] );
    }
}


TEST_F( qarrayTest, mult ) {
    double result[4];

    qarray::mult ( 1, q1, q2, result );

    for ( size_t i = 0; i < 4; ++i ) {
        EXPECT_FLOAT_EQ( mult_result[i], result[i] );
    }
}


TEST_F( qarrayTest, multarray ) {
    size_t n = 3;
    double in1[4*n];
    double in2[4*n];
    double result[4*n];
    double null[4*n];

    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < 4; ++j ) {
            in1[4*i+j] = q1[j];
            in2[4*i+j] = q2[j];
            null[4*i+j] = 0.0;
            if ( j == 3 ) {
                null[4*i+j] = 1.0;
            }
        }
    }

    qarray::mult ( n, in1, in2, result );

    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < 4; ++j ) {
            EXPECT_FLOAT_EQ( mult_result[j], result[4*i+j] );
        }
    }

    qarray::mult ( n, in1, null, result );

    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < 4; ++j ) {
            EXPECT_FLOAT_EQ( in1[j], result[4*i+j] );
        }
    }
}


TEST_F( qarrayTest, rot1 ) {
    double result[3];

    qarray::rotate ( 1, q1, vec, result );

    for ( size_t i = 0; i < 3; ++i ) {
        EXPECT_FLOAT_EQ( rot_by_q1[i], result[i] );
    }
}


TEST_F( qarrayTest, rotarray ) {
    size_t n = 2;
    double qin[4*n];
    double vin[3*n];
    double result[3*n];

    for ( size_t i = 0; i < 4; ++i ) {
        qin[i] = q1[i];
        qin[4+i] = q2[i];
    }

    for ( size_t i = 0; i < n; ++i ) {
        for ( size_t j = 0; j < 3; ++j ) {
            vin[3*i+j] = vec[j];
        }
    }

    qarray::rotate ( n, qin, vin, result );

    for ( size_t i = 0; i < 3; ++i ) {
        EXPECT_FLOAT_EQ( rot_by_q1[i], result[i] );
        EXPECT_FLOAT_EQ( rot_by_q2[i], result[3+i] );
    }
}


TEST_F( qarrayTest, slerp ) {
    size_t n = 2;
    size_t ninterp = 4;

    double q[8] = { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    double qinterp[16];
    double time[2] = { 0.0, 9.0 };
    double targettime[4] = { 0.0, 3.0, 4.5, 9.0 };
    double qcheck1[4];
    double qcheck2[4];

    qarray::normalize_inplace ( n, 4, 4, q );

    qarray::slerp ( n, ninterp, time, targettime, q, qinterp );

    for ( size_t i = 0; i < 4; ++i ) {
        qcheck1[i] = (2.0 / 3.0) * q[i] + (1.0 / 3.0) * q[4+i];
        qcheck2[i] = 0.5 * (q[i] + q[4+i]);
    }

    qarray::normalize_inplace( 1, 4, 4, qcheck1 );
    qarray::normalize_inplace( 1, 4, 4, qcheck2 );

    for ( size_t i = 0; i < 4; ++i ) {
        EXPECT_FLOAT_EQ( q[i], qinterp[i] );
        EXPECT_FLOAT_EQ( q[4+i], qinterp[12+i] );
        ASSERT_NEAR( qcheck1[i], qinterp[4+i], 1.0e-4 );
        ASSERT_NEAR( qcheck2[i], qinterp[8+i], 1.0e-4 );
    }
}


TEST_F( qarrayTest, rotation ) {
    double result[4];
    double axis[3] = { 0.0, 0.0, 1.0 };
    double ang = PI * 30.0 / 180.0;

    qarray::from_axisangle ( 1, axis, &ang, result );

    EXPECT_FLOAT_EQ( 0.0, result[0] );
    EXPECT_FLOAT_EQ( 0.0, result[1] );
    EXPECT_FLOAT_EQ( ::sin(15.0 * PI / 180.0), result[2] );
    EXPECT_FLOAT_EQ( ::cos(15.0 * PI / 180.0), result[3] );
}


TEST_F( qarrayTest, toaxisangle ) {
    double in[4] = { 0.0, 0.0, ::sin(15.0 * PI / 180.0), ::cos(15.0 * PI / 180.0) };
    double axis[3];
    double ang;
    double checkaxis[3] = { 0.0, 0.0, 1.0 };
    double checkang = 30.0 * PI / 180.0;

    qarray::to_axisangle ( 1, in, axis, &ang );

    EXPECT_FLOAT_EQ( checkang, ang );
    for ( size_t i = 0; i < 3; ++i ) {
        EXPECT_FLOAT_EQ( checkaxis[i], axis[i] );
    }
}


TEST_F( qarrayTest, exp ) {
    double result[8];
    double check[8] = { 0.71473568, 0.71473568, 0.23824523, 2.22961712, 0.71473568, 0.71473568, 0.23824523, 2.22961712 };

    qarray::exp ( 2, qeasy, result );

    for ( size_t i = 0; i < 8; ++i ) {
        EXPECT_FLOAT_EQ( check[i], result[i] );
    }
}


TEST_F( qarrayTest, ln ) {
    double result[8];
    double check[8] = { 0.31041794, 0.31041794, 0.10347265, 0.0, 0.31041794, 0.31041794, 0.10347265, 0.0 };

    qarray::ln ( 2, qeasy, result );

    for ( size_t i = 0; i < 8; ++i ) {
        EXPECT_FLOAT_EQ( check[i], result[i] );
    }
}


TEST_F( qarrayTest, pow ) {
    double p[2];
    double result[8];
    double check1[8] = { 0.672, 0.672, 0.224, 0.216, 0.672, 0.672, 0.224, 0.216 };
    double check2[8] = { 0.03103127, 0.03103127, 0.01034376, 0.99898305, 0.03103127, 0.03103127, 0.01034376, 0.99898305 };

    p[0] = 3.0;
    p[1] = 3.0;
    qarray::pow ( 2, p, qeasy, result );

    for ( size_t i = 0; i < 8; ++i ) {
        EXPECT_FLOAT_EQ( check1[i], result[i] );
    }

    p[0] = 0.1;
    p[1] = 0.1;
    qarray::pow ( 2, p, qeasy, result );

    for ( size_t i = 0; i < 8; ++i ) {
        EXPECT_FLOAT_EQ( check2[i], result[i] );
    }
}


TEST_F( qarrayTest, torotmat ) {
    double result[9];
    double check[9] = { 8.00000000e-01, -2.77555756e-17, 6.00000000e-01, 3.60000000e-01, 8.00000000e-01, -4.80000000e-01, -4.80000000e-01, 6.00000000e-01, 6.40000000e-01 };

    qarray::to_rotmat ( qeasy, result );

    for ( size_t i = 0; i < 9; ++i ) {
        EXPECT_FLOAT_EQ( check[i], result[i] );
    }
}


TEST_F( qarrayTest, fromrotmat ) {
    double result[9];
    double qresult[4];

    qarray::to_rotmat ( qeasy, result );
    qarray::from_rotmat ( result, qresult );

    for ( size_t i = 0; i < 4; ++i ) {
        EXPECT_FLOAT_EQ( qeasy[i], qresult[i] );
    }
}


TEST_F( qarrayTest, fromvectors ) {
    double result[4];
    double check[4] = { 0.0, 0.0, ::sin(15.0 * PI / 180.0), ::cos(15.0 * PI / 180.0) };
    double ang = 30.0 * PI / 180.0;
    double v1[3] = { 1.0, 0.0, 0.0 };
    double v2[3] = { ::cos(ang), ::sin(ang), 0.0 };

    qarray::from_vectors ( v1, v2, result );

    for ( size_t i = 0; i < 4; ++i ) {
        EXPECT_FLOAT_EQ( check[i], result[i] );
    }
}


