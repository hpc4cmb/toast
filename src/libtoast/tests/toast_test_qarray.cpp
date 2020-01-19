
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>

#include <cmath>

#include <limits>


void TOASTqarrayTest::SetUp() {
    q1 = {0.50487417,  0.61426059,  0.60118994,  0.07972857};
    q1inv =
    {-0.50487417,  -0.61426059,  -0.60118994,  0.07972857};
    q2 = {0.43561544,  0.33647027,  0.40417115,  0.73052901};
    qtonormalize = {1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0};
    qnormalized =
    {0.18257419, 0.36514837, 0.54772256, 0.73029674, 0.27216553, 0.40824829,
     0.54433105, 0.68041382};
    vec = {0.57734543, 0.30271255, 0.75831218};
    vec2 =
    {0.57734543, 8.30271255, 5.75831218, 1.57734543, 3.30271255, 0.75831218};
    qeasy = {0.3, 0.3, 0.1, 0.9, 0.3, 0.3, 0.1, 0.9};
    mult_result =
    {0.44954009, 0.53339352, 0.37370443, -0.61135101};
    rot_by_q1 = {0.4176698, 0.84203849, 0.34135482};
    rot_by_q2 = {0.8077876, 0.3227185, 0.49328689};
    return;
}

TEST_F(TOASTqarrayTest, arraylist_dot1) {
    double check;
    double result;
    toast::AlignedVector <double> pone(3);

    check = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        pone[i] = vec[i] + 1.0;
        check += vec[i] * pone[i];
    }

    toast::qa_list_dot(1, 3, 3, vec.data(), pone.data(), &result);

    EXPECT_DOUBLE_EQ(check, result);
}


TEST_F(TOASTqarrayTest, arraylist_dot2) {
    double check[2];
    toast::AlignedVector <double> result(2);
    toast::AlignedVector <double> pone(6);

    for (size_t i = 0; i < 2; ++i) {
        check[i] = 0.0;
        for (size_t j = 0; j < 3; ++j) {
            pone[3 * i + j] = vec2[3 * i + j] + 1.0;
            check[i] += vec2[3 * i + j] * pone[3 * i + j];
        }
    }

    toast::qa_list_dot(2, 3, 3, vec2.data(), pone.data(), result.data());

    for (size_t i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(check[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, inv) {
    toast::AlignedVector <double> result(4);

    for (size_t i = 0; i < 4; ++i) {
        result[i] = q1[i];
    }

    toast::qa_inv(1, result.data());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(q1inv[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, norm) {
    toast::AlignedVector <double> result(4);

    toast::qa_normalize(1, 4, 4, qtonormalize.data(), result.data());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(qnormalized[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, mult) {
    toast::AlignedVector <double> result(4);

    toast::qa_mult(1, q1.data(), 1, q2.data(), result.data());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(mult_result[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, multarray) {
    size_t n = 3;
    toast::AlignedVector <double> in1(4 * n);
    toast::AlignedVector <double> in2(4 * n);
    toast::AlignedVector <double> result(4 * n);
    toast::AlignedVector <double> null(4 * n);

    null[0] = 0.0;
    null[1] = 0.0;
    null[2] = 0.0;
    null[3] = 1.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            in1[4 * i + j] = q1[j];
            in2[4 * i + j] = q2[j];
        }
    }

    toast::qa_mult(n, in1.data(), n, in2.data(), result.data());

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(mult_result[j], result[4 * i + j]);
        }
    }

    toast::qa_mult(n, in1.data(), 1, null.data(), result.data());

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(in1[j], result[4 * i + j]);
        }
    }
}


TEST_F(TOASTqarrayTest, rot1) {
    toast::AlignedVector <double> result(3);

    toast::qa_rotate(1, q1.data(), 1, vec.data(), result.data());

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(rot_by_q1[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, rotarray) {
    size_t n = 2;
    toast::AlignedVector <double> qin(4 * n);
    toast::AlignedVector <double> vin(3 * n);
    toast::AlignedVector <double> result(3 * n);

    for (size_t i = 0; i < 4; ++i) {
        qin[i] = q1[i];
        qin[4 + i] = q2[i];
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            vin[3 * i + j] = vec[j];
        }
    }

    toast::qa_rotate(n, qin.data(), n, vin.data(), result.data());

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(rot_by_q1[i], result[i]);
        EXPECT_FLOAT_EQ(rot_by_q2[i], result[3 + i]);
    }
}


TEST_F(TOASTqarrayTest, slerp) {
    size_t n = 2;
    size_t ninterp = 4;

    toast::AlignedVector <double> q =
    {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    toast::AlignedVector <double> qinterp(16);
    toast::AlignedVector <double> time = {0.0, 9.0};
    toast::AlignedVector <double> targettime =
    {0.0, 3.0, 4.5, 9.0};
    toast::AlignedVector <double> qcheck1(4);
    toast::AlignedVector <double> qcheck2(4);

    toast::qa_normalize_inplace(n, 4, 4, q.data());

    toast::qa_slerp(n, ninterp, time.data(), targettime.data(),
                    q.data(), qinterp.data());

    for (size_t i = 0; i < 4; ++i) {
        qcheck1[i] = (2.0 / 3.0) * q[i] + (1.0 / 3.0) * q[4 + i];
        qcheck2[i] = 0.5 * (q[i] + q[4 + i]);
    }

    toast::qa_normalize_inplace(1, 4, 4, qcheck1.data());
    toast::qa_normalize_inplace(1, 4, 4, qcheck2.data());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(q[i], qinterp[i]);
        EXPECT_FLOAT_EQ(q[4 + i], qinterp[12 + i]);
        ASSERT_NEAR(qcheck1[i], qinterp[4 + i], 1.0e-4);
        ASSERT_NEAR(qcheck2[i], qinterp[8 + i], 1.0e-4);
    }
}


TEST_F(TOASTqarrayTest, rotation) {
    toast::AlignedVector <double> result(4);
    toast::AlignedVector <double> axis = {0.0, 0.0, 1.0};
    double ang = toast::PI * 30.0 / 180.0;

    toast::qa_from_axisangle(1, axis.data(), 1, &ang, result.data());

    EXPECT_FLOAT_EQ(0.0, result[0]);
    EXPECT_FLOAT_EQ(0.0, result[1]);
    EXPECT_FLOAT_EQ(::sin(15.0 * toast::PI / 180.0), result[2]);
    EXPECT_FLOAT_EQ(::cos(15.0 * toast::PI / 180.0), result[3]);
}


TEST_F(TOASTqarrayTest, toaxisangle) {
    double in[4] =
    {0.0, 0.0, ::sin(15.0 * toast::PI / 180.0),
     ::cos(15.0 * toast::PI / 180.0)};
    double axis[3];
    double ang;
    double checkaxis[3] = {0.0, 0.0, 1.0};
    double checkang = 30.0 * toast::PI / 180.0;

    toast::qa_to_axisangle(1, in, axis, &ang);

    EXPECT_FLOAT_EQ(checkang, ang);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(checkaxis[i], axis[i]);
    }
}


TEST_F(TOASTqarrayTest, exp) {
    toast::AlignedVector <double> result(8);
    toast::AlignedVector <double> check =
    {0.71473568, 0.71473568, 0.23824523, 2.22961712, 0.71473568, 0.71473568,
     0.23824523, 2.22961712};

    toast::qa_exp(2, qeasy.data(), result.data());

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(check[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, ln) {
    toast::AlignedVector <double> result(8);
    toast::AlignedVector <double> check =
    {0.31041794, 0.31041794, 0.10347265, 0.0, 0.31041794, 0.31041794,
     0.10347265, 0.0};

    toast::qa_ln(2, qeasy.data(), result.data());

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(check[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, pow) {
    toast::AlignedVector <double> p(2);
    toast::AlignedVector <double> result(8);
    toast::AlignedVector <double> check1 =
    {0.672, 0.672, 0.224, 0.216, 0.672, 0.672, 0.224, 0.216};
    toast::AlignedVector <double> check2 =
    {0.03103127, 0.03103127, 0.01034376, 0.99898305, 0.03103127, 0.03103127,
     0.01034376, 0.99898305};

    p[0] = 3.0;
    p[1] = 3.0;
    toast::qa_pow(2, 2, p.data(), qeasy.data(), result.data());

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(check1[i], result[i]);
    }

    p[0] = 0.1;
    p[1] = 0.1;
    toast::qa_pow(2, 2, p.data(), qeasy.data(), result.data());

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(check2[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, torotmat) {
    toast::AlignedVector <double> result(9);
    toast::AlignedVector <double> check =
    {8.00000000e-01, -2.77555756e-17, 6.00000000e-01, 3.60000000e-01,
     8.00000000e-01, -4.80000000e-01, -4.80000000e-01, 6.00000000e-01,
     6.40000000e-01};

    toast::qa_to_rotmat(qeasy.data(), result.data());

    for (size_t i = 0; i < 9; ++i) {
        if (::fabs(check[i]) > 1.0e-12) {
            EXPECT_FLOAT_EQ(check[i], result[i]);
        }
    }
}


TEST_F(TOASTqarrayTest, fromrotmat) {
    toast::AlignedVector <double> result(9);
    toast::AlignedVector <double> qresult(4);

    toast::qa_to_rotmat(qeasy.data(), result.data());
    toast::qa_from_rotmat(result.data(), qresult.data());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(qeasy[i], qresult[i]);
    }
}


TEST_F(TOASTqarrayTest, fromvectors) {
    toast::AlignedVector <double> result(4);
    toast::AlignedVector <double> check =
    {0.0, 0.0, ::sin(15.0 * toast::PI / 180.0),
     ::cos(15.0 * toast::PI / 180.0)};
    double ang = 30.0 * toast::PI / 180.0;
    toast::AlignedVector <double> v1 = {1.0, 0.0, 0.0};
    toast::AlignedVector <double> v2 =
    {::cos(ang), ::sin(ang), 0.0};

    toast::qa_from_vectors(1, v1.data(), v2.data(), result.data());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(check[i], result[i]);
    }
}


TEST_F(TOASTqarrayTest, thetaphipa) {
    size_t n_theta = 5;
    size_t n_phi = 5;
    size_t n = n_theta * n_phi;

    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};

    toast::AlignedVector <double> theta(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <double> pa(n);

    toast::AlignedVector <double> check_theta(n);
    toast::AlignedVector <double> check_phi(n);
    toast::AlignedVector <double> check_pa(n);

    toast::AlignedVector <double> quat(4 * n);

    // First run tests in Healpix convention...

    for (size_t i = 0; i < n_theta; ++i) {
        for (size_t j = 0; j < n_phi; ++j) {
            theta[i * n_phi + j] = (0.5 + (double)i) * toast::PI /
                                   (double)n_theta;
            phi[i * n_phi + j] = (double)j * toast::TWOPI / (double)n_phi;
            pa[i * n_phi + j] = (double)j * toast::TWOPI / (double)n_phi -
                                toast::PI;
        }
    }

    // convert to quaternions

    toast::qa_from_angles(n, theta.data(), phi.data(), pa.data(),
                          quat.data(), false);

    // check that the resulting quaternions rotate the Z and X
    // axes to the correct place.

    double dir[3];
    double orient[3];
    double check;

    for (size_t i = 0; i < n; ++i) {
        toast::qa_rotate(1, &(quat[4 * i]), 1, zaxis, dir);
        toast::qa_rotate(1, &(quat[4 * i]), 1, xaxis, orient);

        ASSERT_NEAR(toast::PI_2 - ::asin(dir[2]), theta[i], 1.0e-6);

        check = ::atan2(dir[1], dir[0]);

        if (check < 0.0) {
            check += toast::TWOPI;
        }
        if (check >= toast::TWOPI) {
            check -= toast::TWOPI;
        }
        if (::fabs(check) < 2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }
        if (::fabs(check - toast::TWOPI) <
            2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }

        ASSERT_NEAR(check, phi[i], 1.0e-6);

        check = ::atan2(orient[0] * dir[1] - orient[1] * dir[0],
                        -(orient[0] * dir[2] * dir[0])
                        - (orient[1] * dir[2] * dir[1])
                        + (orient[2] * (dir[0] * dir[0] + dir[1] * dir[1])));

        ASSERT_NEAR(check, pa[i], 1.0e-6);
    }

    toast::qa_to_angles(n, quat.data(), check_theta.data(),
                        check_phi.data(), check_pa.data(), false);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(theta[i], check_theta[i], 1.0e-6);

        check = check_phi[i];
        if (check < 0.0) {
            check += toast::TWOPI;
        }
        if (check >= toast::TWOPI) {
            check -= toast::TWOPI;
        }
        if (::fabs(check) < 2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }
        if (::fabs(check - toast::TWOPI) <
            2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }

        ASSERT_NEAR(phi[i], check, 1.0e-6);

        ASSERT_NEAR(pa[i], check_pa[i], 1.0e-6);
    }

    toast::qa_to_position(n, quat.data(), check_theta.data(),
                          check_phi.data());

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(theta[i], check_theta[i], 1.0e-6);

        check = check_phi[i];
        if (check < 0.0) {
            check += toast::TWOPI;
        }
        if (check >= toast::TWOPI) {
            check -= toast::TWOPI;
        }
        if (::fabs(check) < 2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }
        if (::fabs(check - toast::TWOPI) <
            2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }

        ASSERT_NEAR(phi[i], check, 1.0e-6);
    }

    // Now run tests in IAU convention...

    for (size_t i = 0; i < n_theta; ++i) {
        for (size_t j = 0; j < n_phi; ++j) {
            theta[i * n_phi + j] = (0.5 + (double)i) * toast::PI /
                                   (double)n_theta;
            phi[i * n_phi + j] = (double)j * toast::TWOPI / (double)n_phi;
            pa[i * n_phi + j] = -(double)j * toast::TWOPI / (double)n_phi +
                                toast::PI;
        }
    }

    // convert to quaternions

    toast::qa_from_angles(n, theta.data(), phi.data(), pa.data(),
                          quat.data(), true);

    // check that the resulting quaternions rotate the Z and X
    // axes to the correct place.

    for (size_t i = 0; i < n; ++i) {
        toast::qa_rotate(1, &(quat[4 * i]), 1, zaxis, dir);
        toast::qa_rotate(1, &(quat[4 * i]), 1, xaxis, orient);

        ASSERT_NEAR(toast::PI_2 - ::asin(dir[2]), theta[i], 1.0e-6);

        check = ::atan2(dir[1], dir[0]);

        if (check < 0.0) {
            check += toast::TWOPI;
        }
        if (check >= toast::TWOPI) {
            check -= toast::TWOPI;
        }
        if (::fabs(check) < 2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }
        if (::fabs(check - toast::TWOPI) <
            2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }

        ASSERT_NEAR(check, phi[i], 1.0e-6);

        check = -::atan2(orient[0] * dir[1] - orient[1] * dir[0],
                         -(orient[0] * dir[2] * dir[0])
                         - (orient[1] * dir[2] * dir[1])
                         + (orient[2] * (dir[0] * dir[0] + dir[1] * dir[1])));

        if (::fabs(::fabs(check - pa[i]) - toast::TWOPI) <
            std::numeric_limits <float>::epsilon()) {
            // we are at the same angle, just with 2PI rotation.
        } else if (::fabs(::fabs(pa[i] - check) - toast::TWOPI) <
                   std::numeric_limits <float>::epsilon()) {
            // we are at the same angle, just with 2PI rotation.
        } else {
            ASSERT_NEAR(check, pa[i], 1.0e-6);
        }
    }

    toast::qa_to_angles(n, quat.data(), check_theta.data(),
                        check_phi.data(), check_pa.data(), true);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(theta[i], check_theta[i], 1.0e-6);

        check = check_phi[i];
        if (check < 0.0) {
            check += toast::TWOPI;
        }
        if (check >= toast::TWOPI) {
            check -= toast::TWOPI;
        }
        if (::fabs(check) < 2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }
        if (::fabs(check - toast::TWOPI) <
            2.0 * std::numeric_limits <float>::epsilon()) {
            check = 0.0;
        }

        ASSERT_NEAR(phi[i], check, 1.0e-6);

        ASSERT_NEAR(pa[i], check_pa[i], 1.0e-6);
    }
}
