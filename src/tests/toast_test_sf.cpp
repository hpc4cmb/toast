
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>


const int TOASTsfTest::size = 1000;


void TOASTsfTest::SetUp() {
    angin.resize(size);
    sinout.resize(size);
    cosout.resize(size);
    xin.resize(size);
    yin.resize(size);
    atanout.resize(size);
    sqin.resize(size);
    sqout.resize(size);
    rsqin.resize(size);
    rsqout.resize(size);
    expin.resize(size);
    expout.resize(size);
    login.resize(size);
    logout.resize(size);

    for (int i = 0; i < size; ++i) {
        angin[i] = (double)(i + 1) * (2.0 * toast::PI / (double)(size + 1));
        sinout[i] = ::sin(angin[i]);
        cosout[i] = ::cos(angin[i]);
        xin[i] = cosout[i];
        yin[i] = sinout[i];
        atanout[i] = ::atan2(yin[i], xin[i]);

        sqin[i] = (double)i / (double)size;
        rsqin[i] = sqin[i];
        sqout[i] = ::sqrt(sqin[i]);
        rsqout[i] = 1.0 / ::sqrt(rsqin[i]);

        expin[i] = -10.0 + (double)i * 20.0 / (double)size;
        expout[i] = ::exp(expin[i]);
        login[i] = expout[i];
        logout[i] = ::log(login[i]);
    }

    return;
}

void TOASTsfTest::TearDown() {
    return;
}

TEST_F(TOASTsfTest, trig) {
    std::vector <double, toast::simd_allocator <double> > comp1(size);
    std::vector <double, toast::simd_allocator <double> > comp2(size);

    toast::vsin(size, angin.data(), comp1.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(sinout[i], comp1[i]);
    }

    toast::vcos(size, angin.data(), comp2.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(cosout[i], comp2[i]);
    }

    toast::vsincos(size, angin.data(), comp1.data(), comp2.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(sinout[i], comp1[i]);
        EXPECT_DOUBLE_EQ(cosout[i], comp2[i]);
    }

    toast::vatan2(size, yin.data(), xin.data(), comp1.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(atanout[i], comp1[i]);
    }
}


TEST_F(TOASTsfTest, fasttrig) {
    std::vector <double, toast::simd_allocator <double> > comp1(size);
    std::vector <double, toast::simd_allocator <double> > comp2(size);

    toast::vfast_sin(size, angin.data(), comp1.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(sinout[i], comp1[i]);
    }

    toast::vfast_cos(size, angin.data(), comp2.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(cosout[i], comp2[i]);
    }

    toast::vfast_sincos(size, angin.data(), comp1.data(), comp2.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(sinout[i], comp1[i]);
        EXPECT_FLOAT_EQ(cosout[i], comp2[i]);
    }

    toast::vfast_atan2(size, yin.data(), xin.data(), comp1.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(atanout[i], comp1[i]);
    }
}


TEST_F(TOASTsfTest, sqrtlog) {
    std::vector <double, toast::simd_allocator <double> > comp(size);

    toast::vsqrt(size, sqin.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(sqout[i], comp[i]);
    }

    toast::vrsqrt(size, rsqin.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        if (std::isfinite(comp[i]) || std::isfinite(rsqout[i])) {
            EXPECT_DOUBLE_EQ(rsqout[i], comp[i]);
        }
    }

    toast::vexp(size, expin.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(expout[i], comp[i]);
    }

    toast::vlog(size, login.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(logout[i], comp[i]);
    }
}


TEST_F(TOASTsfTest, fast_sqrtlog) {
    std::vector <double, toast::simd_allocator <double> > comp(size);

    toast::vfast_sqrt(size, sqin.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(sqout[i], comp[i]);
    }

    toast::vfast_rsqrt(size, rsqin.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        if (std::isfinite(comp[i]) || std::isfinite(rsqout[i])) {
            EXPECT_FLOAT_EQ(rsqout[i], comp[i]);
        }
    }

    toast::vfast_exp(size, expin.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(expout[i], comp[i]);
    }

    toast::vfast_log(size, login.data(), comp.data());
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(logout[i], comp[i]);
    }
}


TEST_F(TOASTsfTest, fast_erfinv) {
    std::vector <double, toast::simd_allocator <double> > in = {
        -9.990000e-01,
        -7.770000e-01,
        -5.550000e-01,
        -3.330000e-01,
        -1.110000e-01,
        1.110000e-01,
        3.330000e-01,
        5.550000e-01,
        7.770000e-01,
        9.990000e-01
    };

    std::vector <double, toast::simd_allocator <double> > check = {
        -2.326753765513524e+00,
        -8.616729665092674e-01,
        -5.400720684419686e-01,
        -3.042461029341061e-01,
        -9.869066534119145e-02,
        9.869066534119164e-02,
        3.042461029341063e-01,
        5.400720684419690e-01,
        8.616729665092677e-01,
        2.326753765513546e+00
    };

    std::vector <double, toast::simd_allocator <double> > out(10);

    toast::vfast_erfinv(10, in.data(), out.data());

    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(out[i], check[i]);
    }
}
