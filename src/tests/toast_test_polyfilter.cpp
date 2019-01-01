
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const int64_t TOASTpolyfilterTest::order = 3;
const int64_t TOASTpolyfilterTest::n = 1000;


TEST_F(TOASTpolyfilterTest, filter) {
    vector <double> signal1(n);
    vector <double> signal2(n);
    vector <double> signal3(n);
    vector <uint8_t> flags(n, 0);

    double * signals[3];

    signals[0] = signal1.data();
    signals[1] = signal2.data();
    signals[2] = signal3.data();

    for (int i = 0; i < n; ++i) {
        signal1[i] = 1;
        signal2[i] = i;
        signal3[i] = i * i;
    }
    size_t nsignal = 3;

    int64_t starts[] = {0, n / 2};
    int64_t stops[] = {n / 2 - 1, n - 1};
    size_t nscan = 2;

    double rms1start = 0, rms2start = 0, rms3start = 0;

    for (int64_t i = 0; i < n; ++i) {
        rms1start += signal1[i] * signal1[i];
        rms2start += signal2[i] * signal2[i];
        rms3start += signal3[i] * signal3[i];
    }

    rms1start = sqrt(rms1start / (double)n);
    rms2start = sqrt(rms2start / (double)n);
    rms3start = sqrt(rms3start / (double)n);

    toast::filter_polynomial(order, signals, flags.data(), n, nsignal,
                             starts, stops, nscan);

    double rms1 = 0, rms2 = 0, rms3 = 0;

    for (int64_t i = 0; i < n; ++i) {
        rms1 += signal1[i] * signal1[i];
        rms2 += signal2[i] * signal2[i];
        rms3 += signal3[i] * signal3[i];
    }

    rms1 = sqrt(rms1 / (double)n);
    rms2 = sqrt(rms2 / (double)n);
    rms3 = sqrt(rms3 / (double)n);

    EXPECT_LT(std::abs(rms1 / rms1start), 1e-10);
    EXPECT_LT(std::abs(rms2 / rms2start), 1e-10);
    EXPECT_LT(std::abs(rms3 / rms3start), 1e-10);
}


TEST_F(TOASTpolyfilterTest, filter_with_flags) {
    vector <double> signal1(n);
    vector <double> signal2(n);
    vector <double> signal3(n);
    vector <unsigned char> flags(n, 0);

    for (int64_t i = 0; i < n / 20; ++i) {
        flags[i] = 1;
        flags[16 * n / 20 + i] = 1;
    }

    int64_t ngood = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (flags[i]) continue;
        ngood++;
    }

    double * signals[3];

    signals[0] = signal1.data();
    signals[1] = signal2.data();
    signals[2] = signal3.data();

    for (int64_t i = 0; i < n; ++i) {
        signal1[i] = 1;
        signal2[i] = i;
        signal3[i] = i * i;
    }
    size_t nsignal = 3;

    int64_t starts[] = {0, n / 2};
    int64_t stops[] = {n / 2 - 1, n - 1};
    size_t nscan = 2;

    double rms1start = 0, rms2start = 0, rms3start = 0;

    for (int64_t i = 0; i < n; ++i) {
        if (flags[i]) continue;
        rms1start += signal1[i] * signal1[i];
        rms2start += signal2[i] * signal2[i];
        rms3start += signal3[i] * signal3[i];
    }

    rms1start = sqrt(rms1start / (double)ngood);
    rms2start = sqrt(rms2start / (double)ngood);
    rms3start = sqrt(rms3start / (double)ngood);

    toast::filter_polynomial(order, signals, flags.data(), n, nsignal,
                             starts, stops, nscan);

    double rms1 = 0, rms2 = 0, rms3 = 0;

    for (int64_t i = 0; i < n; ++i) {
        if (flags[i]) continue;
        rms1 += signal1[i] * signal1[i];
        rms2 += signal2[i] * signal2[i];
        rms3 += signal3[i] * signal3[i];
    }

    rms1 = sqrt(rms1 / (double)ngood);
    rms2 = sqrt(rms2 / (double)ngood);
    rms3 = sqrt(rms3 / (double)ngood);

    EXPECT_LT(std::abs(rms1 / rms1start), 1e-10);
    EXPECT_LT(std::abs(rms2 / rms2start), 1e-10);
    EXPECT_LT(std::abs(rms3 / rms3start), 1e-10);
}
