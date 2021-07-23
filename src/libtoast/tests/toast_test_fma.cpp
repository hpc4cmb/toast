
// Copyright (c) 2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>


const int TOASTfmaTest::n_out = 128;
const int TOASTfmaTest::n_weights = 64;


void TOASTfmaTest::SetUp() {
    weights.resize(n_weights);
    answer.resize(n_out);
    arrays_flat.resize(n_weights * n_out);

    // create some inputs that are different for different i, j
    // and also the final result would be integer
    for (int i = 0; i < n_weights; ++i) {
        weights[i] = (double)12 * (i + 1);
    }
    for (int i = 0; i < n_weights; ++i) {
        for (int j = 0; j < n_out; ++j) {
            arrays_flat[i * n_out + j] = (double)(i + 1) * (i + j + 1) * (j + 1);
        }
    }

    // and calculate the analytically expected value and compare later
    const int n = n_weights;
    for (int j = 0; j < n_out; ++j) {
        answer[j] = (double)(1 + j) * n * (1 + n) * (3 * n * (1 + n) + j * (2 + 4 * n));
    }
    return;
}

TEST_F(TOASTfmaTest, inplace_weighted_sum) {
    // not in SetUp: this should be done just before passing to the function below that
    // modifies it
    out.assign(n_out, 0);
    double ** arrays = new double *[n_weights];
    for (int i = 0; i < n_weights; ++i) {
        arrays[i] = &arrays_flat[i * n_out];
    }
    toast::inplace_weighted_sum(n_out, n_weights, out.data(), weights.data(), arrays);
    delete[] arrays;

    for (int j = 0; j < n_out; ++j) {
        EXPECT_DOUBLE_EQ(out[j], answer[j]);
    }
}
