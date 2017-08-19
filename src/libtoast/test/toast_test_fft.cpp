/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const int64_t fftTest::length = 65536;
const int64_t fftTest::n = 5;

void fftTest::runbatch(int64_t nbatch) {
    std::vector < fft::fft_data > compare ( nbatch );

    // create FFT plans

    fft::r1d_p forward ( fft::r1d::create ( length, nbatch, fft::plan_type::fast, fft::direction::forward, 1.0 ) );
    fft::r1d_p reverse ( fft::r1d::create ( length, nbatch, fft::plan_type::fast, fft::direction::backward, 1.0 ) );

    // First generate some gaussian random noise

    for ( int64_t i = 0; i < nbatch; ++i ) {
        rng::dist_normal ( length, 0, 0, 0, i*length, forward->tdata()[i] );
        for ( int64_t j = 0; j < length; ++j ) {
            compare[i].push_back ( forward->tdata()[i][j] );
        }
    }

    // Do forward transform

    forward->exec();

    // Verify that normalization and spectrum are correct.

    double sigma = ((double)length / 2.0) * ::sqrt( 2.0 / ((double)length - 1.0));

    for ( int64_t i = 0; i < nbatch; ++i ) {
        double mean = 0.0;
        for ( int64_t j = 0; j < length; ++j ) {
            mean += forward->fdata()[i][j];
            //std::cout << forward->fdata()[0][j] << std::endl;
        }
        mean /= (double)length;
        //std::cout << "mean[" << i << "] = " << mean << std::endl;

        double var = 0.0;
        for ( int64_t j = 0; j < length; ++j ) {
            var += ( forward->fdata()[i][j] - mean ) * ( forward->fdata()[i][j] - mean );
        }
        var /= (double)length;

        double outlier = ::fabs( var - ((double)length / 2.0) );

        //std::cout << "var[" << i << "] = " << var << ", (len / 2) = " << ((double)length / 2.0) << " sigma = " << sigma << " outlier = " << outlier << std::endl;
        ASSERT_TRUE( outlier < 3.0 * sigma );
    }


    // Copy data to reverse transform

    for ( int64_t i = 0; i < nbatch; ++i ) {
        std::copy ( forward->fdata()[i], forward->fdata()[i] + length, reverse->fdata()[i] );
    }

    // Do reverse transform

    reverse->exec();

    // Verify roundtrip values

    for ( int64_t i = 0; i < nbatch; ++i ) {
        for ( int64_t j = 0; j < length; ++j ) {
            EXPECT_FLOAT_EQ( compare[i][j], reverse->tdata()[i][j] );
        }
    }

    return;
}


TEST_F( fftTest, roundtrip_single ) {
    runbatch(1);
}

TEST_F( fftTest, roundtrip_multi ) {
    runbatch(n);
}

