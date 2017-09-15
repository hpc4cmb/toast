/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const int64_t fftTest::length = 32;
const int64_t fftTest::n = 3;



void fftTest::runbatch(int64_t nbatch, fft::r1d_p forward, 
    fft::r1d_p reverse) {
    bool debug = false;

    if ( debug ) {
        std::cout << "------- FFT batch of " << nbatch << " --------" << std::endl;
    }

    std::vector < fft::fft_data > compare ( nbatch );

    // First generate some gaussian random noise

    for ( int64_t i = 0; i < nbatch; ++i ) {
        rng::dist_normal ( length, 0, 0, 0, i*length, forward->tdata()[i] );
        compare[i].resize ( length );
        for ( int64_t j = 0; j < length; ++j ) {
            compare[i][j] = forward->tdata()[i][j];
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
        }
        mean /= (double)length;
        if ( debug ) {
            std::cout << "  fdata[" << i << "] mean = " << mean << std::endl;
        }

        double var = 0.0;
        for ( int64_t j = 0; j < length; ++j ) {
            var += ( forward->fdata()[i][j] - mean ) * ( forward->fdata()[i][j] - mean );
        }
        var /= (double)length;

        double outlier = ::fabs( var - ((double)length / 2.0) );

        if ( debug ) {
            std::cout << "  fdata[" << i << "] var = " << var << 
                ", (len / 2) = " << ((double)length / 2.0) << 
                " sigma = " << sigma << " outlier = " << outlier 
                << std::endl;
        }

        if ( ! debug ) {
            ASSERT_TRUE( outlier < 3.0 * sigma );
        }
    }

    // Copy data to reverse transform

    for ( int64_t i = 0; i < nbatch; ++i ) {
        std::copy ( forward->fdata()[i], forward->fdata()[i] + length, reverse->fdata()[i] );
    }

    // Do reverse transform

    reverse->exec();

    // Verify roundtrip values

    for ( int64_t i = 0; i < nbatch; ++i ) {
        if ( debug ) {
            std::cout << "  fft " << i << ":" << std::endl;
        }
        for ( int64_t j = 0; j < length; ++j ) {
            if ( debug ) {
                std::cout << "    (" << j << ")" << compare[i][j] << " " << forward->fdata()[i][j] << " " << reverse->tdata()[i][j] << std::endl;
            } else {
                EXPECT_FLOAT_EQ( compare[i][j], reverse->tdata()[i][j] );
            }
        }
    }

    return;
}


TEST_F( fftTest, roundtrip_single ) {
    // create FFT plans
    fft::r1d_p fplan ( fft::r1d::create ( length, 1, fft::plan_type::fast, 
        fft::direction::forward, 1.0 ) );
    fft::r1d_p rplan ( fft::r1d::create ( length, 1, fft::plan_type::fast, 
        fft::direction::backward, 1.0 ) );
    // run test
    runbatch(1, fplan, rplan);
}

TEST_F( fftTest, roundtrip_multi ) {
    // create FFT plans
    fft::r1d_p fplan ( fft::r1d::create ( length, n, fft::plan_type::fast, 
        fft::direction::forward, 1.0 ) );
    fft::r1d_p rplan ( fft::r1d::create ( length, n, fft::plan_type::fast, 
        fft::direction::backward, 1.0 ) );
    // run test
    runbatch(n, fplan, rplan);
}

TEST_F( fftTest, plancache_single ) {
    // use the plan store.  test both reuse of plans and
    // creation after a clear().
    fft::r1d_plan_store & store = fft::r1d_plan_store::get();
    store.clear();

    fft::r1d_p fplan = store.forward ( length, 1 );
    fft::r1d_p rplan = store.backward ( length, 1 );
    runbatch(1, fplan, rplan);

    fplan = store.forward ( length, 1 );
    rplan = store.backward ( length, 1 );
    runbatch(1, fplan, rplan);

    store.clear();
    fplan = store.forward ( length, 1 );
    rplan = store.backward ( length, 1 );
    runbatch(1, fplan, rplan);

}


TEST_F( fftTest, plancache_multi ) {
    // use the plan store.  test both reuse of plans and
    // creation after a clear().
    fft::r1d_plan_store & store = fft::r1d_plan_store::get();
    store.clear();

    fft::r1d_p fplan = store.forward ( length, n );
    fft::r1d_p rplan = store.backward ( length, n );
    runbatch(n, fplan, rplan);

    fplan = store.forward ( length, n );
    rplan = store.backward ( length, n );
    runbatch(n, fplan, rplan);

    store.clear();

    fplan = store.forward ( length, n );
    rplan = store.backward ( length, n );
    runbatch(n, fplan, rplan);

}

