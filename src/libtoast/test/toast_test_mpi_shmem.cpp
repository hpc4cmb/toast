/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>

#include <cmath>


using namespace std;
using namespace toast;


const size_t TOASTmpiShmemTest::n = 100;


TEST_F( TOASTmpiShmemTest, instantiate ) {

    toast::mpi_shmem::mpi_shmem<double> shmem;
    shmem.allocate( n );

    toast::mpi_shmem::mpi_shmem<double> shmem2( n );

    size_t sz = shmem.size();

    EXPECT_EQ( shmem2.size(), n );

    EXPECT_EQ( shmem2.size(), shmem.size() );

}

TEST_F( TOASTmpiShmemTest, access ) {

    toast::mpi_shmem::mpi_shmem<double> shmem( n );

    shmem.set( 20 );

    shmem.resize( 2*n );

    double *p = shmem.data();

    shmem[n-1] = 10;

    EXPECT_FLOAT_EQ( shmem[n-2], 20 );
    EXPECT_FLOAT_EQ( shmem[n-1], 10 );
    EXPECT_EQ( shmem[n-1], p[n-1] );

}
