/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>


int toast::test::runner ( int argc, char *argv[] ) {

    ::testing::GTEST_FLAG(filter) = std::string("TOAST*");
    ::testing::InitGoogleTest ( &argc, argv );

    toast::init ( argc, argv );

    // Disable result printing from all processes except the root one.

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    ::testing::TestEventListeners & listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    if ( rank != 0 ) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // FIXME:  rank 0 of MPI_COMM_WORLD should create the test
    // output directory here if it does not exist.  Currently none of the
    // C++ unit tests write or read data, so this is not yet an issue.

    int result = RUN_ALL_TESTS();

    return result;
}
