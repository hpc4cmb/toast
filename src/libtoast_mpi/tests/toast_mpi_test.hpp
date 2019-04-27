
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MPI_TEST_TEST_HPP
#define TOAST_MPI_TEST_TEST_HPP

#include <toast_mpi.hpp>

#include <gtest/gtest.h>


class MPITOASTShmemTest : public ::testing::Test {
    public:

        MPITOASTShmemTest() {}

        ~MPITOASTShmemTest() {}

        virtual void SetUp() {}

        virtual void TearDown() {}

        static const size_t n;
};


#endif // ifndef TOAST_MPI_TEST_HPP
