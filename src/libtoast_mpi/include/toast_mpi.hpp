
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MPI_HPP
#define TOAST_MPI_HPP

#include <mpi.h>

#include <toast.hpp>

#include <toast/mpi_shmem.hpp>
#include <toast/mpi_test.hpp>


namespace toast {
void mpi_init(int argc, char * argv[]);

void mpi_finalize();
}

#endif // ifndef TOAST_MPI_HPP
