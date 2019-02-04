
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_mpi.hpp>


void toast::mpi_init(int argc, char * argv[]) {
    // If MPI is not yet initialized (by mpi4py or some other place),
    // then initialize it here.

    int ret;
    int initialized;
    int threadprovided;
    int rank;

    ret = MPI_Initialized(&initialized);

    if (!initialized) {
        ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                              &threadprovided);
    }

    return;
}

void toast::mpi_finalize() {
    int ret = MPI_Finalize();
    return;
}
