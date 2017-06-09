/*
  Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
  All rights reserved.  Use of this source code is governed by
  a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_MPI_SHMEM_HPP
#define TOAST_MPI_SHMEM_HPP

#include "mpi.h"


namespace toast { namespace mpi_shmem {

template < typename T >
class mpi_shmem {

public:

    mpi_shmem ( MPI_Comm comm=MPI_COMM_WORLD ) : comm( comm ) {

        // Split the provided communicator into groups that share
        // memory (are on the same node).

        MPI_Comm_split_type( comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                             &shmcomm );
        MPI_Comm_size( shmcomm, &ntasks );
        MPI_Comm_rank( shmcomm, &rank );

    }

    mpi_shmem ( size_t n, MPI_Comm comm=MPI_COMM_WORLD ) : mpi_shmem( comm ) {
        allocate( n );
    }

    T operator[] ( int i ) const { return global[i]; }
    T & operator[] ( int i ) { return global[i]; }
    

    T * allocate( size_t n ) {

        // Determine the number of elements each process should offer
        // for the shared allocation

        int my_n = n / ntasks;

        if ( my_n * ntasks < n ) my_n += 1;
        if ( my_n * (rank + 1) > n ) my_n = n - my_n * rank;
        if ( my_n < 0 ) my_n = 0;

        // Allocate the shared memory

        MPI_Win_allocate_shared( my_n * sizeof( T ), sizeof( T ), MPI_INFO_NULL,
                                 shmcomm, &local, &win );

        // Get a pointer to the beginning of the shared memory
        // on rank # 0

        MPI_Aint nn;
        int disp;

        MPI_Win_shared_query( win, 0, &nn, &disp, &global );

        return global;
    }

    void free() {

        if ( global ) {
            MPI_Win_free( &win );
            local = NULL;
            global = NULL;
            n = 0;
        }

    }

    T * resize( size_t n ) {
        free();

        return allocate( n );
    }


    T * data() { return global; }

    size_t size() { return n; }

    ~mpi_shmem() { free(); }

private:

    T * local = NULL;
    T * global = NULL;
    size_t n = 0;
    MPI_Comm comm, shmcomm;
    MPI_Win win=MPI_WIN_NULL;
    int ntasks, rank;

};

} }

#endif
