/*
  Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
  All rights reserved.  Use of this source code is governed by
  a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_MPI_SHMEM_HPP
#define TOAST_MPI_SHMEM_HPP

#include "mpi.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdlib.h>

namespace toast { namespace mpi_shmem {

    template < typename T >
    class mpi_shmem {

        public:

            mpi_shmem ( MPI_Comm comm=MPI_COMM_WORLD ) : comm_( comm ) {

                // Split the provided communicator into groups that share
                // memory (are on the same node).

                if ( MPI_Comm_split_type( comm, MPI_COMM_TYPE_SHARED, 0,
                                          MPI_INFO_NULL, &shmcomm_ ) )
                    throw std::runtime_error( "Failed to split communicator by node." );

                if ( MPI_Comm_size( shmcomm_, &ntasks_ ) )
                    throw std::runtime_error( "Failed to get node communicator size" );

                if ( MPI_Comm_rank( shmcomm_, &rank_ ) )
                    throw std::runtime_error( "Failed to get node communicator rank" );
            }

            mpi_shmem ( size_t n, MPI_Comm comm=MPI_COMM_WORLD ) : mpi_shmem( comm ) {
                allocate( n );
            }

            T operator[] ( int i ) const { return global_[i]; }
            T & operator[] ( int i ) { return global_[i]; }


            T * allocate( size_t n ) {

                // Determine the number of elements each process should offer
                // for the shared allocation

                nlocal_ = n / ntasks_;

                if ( nlocal_ * ntasks_ < n ) nlocal_ += 1;
                if ( nlocal_ * (rank_ + 1) > n ) nlocal_ = n - nlocal_ * rank_;
                if ( nlocal_ < 0 ) nlocal_ = 0;

                // Allocate the shared memory

                if ( MPI_Win_allocate_shared(
                         nlocal_ * sizeof( T ), sizeof( T ), MPI_INFO_NULL, shmcomm_,
                         &local_, &win_ ) ) {
                    std::ostringstream o;
                    o << " Failed to allocate " << n / 1024. / 1024.
                      << " MB of shared memory with " << ntasks_
                      <<" tasks.";
                    throw std::runtime_error( o.str().c_str() );
                }
                n_ = n;

                // Get a pointer to the beginning of the shared memory
                // on rank # 0

                MPI_Aint nn;
                int disp;

                if ( MPI_Win_shared_query( win_, 0, &nn, &disp, &global_ ) )
                    throw std::runtime_error( "Failed to query shared memory address." );

                return global_;
            }

            void free() {
                if ( global_ ) {
                    if ( MPI_Win_free( &win_ ) )
                        throw std::runtime_error( "Failed to free shared memory." );
                    local_ = NULL;
                    global_ = NULL;
                    n_ = 0;
                    nlocal_ = 0;
                }
            }

            void set( T val ) {
                for ( int i=0; i<nlocal_; ++i )
                    local_[i] = val;
            }

            T * resize( size_t n ) {

                // If there is memory already allocated, preserve its
                // contents.

                size_t n_copy = 0;
                T *temp;

                if (n < n_) {
                    // We are shrinking the memory
                    n_copy = n;
                } else if (n_ > 0) {
                    // We are expanding the memory AND
                    // we currently have a non-zero memory size
                    n_copy = n_;
                }

                if ( n_copy > 0 && rank_ == 0 ) {
                    temp = ( T* ) std::malloc( sizeof(T) * n_copy );
                    std::memcpy( temp, global_, sizeof(T) * n_copy );
                }

                free();
                allocate( n );

                if ( n_copy > 0 && rank_ == 0 ) {
                    std::memcpy( global_, temp, sizeof(T) * n_copy );
                    std::free( temp );
                }

                return global_;
            }

            T * data() { return global_; }
            size_t size() { return n_; }
            int rank() { return rank_; }
            int ntasks() { return ntasks_; }

            ~mpi_shmem() { free(); }

        private:

            T * local_ = NULL;
            T * global_ = NULL;
            size_t n_ = 0, nlocal_ = 0;
            MPI_Comm comm_, shmcomm_;
            MPI_Win win_ = MPI_WIN_NULL;
            int ntasks_, rank_;

    };

} }

#endif
