/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_internal.hpp>

#include <unistd.h>
#include <climits>

#ifdef HAVE_ELEMENTAL
#   include <El.hpp>
#endif

#ifdef USE_TBB
#   include <tbb/tbb.h>
#   include <tbb/task_scheduler_init.h>
#endif

// Initialize MPI in a consistent way

#ifdef USE_TBB
static tbb::task_scheduler_init* tbb_scheduler = nullptr;
#endif

//============================================================================//

void toast::init ( int argc, char *argv[] )
{
    int ret;
    int initialized;
    int threadprovided;

    ret = MPI_Initialized( &initialized );

    if ( ! initialized )
    {
#   if defined(HAVE_ELEMENTAL)
        // If we are using Elemental, let it initialize MPI
        El::Initialize ( argc, argv );
#   else
        ret = MPI_Init_thread ( &argc, &argv, MPI_THREAD_FUNNELED,
                                &threadprovided );
#   endif
    }

    // Initialize the TBB task scheduler after MPI has been initialized
#   if defined(USE_TBB)
    // toast::get_num_threads checks for environment TOAST_NUM_THREADS
    // and if undefined it uses std::thread::hardware_conncurrency()
    if(!tbb_scheduler)
        tbb_scheduler = new tbb::task_scheduler_init(toast::get_num_threads());
#   endif

    return;
}

//============================================================================//

void toast::finalize ( )
{
    int ret;

    // delete tbb task scheduler
#if defined(USE_TBB)
    // delete the task scheduler
    delete tbb_scheduler;
    tbb_scheduler = nullptr;
#endif // USE_TBB

#if defined(HAVE_ELEMENTAL)
    // If we are using Elemental, let it finalize MPI
    El::Finalize ( );
#else
    ret = MPI_Finalize ( );
#endif

    return;
}

//============================================================================//
