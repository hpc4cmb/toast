/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_internal.hpp>

#include <unistd.h>
#include <climits>
#include <thread>

#ifdef HAVE_ELEMENTAL
#   include <El.hpp>
#endif

#ifdef USE_TBB
#   include <tbb/tbb.h>
#   include <tbb/task_scheduler_init.h>
#endif

#ifdef _OPENMP
#   include <omp.h>
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
    int rank;

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

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // THREADING
    // default to number of cores
    int32_t hw_threads = std::thread::hardware_concurrency();
    // get OMP_NUM_THREADS env variable
    int32_t omp_nthreads = toast::get_env<int32_t>("OMP_NUM_THREADS",
                                                   hw_threads);
    // toast::get_num_threads checks for environment TOAST_NUM_THREADS
    // and if undefined it uses std::thread::hardware_conncurrency()
    int32_t toast_nthreads = toast::get_num_threads();
    // if TOAST_NUM_THREADS not defined, use OMP_NUM_THREADS
    if(!std::getenv("TOAST_NUM_THREADS"))
        toast_nthreads = omp_nthreads;

    // Initialize the TBB task scheduler after MPI has been initialized
#   if defined(USE_TBB)
    // only create if doesn't exist already
    if(!tbb_scheduler)
        tbb_scheduler = new tbb::task_scheduler_init(toast_nthreads);
    // report
    if(rank == 0)
        std::cout << "TOAST number of threads (used by TBB): "
                  << toast_nthreads << std::endl;
#   endif

    // Enable the signal detection
    toast::EnableSignalDetection();

    // Verbose parameter (echo's signal detection info and possibly others
    // in future)
    if(toast::get_env<int32_t>("TOAST_VERBOSE", 0) > 0)
        std::cout << toast::signal_settings::str() << std::endl;

    // the function called when exiting with error code
    auto _exit_func = [] (int errcode)
    {
        auto tman = toast::util::timing_manager::instance();
        std::stringstream sserr;
        tman->report(sserr);
        std::cerr << sserr.str() << std::endl;
        std::stringstream ss;
        ss << "timing_report_err_" << errcode << ".out";
        std::ofstream ferr(ss.str().c_str());
        tman->report(ferr);
    };

    // set the exit function
    toast::signal_settings::set_exit_action(_exit_func);

#if defined(_OPENMP)
    // If TOAST_NUM_THREADS not defined, we use OMP_NUM_THREADS.
    // However, if TOAST_NUM_THREADS is defined, we let it take precedence over
    // OMP_NUM_THREADS since TOAST_NUM_THREADS is specific to our application
    // and OMP_NUM_THREADS is may be used to set other libraries etc. using
    // OpenMP
    if(omp_nthreads < toast_nthreads) {
        if(rank == 0) {
            std::cerr << "Warning! Overridding OMP_NUM_THREADS (= "
                      << omp_nthreads << ") with "
                      << "TOAST_NUM_THREADS (= " << toast_nthreads << ")..."
                      << std::endl;
        }
        omp_nthreads = toast_nthreads;
        omp_set_num_threads(toast_nthreads);
    }
    if(rank == 0) {
        std::cout << "OpenMP number of threads: "
                  << omp_nthreads << std::endl;
    }
#endif

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
