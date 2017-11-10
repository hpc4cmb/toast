/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_MATH_INTERNAL_HPP
#define TOAST_MATH_INTERNAL_HPP

#include <config.h>
#include <toast/math.hpp>

#if defined(_OPENMP)
#   include <omp.h>
#endif

#if defined(HAVE_TBB)
#   include <tbb/tbb.h>
#   include <tbb/partitioner.h>
#   include <tbb/blocked_range.h>
#   include <tbb/parallel_for.h>
#endif

namespace toast {
namespace math {

//----------------------------------------------------------------------------//
//  This is the abstraction of running the parallel for loops
//  - OpenMP/serial implementation in execute_omp
//  - TBB implementation in execute_tbb
//
//  - use execute_mt "typedef" for compile time resolution of preferred method
//
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
// OpenMP/serial interface
template <typename InputIterator_t, typename Func_t>
void execute_omp(Func_t func,
                 InputIterator_t _beg,
                 InputIterator_t _end,
                 InputIterator_t _incr)
{
    InputIterator_t itr;
#   pragma omp parallel for schedule(static)
    for(itr = _beg; itr < _end; itr += _incr)
    {
        func(itr, itr+_incr);
    }
}

//----------------------------------------------------------------------------//
//  TBB interface
#if defined(HAVE_TBB)
template <typename InputIterator_t, typename Func_t>
void execute_tbb(Func_t func,
                 InputIterator_t _beg,
                 InputIterator_t _end,
                 InputIterator_t _incr)
{
#   if defined(USE_TBB_AFFINITY_PART)
    // affinity partitioner likes to determine it's own grainsize
    auto range = tbb::blocked_range<InputIterator_t>(_beg, _end);
    auto partitioner = tbb::affinity_partitioner();
#   else
    auto range = tbb::blocked_range<InputIterator_t>(_beg, _end, _incr);
    auto partitioner = tbb::auto_partitioner();
#   endif

    auto tbbrun = [=] (const tbb::blocked_range<InputIterator_t>& r)
    {
        func(r.begin(), r.end());
    };

    tbb::parallel_for(range, tbbrun, partitioner);
}
#endif

//----------------------------------------------------------------------------//
//  Generic function call interface for MT method
//----------------------------------------------------------------------------//

#if defined(USE_TBB) && defined(HAVE_TBB)

template <typename InputIterator_t, typename Func_t>
inline void execute_mt(Func_t f,
                InputIterator_t i1,
                InputIterator_t i2,
                InputIterator_t i3)
{
    execute_tbb<InputIterator_t, Func_t>(f, i1, i2, i3);
}

#else

template <typename InputIterator_t, typename Func_t>
inline void execute_mt(Func_t f,
                InputIterator_t i1,
                InputIterator_t i2,
                InputIterator_t i3)
{
    execute_omp<InputIterator_t, Func_t>(f, i1, i2, i3);
}

#endif

//----------------------------------------------------------------------------//

}   // namespace math
}   // namespace toast

#endif
