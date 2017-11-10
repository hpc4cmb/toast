//
//  Time Ordered Astrophysics Scalable Tools (TOAST)
//
//  Copyright (c) 2015-2017, The Regents of the University of California
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//  this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  andther materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//

#include <toast_test.hpp>
#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>

#include <toast/timing_manager.hpp>

using namespace std;
using namespace toast;
using namespace toast::util;

typedef toast::util::timer          toast_timer_t;
typedef toast::util::timing_manager timing_manager_t;

typedef std::chrono::duration<int64_t>                      seconds_type;
typedef std::chrono::duration<int64_t, std::milli>          milliseconds_type;
typedef std::chrono::duration<int64_t, std::ratio<60*60>>   hours_type;

// ASSERT_NEAR
// EXPECT_EQ
// EXPECT_FLOAT_EQ
// EXPECT_DOUBLE_EQ

//----------------------------------------------------------------------------//
// fibonacci calculation
int64_t fibonacci(int32_t n)
{
    if (n < 2) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}
//----------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t time_fibonacci_b(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "fibonacci(" << n << ")";
    return tman->time_function<int64_t>(ss.str(), fibonacci, n);
}
//----------------------------------------------------------------------------//
// time fibonacci with return type and no arguments
// e.g. std::function < int32_t () >
int64_t time_fibonacci_l(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "fibonacci(" << n << ")";
    auto _fib = [=] () -> int64_t { return fibonacci(n); };
    return tman->time_function<int64_t>(ss.str(), _fib);
}
//----------------------------------------------------------------------------//
// time fibonacci with no return type and arguments
// e.g. std::function < void ( int32_t ) >
void time_fibonacci_v(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "fibonacci(" << n << ")";
    tman->time_function(ss.str(), fibonacci, n);
}
//----------------------------------------------------------------------------//
// time fibonacci with no return type and no arguments
// e.g. std::function < void () >
void time_fibonacci_lv(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "fibonacci(" << n << ")";
    auto _fib = [=] () { fibonacci(n); };
    tman->time_function(ss.str(), _fib);
}
//----------------------------------------------------------------------------//


//============================================================================//

TEST_F( timingTest, manager )
{
    timing_manager_t* tman = timing_manager_t::instance();
    tman->clear();

    toast_timer_t& t = tman->timer("tmanager test");
    t.start();

    for(auto itr : { 39, 35, 43, 39 })
    {
        time_fibonacci_v(itr-2);
        time_fibonacci_l(itr-1);
        time_fibonacci_b(itr);
        time_fibonacci_lv(itr+1);
    }

    tman->set_output_streams("timing_report_tot.out", "timing_report_avg.out");
    tman->report();

    EXPECT_EQ(timing_manager::instance()->size(), 13);

    for(const auto& itr : *tman)
    {
        ASSERT_FALSE(itr.second.real_elapsed() < 0.0);
        ASSERT_FALSE(itr.second.user_elapsed() < 0.0);
    }

}

//============================================================================//
