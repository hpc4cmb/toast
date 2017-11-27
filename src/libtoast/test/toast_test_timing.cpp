/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include <toast_test.hpp>
#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>

#include <toast/timing_manager.hpp>
#include <toast/auto_timer.hpp>

using namespace std;
using namespace toast;
using namespace toast::util;

typedef toast::util::timer          toast_timer_t;
typedef toast::util::timing_manager timing_manager_t;

typedef std::chrono::duration<int64_t>                      seconds_type;
typedef std::chrono::duration<int64_t, std::milli>          milliseconds_type;
typedef std::chrono::duration<int64_t, std::ratio<60*60>>   hours_type;

typedef toast::util::auto_timer                     auto_timer_t;
typedef std::string                                 string_t;

#define TOAST_FUNCTION_TIME(num) \
    auto_timer_t macro_auto_timer(string_t("") + \
        string_t(__FUNCTION__) + num, __LINE__);

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
    ss << "(" << n << ")";
    TOAST_FUNCTION_TIME(ss.str());
    return tman->time<int64_t>("fibonacci" + ss.str(), fibonacci, n);
}
//----------------------------------------------------------------------------//
// time fibonacci with return type and no arguments
// e.g. std::function < int32_t () >
int64_t time_fibonacci_l(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "(" << n << ")";
    TOAST_FUNCTION_TIME(ss.str());
    auto _fib = [=] () -> int64_t { return fibonacci(n); };
    return tman->time<int64_t>("fibonacci" + ss.str(), _fib);
}
//----------------------------------------------------------------------------//
// time fibonacci with no return type and arguments
// e.g. std::function < void ( int32_t ) >
void time_fibonacci_v(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "(" << n << ")";
    TOAST_FUNCTION_TIME(ss.str());
    tman->time("fibonacci" + ss.str(), fibonacci, n);
}
//----------------------------------------------------------------------------//
// time fibonacci with no return type and no arguments
// e.g. std::function < void () >
void time_fibonacci_lv(int32_t n)
{
    timing_manager_t* tman = timing_manager_t::instance();
    std::stringstream ss;
    ss << "(" << n << ")";
    TOAST_FUNCTION_TIME(ss.str());
    auto _fib = [=] () { fibonacci(n); };
    tman->time("fibonacci" + ss.str(), _fib);
}
//----------------------------------------------------------------------------//


//============================================================================//

TEST_F( timingTest, manager )
{
    timing_manager_t* tman = timing_manager_t::instance();
    tman->clear();

    toast_timer_t& t = tman->timer("tmanager_test");
    t.start();

    for(auto itr : { 35, 39, 43, 39, 35 })
    {
        time_fibonacci_v(itr-2);
        time_fibonacci_l(itr-1);
        time_fibonacci_b(itr);
        time_fibonacci_lv(itr+1);
    }

    t.stop();

    tman->set_output_stream("timing_report.out");
    tman->report();
    tman->write_json("timing_report.json");

    EXPECT_EQ(timing_manager::instance()->size(), 25);

    for(const auto& itr : *tman)
    {
        ASSERT_FALSE(itr.timer().real_elapsed() < 0.0);
        ASSERT_FALSE(itr.timer().user_elapsed() < 0.0);
    }

}

//============================================================================//
