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

#ifndef base_timer_hpp_
#define base_timer_hpp_

//----------------------------------------------------------------------------//

#include <unistd.h>
#include <sys/times.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <unordered_map>

//----------------------------------------------------------------------------//
// essentially an implementation of boost::auto_cpu_timer
//----------------------------------------------------------------------------//

namespace toast
{
namespace util
{
namespace details
{

class base_timer
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef std::string                 string_t;
    typedef string_t::size_type         size_type;
    typedef std::recursive_mutex        mutex_t;
    typedef std::ostream                ostream_t;
    typedef uomap<ostream_t*, mutex_t>  mutex_map_t;
    typedef std::lock_guard<mutex_t>    auto_lock_t;
    typedef tms                         tms_t;

public:
    base_timer(uint16_t = 3, const string_t& =
               "%w wall, %u user + %s system = %t CPU [seconds] (%p%)\n",
               ostream_t* = &std::cout);
    virtual ~base_timer();

public:
    inline void start();
    inline void stop();
    inline bool is_valid() const;
    double real_elapsed() const;
    double system_elapsed() const;
    double user_elapsed() const;
    inline const char* clock_time() const;
    inline void pause();
    inline void resume();
    inline void lap();
    inline size_type laps() const;

public:
    void report(ostream_t&, bool endline = true, bool avg = false) const;
    inline void report(bool endline = true) const;
    inline void report_average(bool endline = true) const;
    inline void report_average(ostream_t& os, bool endline = true) const;

protected:
    enum CLOCK_TYPE { WALL, USER, SYSTEM, CPU, PERCENT };

    typedef std::pair<size_type, CLOCK_TYPE>    clockpos_t;
    typedef std::pair<string_t,  CLOCK_TYPE>    clockstr_t;
    typedef std::vector<clockstr_t>             str_list_t;
    typedef std::vector<clockpos_t>             pos_list_t;

protected:
    struct timing
    {
        clock_t     m_start_real_time;
        clock_t     m_end_real_time;
        tms_t       m_start_times;
        tms_t       m_end_times;
    };

protected:
    typedef std::vector<timing>                 timing_list_t;

protected:
    void parse_format();

protected:
    mutable bool        m_valid_times;
    mutable bool        m_running;
    uint16_t            m_places;
    size_type           m_laps;
    string_t            m_format_string;
    string_t            m_output_format;
    // structures
    timing              m_main;
    timing              m_wait;
    // lists
    pos_list_t          m_format_positions;
    double              m_wait_history_stime;
    double              m_wait_history_utime;
    timing_list_t       m_wait_history;
    timing_list_t       m_lap_history;
    ostream_t*          m_os;

private:
    // world mutex map, thread-safe ostreams
    static mutex_map_t  w_mutex_map;

};

//----------------------------------------------------------------------------//
// Print timer status n std::ostream
static inline
std::ostream& operator<<(std::ostream& os, const base_timer& t)
{
    bool restart = !t.is_valid();
    if(restart)
        const_cast<base_timer&>(t).pause();
    t.report(os);
    if(restart)
        const_cast<base_timer&>(t).resume();

    return os;
}
//----------------------------------------------------------------------------//
inline
double base_timer::real_elapsed() const
{
    if (!m_valid_times)
    {
        throw std::runtime_error("base_timer::real_elapsed() - InvalidCondition"
                                 " base_timer not stopped or times not recorded"
                                 "!");
    }
    double diff = m_main.m_end_real_time - m_main.m_start_real_time;
    return diff/sysconf(_SC_CLK_TCK);
}
//----------------------------------------------------------------------------//
inline
double base_timer::system_elapsed() const
{
    if (!m_valid_times)
    {
        throw std::runtime_error("base_timer::system_elapsed() - "
                                 "InvalidCondition: base_timer not stopped or "
                                 "times not recorded!");
    }
    double diff = m_main.m_end_times.tms_stime - m_main.m_start_times.tms_stime;
    diff -= m_wait_history_stime;
    return diff/sysconf(_SC_CLK_TCK);
}
//----------------------------------------------------------------------------//
inline
double base_timer::user_elapsed() const
{
    if (!m_valid_times)
    {
        throw std::runtime_error("base_timer::user_elapsed() - InvalidCondition"
                                 ": base_timer not stopped or times not "
                                 "recorded!");
    }
    double diff = m_main.m_end_times.tms_utime - m_main.m_start_times.tms_utime;
    diff -= m_wait_history_utime;
    return diff/sysconf(_SC_CLK_TCK);
}
//----------------------------------------------------------------------------//
inline
void base_timer::start()
{
    if(m_running)
        return;
    m_valid_times = false;
    m_running = true;
    m_laps = 0;
    m_main.m_start_real_time = times(&m_main.m_start_times);
    m_wait.m_start_real_time = times(&m_wait.m_start_times);
}
//----------------------------------------------------------------------------//
inline
void base_timer::stop()
{
    if(!m_running)
        return;
    m_main.m_end_real_time = times(&m_main.m_end_times);
    m_valid_times = true;
    m_running = false;
}
//----------------------------------------------------------------------------//
inline
void base_timer::pause()
{
    m_valid_times = false;
    m_wait.m_start_real_time = times(&m_wait.m_start_times);
}
//----------------------------------------------------------------------------//
inline
void base_timer::resume()
{
    if(!m_running)
    {
        start();
        return;
    }
    m_wait.m_end_real_time = times(&m_wait.m_end_times);
    m_wait_history_stime +=
            m_wait.m_end_times.tms_stime - m_wait.m_start_times.tms_stime;
    m_wait_history_utime +=
            m_wait.m_end_times.tms_utime - m_wait.m_start_times.tms_utime;
    m_wait.m_end_real_time = m_wait.m_start_real_time;
    m_valid_times = true;
}
//----------------------------------------------------------------------------//
inline
void base_timer::lap()
{
    ++m_laps;
    m_valid_times = true;
}
//----------------------------------------------------------------------------//
inline
base_timer::size_type base_timer::laps() const
{
    return m_laps;
}
//----------------------------------------------------------------------------//
inline
bool base_timer::is_valid() const
{
    return m_valid_times;
}
//----------------------------------------------------------------------------//
inline const char* base_timer::clock_time() const
{
    time_t rawtime;
    struct tm* timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    return asctime (timeinfo);
}
//----------------------------------------------------------------------------//
inline
void base_timer::report(bool endline) const
{
    this->report(*m_os, endline);
}
//----------------------------------------------------------------------------//
inline
void base_timer::report_average(bool endline) const
{
    this->report(*m_os, endline, true);
}
//----------------------------------------------------------------------------//
inline
void base_timer::report_average(ostream_t& os, bool endline) const
{
    this->report(os, endline, true);
}
//----------------------------------------------------------------------------//

} // namespace details

} // namespace util

} // namespace toast

//----------------------------------------------------------------------------//

#endif // base_timer_hpp_
