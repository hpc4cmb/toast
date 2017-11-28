/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef base_timer_hpp_
#define base_timer_hpp_

//----------------------------------------------------------------------------//

#include "base_clock.hpp"
#include <fstream>
#include <string>

#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/macros.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>

//----------------------------------------------------------------------------//

namespace toast
{

enum class clock_type
{
    wall,
    user,
    system,
    cpu,
    percent
};

namespace util
{
namespace details
{

//----------------------------------------------------------------------------//

class base_timer_data
{
public:
    typedef std::micro                                      ratio_t;
    typedef toast::util::base_clock<ratio_t>                clock_t;
    typedef clock_t::time_point                             time_point_t;
    typedef std::tuple<time_point_t, time_point_t>          data_type;
    typedef std::chrono::duration<clock_t, ratio_t>         duration_t;

public:
    time_point_t& start() { return std::get<0>(m_data); }
    time_point_t& stop() { return std::get<1>(m_data); }

    const time_point_t& start() const { return std::get<0>(m_data); }
    const time_point_t& stop() const { return std::get<1>(m_data); }

    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar( cereal::make_nvp("start", std::get<0>(m_data)),
            cereal::make_nvp("stop", std::get<1>(m_data)));
    }
protected:
    data_type m_data;
};

//----------------------------------------------------------------------------//

class base_timer
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef std::string                         string_t;
    typedef string_t::size_type                 size_type;
    typedef std::mutex                          mutex_t;
    typedef std::recursive_mutex                rmutex_t;
    typedef std::ostream                        ostream_t;
    typedef std::ofstream                       ofstream_t;
    typedef uomap<ostream_t*, rmutex_t>         mutex_map_t;
    typedef std::lock_guard<mutex_t>            auto_lock_t;
    typedef std::lock_guard<rmutex_t>           recursive_lock_t;
    typedef tms                                 tms_t;
    typedef base_timer_data                     data_t;
    typedef data_t::ratio_t                     ratio_t;
    typedef data_t::clock_t                     base_clock_t;
    typedef data_t::time_point_t                time_point_t;
    typedef std::vector<data_t>                 data_list_t;
    typedef data_t::duration_t                  duration_t;
    typedef base_timer                          this_type;
    typedef uomap<const base_timer*, data_t>    data_map_t;

public:
    base_timer(uint16_t = 3, const string_t& =
               "%w wall, %u user + %s system = %t CPU [seconds] (%p%)\n",
               ostream_t* = &std::cout);
    virtual ~base_timer();

    static uint64_t& get_instance_count() { return f_instance_count; }
    static uint64_t& get_instance_hash()  { return f_instance_hash; }

public:
    inline void start();
    inline void stop();
    inline bool is_valid() const;
    double real_elapsed() const;
    double system_elapsed() const;
    double user_elapsed() const;
    double cpu_elapsed() const { return user_elapsed() + system_elapsed(); }
    double cpu_utilization() const { return cpu_elapsed() / real_elapsed() * 100.; }
    inline const char* clock_time() const;
    inline size_type laps() const { return t_main_list.size(); }

public:
    void report(ostream_t&, bool endline = true, bool avg = false) const;
    inline void report(bool endline = true) const;
    inline void report_average(bool endline = true) const;
    inline void report_average(ostream_t& os, bool endline = true) const;

protected:
    typedef std::pair<size_type, clock_type>    clockpos_t;
    typedef std::pair<string_t,  clock_type>    clockstr_t;
    typedef std::vector<clockstr_t>             str_list_t;
    typedef std::vector<clockpos_t>             pos_list_t;

protected:
    void parse_format();
    virtual void compose() = 0;
    data_t& m_timer() const;

protected:
    // PODs
    mutable bool            m_running;
    uint16_t                m_precision;
    // pointers
    ostream_t*              m_os;
    // lists
    pos_list_t              m_format_positions;
    mutable data_list_t     t_main_list;
    // strings
    string_t                m_format_string;
    string_t                m_output_format;

private:
    // world mutex map, thread-safe ostreams
    static thread_local uint64_t    f_instance_count;
    static thread_local uint64_t    f_instance_hash;
    static thread_local data_map_t* f_data_map;
    static mutex_map_t              w_mutex_map;

public:
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(cereal::make_nvp("laps", t_main_list.size()),
           cereal::make_nvp("wall_elapsed", this->real_elapsed()),
           cereal::make_nvp("system_elapsed", this->system_elapsed()),
           cereal::make_nvp("user_elapsed", this->user_elapsed()),
           cereal::make_nvp("cpu_elapsed", this->cpu_elapsed()),
           cereal::make_nvp("cpu_util", this->cpu_utilization()),
           cereal::make_nvp("to_seconds_ratio_num", ratio_t::num),
           cereal::make_nvp("to_seconds_ratio_den", ratio_t::den),
           cereal::make_nvp("raw_history", t_main_list));
    }

};

//----------------------------------------------------------------------------//
// Print timer status n std::ostream
static inline
std::ostream& operator<<(std::ostream& os, const base_timer& t)
{
    bool restart = !t.is_valid();
    if(restart)
        const_cast<base_timer&>(t).stop();
    t.report(os);
    if(restart)
        const_cast<base_timer&>(t).start();

    return os;
}
//----------------------------------------------------------------------------//
inline                                                          // Wall time
double base_timer::real_elapsed() const
{
    if(m_running)
    {
        throw std::runtime_error("base_timer::real_elapsed() - InvalidCondition"
                                 " base_timer not stopped or times not recorded"
                                 "!");
    }

    double diff = 0.0;
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = std::get<2>(t_main_list[i].start().time_since_epoch().count().data);
        auto _te = std::get<2>(t_main_list[i].stop().time_since_epoch().count().data);
        diff += (_te - _ts);
    }

    return diff / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline                                                          // System time
double base_timer::system_elapsed() const
{
    if(m_running)
    {
        throw std::runtime_error("base_timer::system_elapsed() - "
                                 "InvalidCondition: base_timer not stopped or "
                                 "times not recorded!");
    }

    double diff = 0.0;
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = std::get<1>(t_main_list[i].start().time_since_epoch().count().data);
        auto _te = std::get<1>(t_main_list[i].stop().time_since_epoch().count().data);
        diff += (_te - _ts);
    }

    return diff / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline                                                          // CPU time
double base_timer::user_elapsed() const
{
    if(m_running)
    {
        throw std::runtime_error("base_timer::user_elapsed() - InvalidCondition"
                                 ": base_timer not stopped or times not "
                                 "recorded!");
    }

    double diff = 0.0;
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = std::get<0>(t_main_list[i].start().time_since_epoch().count().data);
        auto _te = std::get<0>(t_main_list[i].stop().time_since_epoch().count().data);
        diff += (_te - _ts);
    }

    return diff / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline
void base_timer::start()
{
    if(!m_running)
    {
        m_running = true;
        m_timer().start() = base_clock_t::now();
    }
}
//----------------------------------------------------------------------------//
inline
void base_timer::stop()
{
    if(m_running)
    {
        m_timer().stop() = base_clock_t::now();
        static mutex_t _mutex;
        auto_lock_t l(_mutex);
        t_main_list.push_back(m_timer());
        m_running = false;
    }
}
//----------------------------------------------------------------------------//
inline
bool base_timer::is_valid() const
{
    return (m_running) ? false : true;
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
inline
base_timer::data_t& base_timer::m_timer() const
{
    if(!f_data_map)
        f_data_map = new data_map_t();
    if(f_data_map->find(this) == f_data_map->end())
        f_data_map->insert(std::make_pair(this, data_t()));
    return f_data_map->find(this)->second;
}
//----------------------------------------------------------------------------//

} // namespace details

} // namespace util

} // namespace toast

//----------------------------------------------------------------------------//

namespace internal
{
typedef typename toast::util::details::base_timer_data::ratio_t base_ratio_t;
typedef toast::util::base_clock<base_ratio_t>   base_clock_t;
typedef toast::util::base_clock_data<base_ratio_t> base_clock_data_t;
typedef std::chrono::duration<base_clock_data_t, base_ratio_t> base_duration_t;
typedef std::chrono::time_point<base_clock_t, base_duration_t>  base_time_point_t;
typedef std::tuple<base_time_point_t, base_time_point_t> base_time_pair_t;
}

//----------------------------------------------------------------------------//

#endif // base_timer_hpp_
