/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef base_timer_hpp_
#define base_timer_hpp_

//----------------------------------------------------------------------------//

#include "base_clock.hpp"
#include "rss.hpp"
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
    typedef toast::rss::usage                   rss_usage_t;

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
    inline void rss_init();
    inline void rss_record();

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
    // memory usage
    rss_usage_t             m_rss_tot;
    rss_usage_t             m_rss_self;
    rss_usage_t             m_rss_tmp;

private:
    // world mutex map, thread-safe ostreams
    static thread_local uint64_t    f_instance_count;
    static thread_local uint64_t    f_instance_hash;
    static thread_local data_map_t* f_data_map;
    static mutex_map_t              w_mutex_map;

protected:
    template <int N> uint64_t get_min() const;
    template <int N> uint64_t get_max() const;
    template <int N> uint64_t get_sum() const;
    template <int N> uint64_t get_start(size_t i) const;
    template <int N> uint64_t get_stop(size_t i) const;

public:
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(cereal::make_nvp("laps", t_main_list.size()),
           // user clock elapsed
           cereal::make_nvp("user_elapsed",     get_sum<0>()),
           cereal::make_nvp("user_elapsed_min", get_min<0>()),
           cereal::make_nvp("user_elapsed_max", get_max<0>()),
           // system clock elapsed
           cereal::make_nvp("system_elapsed",      get_sum<1>()),
           cereal::make_nvp("system_elapsed_min",  get_min<1>()),
           cereal::make_nvp("system_elapsed_max",  get_max<1>()),
           // wall clock elapsed
           cereal::make_nvp("wall_elapsed",     get_sum<2>()),
           cereal::make_nvp("wall_elapsed_min", get_min<2>()),
           cereal::make_nvp("wall_elapsed_max", get_max<2>()),
           // cpu elapsed
           cereal::make_nvp("cpu_elapsed",     get_sum<3>()),
           cereal::make_nvp("cpu_elapsed_min", get_min<3>()),
           cereal::make_nvp("cpu_elapsed_max", get_max<3>()),
           // cpu utilization
           cereal::make_nvp("cpu_util",     get_sum<4>()),
           cereal::make_nvp("cpu_util_min", get_min<4>()),
           cereal::make_nvp("cpu_util_max", get_max<4>()),
           // conversion to seconds
           cereal::make_nvp("to_seconds_ratio_num", ratio_t::num),
           cereal::make_nvp("to_seconds_ratio_den", ratio_t::den),
           // memory usage
           cereal::make_nvp("rss_max", m_rss_tot),
           cereal::make_nvp("rss_self", m_rss_self));
    }

};

//----------------------------------------------------------------------------//
inline void base_timer::rss_init()
{
    m_rss_tmp.record();
}
//----------------------------------------------------------------------------//
inline void base_timer::rss_record()
{
    m_rss_self.record(m_rss_tmp);
    m_rss_tot.record();
}
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
        throw std::runtime_error("Error! base_timer::real_elapsed() - "
                                 "timer not stopped or no times recorded!");
    return get_sum<2>() / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline                                                          // System time
double base_timer::system_elapsed() const
{
    if(m_running)
        throw std::runtime_error("Error! base_timer::system_elapsed() - "
                                 "timer not stopped or no times recorded!");
    return get_sum<1>() / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline                                                          // CPU time
double base_timer::user_elapsed() const
{
    if(m_running)
        throw std::runtime_error("Error! base_timer::user_elapsed() - "
                                 "timer not stopped or no times recorded!");
    return get_sum<0>() / static_cast<double>(ratio_t::den);
}
//----------------------------------------------------------------------------//
inline
void base_timer::start()
{
    if(!m_running)
    {
        m_running = true;
        m_timer().start() = base_clock_t::now();
        rss_init();
    }
}
//----------------------------------------------------------------------------//
inline
void base_timer::stop()
{
    if(m_running)
    {
        m_timer().stop() = base_clock_t::now();
        rss_record();
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
template <int N> inline uint64_t
base_timer::get_start(size_t i) const
{
    return std::get<N>(t_main_list[i].start().time_since_epoch().count().data);
}
//----------------------------------------------------------------------------//
template <int N> inline uint64_t
base_timer::get_stop(size_t i) const
{
    return std::get<N>(t_main_list[i].stop().time_since_epoch().count().data);
}
//----------------------------------------------------------------------------//
template <int N> inline uint64_t
base_timer::get_sum() const
{
    uint64_t _val = 0;
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<N>(i);
        auto _te = get_stop<N>(i);
        _val += (_te > _ts) ? (_te - _ts) : uint64_t(0);
    }
    return _val;
}
//----------------------------------------------------------------------------//
template <int N> inline uint64_t
base_timer::get_min() const
{
    uint64_t _val = std::numeric_limits<uint64_t>::max();
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<N>(i);
        auto _te = get_stop<N>(i);
        uint64_t diff = (_te > _ts) ? (_te - _ts) : uint64_t(0);
        _val = std::min(_val, diff);
    }
    return _val;
}
//----------------------------------------------------------------------------//
template <int N> inline uint64_t
base_timer::get_max() const
{
    uint64_t _val = std::numeric_limits<uint64_t>::min();
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<N>(i);
        auto _te = get_stop<N>(i);
        uint64_t diff = (_te > _ts) ? (_te - _ts) : uint64_t(0);
        _val = std::max(_val, diff);
    }
    return _val;
}

//============================================================================//
//
//  Partial specializations
//
//============================================================================//

//----------------------------------------------------------------------------//
// partial specialization of get_sum() for CPU time
template <> inline uint64_t
base_timer::get_sum<3>() const
{
    return get_sum<0>() + get_sum<1>();
}
//----------------------------------------------------------------------------//
// partial specialization of get_min() for CPU time
template <> inline uint64_t
base_timer::get_min<3>() const
{
    uint64_t _val = std::numeric_limits<uint64_t>::max();
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<0>(i) + get_start<1>(i);
        auto _te = get_stop<0>(i) + get_stop<0>(i);
        uint64_t diff = (_te > _ts) ? (_te - _ts) : uint64_t(0);
        _val = std::min(_val, diff);
    }
    return _val;
}
//----------------------------------------------------------------------------//
// partial specialization of get_max() for CPU time
template <> inline uint64_t
base_timer::get_max<3>() const
{
    uint64_t _val = std::numeric_limits<uint64_t>::min();
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<0>(i) + get_start<1>(i);
        auto _te = get_stop<0>(i) + get_stop<0>(i);
        uint64_t diff = (_te > _ts) ? (_te - _ts) : uint64_t(0);
        _val = std::max(_val, diff);
    }
    return _val;
}
//----------------------------------------------------------------------------//
// partial specialization of get_sum() for CPU utilization
template <> inline uint64_t
base_timer::get_sum<4>() const
{
    return (100 * get_sum<3>()) / static_cast<double>(get_sum<2>());
}
//----------------------------------------------------------------------------//
// partial specialization of get_min() for CPU utilization
template <> inline uint64_t
base_timer::get_min<4>() const
{
    uint64_t _val = std::numeric_limits<uint64_t>::max();
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<0>(i) + get_start<1>(i);
        auto _te = get_stop<0>(i) + get_stop<0>(i);
        uint64_t _tn = (_te > _ts) ? (100 * (_te - _ts)) : uint64_t(0);
        auto _td = get_stop<2>(i) - get_start<2>(i);
        _val = std::min(_val, _tn / _td );
    }
    return _val;
}
//----------------------------------------------------------------------------//
// partial specialization of get_max() for CPU utilization
template <> inline uint64_t
base_timer::get_max<4>() const
{
    uint64_t _val = std::numeric_limits<uint64_t>::min();
    for(unsigned i = 0; i < t_main_list.size(); ++i)
    {
        auto _ts = get_start<0>(i) + get_start<1>(i);
        auto _te = get_stop<0>(i) + get_stop<0>(i);
        uint64_t _tn = (_te > _ts) ? (100 * (_te - _ts)) : uint64_t(0);
        auto _td = get_stop<2>(i) - get_start<2>(i);
        _val = std::max(_val, _tn / _td );
    }
    return _val;
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
