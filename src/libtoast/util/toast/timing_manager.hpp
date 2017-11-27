/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#ifndef timing_manager_hpp_
#define timing_manager_hpp_

//----------------------------------------------------------------------------//

#include <unordered_map>
#include <deque>
#include <string>
#include "timer.hpp"

#include <mpi.h>

#include <cereal/cereal.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/archives/adapters.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>

namespace toast
{
namespace util
{

//----------------------------------------------------------------------------//

inline bool mpi_is_initialized()
{
    int32_t _init = 0;
    MPI_Initialized(&_init);
    return (_init != 0) ? true : false;
}

//----------------------------------------------------------------------------//

inline int32_t mpi_rank()
{
    int32_t _rank = 0;
    if(mpi_is_initialized())
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    return std::max(_rank, (int32_t) 0);
}

//----------------------------------------------------------------------------//

inline int32_t mpi_size()
{
    int32_t _size = 1;
    if(mpi_is_initialized())
        MPI_Comm_size(MPI_COMM_WORLD, &_size);
    return std::max(_size, (int32_t) 1);
}

//----------------------------------------------------------------------------//

struct timer_tuple : public std::tuple<std::string, std::string, toast::util::timer&>
{
    typedef std::string                                     string_t;
    typedef toast::util::timer                              toast_timer_t;
    typedef std::tuple<string_t, string_t, toast_timer_t&>  base_type;
    typedef string_t                                        first_type;
    typedef string_t                                        second_type;
    typedef toast_timer_t&                                  third_type;

    timer_tuple(const base_type& _data) : base_type(_data) { }
    timer_tuple(first_type _f, second_type _s, third_type _t)
    : base_type(_f, _s, _t) { }

    timer_tuple& operator=(const base_type& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator =(rhs);
        return *this;
    }

    first_type key() { return std::get<0>(*this); }
    const first_type key() const { return std::get<0>(*this); }

    second_type tag() { return std::get<1>(*this); }
    const second_type tag() const { return std::get<1>(*this); }

    third_type timer() { return std::get<2>(*this); }
    const third_type timer() const { return std::get<2>(*this); }

    // serialization function
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(cereal::make_nvp("timer.key", std::get<0>(*this)),
           cereal::make_nvp("timer.tag", std::get<1>(*this)),
           cereal::make_nvp("timer.ref", std::get<2>(*this)));
    }
};

//----------------------------------------------------------------------------//

class timing_manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef toast::util::timer                  toast_timer_t;
    typedef toast_timer_t::string_t             string_t;
    typedef timer_tuple                         timer_tuple_t;
    typedef std::deque<timer_tuple_t>           timer_list_t;
    typedef timer_list_t::iterator              iterator;
    typedef timer_list_t::const_iterator        const_iterator;
    typedef timer_list_t::size_type             size_type;
    typedef uomap<string_t, toast_timer_t>      timer_map_t;
    typedef toast_timer_t::ostream_t            ostream_t;
    typedef toast_timer_t::ofstream_t           ofstream_t;
    typedef toast::clock_type                   clock_type;

public:
	// Constructor and Destructors
    timing_manager();
    virtual ~timing_manager();

public:
    // Public static functions
    static timing_manager* instance();
    static void write_json(const string_t& _fname);

public:
    // Public member functions
    size_type size() const { return m_timer_list.size(); }
    void clear();

    toast_timer_t& timer(const string_t& key,
                         const string_t& tag = "cxx",
                         int32_t ncount = -1,
                         int32_t nhash = 0);

    // time a function with a return type and no arguments
    template <typename _Ret, typename _Func>
    _Ret time(const string_t& key, _Func);

    // time a function with a return type and arguments
    template <typename _Ret, typename _Func, typename... _Args>
    _Ret time(const string_t& key, _Func, _Args...);

    // time a function with no return type and no arguments
    template <typename _Func>
    void time(const string_t& key, _Func);

    // time a function with no return type and arguments
    template <typename _Func, typename... _Args>
    void time(const string_t& key, _Func, _Args...);

    // serialization function
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/);

    // iteration of timers
    iterator        begin()         { return m_timer_list.begin(); }
    const_iterator  begin() const   { return m_timer_list.cbegin(); }
    const_iterator  cbegin() const  { return m_timer_list.cbegin(); }

    iterator        end()           { return m_timer_list.end(); }
    const_iterator  end() const     { return m_timer_list.cend(); }
    const_iterator  cend() const    { return m_timer_list.cend(); }

    void report() const;
    void set_output_stream(ostream_t&);
    void set_output_stream(const string_t&);

    toast_timer_t& at(size_type i) { return m_timer_list.at(i).timer(); }
    toast_timer_t& at(string_t key, const string_t& tag = "cxx");

private:
    // Private functions
    ofstream_t* get_ofstream(ostream_t* m_os) const;
    void report(ostream_t*) const;

private:
	// Private variables
    static timing_manager*   fgInstance;
    // hashed string map for fast lookup
    timer_map_t             m_timer_map;
    // ordered list for output (outputs in order of timer instantiation)
    timer_list_t            m_timer_list;
    // output stream for total timing report
    ostream_t*              m_report;
};

//----------------------------------------------------------------------------//
inline void
timing_manager::clear()
{
    m_timer_list.clear();
    m_timer_map.clear();
    details::base_timer::get_instance_count() = 0;
}
//----------------------------------------------------------------------------//
template <typename _Ret, typename _Func>
inline _Ret
timing_manager::time(const string_t& key, _Func func)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.start();
    _Ret _ret = func();
    _t.stop();
    return _ret;
}
//----------------------------------------------------------------------------//
template <typename _Ret, typename _Func, typename... _Args>
inline _Ret
timing_manager::time(const string_t& key, _Func func, _Args... args)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.start();
    _Ret _ret = func(args...);
    _t.stop();
    return _ret;
}
//----------------------------------------------------------------------------//
template <typename _Func>
inline void
timing_manager::time(const string_t& key, _Func func)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.start();
    func();
    _t.stop();
}
//----------------------------------------------------------------------------//
template <typename _Func, typename... _Args>
inline void
timing_manager::time(const string_t& key, _Func func, _Args... args)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.start();
    func(args...);
    _t.stop();
}
//----------------------------------------------------------------------------//
template <typename Archive>
inline void
timing_manager::serialize(Archive& ar, const unsigned int /*version*/)
{
    ar(cereal::make_nvp("timers", m_timer_list));
}
//----------------------------------------------------------------------------//

} // namespace util

} // namespace toast

#endif // timing_manager_hpp_
