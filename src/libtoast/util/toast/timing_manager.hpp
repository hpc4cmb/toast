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

namespace toast
{
namespace util
{

//----------------------------------------------------------------------------//

class timing_manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef toast::util::timer                  toast_timer_t;
    typedef toast_timer_t::string_t             string_t;
    typedef std::pair<string_t, toast_timer_t&> timer_pair_t;
    typedef std::deque<timer_pair_t>            timer_list_t;
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
	// Virtual destructors are required by abstract classes 
	// so add it by default, just in case
    virtual ~timing_manager();

public:
    // Public functions
    static timing_manager* instance();

    size_type size() const { return m_timer_list.size(); }
    void clear() { m_timer_list.clear(); m_timer_map.clear(); }

    toast_timer_t& timer(const string_t& key);

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

    // iteration of timers
    iterator        begin()         { return m_timer_list.begin(); }
    const_iterator  begin() const   { return m_timer_list.begin(); }
    const_iterator  cbegin() const  { return m_timer_list.begin(); }

    iterator        end()           { return m_timer_list.end(); }
    const_iterator  end() const     { return m_timer_list.end(); }
    const_iterator  cend() const    { return m_timer_list.end(); }

    void report() const;
    void set_output_streams(ostream_t&, ostream_t&);
    void set_output_streams(const string_t&, const string_t&);

    toast_timer_t& at(size_type i) { return m_timer_list.at(i).second; }
    toast_timer_t& at(string_t i);

protected:
    ofstream_t* get_ofstream(ostream_t* m_os) const;

private:
	// Private variables
    static timing_manager*   fgInstance;
    // hashed string map for fast lookup
    timer_map_t             m_timer_map;
    // ordered list for output (outputs in order of timer instantiation)
    timer_list_t            m_timer_list;
    // output stream for total timing report
    ostream_t*              m_report_tot;
    // output stream for average timing report
    ostream_t*              m_report_avg;
};

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
inline void
timing_manager::report() const
{
    ostream_t* os_avg = m_report_avg;
    ostream_t* os_tot = m_report_tot;

    auto check_stream = [&] (ostream_t*& os, const string_t& id)
    {
        if(os == &std::cout)
            return;
        ofstream_t* fos = get_ofstream(os);
        if(!(*fos && fos->is_open()))
        {
            std::cerr << "Output stream for " << id << " is not open/valid"
                      << ". Redirecting to stdout..." << std::endl;
            os = &std::cout;
        }
    };

    check_stream(os_avg, "average timing report");
    check_stream(os_tot, "total timing report");

    for(const auto& itr : *this)
        itr.second.stop();

    for(const auto& itr : *this)
        itr.second.report(*os_tot);

    for(const auto& itr : *this)
        itr.second.report_average(*os_avg);

    os_avg->flush();
    os_tot->flush();
}
//----------------------------------------------------------------------------//
inline void
timing_manager::set_output_streams(ostream_t& _tot_os, ostream_t& _avg_os)
{
    m_report_tot = &_tot_os;
    m_report_avg = &_avg_os;
}
//----------------------------------------------------------------------------//
inline void
timing_manager::set_output_streams(const string_t& totf, const string_t& avgf)
{
    auto ostreamop = [&] (ostream_t*& m_os, const string_t& _fname)
    {
        if(m_os != &std::cout)
            delete m_os;

        auto* _tos = new ofstream_t;
        _tos->open(_fname);
        if(*_tos)
            m_os = _tos;
        else
        {
            std::cerr << "Warning! Unable to open file " << _fname
                      << ". Redirecting to stdout..." << std::endl;
            m_os = &std::cout;
        }
    };

    ostreamop(m_report_tot, totf);
    ostreamop(m_report_avg, avgf);
}
//----------------------------------------------------------------------------//
inline timing_manager::toast_timer_t&
timing_manager::at(string_t i)
{
    if(m_timer_map.find(i) == m_timer_map.end())
        return this->timer(i);
    return m_timer_map[i];
}
//----------------------------------------------------------------------------//
inline timing_manager::ofstream_t*
timing_manager::get_ofstream(ostream_t* m_os) const
{
    return dynamic_cast<ofstream_t*>(m_os);
}
//----------------------------------------------------------------------------//

} // namespace util

} // namespace toast

#endif // timing_manager_hpp_
