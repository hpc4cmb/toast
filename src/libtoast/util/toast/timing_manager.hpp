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

#ifndef timing_manager_hpp_
#define timing_manager_hpp_

#include <unordered_map>
#include <deque>

#include <toast/timer.hpp>

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

    typedef toast::util::timer              toast_timer_t;
    typedef toast_timer_t::string_t               string_t;
    typedef std::pair<string_t, toast_timer_t&>   timer_pair_t;
    typedef std::deque<timer_pair_t>        timer_list_t;
    typedef timer_list_t::iterator          iterator;
    typedef timer_list_t::const_iterator    const_iterator;
    typedef timer_list_t::size_type         size_type;
    typedef uomap<string_t, toast_timer_t>        timer_map_t;

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

    template <typename _Ret, typename _Func>
    _Ret time_function(const string_t& key, _Func);

    template <typename _Ret, typename _Func, typename... _Args>
    _Ret time_function(const string_t& key, _Func, _Args...);

    template <typename _Func>
    void time_function(const string_t& key, _Func);

    template <typename _Func, typename... _Args>
    void time_function(const string_t& key, _Func, _Args...);

    iterator        begin()         { return m_timer_list.begin(); }
    const_iterator  begin() const   { return m_timer_list.begin(); }
    const_iterator  cbegin() const  { return m_timer_list.begin(); }

    iterator        end()           { return m_timer_list.end(); }
    const_iterator  end() const     { return m_timer_list.end(); }
    const_iterator  cend() const    { return m_timer_list.end(); }

private:
	// Private functions

private:
	// Private variables
    static timing_manager*   fgInstance;
    timer_map_t             m_timer_map;
    timer_list_t            m_timer_list;
};

//----------------------------------------------------------------------------//
template <typename _Ret, typename _Func>
inline
_Ret timing_manager::time_function(const string_t& key, _Func func)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.resume();
    _t.lap();
    _Ret _ret = func();
    _t.pause();
    return _ret;
}
//----------------------------------------------------------------------------//
template <typename _Ret, typename _Func, typename... _Args>
inline
_Ret timing_manager::time_function(const string_t& key, _Func func, _Args... args)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.resume();
    _t.lap();
    _Ret _ret = func(args...);
    _t.pause();
    return _ret;
}
//----------------------------------------------------------------------------//
template <typename _Func>
inline
void timing_manager::time_function(const string_t& key, _Func func)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.resume();
    _t.lap();
    func();
    _t.pause();
}
//----------------------------------------------------------------------------//
template <typename _Func, typename... _Args>
inline
void timing_manager::time_function(const string_t& key, _Func func, _Args... args)
{
    toast_timer_t& _t = this->instance()->timer(key);
    _t.resume();
    _t.lap();
    func(args...);
    _t.pause();
}
//----------------------------------------------------------------------------//

} // namespace util

} // namespace toast

#endif // timing_manager_hpp_
