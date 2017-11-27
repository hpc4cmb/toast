//
// created by jrmadsen on Thu Nov 16 00:55:30 2017
//

#ifndef auto_timer_hpp_
#define auto_timer_hpp_

#include "timing_manager.hpp"
#include <string>
#include <cstdint>

namespace toast
{
namespace util
{

class auto_timer
{
public:
    typedef toast::util::timing_manager::toast_timer_t  toast_timer_t;

public:
    // Constructor and Destructors
    auto_timer(const std::string&, const int32_t& lineno);
    virtual ~auto_timer();

private:
    static uint64_t& ncount()
    { return details::base_timer::get_instance_count(); }

    static uint64_t& nhash()
    { return details::base_timer::get_instance_hash(); }

private:
    uint64_t m_hash;
    toast_timer_t& m_timer;
};

//----------------------------------------------------------------------------//
inline auto_timer::auto_timer(const std::string& timer_tag,
                              const int32_t& lineno)
: m_hash(lineno),
  m_timer(timing_manager::instance()->timer(timer_tag, "cxx",
                                            ++auto_timer::ncount(),
                                            auto_timer::nhash() += m_hash))
{
    m_timer.start();
}
//----------------------------------------------------------------------------//
inline auto_timer::~auto_timer()
{
    m_timer.stop();
    if(auto_timer::ncount() > 0)
        auto_timer::ncount()--;

    auto_timer::nhash() -= m_hash;
}
//----------------------------------------------------------------------------//

} // namespace util

} // namespace toast

#endif

