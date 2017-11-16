//
// created by jrmadsen on Thu Nov 16 00:55:30 2017
//

#ifndef auto_timer_hpp_
#define auto_timer_hpp_

#include "timing_manager.hpp"
#include <string>

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
    auto_timer(const std::string&);
    virtual ~auto_timer();

private:
    toast_timer_t& m_timer;
};

//----------------------------------------------------------------------------//
inline auto_timer::auto_timer(const std::string& timer_tag)
: m_timer(toast::util::timing_manager::instance()->timer(timer_tag))
{
    m_timer.start();
}
//----------------------------------------------------------------------------//
inline auto_timer::~auto_timer()
{
    m_timer.stop();
}
//----------------------------------------------------------------------------//

} // namespace util

} // namespace toast

#endif

