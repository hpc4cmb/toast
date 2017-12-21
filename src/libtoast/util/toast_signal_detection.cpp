//
//
// created by jmadsen on Wed Oct 19 16:38:09 2016
//
//
//
//

#include "toast_util_internal.hpp"
#include <sstream>
#include <string>
#include <cstdlib>


namespace toast
{

//============================================================================//

namespace internal { void dummy_func(int) { return; } }

//============================================================================//

bool signal_settings::signals_active = false;

//============================================================================//

std::set<sys_signal> signal_settings::signals_default
= {
    sys_signal::abort_signal,
    sys_signal::bus_signal,
    sys_signal::hangup_signal,
    sys_signal::illegal_signal,
    sys_signal::interrupt_signal,
    sys_signal::kill_signal,
    sys_signal::quit_signal,
    sys_signal::seg_fault
  };

//============================================================================//

std::set<sys_signal> signal_settings::signals_enabled = signal_settings::signals_default;

//============================================================================//

std::set<sys_signal> signal_settings::signals_disabled
= {
  };

//============================================================================//

signal_settings::signal_function_t signal_settings::signals_exit_func =
        internal::dummy_func;

//============================================================================//

void insert_and_remove(const sys_signal& _type,            // fpe type
                       signal_settings::signal_set_t* _ins, // set to insert into
                       signal_settings::signal_set_t* _rem  // set to remove from
                       )
{
    _ins->insert(_type);
    auto itr = _rem->find(_type);
    if(itr != _rem->end())
        _rem->erase(itr);
}

//============================================================================//

void signal_settings::enable(const sys_signal& _type)
{
    insert_and_remove(_type, &signals_enabled, &signals_disabled);
}

//============================================================================//

void signal_settings::disable(const sys_signal& _type)
{
    insert_and_remove(_type, &signals_disabled, &signals_enabled);
}

//============================================================================//

void signal_settings::check_environment()
{
    typedef std::pair<std::string, sys_signal> match_t;

    auto _list =
    {
        match_t("ABORT",    sys_signal::abort_signal),
        match_t("BUS",      sys_signal::bus_signal),
        match_t("HUP",      sys_signal::hangup_signal),
        match_t("ILL",      sys_signal::illegal_signal),
        match_t("INT",      sys_signal::interrupt_signal),
        match_t("KILL",     sys_signal::kill_signal),
        match_t("QUIT",     sys_signal::quit_signal),
        match_t("SEGF",     sys_signal::seg_fault)
    };

    for(auto itr : _list)
    {
        int _enable = get_env<int>("SIGNAL_ENABLE_" + itr.first, 0);
        int _disable = get_env<int>("SIGNAL_DISABLE_" + itr.first, 0);

        if(_enable > 0)
            signal_settings::enable(itr.second);
        if(_disable > 0)
            signal_settings::disable(itr.second);
    }

    int _enable_all = get_env<int>("SIGNAL_ENABLE_ALL", 0);
    if(_enable_all > 0)
        for(const auto& itr : signal_settings::signals_disabled)
            signal_settings::enable(itr);

    int _disable_all = get_env<int>("SIGNAL_DISABLE_ALL", 0);
    if(_disable_all > 0)
        for(const auto& itr : signal_settings::signals_enabled)
            signal_settings::disable(itr);

}

//============================================================================//

std::string signal_settings::str(const sys_signal& _type)
{
    std::stringstream ss;
    auto descript = [&] (const std::string& _name, const int& _err)
    {
        ss << " Signal: " << _name << " (error code: " << _err << ") ";
    };

    switch (_type)
    {
        case sys_signal::abort_signal:
            descript("SIGABRT", SIGABRT);
            break;
        case sys_signal::bus_signal:
            descript("SIGBUS", SIGBUS);
            break;
        case sys_signal::hangup_signal:
            descript("SIGHUP", SIGHUP);
            break;
        case sys_signal::illegal_signal:
            descript("SIGILL", SIGILL);
            break;
        case sys_signal::interrupt_signal:
            descript("SIGINT", SIGINT);
            break;
        case sys_signal::kill_signal:
            descript("SIGKILL", SIGKILL);
            break;
        case sys_signal::quit_signal:
            descript("SIGQUIT", SIGQUIT);
            break;
        case sys_signal::seg_fault:
            descript("SIGSEGV", SIGSEGV);
            break;
    }

    return ss.str();
}

//============================================================================//

std::string signal_settings::str()
{

    std::stringstream ss;
    auto spacer = [&] () { return "    "; };

#if defined(SIGNAL_AVAILABLE)

    ss << std::endl
       << spacer() << "Signal detection activated. Signal exception settings:\n"
       << std::endl;

    ss << spacer() << "Enabled:" << std::endl;
    for(const auto& itr : signals_enabled)
        ss << spacer() << spacer() << signal_settings::str(itr) << std::endl;

    ss << "\n" << spacer() << "Disabled:" << std::endl;
    for(const auto& itr : signals_disabled)
        ss << spacer() << spacer() << signal_settings::str(itr) << std::endl;

#else

    ss << std::endl
       << spacer()
       << "Signal detection not available" << std::endl;

#endif

    return ss.str();
}

//============================================================================//

} // namespace toast

//============================================================================//
