/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include "toast_util_internal.hpp"
#include "toast_math_internal.hpp"

#include <sstream>
#include <algorithm>

//============================================================================//

toast::util::timing_manager* toast::util::timing_manager::fgInstance = nullptr;

//============================================================================//

toast::util::timing_manager* toast::util::timing_manager::instance()
{
    if(!fgInstance) new toast::util::timing_manager();
	return fgInstance;
}

//============================================================================//

toast::util::timing_manager::timing_manager()
: m_report_tot(&std::cout),
  m_report_avg(&std::cout)
{
	if(!fgInstance) { fgInstance = this; }
    else
    {
        std::ostringstream ss;
        ss << "toast::util::timing_manager singleton has already been created";
        TOAST_THROW( ss.str().c_str() );
    }
}

//============================================================================//

toast::util::timing_manager::~timing_manager()
{
    auto close_ostream = [&] (ostream_t*& m_os)
    {
        ofstream_t* m_fos = get_ofstream(m_os);
        if(!m_fos)
            return;
        m_fos->close();
    };

    close_ostream(m_report_tot);
    close_ostream(m_report_avg);

    fgInstance = nullptr;
}

//============================================================================//

toast::util::timer& toast::util::timing_manager::timer(const string_t& key)
{
    if(m_timer_map.find(key) != m_timer_map.end())
        return m_timer_map.find(key)->second;

    std::stringstream ss;
    ss << "> " << std::setw(45) << std::left << key << std::right << " : ";
    m_timer_map[key] = toast_timer_t(3, ss.str(), string_t(""));

    timer_pair_t _pair(key, m_timer_map[key]);
    m_timer_list.push_back(_pair);

    return m_timer_map[key];
}

//============================================================================//



























