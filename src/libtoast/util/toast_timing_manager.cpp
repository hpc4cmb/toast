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

#include "toast_util_internal.hpp"
#include "toast_math_internal.hpp"

#include <sstream>

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

    m_timer_map[key] = toast_timer_t(3,
                                     string_t("> ") + key + string_t(" : "),
                                     string_t(""));
    timer_pair_t _pair(key, m_timer_map[key]);
    m_timer_list.push_back(_pair);

    return m_timer_map[key];
}
//============================================================================//


