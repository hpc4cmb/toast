//
// Time Ordered Astrophysics Scalable Tools (TOAST)
//
// Copyright (c) 2015-2017, The Regents of the University of California
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "toast_util_internal.hpp"
#include <algorithm>
#include <cassert>

//============================================================================//

CEREAL_CLASS_VERSION(toast::util::timer, TOAST_TIMER_VERSION)

//============================================================================//

thread_local uint64_t toast::util::timer::f_output_width = 10;

//============================================================================//

std::string toast::util::timer::default_format
    =  " : %w wall, %u user + %s system = %t CPU [sec] (%p%)"
       // bash expansion:
       //   total_RSS_current, total_RSS_peak
       //   self_RSS_current self_RSS_peak
       " : RSS {tot,self}_{curr,peak}"
       " : (%c|%m)"
       " | (%C|%M) [MB]";

//============================================================================//

uint16_t toast::util::timer::default_precision = 3;

//============================================================================//

void toast::util::timer::propose_output_width(uint64_t _w)
{
    f_output_width = std::max(f_output_width, _w);
}

//============================================================================//

toast::util::timer::timer(const string_t& _begin,
                          const string_t& _close,
                          bool _use_static_width,
                          uint16_t prec)
: base_type(prec, _begin + default_format + _close),
  m_use_static_width(_use_static_width),
  m_begin(_begin), m_close(_close)
{ }

//============================================================================//

toast::util::timer::timer(const string_t& _begin,
                          const string_t& _end,
                          const string_t& _fmt,
                          bool _use_static_width,
                          uint16_t prec)
: base_type(prec, _begin + _fmt + _end),
  m_use_static_width(_use_static_width),
  m_begin(_begin), m_close(_end)
{ }

//============================================================================//

toast::util::timer::~timer()
{ }

//============================================================================//

void toast::util::timer::compose()
{
    std::stringstream ss;
    if(m_use_static_width)
    {
        ss << std::setw(f_output_width + 2)
           << std::left << m_begin
           << std::right << default_format
           << m_close;
    }
    else
    {
        ss << std::left << m_begin
           << std::right << default_format
           << m_close;
    }
    m_format_string = ss.str();
}

//============================================================================//
