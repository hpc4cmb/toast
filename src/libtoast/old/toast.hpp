/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_HPP
#define TOAST_HPP

#include <mpi.h>

#include <string>
#include <sstream>
#include <thread>

namespace toast
{
    //------------------------------------------------------------------------//

    void init ( int argc, char * argv[] );

    void finalize ( );

    //------------------------------------------------------------------------//

    namespace util
    {

        template <typename _Tp>
        _Tp get_env(const std::string& env_id, _Tp _default = _Tp())
        {
            char* env_var = std::getenv(env_id.c_str());
            if(env_var)
            {
                std::string str_var = std::string(env_var);
                std::istringstream iss(str_var);
                _Tp var = _Tp();
                iss >> var;
                return var;
            }
            // return default if not specified in environment
            return _default;
        }
    }

    //------------------------------------------------------------------------//

    // same as using std::uint*_t but my IDE recognizes ensuing
    // references as types so this simply looks better to me...
    typedef std::uint16_t   uint16_t;
    typedef std::uint32_t   uint32_t;
    typedef std::uint64_t   uint64_t;

    //------------------------------------------------------------------------//

    inline uint32_t get_num_threads()
    {
        // we don't want to check enviroment every time (hence: static)
        // and we don't want threads potentially having to
        // reach for a remote place in memory (hence: thread_local)
        static thread_local uint32_t nthread = 0;
        if(nthread == 0)
            nthread = util::get_env<uint32_t>("TOAST_NUM_THREADS",
                                        std::thread::hardware_concurrency());
        return nthread;
    }

    //------------------------------------------------------------------------//

}

#include <toast/math.hpp>
#include <toast/atm.hpp>
#include <toast/tod.hpp>
#include <toast/fod.hpp>
#include <toast/map.hpp>

#endif

