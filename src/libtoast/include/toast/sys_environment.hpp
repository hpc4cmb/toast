
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_SYS_ENVIRONMENT_HPP
#define TOAST_SYS_ENVIRONMENT_HPP

#include <string>
#include <vector>
#include <map>
#include <sstream>

#include <toast/sys_utils.hpp>


namespace toast {
class Environment {
    // Singleton containing runtime settings.

    public:

        // Singleton access
        static Environment & get();

        std::string log_level() const;
        void set_log_level(char const * level);
        std::vector <std::string> signals() const;
        std::vector <std::string> info() const;
        void print() const;
        bool function_timers() const;
        void enable_function_timers();
        void disable_function_timers();
        int max_threads() const;
        int current_threads() const;
        void set_threads(int nthread);
        std::string version() const;
        int64_t tod_buffer_length() const;

    private:

        // This class is a singleton- constructor is private.
        Environment();

        std::string loglvl_;
        std::vector <std::string> signals_avail_;
        std::map <std::string, int> signals_value_;
        std::map <std::string, bool> signals_enabled_;
        bool func_timers_;
        int max_threads_;
        int cur_threads_;
        std::string git_version_;
        std::string release_version_;
        std::string version_;
        int64_t tod_buffer_length_;
};
}

#endif // ifndef TOAST_ENVIRONMENT_HPP
