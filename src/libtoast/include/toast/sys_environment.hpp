
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_SYS_ENVIRONMENT_HPP
#define TOAST_SYS_ENVIRONMENT_HPP

#include <string>
#include <vector>
#include <map>


namespace toast {

class Environment {
    // Singleton containing runtime settings.

    public:

        // Singleton access
        static Environment & get();

        std::string log_level() const;
        void set_log_level(char const * level);
        std::vector <std::string> signals() const;
        void print() const;
        bool use_mpi() const;
        bool function_timers() const;
        int max_threads() const;
        int current_threads() const;
        void set_threads(int nthread);
        std::string version() const;

    private:

        // This class is a singleton- constructor is private.
        Environment();

        std::string loglvl_;
        std::vector <std::string> signals_avail_;
        std::map <std::string, int> signals_value_;
        std::map <std::string, bool> signals_enabled_;
        bool have_mpi_;
        bool use_mpi_;
        bool func_timers_;
        bool at_nersc_;
        bool in_slurm_;
        int max_threads_;
        int cur_threads_;
        std::string git_version_;
        std::string release_version_;
        std::string version_;
};

}

#endif // ifndef TOAST_ENVIRONMENT_HPP
