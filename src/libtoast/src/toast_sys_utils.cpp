
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_environment.hpp>
#include <toast/sys_utils.hpp>

#include <cstring>
#include <sstream>

#include <vector>
#include <algorithm>

#ifdef HAVE_CUDALIBS
#include <cuda_runtime_api.h>
#endif

std::string toast::format_here(std::pair <std::string, int> const & here) {
    std::ostringstream h;
    h << "file \"" << here.first << "\", line " << here.second;
    return std::string(h.str());
}

void * toast::aligned_alloc(size_t size, size_t align) {
    void * mem = NULL;
    #ifdef HAVE_CUDALIBS
        // allocates with CUDA to get unified memory that can be accessed from CPU and GPU transparently
        // garantees that the memory will be "suitably aligned for any kind of variable"
        int ret = cudaMallocManaged(&mem, size);
    #else
        int ret = posix_memalign(&mem, align, size);
    #endif
    if (ret != 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "cannot allocate " << size
          << " bytes of memory with alignment " << align;
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }
    memset(mem, 0, size);
    return mem;
}

void toast::aligned_free(void * ptr) {
    #ifdef HAVE_CUDALIBS
        // frees with CUDA when using unified memory
        cudaFree(ptr);
    #else
        free(ptr);
    #endif
    return;
}

toast::Timer::Timer() {
    clear();
}

toast::Timer::Timer(double init_time, size_t init_calls) {
    start_ = time_point();
    stop_ = time_point();
    running_ = false;
    total_ = init_time;
    calls_ = init_calls;
}

void toast::Timer::start() {
    if (!running_) {
        start_ = std::chrono::high_resolution_clock::now();
        running_ = true;
        calls_++;
    }
    return;
}

void toast::Timer::stop() {
    if (running_) {
        stop_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration <double> elapsed =
            std::chrono::duration_cast <std::chrono::duration <double> >
                (stop_ - start_);
        total_ += elapsed.count();
        running_ = false;
    }
    return;
}

void toast::Timer::clear() {
    start_ = time_point();
    stop_ = time_point();
    running_ = false;
    calls_ = 0;
    total_ = 0.0;
    return;
}

double toast::Timer::seconds() const {
    if (running_) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("Timer is still running!");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return total_;
}

double toast::Timer::elapsed_seconds() const {
    /* Return the current reading on the timer without incrementing the calls_ counter
       or stopping the timer
     */
    if (not running_) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("Timer is not running!");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration <double> elapsed =
        std::chrono::duration_cast <std::chrono::duration <double> >
            (now - start_);
    return total_ + elapsed.count();
}

size_t toast::Timer::calls() const {
    if (running_) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("Timer is still running!");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    return calls_;
}

bool toast::Timer::is_running() const {
    return running_;
}

void toast::Timer::report(char const * message) {
    double t = seconds();
    toast::Logger & logger = toast::Logger::get();
    std::ostringstream msg;

    msg.precision(2);
    msg << std::fixed << message << ":  " << t << " seconds ("
        << calls_ << " calls)";
    logger.info(msg.str().c_str());
    return;
}

void toast::Timer::report_clear(char const * message) {
    bool was_running = running_;
    if (was_running) {
        stop();
    }
    report(message);
    clear();
    if (was_running) {
        start();
    }
    return;
}

void toast::Timer::report_elapsed(char const * message) {
    /* Report elapsed time from a running timer without changing its state
     */
    double t = elapsed_seconds();
    toast::Logger & logger = toast::Logger::get();
    std::ostringstream msg;

    msg.precision(2);
    msg << std::fixed << message << ":  " << t << " seconds ("
        << calls_ << " calls)";
    logger.info(msg.str().c_str());
    return;
}

toast::GlobalTimers::GlobalTimers() {
    data.clear();
}

toast::GlobalTimers & toast::GlobalTimers::get() {
    static toast::GlobalTimers instance;

    return instance;
}

std::vector <std::string> toast::GlobalTimers::names() const {
    std::vector <std::string> ret;
    for (auto const & it : data) {
        ret.push_back(it.first);
    }
    std::stable_sort(ret.begin(), ret.end());
    return ret;
}

void toast::GlobalTimers::start(std::string const & name) {
    if (data.count(name) == 0) {
        data[name].clear();
    }
    data.at(name).start();
    return;
}

void toast::GlobalTimers::clear(std::string const & name) {
    data[name].clear();
    return;
}

void toast::GlobalTimers::stop(std::string const & name) {
    if (data.count(name) == 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Cannot stop timer " << name << " which does not exist";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }
    data.at(name).stop();
    return;
}

double toast::GlobalTimers::seconds(std::string const & name) const {
    if (data.count(name) == 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Cannot get seconds for timer " << name
          << " which does not exist";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }
    return data.at(name).seconds();
}

size_t toast::GlobalTimers::calls(std::string const & name) const {
    if (data.count(name) == 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Cannot get seconds for timer " << name
          << " which does not exist";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }
    return data.at(name).calls();
}

bool toast::GlobalTimers::is_running(std::string const & name) const {
    if (data.count(name) == 0) {
        return false;
    }
    return data.at(name).is_running();
}

void toast::GlobalTimers::stop_all() {
    for (auto & tm : data) {
        tm.second.stop();
    }
    return;
}

void toast::GlobalTimers::clear_all() {
    for (auto & tm : data) {
        tm.second.clear();
    }
    return;
}

void toast::GlobalTimers::report() {
    stop_all();
    std::vector <std::string> names;
    for (auto const & tm : data) {
        names.push_back(tm.first);
    }
    std::stable_sort(names.begin(), names.end());
    std::ostringstream msg;
    for (auto const & nm : names) {
        msg.str("");
        msg << "Global timer: " << nm;
        data.at(nm).report(msg.str().c_str());
    }
    return;
}

toast::Logger::Logger() {
    // Prefix for messages
    prefix_ = std::string("TOAST ");
    return;
}

toast::Logger & toast::Logger::get() {
    static toast::Logger instance;

    // Check the level every time we get a reference to the singleton,
    // in case the level has been changed manually during runtime.
    instance.check_level();
    return instance;
}

void toast::Logger::check_level() {
    auto & env = toast::Environment::get();
    std::string val = env.log_level();
    if (strncmp(val.c_str(), "VERBOSE", 7) == 0) {
        level_ = log_level::verbose;
    } else if (strncmp(val.c_str(), "DEBUG", 5) == 0) {
        level_ = log_level::debug;
    } else if (strncmp(val.c_str(), "INFO", 4) == 0) {
        level_ = log_level::info;
    } else if (strncmp(val.c_str(), "WARNING", 7) == 0) {
        level_ = log_level::warning;
    } else if (strncmp(val.c_str(), "ERROR", 5) == 0) {
        level_ = log_level::error;
    } else if (strncmp(val.c_str(), "CRITICAL", 8) == 0) {
        level_ = log_level::critical;
    } else {
        level_ = log_level::none;
    }
    return;
}

void toast::Logger::verbose(char const * msg) {
    if (level_ <= log_level::verbose) {
        fprintf(stdout, "%sVERBOSE: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}

void toast::Logger::verbose(char const * msg,
                            std::pair <std::string, int> const & here) {
    if (level_ <= log_level::verbose) {
        std::string hstr = toast::format_here(here);
        fprintf(stdout, "%sVERBOSE: %s (%s)\n", prefix_.c_str(), msg,
                hstr.c_str());
        fflush(stdout);
    }
    return;
}

void toast::Logger::debug(char const * msg) {
    if (level_ <= log_level::debug) {
        fprintf(stdout, "%sDEBUG: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}

void toast::Logger::debug(char const * msg,
                          std::pair <std::string, int> const & here) {
    if (level_ <= log_level::debug) {
        std::string hstr = toast::format_here(here);
        fprintf(stdout, "%sDEBUG: %s (%s)\n", prefix_.c_str(), msg,
                hstr.c_str());
        fflush(stdout);
    }
    return;
}

void toast::Logger::info(char const * msg) {
    if (level_ <= log_level::info) {
        fprintf(stdout, "%sINFO: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}

void toast::Logger::info(char const * msg,
                         std::pair <std::string, int> const & here) {
    if (level_ <= log_level::info) {
        std::string hstr = toast::format_here(here);
        fprintf(stdout, "%sINFO: %s (%s)\n", prefix_.c_str(), msg,
                hstr.c_str());
        fflush(stdout);
    }
    return;
}

void toast::Logger::warning(char const * msg) {
    if (level_ <= log_level::warning) {
        fprintf(stdout, "%sWARNING: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}

void toast::Logger::warning(char const * msg,
                            std::pair <std::string, int> const & here) {
    if (level_ <= log_level::warning) {
        std::string hstr = toast::format_here(here);
        fprintf(stdout, "%sWARNING: %s (%s)\n", prefix_.c_str(), msg,
                hstr.c_str());
        fflush(stdout);
    }
    return;
}

void toast::Logger::error(char const * msg) {
    if (level_ <= log_level::error) {
        fprintf(stdout, "%sERROR: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}

void toast::Logger::error(char const * msg,
                          std::pair <std::string, int> const & here) {
    if (level_ <= log_level::error) {
        std::string hstr = toast::format_here(here);
        fprintf(stdout, "%sERROR: %s (%s)\n", prefix_.c_str(), msg,
                hstr.c_str());
        fflush(stdout);
    }
    return;
}

void toast::Logger::critical(char const * msg) {
    if (level_ <= log_level::critical) {
        fprintf(stdout, "%sCRITICAL: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}

void toast::Logger::critical(char const * msg,
                             std::pair <std::string, int> const & here) {
    if (level_ <= log_level::critical) {
        std::string hstr = toast::format_here(here);
        fprintf(stdout, "%sCRITICAL: %s (%s)\n", prefix_.c_str(), msg,
                hstr.c_str());
        fflush(stdout);
    }
    return;
}
