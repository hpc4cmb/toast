
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/environment.hpp>
#include <toast/utils.hpp>

#include <cstring>
#include <sstream>

#include <vector>
#include <algorithm>


toast::Timer::Timer() {
    clear();
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
        throw std::runtime_error("Timer is still running!");
    }
    return total_;
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

toast::GlobalTimers::GlobalTimers() {
    data.clear();
}

toast::GlobalTimers & toast::GlobalTimers::get() {
    static toast::GlobalTimers instance;

    return instance;
}

void toast::GlobalTimers::start(std::string const & name) {
    if (data.count(name) == 0) {
        data[name].clear();
    }
    data.at(name).start();
    return;
}

void toast::GlobalTimers::stop(std::string const & name) {
    if (data.count(name) == 0) {
        std::ostringstream o;
        o << "Cannot stop timer " << name << " which does not exist";
        throw std::runtime_error(o.str().c_str());
    }
    data.at(name).stop();
    return;
}

double toast::GlobalTimers::seconds(std::string const & name) const {
    if (data.count(name) == 0) {
        std::ostringstream o;
        o << "Cannot get seconds for timer " << name
          << " which does not exist";
        throw std::runtime_error(o.str().c_str());
    }
    return data.at(name).seconds();
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

    // Check log level:
    auto & env = toast::Environment::get();
    level_ = log_level::none;
    std::string val = env.log_level();
    if (strncmp(val.c_str(), "DEBUG", 5) == 0) {
        level_ = log_level::debug;
    } else if (strncmp(val.c_str(), "INFO", 4) == 0) {
        level_ = log_level::info;
    } else if (strncmp(val.c_str(), "WARNING", 7) == 0) {
        level_ = log_level::warning;
    } else if (strncmp(val.c_str(), "ERROR", 5) == 0) {
        level_ = log_level::error;
    } else if (strncmp(val.c_str(), "CRITICAL", 8) == 0) {
        level_ = log_level::critical;
    }
    return;
}

toast::Logger & toast::Logger::get() {
    static toast::Logger instance;

    return instance;
}

void toast::Logger::debug(char const * msg) {
    if (level_ <= log_level::debug) {
        fprintf(stdout, "%sDEBUG: %s\n", prefix_.c_str(), msg);
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

void toast::Logger::warning(char const * msg) {
    if (level_ <= log_level::warning) {
        fprintf(stdout, "%sWARNING: %s\n", prefix_.c_str(), msg);
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

void toast::Logger::critical(char const * msg) {
    if (level_ <= log_level::critical) {
        fprintf(stdout, "%sCRITICAL: %s\n", prefix_.c_str(), msg);
        fflush(stdout);
    }
    return;
}
