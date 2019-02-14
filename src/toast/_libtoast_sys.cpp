
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_sys(py::module & m) {
    py::class_ <toast::Environment,
                std::unique_ptr <toast::Environment, py::nodelete> > (
        m, "Environment",
        R"(
        Global runtime environment.

        This singleton class provides a unified place to parse environment
        variables at runtime and to change global settings that impact the
        overall package.

        )")
    .def("get", []() {
             return std::unique_ptr <toast::Environment, py::nodelete>
             (
                 &toast::Environment::get());
         }, R"(
            Get a handle to the global environment class.
        )")
    .def("log_level", &toast::Environment::log_level,
         R"(
            Return the string of the current Logging level.
        )")
    .def("version", &toast::Environment::version,
         R"(
            Return the current source code version string.
        )")
    .def("set_log_level", &toast::Environment::set_log_level,
         R"(
            Set the Logging level.

            Args:
                level (str):  one of DEBUG, INFO, WARNING, ERROR or
                    CRITICAL.

            Returns:
                None

        )")
    .def("signals", &toast::Environment::signals,
         R"(
            Return a list of the currently available signals.
        )")
    .def("print", &toast::Environment::print,
         R"(
            Print the current environment to STDOUT.
        )")
    .def("use_mpi", &toast::Environment::use_mpi,
         R"(
            Return True if TOAST was compiled with MPI support **and** MPI
            is supported in the current runtime environment.
        )")
    .def("max_threads", &toast::Environment::max_threads,
         R"(
            Returns the maximum number of threads used by compiled code.
        )");

    // Simple timer

    py::class_ <toast::Timer, toast::Timer::puniq> (
        m, "Timer",
        R"(
        Simple timer class.

        This class is just a timer that you can start / stop / clear
        and report the results.  It tracks the elapsed time and the number
        of times it was started.

        )")
    .def(py::init <> ())
    .def("start", &toast::Timer::start,
         R"(
            Start the timer.
        )")
    .def("stop", &toast::Timer::stop,
         R"(
            Stop the timer.
        )")
    .def("clear", &toast::Timer::clear,
         R"(
            Clear the timer.
        )")
    .def("is_running", &toast::Timer::is_running,
         R"(
            Is the timer running?
        )")
    .def("seconds",
         [](toast::Timer const & self) {
             if (self.is_running()) {
                 return -1.0;
             } else {
                 return self.seconds();
             }
         }, R"(
            Return the elapsed seconds (if stopped) else -1.
        )")
    .def("report", &toast::Timer::report,
         R"(
            Report results of the timer to STDOUT.

            Args:
                message (str): A message to prepend to the timing results.

            Returns:
                None

        )")
    .def("__repr__",
         [](toast::Timer const & self) {
             std::ostringstream o;
             o.precision(2);
             o << std::fixed;
             o << "<toast.Timer ";
             if (self.is_running()) {
                 o << "(still running)";
             } else {
                 double elapsed = self.seconds();
                 o << "(stopped at " << elapsed << " seconds)";
             }
             o << ">";
             return o.str();
         });


    py::class_ <toast::GlobalTimers,
                std::unique_ptr <toast::GlobalTimers, py::nodelete> > (
        m, "GlobalTimers",
        R"(
        Global timer registry.

        This singleton class stores timers that can be started / stopped
        anywhere in the code to accumulate the total time for different
        operations.
        )")
    .def("get", []() {
             return std::unique_ptr <toast::GlobalTimers, py::nodelete>
                 (&toast::GlobalTimers::get());
         }, R"(
            Get a handle to the singleton class.
        )")
    .def("names", &toast::GlobalTimers::names,
         R"(
        Return the names of all currently registered timers.
        )")
    .def("start", &toast::GlobalTimers::start,
         R"(
        Start a named timer.  The timer is created if it does not exist.

        Args:
            name (str):  The name of the timer.

        )")
    .def("stop", &toast::GlobalTimers::stop,
         R"(
        Stop a named timer.

        Args:
            name (str):  The name of the timer.

        )")
    .def("seconds", &toast::GlobalTimers::seconds,
         R"(
        Get the elapsed time for a stopped timer.

        Args:
            name (str):  The name of the timer.

        Returns:
            (float):  The elapsed time in seconds.

        )")
    .def("is_running", &toast::GlobalTimers::is_running,
         R"(
        Check the state of a timer.

        Args:
            name (str):  The name of the timer.

        Returns:
            (bool):  True if the timer is running, else False.

        )")
    .def("stop_all", &toast::GlobalTimers::stop_all,
         R"(
        Stop all timers.
        )")
    .def("clear_all", &toast::GlobalTimers::clear_all,
         R"(
        Clear all timers.
        )")
    .def("report", &toast::GlobalTimers::report,
         R"(
        Print the status of all timers to STDOUT.
        )");


    py::class_ <toast::Logger,
                std::unique_ptr <toast::Logger, py::nodelete> > (
        m, "Logger",
        R"(
        Simple Logging class.

        This class mimics the python logger in C++.  The log level is
        controlled by the TOAST_LOGLEVEL environment variable.  Valid levels
        are DEBUG, INFO, WARNING, ERROR and CRITICAL.  The default is INFO.

        )")
    .def("get", []() {
             return std::unique_ptr <toast::Logger, py::nodelete>
                 (&toast::Logger::get());
         }, R"(
            Get a handle to the global logger.
        )")
    .def("debug", &toast::Logger::debug,
         R"(
            Print a DEBUG level message.
        )")
    .def("info", &toast::Logger::info,
         R"(
            Print an INFO level message.
        )")
    .def("warning", &toast::Logger::warning,
         R"(
            Print a WARNING level message.
        )")
    .def("error", &toast::Logger::error,
         R"(
            Print an ERROR level message.
        )")
    .def("critical", &toast::Logger::critical,
         R"(
            Print a CRITICAL level message.
        )");
}
