
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


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
         py::arg(
             "level"), R"(
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
    .def("function_timers", &toast::Environment::function_timers,
         R"(
            Return True if function timing has been enabled.
        )")
    .def("enable_function_timers", &toast::Environment::enable_function_timers,
         R"(
            Explicitly enable function timers.
        )")
    .def("disable_function_timers", &toast::Environment::disable_function_timers,
         R"(
            Explicitly disable function timers.
        )")
    .def("tod_buffer_length", &toast::Environment::tod_buffer_length,
         R"(
            Returns the number of samples to buffer for TOD operations.
        )")
    .def("max_threads", &toast::Environment::max_threads,
         R"(
            Returns the maximum number of threads used by compiled code.
        )")
    .def("current_threads", &toast::Environment::current_threads,
         R"(
            Return the current threading concurrency in use.
        )")
    .def("set_threads", &toast::Environment::set_threads,
         py::arg(
             "nthread"), R"(
            Set the number of threads in use.

            Args:
                nthread (int): The number of threads to use.

            Returns:
                None

        )")
    .def("set_acc", &toast::Environment::set_acc,
         py::arg("n_acc_device"), py::arg("n_acc_proc_per_device"),
         py::arg(
             "my_acc_device"), R"(
            Set the OpenACC device properties.

            Args:
                n_acc_device (int):  The number of accelerator devices.
                n_acc_proc_per_device (int):  The number of processes sharing
                    each device.
                my_acc_device (int):  The device to use for this process.

            Returns:
                None

        )")
    .def("get_acc",
         [](toast::Environment const & self) {
             int n_acc_device;
             int n_acc_proc_per_device;
             int my_acc_device;
             self.get_acc(&n_acc_device, &n_acc_proc_per_device, &my_acc_device);
             return py::make_tuple(n_acc_device, n_acc_proc_per_device,
                                   my_acc_device);
         }, R"(
            Get the OpenACC device properties.

            Returns:
                (tuple):  The (num devices, proc per device, my device) integers.

        )")
    .def("__repr__",
         [](toast::Environment const & self) {
             std::ostringstream o;
             o << "<toast.Environment" << std::endl;
             auto strngs = self.info();
             for (auto const & line : strngs) {
                 o << "  " << line << std::endl;
             }
             o << ">";
             return o.str();
         });

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
    .def(py::init <double, size_t> (), py::arg("init_time"), py::arg(
             "init_calls"), R"(
           Create the timer with some initial state.

           Used mainly when pickling / communicating the timer.  The timer is created
           in the "stopped" state.

           Args:
               init_time (float):  Initial elapsed seconds.
               init_calls (int):  Inital number of calls.

       )")
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

            Returns:
               (bool): True if the timer is running, else False.

        )")
    .def("seconds",
         [](toast::Timer const & self) {
             if (self.is_running()) {
                 return -1.0;
             } else {
                 return self.seconds();
             }
         }, R"(
            Return the elapsed seconds.

            Returns:
                (float): The elapsed seconds (if timer is stopped) else -1.

        )")
    .def("elapsed_seconds",
         [](toast::Timer const & self) {
             return self.elapsed_seconds();
         }, R"(
            Return the elapsed seconds from a running timer without
            modifying the timer state.

            Returns:
                (float): The elapsed seconds (if timer is running).

        )")
    .def("calls",
         [](toast::Timer const & self) {
             if (self.is_running()) {
                 return size_t(0);
             } else {
                 return self.calls();
             }
         }, R"(
            Return the number of calls.

            Returns:
                (int): The number of calls (if timer is stopped) else 0.

        )")
    .def("report", &toast::Timer::report, py::arg(
             "message"),
         R"(
            Report results of the timer to STDOUT.

            Args:
                message (str): A message to prepend to the timing results.

            Returns:
                None

        )")
    .def("report_clear", &toast::Timer::report_clear, py::arg(
             "message"),
         R"(
            Report results of the timer to STDOUT and clear the timer.

            If the timer was running, it is stopped before reporting and clearing and
            then restarted.  If the timer was stopped, then it is left in the stopped
            state after reporting and clearing.

            Args:
                message (str): A message to prepend to the timing results.

            Returns:
                None

        )")
    .def("report_elapsed", &toast::Timer::report_elapsed, py::arg(
             "message"),
         R"(
            Report results of a running timer to STDOUT without
            modifying the timer state.

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
         })
    .def(
        py::pickle(
            [](toast::Timer const & p) { // __getstate__
                return py::make_tuple(p.seconds(), p.calls());
            },
            [](py::tuple t) {            // __setstate__
                if (t.size() != 2) {
                    auto log = toast::Logger::get();
                    std::ostringstream o;
                    o << "Unpickling: wrong number of tuple members";
                    log.error(o.str().c_str());
                    throw std::runtime_error(o.str().c_str());
                }
                toast::Timer ret(
                    t[0].cast <double>(),
                    t[1].cast <size_t>()
                );
                return ret;
            }));


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

        Returns:
            (list): The names of the timers.

        )")
    .def("start", &toast::GlobalTimers::start, py::arg(
             "name"), R"(
            Start the specified timer.

            If the named timer does not exist, it is first created before
            being started.

            Args:
                name (str): The name of the global timer.

            Returns:
                None
        )")
    .def("stop", &toast::GlobalTimers::stop, py::arg(
             "name"), R"(
            Stop the specified timer.

            The timer must already exist.

            Args:
                name (str): The name of the global timer.

            Returns:
                None
        )")
    .def("seconds", &toast::GlobalTimers::seconds, py::arg(
             "name"), R"(
            Get the elapsed time for a timer.

            The timer must be stopped.

            Args:
                name (str): The name of the global timer.

            Returns:
                (float): The elapsed time in seconds.
        )")
    .def("is_running", &toast::GlobalTimers::is_running, py::arg(
             "name"), R"(
            Is the specified timer running?

            Args:
                name (str): The name of the global timer.

            Returns:
                (bool): True if the timer is running, else False.
        )")
    .def("stop_all", &toast::GlobalTimers::stop_all,
         R"(
        Stop all global timers.
        )")
    .def("clear_all", &toast::GlobalTimers::clear_all,
         R"(
        Clear all global timers.
        )")
    .def("report", &toast::GlobalTimers::report,
         R"(
        Report results of all global timers to STDOUT.
        )")
    .def("collect", [](toast::GlobalTimers & self) {
             self.stop_all();
             py::dict result;
             for (auto const & nm : self.names()) {
                 auto cur = self.seconds(nm);
                 auto cal = self.calls(nm);
                 result[py::cast(nm)] = py::cast(toast::Timer(cur,
                                                              cal));
             }
             return result;
         }, R"(
            Stop all timers and return the current state.

            Returns:
                (dict):  A dictionary of Timers.

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
    .def("verbose",
         (void (toast::Logger::*)(char const *)) & toast::Logger::verbose,
         py::arg(
             "msg"), R"(
            Print a VERBOSE level message.

            Args:
                msg (str): The message to print.

            Returns:
                None

        )")
    .def("debug",
         (void (toast::Logger::*)(char const *)) & toast::Logger::debug,
         py::arg(
             "msg"), R"(
            Print a DEBUG level message.

            Args:
                msg (str): The message to print.

            Returns:
                None

        )")
    .def("info",
         (void (toast::Logger::*)(char const *)) & toast::Logger::info,
         py::arg(
             "msg"), R"(
            Print an INFO level message.

            Args:
                msg (str): The message to print.

            Returns:
                None

        )")
    .def("warning",
         (void (toast::Logger::*)(char const *)) & toast::Logger::warning,
         py::arg(
             "msg"), R"(
            Print a WARNING level message.

            Args:
                msg (str): The message to print.

            Returns:
                None

        )")
    .def("error",
         (void (toast::Logger::*)(char const *)) & toast::Logger::error,
         py::arg(
             "msg"), R"(
            Print an ERROR level message.

            Args:
                msg (str): The message to print.

            Returns:
                None

        )")
    .def("critical",
         (void (toast::Logger::*)(char const *)) & toast::Logger::critical,
         py::arg(
             "msg"), R"(
            Print a CRITICAL level message.

            Args:
                msg (str): The message to print.

            Returns:
                None

        )");

    auto env = toast::Environment::get();

    m.def("threading_state",
          []() {
              int max = 0;
              #ifdef _OPENMP
              max = omp_get_max_threads();
              #endif // ifdef _OPENMP

              int cur;
              #pragma omp parallel
              {
                  cur = 0;
                  #ifdef _OPENMP
                  cur = omp_get_num_threads();
                  #endif // ifdef _OPENMP
              }
              return py::make_tuple(max,
                                    cur);
          },
          R"(
        Get the currently configured OpenMP threading state.

        Returns:
            (tuple):  The (global max, current max) number of threads.

    )");
}
