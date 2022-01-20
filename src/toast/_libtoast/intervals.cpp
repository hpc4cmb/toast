// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <intervals.hpp>


void init_intervals(py::module & m) {
    py::class_ <Interval> (
        m, "Interval",
        R"(
        Numpy dtype for an interval
        )")
    .def(py::init([]() {
                      return Interval();
                  }))
    .def_readwrite("start", &Interval::start)
    .def_readwrite("stop", &Interval::stop)
    .def_readwrite("first", &Interval::first)
    .def_readwrite("last", &Interval::last)
    .def("astuple",
         [](const Interval & self) {
             return py::make_tuple(self.start, self.stop, self.first, self.last);
         })
    .def_static("fromtuple", [](const py::tuple & tup) {
                    if (py::len(tup) != 4) {
                        throw py::cast_error("Invalid size");
                    }
                    return Interval{tup[0].cast <double>(),
                                    tup[1].cast <double>(),
                                    tup[2].cast <int64_t>(),
                                    tup[3].cast <int64_t>()};
                });


    PYBIND11_NUMPY_DTYPE(Interval, start, stop, first, last);
}
