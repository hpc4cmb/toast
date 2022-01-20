// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <intervals.hpp>


// Interval::Interval() {
//     start = 0.0;
//     stop = 0.0;
//     first = -1;
//     last = -1;
// }

// Interval::Interval(
//     double start_time,
//     double stop_time,
//     int64_t first_samp,
//     int64_t last_samp
// ) {
//     start = start_time;
//     stop = stop_time;
//     first = first_samp;
//     last = last_samp;
// }

// bool Interval::operator==(Interval const & other) const {
//     if (fabs(self.start - other.start) > std::numeric_limits<double>::epsilon) {
//         return false;
//     }
//     if (fabs(self.stop - other.stop) > std::numeric_limits<double>::epsilon) {
//         return false;
//     }
//     if (self.first != other.first) {
//         return false;
//     }
//     if (self.last != other.last) {
//         return false;
//     }
//     return true;
// }

// bool Interval::operator==(Interval const & other) const {
//     if (fabs(self.start - other.start) > std::numeric_limits<double>::epsilon) {
//         return true;
//     }
//     if (fabs(self.stop - other.stop) > std::numeric_limits<double>::epsilon) {
//         return true;
//     }
//     if (self.first != other.first) {
//         return true;
//     }
//     if (self.last != other.last) {
//         return true;
//     }
//     return false;
// }

void init_intervals(py::module & m) {

    py::class_ <Interval> (
        m, "Interval",
        R"(
        Numpy dtype for an interval
        )")
        .def(py::init([]() { return Interval(); }))
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
            return Interval{tup[0].cast<double>(),
                                  tup[1].cast<double>(),
                                  tup[2].cast<int64_t>(),
                                  tup[3].cast<int64_t>()};
        });


    PYBIND11_NUMPY_DTYPE(Interval, start, stop, first, last);


    // py::class_ <Interval> (
    //     m, "Interval",
    //     R"(
    //     Simple container representing a single interval.

    //     Args:
    //         start (float):  The start time.
    //         stop (float):  The stop time.
    //         first (int):  The first sample.
    //         last (int):  The last sample.

    //     )")
    // .def(py::init <> ())
    // .def(py::init <double, double, int64_t, int64_t> (), py::arg("start"), py::arg("stop"), py::arg("first"), py::arg("last"))
    // .def_readwrite("start", &Interval::start)
    // .def_readwrite("stop", &Interval::stop)
    // .def_readwrite("first", &Interval::first)
    // .def_readwrite("last", &Interval::last)
    // .def("__eq__", [](const Interval & self, const Interval & other) {
    //     return (self == other);
    // })
    // .def("__ne__", [](const Interval & self, const Interval & other) {
    //     return (self != other);
    // });


    // py::class_ <IntervalList> (m, "IntervalList", py::buffer_protocol())
    // .def(py::init <>())
    // .def(py::init <IntervalList::size_type>())
    // .def("pop_back", &IntervalList::pop_back)
    // .def("push_back", (void (IntervalList::*)(
    //                        const IntervalList::value_type &)) &IntervalList::push_back)
    // .def("resize", (void (IntervalList::*)(IntervalList::size_type count)) &IntervalList::resize)
    // .def("size", &IntervalList::size)
    // .def("clear", [](IntervalList & self) {
    //     IntervalList().swap(self);
    //     return;
    // })
    // .def("address", [](IntervalList & self) {
    //     return (int64_t)((void *)self.data());
    // })
    // .def_buffer(
    //     [](IntervalList & self) -> py::buffer_info {
    //         std::string format =
    //             py::format_descriptor <IntervalList::value_type>::format();
    //         return py::buffer_info(
    //             static_cast <void *> (self.data()),
    //             sizeof(IntervalList::value_type),
    //             format,
    //             1,
    //             {self.size()},
    //             {sizeof(IntervalList::value_type)}
    //             );
    //     })
    // .def("__len__", [](const IntervalList & self) {
    //     return self.size();
    // })
    // .def("__iter__", [](IntervalList & self) {
    //     return py::make_iterator(self.begin(), self.end());
    // }, py::keep_alive <0, 1>())

    // // Set and get individual elements
    // .def("__setitem__",
    //      [](IntervalList & self, IntervalList::size_type i,
    //         const IntervalList::value_type & t) {
    //          if (i >= self.size()) {
    //              throw py::index_error();
    //          }
    //          self[i] = t;
    //      })
    // .def("__getitem__",
    //      [](IntervalList & self, IntervalList::size_type i) -> IntervalList::value_type & {
    //          if (i >= self.size()) {
    //              throw py::index_error();
    //          }
    //          return self[i];
    //      })

    // // Set and get a slice
    // .def("__setitem__",
    //      [](IntervalList & self, py::slice slice, py::buffer other) {
    //          size_t start, stop, step, slicelength;
    //          if (!slice.compute(self.size(), &start, &stop, &step,
    //                             &slicelength)) {
    //              throw py::error_already_set();
    //          }
    //          pybuffer_check_1D <IntervalList::value_type> (other);
    //          py::buffer_info info = other.request();
    //          IntervalList::value_type * raw =
    //              reinterpret_cast <IntervalList::value_type *> (info.ptr);

    //          if (slicelength != info.size) {
    //              throw std::runtime_error(
    //                 "Left and right hand size of slice assignment have different sizes!");
    //          }

    //          for (size_t i = 0; i < slicelength; ++i) {
    //              self[start] = raw[i];
    //              start += step;
    //          }
    //      })
    // .def("__setitem__",
    //      [](IntervalList & self, py::slice slice, const IntervalList::value_type & t) {
    //          size_t start, stop, step, slicelength;
    //          if (!slice.compute(self.size(), &start, &stop, &step,
    //                             &slicelength)) {
    //              throw py::error_already_set();
    //          }
    //          for (size_t i = 0; i < slicelength; ++i) {
    //              self[start] = t;
    //              start += step;
    //          }
    //      })
    // .def("__getitem__",
    //      [](IntervalList & self, py::slice slice) {
    //          size_t start, stop, step, slicelength;
    //          if (!slice.compute(self.size(), &start, &stop, &step,
    //                             &slicelength)) {
    //              throw py::error_already_set();
    //          }
    //          std::unique_ptr <IntervalList> ret(new IntervalList(slicelength));
    //          for (size_t i = 0; i < slicelength; ++i) {
    //              (*ret)[i] = self[start];
    //              start += step;
    //          }
    //          return ret;
    //      })

    // // Set and get explicit indices
    // .def("__setitem__",
    //      [](IntervalList & self, py::array_t <int64_t> indices, py::buffer other) {
    //          pybuffer_check_1D <IntervalList::value_type> (other);
    //          py::buffer_info info = other.request();
    //          IntervalList::value_type * raw =
    //              reinterpret_cast <IntervalList::value_type *> (info.ptr);

    //          if (indices.size() != info.size) {
    //              throw std::runtime_error(
    //                        "Left and right hand indexed assignment have different sizes!");
    //          }

    //          auto * dat = indices.data();

    //          for (size_t i = 0; i < info.size; ++i) {
    //              self[dat[i]] = raw[i];
    //          }
    //      })
    // .def("__setitem__",
    //      [](IntervalList & self, py::array_t <int64_t> indices, const IntervalList::value_type & t) {
    //          auto * dat = indices.data();
    //          for (size_t i = 0; i < indices.size(); ++i) {
    //              self[dat[i]] = t;
    //          }
    //      })
    // .def("__getitem__",
    //      [](IntervalList & self, py::array_t <int64_t> indices) {
    //          auto * dat = indices.data();
    //          std::unique_ptr <IntervalList> ret(new IntervalList(indices.size()));
    //          for (size_t i = 0; i < indices.size(); ++i) {
    //              (*ret)[i] = self[dat[i]];
    //          }
    //          return ret;
    //      })
    // .def("__lt__", [](const IntervalList & self, IntervalList::value_type val) {
    //          py::array_t <bool> ret(self.size());
    //          auto result = ret.mutable_data();
    //          for (size_t i = 0; i < self.size(); ++i) {
    //              result[i] = (self[i] < val);
    //          }
    //          return ret;
    //      })
    // .def("__le__", [](const IntervalList & self, IntervalList::value_type val) {
    //          py::array_t <bool> ret(self.size());
    //          auto result = ret.mutable_data();
    //          for (size_t i = 0; i < self.size(); ++i) {
    //              result[i] = (self[i] <= val);
    //          }
    //          return ret;
    //      })
    // .def("__gt__", [](const IntervalList & self, IntervalList::value_type val) {
    //          py::array_t <bool> ret(self.size());
    //          auto result = ret.mutable_data();
    //          for (size_t i = 0; i < self.size(); ++i) {
    //              result[i] = (self[i] > val);
    //          }
    //          return ret;
    //      })
    // .def("__ge__", [](const IntervalList & self, IntervalList::value_type val) {
    //          py::array_t <bool> ret(self.size());
    //          auto result = ret.mutable_data();
    //          for (size_t i = 0; i < self.size(); ++i) {
    //              result[i] = (self[i] >= val);
    //          }
    //          return ret;
    //      })
    // .def("__eq__", [](const IntervalList & self, IntervalList::value_type val) {
    //          py::array_t <bool> ret(self.size());
    //          auto result = ret.mutable_data();
    //          for (size_t i = 0; i < self.size(); ++i) {
    //              result[i] = (self[i] == val);
    //          }
    //          return ret;
    //      })
    // .def("__ne__", [](const IntervalList & self, IntervalList::value_type val) {
    //          py::array_t <bool> ret(self.size());
    //          auto result = ret.mutable_data();
    //          for (size_t i = 0; i < self.size(); ++i) {
    //              result[i] = (self[i] != val);
    //          }
    //          return ret;
    //      })

    // // string representation
    // .def("__repr__",
    //      [name](IntervalList const & self) {
    //          size_t npre = 1;
    //          if (self.size() > 2) {
    //              npre = 2;
    //          }
    //          size_t npost = 0;
    //          if (self.size() > 1) {
    //              npost = 1;
    //          }
    //          if (self.size() > 3) {
    //              npost = 2;
    //          }
    //          std::string dots = "";
    //          if (self.size() > 4) {
    //              dots = " ...";
    //          }
    //          std::ostringstream o;
    //          o << "<" << name << " " << self.size() << " elements:";
    //          for (size_t i = 0; i < npre; ++i) {
    //              o << " " << self[i];
    //          }
    //          o << dots;
    //          for (size_t i = 0; i < npost; ++i) {
    //              o << " " << self[self.size() - npost + i];
    //          }
    //          o << ">";
    //          return o.str();
    //      });


}
