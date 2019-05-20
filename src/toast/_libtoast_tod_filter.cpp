
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_tod_filter(py::module & m) {
    m.def("filter_polynomial",
          [](int64_t order, py::buffer flags, py::list signals, py::buffer starts,
             py::buffer stops) {
              pybuffer_check_1D <uint8_t> (flags);
              pybuffer_check_1D <int64_t> (starts);
              pybuffer_check_1D <int64_t> (stops);
              py::buffer_info info_starts = starts.request();
              py::buffer_info info_stops = stops.request();
              py::buffer_info info_flags = flags.request();
              size_t nsignal = signals.size();
              size_t nsamp = info_flags.size;
              size_t nscan = info_starts.size;
              if (info_stops.size != nscan) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Starts / stops buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              std::vector <double *> sigs;
              for (auto const & sg : signals) {
                  auto sgbuf = sg.cast <py::buffer> ();
                  pybuffer_check_1D <double> (sgbuf);
                  py::buffer_info info_sg = sgbuf.request();
                  int64_t * rawstarts = reinterpret_cast <int64_t *> (info_starts.ptr);
                  if (info_sg.size != nsamp) {
                      auto log = toast::Logger::get();
                      std::ostringstream o;
                      o << "Signal and flag buffer sizes are not consistent.";
                      log.error(o.str().c_str());
                      throw std::runtime_error(o.str().c_str());
                  }
                  sigs.push_back(reinterpret_cast <double *> (info_sg.ptr));
              }
              uint8_t * rawflags = reinterpret_cast <uint8_t *> (info_flags.ptr);
              int64_t * rawstarts = reinterpret_cast <int64_t *> (info_starts.ptr);
              int64_t * rawstops = reinterpret_cast <int64_t *> (info_stops.ptr);
              toast::filter_polynomial(order, nsamp, rawflags, sigs, nscan,
                                       rawstarts, rawstops);
              return;
          }, py::arg("order"), py::arg("flags"), py::arg("signals"), py::arg("starts"),
          py::arg(
              "stops"), R"(
        Fit and subtract a polynomial from one or more signals.

        Args:
            order (int):  The order of the polynomial.
            flags (array, uint8):  The common flags to use for all signals
            signals (list):  A list of float64 arrays containing the signals.
            starts (array, int64):  The start samples of each scan.
            stops (array, int64):  The stop samples of each scan.

        Returns:
            None.

    )");

    return;
}
