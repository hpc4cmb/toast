
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_tod_filter(py::module & m) {
    m.def("chebyshev",
          [](py::buffer x, py::buffer templates, size_t start_order,
             size_t stop_order) {
              pybuffer_check_1D <double> (x);
              py::buffer_info info_x = x.request();
              py::buffer_info info_templates = templates.request();

              size_t nsample = info_x.size;
              size_t ntemplate = info_templates.size / nsample;
              if (ntemplate != stop_order - start_order) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Size of templates does not match x, and order";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }

              double * px = reinterpret_cast <double *> (info_x.ptr);
              double * ptemplates = reinterpret_cast <double *> (info_templates.ptr);
              toast::chebyshev(px, ptemplates, start_order, stop_order, nsample);

              return;
          }, py::arg("x"), py::arg("templates"), py::arg("start_order"),
          py::arg(
              "stop_order"),
          R"(
        Populate an array of Chebyshev polynomials at x (in range [-1, 1])

        Args:

        Returns:
            None.

    )");

    m.def("bin_templates",
          [](py::buffer signal, py::buffer templates, py::buffer good,
             py::buffer invcov,
             py::buffer proj) {
              pybuffer_check_1D <double> (signal);
              pybuffer_check_1D <double> (proj);
              pybuffer_check_1D <uint8_t> (good);
              py::buffer_info info_signal = signal.request();
              py::buffer_info info_templates = templates.request();
              py::buffer_info info_good = good.request();
              py::buffer_info info_invcov = invcov.request();
              py::buffer_info info_proj = proj.request();

              size_t nsample = info_signal.size;
              size_t ntemplate = info_templates.size / nsample;
              if ((ntemplate * ntemplate != info_invcov.size) ||
                  (ntemplate != info_proj.size)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "templates, invcov and proj must have consistent sizes, not " << ntemplate << ", " << info_invcov.size << ", " << info_proj.size;
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }

              double * psignal = reinterpret_cast <double *> (info_signal.ptr);
              double * ptemplates = reinterpret_cast <double *> (info_templates.ptr);
              double * pinvcov = reinterpret_cast <double *> (info_invcov.ptr);
              double * pproj = reinterpret_cast <double *> (info_proj.ptr);
              uint8_t * pgood = reinterpret_cast <uint8_t *> (info_good.ptr);
              toast::bin_templates(psignal, ptemplates, pgood, pinvcov, pproj, nsample,
                                   ntemplate);

              return;
          }, py::arg("signal"), py::arg("templates"), py::arg("good"), py::arg(
              "invcov"), py::arg(
              "proj"),
          R"(
        Perform dot products between signal and templates

        Args:

        Returns:
            None.

    )");

    m.def("add_templates",
          [](py::buffer signal, py::buffer templates, py::buffer coeff) {
              pybuffer_check_1D <double> (signal);
              py::buffer_info info_signal = signal.request();
              py::buffer_info info_templates = templates.request();
              py::buffer_info info_coeff = coeff.request();

              size_t nsample = info_signal.size;
              size_t ntemplate = info_coeff.size;
              if (ntemplate * nsample != info_templates.size) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "signal, templates and coeff have inconsistent sizes";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }

              double * psignal = reinterpret_cast <double *> (info_signal.ptr);
              double * ptemplates = reinterpret_cast <double *> (info_templates.ptr);
              double * pcoeff = reinterpret_cast <double *> (info_coeff.ptr);
              toast::add_templates(psignal, ptemplates, pcoeff, nsample, ntemplate);

              return;
          }, py::arg("signal"), py::arg("templates"), py::arg(
              "coeff"),
          R"(
        Co-add templates using coeff onto signal

        Args:

        Returns:
            None.

    )");

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
