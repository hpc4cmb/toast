
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_fod_psd(py::module & m) {
    m.def("fod_crosssums", [](py::buffer x, py::buffer y, py::buffer good,
                              int64_t lagmax, py::buffer sums, py::buffer hits) {
              pybuffer_check_1D <double> (x);
              pybuffer_check_1D <double> (y);
              pybuffer_check_1D <uint8_t> (good);
              pybuffer_check_1D <double> (sums);
              pybuffer_check_1D <int64_t> (hits);
              py::buffer_info info_x = x.request();
              py::buffer_info info_y = y.request();
              py::buffer_info info_good = good.request();
              size_t n = info_x.size;
              if ((info_y.size != n) ||
                  (info_good.size != n)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              py::buffer_info info_sums = sums.request();
              py::buffer_info info_hits = hits.request();
              size_t nlag = (size_t)lagmax;
              if ((info_sums.size != nlag) ||
                  (info_hits.size != nlag)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawx = reinterpret_cast <double *> (info_x.ptr);
              double * rawy = reinterpret_cast <double *> (info_y.ptr);
              uint8_t * rawgood = reinterpret_cast <uint8_t *> (info_good.ptr);
              double * rawsums = reinterpret_cast <double *> (info_sums.ptr);
              int64_t * rawhits = reinterpret_cast <int64_t *> (info_hits.ptr);
              toast::fod_crosssums(n, rawx, rawy, rawgood, lagmax, rawsums, rawhits);
              return;
          }, py::arg("x"), py::arg("y"), py::arg("good"), py::arg("lagmax"),
          py::arg("sums"), py::arg(
              "hits"), R"(
        Accumulate the time domain covariance between two vectors.

        Args:
            x (array_like, float64): The first timestream.
            y (array_like, float64): The second timestream.
            good (array_like, uint8_t): The flags (zero means *BAD*).
            lagmax (int): The maximum sample distance to consider.
            sums (array like, float64): The vector of sums^2 to accumulate for each lag.
            hits (array_like, int64):  The vector of hits to accumulate for each lag.

        Returns:
            None.

    )");

    m.def("fod_autosums", [](py::buffer x, py::buffer good, int64_t lagmax,
                             py::buffer sums, py::buffer hits) {
              pybuffer_check_1D <double> (x);
              pybuffer_check_1D <uint8_t> (good);
              pybuffer_check_1D <double> (sums);
              pybuffer_check_1D <int64_t> (hits);
              py::buffer_info info_x = x.request();
              py::buffer_info info_good = good.request();
              size_t n = info_x.size;
              if (info_good.size != n) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              py::buffer_info info_sums = sums.request();
              py::buffer_info info_hits = hits.request();
              size_t nlag = (size_t)lagmax;
              if ((info_sums.size != nlag) ||
                  (info_hits.size != nlag)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawx = reinterpret_cast <double *> (info_x.ptr);
              uint8_t * rawgood = reinterpret_cast <uint8_t *> (info_good.ptr);
              double * rawsums = reinterpret_cast <double *> (info_sums.ptr);
              int64_t * rawhits = reinterpret_cast <int64_t *> (info_hits.ptr);
              toast::fod_autosums(n, rawx, rawgood, lagmax, rawsums, rawhits);
              return;
          }, py::arg("x"), py::arg("good"), py::arg("lagmax"),
          py::arg("sums"), py::arg(
              "hits"), R"(
        Accumulate the time domain covariance.

        Args:
            x (array_like, float64): The timestream.
            good (array_like, uint8_t): The flags (zero means *BAD*).
            lagmax (int): The maximum sample distance to consider.
            sums (array like, float64): The vector of sums^2 to accumulate for each lag.
            hits (array_like, int64):  The vector of hits to accumulate for each lag.

        Returns:
            None.

    )");

    return;
}
