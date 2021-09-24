
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_tod_pointing(py::module & m) {
    m.def("healpix_pixels",
          [](toast::HealpixPixels const & hpix, bool nest,
             py::buffer pdata, py::object flags, py::buffer pixels) {
              pybuffer_check_1D <double> (pdata);
              pybuffer_check_1D <int64_t> (pixels);
              py::buffer_info info_pdata = pdata.request();
              py::buffer_info info_pixels = pixels.request();
              size_t n = (size_t)(info_pdata.size / 4);
              if (info_pixels.size != n) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawpdata = reinterpret_cast <double *> (info_pdata.ptr);
              int64_t * rawpixels = reinterpret_cast <int64_t *> (info_pixels.ptr);
              uint8_t * rawflags = NULL;
              if (!flags.is_none()) {
                  auto flagbuf = py::cast <py::buffer> (flags);
                  pybuffer_check_1D <uint8_t> (flagbuf);
                  py::buffer_info info_flags = flagbuf.request();
                  if (info_flags.size != n) {
                      auto log = toast::Logger::get();
                      std::ostringstream o;
                      o << "Flag buffer size is not consistent.";
                      log.error(o.str().c_str());
                      throw std::runtime_error(o.str().c_str());
                  }
                  rawflags = reinterpret_cast <uint8_t *> (info_flags.ptr);
              }
              toast::healpix_pixels(hpix, nest, n, rawpdata, rawflags,
                                    rawpixels);
              return;
          }, py::arg("hpix"), py::arg("nest"), py::arg("pdata"), py::arg("flags"),
          py::arg(
              "pixels"), R"(
        Compute the healpix pixel indices for one detector.

        Args:
            hpix (HealpixPixels):  The healpix projection object.
            nest (bool):  If True, then use NESTED ordering, else RING.
            pdata (array, float64):  The flat-packed array of detector quaternions.
            flags (array, uint8):  The pointing flags.
            pixels (array, int64):  The detector pixel indices to store the result.

        Returns:
            None.

    )");

    m.def("stokes_weights",
          [](double eps, double cal, std::string const & mode, py::buffer pdata,
             py::object hwpang, py::object flags, py::buffer weights) {
              pybuffer_check_1D <double> (pdata);
              pybuffer_check_1D <double> (weights);
              py::buffer_info info_pdata = pdata.request();
              py::buffer_info info_weights = weights.request();
              size_t n = (size_t)(info_pdata.size / 4);
              size_t nw = n;
              if (mode.compare("IQU") == 0) {
                  nw = (size_t)(info_weights.size / 3);
              }
              if (nw != n) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawpdata = reinterpret_cast <double *> (info_pdata.ptr);
              double * rawweights = reinterpret_cast <double *> (info_weights.ptr);
              uint8_t * rawflags = NULL;
              if (!flags.is_none()) {
                  auto flagbuf = py::cast <py::buffer> (flags);
                  pybuffer_check_1D <uint8_t> (flagbuf);
                  py::buffer_info info_flags = flagbuf.request();
                  if (info_flags.size != n) {
                      auto log = toast::Logger::get();
                      std::ostringstream o;
                      o << "Flag buffer size is not consistent.";
                      log.error(o.str().c_str());
                      throw std::runtime_error(o.str().c_str());
                  }
                  rawflags = reinterpret_cast <uint8_t *> (info_flags.ptr);
              }
              double * rawhwpang = NULL;
              if (!hwpang.is_none()) {
                  auto hwpbuf = py::cast <py::buffer> (hwpang);
                  pybuffer_check_1D <double> (hwpbuf);
                  py::buffer_info info_hwpang = hwpbuf.request();
                  if (info_hwpang.size != n) {
                      auto log = toast::Logger::get();
                      std::ostringstream o;
                      o << "HWP buffer size is not consistent.";
                      log.error(o.str().c_str());
                      throw std::runtime_error(o.str().c_str());
                  }
                  rawhwpang = reinterpret_cast <double *> (info_hwpang.ptr);
              }
              toast::stokes_weights(eps, cal, mode, n, rawpdata, rawhwpang,
                                    rawflags, rawweights);
              return;
          }, py::arg("eps"), py::arg("cal"), py::arg("mode"), py::arg("pdata"),
          py::arg("hwpang").none(true), py::arg("flags"),
          py::arg(
              "weights"), R"(
        Compute the Stokes weights for one detector.

        Args:
            eps (float):  The cross polar response.
            cal (float):  A constant to apply to the pointing weights.
            mode (str):  Either "I" or "IQU".
            pdata (array, float64):  The flat-packed array of detector quaternions.
            hwpang (array, float64):  The HWP angles.
            flags (array, uint8):  The pointing flags.
            weights (array, float64):  The flat packed detector weights for the specified
               mode.

        Returns:
            None.

    )");

    return;
}
