
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void sum_detectors(py::array_t <int64_t,
                                py::array::c_style | py::array::forcecast> detectors,
                   py::array_t <unsigned char,
                                py::array::c_style | py::array::forcecast> shared_flags,
                   unsigned char shared_flag_mask,
                   py::array_t <double,
                                py::array::c_style | py::array::forcecast> det_data,
                   py::array_t <unsigned char,
                                py::array::c_style | py::array::forcecast> det_flags,
                   unsigned char det_flag_mask,
                   py::array_t <double,
                                py::array::c_style | py::array::forcecast> sum_data,
                   py::array_t <int64_t,
                                py::array::c_style | py::array::forcecast> hits) {
    auto fast_detectors = detectors.unchecked <1>();
    auto fast_shared_flags = shared_flags.unchecked <1>();
    auto fast_det_data = det_data.unchecked <2>();
    auto fast_det_flags = det_flags.unchecked <2>();
    auto fast_sum_data = sum_data.mutable_unchecked <1>();
    auto fast_hits = hits.mutable_unchecked <1>();

    size_t ndet = fast_detectors.shape(0);
    size_t nsample = fast_det_data.shape(1);

    size_t buflen = 10000;
    size_t nbuf = int(ceilf(double(nsample) / buflen));

    #pragma omp parallel for schedule(static, 1)
    for (size_t ibuf = 0; ibuf < nbuf; ++ibuf) {
        size_t sample_start = ibuf * buflen;
        size_t sample_stop = std::min(sample_start + buflen, nsample);
        for (size_t idet = 0; idet < ndet; ++idet) {
            int64_t row = fast_detectors(idet);
            for (size_t sample = sample_start; sample < sample_stop; ++sample) {
                if ((fast_shared_flags(sample) & shared_flag_mask) != 0) continue;
                if ((fast_det_flags(row, sample) & det_flag_mask) != 0) continue;
                fast_sum_data(sample) += fast_det_data(row, sample);
                fast_hits(sample)++;
            }
        }
    }

    return;
}

void subtract_mean(py::array_t <int64_t,
                                py::array::c_style | py::array::forcecast> detectors,
                   py::array_t <double,
                                py::array::c_style | py::array::forcecast> det_data,
                   py::array_t <double,
                                py::array::c_style | py::array::forcecast> sum_data,
                   py::array_t <int64_t,
                                py::array::c_style | py::array::forcecast> hits) {
    auto fast_detectors = detectors.unchecked <1>();
    auto fast_det_data = det_data.mutable_unchecked <2>();
    auto fast_sum_data = sum_data.mutable_unchecked <1>();
    auto fast_hits = hits.unchecked <1>();

    size_t ndet = fast_detectors.shape(0);
    size_t nsample = fast_det_data.shape(1);

    size_t buflen = 10000;
    size_t nbuf = int(ceilf(double(nsample) / buflen));

    #pragma omp parallel for schedule(static, 10000)
    for (size_t sample = 0; sample < nsample; ++sample) {
        if (fast_hits(sample) != 0) {
            fast_sum_data(sample) /= fast_hits(sample);
        }
    }

    #pragma omp parallel for schedule(static, 1)
    for (size_t ibuf = 0; ibuf < nbuf; ++ibuf) {
        size_t sample_start = ibuf * buflen;
        size_t sample_stop = std::min(sample_start + buflen, nsample);
        for (size_t idet = 0; idet < ndet; ++idet) {
            int64_t row = fast_detectors(idet);
            for (size_t sample = sample_start; sample < sample_stop; ++sample) {
                fast_det_data(row, sample) -= fast_sum_data(sample);
            }
        }
    }

    return;
}

void init_tod_filter(py::module & m) {
    m.def("legendre",
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
              toast::legendre(px, ptemplates, start_order, stop_order, nsample);

              return;
          }, py::arg("x"), py::arg("templates"), py::arg("start_order"),
          py::arg(
              "stop_order"),
          R"(
        Populate an array of Legendre polynomials at x (in range [-1, 1])
        Args:
        Returns:
            None.
    )");

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

    m.def("bin_proj",
          [](py::buffer signal, py::buffer templates, py::buffer good,
             py::buffer proj) {
              pybuffer_check_1D <double> (signal);
              pybuffer_check_1D <double> (proj);
              pybuffer_check_1D <uint8_t> (good);
              py::buffer_info info_signal = signal.request();
              py::buffer_info info_templates = templates.request();
              py::buffer_info info_good = good.request();
              py::buffer_info info_proj = proj.request();

              size_t nsample = info_signal.size;
              size_t ntemplate = info_templates.size / nsample;
              if (ntemplate != info_proj.size) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "templates and proj must have consistent sizes, not " << ntemplate << ", " << info_proj.size;
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }

              double * psignal = reinterpret_cast <double *> (info_signal.ptr);
              double * ptemplates = reinterpret_cast <double *> (info_templates.ptr);
              double * pproj = reinterpret_cast <double *> (info_proj.ptr);
              uint8_t * pgood = reinterpret_cast <uint8_t *> (info_good.ptr);
              toast::bin_proj(psignal, ptemplates, pgood, pproj, nsample, ntemplate);

              return;
          }, py::arg("signal"), py::arg("templates"), py::arg("good"), py::arg(
              "proj"),
          R"(
        Perform dot products between signal and templates

        Args:

        Returns:
            None.

    )");

    m.def("bin_invcov",
          [](py::buffer templates, py::buffer good,
             py::buffer invcov) {
              pybuffer_check_1D <uint8_t> (good);
              py::buffer_info info_templates = templates.request();
              py::buffer_info info_good = good.request();
              py::buffer_info info_invcov = invcov.request();

              size_t nsample = info_good.size;
              size_t ntemplate = info_templates.size / nsample;
              if (ntemplate * ntemplate != info_invcov.size) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "templates and invcov must have consistent sizes, not " << ntemplate << ", " << info_invcov.size;
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }

              double * ptemplates = reinterpret_cast <double *> (info_templates.ptr);
              double * pinvcov = reinterpret_cast <double *> (info_invcov.ptr);
              uint8_t * pgood = reinterpret_cast <uint8_t *> (info_good.ptr);
              toast::bin_invcov(ptemplates, pgood, pinvcov, nsample, ntemplate);

              return;
          }, py::arg("templates"), py::arg("good"), py::arg(
              "invcov"),
          R"(
        Perform dot products between templates

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

    m.def("filter_poly2D",
          [](py::buffer det_groups, py::buffer templates, py::buffer signals,
             py::buffer masks, py::buffer coeff) {
              pybuffer_check <uint8_t> (masks);
              pybuffer_check <int32_t> (det_groups);
              pybuffer_check <double> (templates);
              pybuffer_check <double> (signals);
              pybuffer_check <double> (coeff);
              py::buffer_info info_masks = masks.request();
              py::buffer_info info_detgroups = det_groups.request();
              py::buffer_info info_templates = templates.request();
              py::buffer_info info_signals = signals.request();
              py::buffer_info info_coeff = coeff.request();

// Check dimensions
              if (info_signals.ndim != 2) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Signals array should have 2 dimensions.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t nsample = info_signals.shape[0];
              int32_t ndet = info_signals.shape[1];
              if (info_masks.ndim != 2) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Masks array should have 2 dimensions.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              if ((info_masks.shape[0] != nsample) || (info_masks.shape[1] != ndet)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Masks array dimensions are different than signals array";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              if (info_templates.ndim != 2) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Templates array should have 2 dimensions.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              if (info_templates.shape[0] != ndet) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "First dimension of templates array should be number of dets.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int32_t nmodes = info_templates.shape[1];
              if (info_coeff.ndim != 3) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Coefficient array should have 3 dimensions.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              if ((info_coeff.shape[0] != nsample) || (info_coeff.shape[2] != nmodes)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Coeff array dimensions are inconsistent";
                  log.error(o.str().c_str());
                  throw std::runtime_error(
                            o.str().c_str());
              }
              int32_t ngroup = info_coeff.shape[1];
              int32_t * raw_detgroups = reinterpret_cast <int32_t *> (info_detgroups.ptr);
              for (int32_t i = 0; i < ndet; ++i) {
                  if (raw_detgroups[i] >= ngroup) {
                      auto log = toast::Logger::get();
                      std::ostringstream o;
                      o << "det " << i << ": group " << raw_detgroups[i] << " invalid for " << ngroup << " groups";
                      log.error(o.str().c_str());
                      throw std::runtime_error(o.str().c_str());
                  }
              }
              uint8_t * raw_masks = reinterpret_cast <uint8_t *> (info_masks.ptr);

              double * raw_templates = reinterpret_cast <double *> (info_templates.ptr);
              double * raw_signals = reinterpret_cast <double *> (info_signals.ptr);
              double * raw_coeff = reinterpret_cast <double *> (info_coeff.ptr);
              toast::filter_poly2D_solve(nsample, ndet, ngroup, nmodes, raw_detgroups,
                                         raw_templates, raw_masks, raw_signals,
                                         raw_coeff);
              return;
          }, py::arg("det_groups"), py::arg("templates"), py::arg("signals"),
          py::arg("masks"),
          py::arg(
              "coeff"), R"(
        Solves for 2D polynomial coefficients at each sample.

        Args:
            det_groups (array, int32):  The group index for each detector index.
            templates (array, float64):  The N_detectors x N_modes templates.
            signals (array, float64):  The N_sample x N_detector data.
            masks (array, uint8):  The N_sample x N_detector mask.
            coeff (array, float64):  The N_sample x N_group x N_mode output
                coefficients.

        Returns:
            None.

    )");

    m.def("sum_detectors", &sum_detectors);
    m.def("subtract_mean", &subtract_mean);

    return;
}
