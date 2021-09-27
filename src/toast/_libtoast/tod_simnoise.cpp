
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_tod_simnoise(py::module & m) {
    m.def("tod_sim_noise_timestream",
          [](uint64_t realization, uint64_t telescope, uint64_t component,
             uint64_t obsindx, uint64_t detindx, double rate, int64_t firstsamp,
             int64_t oversample, py::buffer freq, py::buffer psd, py::buffer noise) {
              pybuffer_check_1D <double> (freq);
              pybuffer_check_1D <double> (psd);
              pybuffer_check_1D <double> (noise);
              py::buffer_info info_freq = freq.request();
              py::buffer_info info_psd = psd.request();
              py::buffer_info info_noise = noise.request();
              int64_t psdlen = info_freq.size;
              int64_t samples = info_noise.size;
              if ((int64_t)info_psd.size != psdlen) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawfreq = reinterpret_cast <double *> (info_freq.ptr);
              double * rawpsd = reinterpret_cast <double *> (info_psd.ptr);
              double * rawnoise = reinterpret_cast <double *> (info_noise.ptr);
              toast::tod_sim_noise_timestream(
                  realization, telescope, component, obsindx, detindx, rate, firstsamp,
                  samples, oversample, rawfreq, rawpsd, psdlen, rawnoise);
              return;
          }, py::arg("realization"), py::arg("telescope"), py::arg("component"),
          py::arg("obsindx"), py::arg("detindx"), py::arg("rate"),
          py::arg("firstsamp"), py::arg("oversample"), py::arg("freq"),
          py::arg("psd"), py::arg(
              "noise"), R"(
        Generate a noise timestream, given a starting RNG state.

        Use the RNG parameters to generate unit-variance Gaussian samples
        and then modify the Fourier domain amplitudes to match the desired
        PSD.

        The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
        which each consist of two unsigned 64bit integers.  These four
        numbers together uniquely identify a single sample.  We construct
        those four numbers in the following way:

        key1 = realization * 2^32 + telescope * 2^16 + component
        key2 = obsindx * 2^32 + detindx
        counter1 = currently unused (0)
        counter2 = sample in stream

        counter2 is incremented internally by the RNG function as it calls
        the underlying Random123 library for each sample.

        Args:
            realization (int): the Monte Carlo realization.
            telescope (int): a unique index assigned to a telescope.
            component (int): a number representing the type of timestream
                we are generating (detector noise, common mode noise,
                atmosphere, etc).
            obsindx (int): the global index of this observation.
            detindx (int): the global index of this detector.
            rate (float): the sample rate.
            firstsamp (int): the start sample in the stream.
            oversample (int): the factor by which to expand the FFT length
                beyond the number of samples.
            freq (array): the frequency points of the PSD.
            psd (array): the PSD values.
            noise (array): the output noise timestream.

        Returns:
            None

    )");

    m.def("tod_sim_noise_timestream_batch",
          [](uint64_t realization, uint64_t telescope, uint64_t component,
             uint64_t obsindx, double rate, int64_t firstsamp,
             int64_t oversample, py::buffer detindices, py::buffer freq,
             py::buffer psds, py::buffer noise) {
              pybuffer_check_1D <uint64_t> (detindices);
              pybuffer_check_1D <double> (freq);

              pybuffer_check <double> (psds);
              pybuffer_check <double> (noise);

              py::buffer_info info_detind = detindices.request();
              py::buffer_info info_freq = freq.request();
              py::buffer_info info_psd = psds.request();
              py::buffer_info info_noise = noise.request();

              int64_t ndet = info_detind.size;
              int64_t psdlen = info_freq.size;
              if ((info_psd.ndim != 2) || (info_noise.ndim != 2)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "psds and noise should be 2D arrays.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              if ((info_psd.shape[0] != ndet) || (info_noise.shape[0] != ndet)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "first dimension of psds and noise should match length of detindices.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              if (info_psd.shape[1] != psdlen) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Number of psd points does not match frequency array.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t samples = info_noise.shape[1];

              uint64_t * rawdetind = reinterpret_cast <uint64_t *> (info_detind.ptr);
              double * rawfreq = reinterpret_cast <double *> (info_freq.ptr);
              double * rawpsd = reinterpret_cast <double *> (info_psd.ptr);
              double * rawnoise = reinterpret_cast <double *> (info_noise.ptr);

              toast::tod_sim_noise_timestream_batch(realization, telescope, component,
                                                    obsindx, rate, firstsamp, samples,
                                                    oversample, ndet, rawdetind, psdlen,
                                                    rawfreq,
                                                    rawpsd, rawnoise);
              return;
          }, py::arg("realization"), py::arg("telescope"), py::arg("component"),
          py::arg("obsindx"), py::arg("rate"),
          py::arg("firstsamp"), py::arg("oversample"), py::arg("detindices"),
          py::arg("freq"), py::arg("psds"), py::arg(
              "noise"), R"(
        Generate multiple noise timestreams in parallel.

        Use the RNG parameters to generate unit-variance Gaussian samples
        and then modify the Fourier domain amplitudes to match the desired
        PSD.

        The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
        which each consist of two unsigned 64bit integers.  These four
        numbers together uniquely identify a single sample.  We construct
        those four numbers in the following way:

        key1 = realization * 2^32 + telescope * 2^16 + component
        key2 = obsindx * 2^32 + detindx
        counter1 = currently unused (0)
        counter2 = sample in stream

        counter2 is incremented internally by the RNG function as it calls
        the underlying Random123 library for each sample.

        Args:
            realization (int): the Monte Carlo realization.
            telescope (int): a unique index assigned to a telescope.
            component (int): a number representing the type of timestream
                we are generating (detector noise, common mode noise,
                atmosphere, etc).
            obsindx (int): the global index of this observation.
            rate (float): the sample rate.
            firstsamp (int): the start sample in the stream.
            oversample (int): the factor by which to expand the FFT length
                beyond the number of samples.
            detindices (array): array with global index for each detector.
            freq (array): the frequency points of all PSDs.
            psd (array): a 2D array containing the PSD for each detector.
            noise (array): a 2D array of the output noise timestreams.

        Returns:
            None

    )");

    return;
}
