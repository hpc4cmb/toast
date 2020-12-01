
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


template <typename T>
void register_scan_map(py::module & m, char const * name) {
    m.def(name,
          [](int64_t npix_submap, int64_t nmap, py::buffer submap, py::buffer subpix,
             py::buffer mapdata, py::buffer weights, py::buffer tod) {
              pybuffer_check_1D <int64_t> (submap);
              pybuffer_check_1D <int64_t> (subpix);
              pybuffer_check_1D <T> (mapdata);
              pybuffer_check_1D <double> (weights);
              pybuffer_check_1D <double> (tod);
              py::buffer_info info_submap = submap.request();
              py::buffer_info info_subpix = subpix.request();
              py::buffer_info info_mapdata = mapdata.request();
              py::buffer_info info_weights = weights.request();
              py::buffer_info info_tod = tod.request();
              size_t nsamp = info_tod.size;
              if ((info_submap.size != nsamp) ||
                  (info_subpix.size != nsamp)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              size_t nw = (size_t)(info_weights.size / nmap);
              if (nw != nsamp) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t * rawsubmap = reinterpret_cast <int64_t *> (info_submap.ptr);
              int64_t * rawsubpix = reinterpret_cast <int64_t *> (info_subpix.ptr);
              T * rawmapdata = reinterpret_cast <T *> (info_mapdata.ptr);
              double * rawweights = reinterpret_cast <double *> (info_weights.ptr);
              double * rawtod = reinterpret_cast <double *> (info_tod.ptr);
              toast::scan_local_map <T> (rawsubmap, npix_submap, rawweights, nmap,
                                         rawsubpix, rawmapdata, rawtod, nsamp);
              return;
          }, py::arg("npix_submap"), py::arg("nmap"), py::arg("submap"),
          py::arg("subpix"),
          py::arg("mapdata"), py::arg("weights"), py::arg(
              "tod"), R"(
        Sample a map into a timestream.

        This uses a local piece of a distributed map and the local pointing matrix
        to generate timestream values.

        Args:
            npix_submap (int):  The number of pixels in each submap.
            nmap (int):  The number of non-zeros in each row of the pointing matrix.
            submap (array, int64):  For each time domain sample, the submap index
                within the local map (i.e. including only submap)
            subpix (array, int64):  For each time domain sample, the pixel index
                within the submap.
            mapdata (array):  The flattened local piece of the map.
            weights (array, float64):  The pointing matrix weights for each time
                sample and map.
            tod (array, float64):  The timestream on which to accumulate the map
                values.

        Returns:
            None.

    )");
    return;
}

template <typename T>
void register_fast_scanning(py::module & m, char const * name) {
    m.def(name,
          [](py::buffer tod, py::buffer pix, py::buffer weights, py::buffer mapdata) {
              pybuffer_check_1D <int64_t> (pix);
              pybuffer_check_1D <T> (mapdata);
              pybuffer_check_1D <double> (weights);
              pybuffer_check_1D <double> (tod);
              py::buffer_info info_pix = pix.request();
              py::buffer_info info_mapdata = mapdata.request();
              py::buffer_info info_weights = weights.request();
              py::buffer_info info_tod = tod.request();
              size_t nsamp = info_tod.size;
              if (info_pix.size != nsamp) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              size_t nw = (size_t)(info_weights.size / nsamp);
              int64_t * rawpix = reinterpret_cast <int64_t *> (info_pix.ptr);
              T * rawmapdata = reinterpret_cast <T *> (info_mapdata.ptr);
              double * rawweights = reinterpret_cast <double *> (info_weights.ptr);
              double * rawtod = reinterpret_cast <double *> (info_tod.ptr);
              toast::fast_scanning <T> (rawtod, nsamp, rawpix, rawweights,
                                        nw, rawmapdata);
              return;
          }, py::arg("tod"), py::arg("pix"), py::arg("weights"), py::arg(
              "mapdata"), R"(
        Scan global maps into timestreams.

        Args:
            tod (array, float64):  The timestream on which to accumulate the map
                values.
            pix (array, int64):  For each time domain sample, the pixel index.
            weights (array, float64):  The pointing matrix weights for each time
                sample and map.
            mapdata (array):  The flattened local piece of the map.

        Returns:
            None.

    )");
    return;
}

void init_todmap_scanning(py::module & m) {
    register_scan_map <double> (m, "scan_map_float64");
    register_scan_map <float> (m, "scan_map_float32");
    register_scan_map <int64_t> (m, "scan_map_int64");
    register_scan_map <int32_t> (m, "scan_map_int32");
    register_fast_scanning <double> (m, "fast_scanning_float64");
    register_fast_scanning <float> (m, "fast_scanning_float32");
    return;
}
