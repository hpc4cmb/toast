
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_map_cov(py::module & m) {
    m.def("cov_accum_diag",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer submap,
             py::buffer subpix,
             py::buffer weights, double scale, py::buffer tod, py::buffer invnpp,
             py::buffer hits, py::buffer zmap) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_accum_diag");
              pybuffer_check_1D <int64_t> (submap);
              pybuffer_check_1D <int64_t> (subpix);
              pybuffer_check_1D <double> (invnpp);
              pybuffer_check_1D <int64_t> (hits);
              pybuffer_check_1D <double> (weights);
              pybuffer_check_1D <double> (zmap);
              pybuffer_check_1D <double> (tod);
              py::buffer_info info_submap = submap.request();
              py::buffer_info info_subpix = subpix.request();
              py::buffer_info info_invnpp = invnpp.request();
              py::buffer_info info_hits = hits.request();
              py::buffer_info info_zmap = zmap.request();
              py::buffer_info info_weights = weights.request();
              py::buffer_info info_tod = tod.request();
              size_t nsamp = info_submap.size;
              size_t nw = (size_t)(info_weights.size / nnz);
              if ((info_subpix.size != nsamp) ||
                  (info_tod.size != nsamp) ||
                  (nw != nsamp)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t * rawsubmap = reinterpret_cast <int64_t *> (info_submap.ptr);
              int64_t * rawsubpix = reinterpret_cast <int64_t *> (info_subpix.ptr);
              int64_t * rawhits = reinterpret_cast <int64_t *> (info_hits.ptr);
              double * rawzmap = reinterpret_cast <double *> (info_zmap.ptr);
              double * rawinvnpp = reinterpret_cast <double *> (info_invnpp.ptr);
              double * rawweights = reinterpret_cast <double *> (info_weights.ptr);
              double * rawtod = reinterpret_cast <double *> (info_tod.ptr);
              toast::cov_accum_diag(
                  nsub, nsubpix, nnz, nsamp, rawsubmap, rawsubpix, rawweights, scale,
                  rawtod, rawzmap, rawhits, rawinvnpp);
              gt.stop("cov_accum_diag");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("submap"),
          py::arg("subpix"), py::arg("weights"), py::arg("scale"), py::arg("tod"),
          py::arg("invnpp"), py::arg("hits"), py::arg(
              "zmap"), R"(
        Accumulate block diagonal noise products

        This uses a pointing matrix and timestream data to accumulate the local pieces
        of the inverse diagonal pixel covariance, hits, and noise weighted map.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            submap (array, int64):  For each time domain sample, the submap index
                within the local map (i.e. including only locally stored submaps)
            subpix (array, int64):  For each time domain sample, the pixel index
                within the submap.
            weights (array, float64):  The pointing matrix weights for each time
                sample and map.
            scale (float):  Optional scaling factor.
            tod (array, float64):  The timestream to accumulate in the noise weighted
                map.
            invnpp (array, float64):  The local buffer of diagonal inverse pixel
                covariances, stored as the lower triangle for each pixel.
            hits (array, int64):  The local hitmap buffer to accumulate.
            zmap (array, float64):  The local noise weighted map buffer.

        Returns:
            None.

    )");

    m.def("cov_accum_diag_hits",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer submap,
             py::buffer subpix, py::buffer hits) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_accum_diag_hits");
              pybuffer_check_1D <int64_t> (submap);
              pybuffer_check_1D <int64_t> (subpix);
              pybuffer_check_1D <int64_t> (hits);
              py::buffer_info info_submap = submap.request();
              py::buffer_info info_subpix = subpix.request();
              py::buffer_info info_hits = hits.request();
              size_t nsamp = info_submap.size;
              if (info_subpix.size != nsamp) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t * rawsubmap = reinterpret_cast <int64_t *> (info_submap.ptr);
              int64_t * rawsubpix = reinterpret_cast <int64_t *> (info_subpix.ptr);
              int64_t * rawhits = reinterpret_cast <int64_t *> (info_hits.ptr);
              toast::cov_accum_diag_hits(
                  nsub, nsubpix, nnz, nsamp, rawsubmap, rawsubpix, rawhits);
              gt.stop("cov_accum_diag_hits");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("submap"),
          py::arg("subpix"), py::arg(
              "hits"), R"(
        Accumulate hit map.

        This uses a pointing matrix to accumulate the local pieces of the hit map.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            submap (array, int64):  For each time domain sample, the submap index
                within the local map (i.e. including only locally stored submaps)
            subpix (array, int64):  For each time domain sample, the pixel index
                within the submap.
            hits (array, int64):  The local hitmap buffer to accumulate.

        Returns:
            None.

    )");

    m.def("cov_accum_diag_invnpp",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer submap,
             py::buffer subpix, py::buffer weights, double scale, py::buffer invnpp) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_accum_diag_invnpp");
              pybuffer_check_1D <int64_t> (submap);
              pybuffer_check_1D <int64_t> (subpix);
              pybuffer_check_1D <double> (invnpp);
              pybuffer_check_1D <double> (weights);
              py::buffer_info info_submap = submap.request();
              py::buffer_info info_subpix = subpix.request();
              py::buffer_info info_invnpp = invnpp.request();
              py::buffer_info info_weights = weights.request();
              size_t nsamp = info_submap.size;
              size_t nw = (size_t)(info_weights.size / nnz);
              if ((info_subpix.size != nsamp) ||
                  (nw != nsamp)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t * rawsubmap = reinterpret_cast <int64_t *> (info_submap.ptr);
              int64_t * rawsubpix = reinterpret_cast <int64_t *> (info_subpix.ptr);
              double * rawinvnpp = reinterpret_cast <double *> (info_invnpp.ptr);
              double * rawweights = reinterpret_cast <double *> (info_weights.ptr);
              toast::cov_accum_diag_invnpp(
                  nsub, nsubpix, nnz, nsamp, rawsubmap, rawsubpix, rawweights, scale,
                  rawinvnpp);
              gt.stop("cov_accum_diag_invnpp");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("submap"),
          py::arg("subpix"), py::arg("weights"), py::arg("scale"), py::arg(
              "invnpp"), R"(
        Accumulate block diagonal noise covariance.

        This uses a pointing matrix to accumulate the local pieces
        of the inverse diagonal pixel covariance.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            submap (array, int64):  For each time domain sample, the submap index
                within the local map (i.e. including only locally stored submaps)
            subpix (array, int64):  For each time domain sample, the pixel index
                within the submap.
            weights (array, float64):  The pointing matrix weights for each time
                sample and map.
            scale (float):  Optional scaling factor.
            invnpp (array, float64):  The local buffer of diagonal inverse pixel
                covariances, stored as the lower triangle for each pixel.

        Returns:
            None.

    )");

    m.def("cov_accum_zmap",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer submap,
             py::buffer subpix, py::buffer weights, double scale, py::buffer tod,
             py::buffer zmap) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_accum_zmap");
              pybuffer_check_1D <int64_t> (submap);
              pybuffer_check_1D <int64_t> (subpix);
              pybuffer_check_1D <double> (weights);
              pybuffer_check_1D <double> (zmap);
              pybuffer_check_1D <double> (tod);
              py::buffer_info info_submap = submap.request();
              py::buffer_info info_subpix = subpix.request();
              py::buffer_info info_zmap = zmap.request();
              py::buffer_info info_weights = weights.request();
              py::buffer_info info_tod = tod.request();
              size_t nsamp = info_submap.size;
              size_t nw = (size_t)(info_weights.size / nnz);
              if ((info_subpix.size != nsamp) ||
                  (info_tod.size != nsamp) ||
                  (nw != nsamp)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t * rawsubmap = reinterpret_cast <int64_t *> (info_submap.ptr);
              int64_t * rawsubpix = reinterpret_cast <int64_t *> (info_subpix.ptr);
              double * rawzmap = reinterpret_cast <double *> (info_zmap.ptr);
              double * rawweights = reinterpret_cast <double *> (info_weights.ptr);
              double * rawtod = reinterpret_cast <double *> (info_tod.ptr);
              toast::cov_accum_zmap(
                  nsub, nsubpix, nnz, nsamp, rawsubmap, rawsubpix, rawweights, scale,
                  rawtod, rawzmap);
              gt.stop("cov_accum_zmap");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("submap"),
          py::arg("subpix"), py::arg("weights"), py::arg("scale"), py::arg("tod"),
          py::arg(
              "zmap"), R"(
        Accumulate the noise weighted map.

        This uses a pointing matrix and timestream data to accumulate the local pieces
        of the noise weighted map.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            submap (array, int64):  For each time domain sample, the submap index
                within the local map (i.e. including only locally stored submaps)
            subpix (array, int64):  For each time domain sample, the pixel index
                within the submap.
            weights (array, float64):  The pointing matrix weights for each time
                sample and map.
            scale (float):  Optional scaling factor.
            tod (array, float64):  The timestream to accumulate in the noise weighted
                map.
            invnpp (array, float64):  The local buffer of diagonal inverse pixel
                covariances, stored as the lower triangle for each pixel.
            hits (array, int64):  The local hitmap buffer to accumulate.
            zmap (array, float64):  The local noise weighted map buffer.

        Returns:
            None.

    )");

    m.def("cov_eigendecompose_diag",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer data,
             py::buffer cond, double threshold, bool invert) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_eigendecompose_diag");
              pybuffer_check_1D <double> (data);
              pybuffer_check_1D <double> (cond);
              py::buffer_info info_data = data.request();
              py::buffer_info info_cond = cond.request();
              int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
              size_t nb = (size_t)(info_data.size / block);
              double * rawdata = reinterpret_cast <double *> (info_data.ptr);
              double * rawcond;
              if (info_cond.size > 0) {
                  if (info_cond.size != nb) {
                      auto log = toast::Logger::get();
                      std::ostringstream o;
                      o << "Buffer sizes are not consistent.";
                      log.error(o.str().c_str());
                      throw std::runtime_error(o.str().c_str());
                  }
                  rawcond = reinterpret_cast <double *> (info_cond.ptr);
                  toast::cov_eigendecompose_diag(nsub, nsubpix, nnz, rawdata, rawcond,
                                                 threshold, invert);
              } else {
                  rawcond = NULL;
                  toast::cov_eigendecompose_diag(nsub, nsubpix, nnz, rawdata, rawcond,
                                                 threshold, invert);
              }
              gt.stop("cov_eigendecompose_diag");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("data"),
          py::arg("cond"), py::arg("threshold"), py::arg(
              "invert"), R"(
        Compute the condition number and optionally invert a covariance.

        This performs and eigendecomposition of the covariance at each pixel and
        computes the condition number.  The covariance is optionally inverted.  Pixels
        where the condition number exceeds the threshold have their covariance set to
        zero.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            data (array, float64):  The local buffer of diagonal pixel covariances,
                stored as the lower triangle for each pixel.
            cond (array, float64):  The local buffer of condition numbers (one per
                pixel).
            threshold (float64):  The threshold on the condition number.
            invert (bool):  Whether to invert the covariance in place.

        Returns:
            None.

    )");

    m.def("cov_mult_diag",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer data1,
             py::buffer data2) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_mult_diag");
              pybuffer_check_1D <double> (data1);
              pybuffer_check_1D <double> (data2);
              py::buffer_info info_data1 = data1.request();
              py::buffer_info info_data2 = data2.request();
              if (info_data1.size != info_data2.size) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawdata1 = reinterpret_cast <double *> (info_data1.ptr);
              double * rawdata2 = reinterpret_cast <double *> (info_data2.ptr);
              toast::cov_mult_diag(nsub, nsubpix, nnz, rawdata1, rawdata2);
              gt.stop("cov_mult_diag");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("data1"),
          py::arg(
              "data2"), R"(
        Multiply two block diagonal covariances.

        This multiplies the covariances within each corresponding pixel in the two
        data buffers.  The result is stored in the first buffer and overwrites the
        input.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            data1 (array, float64):  The first local buffer of diagonal pixel
                covariances, stored as the lower triangle for each pixel.
            data2 (array, float64):  The second local buffer of diagonal pixel
                covariances, stored as the lower triangle for each pixel.

        Returns:
            None.

    )");

    m.def("cov_apply_diag",
          [](int64_t nsub, int64_t nsubpix, int64_t nnz, py::buffer mat,
             py::buffer vec) {
              auto & gt = toast::GlobalTimers::get();
              gt.start("cov_apply_diag");
              pybuffer_check_1D <double> (mat);
              pybuffer_check_1D <double> (vec);
              py::buffer_info info_mat = mat.request();
              py::buffer_info info_vec = vec.request();
              int64_t block = (int64_t)(nnz * (nnz + 1) / 2);
              size_t nb = (size_t)(info_mat.size / block);
              size_t nv = (size_t)(info_vec.size / nnz);
              if (nv != nb) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "Buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * rawmat = reinterpret_cast <double *> (info_mat.ptr);
              double * rawvec = reinterpret_cast <double *> (info_vec.ptr);
              toast::cov_apply_diag(nsub, nsubpix, nnz, rawmat, rawvec);
              gt.stop("cov_apply_diag");
              return;
          }, py::arg("nsub"), py::arg("nsubpix"), py::arg("nnz"), py::arg("mat"),
          py::arg(
              "vec"), R"(
        Apply a covariance to a vector at each pixel.

        This does a matrix-vector multiply at each pixel.

        Args:
            nsub (int):  The number of locally stored submaps.
            nsubpix (int):  The number of pixels in each submap.
            nnz (int):  The number of non-zeros in each row of the pointing matrix.
            mat (array, float64):  The local buffer of diagonal pixel
                covariances, stored as the lower triangle for each pixel.
            vec (array, float64):  The local buffer of vectors for each pixel.

        Returns:
            None.

    )");

    return;
}
