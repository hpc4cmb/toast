
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>
#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


void apply_flags_to_pixels(py::array_t <unsigned char> common_flags,
                           unsigned char common_flag_mask,
                           py::array_t <unsigned char> detector_flags,
                           unsigned char detector_flag_mask,
                           py::array_t <int64_t> pixels) {
    auto fast_common_flags = common_flags.unchecked <1>();
    auto fast_detector_flags = detector_flags.unchecked <1>();
    auto fast_pixels = pixels.mutable_unchecked <1>();

    size_t nsamp = pixels.size();
    #pragma omp parallel for schedule(static, 64)
    for (size_t i = 0; i < nsamp; ++i) {
        unsigned char common_flag = fast_common_flags(i);
        unsigned char detector_flag = fast_detector_flags(i);
        if ((common_flag & common_flag_mask) || (detector_flag & detector_flag_mask)) {
            fast_pixels(i) = -1;
        }
    }
}

void add_offsets_to_signal(py::array_t <double> ref, py::list todslices,
                           py::array_t <double> amplitudes,
                           py::array_t <int64_t> itemplates) {
    auto fast_ref = ref.mutable_unchecked <1>();
    auto fast_amplitudes = amplitudes.unchecked <1>();
    auto fast_itemplates = itemplates.unchecked <1>();
    size_t ntemplate = itemplates.size();

    // Parsing the slices cannot be threaded due to GIL
    std::vector <std::pair <size_t, size_t> > slices;
    for (int i = 0; i < ntemplate; ++i) {
        py::slice todslice = py::slice(todslices[i]);
        py::size_t istart, istop, istep, islicelength;
        if (!todslice.compute(ref.size(), &istart, &istop, &istep,
                              &islicelength)) throw py::error_already_set();
        slices.push_back(std::make_pair(istart, istop));
    }

    // Enabling parallelization made this loop run 10% slower in testing...
    // #pragma omp parallel for
    for (size_t i = 0; i < ntemplate; ++i) {
        int itemplate = fast_itemplates(i);
        double offset = fast_amplitudes(itemplate);
        for (size_t j = slices[i].first; j < slices[i].second; ++j) {
            fast_ref(j) += offset;
        }
    }
}

void project_signal_offsets(py::array_t <double> ref, py::list todslices,
                            py::array_t <double> amplitudes,
                            py::array_t <int64_t> itemplates) {
    auto fast_ref = ref.unchecked <1>();
    auto fast_amplitudes = amplitudes.mutable_unchecked <1>();
    auto fast_itemplates = itemplates.unchecked <1>();
    size_t ntemplate = itemplates.size();

    // Parsing the slices cannot be threaded due to GIL
    std::vector <std::pair <size_t, size_t> > slices;
    for (int i = 0; i < ntemplate; ++i) {
        py::slice todslice = py::slice(todslices[i]);
        py::size_t istart, istop, istep, islicelength;
        if (!todslice.compute(ref.size(), &istart, &istop, &istep,
                              &islicelength)) throw py::error_already_set();
        slices.push_back(std::make_pair(istart, istop));
    }

    // Enabling parallelization made this loop run 20% slower in testing...
    // #pragma omp parallel for
    for (size_t i = 0; i < ntemplate; ++i) {
        double sum = 0;
        for (size_t j = slices[i].first; j < slices[i].second; ++j) {
            sum += fast_ref(j);
        }
        int itemplate = fast_itemplates(i);
        fast_amplitudes(itemplate) += sum;
    }
}

void expand_matrix(py::array_t <double> compressed_matrix,
                   py::array_t <int64_t> local_to_global,
                   int64_t npix,
                   int64_t nnz,
                   py::array_t <int64_t> indices,
                   py::array_t <int64_t> indptr
) {
    auto fast_matrix = compressed_matrix.unchecked <2>();
    auto fast_local_to_global = local_to_global.unchecked <1>();
    auto fast_indices = indices.mutable_unchecked <1>();
    auto fast_indptr = indptr.mutable_unchecked <1>();

    size_t nlocal = fast_local_to_global.shape(0);
    size_t nlocal_tot = fast_matrix.shape(0);
    std::vector <int64_t> col_indices;

    size_t offset = 0;
    for (size_t inz = 0; inz < nnz; ++inz) {
        for (size_t ilocal = 0; ilocal < nlocal; ++ilocal) {
            size_t iglobal = fast_local_to_global[ilocal];
            col_indices.push_back(iglobal + offset);
        }
        offset += npix;
    }

    size_t global_row = 0;
    offset = 0;
    for (size_t inz = 0; inz < nnz; ++inz) {
        size_t global_pixel = 0;
        for (size_t ilocal = 0; ilocal < nlocal; ++ilocal) {
            size_t iglobal = fast_local_to_global[ilocal];
            while (global_pixel < iglobal) {
                fast_indptr[global_row + 1] = offset;
                global_row++;
                global_pixel++;
            }
            for (auto ind : col_indices) {
                fast_indices[offset++] = ind;
            }
            fast_indptr[global_row + 1] = offset;
            global_pixel++;
            global_row++;
        }
        while (global_pixel < npix) {
            fast_indptr[global_row + 1] = offset;
            global_row++;
            global_pixel++;
        }
    }
}

void build_template_covariance(std::vector <int64_t> & starts,
                               std::vector <int64_t> & stops,
                               std::vector <py::array_t <double,
                                                         py::array::c_style | py::array::forcecast> > & templates,
                               py::array_t <double,
                                            py::array::c_style | py::array::forcecast> good,
                               py::array_t <double,
                                            py::array::c_style |
                                            py::array::forcecast> template_covariance) {
    auto fast_good = good.unchecked <1>();
    auto fast_covariance = template_covariance.mutable_unchecked <2>();

    size_t ntemplate = templates.size();

    #pragma omp parallel for schedule(static, 1)
    for (size_t row = 0; row < ntemplate; ++row) {
        auto rowtemplate = templates[row].unchecked <1>();
        size_t rowoffset = starts[row];
        for (size_t col = row; col < ntemplate; ++col) {
            auto coltemplate = templates[col].unchecked <1>();
            size_t coloffset = starts[col];
            double val = 0;
            size_t istart = std::max(starts[row], starts[col]);
            size_t istop = std::min(stops[row], stops[col]);
            if ((row == col) && (istop - istart <= 1)) val = 1;
            for (size_t isample = istart; isample < istop; ++isample) {
                val += rowtemplate(isample - rowoffset)
                       * coltemplate(isample - coloffset)
                       * fast_good(isample);
            }
            fast_covariance(row, col) = val;
            if (row != col) {
                fast_covariance(col, row) = val;
            }
        }
    }
}

void accumulate_observation_matrix(py::array_t <double,
                                                py::array::c_style | py::array::forcecast> c_obs_matrix,
                                   py::array_t <int64_t, py::array::c_style | py::array::forcecast> c_pixels,
                                   py::array_t <double, py::array::c_style | py::array::forcecast> weights,
                                   py::array_t <double, py::array::c_style | py::array::forcecast> templates,
                                   py::array_t <double,
                                                py::array::c_style | py::array::forcecast> template_covariance,
                                   py::array_t <unsigned char, py::array::c_style | py::array::forcecast> good_fit,
                                   py::array_t <unsigned char,
                                                py::array::c_style |
                                                py::array::forcecast> good_bin) {
    /* This function evaluates the cumulative parts of the observation matrix:  P^T N^-1
       Z P,  where Z = I - F(F^T N^-1_F F)^-1 F^T N^-1_F and F is the template matrix,
       (F^T N^-1_F F)^-1 is the template covariance matrix, N^-1_F diagonal has the
       fitting noise weights, N^-1 diagonal has the binning noise weights and P is the
       pointing matrix
     */

    auto fast_obs_matrix = c_obs_matrix.mutable_unchecked <2>();
    auto fast_pixels = c_pixels.unchecked <1>();
    auto fast_weights = weights.unchecked <2>();
    auto fast_templates = templates.unchecked <2>();
    auto fast_covariance = template_covariance.unchecked <2>();
    auto fast_good_fit = good_fit.unchecked <1>();
    auto fast_good_bin = good_bin.unchecked <1>();

    size_t nsample = fast_pixels.shape(0);
    size_t nnz = fast_weights.shape(1);
    size_t ntemplate = fast_templates.shape(1);
    size_t npixtot = fast_obs_matrix.shape(0);
    size_t npix = npixtot / nnz;

    // Count number of dense templates in the beginning of template array
    size_t ndense = 0;
    for (size_t itemplate = 0; itemplate < ntemplate; ++itemplate) {
        size_t nhit = 0;
        for (size_t isample = 0; isample < nsample; ++isample) {
            if (fast_templates(isample, itemplate) != 0) ++nhit;
        }
        if (nhit < 0.1 * nsample) break;
        ++ndense;
    }

    // Build lists of non-zeros for each row of the template matrix
    std::vector <std::vector <size_t> > nonzeros(nsample);
    #pragma omp parallel for schedule(static, 1)
    for (size_t isample = 0; isample < nsample; ++isample) {
        for (size_t itemplate = ndense; itemplate < ntemplate; ++itemplate) {
            if (fast_templates(isample, itemplate) != 0) {
                nonzeros[isample].push_back(itemplate);
            }
        }
    }

    #pragma omp parallel
    {
        int nthreads = 1;
        int idthread = 0;
        #ifdef _OPENMP
        nthreads = omp_get_num_threads();
        idthread = omp_get_thread_num();
        #endif // ifdef _OPENMP

        for (size_t isample = 0; isample < nsample; ++isample) {
            if (!fast_good_bin(isample)) continue;

            const int64_t ipixel = fast_pixels(isample);
            if (ipixel % nthreads != idthread) continue;
            const double * ipointer = &fast_templates(isample, 0);

            for (size_t jsample = 0; jsample < nsample; ++jsample) {
                const int64_t jpixel = fast_pixels(jsample);
                double filter_matrix = 0;
                const double * jpointer = &fast_templates(jsample, 0);
                const double * pcov = &fast_covariance(0, 0);

                // Evaluate F(F^T N^-1_F F)^-1 F^T N^-1_F at
                // (isample, jsample)

                if (fast_good_fit(jsample)) {
                    // dense templates
                    for (size_t itemplate = 0; itemplate < ndense; ++itemplate) {
                        const double * covpointer = pcov + itemplate * ntemplate;
                        double dtemp = 0;

                        // dense X dense
                        for (size_t jtemplate = 0; jtemplate < ndense; ++jtemplate) {
                            dtemp += jpointer[jtemplate] * covpointer[jtemplate];
                        }

                        // dense X sparse
                        for (const auto jtemplate : nonzeros[jsample]) {
                            dtemp += jpointer[jtemplate] * covpointer[jtemplate];
                        }
                        filter_matrix += dtemp * ipointer[itemplate];
                    }

                    // sparse templates
                    for (const auto itemplate : nonzeros[isample]) {
                        const double * covpointer = pcov + itemplate * ntemplate;
                        double dtemp = 0;

                        // sparse X dense
                        for (size_t jtemplate = 0; jtemplate < ndense; ++jtemplate) {
                            dtemp += jpointer[jtemplate] * covpointer[jtemplate];
                        }

                        // sparse X sparse
                        for (const auto jtemplate : nonzeros[jsample]) {
                            dtemp += jpointer[jtemplate] * covpointer[jtemplate];
                        }
                        filter_matrix += dtemp * ipointer[itemplate];
                    }
                }

                // Now get Z = I - filter_matrix

                if (isample == jsample) {
                    filter_matrix = 1 - filter_matrix;
                } else {
                    filter_matrix = -filter_matrix;
                }

                // `filter_matrix` holds the value of `Z` at
                // (isample, jsample).  Now accumulate that value to
                // approriate (ipixel, jpixel) in the observatio matrix.

                for (size_t inz = 0; inz < nnz; ++inz) {
                    double iweight = fast_weights(isample, inz) * filter_matrix;
                    for (size_t jnz = 0; jnz < nnz; ++jnz) {
                        fast_obs_matrix(ipixel + inz * npix, jpixel + jnz * npix)
                            += iweight * fast_weights(jsample, jnz);
                    }
                }
            }
        }
    }
}

void init_todmap_mapmaker(py::module & m) {
    m.doc() = "Compiled kernels to support TOAST mapmaker";

    m.def("project_signal_offsets", &project_signal_offsets);
    m.def("add_offsets_to_signal", &add_offsets_to_signal);
    m.def("apply_flags_to_pixels", &apply_flags_to_pixels);
    m.def("accumulate_observation_matrix", &accumulate_observation_matrix);
    m.def("expand_matrix", &expand_matrix);
    m.def("build_template_covariance", &build_template_covariance);
}
