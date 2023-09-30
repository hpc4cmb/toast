
// Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
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
    //#pragma omp parallel for schedule(static, 64)
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

void add_matrix(
    py::array_t <double, py::array::c_style | py::array::forcecast> data1,
    py::array_t <int64_t, py::array::c_style | py::array::forcecast> indices1,
    py::array_t <int64_t, py::array::c_style | py::array::forcecast> indptr1,
    py::array_t <double, py::array::c_style | py::array::forcecast> data2,
    py::array_t <int64_t, py::array::c_style | py::array::forcecast> indices2,
    py::array_t <int64_t, py::array::c_style | py::array::forcecast> indptr2,
    py::array_t <double, py::array::c_style | py::array::forcecast> data3,
    py::array_t <int64_t, py::array::c_style | py::array::forcecast> indices3,
    py::array_t <int64_t, py::array::c_style | py::array::forcecast> indptr3
) {
    /* Compiled kernel to add two CSR matrices
     */
    auto fast_data1 = data1.unchecked <1>();
    auto fast_indices1 = indices1.unchecked <1>();
    auto fast_indptr1 = indptr1.unchecked <1>();
    auto fast_data2 = data2.unchecked <1>();
    auto fast_indices2 = indices2.unchecked <1>();
    auto fast_indptr2 = indptr2.unchecked <1>();
    auto fast_data3 = data3.mutable_unchecked <1>();
    auto fast_indices3 = indices3.mutable_unchecked <1>();
    auto fast_indptr3 = indptr3.mutable_unchecked <1>();

    const size_t n1 = fast_data1.shape(0);
    const size_t n2 = fast_data2.shape(0);
    const size_t n3 = fast_data3.shape(0);
    if ((indptr1.shape(0) != indptr2.shape(0)) ||
        (indptr1.shape(0) != indptr3.shape(0))) {
        throw std::length_error("Input and output sizes do not agree");
    }
    const size_t nrow = indptr1.shape(0) - 1;

    // Collect each row's data into vectors
    std::vector <std::vector <double> > row_data(nrow);
    std::vector <std::vector <int64_t> > row_indices(nrow);

    //#pragma omp parallel for schedule(static, 4)
    for (size_t row = 0; row < nrow; ++row) {
        const size_t start1 = fast_indptr1[row];
        const size_t stop1 = fast_indptr1[row + 1];
        const size_t start2 = fast_indptr2[row];
        const size_t stop2 = fast_indptr2[row + 1];
        if ((start1 == stop1) && (start2 == stop2)) continue;
        size_t n;
        if (start2 == stop2) {
            // Only first matrix has entries
            n = stop1 - start1;
            row_indices[row].resize(n);
            memcpy(row_indices[row].data(),
                   &fast_indices1[start1],
                   n * sizeof(int64_t));
            row_data[row].resize(n);
            memcpy(row_data[row].data(),
                   &fast_data1[start1],
                   n * sizeof(double));
        } else if (start1 == stop1) {
            // Only second matrix has entries
            n = stop2 - start2;
            row_indices[row].resize(n);
            memcpy(row_indices[row].data(),
                   &fast_indices2[start2],
                   n * sizeof(int64_t));
            row_data[row].resize(n);
            memcpy(row_data[row].data(),
                   &fast_data2[start2],
                   n * sizeof(double));
        } else {
            // Both matrices have entries
            std::map <size_t, double> datamap;
            for (size_t ind1 = start1; ind1 < stop1; ++ind1) {
                const size_t col1 = fast_indices1[ind1];
                const double value1 = fast_data1[ind1];
                datamap[col1] = value1;
            }
            for (size_t ind2 = start2; ind2 < stop2; ++ind2) {
                const size_t col2 = fast_indices2[ind2];
                const double value2 = fast_data2[ind2];
                const auto & match = datamap.find(col2);
                if (match == datamap.end()) {
                    datamap[col2] = value2;
                } else {
                    match->second += value2;
                }
            }
            n = datamap.size();
            row_indices[row].resize(n);
            row_data[row].resize(n);
            size_t offset = 0;
            for (const auto & item : datamap) {
                row_indices[row][offset] = item.first;
                row_data[row][offset] = item.second;
                ++offset;
            }
        }

        // Store the number of nonzeros in the index pointer array.
        // We take a cumulative sum later
        fast_indptr3[row + 1] = n;
    }

    // Cumulative sum of the nonzeros is the offset
    for (size_t row = 2; row < nrow + 1; ++row) {
        fast_indptr3[row] += fast_indptr3[row - 1];
    }

    // Pack the row vectors into the output arrays
    //#pragma omp parallel for schedule(static, 4)
    for (size_t row = 0; row < nrow; ++row) {
        size_t n = row_data[row].size();
        if (n == 0) continue;
        size_t offset = fast_indptr3[row];
        memcpy(&fast_indices3[offset],
               row_indices[row].data(),
               n * sizeof(int64_t));
        memcpy(&fast_data3[offset],
               row_data[row].data(),
               n * sizeof(double));
    }
}

void expand_matrix(py::array_t <double> compressed_matrix,
                   py::array_t <int64_t> local_to_global,
                   int64_t npix,
                   int64_t nnz,
                   py::array_t <double> values,
                   py::array_t <int64_t> indices,
                   py::array_t <int64_t> indptr
) {
    auto fast_matrix = compressed_matrix.unchecked <2>();
    auto fast_local_to_global = local_to_global.unchecked <1>();
    auto fast_values = values.mutable_unchecked <1>();
    auto fast_indices = indices.mutable_unchecked <1>();
    auto fast_indptr = indptr.mutable_unchecked <1>();

    size_t nlocal = fast_local_to_global.shape(0);
    size_t nlocal_tot = fast_matrix.shape(0);
    std::vector <size_t> col_indices;

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
            size_t local_row = ilocal + inz * nlocal;
            size_t iglobal = fast_local_to_global[ilocal];

            // Skip empty rows before the current pixel index
            while (global_pixel < iglobal) {
                fast_indptr[global_row + 1] = offset;
                global_row++;
                global_pixel++;
            }

            // Copy a row of the dense matrix into the sparse one
            for (size_t local_col = 0; local_col < col_indices.size(); ++local_col) {
                size_t ind = col_indices[local_col];
                double value = fast_matrix(local_row, local_col);
                if (value != 0) {
                    fast_values[offset] = value;
                    fast_indices[offset++] = ind;
                }
            }
            fast_indptr[global_row + 1] = offset;
            global_row++;
            global_pixel++;
        }

        // Skip empty rows trailing the last pixel in the dense matrix
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

    //#pragma omp parallel for schedule(static, 1)
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
    //#pragma omp parallel for schedule(static, 1)
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
                // approriate (ipixel, jpixel) in the observation matrix.

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
    m.def("add_matrix", &add_matrix);
    m.def("build_template_covariance", &build_template_covariance);
}
