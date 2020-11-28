
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>
#ifdef _OPENMP
#include <omp.h>
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
    auto fast_matrix = compressed_matrix.unchecked<2>();
    auto fast_local_to_global = local_to_global.unchecked<1>();
    auto fast_indices = indices.mutable_unchecked<1>();
    auto fast_indptr = indptr.mutable_unchecked<1>();

    size_t nlocal = fast_local_to_global.shape(0);
    size_t nlocal_tot = fast_matrix.shape(0);
    std::vector<int64_t> col_indices;

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


void build_template_covariance(py::array_t <double, py::array::c_style | py::array::forcecast> templates,
                               py::array_t <double, py::array::c_style | py::array::forcecast> good,
                               py::array_t <double, py::array::c_style | py::array::forcecast> template_covariance) {
    auto fast_templates = templates.unchecked<2>();
    auto fast_good = good.unchecked<1>();
    auto fast_covariance = template_covariance.mutable_unchecked<2>();

    size_t ntemplate = fast_templates.shape(0);
    size_t nsample = fast_templates.shape(1);

    std::vector<size_t> starts(ntemplate);
    std::vector<size_t> stops(ntemplate);
#pragma omp parallel for schedule(static, 1)
    for (size_t itemplate = 0; itemplate < ntemplate; ++itemplate) {
        size_t istart;
        for (istart = 0; istart < nsample; ++istart) {
            if (fast_templates(itemplate, istart) != 0 && fast_good[istart] != 0) break;
        }
        starts[itemplate] = istart;

        size_t istop;
        for (istop = nsample - 1; istop >= 0; --istop) {
            if (fast_templates(itemplate, istop) != 0 && fast_good[istop] != 0) break;
        }
        stops[itemplate] = istop + 1;
    }

#pragma omp parallel for schedule(static, 1)
    for (size_t itemplate = 0; itemplate < ntemplate; ++itemplate) {
        for (size_t jtemplate = itemplate; jtemplate < ntemplate; ++jtemplate) {
            double val = 0;
            size_t istart = std::max(starts[itemplate], starts[jtemplate]);
            size_t istop = std::min(stops[itemplate], stops[jtemplate]);
            for (size_t isample = istart; isample < istop; ++isample) {
                val += fast_templates(itemplate, isample)
                    * fast_templates(jtemplate, isample)
                    * fast_good(isample);
            }
            fast_covariance(itemplate, jtemplate) = val;
            if (itemplate != jtemplate) {
                fast_covariance(jtemplate, itemplate) = val;
            }
        }
    }
}



void accumulate_observation_matrix(py::array_t <double, py::array::c_style | py::array::forcecast> c_obs_matrix,
                                   py::array_t <int64_t, py::array::c_style | py::array::forcecast> c_pixels,
                                   py::array_t <double, py::array::c_style | py::array::forcecast> weights,
                                   py::array_t <double, py::array::c_style | py::array::forcecast> templates,
                                   py::array_t <double, py::array::c_style | py::array::forcecast> template_covariance) {
    auto fast_obs_matrix = c_obs_matrix.mutable_unchecked<2>();
    auto fast_pixels = c_pixels.unchecked<1>();
    auto fast_weights = weights.unchecked<2>();
    auto fast_templates = templates.unchecked<2>();
    auto fast_covariance = template_covariance.unchecked<2>();

    size_t nsample = fast_pixels.shape(0);
    size_t nnz = fast_weights.shape(1);
    size_t ntemplate = fast_templates.shape(1);
    size_t npixtot = fast_obs_matrix.shape(0);
    size_t npix = npixtot / nnz;

    // Build lists of non-zeros for each row of the template matrix
    std::vector<std::vector<size_t>> nonzeros(nsample);
# pragma omp parallel for schedule(static, 1)
    for (size_t isample = 0; isample < nsample; ++isample) {
        for (size_t itemplate = 0; itemplate < ntemplate; ++itemplate) {
            if (fast_templates(isample, itemplate) != 0) {
                nonzeros[isample].push_back(itemplate);
            }
        }
    }

# pragma omp parallel
    {
        int nthreads = 1;
        int idthread = 0;
#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        idthread = omp_get_thread_num();
#endif // ifdef _OPENMP

        for (size_t isample = 0; isample < nsample; ++isample) {
            size_t ipixel = fast_pixels(isample);
            if (ipixel % nthreads != idthread) continue;
            for (size_t jsample = 0; jsample < nsample; ++jsample) {
                size_t jpixel = fast_pixels(jsample);
                double filter_matrix = 0;
                for (auto itemplate : nonzeros[isample]) {
                    for (auto jtemplate : nonzeros[jsample]) {
                        filter_matrix
                            += fast_templates(isample, itemplate)
                            * fast_templates(jsample, jtemplate)
                            * fast_covariance(itemplate, jtemplate);
                    }
                }
                if (isample == jsample) {
                    filter_matrix = 1 - filter_matrix;

                } else {
                    filter_matrix = -filter_matrix;
                }
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


void init_todmap_mapmaker(py::module & m)
{
    m.doc() = "Compiled kernels to support TOAST mapmaker";

    m.def("project_signal_offsets", &project_signal_offsets);
    m.def("add_offsets_to_signal", &add_offsets_to_signal);
    m.def("apply_flags_to_pixels", &apply_flags_to_pixels);
    m.def("accumulate_observation_matrix", &accumulate_observation_matrix);
    m.def("expand_matrix", &expand_matrix);
    m.def("build_template_covariance", &build_template_covariance);
}
