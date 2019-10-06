
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


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
        if (common_flag & common_flag_mask || detector_flag & detector_flag_mask) {
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

void init_todmap_mapmaker(py::module & m)
{
    m.doc() = "Compiled kernels to support TOAST mapmaker";

    m.def("project_signal_offsets", &project_signal_offsets);
    m.def("add_offsets_to_signal", &add_offsets_to_signal);
    m.def("apply_flags_to_pixels", &apply_flags_to_pixels);
}
