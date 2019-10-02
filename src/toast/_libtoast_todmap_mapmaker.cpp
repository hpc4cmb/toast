
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>

/*
   last_obs = None
   last_det = None
   last_ref = None
   for itemplate, iobs, det, todslice, sqsigma in self.offset_templates:
   if iobs != last_obs or det != last_det:
      last_obs = iobs
      last_det = det
      last_ref = signal[iobs, det, :]
   offset_amplitudes[itemplate] += np.sum(last_ref[todslice])
 */
void add_offset_to_signal(py::array_t <double> ref, py::slice todslice,
                          py::array_t <double> amplitudes, int itemplate) {
    py::size_t istart, istop, istep, islicelength;
    if (!todslice.compute(ref.size(), &istart, &istop, &istep,
                          &islicelength)) throw py::error_already_set();
    auto fast_amplitudes = amplitudes.unchecked <1>();
    double offset = fast_amplitudes(itemplate);
    auto fast_ref = ref.mutable_unchecked <1>();
    for (ssize_t i = istart; i < istop; ++i) {
        fast_ref(i) += offset;
    }
}

void add_offsets_to_signal(py::array_t <double> ref, py::list todslices,
                           py::array_t <double> amplitudes,
                           py::array_t <long> itemplates) {
    auto fast_ref = ref.mutable_unchecked <1>();
    auto fast_amplitudes = amplitudes.unchecked <1>();
    auto fast_itemplates = itemplates.unchecked <1>();

    for (int i = 0; i < itemplates.size(); ++i)
    {
        py::slice todslice = py::slice(todslices[i]);
        py::size_t istart, istop, istep, islicelength;
        if (!todslice.compute(ref.size(), &istart, &istop, &istep,
                              &islicelength)) throw py::error_already_set();
        int itemplate = fast_itemplates(i);
        double offset = fast_amplitudes(itemplate);
        for (ssize_t i = istart; i < istop; ++i) {
            fast_ref(i) += offset;
        }
    }
}

void project_signal_offset(py::array_t <double> ref, py::slice todslice,
                           py::array_t <double> amplitudes, int itemplate) {
    py::size_t istart, istop, istep, islicelength;
    if (!todslice.compute(ref.size(), &istart, &istop, &istep,
                          &islicelength)) throw py::error_already_set();
    auto fast_ref = ref.unchecked <1>();
    double sum = 0;
    for (ssize_t i = istart; i < istop; ++i) {
        sum += fast_ref(i);
    }
    auto fast_amplitudes = amplitudes.mutable_unchecked <1>();
    fast_amplitudes(itemplate) += sum;
}

void project_signal_offsets(py::array_t <double> ref, py::list todslices,
                            py::array_t <double> amplitudes,
                            py::array_t <long> itemplates) {
    auto fast_ref = ref.unchecked <1>();
    auto fast_amplitudes = amplitudes.mutable_unchecked <1>();
    auto fast_itemplates = itemplates.unchecked <1>();

    for (int i = 0; i < itemplates.size(); ++i)
    {
        py::slice todslice = py::slice(todslices[i]);
        py::size_t istart, istop, istep, islicelength;
        if (!todslice.compute(ref.size(), &istart, &istop, &istep,
                              &islicelength)) throw py::error_already_set();
        double sum = 0;
        for (ssize_t i = istart; i < istop; ++i) {
            sum += fast_ref(i);
        }
        int itemplate = fast_itemplates(i);
        fast_amplitudes(itemplate) += sum;
    }
}

void init_todmap_mapmaker(py::module & m)
{
    m.doc() = "Compiled kernels to support TOAST mapmaker";

    m.def("project_signal_offset", &project_signal_offset);
    m.def("project_signal_offsets", &project_signal_offsets);
    m.def("add_offset_to_signal", &add_offset_to_signal);
    m.def("add_offsets_to_signal", &add_offsets_to_signal);
}
