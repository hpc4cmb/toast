
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#ifdef HAVE_OPENACC
# include <openacc.h>
#endif // ifdef HAVE_OPENACC


// FIXME:  docstrings need to be updated if we keep these accelerator versions of the
// code.

void init_template_offset(py::module & m) {
    m.def(
        "template_offset_add_to_signal", [](
            bool use_acc,
            int64_t step_length,
            int64_t amp_offset,
            int64_t n_amp,
            int64_t samp_offset,
            int64_t n_det_samp,
            int64_t det_indx,
            py::buffer amplitudes,
            py::buffer det_data
        ) {
            auto info_amps = amplitudes.request();
            double * raw_amps = reinterpret_cast <double *> (info_amps.ptr);

            auto info_detdata = det_data.request();
            double * raw_data = reinterpret_cast <double *> (info_detdata.ptr);

            size_t n_det = info_detdata.shape[0];
            size_t n_all_samp = info_detdata.shape[1];

            if (use_acc) {
                #pragma \
                acc data copyin(step_length, amp_offset, n_amp, samp_offset, n_det_samp, det_indx, n_det, n_all_samp) present(raw_amps[amp_offset:n_amp], raw_data[samp_offset:n_det_samp])
                {
                    if (fake_openacc()) {
                        // Set all "present" data to point at the fake device pointers
                        auto & fake = FakeMemPool::get();
                        raw_data = (double *)fake.device_ptr(raw_data);
                        raw_amps = (double *)fake.device_ptr(raw_amps);
                    }
                    #pragma acc parallel
                    {
                        // All but the last amplitude have the same number of samples.
                        #pragma acc loop independent
                        for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                            //std::cout << "DBG add_to_signal " << amp_offset + iamp << ": " << raw_amps[amp_offset + iamp] << std::endl;
                            int64_t doff = det_indx * n_all_samp + samp_offset + iamp * step_length;
                            int64_t nd;
                            if (iamp == n_amp - 1) {
                                nd = n_det_samp - (samp_offset + (n_amp - 1) * step_length);
                            } else {
                                nd = step_length;
                            }
                            for (int64_t j = 0; j < nd; ++j) {
                                //std::cout << "DBG add_to_signal   " << doff + j << ":   " << raw_data[doff + j];
                                raw_data[doff + j] += raw_amps[amp_offset + iamp];
                                //std::cout << " -> " << raw_data[doff + j] << std::endl;
                            }
                        }
                    }
                }
            } else {
                for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                    //std::cout << "DBG add_to_signal " << amp_offset + iamp << ": " << raw_amps[amp_offset + iamp] << std::endl;
                    int64_t doff = det_indx * n_all_samp + samp_offset + iamp * step_length;
                    int64_t nd;
                    if (iamp == n_amp - 1) {
                        nd = n_det_samp - (samp_offset + (n_amp - 1) * step_length);
                    } else {
                        nd = step_length;
                    }
                    for (int64_t j = 0; j < nd; ++j) {
                        //std::cout << "DBG add_to_signal   " << doff + j << ":   " << raw_data[doff + j];
                        raw_data[doff + j] += raw_amps[amp_offset + iamp];
                        //std::cout << " -> " << raw_data[doff + j] << std::endl;
                    }
                }
            }
            return;
        });

    m.def(
        "template_offset_project_signal", [](
            bool use_acc,
            int64_t step_length,
            int64_t amp_offset,
            int64_t n_amp,
            int64_t samp_offset,
            int64_t n_det_samp,
            int64_t det_indx,
            py::buffer det_data,
            py::buffer amplitudes
        ) {
            auto info_amps = amplitudes.request();
            double * raw_amps = reinterpret_cast <double *> (info_amps.ptr);

            auto info_detdata = det_data.request();
            double * raw_data = reinterpret_cast <double *> (info_detdata.ptr);

            size_t n_det = info_detdata.shape[0];
            size_t n_all_samp = info_detdata.shape[1];

            if (use_acc) {
                #pragma \
                acc data copyin(step_length, amp_offset, n_amp, samp_offset, n_det_samp, det_indx, n_det, n_all_samp) present(raw_amps[amp_offset:n_amp], raw_data[samp_offset:n_det_samp])
                {
                    if (fake_openacc()) {
                        // Set all "present" data to point at the fake device pointers
                        auto & fake = FakeMemPool::get();
                        raw_data = (double *)fake.device_ptr(raw_data);
                        raw_amps = (double *)fake.device_ptr(raw_amps);
                    }
                    #pragma acc parallel
                    {
                        // All but the last amplitude have the same number of samples.
                        #pragma acc loop independent
                        for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                            //std::cout << "DBG project_signal " << amp_offset + iamp << ": start = " << raw_amps[amp_offset + iamp] << std::endl;
                            int64_t doff = det_indx * n_all_samp + samp_offset + iamp * step_length;
                            int64_t nd;
                            if (iamp == n_amp - 1) {
                                nd = n_det_samp - (samp_offset + (n_amp - 1) * step_length);
                            } else {
                                nd = step_length;
                            }
                            for (int64_t j = 0; j < nd; ++j) {
                                //std::cout << "DBG project_signal   " << doff + j << ": += " << raw_data[doff + j];
                                raw_amps[amp_offset + iamp] += raw_data[doff + j];
                                //std::cout << " -> " << raw_amps[amp_offset + iamp] << std::endl;
                            }
                        }
                    }
                }
            } else {
                for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                    //std::cout << "DBG project_signal " << amp_offset + iamp << ": start = " << raw_amps[amp_offset + iamp] << std::endl;
                    int64_t doff = det_indx * n_all_samp + samp_offset + iamp * step_length;
                    int64_t nd;
                    if (iamp == n_amp - 1) {
                        nd = n_det_samp - (samp_offset + (n_amp - 1) * step_length);
                    } else {
                        nd = step_length;
                    }
                    for (int64_t j = 0; j < nd; ++j) {
                        //std::cout << "DBG project_signal   " << doff + j << ": += " << raw_data[doff + j];
                        raw_amps[amp_offset + iamp] += raw_data[doff + j];
                        //std::cout << " -> " << raw_amps[amp_offset + iamp] << std::endl;
                    }
                }
            }
            return;

        });

    m.def(
        "template_offset_apply_diag_precond", [](
            bool use_acc,
            py::buffer offset_var,
            py::buffer amplitudes_in,
            py::buffer amplitudes_out
        ) {
            auto info_var = offset_var.request();
            // std::cout << "DBG offsetvar = " << info_var.shape[0] << " at " << info_var.ptr << std::endl;
            double * raw_var = reinterpret_cast <double *> (info_var.ptr);

            auto info_amps_in = amplitudes_in.request();
            double * raw_amps_in = reinterpret_cast <double *> (info_amps_in.ptr);

            auto info_amps_out = amplitudes_out.request();
            double * raw_amps_out = reinterpret_cast <double *> (info_amps_out.ptr);

            size_t n_amp = info_amps_in.shape[0];

            // std::cout << "DBG offset var = ";
            // for (size_t i = 0; i < n_amp; i++) {
            //     std::cout << raw_var[i] << ", ";
            // }
            // std::cout << std::endl;

            if (use_acc) {
                #pragma \
                acc data copyin(n_amp, raw_var[0:n_amp]) present(raw_amps_in[0:n_amp], raw_amps_out[0:n_amp])
                {
                    if (fake_openacc()) {
                        // Set all "present" data to point at the fake device pointers
                        auto & fake = FakeMemPool::get();
                        raw_amps_in = (double *)fake.device_ptr(raw_amps_in);
                        raw_amps_out = (double *)fake.device_ptr(raw_amps_out);
                    }
                    #pragma acc parallel
                    {
                        #pragma acc loop independent
                        for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                            //std::cout << "DBG apply_precond " << iamp << ": " << raw_amps_in[iamp] << " * " << raw_var[iamp] << " = ";
                            raw_amps_out[iamp] = raw_amps_in[iamp];
                            raw_amps_out[iamp] *= raw_var[iamp];
                            //std::cout << raw_amps_out[iamp] << std::endl;
                        }
                    }
                }
            } else {
                for (int64_t iamp = 0; iamp < n_amp; iamp++) {
                    //std::cout << "DBG apply_precond " << iamp << ": " << raw_amps_in[iamp] << " * " << raw_var[iamp] << " = ";
                    raw_amps_out[iamp] = raw_amps_in[iamp];
                    raw_amps_out[iamp] *= raw_var[iamp];
                    //std::cout << raw_amps_out[iamp] << std::endl;
                }
            }
            return;

        });

    // m.def(
    //     "template_offset_add_to_signal", [](int64_t step_length, py::buffer amplitudes,
    //                                         py::buffer data) {
    //         pybuffer_check_1D <double> (amplitudes);
    //         pybuffer_check_1D <double> (data);
    //         py::buffer_info info_amplitudes = amplitudes.request();
    //         py::buffer_info info_data = data.request();
    //         int64_t n_amp = info_amplitudes.size;
    //         int64_t n_data = info_data.size;
    //         double * raw_amplitudes = reinterpret_cast <double *> (info_amplitudes.ptr);
    //         double * raw_data = reinterpret_cast <double *> (info_data.ptr);
    //         toast::template_offset_add_to_signal(step_length, n_amp, raw_amplitudes,
    //                                              n_data, raw_data);
    //         return;
    //     }, py::arg("step_length"), py::arg("amplitudes"), py::arg(
    //         "data"), R"(
    //     Accumulate offset amplitudes to timestream data.

    //     Each amplitude value is accumulated to `step_length` number of samples.  The
    //     final offset will be at least this many samples, but may be more if the step
    //     size does not evenly divide into the number of samples.

    //     Args:
    //         step_length (int64):  The minimum number of samples for each offset.
    //         amplitudes (array):  The float64 amplitude values.
    //         data (array):  The float64 timestream values to accumulate.

    //     Returns:
    //         None.

    // )");

    // m.def(
    //     "template_offset_project_signal", [](int64_t step_length, py::buffer data,
    //                                          py::buffer amplitudes) {
    //         pybuffer_check_1D <double> (amplitudes);
    //         pybuffer_check_1D <double> (data);
    //         py::buffer_info info_amplitudes = amplitudes.request();
    //         py::buffer_info info_data = data.request();
    //         int64_t n_amp = info_amplitudes.size;
    //         int64_t n_data = info_data.size;
    //         double * raw_amplitudes = reinterpret_cast <double *> (info_amplitudes.ptr);
    //         double * raw_data = reinterpret_cast <double *> (info_data.ptr);
    //         toast::template_offset_project_signal(step_length, n_data, raw_data,
    //                                               n_amp, raw_amplitudes);
    //         return;
    //     }, py::arg("step_length"), py::arg("data"), py::arg(
    //         "amplitudes"), R"(
    //     Accumulate timestream data into offset amplitudes.

    //     Chunks of `step_length` number of samples are accumulated into the offset
    //     amplitudes.  If step_length does not evenly divide into the total number of
    //     samples, the final amplitude will be extended to include the remainder.

    //     Args:
    //         step_length (int64):  The minimum number of samples for each offset.
    //         data (array):  The float64 timestream values.
    //         amplitudes (array):  The float64 amplitude values.

    //     Returns:
    //         None.

    // )");

    return;
}
