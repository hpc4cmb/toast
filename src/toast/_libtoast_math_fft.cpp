
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_math_fft(py::module & m) {
    py::enum_ <toast::fft_plan_type> (m, "FFTPlanType", py::arithmetic(),
                                      "FFT Plan Type")
    .value("fast", toast::fft_plan_type::fast)
    .value("best", toast::fft_plan_type::best);

    py::enum_ <toast::fft_direction> (m, "FFTDirection", py::arithmetic(),
                                      "FFT Direction")
    .value("forward", toast::fft_direction::forward)
    .value("backward", toast::fft_direction::backward);


    py::class_ <toast::FFTPlanReal1D, toast::FFTPlanReal1D::pshr> (
        m, "FFTPlanReal1D", py::arg("length"), py::arg("n"), py::arg("type"),
        py::arg("dir"), py::arg(
            "scale"), R"(
        FFT plan for one dimensional real transforms.

        The plan is valid for a particular length and for a batch of a
        fixed number of data buffers.

        Args:
            length (int):  The length of the FFTs.
            n (int):  The number of data buffers.
            type (FFTPlanType):  The type (fast, best) of the transform.
            dir (FFTDirection):  The direction of the transform.
            scale (float):  The scale factor to apply to the result.

        )")
    .def(
        py::init(
            [](int64_t length, int64_t n, toast::fft_plan_type type,
               toast::fft_direction dir, double scale) {
                return toast::FFTPlanReal1D::pshr(
                    toast::FFTPlanReal1D::create(
                        length, n, type, dir, scale));
            })
        )
    .def("exec", &toast::FFTPlanReal1D::exec,
         R"(
        Execute the plan on the current state of the data buffers.
        )")
    .def("length", &toast::FFTPlanReal1D::length,
         R"(
        Return the length of the FFTs.

        Returns:
            (int): The FFT length for this plan.
        )")
    .def("count", &toast::FFTPlanReal1D::count,
         R"(
        Return the number of FFTs in this plan.

        Returns:
            (int): The number of FFTs.
        )")
    .def("tdata", [](toast::FFTPlanReal1D & self) {
             auto raw = self.tdata();
             int64_t leng = self.length();
             py::list ret;
             for (auto & raw_p : raw) {
                 ret.append(py::array_t <double> ({leng}, {8},
                                                  raw_p));
             }
             return ret;
         }, R"(
        Return references to the time domain buffers.

        Returns:
            (list): List of arrays which are a view of the internal buffers.
        )")
    .def("fdata", [](toast::FFTPlanReal1D & self) {
             auto raw = self.fdata();
             int64_t leng = self.length();
             py::list ret;
             for (auto & raw_p : raw) {
                 ret.append(py::array_t <double> ({leng}, {8},
                                                  raw_p));
             }
             return ret;
         }, R"(
        Return references to the frequency domain buffers.

        Returns:
            (list): List of arrays which are a view of the internal buffers.
        )");


    py::class_ <toast::FFTPlanReal1DStore,
                std::unique_ptr <toast::FFTPlanReal1DStore, py::nodelete> > (
        m, "FFTPlanReal1DStore",
        R"(
        Global cache of FFT plans.

        This singleton class allows re-use of FFT plans within a single
        process.

        )")
    .def("get", []() {
             return std::unique_ptr <toast::FFTPlanReal1DStore, py::nodelete>
             (
                 &toast::FFTPlanReal1DStore::get());
         }, R"(
            Get a handle to the global FFT plan cache.
        )")
    .def("clear", &toast::FFTPlanReal1DStore::clear,
         R"(
        Clear all plans in the store.
        )")
    .def("cache", &toast::FFTPlanReal1DStore::cache, py::args("length"),
         py::arg(
             "n"),  R"(
            Add a plan to the store.

            This adds a forward and reverse "fast" plan to the global cache.

            Args:
                length (int):  The length of the FFTs.
                n (int):  The number of data buffers.

            Returns:
                None

        )")
    .def("forward", &toast::FFTPlanReal1DStore::forward, py::args("length"),
         py::arg(
             "n"),  R"(
            Retrieve a plan from the store.

            A forward plan of the specified length and number of buffers is
            returned.

            Args:
                length (int):  The length of the FFTs.
                n (int):  The number of data buffers.

            Returns:
                (FFTPlanReal1D):  The plan.

        )")
    .def("backward", &toast::FFTPlanReal1DStore::backward, py::args("length"),
         py::arg(
             "n"),  R"(
            Retrieve a plan from the store.

            A backward plan of the specified length and number of buffers is
            returned.

            Args:
                length (int):  The length of the FFTs.
                n (int):  The number of data buffers.

            Returns:
                (FFTPlanReal1D):  The plan.

        )");

    return;
}
