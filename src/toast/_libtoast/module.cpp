
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

using size_container = py::detail::any_container <ssize_t>;


PYBIND11_MODULE(_libtoast, m) {
    m.doc() = R"(
    Interface to C++ TOAST library.

    )";

    // Register aligned array types
    register_aligned <toast::AlignedI8> (m, "AlignedI8");
    register_aligned <toast::AlignedU8> (m, "AlignedU8");
    register_aligned <toast::AlignedI16> (m, "AlignedI16");
    register_aligned <toast::AlignedU16> (m, "AlignedU16");
    register_aligned <toast::AlignedI32> (m, "AlignedI32");
    register_aligned <toast::AlignedU32> (m, "AlignedU32");
    register_aligned <toast::AlignedI64> (m, "AlignedI64");
    register_aligned <toast::AlignedU64> (m, "AlignedU64");
    register_aligned <toast::AlignedF32> (m, "AlignedF32");
    register_aligned <toast::AlignedF64> (m, "AlignedF64");

    init_sys(m);
    init_intervals(m);
    init_math_misc(m);
    init_math_sf(m);
    init_math_rng(m);
    init_math_qarray(m);
    init_math_healpix(m);
    init_math_fft(m);
    init_fod_psd(m);
    init_tod_filter(m);
    init_tod_pointing(m);
    init_tod_simnoise(m);
    init_todmap_scanning(m);
    init_map_cov(m);
    init_pixels(m);
    init_todmap_mapmaker(m);
    init_atm(m);
    init_template_offset(m);
    init_accelerator(m);
    init_ops_pointing_detector(m);
    init_ops_stokes_weights(m);
    init_ops_pixels_healpix(m);
    init_ops_mapmaker_utils(m);

    // Internal unit test runner
    m.def(
        "libtoast_tests", [](py::list argv) {
            size_t narg = argv.size();
            std::vector <std::string> argbuffer;
            for (auto const & a : argv) {
                argbuffer.push_back(py::cast <std::string> (a));
            }
            char ** carg = (char **)std::malloc(narg * sizeof(char *));
            for (size_t i = 0; i < narg; ++i) {
                carg[i] = &(argbuffer[i][0]);
            }
            toast::test_runner(narg, carg);
            free(carg);
            return;
        }, py::arg(
            "argv"), R"(
        Run serial compiled tests from the internal libtoast.

        Args:
            argv (list):  The sys.argv or compatible list.

        Returns:
            None

    )");
}
