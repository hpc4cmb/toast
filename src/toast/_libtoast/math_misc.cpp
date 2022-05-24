
// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>
#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


double integrate_simpson(py::array_t <double> x, py::array_t <double> f) {
    auto fast_x = x.unchecked <1>();
    auto fast_f = f.unchecked <1>();

    size_t n = x.size();
    double result = 0;

    #pragma omp parallel for reduction(+: result)
    for (size_t i = 0; i < (n - 1) / 2; ++i) {
        size_t ii = 2 * i;
        double h1 = fast_x(ii + 1) - fast_x(ii);
        double h2 = fast_x(ii + 2) - fast_x(ii + 1);
        double f1 = fast_f(ii);
        double f2 = fast_f(ii + 1);
        double f3 = fast_f(ii + 2);

        result += (h1 + h2) / 6 * (
            (2 - h2 / h1) * f1 +
            pow(h1 + h2, 2) / (h1 * h2) * f2 +
            (2 - h1 / h2) * f3
        );
    }

    if (n % 2 == 0) {
        double h1 = fast_x(n - 1) - fast_x(n - 2);
        double h2 = fast_x(n - 2) - fast_x(n - 3);
        double f1 = fast_f(n - 1);
        double f2 = fast_f(n - 2);
        double f3 = fast_f(n - 3);
        result += (
            (2 * pow(h1, 2) + 3 * h1 * h2) / (6 * (h2 + h1)) * f1 +
            (pow(h1, 2) + 3 * h1 * h2) / (6 * h2) * f2 -
            pow(h1, 3) / (6 * h2 * (h2 + h1)) * f2
        );
    }

    return result;
}

void init_math_misc(py::module & m) {
    m.doc() = "Miscellaneous match functions";

    m.def("integrate_simpson", &integrate_simpson);
}
