
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TEMPLATE_OFFSET_HPP
#define TOAST_TEMPLATE_OFFSET_HPP

#include <vector>


namespace toast {
void template_offset_add_to_signal(int64_t step_length, int64_t n_amp,
                                   double * amplitudes, int64_t n_data, double * data);

void template_offset_project_signal(int64_t step_length, int64_t n_data, double * data,
                                    int64_t n_amp, double * amplitudes);
}

#endif // ifndef TOAST_TEMPLATE_OFFSET_HPP
