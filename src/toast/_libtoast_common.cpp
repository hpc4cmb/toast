
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast_common.hpp>

// FIXME: we could use configure checks to detect whether we are on a 64bit
// system and whether "l" and "L" are equivalent to "q" and "Q".

template <>
std::vector <char> align_format <int8_t> () {
    return std::vector <char> ({'b'});
}

template <>
std::vector <char> align_format <int16_t> () {
    return std::vector <char> ({'h'});
}

template <>
std::vector <char> align_format <int32_t> () {
    return std::vector <char> ({'i'});
}

template <>
std::vector <char> align_format <int64_t> () {
    return std::vector <char> ({'q', 'l'});
}

template <>
std::vector <char> align_format <uint8_t> () {
    return std::vector <char> ({'B'});
}

template <>
std::vector <char> align_format <uint16_t> () {
    return std::vector <char> ({'H'});
}

template <>
std::vector <char> align_format <uint32_t> () {
    return std::vector <char> ({'I'});
}

template <>
std::vector <char> align_format <uint64_t> () {
    return std::vector <char> ({'Q', 'L'});
}

template <>
std::vector <char> align_format <float> () {
    return std::vector <char> ({'f'});
}

template <>
std::vector <char> align_format <double> () {
    return std::vector <char> ({'d'});
}
