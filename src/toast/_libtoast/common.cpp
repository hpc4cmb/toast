
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <common.hpp>

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

std::string get_format(std::string const & input) {
    // Machine endianness
    int32_t test = 1;
    bool little_endian = (*(int8_t *)&test == 1);

    std::string format = input;

    // Remove leading caret, if present.
    if (format.substr(0, 1) == "^") {
        format = input.substr(1, input.length() - 1);
    }

    std::string fmt = format;
    if (format.length() > 1) {
        // The format string includes endianness information or
        // is a compound type.
        std::string endianness = format.substr(0, 1);
        if (
            ((endianness == ">") &&  !little_endian)  ||
            ((endianness == "<") &&  little_endian)  ||
            (endianness == "=")
        ) {
            fmt = format.substr(1, format.length() - 1);
        } else if (
            ((endianness == ">") &&  little_endian)  ||
            ((endianness == "<") &&  !little_endian)
        ) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Object has different endianness than system- cannot use";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    return fmt;
}
