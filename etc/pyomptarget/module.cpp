
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <map>
#include <limits>

#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include <pybind11/stl_bind.h>

namespace py = pybind11;

using size_container = py::detail::any_container <ssize_t>;


// Device code

#pragma omp declare target

struct Interval {
    double start;
    double stop;
    int64_t first;
    int64_t last;
};

void qa_mult(double const * p, double const * q, double * r) {
    r[0] =  p[0] * q[3] + p[1] * q[2] -
           p[2] * q[1] + p[3] * q[0];
    r[1] = -p[0] * q[2] + p[1] * q[3] +
           p[2] * q[0] + p[3] * q[1];
    r[2] =  p[0] * q[1] - p[1] * q[0] +
           p[2] * q[3] + p[3] * q[2];
    r[3] = -p[0] * q[0] - p[1] * q[1] -
           p[2] * q[2] + p[3] * q[3];
    return;
}

void pointing_detector_inner(
    int32_t const * q_index,
    uint8_t const * flags,
    double const * boresight,
    double const * fp,
    double * quats,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet
) {
    int32_t qidx = q_index[idet];
    double temp_bore[4];
    uint8_t check = flags[isamp] & 255;
    if (check == 0) {
        temp_bore[0] = boresight[4 * isamp];
        temp_bore[1] = boresight[4 * isamp + 1];
        temp_bore[2] = boresight[4 * isamp + 2];
        temp_bore[3] = boresight[4 * isamp + 3];
    } else {
        temp_bore[0] = 0.0;
        temp_bore[1] = 0.0;
        temp_bore[2] = 0.0;
        temp_bore[3] = 1.0;
    }
    qa_mult(
        temp_bore,
        &(fp[4 * idet]),
        &(quats[(qidx * 4 * n_samp) + 4 * isamp])
    );
    return;
}

#pragma omp end declare target

template <typename T>
void host_to_device(
    std::unordered_map <void *, void *> & mem,
    int64_t n_elem,
    T * data,
    std::string const & name
) {
    int target = omp_get_num_devices() - 1;
    int host = omp_get_initial_device();

    void * vdata = (void*)(&data[0]);

    size_t n_bytes = n_elem * sizeof(T);
    std::cerr << name << ": Allocating " << n_bytes << " device bytes (" << n_elem << " x " << sizeof(T) << ")" << std::endl;
    void * buffer = omp_target_alloc(n_bytes, target);

    std::cerr << name << ": Associating host pointer " << vdata << std::endl;
    int failed = omp_target_associate_ptr(vdata, buffer, n_bytes, 0, target);
    if (failed != 0) {
        std::cerr << "Failed to associate dev pointer" << std::endl;
    }

    failed = omp_target_memcpy(buffer, vdata, n_bytes, 0, 0, target, host);
    if (failed != 0) {
        std::cerr << "Failed to copy " << name << " host to device" << std::endl;
    }
    std::cerr << name << ": map host " << vdata << " --> " << buffer << std::endl;
    mem[vdata] = buffer;
    return;
}

template <typename T>
void device_to_host(
    std::unordered_map <void *, void *> & mem,
    int64_t n_elem,
    T * data,
    std::string const & name
) {
    int target = omp_get_num_devices() - 1;
    int host = omp_get_initial_device();

    void * vdata = (void*)(&data[0]);

    size_t n_bytes = n_elem * sizeof(T);

    void * buffer = mem.at(vdata);

    int failed = omp_target_memcpy(vdata, buffer, n_bytes, 0, 0, host, target);
    if (failed != 0) {
        std::cerr << "Failed to copy " << name << " device to host" << std::endl;
    }
    return;
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
            std::ostringstream o;
            o << "Object has different endianness than system- cannot use";
            throw std::runtime_error(o.str().c_str());
        }
    }
    return fmt;
}

template <typename T>
T * extract_buffer(
    py::buffer data,
    char const * name,
    size_t assert_dims,
    std::vector <int64_t> & shape,
    std::vector <int64_t> assert_shape
) {
    // Get buffer info structure
    auto info = data.request();

    // Extract the format character for the target type
    std::string target_format = get_format(py::format_descriptor <T>::format());

    // Extract the format for the input buffer
    std::string buffer_format = get_format(info.format);

    // Verify format string
    if (buffer_format != target_format) {
        // On 64bit linux, numpy is internally inconsistent with the
        // format codes for int64_t and long long:
        //   https://github.com/numpy/numpy/issues/12264
        // Here we treat them as equivalent.
        if (((buffer_format == "q") || (buffer_format == "l"))
            && ((target_format == "q") || (target_format == "l"))) {
            // What could possibly go wrong...
        } else {
            std::ostringstream o;
            o << "Object " << name << " has format \"" << buffer_format
              << "\" instead of \"" << target_format << "\"";
            throw std::runtime_error(o.str().c_str());
        }
    }

    // Verify itemsize
    if (info.itemsize != sizeof(T)) {
        std::ostringstream o;
        o << "Object " << name << " has item size of "
          << info.itemsize << " instead of " << sizeof(T);
        throw std::runtime_error(o.str().c_str());
    }

    // Verify number of dimensions
    if (info.ndim != assert_dims) {
        std::ostringstream o;
        o << "Object " << name << " has " << info.ndim
          << " dimensions instead of " << assert_dims;
        throw std::runtime_error(o.str().c_str());
    }

    // Get array dimensions
    for (py::ssize_t d = 0; d < info.ndim; d++) {
        shape[d] = info.shape[d];
    }

    // Check strides and verify that memory is contiguous
    size_t stride = info.itemsize;
    for (int d = info.ndim - 1; d >= 0; d--) {
        if (info.strides[d] != stride) {
            std::ostringstream o;
            o << "Object " << name
              << ": python buffers must be contiguous in memory.";
            throw std::runtime_error(o.str().c_str());
        }
        stride *= info.shape[d];
    }

    // If the user wants to verify any of the dimensions, do that
    for (py::ssize_t d = 0; d < info.ndim; d++) {
        if (assert_shape[d] >= 0) {
            // We are checking this dimension
            if (assert_shape[d] != shape[d]) {
                std::ostringstream o;
                o << "Object " << name << " dimension " << d
                  << " has length " << shape[d]
                  << " instead of " << assert_shape[d];
                throw std::runtime_error(o.str().c_str());
            }
        }
    }

    return static_cast <T *> (info.ptr);
}


PYBIND11_MODULE(pyomptarget, m) {
    m.doc() = R"(
    Compiled extension using OpenMP target offload.

    )";

    py::class_ <Interval> (
        m, "Interval",
        R"(
        Numpy dtype for an interval
        )")
    .def(py::init([]() {
                      return Interval();
                  }))
    .def_readwrite("start", &Interval::start)
    .def_readwrite("stop", &Interval::stop)
    .def_readwrite("first", &Interval::first)
    .def_readwrite("last", &Interval::last)
    .def("astuple",
         [](const Interval & self) {
             return py::make_tuple(self.start, self.stop, self.first, self.last);
         })
    .def_static("fromtuple", [](const py::tuple & tup) {
                    if (py::len(tup) != 4) {
                        throw py::cast_error("Invalid size");
                    }
                    return Interval{tup[0].cast <double>(),
                                    tup[1].cast <double>(),
                                    tup[2].cast <int64_t>(),
                                    tup[3].cast <int64_t>()};
                });


    PYBIND11_NUMPY_DTYPE(Interval, start, stop, first, last);

    m.def(
        "stage_data", [](
            py::buffer boresight,
            py::buffer quats,
            py::buffer intervals,
            py::buffer shared_flags
        ) {
            int ndev = omp_get_num_devices();
            std::cout << "OMP found " << ndev << " available target offload devices" << std::endl;
            int target = ndev - 1;
            int host = omp_get_initial_device();
            int defdev = omp_get_default_device();
            std::cout << "OMP initial host device = " << host << std::endl;
            std::cout << "OMP target device = " << target << std::endl;
            std::cout << "OMP default device = " << defdev << std::endl;

            std::unordered_map <void *, void *> mem;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            double * raw_boresight = extract_buffer <double> (
                boresight, "boresight", 2, temp_shape, {-1, 4}
            );
            int64_t n_samp = temp_shape[0];

            uint8_t * raw_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {n_samp}
            );

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, n_samp, 4}
            );
            int64_t n_det = temp_shape[0];

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            host_to_device(mem, 4 * n_samp, raw_boresight, "boresight");

            host_to_device(mem, n_samp, raw_flags, "flags");

            host_to_device(mem, n_view, raw_intervals, "intervals");

            host_to_device(mem, n_samp * n_det * 4, raw_quats, "quats");

            return mem;
        });

    m.def(
        "unstage_data", [](
            std::unordered_map <void *, void *> mem,
            py::buffer quats
        ) {
            std::vector <int64_t> temp_shape(3);
            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, -1, 4}
            );
            int64_t n_det = temp_shape[0];
            int64_t n_samp = temp_shape[1];
            device_to_host(mem, n_samp * n_det * 4, raw_quats, "quats");

            int target = omp_get_num_devices() - 1;
            for(auto & p : mem) {
                std::cerr << "Disassociate host pointer " << p.first << std::endl;
                int fail = omp_target_disassociate_ptr(p.first, target);
                if (fail != 0) {
                    std::cerr << "Failed to disassociate host pointer " << p.first << std::endl;
                }
                omp_target_free(p.second, target);
            }
        });

    m.def(
        "pointing_detector", [](
            std::unordered_map <void *, void *> mem,
            py::buffer focalplane,
            py::buffer boresight,
            py::buffer quat_index,
            py::buffer quats,
            py::buffer intervals,
            py::buffer shared_flags
        ) {
            // What if quats has more dets than we are considering in quat_index?

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_quat_index = extract_buffer <int32_t> (
                quat_index, "quat_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            double * raw_focalplane = extract_buffer <double> (
                focalplane, "focalplane", 2, temp_shape, {n_det, 4}
            );

            double * raw_boresight = extract_buffer <double> (
                boresight, "boresight", 2, temp_shape, {-1, 4}
            );
            int64_t n_samp = temp_shape[0];
            double * dev_boresight = (double*)mem.at(raw_boresight);

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, n_samp, 4}
            );
            double * dev_quats = (double*)mem.at(raw_quats);

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            Interval * dev_intervals = (Interval*)mem.at(raw_intervals);
            int64_t n_view = temp_shape[0];
            std::cerr << "interval 0 = " << raw_intervals[0].first << ", " << raw_intervals[0].last << std::endl;

            uint8_t * raw_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {n_samp}
            );
            uint8_t * dev_flags = (uint8_t*)mem.at(raw_flags);

            int present = omp_target_is_present(raw_boresight, 0);
            std::cerr << "target present boresight = " << present << std::endl;
            present = omp_target_is_present(raw_quats, 0);
            std::cerr << "target present quats = " << present << std::endl;
            present = omp_target_is_present(raw_intervals, 0);
            std::cerr << "target present intervals = " << present << std::endl;
            present = omp_target_is_present(raw_flags, 0);
            std::cerr << "target present flags = " << present << std::endl;

            # pragma omp target data           \
                device(0)                      \
                map(to:                        \
                    raw_focalplane[0:4*n_det], \
                    raw_quat_index[0:n_det],   \
                    n_view,                    \
                    n_det,                     \
                    n_samp                     \
                )
            {
                # pragma omp target teams distribute collapse(2) \
                    is_device_ptr(                 \
                        dev_boresight,             \
                        dev_quats,                 \
                        dev_intervals,             \
                        dev_flags                  \
                    )
                for (int64_t idet = 0; idet < n_det; idet++) {
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        # pragma omp parallel for default(shared)
                        for (
                            int64_t isamp = raw_intervals[iview].first;
                            isamp <= raw_intervals[iview].last;
                            isamp++
                        ) {
                            pointing_detector_inner(
                                raw_quat_index,
                                dev_flags,
                                dev_boresight,
                                raw_focalplane,
                                dev_quats,
                                isamp,
                                n_samp,
                                idet
                            );
                        }
                    }
                }
            }

            return;
        });

}
