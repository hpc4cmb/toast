
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

// 2/PI
#define TWOINVPI 0.63661977236758134308

// 2/3
#define TWOTHIRDS 0.66666666666666666667

// Helper table initialization
void hpix_init_utab(uint64_t * utab) {
    for (uint64_t m = 0; m < 256; ++m) {
        utab[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
            ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
            ((m & 0x40) << 6) | ((m & 0x80) << 7);
    }
    return;
}

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

void qa_rotate(double const * q_in, double const * v_in, double * v_out) {
    // The input quaternion has already been normalized on the host.

    double xw =  q_in[3] * q_in[0];
    double yw =  q_in[3] * q_in[1];
    double zw =  q_in[3] * q_in[2];
    double x2 = -q_in[0] * q_in[0];
    double xy =  q_in[0] * q_in[1];
    double xz =  q_in[0] * q_in[2];
    double y2 = -q_in[1] * q_in[1];
    double yz =  q_in[1] * q_in[2];
    double z2 = -q_in[2] * q_in[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                    (yw + xz) * v_in[2]) + v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) + v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) + v_in[2];

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

uint64_t hpix_xy2pix(uint64_t * utab, uint64_t x, uint64_t y) {
    return utab[x & 0xff] | (utab[(x >> 8) & 0xff] << 16) |
           (utab[(x >> 16) & 0xff] << 32) |
           (utab[(x >> 24) & 0xff] << 48) |
           (utab[y & 0xff] << 1) | (utab[(y >> 8) & 0xff] << 17) |
           (utab[(y >> 16) & 0xff] << 33) |
           (utab[(y >> 24) & 0xff] << 49);
}

void hpix_vec2zphi(double const * vec,
                   double * phi, int * region, double * z,
                   double * rtz) {
    // region encodes BOTH the sign of Z and whether its
    // absolute value is greater than 2/3.
    (*z) = vec[2];
    double za = fabs(*z);
    int itemp = ((*z) > 0.0) ? 1 : -1;
    (*region) = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    (*rtz) = sqrt(3.0 * (1.0 - za));
    (*phi) = atan2(vec[1], vec[0]);
    return;
}

void hpix_zphi2nest(int64_t nside, int64_t factor, uint64_t * utab, 
                    double phi, int region, double z,
                    double rtz, int64_t * pix) {
    double tt = (phi >= 0.0) ? phi * TWOINVPI : phi * TWOINVPI + 4.0;
    int64_t x;
    int64_t y;
    double temp1;
    double temp2;
    int64_t jp;
    int64_t jm;
    int64_t ifp;
    int64_t ifm;
    int64_t face;
    int64_t ntt;
    double tp;

    double dnside = static_cast <double> (nside);
    int64_t twonside = 2 * nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    int64_t nsideminusone = nside - 1;

    if ((region == 1) || (region == -1)) {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ifp = jp >> factor;
        ifm = jm >> factor;

        if (ifp == ifm) {
            face = (ifp == 4) ? (int64_t)4 : ifp + 4;
        } else if (ifp < ifm) {
            face = ifp;
        } else {
            face = ifm + 8;
        }

        x = jm & nsideminusone;
        y = nsideminusone - (jp & nsideminusone);
    } else {
        ntt = (int64_t)tt;

        tp = tt - (double)ntt;

        temp1 = dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);

        if (jp >= nside) {
            jp = nsideminusone;
        }
        if (jm >= nside) {
            jm = nsideminusone;
        }

        if (z >= 0) {
            face = ntt;
            x = nsideminusone - jm;
            y = nsideminusone - jp;
        } else {
            face = ntt + 8;
            x = jp;
            y = jm;
        }
    }

    uint64_t sipf = hpix_xy2pix(utab, (uint64_t)x, (uint64_t)y);

    (*pix) = (int64_t)sipf + (face << (2 * factor));

    return;
}

void pixels_healpix_nest_inner(
    int64_t nside,
    int64_t factor,
    uint64_t * utab,
    int32_t const * quat_index,
    int32_t const * pixel_index,
    double const * quats,
    uint8_t const * flags,
    uint8_t * hsub,
    int64_t * pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet
) {
    const double zaxis[3] = {0.0, 0.0, 1.0};
    int32_t p_indx = pixel_index[idet];
    int32_t q_indx = quat_index[idet];
    double dir[3];
    double z;
    double rtz;
    double phi;
    int region;
    size_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;
    size_t poff = p_indx * n_samp + isamp;
    int64_t sub_map;

    uint8_t check = flags[isamp] & 255;

    if (check != 0) {
        pixels[poff] = -1;
    } else {
        qa_rotate(&(quats[qoff]), zaxis, dir);
        hpix_vec2zphi(dir, &phi, &region, &z, &rtz);
        hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz, &(pixels[poff]));
        sub_map = (int64_t)(pixels[poff] / n_pix_submap);
        hsub[sub_map] = 1;
    }

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
            py::buffer shared_flags,
            py::buffer pixels
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

            int64_t * raw_pixels = extract_buffer <int64_t> (
                pixels, "pixels", 2, temp_shape, {n_det, n_samp}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            host_to_device(mem, 4 * n_samp, raw_boresight, "boresight");

            host_to_device(mem, n_samp, raw_flags, "flags");

            host_to_device(mem, n_samp * n_det, raw_pixels, "pixels");

            host_to_device(mem, n_view, raw_intervals, "intervals");

            host_to_device(mem, n_samp * n_det * 4, raw_quats, "quats");

            return mem;
        });

    m.def(
        "unstage_data", [](
            std::unordered_map <void *, void *> mem,
            py::buffer quats,
            py::buffer pixels
        ) {
            std::vector <int64_t> temp_shape(3);
            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, -1, 4}
            );
            int64_t n_det = temp_shape[0];
            int64_t n_samp = temp_shape[1];
            int64_t * raw_pixels = extract_buffer <int64_t> (
                pixels, "pixels", 2, temp_shape, {n_det, n_samp}
            );
            device_to_host(mem, n_samp * n_det * 4, raw_quats, "quats");
            device_to_host(mem, n_samp * n_det, raw_pixels, "pixels");

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
                            // Test stack allocation with a mapped variable
                            //double dummy[n_det];

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

    m.def(
        "pixels_healpix_nest", [](
            std::unordered_map <void *, void *> mem,
            py::buffer quat_index,
            py::buffer quats,
            py::buffer shared_flags,
            py::buffer pixel_index,
            py::buffer pixels,
            py::buffer intervals,
            py::buffer hit_submaps,
            int64_t n_pix_submap,
            int64_t nside
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_quat_index = extract_buffer <int32_t> (
                quat_index, "quat_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            int32_t * raw_pixel_index = extract_buffer <int32_t> (
                pixel_index, "pixel_index", 1, temp_shape, {n_det}
            );

            int64_t * raw_pixels = extract_buffer <int64_t> (
                pixels, "pixels", 2, temp_shape, {-1, -1}
            );
            int64_t n_samp = temp_shape[1];

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, n_samp, 4}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            uint8_t * raw_hsub = extract_buffer <uint8_t> (
                hit_submaps, "hit_submaps", 1, temp_shape, {-1}
            );
            int64_t n_submap = temp_shape[0];

            uint8_t * raw_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {n_samp}
            );

            uint64_t utab[256];
            hpix_init_utab(utab);

            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            double * dev_quats = (double*)mem.at(raw_quats);
            
            int64_t * dev_pixels = (int64_t*)mem.at(raw_pixels);

            Interval * dev_intervals = (Interval*)mem.at(raw_intervals);

            uint8_t * dev_flags = (uint8_t*)mem.at(raw_flags);

            // Make sure the lookup table exists on device
            // if (mem.count(vutab) == 0) {
            //     host_to_device(mem, 256, utab, "hpix_utab");
            // }
            // uint64_t * dev_utab = (uint64_t*)mem.at(utab);

            # pragma omp target data  \
            device(0)               \
            map(to:                   \
            utab[0:256],              \
            raw_pixel_index[0:n_det], \
            raw_quat_index[0:n_det],  \
            n_pix_submap,             \
            nside,                    \
            factor,                   \
            n_view,                   \
            n_det,                    \
            n_samp                   \
            )                         \
            map(tofrom: raw_hsub[0:n_submap])
            {
                # pragma omp target teams distribute collapse(2) \
                is_device_ptr(                                   \
                dev_pixels,                                      \
                dev_quats,                                       \
                dev_flags,                                       \
                dev_intervals                                   \
                )
                for (int64_t idet = 0; idet < n_det; idet++) {
                    for (int64_t iview = 0; iview < n_view; iview++) {
                        # pragma omp parallel for default(shared)
                        for (
                            int64_t isamp = dev_intervals[iview].first;
                            isamp <= dev_intervals[iview].last;
                            isamp++
                        ) {
                            pixels_healpix_nest_inner(
                                nside,
                                factor,
                                utab,
                                raw_quat_index,
                                raw_pixel_index,
                                dev_quats,
                                dev_flags,
                                raw_hsub,
                                dev_pixels,
                                n_pix_submap,
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
