// Copyright (c) 2026 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#include <intervals.hpp>

#include <cmath>


// The RING helpers below follow healpix_base.cc in HEALPix C++, so the bilinear
// stencil matches healpy.get_interp_weights.

// The ring just north of z = cos(theta), numbered from 1, or 0 north of the first
// ring.

int64_t real_space_convolve_ring_above(int64_t const & nside, double const & z) {
    double az = ::fabs(z);
    if (az <= 2.0 / 3.0) {
        return (int64_t)(nside * (2.0 - 1.5 * z));
    }
    int64_t iring = (int64_t)(nside * ::sqrt(3.0 * (1.0 - az)));
    return (z > 0.0) ? iring : 4 * nside - iring - 1;
}

// First pixel, pixel count and longitude shift of a ring (1 to 4 nside - 1).

void real_space_convolve_ring_layout(
    int64_t const & nside,
    int64_t const & ring,
    int64_t & first_pix,
    int64_t & n_ring_pix,
    bool & shifted
) {
    int64_t north_ring = (ring > 2 * nside) ? 4 * nside - ring : ring;
    if (north_ring < nside) {
        n_ring_pix = 4 * north_ring;
        shifted = true;
        first_pix = 2 * north_ring * (north_ring - 1);
    } else {
        n_ring_pix = 4 * nside;
        shifted = ((north_ring - nside) & 1) == 0;
        first_pix = 2 * nside * (nside - 1) + (north_ring - nside) * n_ring_pix;
    }
    if (north_ring != ring) {
        first_pix = 12 * nside * nside - first_pix - n_ring_pix;
    }
    return;
}

// Colatitude, and its cosine and sine, of the pixel centers of a ring.

void real_space_convolve_ring_angles(
    int64_t const & nside,
    int64_t const & ring,
    double & theta,
    double & z,
    double & sth
) {
    double fact2 = 4.0 / (double)(12 * nside * nside);
    double fact1 = (double)(2 * nside) * fact2;
    int64_t north_ring = (ring > 2 * nside) ? 4 * nside - ring : ring;
    if (north_ring < nside) {
        double tmp = (double)(north_ring * north_ring) * fact2;
        z = 1.0 - tmp;
        sth = ::sqrt(tmp * (2.0 - tmp));
        theta = ::atan2(sth, z);
    } else {
        z = (double)(2 * nside - north_ring) * fact1;
        sth = ::sqrt((1.0 - z) * (1.0 + z));
        theta = ::acos(z);
    }
    if (north_ring != ring) {
        theta = M_PI - theta;
        z = -z;
    }
    return;
}

// The two pixels of a ring that bracket longitude phi, their linear weights and
// the longitudes of their centers.

void real_space_convolve_ring_pair(
    int64_t const & nside,
    double const * ring_angles,
    int64_t const & ring,
    double const & phi,
    int64_t * pix,
    double * wgt,
    double * pix_phi,
    double & theta,
    double & z,
    double & sth
) {
    int64_t first_pix;
    int64_t n_ring_pix;
    bool shifted;
    real_space_convolve_ring_layout(nside, ring, first_pix, n_ring_pix, shifted);
    theta = ring_angles[3 * ring];
    z = ring_angles[3 * ring + 1];
    sth = ring_angles[3 * ring + 2];
    double dphi = 2.0 * M_PI / (double)n_ring_pix;
    double inv_dphi = (double)n_ring_pix / (2.0 * M_PI);
    double shift = shifted ? 0.5 : 0.0;
    double tmp = phi * inv_dphi - shift;
    int64_t i1 = (tmp < 0.0) ? (int64_t)tmp - 1 : (int64_t)tmp;
    double w1 = (phi - ((double)i1 + shift) * dphi) * inv_dphi;
    int64_t i2 = i1 + 1;

    // Phi comes from atan2 and lies in [-pi, pi], so either index can fall
    // outside the ring on either side.
    if (i1 < 0) {
        i1 += n_ring_pix;
    } else if (i1 >= n_ring_pix) {
        i1 -= n_ring_pix;
    }
    if (i2 < 0) {
        i2 += n_ring_pix;
    } else if (i2 >= n_ring_pix) {
        i2 -= n_ring_pix;
    }
    pix[0] = first_pix + i1;
    pix[1] = first_pix + i2;
    wgt[0] = 1.0 - w1;
    wgt[1] = w1;
    pix_phi[0] = ((double)i1 + shift) * dphi;
    pix_phi[1] = ((double)i2 + shift) * dphi;
    return;
}

// Bilinear interpolation on a RING map at colatitude theta, with z = cos(theta),
// and longitude phi.  Returns the four pixels, their weights, and the cosine and
// sine of the colatitude and the longitude of each pixel center, with the first
// two pixels on the northern ring, as healpy orders them.  The ring_angles
// table holds the output of real_space_convolve_ring_angles for every ring.

void real_space_convolve_interp(
    int64_t const & nside,
    double const * ring_angles,
    double const & z,
    double const & theta,
    double const & phi,
    int64_t * pix,
    double * wgt,
    double * pix_z,
    double * pix_sth,
    double * pix_phi
) {
    int64_t ring1 = real_space_convolve_ring_above(nside, z);
    int64_t ring2 = ring1 + 1;
    double theta1 = 0.0;
    double z1 = 0.0;
    double sth1 = 0.0;
    double theta2 = 0.0;
    double z2 = 0.0;
    double sth2 = 0.0;
    if (ring1 > 0) {
        real_space_convolve_ring_pair(
            nside, ring_angles, ring1, phi, &pix[0], &wgt[0], &pix_phi[0],
            theta1, z1, sth1
        );
    }
    if (ring2 < 4 * nside) {
        real_space_convolve_ring_pair(
            nside, ring_angles, ring2, phi, &pix[2], &wgt[2], &pix_phi[2],
            theta2, z2, sth2
        );
    }
    if (ring1 == 0) {
        // Above the first ring, the missing pair is the opposite side of that ring.
        double wtheta = theta / theta2;
        double fac = (1.0 - wtheta) * 0.25;
        wgt[0] = fac;
        wgt[1] = fac;
        wgt[2] = wgt[2] * wtheta + fac;
        wgt[3] = wgt[3] * wtheta + fac;
        pix[0] = (pix[2] + 2) & 3;
        pix[1] = (pix[3] + 2) & 3;
        pix_phi[0] = ((double)pix[0] + 0.5) * M_PI_2;
        pix_phi[1] = ((double)pix[1] + 0.5) * M_PI_2;
        z1 = z2;
        sth1 = sth2;
    } else if (ring2 == 4 * nside) {
        // Below the last ring, the same with the south pole.
        double wtheta = (theta - theta1) / (M_PI - theta1);
        double fac = wtheta * 0.25;
        int64_t last_first = 12 * nside * nside - 4;
        wgt[0] = wgt[0] * (1.0 - wtheta) + fac;
        wgt[1] = wgt[1] * (1.0 - wtheta) + fac;
        wgt[2] = fac;
        wgt[3] = fac;
        pix[2] = ((pix[0] + 2) & 3) + last_first;
        pix[3] = ((pix[1] + 2) & 3) + last_first;
        pix_phi[2] = ((double)(pix[2] - last_first) + 0.5) * M_PI_2;
        pix_phi[3] = ((double)(pix[3] - last_first) + 0.5) * M_PI_2;
        z2 = z1;
        sth2 = sth1;
    } else {
        double wtheta = (theta - theta1) / (theta2 - theta1);
        wgt[0] *= 1.0 - wtheta;
        wgt[1] *= 1.0 - wtheta;
        wgt[2] *= wtheta;
        wgt[3] *= wtheta;
    }
    pix_z[0] = z1;
    pix_z[1] = z1;
    pix_z[2] = z2;
    pix_z[3] = z2;
    pix_sth[0] = sth1;
    pix_sth[1] = sth1;
    pix_sth[2] = sth2;
    pix_sth[3] = sth2;
    return;
}

// Spin-2 parallel transport from a pixel center to the line of sight, as
// exp(2i delta) with delta = alpha - gamma.  Alpha is the bearing from the pixel
// center towards the line of sight and gamma the bearing from the line of sight
// back towards the pixel center, both east of north.

void real_space_convolve_spin2_transport(
    double const & z_pix,
    double const & sth_pix,
    double const & phi_pix,
    double const & z_los,
    double const & sth_los,
    double const & phi_los,
    double & cos_2d,
    double & sin_2d
) {
    double dphi = phi_los - phi_pix;
    double ch = ::cos(0.5 * dphi);
    double sh = ::sin(0.5 * dphi);
    double sin_dphi = 2.0 * ch * sh;
    double sh2 = sh * sh;
    double cos_dphi = 1.0 - 2.0 * sh2;

    double dz = z_los - z_pix;
    double ds = sth_los - sth_pix;
    double st2 = 0.25 * (dz * dz + ds * ds);
    double prod_sth = sth_pix * sth_los;
    double hav = st2 + prod_sth * sh2;

    double num = 2.0 * sin_dphi * (z_los + z_pix) * hav;
    double den = 4.0 * st2 * (1.0 - st2) * cos_dphi -
                 4.0 * prod_sth * z_los * z_pix * sh2 * sh2 +
                 prod_sth * sin_dphi * sin_dphi;

    double norm = num * num + den * den;
    if (norm == 0.0) {
        // Coincident points need no rotation.
        cos_2d = 1.0;
        sin_2d = 0.0;
    } else {
        double inv = 1.0 / norm;
        cos_2d = (den * den - num * num) * inv;
        sin_2d = -2.0 * num * den * inv;
    }
    return;
}

// Beam-convolved I, Q and U at one sample, in the meridian basis of the line of
// sight.  Q and U are computed only for a three-component map.

void real_space_convolve_inner(
    int64_t const & nside,
    double const * ring_angles,
    int64_t const & nnz,
    double const * maps,
    int64_t const & n_node,
    double const * node_dirs,
    double const * node_weights_I,
    double const * node_weights_P,
    double const * quat,
    double * stokes
) {
    double const xaxis[3] = {1.0, 0.0, 0.0};
    double const yaxis[3] = {0.0, 1.0, 0.0};
    double const zaxis[3] = {0.0, 0.0, 1.0};
    double rot_x[3];
    double rot_y[3];
    double los[3];
    qa_rotate(quat, xaxis, rot_x);
    qa_rotate(quat, yaxis, rot_y);
    qa_rotate(quat, zaxis, los);
    double z_los = los[2];
    double sth_los = ::hypot(los[0], los[1]);
    double phi_los = ::atan2(los[1], los[0]);

    int64_t n_pix = 12 * nside * nside;

    int64_t pix[4];
    double wgt[4];
    double pix_z[4];
    double pix_sth[4];
    double pix_phi[4];
    double vec[3];

    // Nearby nodes share stencil pixels, so the transport of each pixel to this
    // line of sight is kept in a direct-mapped table.  The slot is a
    // multiplicative hash of the pixel index, because the low bits alias between
    // adjacent rings, whose lengths are multiples of a power of two.
    constexpr int64_t cache_bits = 10;
    constexpr int64_t n_cache = 1 << cache_bits;
    int64_t cache_pix[n_cache];
    double cache_cos[n_cache];
    double cache_sin[n_cache];
    if (nnz == 3) {
        for (int64_t slot = 0; slot < n_cache; slot++) {
            cache_pix[slot] = -1;
        }
    }

    stokes[0] = 0.0;
    stokes[1] = 0.0;
    stokes[2] = 0.0;
    for (int64_t inode = 0; inode < n_node; inode++) {
        double const * dir = &(node_dirs[3 * inode]);
        for (int64_t k = 0; k < 3; k++) {
            vec[k] = rot_x[k] * dir[0] + rot_y[k] * dir[1] + los[k] * dir[2];
        }

        // acos is accurate away from the poles and asin near them.
        double z = vec[2];
        double sth = ::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
        double theta;
        if (::fabs(z) < sth) {
            theta = ::acos(z);
        } else if (z > 0.0) {
            theta = ::asin(sth);
        } else {
            theta = M_PI - ::asin(sth);
        }

        // TODO: look into how to speed up this atan2
        double phi = ::atan2(vec[1], vec[0]);
        real_space_convolve_interp(
            nside, ring_angles, z, theta, phi, pix, wgt, pix_z, pix_sth, pix_phi
        );

        double val_I = 0.0;
        for (int64_t j = 0; j < 4; j++) {
            val_I += wgt[j] * maps[pix[j]];
        }
        stokes[0] += node_weights_I[inode] * val_I;
        if (nnz < 3) {
            continue;
        }

        // Q + iU of each pixel, multiplied by exp(2i delta).
        double val_Q = 0.0;
        double val_U = 0.0;
        for (int64_t j = 0; j < 4; j++) {
            int64_t slot = (int64_t)(
                ((uint64_t)pix[j] * 0x9E3779B97F4A7C15ull) >> (64 - cache_bits)
            );
            if (cache_pix[slot] != pix[j]) {
                real_space_convolve_spin2_transport(
                    pix_z[j], pix_sth[j], pix_phi[j], z_los, sth_los, phi_los,
                    cache_cos[slot], cache_sin[slot]
                );
                cache_pix[slot] = pix[j];
            }
            double cos_2d = cache_cos[slot];
            double sin_2d = cache_sin[slot];
            double q_pix = maps[n_pix + pix[j]];
            double u_pix = maps[2 * n_pix + pix[j]];
            val_Q += wgt[j] * (q_pix * cos_2d - u_pix * sin_2d);
            val_U += wgt[j] * (q_pix * sin_2d + u_pix * cos_2d);
        }
        stokes[1] += node_weights_P[inode] * val_Q;
        stokes[2] += node_weights_P[inode] * val_U;
    }
    return;
}

void init_ops_real_space_convolve(py::module & m) {
    m.def("ops_real_space_convolve",
          [](
              py::buffer quat_index,
              py::buffer quats,
              py::buffer weight_index,
              py::buffer weights,
              py::buffer det_data_index,
              py::buffer det_data,
              py::buffer intervals,
              int64_t nside,
              py::buffer maps,
              py::buffer node_dirs,
              py::buffer node_weights_I,
              py::buffer node_weights_P)
          {
              // This is used to return the actual shape of each buffer
              std::vector <int64_t> temp_shape(3);

              int32_t * raw_quat_index = extract_buffer <int32_t> (
                  quat_index, "quat_index", 1, temp_shape, {-1}
              );
              int64_t n_det = temp_shape[0];

              double * raw_quats = extract_buffer <double> (
                  quats, "quats", 3, temp_shape, {-1, -1, 4}
              );
              int64_t n_samp = temp_shape[1];

              int32_t * raw_weight_index = extract_buffer <int32_t> (
                  weight_index, "weight_index", 1, temp_shape, {n_det}
              );
              double * raw_weights = extract_buffer <double> (
                  weights, "weights", 3, temp_shape, {-1, n_samp, -1}
              );
              int64_t nnz = temp_shape[2];

              int32_t * raw_det_data_index = extract_buffer <int32_t> (
                  det_data_index, "det_data_index", 1, temp_shape, {n_det}
              );
              double * raw_det_data = extract_buffer <double> (
                  det_data, "det_data", 2, temp_shape, {-1, n_samp}
              );

              Interval * raw_intervals = extract_buffer <Interval> (
                  intervals, "intervals", 1, temp_shape, {-1}
              );
              int64_t n_view = temp_shape[0];

              double * raw_maps = extract_buffer <double> (
                  maps, "maps", 2, temp_shape, {nnz, 12 * nside * nside}
              );

              double * raw_node_dirs = extract_buffer <double> (
                  node_dirs, "node_dirs", 2, temp_shape, {-1, 3}
              );
              int64_t n_node = temp_shape[0];

              double * raw_node_weights_I = extract_buffer <double> (
                  node_weights_I, "node_weights_I", 1, temp_shape, {n_node}
              );
              double * raw_node_weights_P = extract_buffer <double> (
                  node_weights_P, "node_weights_P", 1, temp_shape,
                  {(nnz == 3) ? n_node : -1}
              );

              // Ring angles are shared by all samples, so they are computed once.
              // Rings are numbered from 1.
              std::vector <double> ring_angles(3 * 4 * nside);
              for (int64_t ring = 1; ring < 4 * nside; ring++) {
                  real_space_convolve_ring_angles(
                      nside, ring, ring_angles[3 * ring], ring_angles[3 * ring + 1],
                      ring_angles[3 * ring + 2]
                  );
              }

              for (int64_t idet = 0; idet < n_det; idet++) {
                  int64_t off_q = raw_quat_index[idet] * n_samp;
                  int64_t off_w = raw_weight_index[idet] * n_samp;
                  int64_t off_d = raw_det_data_index[idet] * n_samp;
                  for (int64_t iview = 0; iview < n_view; iview++) {
                      #pragma omp parallel for default(shared) schedule(static)
                      for (
                          int64_t isamp = raw_intervals[iview].first;
                          isamp < raw_intervals[iview].last;
                          isamp++
                      ) {
                          double stokes[3];
                          real_space_convolve_inner(
                              nside,
                              ring_angles.data(),
                              nnz,
                              raw_maps,
                              n_node,
                              raw_node_dirs,
                              raw_node_weights_I,
                              raw_node_weights_P,
                              &(raw_quats[4 * (off_q + isamp)]),
                              stokes
                          );
                          double const * wt = &(raw_weights[nnz * (off_w + isamp)]);
                          double val = 0.0;
                          for (int64_t iweight = 0; iweight < nnz; iweight++) {
                              val += wt[iweight] * stokes[iweight];
                          }
                          raw_det_data[off_d + isamp] += val;
                      }
                  }
              }
              return;
          }
    );
    return;
}
