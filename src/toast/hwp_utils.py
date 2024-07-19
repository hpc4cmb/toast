# Copyright (c) 2023-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.linalg import eigvalsh, lu_factor, lu_solve

from .dist import distribute_uniform
from .mpi import MPI


def hwpss_samples(n_samp, comm):
    """Helper function to distribute samples."""
    if comm is None:
        slc = slice(0, n_samp, 1)
        return slc
    dist = distribute_uniform(n_samp, comm.size)
    # The sample counts and displacement for each process
    sample_count = [x.n_elem for x in dist]
    sample_displ = [x.offset for x in dist]

    # The local sample range as a slice
    slc = slice(
        sample_displ[comm.rank],
        sample_displ[comm.rank] + sample_count[comm.rank],
        1,
    )
    return slc


def hwpss_sincos_buffer(angles, flags, n_harmonics, comm=None):
    slc = hwpss_samples(len(angles), comm)
    ang = np.copy(angles[slc])
    good = flags[slc] == 0
    my_samp = len(ang)
    sample_vals = 2 * n_harmonics
    my_sincos = np.zeros((my_samp, sample_vals), dtype=np.float32)
    for h in range(n_harmonics):
        my_sincos[good, 2 * h] = np.sin((h + 1) * ang[good])
        my_sincos[good, 2 * h + 1] = np.cos((h + 1) * ang[good])
    if comm is None:
        return my_sincos
    sincos = np.vstack(comm.allgather(my_sincos))
    return sincos


def hwpss_compute_coeff_covariance(times, flags, sincos, comm=None):
    slc = hwpss_samples(len(times), comm)
    my_sincos = sincos[slc, :]
    my_times = times[slc]
    my_flags = flags[slc]

    n_harmonics = sincos.shape[1] // 2
    my_cov = np.zeros((4 * n_harmonics, 4 * n_harmonics), dtype=np.float64)
    good = my_flags == 0
    # Compute local upper triangle
    for hr in range(0, n_harmonics):
        for hc in range(hr, n_harmonics):
            my_cov[4 * hr + 0, 4 * hc + 0] = np.dot(
                my_sincos[good, 2 * hr + 0], my_sincos[good, 2 * hc + 0]
            )
            my_cov[4 * hr + 0, 4 * hc + 1] = np.dot(
                my_sincos[good, 2 * hr + 0],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 0, 4 * hc + 2] = np.dot(
                my_sincos[good, 2 * hr + 0], my_sincos[good, 2 * hc + 1]
            )
            my_cov[4 * hr + 0, 4 * hc + 3] = np.dot(
                my_sincos[good, 2 * hr + 0],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )

            my_cov[4 * hr + 1, 4 * hc + 0] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                my_sincos[good, 2 * hc + 0],
            )
            my_cov[4 * hr + 1, 4 * hc + 1] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 1, 4 * hc + 2] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                my_sincos[good, 2 * hc + 1],
            )
            my_cov[4 * hr + 1, 4 * hc + 3] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )

            my_cov[4 * hr + 2, 4 * hc + 0] = np.dot(
                my_sincos[good, 2 * hr + 1], my_sincos[good, 2 * hc + 0]
            )
            my_cov[4 * hr + 2, 4 * hc + 1] = np.dot(
                my_sincos[good, 2 * hr + 1],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 2, 4 * hc + 2] = np.dot(
                my_sincos[good, 2 * hr + 1], my_sincos[good, 2 * hc + 1]
            )
            my_cov[4 * hr + 2, 4 * hc + 3] = np.dot(
                my_sincos[good, 2 * hr + 1],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )

            my_cov[4 * hr + 3, 4 * hc + 0] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                my_sincos[good, 2 * hc + 0],
            )
            my_cov[4 * hr + 3, 4 * hc + 1] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 3, 4 * hc + 2] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                my_sincos[good, 2 * hc + 1],
            )
            my_cov[4 * hr + 3, 4 * hc + 3] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )
    # Accumulate across processes
    if comm is None:
        cov = my_cov
    else:
        cov = np.zeros((4 * n_harmonics, 4 * n_harmonics), dtype=np.float64)
        comm.Allreduce(my_cov, cov, op=MPI.SUM)

    # Fill in lower triangle
    for hr in range(0, 4 * n_harmonics):
        for hc in range(0, hr):
            cov[hr, hc] = cov[hc, hr]
    # Check that condition number is reasonable
    evals = eigvalsh(cov)
    rcond = np.min(evals) / np.max(evals)
    if rcond < 1.0e-8:
        return None
    # LU factorization for later solve
    cov_lu, cov_piv = lu_factor(cov)
    return cov_lu, cov_piv


def hwpss_compute_coeff(detdata, flags, times, sincos, cov_lu, cov_piv):
    n_samp = len(times)
    n_harmonics = sincos.shape[1] // 2
    good = flags == 0
    bad = np.logical_not(good)
    input = np.copy(detdata)
    dc = np.mean(input[good])
    input[:] -= dc
    input[bad] = 0
    rhs = np.zeros(4 * n_harmonics, dtype=np.float64)
    for h in range(n_harmonics):
        rhs[4 * h + 0] = np.dot(input[good], sincos[good, 2 * h])
        rhs[4 * h + 1] = np.dot(
            input[good], np.multiply(sincos[good, 2 * h], times[good])
        )
        rhs[4 * h + 2] = np.dot(input[good], sincos[good, 2 * h + 1])
        rhs[4 * h + 3] = np.dot(
            input[good], np.multiply(sincos[good, 2 * h + 1], times[good])
        )
    coeff = lu_solve((cov_lu, cov_piv), rhs)
    return coeff


def hwpss_build_model(times, flags, sincos, coeff):
    n_samp = len(times)
    n_harmonics = sincos.shape[1] // 2
    good = flags == 0
    model = np.zeros(n_samp, dtype=np.float64)
    for h in range(n_harmonics):
        model[good] += coeff[4 * h + 0] * sincos[good, 2 * h]
        model[good] += coeff[4 * h + 1] * np.multiply(sincos[good, 2 * h], times[good])
        model[good] += coeff[4 * h + 2] * sincos[good, 2 * h + 1]
        model[good] += coeff[4 * h + 3] * np.multiply(
            sincos[good, 2 * h + 1], times[good]
        )
    return model
