# Copyright (c) 2023-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import scipy.interpolate
from scipy.optimize import least_squares

from .dist import distribute_uniform


def hwpss_samples(n_samp, comm):
    """Helper function to distribute samples.

    This distributes slices of samples uniformly among the processes
    of an MPI communicator.

    Args:
        n_samp (int): The number of samples
        comm (MPI.Comm): The communicator.

    Returns:
        (slice):  The local slice on this process.

    """
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
    """Precompute sin / cos terms.

    Compute the harmonic terms involving sin / cos of the HWP angle.
    This is done once in parallel and used by each process for its
    local detectors.

    Args:
        angles (array):  The HWP angle at each time stamp
        flags (array):  The flags indicating bad angle samples.
        n_harmonics (int):  The number of harmonics in the model.
        comm (MPI.Comm):  The optional communicator to parallelize
            the calculation.

    Returns:
        (array):  The full buffer of sin/cos factors on all processes.

    """
    slc = hwpss_samples(len(angles), comm)
    ang = np.copy(angles[slc])
    good = flags[slc] == 0
    my_samp = len(ang)
    sample_vals = 2 * n_harmonics
    my_sincos = np.zeros((my_samp, sample_vals), dtype=np.float64)
    for h in range(n_harmonics):
        my_sincos[good, 2 * h] = np.sin((h + 1) * ang[good])
        my_sincos[good, 2 * h + 1] = np.cos((h + 1) * ang[good])
    if comm is None:
        return my_sincos
    sincos = np.vstack(comm.allgather(my_sincos))
    return sincos


def hwpss_build_model(sincos, flags, coeff):
    """Construct the HWPSS template from coefficients.

    The array of coefficients should either be a one-dimensional array of fixed
    coefficients that are valid for all samples, or a 2D array with a set of
    coefficients for every sample.

    Args:
        sincos (array):  The pre-computed sin / cos terms.
        flags (array):  The flags indicating bad angle samples
        coeff (array):  The model coefficents for this detector.

    Returns:
        (array):  The template.

    """
    n_samp = sincos.shape[0]
    n_harmonics = sincos.shape[1] // 2

    good = flags == 0
    out = np.zeros(n_samp, dtype=np.float64)
    if np.count_nonzero(good) == 0:
        # No good samples
        return out

    # Accumulate the harmonics
    if len(coeff.shape) == 2:
        # Per-sample coefficients
        if coeff.shape[0] != n_samp:
            msg = "coefficient array has incorrect number of samples"
            raise RuntimeError(msg)
        if coeff.shape[1] != 2 * n_harmonics:
            msg = "coefficient array has incorrect number of harmonics"
            raise RuntimeError(msg)
        for h in range(n_harmonics):
            out[good] += coeff[good, 2 * h] * sincos[good, 2 * h]
            out[good] += coeff[good, 2 * h + 1] * sincos[good, 2 * h + 1]
    else:
        # One set of coefficients for all samples
        for h in range(n_harmonics):
            out[good] += coeff[2 * h] * sincos[good, 2 * h]
            out[good] += coeff[2 * h + 1] * sincos[good, 2 * h + 1]
    return out


def hwpss_build_interpolated_model(sincos, flags, coeff, coeff_indx, coeff_wts=None):
    """Construct the HWPSS template from interpolated coefficients.

    This is similar to `hwpss_build_model()`, but takes an array of coefficients
    and the sample indices where those coefficients were estimated.  Then it
    interpolates and accumulates those varying coefficients to build a dynamic
    model of the HWPSS.

    The array of coefficients should be a 2D array where the first dimension
    is the harmonic coefficient and the second dimension contains the values of
    that coefficient at the samples specified by `coeff_indx`.

    Args:
        sincos (array):  The pre-computed sin / cos terms.
        flags (array):  The flags indicating bad angle samples
        coeff (array):  The 2D array containing the piecewise values for each
            harmonic coefficient.
        coeff_indx (array):  The sample indices where each harmonic coefficent
            was estimated.
        coeff_wts (array):  The relative weights for each piecewise set of values

    Returns:
        (array):  The dynamic model of the HWPSS.

    """
    n_samp = sincos.shape[0]
    n_harmonics = sincos.shape[1] // 2
    n_chunk = len(coeff_indx)

    if len(coeff.shape) != 2:
        msg = "Coefficient array should be 2D"
        raise RuntimeError(msg)

    if coeff.shape[0] != 2 * n_harmonics or coeff.shape[1] != n_chunk:
        msg = "Coefficient array should have shape n_harmonics X len(coeff_indx)"
        raise RuntimeError(msg)

    good = flags == 0
    out = np.zeros(n_samp, dtype=np.float64)
    if np.count_nonzero(good) == 0:
        # No good samples
        return out

    if coeff_wts is None:
        wts = np.ones(n_chunk, dtype=np.float64)
    else:
        wts = coeff_wts

    smoothing = float(max(n_chunk - np.sqrt(2 * n_chunk), 4))
    samp_indx = np.arange(n_samp)
    cf_interp = np.empty(n_samp)

    fcoeff_indx = np.array(coeff_indx, dtype=np.float64)
    f_n_samp = float(n_samp)

    # print(f"coeff_indx = {fcoeff_indx}")
    # print(f"coeff = {coeff}")
    # print(f"wts = {wts}")
    # print(f"n_samp = {n_samp}")
    # print(f"smoothing = {smoothing}", flush=True)

    for h in range(n_harmonics):
        # Sine term
        splrep = scipy.interpolate.make_splrep(
            coeff_indx,
            np.ascontiguousarray(coeff[2 * h, :]),
            w=wts,
            xb=0.0,
            xe=f_n_samp,
            k=3,
            s=smoothing,
        )
        cf_interp[:] = scipy.interpolate.splev(samp_indx, splrep, ext=0)
        out[good] += cf_interp[good] * sincos[good, 2 * h]
        # Cosine term
        splrep = scipy.interpolate.make_splrep(
            coeff_indx,
            np.ascontiguousarray(coeff[2 * h + 1, :]),
            w=wts,
            xb=0.0,
            xe=f_n_samp,
            k=3,
            s=smoothing,
        )
        cf_interp[:] = scipy.interpolate.splev(samp_indx, splrep, ext=0)
        out[good] += cf_interp[good] * sincos[good, 2 * h + 1]
    return out


def hwpss_compute_coeff(sincos, detdata, flags, guess=None, xtol=1.0e-10, gtol=1.0e-10):
    """Compute the HWPSS model coefficients.

    The HWPSS model for this function is the one used by the Maxipol and EBEX
    experiments.  See for example Joy Didier's thesis, equation 8.17:

    https://academiccommons.columbia.edu/doi/10.7916/D8MW2HCG

    In this function we ignore the "time drift" terms, since those are captured
    by adjusting the chunk size over which the model is estimated.

    The model with N harmonics and HWP angle H can be written as:

    h(t) = Sum(i=1...N) { C_i0 * cos(i * H(t)) + C_i1 * sin(i * H(t)) }

    Writing this in terms of the vector of coefficients and the matrix of
    of sin / cos factors:

    h(t) = M x C
    h(t) =
    [ cos(H(t_0))  sin(H(t_0))  cos(2H(t_0)) sin(2H(t_0)) ...]
    [ cos(H(t_1))  sin(H(t_1))  cos(2H(t_1)) sin(2H(t_1)) ...]
    [ ... ]  X  [ C_10 C_11 C_20 C_21 C_30 C_31 ... ]^T

    The least squares solution for the coefficients is then

    C = (M^T M)^-1 M^T h(t)

    Which we solve with an iterative approach rather than building and
    inverting the covariance.

    Args:
        sincos (array):  The pre-computed sin / cos terms.
        detdata (array):  The detector data for one detector.
        flags (array):  The detector flags.
        guess (array):  Starting guess for the coefficients
        xtol (float):  The xtol parameter passed to scipy.optimize.least_squares
        gtol (float):  The gtol parameter passed to scipy.optimize.least_squares

    Returns:
        (array):  The coefficients.

    """
    n_samp = len(detdata)
    n_harmonics = sincos.shape[1] // 2

    good = flags == 0
    n_good = np.count_nonzero(good)
    n_coeff = 2 * n_harmonics

    if n_good == 0:
        # No good data, return zeros
        return np.zeros(n_coeff, dtype=np.float64)

    def _func_stepwise(x, *args, **kwargs):
        """Function to compute the current model residuals."""
        cur_model = hwpss_build_model(sincos, flags, x)
        resid = cur_model[:] - detdata[:]
        return resid

    def _jac_stepwise(x, *args, **kwargs):
        """Return Jacobian (partial derivatives) of the model."""
        J = np.zeros((n_samp, x.size))

        # Partial derivative with respect to harmonic terms.  The function is
        # linear in these coefficients and so the derivative is just the sin / cos
        # quantities.
        for h in range(n_harmonics):
            J[good, 2 * h] = sincos[good, 2 * h]
            J[good, 2 * h + 1] = sincos[good, 2 * h + 1]
        return J

    if guess is not None and len(guess) == n_coeff:
        x_0 = guess.copy()
    else:
        x_0 = np.zeros(n_coeff)
    result = least_squares(
        _func_stepwise,
        x_0,
        jac=_jac_stepwise,
        xtol=xtol,
        gtol=gtol,
        verbose=0,
        method="trf",
        tr_solver="exact",
    )
    coeff = np.array(result.x)

    # Scale result based on the number of good samples
    coeff[:] *= n_samp / n_good
    return coeff


def hwpss_build_model_step2f(sincos, flags, coeff):
    """Construct the HWPSS template from coefficients.

    The 2F coefficients are represented by step-wise values.

    Args:
        sincos (array):  The pre-computed sin / cos terms.
        flags (array):  The flags indicating bad angle samples
        coeff (array):  The model coefficents for this detector.

    Returns:
        (array):  The template.

    """
    n_samp = sincos.shape[0]
    n_harmonics = sincos.shape[1] // 2
    n_steps = (len(coeff) - 2 * (n_harmonics - 1)) // 2

    # We assign any "leftover" samples to the final step
    step_samples = n_samp // n_steps

    good = flags == 0
    out = np.zeros(n_samp, dtype=np.float64)
    if np.count_nonzero(good) == 0:
        # No good samples
        return out

    # Accumulate the first harmonic
    out[good] += coeff[0] * sincos[good, 0]
    out[good] += coeff[1] * sincos[good, 1]

    # Accumulate the second harmonic.  We project the stepwise coefficients.
    step_good = np.zeros_like(good)
    for stp in range(n_steps):
        off = stp * step_samples
        if stp == n_steps - 1:
            slc = slice(off, n_samp, 1)
        else:
            slc = slice(off, off + step_samples, 1)
        step_good[:] = False
        step_good[slc] = good[slc]
        out[step_good] += coeff[2 + 2 * stp] * sincos[step_good, 2]
        out[step_good] += coeff[2 + 2 * stp + 1] * sincos[step_good, 3]

    # Accumulate the remaining harmonics
    off = 2 + 2 * n_steps
    for h in range(n_harmonics - 2):
        out[good] += coeff[off + 2 * h] * sincos[good, 2 * (h + 2)]
        out[good] += coeff[off + 2 * h + 1] * sincos[good, 2 * (h + 2) + 1]
    return out


def hwpss_compute_coeff_step2f(
    sincos, detdata, flags, step_size=None, guess=None, xtol=1.0e-10, gtol=1.0e-10
):
    """Compute the HWPSS model coefficients.

    See docstring for `hwpss_compute_coeff`.  This function is similar
    but models the 2F coefficients as piecewise steps.

    Args:
        sincos (array):  The pre-computed sin / cos terms.
        detdata (array):  The detector data for one detector.
        flags (array):  The detector flags.
        step_size (int):  The number of samples per step.  If None, use
            one step for the whole data length.
        guess (array):  Starting guess for the coefficients
        xtol (float):  The xtol parameter passed to scipy.optimize.least_squares
        gtol (float):  The gtol parameter passed to scipy.optimize.least_squares

    Returns:
        (array):  The coefficients.

    """
    n_samp = len(detdata)
    n_harmonics = sincos.shape[1] // 2

    good = flags == 0
    n_good = np.count_nonzero(good)

    # We assign any "leftover" samples to the final step
    if step_size is None:
        step_size = n_samp
    n_steps = n_samp // step_size
    n_coeff = 2 * n_harmonics + 2 * (n_steps - 1)

    if n_good == 0:
        # No good data, return zeros
        return np.zeros(n_coeff, dtype=np.float64)

    def _func_stepwise(x, *args, **kwargs):
        """Function to compute the current model residuals."""
        cur_model = hwpss_build_model_step2f(sincos, flags, x)
        resid = cur_model[:] - detdata[:]
        return resid

    def _jac_stepwise(x, *args, **kwargs):
        """Return Jacobian (partial derivatives) of the model."""
        J = np.zeros((n_samp, x.size))

        # Partial derivative with respect to 1F coefficients.  These are just the
        # sin / cos quantities.
        J[good, 0] = sincos[good, 0]
        J[good, 1] = sincos[good, 1]

        # Partial derivative of the 2F coefficients.  These are the sin / cos terms
        # for the current step and zero otherwise.
        step_good = np.zeros_like(good)
        for stp in range(n_steps):
            off = stp * step_size
            if stp == n_steps - 1:
                slc = slice(off, n_samp, 1)
            else:
                slc = slice(off, off + step_size, 1)
            step_good[:] = False
            step_good[slc] = good[slc]
            J[step_good, 2 + 2 * stp] = sincos[step_good, 2]
            J[step_good, 2 + 2 * stp + 1] = sincos[step_good, 3]

        # Partial derivative with respect to higher harmonics.  The function is
        # linear in these coefficients and so the derivative is just the sin / cos
        # quantities.
        off = 2 + 2 * n_steps
        for h in range(n_harmonics - 2):
            J[good, off + 2 * h] = sincos[good, 2 * (h + 2)]
            J[good, off + 2 * h + 1] = sincos[good, 2 * (h + 2) + 1]
        return J

    if guess is not None and len(guess) == n_coeff:
        x_0 = guess.copy()
    else:
        x_0 = np.zeros(n_coeff)
    result = least_squares(
        _func_stepwise,
        x_0,
        jac=_jac_stepwise,
        xtol=xtol,
        gtol=gtol,
        verbose=0,
        method="trf",
        tr_solver="exact",
    )
    coeff = np.array(result.x)

    # Scale result based on the number of good samples
    coeff[:] *= n_samp / n_good
    return coeff
