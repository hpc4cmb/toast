#!/usr/bin/env python3

from toast.mpi import MPI

import os

import numpy as np

from numba import njit, config

# This function mimics the pysm use of the inner nested njit call to
# compute_spdust_scaling_numba.  In particular, it calls a numpy.interp,
# which might be parallelized in external compiled code.
@njit
def inner_numba_numpy(xdata, ydata, x1, x2):
    # Calling numpy within this function triggers an error
    y1 = np.interp(x1, xdata, ydata)
    y2 = np.interp(x2, xdata, ydata)
    return (y1, y2)


@njit
def inner_numba_nonumpy(xdata, ydata, x1, x2):
    y1 = np.zeros_like(x1)
    y2 = np.zeros_like(x2)

    indx = 0
    for i, x in enumerate(x1):
        # find bounds
        while x < xdata[indx + 1]:
            indx += 1
        while x > xdata[indx + 1]:
            indx -= 1
        y1[i] = ydata[indx] + (x - xdata[indx]) * (ydata[indx + 1] - ydata[indx]) / (
            xdata[indx + 1] - xdata[indx]
        )
    indx = 0
    for i, x in enumerate(x2):
        # find bounds
        while x < xdata[indx + 1]:
            indx += 1
        while x > xdata[indx + 1]:
            indx -= 1
        y2[i] = ydata[indx] + (x - xdata[indx]) * (ydata[indx + 1] - ydata[indx]) / (
            xdata[indx + 1] - xdata[indx]
        )

    return (y1, y2)


# This function mimics the pysm use of a njit parallel calling another njit
# in compute_spdust_emission_pol_numba.
@njit(parallel=True)
def outer_numba(xdata, ydata):
    inp1 = [0.5 + np.arange(1000000.0) for x in range(10)]
    inp2 = [0.25 + np.arange(1000000.0) for x in range(10)]
    out = list()
    for x1, x2 in zip(inp1, inp2):
        # Calling this with numpy causes OMP issue
        # y1, y2 = inner_numba_numpy(xdata, ydata, x1, x2)
        # Calling version without numpy works
        y1, y2 = inner_numba_nonumpy(xdata, ydata, x1, x2)
        out.append((y1, y2))
    return


@njit(parallel=True)
def outer(xdata, ydata):
    inp1 = [0.5 + np.arange(1000000.0) for x in range(10)]
    inp2 = [0.25 + np.arange(1000000.0) for x in range(10)]
    out = list()
    for x1, x2 in zip(inp1, inp2):
        y1 = np.zeros_like(x1)
        y2 = np.zeros_like(x2)
        indx = 0
        for i, x in enumerate(x1):
            # find bounds
            while x < xdata[indx + 1]:
                indx += 1
            while x > xdata[indx + 1]:
                indx -= 1
            y1[i] = ydata[indx] + (x - xdata[indx]) * (
                ydata[indx + 1] - ydata[indx]
            ) / (xdata[indx + 1] - xdata[indx])
        indx = 0
        for i, x in enumerate(x2):
            # find bounds
            while x < xdata[indx + 1]:
                indx += 1
            while x > xdata[indx + 1]:
                indx -= 1
            y2[i] = ydata[indx] + (x - xdata[indx]) * (
                ydata[indx + 1] - ydata[indx]
            ) / (xdata[indx + 1] - xdata[indx])
        out.append((y1, y2))
    return


def test_numba(comm):
    xdata = np.arange(1000001.0)
    ydata = 0.001 * np.arange(1000001.0)
    out = outer_numba(xdata, ydata)
    print(
        "test_numba post:  NUMBA_NUM_THREADS = {} / {}"
        .format(os.environ["NUMBA_NUM_THREADS"], config.NUMBA_NUM_THREADS),
        flush=True
    )
    # out = outer(xdata, ydata)


def main():
    print(
        "main:  NUMBA_NUM_THREADS = {} / {}"
        .format(os.environ["NUMBA_NUM_THREADS"], config.NUMBA_NUM_THREADS),
        flush=True
    )
    comm = None
    if MPI is not None:
        comm = MPI.COMM_WORLD
    test_numba(comm)


if __name__ == "__main__":
    main()
