# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for starting up interactive sessions."""

import os

from IPython.core.extensions import ExtensionManager


def start_parallel(procs=1, threads=1, nice=True, auto_mpi=False, shell=None):
    """Attempt to start up ipyparallel with mpi4py and OpenMP.

    Args:
        procs (int):  The number of processes to use.
        threads (int):  The number of OpenMP threads per process.
        nice (bool):  If True, "nice" the processes to a lower
            priority.

    Returns:
        (int):  The number of processes started.

    """
    print(f"Using {procs} processes with {threads} threads each.")

    os.environ["OMP_NUM_THREADS"] = f"{threads}"

    if procs > 1:
        try:
            import ipyparallel as ipp

            ipp.bind_kernel()
            cluster = ipp.Cluster(engines="mpi", n=procs)
            client = cluster.start_and_connect_sync()
            client.block = True
            if nice:
                # Optionally nice the individual processes if running on a
                # shared node.
                if procs > 1:
                    import psutil

                    psutil.Process().nice(
                        20 if psutil.POSIX else psutil.IDLE_PRIORITY_CLASS
                    )
            # Optionally enable automatic use of MPI
            if auto_mpi and shell is not None:
                # Turn on automatic use of MPI
                shell.run_line_magic("autopx")
        except Exception:
            procs = 1
            print("Failed to start ipyparallel cluster, using one process.")
    extmanager = ExtensionManager()
    try:
        _ = extmanager.load_extension("wurlitzer")
    except Exception:
        # Must be running outside IPython shell
        pass
    return procs
