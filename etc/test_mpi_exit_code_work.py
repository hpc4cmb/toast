#!/usr/bin/env python3

"""Test MPI exit code.

When running an MPI workflow, we want unhandled exceptions to trigger
an Abort on the world communicator.  This prevents stale processes
from hanging in a deadlock case.  This dummy script just raises
such an exception on the last process.

"""

import toast


def main():
    world, procs, rank = toast.mpi.get_world()
    if rank == procs - 1:
        msg = f"Testing unhandled exception on rank {rank}"
        raise RuntimeError(msg)
    if world is not None:
        # Wait here to be killed
        world.barrier()


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world, timeout=2):
        main()
