# py37+
# from __future__ import annotations

"""Simulate crosstalk between detectors in ToD.

A typical use of this is that in your pipeline script,

1. Add `add_crosstalk_args(parser)` to your argparse parser to add cli args,
2. Read & create `op_crosstalk` by adding


    .. highlight:: python
    .. code-block:: python

        if args.crosstalk_matrix is not None:
            op_crosstalk = OpCrosstalk.read(args)

3. Apply the crosstalk matrix to your data after the ToD is prepared
(e.g. signal & noise) by adding


    .. highlight:: python
    .. code-block:: python

        if args.crosstalk_matrix is not None:
            if comm.comm_world is not None:
                comm.comm_world.barrier()
            op_crosstalk.exec(data, "tot_signal")
            if comm.comm_world is not None:
                comm.comm_world.barrier()

Note that you need to create your crosstalk matri(x|ces)
in HDF5 container(s) beforehand. A simple class `SimpleCrosstalkMatrix`
can assist you to write such file,
with the simple conventions that
it has the following datasets:

1. names: ASCII names of your detectors, say of length ``n``
2. data: ``n`` by ``n`` float64 array of your crosstalk matrix ``m``,
where you expect

    ToD_crosstalked = m @ ToD_original

    (Assuming row-major as is standard in Python.)

Note that each crosstalk matrix is dense,
and when multiple matrices are supplied, they are effectively block-diagonal
(where each given dense matrix is a block.)
I.e. in order to provide a block-diagonal crosstalk matrix,
each block (together with the names of the detectors in that block)
should write to a seperate HDF5 file to avoid unnessary computation.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

try:
    from numba import jit
except ImportError:
    jit = None

from ..mpi import get_world
from ..op import Operator
from ..utils import Logger

if TYPE_CHECKING:
    from typing import List, Union, Optional

    from .mpi import Comm
    from .dist import Data


def _fma(
    out: 'np.ndarray[np.float64]',
    weights: 'np.ndarray[np.float64]',
    *arrays: 'np.ndarray[np.float64]',
):
    """Simple fused multiply–add, compiled to avoid Python memory implications.

    :param out: must be zero array in the same shape of each in `arrays`

    If not compiled, a lot of Python objects will be created,
    and as the Python garbage collector is inefficient,
    it would have larger memory footprints.
    """
    for weight, array in zip(weights, arrays):
        out += weight * array


if jit is None:
    Logger.get().warning(
        'Numba not present. '
        '_fma in crosstalk will have more intermediate Numpy array objects '
        'created that uses more memory.'
    )
else:
    # cache is False to avoid IO on HPC.
    _fma = jit(_fma, nopython=True, nogil=True, parallel=True, cache=False)


def add_crosstalk_args(parser: 'argparse.ArgumentParser'):
    """Add crosstalk args to argparse.ArgumentParser object."""
    parser.add_argument(
        "--crosstalk-matrix",
        type=Path,
        nargs='*',
        required=False,
        help="input path(s) to crosstalk matrix in HDF5 container.",
    )
    parser.add_argument(
        "--crosstalk-matrix-debug",
        action='store_true',
        required=False,
        help="if specified, perform more checks and emit more messages. "
             "You may want to set TOAST_LOGLEVEL=DEBUG as well.",
    )


class SimpleCrosstalkMatrix:
    """A thin crosstalk matrix class.

    :param names: detector names.
    :param data: crosstalk matrix.
        The length of `names` should match the dimension of `data`,
        which should be a square array.
    :param debug: emit more debug message and perform runtime validation check.

    This is a simple container storing the crosstalk matrix,
    but not generating it.
    """

    def __init__(
        self,
        names: "Union[np.ndarray['S'], List[str]]",
        data: 'np.ndarray[np.float64]',
        *,
        debug: 'bool' = False,
    ):
        self.names = np.asarray(names, dtype="S")
        self.data = data
        self.debug = debug
        if debug:
            self.__post_init__()

    def __post_init__(self):
        names = self.names
        data = self.data
        try:
            if not isinstance(names.dtype, type(np.dtype('S'))):
                raise TypeError('names has to be a numpy array with dtype("S")')
            if not isinstance(
                data.dtype,
                (
                    type(np.dtype(np.float64)),
                    type(np.dtype(np.float32)),
                )
            ):
                raise TypeError('data has to be a numpy array of float')
            if data.ndim != 2:
                raise TypeError('data should be 2-dimensional')
            if names.ndim != 1:
                raise TypeError('names should be 1-dimensional')
            shape = data.shape
            if not (names.size == shape[0] == shape[1]):
                raise TypeError('The dimensions of names and/or data not matched.')
        except AttributeError:
            raise TypeError('name and data has to be numpy arrays')

    @property
    def names_str(self) -> 'List[str]':
        """names in list of str."""
        return [name.decode() for name in self.names]

    @classmethod
    def load(cls, path: 'Path', *, debug: 'bool' = False):
        """Load from an HDF5 file."""
        with h5py.File(path, 'r') as f:
            names = f["names"][:]
            data = f["data"][:]
        return cls(names, data, debug=debug)

    def dump(
        self,
        path: 'Path',
        *,
        libver: 'str' = 'latest',
        # h5py.File.create_dataset args
        compression: 'str' = 'gzip',
        shuffle: 'bool' = True,
        fletcher32: 'bool' = True,
        track_times: 'bool' = False,
        compression_opts: 'int' = 9,
        **kwargs,
    ):
        """Dump to an HDF5 file.

        `libver` will be passed to h5py.File,
        and the rest keyword arguments will be passed to `h5py.File.create_dataset`
        with sensible defaults.
        """
        with h5py.File(path, 'w', libver=libver) as f:
            f.create_dataset(
                'names',
                data=self.names,
                compression=compression,
                shuffle=shuffle,
                fletcher32=fletcher32,
                track_times=track_times,
                compression_opts=compression_opts,
                **kwargs,
            )
            f.create_dataset(
                'data',
                data=self.data,
                compression=compression,
                shuffle=shuffle,
                fletcher32=fletcher32,
                track_times=track_times,
                compression_opts=compression_opts,
                **kwargs,
            )


class OpCrosstalk(Operator):
    """Operator that applies crosstalk matrix to detector ToDs.

    :param n_crosstalk_matrices: total no. of crosstalk matrices
    :param crosstalk_matrices: the crosstalk matrices.
        In MPI case, this should holds only those matrices owned by a rank dictate
        by the condition `i % world_procs == world_rank`
    :param name: this name is used to save data in tod.cache, so better be unique from other cache

    In a typical scenario, the classmethod `read` is used to create an object
    instead of initiating directly. This classmethod guarantees the condition above holds.

    Note that in the MPI case, the matrices are distributed in the world communicator, while the
    `exec` method will uses the grid commuicator specified per ToD. This is because at the time
    of initiating, we have no knowledge on how the data is going to be distributed yet,
    nor how that is "partitioned" in relation with multiple crosstalk matrix files given.
    These 2 processes (of finding the crosstalk matrix, and finding the ToD) are orthogonal though
    so the operator is completely general.
    """

    def __init__(
        self,
        n_crosstalk_matrices: 'int',
        crosstalk_matrices: 'List[SimpleCrosstalkMatrix]',
        *,
        name: 'str' = "crosstalk",
        debug: 'bool' = False,
    ):
        self.n_crosstalk_matrices = n_crosstalk_matrices
        self.crosstalk_matrices = crosstalk_matrices
        self.name = name
        self.debug = debug

        self.world_comm: 'Optional[Comm]'
        self.world_procs: 'int'
        self.world_rank: 'int'
        self.world_comm, self.world_procs, self.world_rank = get_world()

    @property
    def is_serial(self) -> 'bool':
        return self.world_procs == 1

    def _get_crosstalk_matrix(self, i: 'int') -> 'SimpleCrosstalkMatrix':
        """Get the i-th crosstalk matrix, used this with MPI only.
        """
        debug = self.debug
        logger = Logger.get()

        rank_owner = i % self.world_procs
        # index of the i-th matrix in the local rank
        idx = i // self.world_procs

        if self.world_rank == rank_owner:
            crosstalk_matrix = self.crosstalk_matrices[idx]

            names = crosstalk_matrix.names
            data = crosstalk_matrix.data

            # cast to int for boardcasting
            names_int = names.view(np.uint8)
            # the data from HDF5 is already float64
            # this is needed for comm.Bcast below
            data = data.view(np.float64)

        # prepare lengths for creating arrays
            lengths = np.array([names.size, names.dtype.itemsize], dtype=np.int64)
        else:
            lengths = np.empty(2, dtype=np.int64)
        self.world_comm.Bcast(lengths, root=rank_owner)
        if debug:
            logger.debug(f'crosstalk: Rank {self.world_rank} receives lengths {lengths}')

        # broadcast arrays
        if self.world_rank != rank_owner:
            n = lengths[0]
            name_len = lengths[1]
            names_int = np.empty(n * name_len, dtype=np.uint8)
            names = names_int.view(f'S{name_len}')
            data = np.empty((n, n), dtype=np.float64)
        self.world_comm.Bcast(names_int, root=rank_owner)
        if debug:
            logger.debug(f'crosstalk: Rank {self.world_rank} receives names {names}')
        self.world_comm.Bcast(data, root=rank_owner)
        if debug:
            logger.debug(f'crosstalk: Rank {self.world_rank} receives data {data}')

        return (
            crosstalk_matrix
        ) if self.world_rank == rank_owner else (
            SimpleCrosstalkMatrix(names, data)
        )

    @classmethod
    def read(
        cls,
        args: 'argparse.Namespace',
        *,
        name: 'str' = "crosstalk",
        debug: 'bool' = False,
    ) -> 'OpCrosstalk':
        """Read crosstalk matri(x|ces) from HDF5 file(s).

        This holds only those matrices owned by a rank
        dictate by the condition `i % world_procs == world_rank`.
        """
        _, world_procs, world_rank = get_world()

        paths = args.crosstalk_matrix
        debug = args.crosstalk_matrix_debug

        N = len(paths)

        crosstalk_matrices = [
            SimpleCrosstalkMatrix.load(paths[i], debug=debug)
            for i in range(world_rank, N, world_procs)
        ]

        return cls(N, crosstalk_matrices, name=name, debug=debug)

    def _exec_serial(
        self,
        data: 'Data',
        signal_name: 'str',
    ):
        """Apply crosstalk matrix on ToD in data serially."""
        crosstalk_name = self.name
        logger = Logger.get()

        # loop over crosstalk matrices
        for crosstalk_matrix in self.crosstalk_matrices:
            names = crosstalk_matrix.names_str
            names_set = set(names)
            crosstalk_data = crosstalk_matrix.data
            for obs in data.obs:
                tod = obs["tod"]

                detectors_set = set(tod.detectors)
                if not (names_set & detectors_set):
                    logger.info(
                        f"Crosstalk: skipping tod {tod} as "
                        "it does not include detectors from crosstalk matrix "
                        f"with these detectors: {names}."
                    )
                    continue
                elif not (names_set <= detectors_set):
                    raise ValueError(
                        f"Crosstalk: tod {tod} only include some detectors "
                        "from the crosstalk matrix "
                        f"with these detectors: {names}."
                    )
                del detectors_set

                n_samples = tod.local_samples[1]

                # mat-mul
                # This follows the _exec_mpi mat-mul algorithm
                # but not put them in a contiguous array and use real mat-mul @
                # The advantage is to reduce memory use
                # (if creating an intermediate contiguous array
                # that would requires one more copy of tod than needed below)
                # and perhaps served as a easier-to-understand version of _exec_mpi below
                for name, row in zip(names, crosstalk_data):
                    row_global_total = tod.cache.create(
                        f"{crosstalk_name}_{name}",
                        np.float64,
                        (n_samples,),
                    )
                    tods_list = [
                        tod.cache.reference(f"{signal_name}_{name_j}")
                        for name_j in names
                    ]
                    _fma(row_global_total, row, *tods_list)
                for name in names:
                    # overwrite it in-place
                    # not using tod.cache.put as that will destroy and create
                    tod.cache.reference(f"{signal_name}_{name}")[:] = \
                        tod.cache.reference(f"{crosstalk_name}_{name}")
                    tod.cache.destroy(f"{crosstalk_name}_{name}")

    def _exec_mpi(
        self,
        data: 'Data',
        signal_name: 'str',
    ):
        """Apply crosstalk matrix on ToD in data with MPI."""
        debug = self.debug
        crosstalk_name = self.name
        logger = Logger.get()

        # loop over crosstalk matrices
        for idx_crosstalk_matrix in range(self.n_crosstalk_matrices):
            crosstalk_matrix = self._get_crosstalk_matrix(idx_crosstalk_matrix)
            names = crosstalk_matrix.names_str
            names_set = set(names)
            crosstalk_data = crosstalk_matrix.data
            n = crosstalk_data.shape[0]
            for obs in data.obs:
                tod = obs["tod"]
                comm = tod.grid_comm_col
                procs = tod.grid_size[0]
                rank = tod.grid_ranks[0]

                # all ranks need to check this as they need to perform the same action
                detectors_set = set(tod.detectors)
                if not (names_set & detectors_set):
                    logger.info(
                        f"Crosstalk: skipping tod {tod} as "
                        "it does not include detectors from crosstalk matrix "
                        f"with these detectors: {names}."
                    )
                    continue
                elif not (names_set <= detectors_set):
                    raise ValueError(
                        f"Crosstalk: tod {tod} only include some detectors "
                        f"from the crosstalk matrix with these detectors: {names}."
                    )
                del detectors_set

                n_samples = tod.local_samples[1]
                local_crosstalk_dets_set = set(tod.local_dets) & names_set
                n_local_dets = len(local_crosstalk_dets_set)

                # this is easier to understand and shorter
                # but uses allgather instead of the more efficient Allgather
                # construct detector lookup table
                # local_dets = tod.local_dets
                # global_dets = comm.allgather(local_dets)
                # det_lut = {}
                # for i, dets in enumerate(global_dets):
                #     for det in dets:
                #         det_lut[det] = i
                # log.debug(f'dets lookup table: {dets_lut}')

                # construct det_lut, a lookup table to know which rank holds a detector
                local_has_det = tod.cache.create(
                    f"{crosstalk_name}_local_has_det_{rank}",
                    np.uint8,
                    (n,),
                ).view(np.bool_)
                for i, name in enumerate(names):
                    if name in local_crosstalk_dets_set:
                        local_has_det[i] = True

                global_has_det = tod.cache.create(
                    f"{crosstalk_name}_global_has_det_{rank}",
                    np.uint8,
                    (procs, n),
                ).view(np.bool_)
                comm.Allgather(local_has_det, global_has_det)

                if debug:
                    np.testing.assert_array_equal(local_has_det, global_has_det[rank])
                del local_has_det
                tod.cache.destroy(f"{crosstalk_name}_local_has_det_{rank}")

                det_lut = {}
                for i in range(procs):
                    for j in range(n):
                        if global_has_det[i, j]:
                            det_lut[names[j]] = i
                del global_has_det, i, j
                tod.cache.destroy(f"{crosstalk_name}_global_has_det_{rank}")

                if debug:
                    logger.debug(f'Rank {rank} has detectors lookup table: {det_lut}')
                    for name in local_crosstalk_dets_set:
                        if det_lut[name] != rank:
                            raise RuntimeError(
                                'Error in creating a lookup table '
                                f'from detector name to rank: {det_lut}'
                            )

                # mat-mul
                row_local_total = tod.cache.create(
                    f"{crosstalk_name}_row_local_total_{rank}",
                    np.float64,
                    (n_samples,),
                )
                if n_local_dets > 0:
                    row_local_weights = tod.cache.create(
                        f"{crosstalk_name}_row_local_weights_{rank}",
                        np.float64,
                        (n_local_dets,),
                    )
                    local_det_idxs = tod.cache.create(
                        f"{crosstalk_name}_local_det_idxs_{rank}",
                        np.int64,
                        (n_local_dets,),
                    )
                for i, name in enumerate(local_crosstalk_dets_set):
                    local_det_idxs[i] = names.index(name)
                # row-loop
                # * potentially the tod can have more detectors
                # * than SimpleCrosstalkMatrix.names_str has
                # * and they will be skipped
                for name, row in zip(names, crosstalk_data):
                    rank_owner = det_lut[name]
                    if n_local_dets > 0:
                        row_local_total[:] = 0.
                        row_local_weights[:] = row[local_det_idxs]
                        tods_list = [
                            tod.cache.reference(
                                f"{signal_name}_{names[local_det_idxs[i]]}"
                            )
                            for i in range(n_local_dets)
                        ]
                        _fma(row_local_total, row_local_weights, *tods_list)
                    if rank == rank_owner:
                        row_global_total = tod.cache.create(
                            f"{crosstalk_name}_{name}",
                            np.float64,
                            (n_samples,),
                        )
                        comm.Reduce(
                            row_local_total,
                            row_global_total,
                            root=rank_owner,
                        )
                        # it is reduced into tod.cache and
                        # the python reference can be safely deleted
                        del row_global_total
                    else:
                        comm.Reduce(row_local_total, None, root=rank_owner)
                del det_lut, row_local_total, name, row, rank_owner
                tod.cache.destroy(f"{crosstalk_name}_row_local_total_{rank}")
                if n_local_dets > 0:
                    del row_local_weights, local_det_idxs, tods_list
                    tod.cache.destroy(f"{crosstalk_name}_row_local_weights_{rank}")
                    tod.cache.destroy(f"{crosstalk_name}_local_det_idxs_{rank}")

                for name in local_crosstalk_dets_set:
                    # overwrite it in-place
                    # not using tod.cache.put as that will destroy and create
                    tod.cache.reference(f"{signal_name}_{name}")[:] = \
                        tod.cache.reference(f"{crosstalk_name}_{name}")
                    tod.cache.destroy(f"{crosstalk_name}_{name}")

    def exec(
        self,
        data: 'Data',
        signal_name: 'str',
    ):
        """Apply crosstalk matrix on ToD in data.

        It is dispatched depending if MPI is used."""
        (
            self._exec_serial(data, signal_name)
        ) if self.is_serial else (
            self._exec_mpi(data, signal_name)
        )
