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

from ..mpi import get_world
from ..op import Operator
from ..timing import GlobalTimers
from ..utils import Logger, inplace_weighted_sum

if TYPE_CHECKING:
    from typing import List, Optional

    from .dist import Data
    from .mpi import Comm


def add_crosstalk_args(parser: 'argparse.ArgumentParser'):
    """Add crosstalk args to argparse.ArgumentParser object.

    Args:
      parser: 'argparse.ArgumentParser': argparse' argument parser

    Returns:
        None
    """
    parser.add_argument(
        "--crosstalk-matrix",
        type=Path,
        nargs='*',
        required=False,
        help="input path(s) to crosstalk matrix in HDF5 container.",
    )


class SimpleCrosstalkMatrix:
    """A thin crosstalk matrix class.

    Args:
      names: detector names.
      data: crosstalk matrix.
    The length of `names` should match the dimension of `data`,
    which should be a square array.

    This is a simple container storing the crosstalk matrix,
    but not generating it.
    """

    def __init__(
        self,
        names: "np.ndarray['S']",
        data: 'np.ndarray[np.float64]',
    ):
        self.names = names
        self.data = data

        # py37+: we keep the dataclass semantics here in anticipation to use it in the future
        self.__post_init__()

    def __post_init__(self):
        logger = Logger.get()
        try:
            names = self.names = np.asarray(self.names, dtype="S")
            data = self.data = np.asarray(self.data, dtype=np.float64)
        except ValueError as e:
            logger.critical('name and data has to be Numpy-array-like')
            raise e
        if data.ndim != 2:
            raise TypeError('data should be 2-dimensional')
        if names.ndim != 1:
            raise TypeError('names should be 1-dimensional')
        shape = data.shape
        if not (names.size == shape[0] == shape[1]):
            raise TypeError('The dimensions of names and/or data not matched.')

    @property
    def names_str(self) -> 'List[str]':
        """names in list of str."""
        return [name.decode() for name in self.names]

    @classmethod
    def load(cls, path: 'Path') -> 'SimpleCrosstalkMatrix':
        """Load from an HDF5 file.

        Args:
          path: 'Path': path to an HDF5 file containing the data and names.

        Returns:
            SimpleCrosstalkMatrix
        """
        with h5py.File(path, 'r') as f:
            names = f["names"][:]
            data = f["data"][:]
        return cls(names, data)

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

        Args:
          path: 'Path': output file path.

        Returns:
            None
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

    Args:
      n_crosstalk_matrices: total no. of crosstalk matrices
      crosstalk_matrices: the crosstalk matrices.
        In MPI case, this should holds only those matrices owned by a rank dictate
        by the condition `i % world_procs == world_rank`
      name: this name is used to save data in tod.cache, so better be unique from other cache

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
    ):
        self.n_crosstalk_matrices = n_crosstalk_matrices
        self.crosstalk_matrices = crosstalk_matrices
        self.name = name

        self.world_comm: 'Optional[Comm]'
        self.world_procs: 'int'
        self.world_rank: 'int'
        self.world_comm, self.world_procs, self.world_rank = get_world()

    @property
    def is_serial(self) -> 'bool':
        return self.world_procs == 1

    def _get_crosstalk_matrix(self, i: 'int') -> 'SimpleCrosstalkMatrix':
        """Get the i-th crosstalk matrix, used this with MPI only.

        Args:
          i: 'int': the i-th crosstalk matrix.

        Returns:
            SimpleCrosstalkMatrix
        """
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
        # logger.debug(f'crosstalk: Rank {self.world_rank} receives lengths {lengths}')

        # broadcast arrays
        if self.world_rank != rank_owner:
            n = lengths[0]
            name_len = lengths[1]
            names_int = np.empty(n * name_len, dtype=np.uint8)
            names = names_int.view(f'S{name_len}')
            data = np.empty((n, n), dtype=np.float64)
        self.world_comm.Bcast(names_int, root=rank_owner)
        # logger.debug(f'crosstalk: Rank {self.world_rank} receives names {names}')
        self.world_comm.Bcast(data, root=rank_owner)
        # logger.debug(f'crosstalk: Rank {self.world_rank} receives data {data}')

        logger.info(f'Obtained the {i + 1}-th crosstalk matrix from world-rank {rank_owner}.')
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
    ) -> 'OpCrosstalk':
        """Read crosstalk matri(x|ces) from HDF5 file(s).

        This holds only those matrices owned by a rank
        dictate by the condition `i % world_procs == world_rank`.

        Args:
          args: 'argparse.Namespace': namespace object from argparse' parser
          name: 'str':  (Default value = "crosstalk"): this name is used to
            save data in tod.cache, so better be unique from other cache

        Returns:

        """
        gt = GlobalTimers.get()
        gt.start("OpCrosstalk_read")

        _, world_procs, world_rank = get_world()

        paths = args.crosstalk_matrix

        N = len(paths)

        crosstalk_matrices = [
            SimpleCrosstalkMatrix.load(paths[i])
            for i in range(world_rank, N, world_procs)
        ]

        gt.stop("OpCrosstalk_read")

        return cls(N, crosstalk_matrices, name=name)

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
                    inplace_weighted_sum(row_global_total, row, *tods_list)
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
        crosstalk_name = self.name
        logger = Logger.get()
        gt = GlobalTimers.get()

        # loop over crosstalk matrices
        for idx_crosstalk_matrix in range(self.n_crosstalk_matrices):
            crosstalk_matrix = self._get_crosstalk_matrix(idx_crosstalk_matrix)
            names = crosstalk_matrix.names_str
            names_set = set(names)
            crosstalk_data = crosstalk_matrix.data
            n = crosstalk_data.shape[0]
            for obs_i, obs in enumerate(data.obs):
                tod = obs["tod"]
                comm = tod.grid_comm_col
                procs = tod.grid_size[0]
                rank = tod.grid_ranks[0]

                gt.start(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}")
                gt.start(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_1-check")

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

                gt.stop(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_1-check")
                gt.start(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_2-detector-lut")

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

                # np.testing.assert_array_equal(local_has_det, global_has_det[rank])
                del local_has_det
                tod.cache.destroy(f"{crosstalk_name}_local_has_det_{rank}")

                det_lut = {}
                for i in range(procs):
                    for j in range(n):
                        if global_has_det[i, j]:
                            det_lut[names[j]] = i
                del global_has_det, i, j
                tod.cache.destroy(f"{crosstalk_name}_global_has_det_{rank}")

                # logger.debug(f'Rank {rank} has detectors lookup table: {det_lut}')
                # for name in local_crosstalk_dets_set:
                #     if det_lut[name] != rank:
                #         raise RuntimeError(
                #             'Error in creating a lookup table '
                #             f'from detector name to rank: {det_lut}'
                #         )

                gt.stop(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_2-detector-lut")
                gt.start(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_3-mat-mul")

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
                    try:
                        rank_owner = det_lut[name]
                    # in the unlikely scenario this happened,
                    # it is likely the detector LUT is incorrect.
                    # check the code containing `det_lut` above
                    except KeyError:
                        raise RuntimeError(f'Detector look-up table cannot find rank-owner of {name}: det_lut = {det_lut}')
                    if n_local_dets > 0:
                        row_local_total[:] = 0.
                        row_local_weights[:] = row[local_det_idxs]
                        tods_list = [
                            tod.cache.reference(
                                f"{signal_name}_{names[local_det_idxs[i]]}"
                            )
                            for i in range(n_local_dets)
                        ]
                        inplace_weighted_sum(row_local_total, row_local_weights, *tods_list)
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

                gt.stop(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_3-mat-mul")
                gt.start(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_4-copy-inplace")

                for name in local_crosstalk_dets_set:
                    # overwrite it in-place
                    # not using tod.cache.put as that will destroy and create
                    tod.cache.reference(f"{signal_name}_{name}")[:] = \
                        tod.cache.reference(f"{crosstalk_name}_{name}")
                    tod.cache.destroy(f"{crosstalk_name}_{name}")

                gt.stop(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}_4-copy-inplace")
                gt.stop(f"OpCrosstalk_matrix-{idx_crosstalk_matrix}_observation-{obs_i}")

    def exec(
        self,
        data: 'Data',
        signal_name: 'str',
    ):
        """Apply crosstalk matrix on ToD in data.

        It is dispatched depending if MPI is used.

        Args:
          data: 'Data': the data to be exec on.
          signal_name: 'str': the name of the signal.

        Returns:

        """
        gt = GlobalTimers.get()
        gt.start("OpCrosstalk_exec")
        (
            self._exec_serial(data, signal_name)
        ) if self.is_serial else (
            self._exec_mpi(data, signal_name)
        )
        gt.stop("OpCrosstalk_exec")
