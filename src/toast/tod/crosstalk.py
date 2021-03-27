from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import toast
from numba import jit
from toast.mpi import get_world
from toast.op import Operator
from toast.utils import Logger

if TYPE_CHECKING:
    from typing import List

COMM: toast.mpi.Comm
PROCS: int
RANK: int
COMM, PROCS, RANK = get_world()
LOGGER = Logger.get()
IS_SERIAL = COMM is None

H5_CREATE_KW = {
    'compression': 'gzip',
    # shuffle minimize the output size
    'shuffle': True,
    # checksum for data integrity
    'fletcher32': True,
    # turn off track_times so that identical output gives the same md5sum
    'track_times': False
}


@jit(nopython=True, nogil=True, cache=False)
def fma(out: np.ndarray[np.float64], ws: np.ndarray[np.float64], *arrays: np.ndarray[np.float64]):
    """Simple FMA, compiled to avoid Python memory implications.

    :param out: must be zero array in the same shape of each array in `arrays`

    cache is False to avoid IO on HPC.

    If not compiled, a lot of Python objects will be created,
    and as the Python garbage collector is inefficient,
    it would have larger memory footprints.
    """
    for w, array in zip(ws, arrays):
        out += w * array


def add_crosstalk_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--crosstalk-matrix",
        type=Path,
        nargs='*',
        required=False,
        help="input path(s) to crosstalk matrix in HDF5 container.",
    )


@dataclass
class SimpleCrosstalkMatrix:
    """A thin crosstalk matrix class.

    For feature-rich crosstalk matrix class, see `coscon.toast_helper.CrosstalkMatrix`.
    """
    names: np.ndarray['S']
    data: np.ndarray[np.float64]

    @property
    def names_str(self) -> List[str]:
        """names in list of str"""
        return [name.decode() for name in self.names]

    @classmethod
    def load(cls, path: Path):
        with h5py.File(path, 'r') as f:
            names = f["names"][:]
            data = f["data"][:]
        return cls(names, data)

    def dump(self, path: Path, compress_level: int = 9):
        with h5py.File(path, 'w', libver='latest') as f:
            f.create_dataset(
                'names',
                data=self.names,
                compression_opts=compress_level,
                **H5_CREATE_KW
            )
            f.create_dataset(
                'data',
                data=self.data,
                compression_opts=compress_level,
                **H5_CREATE_KW
            )


@dataclass
class OpCrosstalk(Operator):
    """Operator that apply crosstalk matrix to detector ToDs.
    """
    # total no. of crosstalk matrices
    n_crosstalk_matrices: int
    # in MPI case, this holds only those matrices owned by a rank
    # dictate by the condition i % PROCS == RANK
    crosstalk_matrices: List[SimpleCrosstalkMatrix]
    # this name is used to save data in tod.cache, so better be unique from other cache
    name: str = "crosstalk"

    def _get_crosstalk_matrix(self, i: int) -> SimpleCrosstalkMatrix:
        """Get the i-th crosstalk matrix, used this with MPI only.
        """
        rank_owner = i % PROCS
        # index of the i-th matrix in the local rank
        idx = i // PROCS

        if RANK == rank_owner:
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
        COMM.Bcast(lengths, root=rank_owner)
        LOGGER.debug(f'crosstalk: Rank {RANK} receives lengths {lengths}')

        # broadcast arrays
        if RANK != rank_owner:
            n = lengths[0]
            name_len = lengths[1]
            names_int = np.empty(n * name_len, dtype=np.uint8)
            names = names_int.view(f'S{name_len}')
            data = np.empty((n, n), dtype=np.float64)
        COMM.Bcast(names_int, root=rank_owner)
        LOGGER.debug(f'crosstalk: Rank {RANK} receives names {names}')
        COMM.Bcast(data, root=rank_owner)
        LOGGER.debug(f'crosstalk: Rank {RANK} receives data {data}')

        if RANK == rank_owner:
            return crosstalk_matrix
        else:
            return SimpleCrosstalkMatrix(names, data)

    @staticmethod
    def _read_serial(paths: List[Path]) -> List[SimpleCrosstalkMatrix]:
        return [SimpleCrosstalkMatrix.load(path) for path in paths]

    @staticmethod
    def _read_mpi(paths: List[Path]) -> List[SimpleCrosstalkMatrix]:
        N = len(paths)
        path_idxs_per_rank = range(RANK, N, PROCS)
        return [SimpleCrosstalkMatrix.load(paths[i]) for i in path_idxs_per_rank]

    @classmethod
    def read(
        cls,
        args: argparse.Namespace,
        name: str = "crosstalk",
    ) -> OpCrosstalk:
        paths = args.crosstalk_matrix
        crosstalk_matrices = cls._read_serial(paths) if IS_SERIAL else cls._read_mpi(paths)
        return cls(len(paths), crosstalk_matrices, name=name)

    def _exec_serial(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = False,
    ):
        crosstalk_name = self.name

        # loop over crosstalk matrices
        for crosstalk_matrix in self.crosstalk_matrices:
            names = crosstalk_matrix.names_str
            names_set = set(names)
            crosstalk_data = crosstalk_matrix.data
            for obs in data.obs:
                tod = obs["tod"]
                # TODO: should we only check this only `if debug`?
                detectors_set = set(tod.detectors)
                if not (names_set & detectors_set):
                    LOGGER.info(f"Crosstalk: skipping tod {tod} as it does not include detectors from crosstalk matrix with these detectors: {names}.")
                    continue
                elif not (names_set <= detectors_set):
                    raise ValueError(f"Crosstalk: tod {tod} only include some detectors from the crosstalk matrix with these detectors: {names}.")
                del detectors_set

                n_samples = tod.total_samples

                # mat-mul
                # This follows the _exec_mpi mat-mul algorithm
                # but not put them in a contiguous array and use real mat-mul @
                # The advantage is to reduce memory use
                # (if creating an intermediate contiguous array that would requires one more copy of tod then needed below)
                # and perhaps served as a easier-to-understand version of _exec_mpi below
                for name, row in zip(names, crosstalk_data):
                    row_global_total = tod.cache.create(f"{crosstalk_name}_{name}", np.float64, (n_samples,))
                    tods_list = [tod.cache.reference(f"{signal_name}_{name_j}") for name_j in names]
                    fma(row_global_total, row, *tods_list)
                for name in names:
                    # overwrite it in-place
                    # not using tod.cache.put as that will destroy and create
                    tod.cache.reference(f"{signal_name}_{name}")[:] = tod.cache.reference(f"{crosstalk_name}_{name}")
                    tod.cache.destroy(f"{crosstalk_name}_{name}")

    def _exec_mpi(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = False,
    ):
        crosstalk_name = self.name

        # loop over crosstalk matrices
        for idx_crosstalk_matrix in range(self.n_crosstalk_matrices):
            crosstalk_matrix = self._get_crosstalk_matrix(idx_crosstalk_matrix)
            names = crosstalk_matrix.names_str
            names_set = set(names)
            crosstalk_data = crosstalk_matrix.data
            n = crosstalk_data.shape[0]
            for obs in data.obs:
                tod = obs["tod"]
                # TODO: should we only check this only `if debug`?
                # all ranks need to check this as they need to perform the same action
                detectors_set = set(tod.detectors)
                if not (names_set & detectors_set):
                    LOGGER.info(f"Crosstalk: skipping tod {tod} as it does not include detectors from crosstalk matrix with these detectors: {names}.")
                    continue
                elif not (names_set <= detectors_set):
                    raise ValueError(f"Crosstalk: tod {tod} only include some detectors from the crosstalk matrix with these detectors: {names}.")
                del detectors_set

                n_samples = tod.total_samples
                local_crosstalk_dets_set = set(tod.local_dets) & names_set
                n_local_dets = len(local_crosstalk_dets_set)

                # this is easier to understand and shorter
                # but uses allgather instead of the more efficient Allgather
                # construct detector LUT
                # local_dets = tod.local_dets
                # global_dets = comm.allgather(local_dets)
                # det_lut = {}
                # for i, dets in enumerate(global_dets):
                #     for det in dets:
                #         det_lut[det] = i
                # log.debug(f'dets LUT: {dets_lut}')

                # construct det_lut, a LUT to know which rank holds a detector
                local_has_det = tod.cache.create(f"{crosstalk_name}_local_has_det_{RANK}", np.uint8, (n,)).view(np.bool_)
                for i, name in enumerate(names):
                    if name in local_crosstalk_dets_set:
                        local_has_det[i] = True

                global_has_det = tod.cache.create(f"{crosstalk_name}_global_has_det_{RANK}", np.uint8, (PROCS, n)).view(np.bool_)
                COMM.Allgather(local_has_det, global_has_det)

                if debug:
                    np.testing.assert_array_equal(local_has_det, global_has_det[RANK])
                del local_has_det
                tod.cache.destroy(f"{crosstalk_name}_local_has_det_{RANK}")

                det_lut = {}
                for i in range(PROCS):
                    for j in range(n):
                        if global_has_det[i, j]:
                            det_lut[names[j]] = i
                del global_has_det, i, j
                tod.cache.destroy(f"{crosstalk_name}_global_has_det_{RANK}")

                LOGGER.debug(f'Rank {RANK} has detectors LUT: {det_lut}')

                if debug:
                    for name in local_crosstalk_dets_set:
                        assert det_lut[name] == RANK

                # mat-mul
                row_local_total = tod.cache.create(f"{crosstalk_name}_row_local_total_{RANK}", np.float64, (n_samples,))
                if n_local_dets > 0:
                    row_local_weights = tod.cache.create(f"{crosstalk_name}_row_local_weights_{RANK}", np.float64, (n_local_dets,))
                    local_det_idxs = tod.cache.create(f"{crosstalk_name}_local_det_idxs_{RANK}", np.int64, (n_local_dets,))
                for i, name in enumerate(local_crosstalk_dets_set):
                    local_det_idxs[i] = names.index(name)
                # row-loop
                # * potentially the tod can have more detectors than SimpleCrosstalkMatrix.names_str has
                # * and they will be skipped
                for name, row in zip(names, crosstalk_data):
                    rank_owner = det_lut[name]
                    if n_local_dets > 0:
                        row_local_total[:] = 0.
                        row_local_weights[:] = row[local_det_idxs]
                        tods_list = [tod.cache.reference(f"{signal_name}_{names[local_det_idxs[i]]}") for i in range(n_local_dets)]
                        fma(row_local_total, row_local_weights, *tods_list)
                    if RANK == rank_owner:
                        row_global_total = tod.cache.create(f"{crosstalk_name}_{name}", np.float64, (n_samples,))
                        COMM.Reduce(row_local_total, row_global_total, root=rank_owner)
                        # it is reduced into tod.cache and the python reference can be safely deleted
                        del row_global_total
                    else:
                        COMM.Reduce(row_local_total, None, root=rank_owner)
                del det_lut, row_local_total, name, row, rank_owner
                tod.cache.destroy(f"{crosstalk_name}_row_local_total_{RANK}")
                if n_local_dets > 0:
                    del row_local_weights, local_det_idxs, tods_list
                    tod.cache.destroy(f"{crosstalk_name}_row_local_weights_{RANK}")
                    tod.cache.destroy(f"{crosstalk_name}_local_det_idxs_{RANK}")

                for name in local_crosstalk_dets_set:
                    # overwrite it in-place
                    # not using tod.cache.put as that will destroy and create
                    tod.cache.reference(f"{signal_name}_{name}")[:] = tod.cache.reference(f"{crosstalk_name}_{name}")
                    tod.cache.destroy(f"{crosstalk_name}_{name}")

    def exec(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = False,
    ):
        self._exec_serial(data, signal_name, debug=debug) if IS_SERIAL else self._exec_mpi(data, signal_name, debug=debug)
