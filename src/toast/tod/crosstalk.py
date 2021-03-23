

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

H5_CREATE_KW = {
    'compression': 'gzip',
    # shuffle minimize the output size
    'shuffle': True,
    # checksum for data integrity
    'fletcher32': True,
    # turn off track_times so that identical output gives the same md5sum
    'track_times': False
}

if TYPE_CHECKING:
    from typing import Optional, Tuple, List


@jit(nopython=True, nogil=True, cache=False)
def fma(out, ws, *arrays):
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
        required=False,
        help="input path to crosstalk matrix in HDF5 container.",
    )
    parser.add_argument(
        "--crosstalk-write-tod-input",
        type=Path,
        required=False,
        help="output path to write TOD input. For debug only.",
    )
    parser.add_argument(
        "--crosstalk-write-tod-output",
        type=Path,
        required=False,
        help="output path to write TOD input. For debug only.",
    )


@dataclass
class OpCrosstalk(Operator):
    """Operator that apply crosstalk matrix to detector ToDs.
    """
    crosstalk_names: np.ndarray['S']
    crosstalk_data: np.ndarray[np.float64]
    crosstalk_write_tod_input_path: Optional[Path] = None
    crosstalk_write_tod_output_path: Optional[Path] = None
    name: str = "crosstalk"

    def __post_init__(self):
        self._name = self.name
        self.comm, self.procs, self.rank = get_world()
        self.log = Logger.get()

    @property
    def is_serial(self):
        return self.comm is None

    @property
    def crosstalk_names_str(self) -> List[str]:
        """names in list of str"""
        return [name.decode() for name in self.crosstalk_names]

    @staticmethod
    def read_serial(
        path: Path,
    ) -> Tuple[np.ndarray['S'], np.ndarray[np.float64]]:
        with h5py.File(path, 'r') as f:
            names = f["names"][:]
            data = f["data"][:]
        return names, data

    @staticmethod
    def read_mpi(
        path: Path,
        comm: toast.mpi.Comm,
        procs: int,
        rank: int,
    ) -> Tuple[np.ndarray['S'], np.ndarray[np.float64]]:
        log = Logger.get()
        comm, procs, rank = get_world()

        if rank == 0:
            names, data = OpCrosstalk.read_serial(path)
            lengths = np.array([names.size, names.dtype.itemsize], dtype=np.int64)
            # cast to int for boardcasting
            names_int = names.view(np.uint8)
            # the data from HDF5 is already float64
            # this is needed for comm.Bcast below
            data = data.view(np.float64)
        else:
            lengths = np.empty(2, dtype=np.int64)
        comm.Bcast(lengths, root=0)
        log.debug(f'crosstalk: Rank {rank} receives lengths {lengths}')
        if rank != 0:
            n = lengths[0]
            name_len = lengths[1]
            names_int = np.empty(n * name_len, dtype=np.uint8)
            data = np.empty((n, n), dtype=np.float64)
        comm.Bcast(names_int, root=0)
        if rank != 0:
            names = names_int.view(f'S{name_len}')
        log.debug(f'crosstalk: Rank {rank} receives names {names}')
        comm.Bcast(data, root=0)
        log.debug(f'crosstalk: Rank {rank} receives data {data}')
        return names, data

    @classmethod
    def read(
        cls,
        args: argparse.Namespace,
        name: str = "crosstalk",
    ) -> OpCrosstalk:
        path = args.crosstalk_matrix
        comm, procs, rank = get_world()
        names, data = cls.read_serial(path) if procs == 1 else cls.read_mpi(path, comm, procs, rank)
        return cls(
            names,
            data,
            crosstalk_write_tod_input_path=args.crosstalk_write_tod_input,
            crosstalk_write_tod_output_path=args.crosstalk_write_tod_output,
            name=name,
        )

    def get_tod_serial(
        self,
        tod: toast.tod.TOD,
        signal_name: str,
    ) -> np.ndarray[np.float64]:
        raise NotImplementedError

    def get_tod_mpi(
        self,
        tod: toast.tod.TOD,
        signal_name: str,
    ) -> Optional[np.ndarray[np.float64]]:
        """Obtain the TOD as a contiguous array.

        This is very inefficient as it is for debug only!
        """
        rank = self.rank
        comm = self.comm
        log = self.log
        names = self.crosstalk_names_str
        names_set = set(names)
        n = len(names)
        n_samples = tod.total_samples

        local_dets = tod.local_dets
        send_data = [(det, tod.cache.reference(f"{signal_name}_{det}")) for det in local_dets if det in names_set]
        log.debug(f"Rank {rank} collected local TOD from {local_dets}")
        if rank == 0:
            log.debug("Gathering TOD to root.")
        data = comm.gather(send_data, root=0)
        if rank == 0:
            log.debug("Gathered TOD to root, constructing dict")
            tod_dict = {}
            for datum in data:
                for name, t in datum:
                    tod_dict[name] = t
            # assume all names are found in tod!
            tod_array = np.array([tod_dict[name] for name in names])
            assert tod_array.shape == (n, n_samples)
            log.debug(f"TOD array constructed with shape {(n, n_samples)}.")
            return tod_array
        else:
            return None

    def save_tod_serial(
        self,
        path: Path,
        tod: toast.tod.TOD,
        signal_name: str,
    ):
        raise NotImplementedError

    def save_tod_mpi(
        self,
        path: Path,
        tod: toast.tod.TOD,
        signal_name: str,
        compress_level: int = 1,
    ):
        log = self.log
        rank = self.rank
        # non-root should have None
        tod_array = self.get_tod_mpi(
            tod,
            signal_name,
        )
        if rank == 0:
            log.debug(f"Writing TOD array to file {path}.")
            with h5py.File(path, 'w', libver='latest') as f:
                f.create_dataset(
                    'names',
                    data=self.crosstalk_names,
                    compression_opts=compress_level,
                    **H5_CREATE_KW
                )
                f.create_dataset(
                    'data',
                    data=tod_array,
                    compression_opts=compress_level,
                    **H5_CREATE_KW
                )

    def exec_serial(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = True,  # TODO
    ):
        raise NotImplementedError

    def exec_mpi(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = True,  # TODO
    ):
        log = self.log
        comm = self.comm
        procs = self.procs
        rank = self.rank
        crosstalk_name = self.name
        names = self.crosstalk_names_str
        crosstalk_data = self.crosstalk_data
        n = len(names)

        for obs in data.obs:
            tod = obs["tod"]

            if self.crosstalk_write_tod_input_path:
                if rank == 0:
                    log.warning(f"Saving input TOD to {self.crosstalk_write_tod_input_path}. You should only use it for debug only!")
                self.save_tod_mpi(
                    self.crosstalk_write_tod_input_path,
                    tod,
                    signal_name,
                )

            n_samples = tod.total_samples
            local_dets = tod.local_dets
            n_local_dets = len(local_dets)

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
            local_has_det = tod.cache.create(f"{crosstalk_name}_local_has_det_{rank}", np.uint8, (n,)).view(np.bool)
            local_dets_set = set(tod.local_dets)
            for i, name in enumerate(names):
                if name in local_dets_set:
                    local_has_det[i] = True
            del local_dets_set

            global_has_det = tod.cache.create(f"{crosstalk_name}_global_has_det_{rank}", np.uint8, (procs, n)).view(np.bool)
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
            del global_has_det
            tod.cache.destroy(f"{crosstalk_name}_global_has_det_{rank}")

            log.debug(f'Rank {rank} has detectors LUT: {det_lut}')

            if debug:
                for name in local_dets:
                    assert det_lut[name] == rank

            # mat-mul
            row_local_total = tod.cache.create(f"{crosstalk_name}_row_local_total_{rank}", np.float64, (n_samples,))
            row_local_weights = tod.cache.create(f"{crosstalk_name}_row_local_weights_{rank}", np.float64, (n_local_dets,))
            local_det_idxs = tod.cache.create(f"{crosstalk_name}_local_det_idxs_{rank}", np.int64, (n_local_dets,))
            for i, name in enumerate(local_dets):
                local_det_idxs[i] = names.index(name)
            # row-loop
            # potentially the tod can have more detectors than OpCrosstalk.crosstalk_names has
            # and they will be skipped
            for name, row in zip(names, crosstalk_data):
                rank_owner = det_lut[name]
                # assume each process must have at least one detector
                row_local_total[:] = 0.
                row_local_weights[:] = row[local_det_idxs]
                tods_list = [tod.cache.reference(f"{signal_name}_{names[local_det_idxs[i]]}") for i in range(n_local_dets)]
                fma(row_local_total, row_local_weights, *tods_list)
                if rank == rank_owner:
                    row_global_total = tod.cache.create(f"{crosstalk_name}_{name}", np.float64, (n_samples,))
                    comm.Reduce(row_local_total, row_global_total, root=rank_owner)
                else:
                    comm.Reduce(row_local_total, None, root=rank_owner)
            del row_local_total, row_local_weights, local_det_idxs, tods_list
            tod.cache.destroy(f"{crosstalk_name}_row_local_total_{rank}")
            tod.cache.destroy(f"{crosstalk_name}_row_local_weights_{rank}")
            tod.cache.destroy(f"{crosstalk_name}_local_det_idxs_{rank}")

            # overwrite original tod from cache
            for name in local_dets:
                tod.cache.destroy(f"{signal_name}_{name}")
                tod.cache.add_alias(f"{signal_name}_{name}", f"{crosstalk_name}_{name}")

            if self.crosstalk_write_tod_output_path:
                if rank == 0:
                    log.warning(f"Saving input TOD to {self.crosstalk_write_tod_output_path}. You should only use it for debug only!")
                self.save_tod_mpi(
                    self.crosstalk_write_tod_output_path,
                    tod,
                    signal_name,
                )

    def exec(
        self,
        data: toast.dist.Data,
        signal_name: str,
        debug: bool = True,  # TODO
    ):
        self.exec_serial(data, signal_name, debug=debug) if self.is_serial else self.exec_mpi(data, signal_name, debug=debug)
