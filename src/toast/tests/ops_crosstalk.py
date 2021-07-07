# py37+
# from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng

from ..dist import Data
from ..mpi import get_world
from ..tod import TODCache
from ..tod.crosstalk import OpCrosstalk, SimpleCrosstalkMatrix
from ..utils import Logger
from ._helpers import create_outdir
from .mpi import MPITestCase

if TYPE_CHECKING:
    from typing import List


class FakeData(Data):
    def __init__(
        self,
        obs: "List[dict]",
    ):
        self.obs = obs


class OpCrosstalkTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = Path(create_outdir(self.comm, fixture_name))

        self.world_comm, self.world_procs, self.world_rank = get_world()
        log = Logger.get()

        # just to have more than 1 detectors per self.world_procs, and is irregular
        n_detectors = 4 * self.world_procs + 3
        self.n_samples = 100
        self.signal_name = "signal"

        self.detrankses = [1]
        for i in range(2, self.world_procs + 1):
            if self.world_procs % i == 0:
                self.detrankses.append(i)
        log.info(f'Testing OpCrosstalk against detranks {", ".join(map(str, self.detrankses))}.')

        self.names_str = [f"A{i}" for i in range(n_detectors)]
        self.names = np.array(self.names_str, dtype="S")
        self.names_2_str = [f"B{i}" for i in range(n_detectors)]
        self.names_2 = np.array(self.names_2_str, dtype="S")

        self.tod_array = np.arange(n_detectors * self.n_samples, dtype=np.float64).reshape(
            (n_detectors, self.n_samples)
        )
        self.crosstalk_data = np.arange(n_detectors * n_detectors, dtype=np.float64).reshape(
            (n_detectors, n_detectors)
        )
        self.tod_crosstalked = (
            np.arange(n_detectors * n_detectors).reshape((n_detectors, n_detectors))
            @ np.arange(n_detectors * self.n_samples).reshape((n_detectors, self.n_samples))
        ).astype(np.float64)

        rng = default_rng()
        self.tod_array_random = rng.standard_normal((n_detectors, self.n_samples))
        self.crosstalk_data_2 = np.identity(n_detectors) + np.reciprocal(
            np.arange(10, 10 + n_detectors * n_detectors)
        ).reshape((n_detectors, n_detectors))
        self.tod_crosstalked_random = self.crosstalk_data_2 @ self.tod_array_random

    def _each_op_crosstalk(
        self,
        tod_array: "np.ndarray[np.float64]",
        crosstalk_data: "np.ndarray[np.float64]",
        tod_crosstalked: "np.ndarray[np.float64]",
        detranks: "int",
    ):
        if self.world_rank == 0:
            crosstalk_matrices = [
                SimpleCrosstalkMatrix(self.names, crosstalk_data, debug=True)
            ]
        else:
            crosstalk_matrices = []
        op_crosstalk = OpCrosstalk(1, crosstalk_matrices, debug=True)

        tod = TODCache(self.world_comm, self.names_str, self.n_samples, detranks=detranks)

        # write local detectors
        start, length = tod.local_samples
        local_dets_set = set(tod.local_dets)
        for i, name in enumerate(self.names_str):
            if name in local_dets_set:
                tod.write(detector=name, data=tod_array[i, start:start + length])
        data = FakeData(
            [
                {
                    "tod": tod,
                }
            ]
        )
        op_crosstalk.exec(data, signal_name=self.signal_name)

        # check output local detectors
        # half float64 precision (float64 has 53-bit precison)
        # will not produce bit-identical output as mat-mul
        # is sensitive to the sum-ordering
        for i, name in enumerate(self.names_str):
            if name in local_dets_set:
                output = tod.cache.reference(f"{self.signal_name}_{name}")
                answer = tod_crosstalked[i, start:start + length]
                np.testing.assert_allclose(answer, output)

    def test_op_crosstalk(self):
        cases = []
        for detranks in self.detrankses:
            cases += [
                (
                    self.tod_array,
                    self.crosstalk_data,
                    self.tod_crosstalked,
                    detranks,
                ),
                (
                    self.tod_array_random,
                    self.crosstalk_data_2,
                    self.tod_crosstalked_random,
                    detranks,
                ),
            ]
        for case in cases:
            self._each_op_crosstalk(*case)

    def _each_op_crosstalk_multiple_matrices(
        self,
        names: "List[np.ndarray['S']]",
        names_strs: "List[List[str]]",
        tods_array: "List[np.ndarray[np.float64]]",
        crosstalk_datas: "List[np.ndarray[np.float64]]",
        tods_crosstalked: "List[np.ndarray[np.float64]]",
        detranks: "int",
    ):
        n_crosstalk_matrices = len(self.names)
        idxs_per_rank = range(self.world_rank, n_crosstalk_matrices, self.world_procs)
        crosstalk_matrices = [
            SimpleCrosstalkMatrix(self.names[i], crosstalk_datas[i], debug=True)
            for i in idxs_per_rank
        ]
        op_crosstalk = OpCrosstalk(n_crosstalk_matrices, crosstalk_matrices, debug=True)

        tod = TODCache(self.world_comm, sum(names_strs, []), self.n_samples, detranks=detranks)

        # write local detectors
        start, length = tod.local_samples
        local_dets_set = set(tod.local_dets)
        for self.names_str, tod_array in zip(names_strs, tods_array):
            for i, name in enumerate(self.names_str):
                if name in local_dets_set:
                    tod.write(detector=name, data=tod_array[i, start:start + length])
        data = FakeData(
            [
                {
                    "tod": tod,
                }
            ]
        )
        op_crosstalk.exec(data, signal_name=self.signal_name)

        # check output local detectors
        # half float64 precision (float64 has 53-bit precison)
        # will not produce bit-identical output as mat-mul
        # is sensitive to the sum-ordering
        for self.names_str, tod_crosstalked in zip(names_strs, tods_crosstalked):
            for i, name in enumerate(self.names_str):
                if name in local_dets_set:
                    output = tod.cache.reference(f"{self.signal_name}_{name}")
                    answer = tod_crosstalked[i, start:start + length]
                    np.testing.assert_allclose(answer, output)

    def test_op_crosstalk_multiple_matrices(self):
        cases = []
        for detranks in self.detrankses:
            cases.append(
                (
                    [self.names, self.names_2],
                    [self.names_str, self.names_2_str],
                    [self.tod_array, self.tod_array_random],
                    [self.crosstalk_data, self.crosstalk_data_2],
                    [self.tod_crosstalked, self.tod_crosstalked_random],
                    detranks,
                ),
            )
        for case in cases:
            self._each_op_crosstalk_multiple_matrices(*case)

    def test_op_crosstalk_io(self):
        if self.world_rank == 0:
            path = self.outdir / "simple_crosstalk_matrix.hdf5"
            crosstalk_matrix = SimpleCrosstalkMatrix(self.names, self.crosstalk_data, debug=True)

            crosstalk_matrix.dump(path)
            crosstalk_matrix_read = SimpleCrosstalkMatrix.load(path, debug=True)

            np.testing.assert_array_equal(self.names, crosstalk_matrix_read.names)
            np.testing.assert_array_equal(self.crosstalk_data, crosstalk_matrix_read.data)
