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
    from typing import List, Optional

    from .mpi import Comm


class FakeData(Data):
    def __init__(
        self,
        obs: "List[dict]",
    ):
        self.obs = obs


class OpCrosstalkTest(MPITestCase):
    def setUp(self):
        fixture_name: 'str' = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir: 'Path' = Path(create_outdir(self.comm, fixture_name))

        self.world_comm: 'Optional[Comm]'
        self.world_procs: 'int'
        self.world_rank: 'int'
        self.world_comm, self.world_procs, self.world_rank = get_world()

        logger = Logger.get()

        # just to have more than 1 detectors per self.world_procs, and is irregular
        n_detectors: 'int' = 4 * self.world_procs + 3
        self.n_samples: 'int' = 100
        self.signal_name: 'str' = "signal"

        self.detranks: 'List[int]' = [1]
        for i in range(2, self.world_procs + 1):
            if self.world_procs % i == 0:
                self.detranks.append(i)
        logger.info(f'Testing OpCrosstalk against detranks {", ".join(map(str, self.detranks))}.')

        self.names_strs: 'List[List[str]]' = []
        self.names: 'List[np.ndarray["S"]]' = []
        self.tod: 'List[np.ndarray[np.float64]]' = []
        self.data: 'List[np.ndarray[np.float64]]' = []
        self.tod_crosstalked: 'List[np.ndarray[np.float64]]' = []

        temp = [f"A{i}" for i in range(n_detectors)]
        self.names_strs.append(temp)
        self.names.append(np.array(temp, dtype="S"))

        temp = [f"B{i}" for i in range(n_detectors)]
        self.names_strs.append(temp)
        self.names.append(np.array(temp, dtype="S"))

        x = np.arange(n_detectors * self.n_samples, dtype=np.float64).reshape(
            (n_detectors, self.n_samples)
        )
        self.tod.append(x)

        A = np.arange(n_detectors * n_detectors, dtype=np.float64).reshape(
            (n_detectors, n_detectors)
        )
        self.data.append(A)

        y = (
            np.arange(n_detectors * n_detectors).reshape((n_detectors, n_detectors))
            @ np.arange(n_detectors * self.n_samples).reshape((n_detectors, self.n_samples))
        ).astype(np.float64)
        self.tod_crosstalked.append(y)

        rng = default_rng()
        x = rng.standard_normal((n_detectors, self.n_samples))
        self.tod.append(x)

        A = np.identity(n_detectors) + np.reciprocal(
            np.arange(10, 10 + n_detectors * n_detectors)
        ).reshape((n_detectors, n_detectors))
        self.data.append(A)

        y = A @ x
        self.tod_crosstalked.append(y)

    def _each_op_crosstalk(
        self,
        tod_array: "np.ndarray[np.float64]",
        crosstalk_data: "np.ndarray[np.float64]",
        tod_crosstalked: "np.ndarray[np.float64]",
        detranks: "int",
    ):
        names = self.names[0]
        names_str = self.names_strs[0]

        if self.world_rank == 0:
            crosstalk_matrices = [
                SimpleCrosstalkMatrix(names, crosstalk_data, debug=True)
            ]
        else:
            crosstalk_matrices = []
        op_crosstalk = OpCrosstalk(1, crosstalk_matrices, debug=True)

        tod = TODCache(self.world_comm, names_str, self.n_samples, detranks=detranks)

        # write local detectors
        start, length = tod.local_samples
        local_dets_set = set(tod.local_dets)
        for i, name in enumerate(names_str):
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
        for i, name in enumerate(names_str):
            if name in local_dets_set:
                output = tod.cache.reference(f"{self.signal_name}_{name}")
                answer = tod_crosstalked[i, start:start + length]
                np.testing.assert_allclose(answer, output)

    def test_op_crosstalk(self):
        for i, (tod, data, tod_crosstalked) in enumerate(zip(self.tod, self.data, self.tod_crosstalked)):
            for detranks in self.detranks:
                with self.subTest(msg=f'OpCrosstalk Test case {i + 1} with detranks {detranks}'):
                    self._each_op_crosstalk(
                        (
                            tod,
                            data,
                            tod_crosstalked,
                            detranks,
                        )
                    )

    def _each_op_crosstalk_multiple_matrices(
        self,
        names: "List[np.ndarray['S']]",
        names_strs: "List[List[str]]",
        tods_array: "List[np.ndarray[np.float64]]",
        crosstalk_datas: "List[np.ndarray[np.float64]]",
        tods_crosstalked: "List[np.ndarray[np.float64]]",
        detranks: "int",
    ):
        n_crosstalk_matrices = len(names)
        idxs_per_rank = range(self.world_rank, n_crosstalk_matrices, self.world_procs)
        crosstalk_matrices = [
            SimpleCrosstalkMatrix(names[i], crosstalk_datas[i], debug=True)
            for i in idxs_per_rank
        ]
        op_crosstalk = OpCrosstalk(n_crosstalk_matrices, crosstalk_matrices, debug=True)

        tod = TODCache(self.world_comm, sum(names_strs, []), self.n_samples, detranks=detranks)

        # write local detectors
        start, length = tod.local_samples
        local_dets_set = set(tod.local_dets)
        for names_str, tod_array in zip(names_strs, tods_array):
            for i, name in enumerate(names_str):
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
        for names_str, tod_crosstalked in zip(names_strs, tods_crosstalked):
            for i, name in enumerate(names_str):
                if name in local_dets_set:
                    output = tod.cache.reference(f"{self.signal_name}_{name}")
                    answer = tod_crosstalked[i, start:start + length]
                    np.testing.assert_allclose(answer, output)

    def test_op_crosstalk_multiple_matrices(self):
        for detranks in self.detranks:
            with self.subTest(msg=f'Test OpCrosstalk with multiple matrices and detranks {detranks}'):
                self._each_op_crosstalk_multiple_matrices(
                    self.names,
                    self.names_strs,
                    self.tod,
                    self.data,
                    self.tod_crosstalked,
                    detranks,
                )

    def test_op_crosstalk_io(self):
        names = self.names[0]
        crosstalk_data = self.data[0]
        if self.world_rank == 0:
            path = self.outdir / "simple_crosstalk_matrix.hdf5"
            crosstalk_matrix = SimpleCrosstalkMatrix(names, crosstalk_data, debug=True)

            crosstalk_matrix.dump(path)
            crosstalk_matrix_read = SimpleCrosstalkMatrix.load(path, debug=True)

            np.testing.assert_array_equal(names, crosstalk_matrix_read.names)
            np.testing.assert_array_equal(crosstalk_data, crosstalk_matrix_read.data)
