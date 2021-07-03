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
from ._helpers import create_outdir  # , create_satellite_data
from .mpi import MPITestCase

if TYPE_CHECKING:
    from typing import List


mpiworld, procs, rank = get_world()
log = Logger.get()

# just to have more than 1 detectors per procs, and is irregular
n_detectors = 4 * procs + 3
n_samples = 100
signal_name = "signal"
detrankses = [1]
for i in range(2, procs + 1):
    if procs % i == 0:
        detrankses.append(i)
log.info(f'Testing OpCrosstalk against detranks {", ".join(map(str, detrankses))}.')

names_str = [f"A{i}" for i in range(n_detectors)]
names = np.array(names_str, dtype="S")
names_2_str = [f"B{i}" for i in range(n_detectors)]
names_2 = np.array(names_str, dtype="S")

tod_array = np.arange(n_detectors * n_samples, dtype=np.float64).reshape(
    (n_detectors, n_samples)
)
crosstalk_data = np.arange(n_detectors * n_detectors, dtype=np.float64).reshape(
    (n_detectors, n_detectors)
)
tod_crosstalked = (
    np.arange(n_detectors * n_detectors).reshape((n_detectors, n_detectors))
    @ np.arange(n_detectors * n_samples).reshape((n_detectors, n_samples))
).astype(np.float64)

rng = default_rng()
tod_array_random = rng.standard_normal((n_detectors, n_samples))
crosstalk_data_2 = np.identity(n_detectors) + np.reciprocal(
    np.arange(10, 10 + n_detectors * n_detectors)
).reshape((n_detectors, n_detectors))
tod_crosstalked_random = crosstalk_data_2 @ tod_array_random


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

    @staticmethod
    def _tset_op_crosstalk(
        tod_array: "np.ndarray[np.float64]",
        crosstalk_data: "np.ndarray[np.float64]",
        tod_crosstalked: "np.ndarray[np.float64]",
        detranks: "int",
    ):
        if rank == 0:
            crosstalk_matrices = [
                SimpleCrosstalkMatrix(names, crosstalk_data, debug=True)
            ]
        else:
            crosstalk_matrices = []
        op_crosstalk = OpCrosstalk(1, crosstalk_matrices, debug=True)

        tod = TODCache(mpiworld, names_str, n_samples, detranks=detranks)

        # write local detectors
        start, length = tod.local_samples
        local_dets_set = set(tod.local_dets)
        for i, name in enumerate(names_str):
            if name in local_dets_set:
                tod.write(detector=name, data=tod_array[i, start : start + length])
        data = FakeData(
            [
                {
                    "tod": tod,
                }
            ]
        )
        op_crosstalk.exec(data, signal_name=signal_name)

        # check output local detectors
        # half float64 precision (float64 has 53-bit precison)
        # will not produce bit-identical output as mat-mul
        # is sensitive to the sum-ordering
        ulp_max = 2 ** (53 / 2)
        for i, name in enumerate(names_str):
            if name in local_dets_set:
                output = tod.cache.reference(f"{signal_name}_{name}")
                answer = tod_crosstalked[i, start : start + length]
                ulp = np.testing.assert_array_max_ulp(answer, output, ulp_max)
                log.info(
                    f"Reproducing mat-mul for detector {name} with max ULP: {ulp.max():e}"
                )

    def test_op_crosstalk(self):
        cases = []
        for detranks in detrankses:
            cases += [
                (
                    tod_array,
                    crosstalk_data,
                    tod_crosstalked,
                    detranks,
                ),
                (
                    tod_array_random,
                    crosstalk_data_2,
                    tod_crosstalked_random,
                    detranks,
                ),
            ]
        for case in cases:
            self._tset_op_crosstalk(*case)

    @staticmethod
    def _tset_op_crosstalk_multiple_matrices(
        names: "List[np.ndarray['S']]",
        names_strs: "List[List[str]]",
        tods_array: "List[np.ndarray[np.float64]]",
        crosstalk_datas: "List[np.ndarray[np.float64]]",
        tods_crosstalked: "List[np.ndarray[np.float64]]",
        detranks: "int",
    ):
        n_crosstalk_matrices = len(names)
        idxs_per_rank = range(rank, n_crosstalk_matrices, procs)
        crosstalk_matrices = [
            SimpleCrosstalkMatrix(names[i], crosstalk_datas[i], debug=True)
            for i in idxs_per_rank
        ]
        op_crosstalk = OpCrosstalk(n_crosstalk_matrices, crosstalk_matrices, debug=True)

        tod = TODCache(mpiworld, sum(names_strs, []), n_samples, detranks=detranks)

        # write local detectors
        start, length = tod.local_samples
        local_dets_set = set(tod.local_dets)
        for names_str, tod_array in zip(names_strs, tods_array):
            for i, name in enumerate(names_str):
                if name in local_dets_set:
                    tod.write(detector=name, data=tod_array[i, start : start + length])
        data = FakeData(
            [
                {
                    "tod": tod,
                }
            ]
        )
        op_crosstalk.exec(data, signal_name=signal_name)

        # check output local detectors
        # half float64 precision (float64 has 53-bit precison)
        # will not produce bit-identical output as mat-mul
        # is sensitive to the sum-ordering
        ulp_max = 2 ** (53 / 2)
        for names_str, tod_crosstalked in zip(names_strs, tods_crosstalked):
            for i, name in enumerate(names_str):
                if name in local_dets_set:
                    output = tod.cache.reference(f"{signal_name}_{name}")
                    answer = tod_crosstalked[i, start : start + length]
                    ulp = np.testing.assert_array_max_ulp(answer, output, ulp_max)
                    log.info(
                        f"Reproducing mat-mul for detector {name} with max ULP: {ulp.max():e}"
                    )

    def test_op_crosstalk_multiple_matrices(self):
        cases = []
        for detranks in detrankses:
            cases.append(
                (
                    [names, names_2],
                    [names_str, names_2_str],
                    [tod_array, tod_array_random],
                    [crosstalk_data, crosstalk_data_2],
                    [tod_crosstalked, tod_crosstalked_random],
                    detranks,
                ),
            )
        for case in cases:
            self._tset_op_crosstalk_multiple_matrices(*case)

    def test_op_crosstalk_io(self):
        if rank == 0:
            path = self.outdir / "simple_crosstalk_matrix.hdf5"
            crosstalk_matrix = SimpleCrosstalkMatrix(names, crosstalk_data, debug=True)

            crosstalk_matrix.dump(path)
            crosstalk_matrix_read = SimpleCrosstalkMatrix.load(path, debug=True)

            np.testing.assert_array_equal(names, crosstalk_matrix_read.names)
            np.testing.assert_array_equal(crosstalk_data, crosstalk_matrix_read.data)
