# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import time

import numpy as np
import numpy.testing as nt

from .. import ops
from .._libtoast import test_accel_op_array, test_accel_op_buffer
from ..accelerator import (
    accel_data_create,
    accel_data_delete,
    accel_data_present,
    accel_data_update_device,
    accel_data_update_host,
    accel_enabled,
    use_accel_jax,
    use_accel_omp,
)
from ..data import Data
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..traits import Int, Unicode, trait_docs
from ._helpers import create_comm, create_outdir, create_satellite_data
from .mpi import MPITestCase


@trait_docs
class AccelOperator(ops.Operator):
    """Dummy operator to test device data movement."""

    # Class traits
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        for ob in data.obs:
            if use_accel:
                # Base class has checked that data listed in our requirements
                # is present.  Call compiled code that uses OpenACC to work
                # with this data.
                test_accel_op_buffer(ob.detdata[self.det_data].data)
                test_accel_op_array(ob.detdata[self.det_data].data)
            else:
                # Just use python
                for d in ob.detdata[self.det_data].detectors:
                    ob.detdata[self.det_data][d] *= 4

    def _finalize(self, data, use_accel=False, **kwargs):
        pass

    def _requires(self):
        return {"detdata": [self.det_data]}

    def _provides(self):
        return {"detdata": [self.det_data]}

    def _supports_acc(self):
        return True


class AcceleratorTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        # self.outdir = create_outdir(self.comm, fixture_name)
        self.rank = 0
        self.nproc = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.nproc = self.comm.size
        self.types = {
            "f64": np.float64,
            "f32": np.float32,
            "i64": np.int64,
            "u64": np.uint64,
            "i32": np.int32,
            "u32": np.uint32,
            "i16": np.int16,
            "u16": np.uint16,
            "i8": np.int8,
            "u8": np.uint8,
        }

    def test_memory(self):
        if not (use_accel_omp or use_accel_jax):
            if self.rank == 0:
                print("Not running with accelerator support- skipping memory test")
            return
        data = dict()
        check = dict()
        for tname, tp in self.types.items():
            data[tname] = np.ones(100, dtype=tp)
            check[tname] = 2 * np.array(data[tname])

        # Verify that data is not on the device
        for tname, buffer in data.items():
            self.assertFalse(accel_data_present(buffer))

        # Copy to device
        for tname, buffer in data.items():
            accel_data_create(buffer)
            accel_data_update_device(buffer)

        # Check that it is present
        for tname, buffer in data.items():
            self.assertTrue(accel_data_present(buffer))

        # Change host copy
        for tname, buffer in data.items():
            buffer[:] *= 2

        # Update device copy
        for tname, buffer in data.items():
            accel_data_update_device(buffer)

        # Reset host copy
        for tname, buffer in data.items():
            buffer[:] = 0

        # Update host copy from device
        for tname, buffer in data.items():
            accel_data_update_host(buffer)

        # Check Values
        for tname, buffer in data.items():
            np.testing.assert_array_equal(buffer, check[tname])

        # Delete device copy
        for tname, buffer in data.items():
            accel_data_delete(buffer)

        # Verify that data is not on the device
        for tname, buffer in data.items():
            self.assertFalse(accel_data_present(buffer))

    def test_data_stage(self):
        if not (use_accel_omp or use_accel_jax):
            if self.rank == 0:
                print("Not running with accelerator support- skipping data stage test")
            return
        data = create_satellite_data(self.comm, pixel_per_process=4)
        data.obs = data.obs[:1]
        for ob in data.obs:
            for itp, (tname, tp) in enumerate(self.types.items()):
                for sname, sshape in zip(["1", "2"], [None, (2,)]):
                    name = f"{tname}_{sname}"
                    ob.detdata.create(name, sample_shape=sshape, dtype=tp)
                    ob.detdata[name][:] = itp + 1
                    shp = (ob.n_local_samples,)
                    if sshape is not None:
                        shp += sshape
                    ob.shared.create_column(name, shp, dtype=tp)
                    if ob.comm_col_rank == 0:
                        ob.shared[name].set((itp + 1) * np.ones(shp, dtype=tp))
                    else:
                        ob.shared[name].set(None)

        pix_dist = PixelDistribution(
            n_pix=100,
            n_submap=10,
            local_submaps=[0, 2, 4, 6, 8],
            comm=data.comm.comm_world,
        )

        data["test_pix"] = PixelData(pix_dist, dtype=np.float64, n_value=3)

        # Duplicate for future comparison
        check_data = Data(comm=data.comm)
        for ob in data.obs:
            check_data.obs.append(ob.duplicate())
        check_data["test_pix"] = data["test_pix"].duplicate()

        # print("Start original:")
        # for ob in check_data.obs:
        #     for itp, (tname, tp) in enumerate(self.types.items()):
        #         for sname, sshape in zip(["1", "2"], [None, (2,)]):
        #             name = f"{tname}_{sname}"
        #             print(ob.detdata[name])
        #             print(ob.shared[name])
        # print("Start current:")
        # for ob in data.obs:
        #     for itp, (tname, tp) in enumerate(self.types.items()):
        #         for sname, sshape in zip(["1", "2"], [None, (2,)]):
        #             name = f"{tname}_{sname}"
        #             print(ob.detdata[name])
        #             print(ob.shared[name])

        # The dictionary of data objects.
        dnames = {
            "global": ["test_pix"],
            "meta": list(),
            "detdata": list(),
            "shared": list(),
            "intervals": list(),
        }
        for itp, (tname, tp) in enumerate(self.types.items()):
            for sname, sshape in zip(["1", "2"], [None, (2,)]):
                name = f"{tname}_{sname}"
                dnames["detdata"].append(name)
                dnames["shared"].append(name)

        # Copy data to device
        data.accel_create(dnames)
        data.accel_update_device(dnames)

        # Clear buffers
        for ob in data.obs:
            for itp, (tname, tp) in enumerate(self.types.items()):
                for sname, sshape in zip(["1", "2"], [None, (2,)]):
                    name = f"{tname}_{sname}"
                    ob.detdata[name][:] = 0
                    shp = (ob.n_local_samples,)
                    if sshape is not None:
                        shp += sshape
                    if ob.comm_col_rank == 0:
                        ob.shared[name].set(np.zeros(shp, dtype=tp))
                    else:
                        ob.shared[name].set(None)

        # print("Purge original:")
        # for ob in check_data.obs:
        #     for itp, (tname, tp) in enumerate(self.types.items()):
        #         for sname, sshape in zip(["1", "2"], [None, (2,)]):
        #             name = f"{tname}_{sname}"
        #             print(ob.detdata[name])
        #             print(ob.shared[name])
        # print("Purge current:")
        # for ob in data.obs:
        #     for itp, (tname, tp) in enumerate(self.types.items()):
        #         for sname, sshape in zip(["1", "2"], [None, (2,)]):
        #             name = f"{tname}_{sname}"
        #             print(ob.detdata[name])
        #             print(ob.shared[name])

        # Copy back from device
        data.accel_update_host(dnames)

        # print("Check original:")
        # for ob in check_data.obs:
        #     for itp, (tname, tp) in enumerate(self.types.items()):
        #         for sname, sshape in zip(["1", "2"], [None, (2,)]):
        #             name = f"{tname}_{sname}"
        #             print(ob.detdata[name])
        #             print(ob.shared[name])
        # print("Check current:")
        # for ob in data.obs:
        #     for itp, (tname, tp) in enumerate(self.types.items()):
        #         for sname, sshape in zip(["1", "2"], [None, (2,)]):
        #             name = f"{tname}_{sname}"
        #             print(ob.detdata[name])
        #             print(ob.shared[name])

        # Compare
        for check, ob in zip(check_data.obs, data.obs):
            if ob != check:
                print(f"Original: {check}")
                print(f"Roundtrip:  {ob}")
            self.assertEqual(ob, check)
        if data["test_pix"] != check_data["test_pix"]:
            print(
                f"Original: {check_data['test_pix']} {np.array(check_data['test_pix'].raw)[:]}"
            )
            print(f"Roundtrip: {data['test_pix']} {np.array(data['test_pix'].raw)[:]}")
        self.assertEqual(data["test_pix"], check_data["test_pix"])

        # Now go and shrink the detector buffers

        data.accel_update_device(dnames)

        for check, ob in zip(check_data.obs, data.obs):
            for itp, (tname, tp) in enumerate(self.types.items()):
                for sname, sshape in zip(["1", "2"], [None, (2,)]):
                    name = f"{tname}_{sname}"
                    # This will set the host copy to zero and invalidate the device copy
                    ob.detdata[name].change_detectors(ob.local_detectors[0:2])
                    check.detdata[name].change_detectors(check.local_detectors[0:2])
                    ob.detdata[name].accel_update_host()
                    # Reset host copy
                    ob.detdata[name][:] = itp + 1
                    check.detdata[name][:] = itp + 1
                    # Update device copy
                    ob.detdata[name].accel_update_device()

        data.accel_update_host(dnames)

        # Compare
        for check, ob in zip(check_data.obs, data.obs):
            if ob != check:
                print(f"Original: {check}")
                print(f"Roundtrip:  {ob}")
            self.assertEqual(ob, check)
        if data["test_pix"] != check_data["test_pix"]:
            print(f"Original: {check_data['test_pix']} {check_data['test_pix'].raw}")
            print(f"Roundtrip: {data['test_pix']} {data['test_pix'].raw}")
        self.assertEqual(data["test_pix"], check_data["test_pix"])

        del check_data
        del data

    def test_operator_stage(self):
        if not (use_accel_omp or use_accel_jax):
            if self.rank == 0:
                print("Not running with accelerator support- skipping operator test")
            return

        data = create_satellite_data(self.comm)

        accel_op = AccelOperator()

        # Make a copy for later comparison
        ops.Copy(detdata=[(accel_op.det_data, "original")]).apply(data)

        # Data not staged
        accel_op.apply(data)

        # Stage the data
        data.accel_create(accel_op.requires())
        data.accel_update_device(accel_op.requires())

        # Run with staged data
        accel_op.apply(data)

        # Copy out
        data.accel_update_host(accel_op.provides())

        # Check
        for ob in data.obs:
            for det in ob.local_detectors:
                np.testing.assert_allclose(
                    ob.detdata[accel_op.det_data][det],
                    16.0 * ob.detdata["original"][det],
                )
