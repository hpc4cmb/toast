# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import time

import numpy as np
import numpy.testing as nt

from ..traits import (
    trait_docs,
    Int,
    Unicode,
)

from .. import ops

from .mpi import MPITestCase

from .._libtoast import (
    acc_enabled,
    acc_is_present,
    acc_copyin,
    acc_copyout,
    acc_delete,
    acc_update_device,
    acc_update_self,
    test_acc_op_buffer,
    test_acc_op_array,
)

from ..observation import default_values as defaults

from ._helpers import create_outdir, create_comm, create_satellite_data


@trait_docs
class AccOperator(ops.Operator):
    """Dummy operator to test device data movement."""

    # Class traits
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, use_acc=False, **kwargs):
        for ob in data.obs:
            if use_acc:
                # Base class has checked that data listed in our requirements
                # is present.  Call compiled code that uses OpenACC to work
                # with this data.
                test_acc_op_buffer(
                    ob.detdata[self.det_data].flatdata,
                    len(ob.detdata[self.det_data].detectors),
                )
                test_acc_op_array(ob.detdata[self.det_data].data)
            else:
                # Just use python
                for d in ob.detdata[self.det_data].detectors:
                    ob.detdata[self.det_data][d] *= 4

    def _finalize(self, data, use_acc=False, **kwargs):
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
        if not acc_enabled():
            if self.rank == 0:
                print("Not compiled with OpenACC support- skipping memory test")
            return
        data = dict()
        check = dict()
        for tname, tp in self.types.items():
            data[tname] = np.ones(100, dtype=tp)
            check[tname] = 2 * np.array(data[tname])

        # Verify that data is not on the device
        for tname, buffer in data.items():
            self.assertFalse(acc_is_present(buffer))

        # Copy to device
        for tname, buffer in data.items():
            acc_copyin(buffer)

        # Check that it is present
        for tname, buffer in data.items():
            self.assertTrue(acc_is_present(buffer))

        # Change host copy
        for tname, buffer in data.items():
            buffer[:] *= 2

        # Update device copy
        for tname, buffer in data.items():
            acc_update_device(buffer)

        # Reset host copy
        for tname, buffer in data.items():
            buffer[:] = 0

        # Update host copy from device
        for tname, buffer in data.items():
            acc_update_self(buffer)

        # Check Values
        for tname, buffer in data.items():
            np.testing.assert_array_equal(buffer, check[tname])

        # Delete device copy
        for tname, buffer in data.items():
            acc_delete(buffer)

        # Verify that data is not on the device
        for tname, buffer in data.items():
            self.assertFalse(acc_is_present(buffer))

    def test_operator_stage(self):
        if not acc_enabled():
            if self.rank == 0:
                print("Not compiled with OpenACC support- skipping memory test")
            return
        data = create_satellite_data(self.comm)
        acc_op = AccOperator()

        # Data not staged
        acc_op.apply(data)

        # Stage the data
        data.acc_copyin(acc_op.requires())

        # Run with staged data
        acc_op.apply(data)

        # Copy out
        # data.acc_copyout(acc_op.provides())
