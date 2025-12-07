# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import warnings

from ..io import load_hdf5, save_hdf5
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Dict, Int, List, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


def obs_approx_equal(obs1, obs2):
    """Compare observations with relaxed floating point data comparisons.

    Normal equality tests for detector data may be too stringent in the case where
    float64 data is compressed to 32bit integer FLAC data.  Here we do a normal equality
    test except for detector data that has floating point values, where we instead
    use an increased tolerance.

    Args:
        obs1 (Observation):  The first observation to compare.
        obs2 (Observation):  The second observation to compare.

    Returns:
        (bool):  True if observations are approximately equal.

    """
    log = Logger.get()
    fail = 0
    if obs1.name != obs2.name:
        fail = 1
        log.verbose(
            f"Proc {obs1.comm.world_rank}:  Obs names {obs1.name} != {obs2.name}"
        )
    if obs1.uid != obs2.uid:
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs uid {obs1.uid} != {obs2.uid}")
    if obs1.telescope != obs2.telescope:
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs telescopes not equal")
    if obs1.session != obs2.session:
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs sessions not equal")
    if obs1.dist != obs2.dist:
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs distributions not equal")

    if not obs1.meta_equal(obs2, f"Proc {obs1.comm.world_rank}:  Obs _internal"):
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs metadata not equal")

    # Compare any extra metadata class instances
    extra_objs1 = list()
    for k, v in vars(obs1).items():
        if k.startswith("_"):
            continue
        if hasattr(v, "save_hdf5"):
            extra_objs1.append(k)
    extra_objs2 = list()
    for k, v in vars(obs2).items():
        if k.startswith("_"):
            continue
        if hasattr(v, "save_hdf5"):
            extra_objs2.append(k)

    if extra_objs1 != extra_objs2:
        fail = 1
        log.verbose(
            f"Proc {obs1.comm.world_rank}:  Obs extra metadata obj lists not equal"
        )
    else:
        for exobj in extra_objs1:
            obj1 = getattr(obs1, exobj)
            obj2 = getattr(obs2, exobj)
            if obj1 != obj2:
                fail = 1
                log.verbose(
                    f"Proc {obs1.comm.world_rank}:  Obs extra {exobj} not equal"
                )

    if obs1.shared != obs2.shared:
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs shared data not equal")
    if obs1.intervals != obs2.intervals:
        fail = 1
        log.verbose(f"Proc {obs1.comm.world_rank}:  Obs intervals not equal")

    # Walk through the detector data
    o1d = obs1.detdata
    o2d = obs2.detdata
    if o1d.detectors != o2d.detectors:
        msg = f"Proc {obs1.comm.world_rank}:  Obs detdata detectors"
        msg += f" {o1d.detectors} != {o2d.detectors}"
        log.verbose(msg)
        fail = 1
    if o1d.samples != o2d.samples:
        msg = f"Proc {obs1.comm.world_rank}:  Obs detdata samples "
        msg += f"{o1d.samples} != {o2d.samples}"
        log.verbose(msg)
        fail = 1
    if set(o1d._internal.keys()) != set(o2d._internal.keys()):
        msg = f"Proc {obs1.comm.world_rank}:  Obs detdata keys "
        msg += f"{o1d._internal.keys()} != {o2d._internal.keys()}"
        log.verbose(msg)
        fail = 1
    for k in o1d._internal.keys():
        if o1d[k].detectors != o2d[k].detectors:
            msg = f"Proc {obs1.comm.world_rank}:  Obs detdata {k} detectors "
            msg += f"{o1d[k].detectors} != {o2d[k].detectors}"
            log.verbose(msg)
            fail = 1
        if o1d[k].dtype.char != o2d[k].dtype.char:
            msg = f"Proc {obs1.comm.world_rank}:  Obs detdata {k} dtype "
            msg += f"{o1d[k].dtype} != {o2d[k].dtype}"
            log.verbose(msg)
            fail = 1
        if o1d[k].shape != o2d[k].shape:
            msg = f"Proc {obs1.comm.world_rank}:  Obs detdata {k} shape "
            msg += f"{o1d[k].shape} != {o2d[k].shape}"
            log.verbose(msg)
            fail = 1
        if o1d[k].units != o2d[k].units:
            msg = f"Proc {obs1.comm.world_rank}:  Obs detdata {k} units "
            msg += f"{o1d[k].units} != {o2d[k].units}"
            log.verbose(msg)
            fail = 1
        if o1d[k].dtype == np.dtype(np.float64) or o1d[k].dtype == np.dtype(np.float32):
            # Only compare to 32bit precision
            if not np.allclose(o1d[k].data, o2d[k].data, rtol=1.0e-3, atol=1.0e-5):
                msg = f"Proc {obs1.comm.world_rank}:  Obs detdata {k} array "
                msg += f"{o1d[k].data} != {o2d[k].data}"
                log.verbose(msg)
                fail = 1
        elif not np.array_equal(o1d[k].data, o2d[k].data):
            msg = f"Proc {obs1.comm.world_rank}:  Obs detdata {k} array "
            msg += f"{o1d[k].data} != {o2d[k].data}"
            log.verbose(msg)
            fail = 1
    if obs1.comm.comm_group is not None:
        fail = obs1.comm.comm_group.allreduce(fail, op=MPI.SUM)
    return fail == 0


@trait_docs
class SaveHDF5(Operator):
    """Operator which saves observations to HDF5.

    This creates a file for each observation.  Detector data compression can be enabled
    by specifying a tuple for each item in the detdata list.  The first item in the
    tuple is the field name.  The second item is either None, or a dictionary of FLAC
    comppression properties.  Allowed compression parameters are:

        "level": (int) the compression level
        "quanta": (float) the quantization value, only for floating point data
        "precision": (float) the fixed precision, only for floating point data

    For integer data, an empty dictionary may be passed, and FLAC compression
    will use the default level (5).  Floating point data *must* specify either the
    quanta or precision parameters.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    volume = Unicode(
        "toast_out_hdf5",
        allow_none=True,
        help="Top-level directory for the data volume",
    )

    # FIXME:  We should add a filtering mechanism here to dump a subset of
    # observations and / or detectors, as well as figure out subdirectory organization.

    meta = List([], allow_none=True, help="Only save this list of meta objects")

    detdata = List(
        [defaults.det_data, defaults.det_flags],
        help="Only save this list of detdata objects",
    )

    shared = List([], help="Only save this list of shared objects")

    intervals = List([], help="Only save this list of intervals objects")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    config = Dict({}, help="Write this job config to the file")

    force_serial = Bool(
        False, help="Use serial HDF5 operations, even if parallel support available"
    )

    detdata_float32 = Bool(
        False, help="(Deprecated) Specify the per-field compression parameters."
    )

    detdata_in_place = Bool(
        False,
        help="If True, all compressed detector data will be decompressed and written "
        "over the input data.",
    )

    compress_detdata = Bool(
        False, help="(Deprecated) Specify the per-field compression parameters"
    )

    compress_precision = Int(
        None,
        allow_none=True,
        help="(Deprecated) Specify the per-field compression parameters",
    )

    verify = Bool(False, help="If True, immediately load data back in and verify")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.volume is None:
            msg = "You must set the volume trait prior to calling exec()"
            log.error(msg)
            raise RuntimeError(msg)

        # Warn for deprecated traits that will be removed eventually.

        if self.detdata_float32:
            msg = "The detdata_float32 option is deprecated.  Instead, specify"
            msg = " a compression quanta / precision that is appropriate for"
            msg = " each detdata field."
            warnings.warn(msg, DeprecationWarning)

        if self.compress_detdata:
            msg = "The compress_detdata option is deprecated.  Instead, specify"
            msg = " a compression quanta / precision that is appropriate for"
            msg = " each detdata field."
            warnings.warn(msg, DeprecationWarning)

        if self.compress_precision is not None:
            msg = "The compress_precision option is deprecated.  Instead, specify"
            msg = " a compression quanta / precision that is appropriate for"
            msg = " each detdata field."
            warnings.warn(msg, DeprecationWarning)

        # One process creates the top directory
        if data.comm.world_rank == 0:
            os.makedirs(self.volume, exist_ok=True)
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        meta_fields = None
        if len(self.meta) > 0:
            meta_fields = list(self.meta)

        shared_fields = None
        if len(self.shared) > 0:
            shared_fields = list(self.shared)

        intervals_fields = None
        if len(self.intervals) > 0:
            intervals_fields = list(self.intervals)

        if len(self.detdata) > 0:
            detdata_fields = list(self.detdata)
        else:
            detdata_fields = list()

        # Handle parsing of deprecated global compression options.  All
        # new code should specify the FLAC compression parameters per
        # field.
        for ifield, field in enumerate(detdata_fields):
            if not isinstance(field, str):
                # User already specified compression parameters
                continue
            cprops = {"level": 5}
            if self.compress_detdata:
                # Try to guess what to do.
                if "flag" not in field:
                    # Might be float data
                    if self.compress_precision is None:
                        # Compress to 32bit floats
                        cprops["quanta"] = np.finfo(np.float32).eps
                    else:
                        cprops["precision"] = self.compress_precision
                detdata_fields[ifield] = (field, cprops)
            elif self.detdata_float32:
                # Implement this truncation as just compression to 32bit float
                # precision
                cprops["quanta"] = np.finfo(np.float32).eps
                detdata_fields[ifield] = (field, cprops)
            else:
                # No compression
                detdata_fields[ifield] = (field, None)

        for ob in data.obs:
            # Observations must have a name for this to work
            if ob.name is None:
                raise RuntimeError(
                    "Observations must have a name in order to save to HDF5 format"
                )

            # Check to see if any detector data objects are temporary and have just
            # a partial list of detectors.  Delete these.
            for dd in list(ob.detdata.keys()):
                if ob.detdata[dd].detectors != ob.local_detectors:
                    del ob.detdata[dd]

            outpath = save_hdf5(
                ob,
                self.volume,
                meta=meta_fields,
                detdata=detdata_fields,
                shared=shared_fields,
                intervals=intervals_fields,
                config=self.config,
                times=str(self.times),
                force_serial=self.force_serial,
                detdata_in_place=self.detdata_in_place,
            )

            log.info_rank(f"Wrote {outpath}", comm=data.comm.comm_group)

            if self.verify:
                # We are going to load the data back in, but first we need to make
                # a modified copy of the input observation for comparison.  This is
                # because we may have only a portion of the data on disk and we
                # might have also converted data to 32bit floats.

                loadpath = outpath

                if len(self.detdata) > 0:
                    # There might be compression info
                    if isinstance(detdata_fields[0], (tuple, list)):
                        verify_fields = [x[0] for x in detdata_fields]
                    else:
                        verify_fields = list(detdata_fields)
                else:
                    # We saved nothing
                    verify_fields = list()

                original = ob.duplicate(
                    times=str(self.times),
                    meta=meta_fields,
                    shared=shared_fields,
                    detdata=verify_fields,
                    intervals=intervals_fields,
                )

                compare = load_hdf5(
                    loadpath,
                    data.comm,
                    process_rows=ob.comm_col_size,
                    meta=meta_fields,
                    detdata=verify_fields,
                    shared=shared_fields,
                    intervals=intervals_fields,
                    force_serial=self.force_serial,
                )

                if not obs_approx_equal(compare, original):
                    msg = "Observation HDF5 verify failed:\n"
                    msg += f"Input = {original}\n"
                    msg += f"Loaded = {compare}\n"
                    msg += f"Input signal[0] = {original.detdata['signal'][0]}\n"
                    msg += f"Loaded signal[0] = {compare.detdata['signal'][0]}"
                    log.error(msg)
                    raise RuntimeError(msg)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "detdata": list(),
            "intervals": list(),
            "shared": [
                self.times,
            ],
        }
        if self.meta is not None:
            req["meta"].extend(self.meta)
        if self.detdata is not None:
            req["detdata"].extend(self.detdata)
        if self.intervals is not None:
            req["intervals"].extend(self.intervals)
        if self.shared is not None:
            req["shared"].extend(self.shared)
        return req

    def _provides(self):
        return dict()
