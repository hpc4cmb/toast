# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from ..tod import tidas_available, spt3g_available

if tidas_available:
    from ..tod.tidas import OpTidasExport, TODTidas

if spt3g_available:
    from ..tod.spt3g import Op3GExport, TOD3G


def add_tidas_args(parser):
    """ Add the noise simulation arguments
    """
    parser.add_argument(
        "--tidas", required=False, default=None, help="Output TIDAS export path"
    )
    return


def add_spt3g_args(parser):
    """ Add the noise simulation arguments
    """
    parser.add_argument(
        "--spt3g", required=False, default=None, help="Output SPT3G export path"
    )
    return


@function_timer
def output_tidas(args, comm, data, cache_prefix=None, verbose=True):
    if args.tidas is None:
        return
    if not tidas_available:
        raise RuntimeError(
            "TIDAS not available.  Cannot export to '{}'" "".format(args.tidas)
        )
    log = Logger.get()
    timer = Timer()
    tidas_path = os.path.abspath(args.tidas)

    if comm.world_rank == 0 and verbose:
        log.info("Exporting data to a TIDAS volume at {}".format(tidas_path))

    timer.start()
    export = OpTidasExport(
        tidas_path,
        TODTidas,
        backend="hdf5",
        use_intervals=True,
        create_opts={"group_dets": "sim"},
        ctor_opts={"group_dets": "sim"},
        cache_name=cache_prefix,
    )
    export.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("TIDAS export")
    return


@function_timer
def output_spt3g(args, comm, data, cache_prefix=None, verbose=True):
    if args.spt3g is None:
        return
    if not spt3g_available:
        raise RuntimeError(
            "SPT3G not available.  Cannot export to '{}'" "".format(args.spt3g)
        )
    log = Logger.get()
    timer = Timer()

    spt3g_path = os.path.abspath(args.spt3g)

    if comm.world_rank == 0 and verbose:
        log.info("Exporting data to a SPT3G directory tree at {}" "".format(spt3g_path))

    timer.start()
    export = Op3GExport(
        spt3g_path,
        TOD3G,
        use_intervals=True,
        export_opts={"prefix": "sim"},
        cache_name=cache_prefix,
    )
    export.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("spt3g export")
    return
