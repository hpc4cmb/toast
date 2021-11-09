# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance, Float

from ..timing import function_timer

from .operator import Operator

from ..spt3g import available

if available:
    from spt3g import core as c3g
    from ..spt3g import frame_emitter


@trait_docs
class SaveSpt3g(Operator):
    """Operator which saves observations to SPT3G frame files.

    This creates a directory for each observation, and writes framefiles with
    a target size.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    directory = Unicode("spt3g_data", help="Top-level export directory")

    framefile_mb = Float(100.0, help="Target frame file size in MB")

    # FIXME:  We should add a filtering mechanism here to dump a subset of
    # observations and / or detectors.

    obs_export = Instance(
        klass=object,
        allow_none=True,
        help="Export class to create frames from an observation",
    )

    purge = Bool(False, help="If True, delete observation data as it is saved")

    @traitlets.validate("obs_export")
    def _check_export(self, proposal):
        ex = proposal["value"]
        if ex is not None:
            # Check that this class has an "export_rank" attribute and is callable.
            if not callable(ex):
                raise traitlets.TraitError("obs_export class must be callable")
            if not hasattr(ex, "export_rank"):
                raise traitlets.TraitError(
                    "obs_export class must have an export_rank attribute"
                )
        return ex

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not available:
            raise RuntimeError("spt3g is not available")

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the export class is set
        if self.obs_export is None:
            raise RuntimeError(
                "You must set the obs_export trait before calling exec()"
            )

        # Find the process rank which will have all the frames
        ex_rank = self.obs_export.export_rank
        if ex_rank is None:
            raise RuntimeError(
                "The obs_export class must be configured to export frames on one process"
            )

        # One process creates the top directory
        if data.comm.world_rank == 0:
            os.makedirs(self.directory, exist_ok=True)
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        for ob in data.obs:
            # Observations must have a name for this to work
            if ob.name is None:
                raise RuntimeError(
                    "Observations must have a name in order to save to SPT3G format"
                )

            # Export observation to frames on one process
            frames = self.obs_export(ob)

            # One process writes frame files
            if ob.comm.group_rank == ex_rank:
                # Make observation directory.  This should NOT already exist.
                ob_dir = os.path.join(self.directory, ob.name)
                os.makedirs(ob_dir)

                emitter = frame_emitter(frames=frames)

                save_pipe = c3g.G3Pipeline()
                save_pipe.Add(emitter)
                save_pipe.Add(
                    c3g.G3MultiFileWriter,
                    filename=os.path.join(ob_dir, "frames-%05u.g3"),
                    size_limit=int(self.framefile_mb * 1024 ** 2),
                )
                save_pipe.Run()

                del save_pipe
                del emitter

            if ob.comm.comm_group is not None:
                ob.comm.comm_group.barrier()

            if self.purge:
                ob.clear()

        if self.purge:
            data.obs.clear()
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()

    def _accelerators(self):
        return list()
