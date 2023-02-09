# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ..spt3g import available
from ..timing import function_timer
from ..traits import Instance, Int, List, trait_docs
from ..utils import Logger
from .operator import Operator

if available:
    from spt3g import core as c3g

    from ..spt3g import frame_collector, frame_emitter


@trait_docs
class RunSpt3g(Operator):
    """Operator which runs a G3Pipeline.

    This operator converts each observation to a stream of frames on each process
    and then runs the specified G3 pipeline on the local frames.  If the `obs_import`
    trait is specified, the resulting frames are re-imported to a toast observation
    at the end.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    obs_export = Instance(
        klass=object,
        allow_none=True,
        help="Export class to create frames from an observation",
    )

    obs_import = Instance(
        klass=object,
        allow_none=True,
        help="Import class to create observations from frame files",
    )

    modules = List(
        [],
        help="List of tuples of (callable, **kwargs) that will passed to G3Pipeline.Add()",
    )

    @traitlets.validate("obs_export")
    def _check_export(self, proposal):
        ex = proposal["value"]
        if ex is not None:
            # Check that this class is callable.
            if not callable(ex):
                raise traitlets.TraitError("obs_export class must be callable")
        return ex

    @traitlets.validate("obs_import")
    def _check_import(self, proposal):
        im = proposal["value"]
        if im is not None:
            # Check that this class is callable.
            if not callable(im):
                raise traitlets.TraitError("obs_import class must be callable")
        return im

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

        # Check that the import class is set
        if self.obs_import is None:
            raise RuntimeError(
                "You must set the obs_import trait before calling exec()"
            )

        if len(self.modules) == 0:
            log.debug_rank(
                "No modules specified, nothing to do.", comm=data.comm.comm_world
            )
            return

        n_obs = len(data.obs)

        for iobs in range(n_obs):
            ob = data.obs[iobs]

            # Export observation to frames on all processes
            frames = self.obs_export(ob)

            # Helper class that emits frames
            emitter = frame_emitter(frames=frames)

            # Optional frame collection afterwards
            collector = frame_collector()

            # Set up pipeline
            run_pipe = c3g.G3Pipeline()
            run_pipe.Add(emitter)
            for callable, args in self.modules:
                run_pipe.Add(callable, args)
            if self.obs_import is not None:
                run_pipe.Add(collector)

            # Run it
            run_pipe.Run()

            # Optionally convert back and replace the input observation
            if self.obs_import is not None:
                data.obs[iobs] = self.obs_import(collector.frames)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()
