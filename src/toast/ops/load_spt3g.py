# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os

import numpy as np
import traitlets

from ..dist import distribute_discrete
from ..spt3g import available
from ..timing import function_timer
from ..traits import Instance, Int, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator

if available:
    from spt3g import core as c3g

    from ..spt3g import frame_collector


@trait_docs
class LoadSpt3g(Operator):
    """Operator which loads SPT3G data files into observations.

    This operator expects a top-level data directory.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    directory = Unicode(
        None, allow_none=True, help="Top-level directory of observations"
    )

    # FIXME:  We should add a filtering mechanism here to load a subset of
    # observations and / or detectors.

    obs_import = Instance(
        klass=object,
        allow_none=True,
        help="Import class to create observations from frame files",
    )

    @traitlets.validate("obs_import")
    def _check_import(self, proposal):
        im = proposal["value"]
        if im is not None:
            # Check that this class has an "import_rank" attribute and is callable.
            if not callable(im):
                raise traitlets.TraitError("obs_import class must be callable")
            if not hasattr(im, "import_rank"):
                raise traitlets.TraitError(
                    "obs_import class must have an import_rank attribute"
                )
        return im

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not available:
            raise RuntimeError("spt3g is not available")

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the import class is set
        if self.obs_import is None:
            raise RuntimeError(
                "You must set the obs_import trait before calling exec()"
            )

        # Find the process rank which will load all frames for a given group.
        im_rank = self.obs_import.import_rank
        if im_rank is None:
            raise RuntimeError(
                "The obs_import class must be configured to import frames on one process"
            )

        # FIXME:  we should add a check that self.obs_import.comm is the same as
        # the data.comm.comm_group communicator.

        # One global process computes the list of observations and their approximate
        # relative size (based on the total framefile sizes for each).

        obs_props = list()
        if data.comm.world_rank == 0:
            for root, dirs, files in os.walk(self.directory):
                for d in dirs:
                    framefiles = glob.glob(os.path.join(root, d, "*.g3"))
                    if len(framefiles) > 0:
                        # This is an observation directory
                        flist = list()
                        obtotal = 0
                        for ffile in sorted(framefiles):
                            fsize = os.path.getsize(ffile)
                            obtotal += fsize
                            flist.append(ffile)
                        obs_props.append((d, obtotal, flist))
                break
        obs_props.sort(key=lambda x: x[0])
        if data.comm.comm_world is not None:
            obs_props = data.comm.comm_world.bcast(obs_props, root=0)

        # Distribute observations among groups

        obs_sizes = [x[1] for x in obs_props]
        groupdist = distribute_discrete(obs_sizes, data.comm.ngroups)
        group_firstobs = groupdist[data.comm.group][0]
        group_numobs = groupdist[data.comm.group][1]

        # Every group creates its observations
        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            obname = obs_props[obindx][0]
            obfiles = obs_props[obindx][2]

            collect = frame_collector()

            if data.comm.group_rank == im_rank:
                all_frames = list()
                # We are loading the frames
                for ffile in obfiles:
                    load_pipe = c3g.G3Pipeline()
                    load_pipe.Add(c3g.G3Reader(ffile))
                    load_pipe.Add(collect)
                    load_pipe.Run()

            # Convert collected frames to an observation
            ob = self.obs_import(collect.frames)
            data.obs.append(ob)
            del collect

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()
