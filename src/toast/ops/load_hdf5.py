# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import glob
import os

import h5py
import numpy as np
import traitlets

from ..dist import distribute_discrete
from ..io import load_hdf5
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, List, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class LoadHDF5(Operator):
    """Operator which loads HDF5 data files into observations.

    This operator expects a top-level volume directory.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    volume = Unicode(
        None, allow_none=True, help="Top-level directory containing the data volume"
    )

    # FIXME:  We should add a filtering mechanism here to load a subset of
    # observations and / or detectors, as well as figure out subdirectory organization.

    meta = List(None, allow_none=True, help="Only load this list of meta objects")

    detdata = List(None, allow_none=True, help="Only load this list of detdata objects")

    shared = List(None, allow_none=True, help="Only load this list of shared objects")

    intervals = List(
        None, allow_none=True, help="Only load this list of intervals objects"
    )

    sort_by_size = Bool(
        False, help="If True, sort observations by size before load balancing"
    )

    process_rows = Int(
        None,
        allow_none=True,
        help="The size of the rectangular process grid in the detector direction.",
    )

    force_serial = Bool(
        False, help="Use serial HDF5 operations, even if parallel support available"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        meta_fields = None
        if self.meta is not None and len(self.meta) > 0:
            meta_fields = list(self.meta)

        shared_fields = None
        if self.shared is not None and len(self.shared) > 0:
            shared_fields = list(self.shared)

        detdata_fields = None
        if self.detdata is not None and len(self.detdata) > 0:
            detdata_fields = list(self.detdata)

        intervals_fields = None
        if self.intervals is not None and len(self.intervals) > 0:
            intervals_fields = list(self.intervals)

        # FIXME:  Eventually we will use the volume index / DB to select observations
        # and their sizes for a load-balanced assignment.  For now, we read this from
        # the file.

        # One global process computes the list of observations and their approximate
        # relative size.

        def _get_obs_samples(path):
            n_samp = 0
            with h5py.File(path, "r") as hf:
                n_samp = int(hf.attrs["observation_samples"])
            return n_samp

        obs_props = list()
        if data.comm.world_rank == 0:
            for root, dirs, files in os.walk(self.volume):
                # Process top-level files
                obsfiles = glob.glob(os.path.join(root, "*.h5"))
                if len(obsfiles) > 0:
                    # Root directory contains observations
                    for ofile in sorted(obsfiles):
                        fsize = _get_obs_samples(ofile)
                        obs_props.append((fsize, ofile))
                # Also process sub-directories one level deep
                for d in dirs:
                    obsfiles = glob.glob(os.path.join(root, d, "*.h5"))
                    if len(obsfiles) > 0:
                        # This directory contains observations
                        for ofile in sorted(obsfiles):
                            fsize = _get_obs_samples(ofile)
                            obs_props.append((fsize, ofile))
                break
        if self.sort_by_size:
            obs_props.sort(key=lambda x: x[0])
        else:
            obs_props.sort(key=lambda x: x[1])
        if data.comm.comm_world is not None:
            obs_props = data.comm.comm_world.bcast(obs_props, root=0)

        # Distribute observations among groups

        obs_sizes = [x[0] for x in obs_props]
        groupdist = distribute_discrete(obs_sizes, data.comm.ngroups)
        group_firstobs = groupdist[data.comm.group].offset
        group_numobs = groupdist[data.comm.group].n_elem

        # Every group loads its observations
        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            obfile = obs_props[obindx][1]

            ob = load_hdf5(
                obfile,
                data.comm,
                process_rows=self.process_rows,
                meta=meta_fields,
                detdata=detdata_fields,
                shared=shared_fields,
                intervals=intervals_fields,
                force_serial=self.force_serial,
            )

            data.obs.append(ob)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()
