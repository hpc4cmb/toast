# Copyright (c) 2021-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re

import h5py

from ..dist import distribute_discrete
from ..io import load_hdf5, VolumeIndex
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Dict, Int, List, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class LoadHDF5(Operator):
    """Operator which loads HDF5 data files into observations.

    Selected observation files are distributed across process groups in a load-balanced
    way, using the number of samples and (if using an index), the number of valid
    detectors in each observation.

    The list of observation files can be built in multiple ways:

    - If `files` is specified, that is used and `volume` is ignored.

    - If `volume_select` is specified, it should contain the conditional portion of the
      SQL select command (e.g. "where X < Y").  This will be used to extract the
      observation metadata needed to load files.

    - If `volume_select` is not specified then all observations are chosen (either
      from the volume index if specified, or using the filesystem).  This full list
      of observations then has the regex `pattern` applied to the basename of each
      observation file.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    volume = Unicode(
        None, allow_none=True, help="Top-level directory containing the data volume"
    )

    volume_index = Unicode(
        "DEFAULT",
        allow_none=True,
        help=(
            "Path to index file.  None disables use of index.  "
            "'DEFAULT' uses the default VolumeIndex name at the top of the volume."
        ),
    )

    volume_select = Unicode(
        None,
        allow_none=True,
        help="SQL selection string applied to the observation table of the index.",
    )

    pattern = Unicode(
        r".*\.h5",
        help="Regexp pattern to match files against (if not using volume_select).",
    )

    files = List(
        [],
        help="Override `volume` and load a list of files",
    )

    meta = List([], help="Only load this list of meta objects")

    detdata = List(
        [defaults.det_data, defaults.det_flags],
        help="Only load this list of detdata objects",
    )

    det_select = Dict(
        {}, help="Keep a subset of detectors whose focalplane columns match."
    )

    shared = List([], help="Only load this list of shared objects")

    intervals = List([], help="Only load this list of intervals objects")

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
        meta_fields = None
        if len(self.meta) > 0:
            meta_fields = list(self.meta)

        shared_fields = None
        if len(self.shared) > 0:
            shared_fields = list(self.shared)

        if len(self.detdata) > 0:
            detdata_fields = list(self.detdata)
        else:
            detdata_fields = list()

        intervals_fields = None
        if len(self.intervals) > 0:
            intervals_fields = list(self.intervals)

        # Get our list of observation files and relative sizes for load-balancing
        obs_props = self._get_obs_props(data.comm.comm_world)

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
                det_select=self.det_select,
            )

            data.obs.append(ob)

        return

    def _get_obs_samples(self, path):
        """Helper function to extract the number of samples in each observation."""
        n_samp = 0
        with h5py.File(path, "r") as hf:
            n_samp = int(hf.attrs["observation_samples"])
        return n_samp

    def _get_obs_props(self, comm):
        """Query the index or filesystem for observations and their properties."""
        log = Logger.get()
        if comm is None:
            rank = 0
        else:
            rank = comm.rank

        # Index for the volume, if we are using it.
        if self.volume_index is None or len(self.files) > 0:
            # Either we are loading a list of files or disabling use of the index
            vindx = None
        else:
            if self.volume_index == "DEFAULT":
                index_path = os.path.join(self.volume, VolumeIndex.default_name)
            else:
                index_path = self.volume_index
            index_exists = False
            if rank == 0:
                if os.path.isfile(index_path):
                    index_exists = True
            if comm is not None:
                index_exists = comm.bcast(index_exists, root=0)
            if index_exists:
                vindx = VolumeIndex(index_path)
            else:
                msg = f"Volume index '{index_path}' does not exist, scanning filesystem for observations."
                log.warning_rank(msg, comm=comm)
                vindx = None

        # If using the index, the fields we are querying
        select_prefix = "select name, path, samples, valid_dets from "
        select_prefix += VolumeIndex.obs_table
        select_prefix += " "

        obs_props = None
        if rank == 0:
            # Perform all filesystem and index queries on one process.
            obs_props = list()
            if len(self.files) > 0:
                # The user gave an explicit list of files.  We use the number of
                # samples for load-balancing.
                if self.volume is not None:
                    log.warning(
                        f'LoadHDF5: volume="{self.volume}" trait overridden by '
                        f"files={self.files}"
                    )
                for ofile in self.files:
                    fsize = self._get_obs_samples(ofile)
                    obs_props.append((fsize, ofile))
                msg = "LoadHDF5 using specified file list with sizes: "
                msg += f"{obs_props}"
                log.verbose(msg)
            else:
                # We are using the volume trait.  Get the list of relative file paths
                # for each observation.
                if self.volume is None:
                    msg = "Either the volume or a list of files must be specified"
                    log.error(msg)
                    raise RuntimeError(msg)
                if self.volume_select is None:
                    # We are selecting all observations, and matching a regex.
                    if vindx is None:
                        # We have no index, just check the filesystem.  This is slow
                        # for many observations.  Use the number of samples for load
                        # balancing.
                        rel_files = VolumeIndex.find_observations(
                            self.volume, pattern_str=self.pattern
                        )
                        for rfile in rel_files:
                            full_path = os.path.join(self.volume, rfile)
                            fsize = self._get_obs_samples(full_path)
                            obs_props.append((fsize, full_path))
                        msg = "LoadHDF5 using volume with NO index and matching"
                        msg += f" filename pattern '{self.pattern}' found sizes: "
                        msg += f"{obs_props}"
                        log.verbose(msg)
                    else:
                        # We are using the index.  Get the full list of obs and then
                        # apply the regex.  We use the number of valid detector-samples
                        # for load balancing.
                        pattern = re.compile(self.pattern)
                        all_obs = vindx.select(select_prefix)
                        for oname, opath, osamples, odets in all_obs:
                            if pattern.search(opath) is not None:
                                sz = osamples * odets
                                full_path = os.path.join(self.volume, opath)
                                obs_props.append((sz, full_path))
                        msg = "LoadHDF5 using volume WITH index and matching"
                        msg += f" filename pattern '{self.pattern}' found sizes: "
                        msg += f"{obs_props}"
                        log.verbose(msg)
                else:
                    # Using a custom selection with the index.  Use the number of
                    # valid detector-samples for load balancing.
                    sel_str = f"{select_prefix}{self.volume_select}"
                    sel_obs = vindx.select(sel_str)
                    for oname, opath, osamples, odets in sel_obs:
                        sz = osamples * odets
                        full_path = os.path.join(self.volume, opath)
                        obs_props.append((sz, full_path))
                    msg = f"LoadHDF5 using volume with query '{self.volume_select}'"
                    msg += f" found sizes: {obs_props}"
                    log.verbose(msg)
            if self.sort_by_size:
                obs_props.sort(key=lambda x: x[0])
            else:
                obs_props.sort(key=lambda x: x[1])
        if comm is not None:
            obs_props = comm.bcast(obs_props, root=0)
        return obs_props

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return dict()
