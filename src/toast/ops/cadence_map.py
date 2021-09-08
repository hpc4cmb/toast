# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import h5py
import os
from time import time
import warnings

from astropy import units as u
import numpy as np
import traitlets

from ..mpi import MPI, MPI_Comm, use_mpi, Comm

from .operator import Operator
from .. import qarray as qa
from ..data import Data
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Instance
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned
from ..observation import default_names as obs_names
from ..coordinates import to_MJD


@trait_docs
class CadenceMap(Operator):
    """Tabulate which days each pixel on the map is visited."""

    # Class traits

    pixel_dist = Unicode(
        None,
        allow_none=True,
        help="The Data key containing the submap distribution",
    )

    pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator.  "
        "Used exclusively for pixel numbers, not pointing weights.",
    )

    times = Unicode(obs_names.times, help="Observation shared key for timestamps")

    det_flags = Unicode(
        obs_names.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(255, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        obs_names.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    @traitlets.validate("pointing")
    def _check_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError("pointing should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["pixels", "weights", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = "pointing operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.pixel_dist is None:
            raise RuntimeError(
                "You must set the 'pixel_dist' trait before calling exec()"
            )
        elif self.pixel_dist not in data:
            raise RuntimeError(
                f"Pixel distribution '{self.pixel_dist}' does not exist in data."
            )

        comm = data.comm.comm_world
        rank = data.comm.world_rank

        # Get the total number of pixels from the pixel distribution

        npix = data[self.pixel_dist].n_pix

        if rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        # determine the number of modified Julian days

        tmin = 1e30
        tmax = -1e30
        for obs in data.obs:
            times = obs.shared[self.times].data
            tmin = min(tmin, times[0])
            tmax = max(tmax, times[-1])

        if comm is not None:
            tmin = comm.allreduce(tmin, MPI.MIN)
            tmax = comm.allreduce(tmax, MPI.MAX)

        MJD_start = int(to_MJD(tmin))
        MJD_stop = int(to_MJD(tmax)) + 1
        nday = MJD_stop - MJD_start

        # Flag all pixels that are observed on each MJD

        if rank == 0:
            all_hit = np.zeros([nday, npix], dtype=bool)

        buflen = 10  # Number of days to process at once
        # FIXME : We should use `buflen` also for the HDF5 dataset size
        buf = np.zeros([buflen, npix], dtype=bool)
        day_start = MJD_start
        while day_start < MJD_stop:
            day_stop = min(MJD_stop, day_start + buflen)
            if rank == 0:
                log.debug(
                    f"Processing {MJD_start} <= {day_start} - {day_stop} <= {MJD_stop}"
                )
            buf[:, :] = False
            for obs in data.obs:
                dets = obs.select_local_detectors(detectors)
                times = obs.shared[self.times].data
                days = to_MJD(times).astype(int)
                if days[0] >= day_stop or days[-1] < day_start:
                    continue
                if self.shared_flags:
                    cflag = (
                        obs.shared[self.shared_flags].data & self.shared_flag_mask
                    ) != 0
                for day in range(day_start, day_stop):
                    # Find samples that were collected on target day ...
                    good = days == day
                    if not np.any(good):
                        continue
                    if self.shared_flags:
                        # ... and are not flagged ...
                        good[cflag] = False
                    for det in dets:
                        if self.det_flags:
                            # ... even by detector flags
                            flag = obs.detdata[self.det_flags][det] & self.flag_mask
                            mask = np.logical_and(good, flag == 0)
                        else:
                            mask = good
                        # Compute pixel numbers.  Will do nothing if they already exist.
                        obs_data = Data(comm=data.comm)
                        obs_data._internal = data._internal
                        obs_data.obs = [obs]
                        self.pointing.apply(obs_data, detectors=[det])
                        obs_data.obs.clear()
                        del obs_data
                        # Flag the hit pixels
                        pixels = obs.detdata[self.pointing.pixels][det]
                        mask[pixels < 0] = False
                        buf[day - day_start][pixels[mask]] = True
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.LOR)
            if rank == 0:
                for i in range(day_start, day_stop):
                    all_hit[i - MJD_start] = buf[i - day_start]
            day_start = day_stop

        if rank == 0:
            fname = os.path.join(self.output_dir, f"{self.name}_cadence.h5")
            with h5py.File(fname, "w") as f:
                dset = f.create_dataset("cadence", data=all_hit)
                dset.attrs["MJDSTART"] = MJD_start
                dset.attrs["MJDSTOP"] = MJD_stop
                dset.attrs["NESTED"] = self.pointing.nest
            log.info(f"Wrote cadence map to {fname}.")

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pointing.requires()
        req["shared"].append(self.times)
        req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return {}

    def _accelerators(self):
        return list()
