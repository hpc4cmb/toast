# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import warnings
from time import time

import h5py
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..coordinates import to_MJD
from ..data import Data
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, trait_docs
from ..utils import Environment, GlobalTimers, Logger, Timer, dtype_to_aligned
from .operator import Operator
from .pointing import BuildPixelDistribution


@trait_docs
class CadenceMap(Operator):
    """Tabulate which days each pixel on the map is visited."""

    # Class traits

    pixel_dist = Unicode(
        None,
        allow_none=True,
        help="The Data key containing the submap distribution",
    )

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pixel pointing operator.",
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional telescope flagging",
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    save_pointing = Bool(False, help="If True, do not clear pixel numbers after use")

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "pixel_pointing", "pixel_dist":
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        comm = data.comm.comm_world
        rank = data.comm.world_rank

        # We need the pixel distribution to know total number of pixels

        if self.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                shared_flags=self.shared_flags,
                shared_flag_mask=self.shared_flag_mask,
                save_pointing=self.save_pointing,
            )
            log.info_rank("Caching pixel distribution", comm=data.comm.comm_world)
            pix_dist.apply(data)

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
                obs_data = data.select(obs_uid=obs.uid)
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
                            flag = obs.detdata[self.det_flags][det] & self.det_flag_mask
                            mask = np.logical_and(good, flag == 0)
                        else:
                            mask = good
                        # Compute pixel numbers.  Will do nothing if they already exist.
                        self.pixel_pointing.apply(obs_data, detectors=[det])
                        # Flag the hit pixels
                        pixels = obs.detdata[self.pixel_pointing.pixels][det]
                        mask[pixels < 0] = False
                        buf[day - day_start][pixels[mask]] = True
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.LOR)
            if rank == 0:
                for i in range(day_start, day_stop):
                    all_hit[i - MJD_start] = buf[i - day_start]
            day_start = day_stop

        if rank == 0:
            fname = os.path.join(self.output_dir, f"{self.name}.h5")
            with h5py.File(fname, "w") as f:
                dset = f.create_dataset("cadence", data=all_hit)
                dset.attrs["MJDSTART"] = MJD_start
                dset.attrs["MJDSTOP"] = MJD_stop
                dset.attrs["NESTED"] = self.pixel_pointing.nest
            log.info(f"Wrote cadence map to {fname}.")

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req["shared"].append(self.times)
        req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return {}
