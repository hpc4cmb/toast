# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import ast
import json
from collections import OrderedDict

import h5py
import numpy as np

from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Unicode, trait_docs
from ..utils import Logger
from ..vis import set_matplotlib_backend
from .amplitudes import Amplitudes
from .template import Template


@trait_docs
class Periodic(Template):
    """This template represents amplitudes which are periodic in time.

    The template amplitudes are modeled as a value for each detector of each
    observation in a "bin" of the specified data values.  The min / max values
    of the periodic data are computed for each observation, and the binning
    between these min / max values is set by either the n_bins trait or by
    specifying the increment in the value to use for each bin.

    Although the data values used do not have to be strictly periodic, this
    template works best if the values are varying in a regular way such that
    each bin has approximately the same number of hits.

    The periodic quantity to consider can be either a shared or detdata field.

    The overall TemplateMatrix view is used to indicate valid samples.
    Separately, the `select_view` trait is used to restrict the samples to
    project into amplitudes.  This is useful (for example) if a periodic template
    has different amplitudes for different timespans in an observation, like
    right and left-going scans.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    det_data_units   : The units of the detector data
    #    det_mask         : Bitmask for per-detector flagging
    #    det_flags        : Optional detector solver flags
    #    det_flag_mask    : Bit mask for detector solver flags
    #

    periodic_key = Unicode(
        None, allow_none=True, help="Observation shared key for the periodic quantity"
    )

    periodic_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for flags",
    )

    periodic_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional flagging of periodic_key values",
    )

    select_view = Unicode(
        None,
        allow_none=True,
        help="Only solve for amplitudes within these intervals",
    )

    bins = Int(
        10,
        allow_none=True,
        help="Number of bins between min / max values of periodic key",
    )

    increment = Float(
        None,
        allow_none=True,
        help="The increment of the periodic key for each bin",
    )

    minimum_bin_hits = Int(3, help="Minimum number of samples per amplitude bin")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        log = Logger.get()
        if self.periodic_key is None:
            msg = "You must set periodic_key before initializing"
            raise RuntimeError(msg)

        if self.bins is not None and self.increment is not None:
            msg = "Only one of bins and increment can be specified"
            raise RuntimeError(msg)

        # Good detectors to use for each observation
        self._obs_dets = dict()

        # The sample flags for the combined intervals to use for each observation,
        # which is the intersection of `view` and `select_view` intervals, as well
        # as any flags for the key field.
        self._obs_flags = dict()

        # Find the binning for each observation
        self._obs_min = dict()
        self._obs_max = dict()
        self._obs_incr = dict()
        self._obs_nbins = dict()
        total_bins = 0
        for ob in new_data.obs:
            if self.periodic_key not in ob.shared:
                continue

            # The total intervals we are using for projection
            if self.select_view is not None:
                select = ob.intervals[self.view] & ob.intervals[self.select_view]
            else:
                select = ob.intervals[self.view]

            # The common flags we will use for this observation:  Cut samples
            # outside our selection intervals and any data where the periodic
            # key is flagged.
            self._obs_flags[ob.uid] = np.ones(ob.n_local_samples, dtype=np.uint8)
            for vw in select:
                self._obs_flags[ob.uid][vw.first : vw.last] = 0

            if self.periodic_flags is not None:
                self._obs_flags[ob.uid] |= (
                    ob.shared[self.periodic_flags].data & self.periodic_flag_mask
                )

            good = self._obs_flags[ob.uid] == 0
            omin = np.amin(ob.shared[self.periodic_key].data[good])
            omax = np.amax(ob.shared[self.periodic_key].data[good])

            if omin == omax:
                msg = f"Periodic data {self.periodic_key} is constant for observation "
                msg += f"{ob.name}"
                raise RuntimeError(msg)

            self._obs_min[ob.uid] = omin
            self._obs_max[ob.uid] = omax
            if self.bins is not None:
                obins = int(self.bins)
                oincr = (omax - omin) / obins
            else:
                oincr = float(self.increment)
                obins = int((omax - omin) / oincr)
            if obins == 0 and ob.comm.group_rank == 0:
                msg = f"Template {self.name}, obs {ob.name} has zero amplitude bins"
                log.warning(msg)
            total_bins += obins
            self._obs_nbins[ob.uid] = obins
            self._obs_incr[ob.uid] = oincr

            # Build up detector list
            self._obs_dets[ob.uid] = list()
            ddets = set(ob.detdata[self.det_data].detectors)
            for d in ob.select_local_detectors(flagmask=self.det_mask):
                if d not in ddets:
                    continue
                self._obs_dets[ob.uid].append(d)

        if total_bins == 0:
            msg = f"Template {self.name} process group {new_data.comm.group}"
            msg += " has zero amplitude bins- change the binning size."
            raise RuntimeError(msg)

        # Go through the data and compute the offsets into the amplitudes for each
        # observation and detector.

        self._det_offset = dict()
        offset = 0
        for ob in new_data.obs:
            self._det_offset[ob.uid] = dict()
            for det in self._obs_dets[ob.uid]:
                self._det_offset[ob.uid][det] = offset
                offset += np.sum(self._obs_nbins[ob.uid])

        # Now we know the total number of local amplitudes.

        if offset == 0:
            # This means that no observations included the shared key
            # we are using.
            msg = f"Data has no observations with key '{self.periodic_key}'."
            msg += "  You should disable this template."
            log.error(msg)
            raise RuntimeError(msg)

        self._n_local = offset
        if new_data.comm.comm_world is None:
            self._n_global = self._n_local
        else:
            self._n_global = new_data.comm.comm_world.allreduce(
                self._n_local, op=MPI.SUM
            )

        # Go through all the data and compute the number of hits per amplitude
        # bin and the flagging of bins.

        # Boolean flags
        if self._n_local == 0:
            self._amp_flags = None
        else:
            self._amp_flags = np.zeros(self._n_local, dtype=bool)

        # Hits
        if self._n_local == 0:
            self._amp_hits = None
        else:
            self._amp_hits = np.zeros(self._n_local, dtype=np.int32)

        for ob in new_data.obs:
            for det in self._obs_dets[ob.uid]:
                amp_offset = self._det_offset[ob.uid][det]
                nbins = self._obs_nbins[ob.uid]
                amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
                amp_flags = self._amp_flags[amp_offset : amp_offset + nbins]

                good, amp_indx = self._flags_and_index(ob, det)
                n_good = np.count_nonzero(good)

                np.add.at(
                    amp_hits,
                    amp_indx,
                    np.ones(n_good, dtype=np.int32),
                )
                # Flag amplitudes based on hits
                flag_thresh = amp_hits < self.minimum_bin_hits
                amp_flags[flag_thresh] = True
        return

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        if z.local_flags is not None:
            z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _flags_and_index(self, obs, det):
        """Get the good samples and amplitude indices for one detector."""
        nbins = self._obs_nbins[obs.uid]
        if self.det_flags is None:
            flags = self._obs_flags[obs.uid]
        else:
            flags = np.copy(self._obs_flags[obs.uid])
            flags |= obs.detdata[self.det_flags][det] & self.det_flag_mask
        good = flags == 0
        amp_indx = np.array(
            (
                (obs.shared[self.periodic_key].data[good] - self._obs_min[obs.uid])
                / self._obs_incr[obs.uid]
            ),
            dtype=np.int32,
        )
        # Place any amplitudes at the max value into the final bin
        overflow = amp_indx == nbins
        amp_indx[overflow] = nbins - 1
        return good, amp_indx

    def _add_to_signal(self, obs, detectors, amplitudes, **kwargs):
        for det in detectors:
            if det not in self._det_offset[obs.uid]:
                continue
            amp_offset = self._det_offset[obs.uid][det]
            nbins = self._obs_nbins[obs.uid]
            good, amp_indx = self._flags_and_index(obs, det)
            amps = amplitudes.local[amp_offset : amp_offset + nbins]

            # Accumulate to timestream
            obs.detdata[self.det_data][det][good] += amps[amp_indx]

    def _project_signal(self, obs, detectors, amplitudes, **kwargs):
        for det in detectors:
            if det not in self._det_offset[obs.uid]:
                continue
            amp_offset = self._det_offset[obs.uid][det]
            nbins = self._obs_nbins[obs.uid]
            good, amp_indx = self._flags_and_index(obs, det)
            amps = amplitudes.local[amp_offset : amp_offset + nbins]

            # Accumulate to amplitudes
            np.add.at(
                amps,
                amp_indx,
                obs.detdata[self.det_data][det][good],
            )

    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        # No prior for this template, nothing to accumulate to output.
        return

    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        # Apply weights based on the number of samples hitting each
        # amplitude bin.
        for ob in self.data.obs:
            for det in self._obs_dets[ob.uid]:
                amp_offset = self._det_offset[ob.uid][det]
                nbins = self._obs_nbins[ob.uid]
                amps_in = amplitudes_in.local[amp_offset : amp_offset + nbins]
                amps_out = amplitudes_out.local[amp_offset : amp_offset + nbins]
                amp_flags = amplitudes_in.local_flags[amp_offset : amp_offset + nbins]
                amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
                amp_good = amp_flags == 0
                amps_out[amp_good] = amps_in[amp_good] * amp_hits[amp_good]

    @function_timer
    def write(self, amplitudes, out):
        """Write out amplitude values.

        This stores the amplitudes to a file for debugging / plotting.

        Args:
            amplitudes (Amplitudes):  The amplitude data.
            out (str):  The output file.

        Returns:
            None

        """
        # By definition, when solving for something that is periodic for
        # a given detector in a single observation, we (should) have many
        # fewer template amplitudes than timestream samples.  Because of
        # this we assume we can make several copies for extracting the
        # amplitudes and gathering them for writing.

        obs_det_amps = dict()

        for ob in self.data.obs:
            for det in self._obs_dets[ob.uid]:
                if self.periodic_key not in ob.shared:
                    continue
                if ob.name not in obs_det_amps:
                    obs_det_amps[ob.name] = dict()
                amp_offset = self._det_offset[ob.uid][det]
                nbins = self._obs_nbins[ob.uid]
                amp_data = amplitudes.local[amp_offset : amp_offset + nbins]
                amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
                amp_flags = self._amp_flags[amp_offset : amp_offset + nbins]
                obs_det_amps[ob.name][det] = {
                    "amps": amp_data,
                    "hits": amp_hits,
                    "flags": amp_flags,
                    "min": self._obs_min[ob.uid],
                    "max": self._obs_max[ob.uid],
                    "incr": self._obs_incr[ob.uid],
                }

        if self.data.comm.world_size == 1:
            all_obs_dets_amps = [obs_det_amps]
        else:
            all_obs_dets_amps = self.data.comm.comm_world.gather(obs_det_amps, root=0)

        if self.data.comm.world_rank == 0:
            obs_det_amps = dict()
            for pdata in all_obs_dets_amps:
                for obname in pdata.keys():
                    if obname not in obs_det_amps:
                        obs_det_amps[obname] = dict()
                    obs_det_amps[obname].update(pdata[obname])
            del all_obs_dets_amps
            with h5py.File(out, "w") as hf:
                for obname, obamps in obs_det_amps.items():
                    n_det = len(obamps)
                    det_list = list(sorted(obamps.keys()))
                    det_indx = {y: x for x, y in enumerate(det_list)}
                    indx_to_det = {det_indx[x]: x for x in det_list}
                    n_amp = len(obamps[det_list[0]]["amps"])
                    amp_min = [obamps[x]["min"] for x in det_list]
                    amp_max = [obamps[x]["max"] for x in det_list]
                    amp_incr = [obamps[x]["incr"] for x in det_list]

                    # Create datasets for this observation
                    hg = hf.create_group(obname)
                    hg.attrs["detectors"] = json.dumps(det_list)
                    hg.attrs["min"] = json.dumps(amp_min)
                    hg.attrs["max"] = json.dumps(amp_max)
                    hg.attrs["incr"] = json.dumps(amp_incr)
                    hamps = hg.create_dataset(
                        "amplitudes",
                        (n_det, n_amp),
                        dtype=np.float64,
                    )
                    hhits = hg.create_dataset(
                        "hits",
                        (n_det, n_amp),
                        dtype=np.int32,
                    )
                    hflags = hg.create_dataset(
                        "flags",
                        (n_det, n_amp),
                        dtype=np.uint8,
                    )

                    # Write data
                    for idet in range(n_det):
                        det = indx_to_det[idet]
                        dprops = obamps[det]
                        hslice = (slice(idet, idet + 1, 1), slice(0, n_amp, 1))
                        dslice = (slice(0, n_amp, 1),)
                        hamps.write_direct(dprops["amps"], dslice, hslice)
                        hhits.write_direct(dprops["hits"], dslice, hslice)
                        hflags.write_direct(dprops["flags"], dslice, hslice)


def plot(amp_file, out_root=None):
    """Plot an amplitude dump file.

    This loads an amplitude file and makes a set of plots.

    Args:
        amp_file (str):  The path to the input file of amplitudes.
        out_root (str):  The root of the output files.

    Returns:
        None

    """

    if out_root is not None:
        set_matplotlib_backend(backend="pdf")

    import matplotlib.pyplot as plt

    figdpi = 100

    with h5py.File(amp_file, "r") as hf:
        for obname, obgrp in hf.items():
            det_list = json.loads(obgrp.attrs["detectors"])
            amp_min = list(ast.literal_eval(obgrp.attrs["min"]))
            amp_max = list(ast.literal_eval(obgrp.attrs["max"]))
            amp_incr = list(ast.literal_eval(obgrp.attrs["incr"]))

            hamps = np.array(obgrp["amplitudes"])
            hhits = np.array(obgrp["hits"])
            hflags = np.array(obgrp["flags"])
            n_bin = hamps.shape[1]
            bad = hflags != 0
            hamps[bad] = np.nan

            for idet, det in enumerate(det_list):
                outfile = f"{out_root}_{obname}_{det}.pdf"
                xdata = amp_min[idet] + amp_incr[idet] * np.arange(
                    n_bin, dtype=np.float64
                )
                fig = plt.figure(dpi=figdpi, figsize=(8, 12))
                ax = fig.add_subplot(3, 1, 1)
                ax.step(xdata, hamps[idet], where="post", label=f"{det}")
                ax.set_xlabel("Shared Data Value")
                ax.set_ylabel("Amplitude")
                ax.legend(loc="best")
                ax = fig.add_subplot(3, 1, 2)
                ax.step(xdata, hflags[idet], where="post", label=f"{det}")
                ax.set_xlabel("Shared Data Value")
                ax.set_ylabel("Flags")
                ax.legend(loc="best")
                ax = fig.add_subplot(3, 1, 3)
                ax.step(xdata, hhits[idet], where="post", label=f"{det}")
                ax.set_xlabel("Shared Data Value")
                ax.set_ylabel("Hits")
                ax.legend(loc="best")
                if out_root is None:
                    # Interactive
                    plt.show()
                else:
                    plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
                    plt.close()
