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

    is_detdata_key = Bool(
        False,
        help="If True, the periodic data and flags are detector fields, not shared",
    )

    key = Unicode(
        None, allow_none=True, help="Observation data key for the periodic quantity"
    )

    flags = Unicode(
        None,
        allow_none=True,
        help="Observation data key for flags to use",
    )

    flag_mask = Int(0, help="Bit mask value for flags")

    bins = Int(
        10,
        allow_none=True,
        help="Number of bins between min / max values of data key",
    )

    increment = Float(
        None,
        allow_none=True,
        help="The increment of the data key for each bin",
    )

    minimum_bin_hits = Int(3, help="Minimum number of samples per amplitude bin")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        log = Logger.get()
        if self.key is None:
            msg = "You must set key before initializing"
            raise RuntimeError(msg)

        if self.bins is not None and self.increment is not None:
            msg = "Only one of bins and increment can be specified"
            raise RuntimeError(msg)

        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        all_dets = OrderedDict()

        # Good detectors to use for each observation
        self._obs_dets = dict()

        # Find the binning for each observation and the total detectors on this
        # process.
        self._obs_min = list()
        self._obs_max = list()
        self._obs_incr = list()
        self._obs_nbins = list()
        total_bins = 0
        for iob, ob in enumerate(new_data.obs):
            if self.is_detdata_key:
                if self.key not in ob.detdata:
                    continue
            else:
                if self.key not in ob.shared:
                    continue
            omin = None
            omax = None
            for vw in ob.intervals[self.view].data:
                vw_slc = slice(vw.first, vw.last, 1)
                good = slice(None)
                if self.is_detdata_key:
                    vw_data = ob.detdata[self.key].data[vw_slc]
                    if self.flags is not None:
                        # We have some flags
                        bad = ob.detdata[self.flags].data[vw_slc] & self.flag_mask
                        good = np.logical_not(bad)
                else:
                    vw_data = ob.shared[self.key].data[vw_slc]
                    if self.flags is not None:
                        # We have some flags
                        bad = ob.shared[self.flags].data[vw_slc] & self.flag_mask
                        good = np.logical_not(bad)
                vmin = np.amin(vw_data[good])
                vmax = np.amax(vw_data[good])
                if omin is None:
                    omin = vmin
                    omax = vmax
                else:
                    omin = min(omin, vmin)
                    omax = max(omax, vmax)

            if omin == omax:
                msg = f"Periodic data {self.key} is constant for observation "
                msg += f"{ob.name}"
                raise RuntimeError(msg)
            self._obs_min.append(omin)
            self._obs_max.append(omax)
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
            self._obs_nbins.append(obins)
            self._obs_incr.append(oincr)

            # Build up detector list
            self._obs_dets[iob] = set()
            for d in ob.select_local_detectors(flagmask=self.det_mask):
                if d not in ob.detdata[self.det_data].detectors:
                    continue
                self._obs_dets[iob].add(d)
                if d not in all_dets:
                    all_dets[d] = None

        self._all_dets = list(all_dets.keys())

        if total_bins == 0:
            msg = f"Template {self.name} process group {new_data.comm.group}"
            msg += f" has zero amplitude bins- change the binning size."
            raise RuntimeError(msg)

        # During application of the template, we will be looping over detectors
        # in the outer loop.  So we pack the amplitudes by detector and then by
        # observation.  Compute the per-detector offsets into the amplitudes.

        self._det_offset = dict()

        offset = 0
        for det in self._all_dets:
            self._det_offset[det] = offset
            for iob, ob in enumerate(new_data.obs):
                if det not in self._obs_dets[iob]:
                    continue
                if self.is_detdata_key:
                    if self.key not in ob.detdata:
                        continue
                else:
                    if self.key not in ob.shared:
                        continue
                offset += self._obs_nbins[iob]

        # Now we know the total number of local amplitudes.

        if offset == 0:
            # This means that no observations included the shared key
            # we are using.
            msg = f"Data has no observations with key '{self.key}'."
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

        self._obs_bin_hits = list()
        for det in self._all_dets:
            amp_offset = self._det_offset[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in self._obs_dets[iob]:
                    continue
                if self.is_detdata_key:
                    if self.key not in ob.detdata:
                        continue
                else:
                    if self.key not in ob.shared:
                        continue
                nbins = self._obs_nbins[iob]
                det_indx = ob.detdata[self.det_data].indices([det])[0]
                amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
                amp_flags = self._amp_flags[amp_offset : amp_offset + nbins]
                if self.det_flags is not None:
                    flag_indx = ob.detdata[self.det_flags].indices([det])[0]
                else:
                    flag_indx = None
                for vw in ob.intervals[self.view].data:
                    vw_slc = slice(vw.first, vw.last, 1)
                    if self.is_detdata_key:
                        vw_data = ob.detdata[self.key].data[vw_slc]
                    else:
                        vw_data = ob.shared[self.key].data[vw_slc]
                    good, amp_indx = self._view_flags_and_index(
                        det_indx,
                        iob,
                        ob,
                        vw,
                        flag_indx=flag_indx,
                        det_flags=True,
                    )
                    np.add.at(
                        amp_hits,
                        amp_indx,
                        np.ones(len(vw_data[good]), dtype=np.int32),
                    )
                    flag_thresh = amp_hits < self.minimum_bin_hits
                    amp_flags[flag_thresh] = True
                amp_offset += nbins
        return

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        if z.local_flags is not None:
            z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _view_flags_and_index(
        self, det_indx, ob_indx, ob, view, flag_indx=None, det_flags=False
    ):
        """Get the flags and amplitude indices for one detector and view."""
        vw_slc = slice(view.first, view.last, 1)
        vw_len = view.last - view.first
        incr = self._obs_incr[ob_indx]
        # Determine good samples
        if self.is_detdata_key:
            vw_data = ob.detdata[self.key].data[vw_slc]
            if self.flags is not None:
                # We have some flags
                bad = ob.detdata[self.flags].data[vw_slc] & self.flag_mask
            else:
                bad = np.zeros(vw_len, dtype=np.uint8)
        else:
            vw_data = ob.shared[self.key].data[vw_slc]
            if self.flags is not None:
                # We have some flags
                bad = ob.shared[self.flags].data[vw_slc] & self.flag_mask
            else:
                bad = np.zeros(vw_len, dtype=np.uint8)
        if det_flags and self.det_flags is not None:
            # We have some det flags
            bad |= ob.detdata[self.det_flags][flag_indx, vw_slc] & self.det_flag_mask
        good = np.logical_not(bad)

        # Find the amplitude index for every good sample
        amp_indx = np.array(
            ((vw_data[good] - self._obs_min[ob_indx]) / incr),
            dtype=np.int32,
        )
        overflow = amp_indx >= self._obs_nbins[ob_indx]
        amp_indx[overflow] = self._obs_nbins[ob_indx] - 1

        return good, amp_indx

    def _add_to_signal(self, detector, amplitudes, **kwargs):
        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return

        amp_offset = self._det_offset[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in self._obs_dets[iob]:
                continue
            if self.is_detdata_key:
                if self.key not in ob.detdata:
                    continue
            else:
                if self.key not in ob.shared:
                    continue
            nbins = self._obs_nbins[iob]
            det_indx = ob.detdata[self.det_data].indices([detector])[0]
            amps = amplitudes.local[amp_offset : amp_offset + nbins]
            for vw in ob.intervals[self.view].data:
                vw_slc = slice(vw.first, vw.last, 1)
                good, amp_indx = self._view_flags_and_index(
                    det_indx,
                    iob,
                    ob,
                    vw,
                    det_flags=False,
                )
                # Accumulate to timestream
                ob.detdata[self.det_data][det_indx, vw_slc][good] += amps[amp_indx]
            amp_offset += nbins

    def _project_signal(self, detector, amplitudes, **kwargs):
        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return

        amp_offset = self._det_offset[detector]
        for iob, ob in enumerate(self.data.obs):
            if detector not in self._obs_dets[iob]:
                continue
            if self.is_detdata_key:
                if self.key not in ob.detdata:
                    continue
            else:
                if self.key not in ob.shared:
                    continue
            nbins = self._obs_nbins[iob]
            det_indx = ob.detdata[self.det_data].indices([detector])[0]
            amps = amplitudes.local[amp_offset : amp_offset + nbins]
            if self.det_flags is not None:
                flag_indx = ob.detdata[self.det_flags].indices([detector])[0]
            else:
                flag_indx = None
            for vw in ob.intervals[self.view].data:
                vw_slc = slice(vw.first, vw.last, 1)
                good, amp_indx = self._view_flags_and_index(
                    det_indx,
                    iob,
                    ob,
                    vw,
                    flag_indx=flag_indx,
                    det_flags=True,
                )
                # Accumulate to amplitudes
                np.add.at(
                    amps,
                    amp_indx,
                    ob.detdata[self.det_data][det_indx, vw_slc][good],
                )
            amp_offset += nbins

    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        # No prior for this template, nothing to accumulate to output.
        return

    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        # Apply weights based on the number of samples hitting each
        # amplitude bin.
        for det in self._all_dets:
            amp_offset = self._det_offset[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in self._obs_dets[iob]:
                    continue
                if self.is_detdata_key:
                    if self.key not in ob.detdata:
                        continue
                else:
                    if self.key not in ob.shared:
                        continue
                nbins = self._obs_nbins[iob]
                amps_in = amplitudes_in.local[amp_offset : amp_offset + nbins]
                amps_out = amplitudes_out.local[amp_offset : amp_offset + nbins]
                amp_flags = amplitudes_in.local_flags[amp_offset : amp_offset + nbins]
                amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
                amp_good = amp_flags == 0

                amps_out[amp_good] = amps_in[amp_good] * amp_hits[amp_good]

                amp_offset += nbins

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

        for det in self._all_dets:
            amp_offset = self._det_offset[det]
            for iob, ob in enumerate(self.data.obs):
                if det not in self._obs_dets[iob]:
                    continue
                if self.is_detdata_key:
                    if self.key not in ob.detdata:
                        continue
                else:
                    if self.key not in ob.shared:
                        continue
                if ob.name not in obs_det_amps:
                    obs_det_amps[ob.name] = dict()
                nbins = self._obs_nbins[iob]
                amp_data = amplitudes.local[amp_offset : amp_offset + nbins]
                amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
                amp_flags = self._amp_flags[amp_offset : amp_offset + nbins]
                obs_det_amps[ob.name][det] = {
                    "amps": amp_data,
                    "hits": amp_hits,
                    "flags": amp_flags,
                    "min": self._obs_min[iob],
                    "max": self._obs_max[iob],
                    "incr": self._obs_incr[iob],
                }
                amp_offset += nbins

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
