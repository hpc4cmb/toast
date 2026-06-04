# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import ast
import json
import re
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
class PreComputed(Template):
    """This template represents amplitudes of precomputed timestream contributions.

    Given a constant, pre-computed, time-domain array, this class represents
    amplitudes which are coefficients used to project that array into detector
    timestreams.  The amplitudes are solved per-detector and optionally per-interval.

    If `select_view` is not None, it is a set of intervals used to group sample
    ranges for independent estimation of the template amplitudes.  These intervals
    are taken with the intersection of the solving view passed from the template
    matrix.

    The templates themselves must be in a specific format:

    - The `obs_key` must point to a template dictionary.

    - Within this template dictionary, the `det_to_key` lookup table maps detector
      names to a corresponding template key.

    - The templates must have exactly the same length as the number of samples
      in the detector data.

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

    obs_key = Unicode(
        None, allow_none=True, help="Observation key for timestream templates"
    )

    select_view = Unicode(
        None,
        allow_none=True,
        help="Intervals to use for breaking up regions with independent amplitudes",
    )

    good_fraction = Float(
        0.2,
        help="Fraction of unflagged samples needed to keep a given amplitude estimate",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        log = Logger.get()
        if self.obs_key is None:
            msg = "You must set the obs_key before initializing"
            raise RuntimeError(msg)

        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        # FIXME: not needed once accum_load branch merged
        self._all_dets = OrderedDict()

        # Good detectors to use for each observation
        self._obs_dets = dict()

        # The views for each observation
        self._obs_views = dict()
        self._obs_nview = dict()

        # Go through the data and find the intersection of the solver intervals and
        # the intervals we are using to estimate amplitudes.

        for ob in new_data.obs:
            if self.obs_key not in ob:
                # No templates in this observation
                continue

            # The total intervals we are using for projection
            if self.select_view is not None:
                self._obs_views[ob.uid] = (
                    ob.intervals[self.view] & ob.intervals[self.select_view]
                )
            else:
                self._obs_views[ob.uid] = ob.intervals[self.view]
            self._obs_nview[ob.uid] = len(self._obs_views[ob.uid])

            # Build up detector list
            det_pat = None
            if self.pattern is not None:
                det_pat = re.compile(self.pattern)
            self._obs_dets[ob.uid] = list()
            ddets = set(ob.detdata[self.det_data].detectors)
            for d in ob.select_local_detectors(flagmask=self.det_mask):
                if d not in ddets:
                    continue
                if det_pat is not None and det_pat.match(d) is None:
                    continue
                self._obs_dets[ob.uid].append(d)
                self._all_dets[d] = None

        # Go through the data and compute the offsets into the amplitudes for each
        # observation and detector.

        self._det_start = dict()
        offset = 0
        for ob in new_data.obs:
            if ob.uid not in self._obs_dets:
                continue
            det_list = self._obs_dets[ob.uid]
            self._det_start[ob.uid] = dict()
            for det in det_list:
                self._det_start[ob.uid][det] = offset
                offset += self._obs_nview[ob.uid]

        # Now we know the total number of amplitudes.

        self._n_local = offset
        if new_data.comm.comm_world is None:
            self._n_global = self._n_local
        else:
            self._n_global = new_data.comm.comm_world.allreduce(
                self._n_local, op=MPI.SUM
            )

        # Now that we know the number of amplitudes, we go through the solver flags
        # and determine what amplitudes, if any, are poorly constrained.

        # Boolean flags
        self._amp_flags = np.zeros(self._n_local, dtype=bool)

        # The relative weights for each amplitude, used in preconditioning
        self._amp_weights = np.zeros(self._n_local, dtype=np.float32)

        # The amplitude covariance.  For each amplitude representing a sample slice
        # of a time-domain template T, this is just:
        #
        # (T * transpose(T))^-1.
        #
        # This is a scalar.
        self._amp_cov = np.zeros(self._n_local, dtype=np.float64)

        for ob in new_data.obs:
            if ob.uid not in self._obs_dets:
                continue
            det_list = self._obs_dets[ob.uid]
            for det in det_list:
                template_key = ob[self.obs_key]["det_to_key"][det]
                det_templates = ob[self.obs_key][template_key]
                if isinstance(det_templates, dict):
                    if len(det_templates.keys()) > 1:
                        msg = f"Det {det} has templates {det_templates.keys()}. "
                        msg += " Only one template per det is supported"
                        raise NotImplementedError(msg)
                    template = list(det_templates.values())[0]
                else:
                    template = det_templates
                offset = self._det_start[ob.uid][det]
                if self.det_flags is not None:
                    flags = ob.detdata[self.det_flags][det] & self.det_flag_mask
                else:
                    flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
                for ivw, vw in enumerate(self._obs_views[ob.uid]):
                    vsamp = vw.last - vw.first
                    vslc = slice(vw.first, vw.last, 1)
                    vflags = flags[vslc]
                    vtemp = template[vslc]
                    amp_idx = offset + ivw

                    good = vflags == 0
                    ngood = np.count_nonzero(good)
                    good_frac = ngood / vsamp
                    if good_frac < self.good_fraction:
                        self._amp_flags[amp_idx] = 1
                        self._amp_cov[amp_idx] = 0.0
                    else:
                        self._amp_weights[amp_idx] = good_frac
                        dot_temp = np.dot(vtemp[good], vtemp[good])
                        if dot_temp == 0:
                            msg = f"Obs {ob.name}, det {det}, interval {ivw}"
                            msg += " has a template of all zeros"
                            raise RuntimeError(msg)
                        self._amp_cov[amp_idx] = 1.0 / dot_temp

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        if z.local_flags is not None:
            z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _add_to_signal(self, detector, amplitudes, **kwargs):
        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return

        for ob in self.data.obs:
            if ob.uid not in self._obs_dets:
                continue
            if detector not in self._det_start[ob.uid]:
                continue
            amp_offset = self._det_start[ob.uid][detector]
            amps = amplitudes.local[amp_offset : amp_offset + self._obs_nview[ob.uid]]

            template_key = ob[self.obs_key]["det_to_key"][detector]
            det_templates = ob[self.obs_key][template_key]
            if isinstance(det_templates, dict):
                template = list(det_templates.values())[0]
            else:
                template = det_templates

            # Accumulate to timestream
            for ivw, vw in enumerate(self._obs_views[ob.uid]):
                vslc = slice(vw.first, vw.last, 1)
                ob.detdata[self.det_data][detector][vslc] += (
                    amps[ivw] * template[vslc]
                )

    def _project_signal(self, detector, amplitudes, **kwargs):
        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return

        for ob in self.data.obs:
            if ob.uid not in self._obs_dets:
                continue
            if detector not in self._det_start[ob.uid]:
                continue
            amp_offset = self._det_start[ob.uid][detector]
            amps = amplitudes.local[amp_offset : amp_offset + self._obs_nview[ob.uid]]

            if self.det_flags is not None:
                flags = ob.detdata[self.det_flags][detector] & self.det_flag_mask
            else:
                flags = np.zeros(ob.n_local_samples, dtype=np.uint8)

            template_key = ob[self.obs_key]["det_to_key"][detector]
            det_templates = ob[self.obs_key][template_key]
            if isinstance(det_templates, dict):
                template = list(det_templates.values())[0]
            else:
                template = det_templates

            # Accumulate to amplitudes
            for ivw, vw in enumerate(self._obs_views[ob.uid]):
                vslc = slice(vw.first, vw.last, 1)
                amp_idx = amp_offset + ivw
                vflags = flags[vslc]
                good = vflags == 0
                vtemp = template[vslc]
                vdata = ob.detdata[self.det_data][detector][vslc]
                val = self._amp_cov[amp_idx] * np.dot(vtemp[good], vdata[good])
                amps[ivw] += val

    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        # No prior for this template, nothing to accumulate to output.
        return

    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        # Apply weights based on the number of samples hitting each
        # amplitude bin.
        for ob in self.data.obs:
            if ob.uid not in self._obs_dets:
                continue
            for det, amp_offset in self._det_start[ob.uid].items():
                for ivw, vw in enumerate(self._obs_views[ob.uid]):
                    amp_idx = amp_offset + ivw
                    amp_in = amplitudes_in.local[amp_idx]
                    amplitudes_out.local[amp_idx] = amp_in * self._amp_weights[amp_idx]


#     @function_timer
#     def write(self, amplitudes, out):
#         """Write out amplitude values.

#         This stores the amplitudes to a file for debugging / plotting.

#         Args:
#             amplitudes (Amplitudes):  The amplitude data.
#             out (str):  The output file.

#         Returns:
#             None

#         """
#         # By definition, when solving for something that is periodic for
#         # a given detector in a single observation, we (should) have many
#         # fewer template amplitudes than timestream samples.  Because of
#         # this we assume we can make several copies for extracting the
#         # amplitudes and gathering them for writing.

#         obs_det_amps = dict()

#         for det in self._all_dets:
#             amp_offset = self._det_offset[det]
#             for iob, ob in enumerate(self.data.obs):
#                 if det not in self._obs_dets[iob]:
#                     continue
#                 if self.is_detdata_key:
#                     if self.key not in ob.detdata:
#                         continue
#                 else:
#                     if self.key not in ob.shared:
#                         continue
#                 if ob.name not in obs_det_amps:
#                     obs_det_amps[ob.name] = dict()
#                 nbins = self._obs_nbins[iob]
#                 amp_data = amplitudes.local[amp_offset : amp_offset + nbins]
#                 amp_hits = self._amp_hits[amp_offset : amp_offset + nbins]
#                 amp_flags = self._amp_flags[amp_offset : amp_offset + nbins]
#                 obs_det_amps[ob.name][det] = {
#                     "amps": amp_data,
#                     "hits": amp_hits,
#                     "flags": amp_flags,
#                     "min": self._obs_min[iob],
#                     "max": self._obs_max[iob],
#                     "incr": self._obs_incr[iob],
#                 }
#                 amp_offset += nbins

#         if self.data.comm.world_size == 1:
#             all_obs_dets_amps = [obs_det_amps]
#         else:
#             all_obs_dets_amps = self.data.comm.comm_world.gather(obs_det_amps, root=0)

#         if self.data.comm.world_rank == 0:
#             obs_det_amps = dict()
#             for pdata in all_obs_dets_amps:
#                 for obname in pdata.keys():
#                     if obname not in obs_det_amps:
#                         obs_det_amps[obname] = dict()
#                     obs_det_amps[obname].update(pdata[obname])
#             del all_obs_dets_amps
#             with h5py.File(out, "w") as hf:
#                 for obname, obamps in obs_det_amps.items():
#                     n_det = len(obamps)
#                     det_list = list(sorted(obamps.keys()))
#                     det_indx = {y: x for x, y in enumerate(det_list)}
#                     indx_to_det = {det_indx[x]: x for x in det_list}
#                     n_amp = len(obamps[det_list[0]]["amps"])
#                     amp_min = [obamps[x]["min"] for x in det_list]
#                     amp_max = [obamps[x]["max"] for x in det_list]
#                     amp_incr = [obamps[x]["incr"] for x in det_list]

#                     # Create datasets for this observation
#                     hg = hf.create_group(obname)
#                     hg.attrs["detectors"] = json.dumps(det_list)
#                     hg.attrs["min"] = json.dumps(amp_min)
#                     hg.attrs["max"] = json.dumps(amp_max)
#                     hg.attrs["incr"] = json.dumps(amp_incr)
#                     hamps = hg.create_dataset(
#                         "amplitudes",
#                         (n_det, n_amp),
#                         dtype=np.float64,
#                     )
#                     hhits = hg.create_dataset(
#                         "hits",
#                         (n_det, n_amp),
#                         dtype=np.int32,
#                     )
#                     hflags = hg.create_dataset(
#                         "flags",
#                         (n_det, n_amp),
#                         dtype=np.uint8,
#                     )

#                     # Write data
#                     for idet in range(n_det):
#                         det = indx_to_det[idet]
#                         dprops = obamps[det]
#                         hslice = (slice(idet, idet + 1, 1), slice(0, n_amp, 1))
#                         dslice = (slice(0, n_amp, 1),)
#                         hamps.write_direct(dprops["amps"], dslice, hslice)
#                         hhits.write_direct(dprops["hits"], dslice, hslice)
#                         hflags.write_direct(dprops["flags"], dslice, hslice)


# def plot(amp_file, out_root=None):
#     """Plot an amplitude dump file.

#     This loads an amplitude file and makes a set of plots.

#     Args:
#         amp_file (str):  The path to the input file of amplitudes.
#         out_root (str):  The root of the output files.

#     Returns:
#         None

#     """

#     if out_root is not None:
#         set_matplotlib_backend(backend="pdf")

#     import matplotlib.pyplot as plt

#     figdpi = 100

#     with h5py.File(amp_file, "r") as hf:
#         for obname, obgrp in hf.items():
#             det_list = json.loads(obgrp.attrs["detectors"])
#             amp_min = list(ast.literal_eval(obgrp.attrs["min"]))
#             amp_max = list(ast.literal_eval(obgrp.attrs["max"]))
#             amp_incr = list(ast.literal_eval(obgrp.attrs["incr"]))

#             hamps = np.array(obgrp["amplitudes"])
#             hhits = np.array(obgrp["hits"])
#             hflags = np.array(obgrp["flags"])
#             n_bin = hamps.shape[1]
#             bad = hflags != 0
#             hamps[bad] = np.nan

#             for idet, det in enumerate(det_list):
#                 outfile = f"{out_root}_{obname}_{det}.pdf"
#                 xdata = amp_min[idet] + amp_incr[idet] * np.arange(
#                     n_bin, dtype=np.float64
#                 )
#                 fig = plt.figure(dpi=figdpi, figsize=(8, 12))
#                 ax = fig.add_subplot(3, 1, 1)
#                 ax.step(xdata, hamps[idet], where="post", label=f"{det}")
#                 ax.set_xlabel("Shared Data Value")
#                 ax.set_ylabel("Amplitude")
#                 ax.legend(loc="best")
#                 ax = fig.add_subplot(3, 1, 2)
#                 ax.step(xdata, hflags[idet], where="post", label=f"{det}")
#                 ax.set_xlabel("Shared Data Value")
#                 ax.set_ylabel("Flags")
#                 ax.legend(loc="best")
#                 ax = fig.add_subplot(3, 1, 3)
#                 ax.step(xdata, hhits[idet], where="post", label=f"{det}")
#                 ax.set_xlabel("Shared Data Value")
#                 ax.set_ylabel("Hits")
#                 ax.legend(loc="best")
#                 if out_root is None:
#                     # Interactive
#                     plt.show()
#                 else:
#                     plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
#                     plt.close()
