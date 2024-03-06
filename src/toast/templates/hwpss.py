# Copyright (c) 2023-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import ast
import json
from collections import OrderedDict

import h5py
import numpy as np
from scipy.linalg import lu_factor, lu_solve, eigvalsh

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
class Hwpss(Template):
    """This template represents the HWP synchronous signal."""

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

    hwp_angle = Unicode(
        defaults.hwp_angle, allow_none=True, help="Observation shared key for HWP angle"
    )

    hwp_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for HWP flags",
    )

    hwp_flag_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask to use when considering valid HWP angle values.",
    )

    harmonics = Int(9, help="Number of harmonics to consider in the expansion")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    debug_plots = Unicode(
        None,
        allow_none=True,
        help="If not None, make debugging plots in this directory",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        log = Logger.get()

        # For each harmonic with the sin, cos, and time drift terms for each.
        self._n_coeff = 4 * self.harmonics

        # Use this as an "Ordered Set".  We want the unique detectors on this process,
        # but sorted in order of occurrence.
        all_dets = OrderedDict()

        # Good detectors to use for each observation
        self._obs_dets = dict()

        # Build up detector list
        for iob, ob in enumerate(new_data.obs):
            if self.hwp_angle not in ob.shared:
                continue
            self._obs_dets[iob] = set()
            for d in ob.select_local_detectors(flagmask=self.det_mask):
                if d not in ob.detdata[self.det_data].detectors:
                    continue
                self._obs_dets[iob].add(d)
                if d not in all_dets:
                    all_dets[d] = None
        self._all_dets = list(all_dets.keys())

        # During application of the template, we will be looping over detectors
        # in the outer loop.  So we pack the amplitudes by detector and then by
        # observation.  For each observation, we precompute the quantities that
        # are common to all detectors.

        self._det_offset = dict()
        self._obs_reltime = dict()
        self._obs_sincos = dict()
        self._obs_cov = dict()
        self._obs_outview = dict()

        offset = 0
        for det in self._all_dets:
            self._det_offset[det] = offset
            for iob, ob in enumerate(new_data.obs):
                if self.hwp_angle not in ob.shared:
                    continue
                if det not in self._obs_dets[iob]:
                    continue
                if self.view is not None:
                    self._obs_outview[iob] = ~ob.intervals[self.view]
                times = np.array(ob.shared[self.times].data, copy=True)
                time_offset = times[0]
                times -= time_offset
                self._obs_reltime[iob] = times.astype(np.float32)
                if self.hwp_flags is None:
                    flags = np.zeros(len(times), dtype=np.uint8)
                else:
                    flags = ob.shared[self.hwp_flags].data & self.hwp_flag_mask
                self._obs_sincos[iob] = self.sincos_buffer(
                    ob.shared[self.hwp_angle], flags, self.harmonics
                )
                self._obs_cov[iob] = self.compute_coeff_covariance(
                    self._obs_reltime[iob], flags, self._obs_sincos[iob]
                )
                offset += self._n_coeff

        # Now we know the total number of local amplitudes.
        if offset == 0:
            # This means that no observations included a HWP angle
            msg = f"Data has no observations with HWP angle '{self.hwp_angle}'."
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

        # Boolean flags
        self._amp_flags = np.zeros(self._n_local, dtype=bool)

        for det in self._all_dets:
            amp_offset = self._det_offset[det]
            for iob, ob in enumerate(new_data.obs):
                if self.hwp_angle not in ob.shared:
                    continue
                if det not in self._obs_dets[iob]:
                    continue
                if self._obs_cov[iob] is None:
                    # This observation has poorly conditioned covariance
                    self._amp_flags[amp_offset : amp_offset + self._n_coeff] = True
                amp_offset += self._n_coeff

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(self.data.comm, self._n_global, self._n_local)
        z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    @classmethod
    def sincos_buffer(cls, angles, flags, n_harmonics):
        ang = np.copy(angles)
        ang[flags != 0] = 0.0
        n_samp = len(ang)
        sample_vals = 2 * n_harmonics
        sincos = np.zeros((n_samp, sample_vals), dtype=np.float32)
        for h in range(n_harmonics):
            sincos[:, 2 * h] = np.sin((h + 1) * ang)
            sincos[:, 2 * h + 1] = np.cos((h + 1) * ang)
        return sincos

    @classmethod
    def compute_coeff_covariance(cls, times, flags, sincos):
        n_samp = len(times)
        n_harmonics = sincos.shape[1] // 2
        cov = np.zeros((4 * n_harmonics, 4 * n_harmonics), dtype=np.float64)
        good = flags == 0
        # Compute upper triangle
        for hr in range(0, n_harmonics):
            for hc in range(hr, n_harmonics):
                cov[4 * hr + 0, 4 * hc + 0] = np.dot(
                    sincos[good, 2 * hr + 0], sincos[good, 2 * hc + 0]
                )
                cov[4 * hr + 0, 4 * hc + 1] = np.dot(
                    sincos[good, 2 * hr + 0],
                    np.multiply(times[good], sincos[good, 2 * hc + 0]),
                )
                cov[4 * hr + 0, 4 * hc + 2] = np.dot(
                    sincos[good, 2 * hr + 0], sincos[good, 2 * hc + 1]
                )
                cov[4 * hr + 0, 4 * hc + 3] = np.dot(
                    sincos[good, 2 * hr + 0],
                    np.multiply(times[good], sincos[good, 2 * hc + 1]),
                )

                cov[4 * hr + 1, 4 * hc + 0] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 0]),
                    sincos[good, 2 * hc + 0],
                )
                cov[4 * hr + 1, 4 * hc + 1] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 0]),
                    np.multiply(times[good], sincos[good, 2 * hc + 0]),
                )
                cov[4 * hr + 1, 4 * hc + 2] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 0]),
                    sincos[good, 2 * hc + 1],
                )
                cov[4 * hr + 1, 4 * hc + 3] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 0]),
                    np.multiply(times[good], sincos[good, 2 * hc + 1]),
                )

                cov[4 * hr + 2, 4 * hc + 0] = np.dot(
                    sincos[good, 2 * hr + 1], sincos[good, 2 * hc + 0]
                )
                cov[4 * hr + 2, 4 * hc + 1] = np.dot(
                    sincos[good, 2 * hr + 1],
                    np.multiply(times[good], sincos[good, 2 * hc + 0]),
                )
                cov[4 * hr + 2, 4 * hc + 2] = np.dot(
                    sincos[good, 2 * hr + 1], sincos[good, 2 * hc + 1]
                )
                cov[4 * hr + 2, 4 * hc + 3] = np.dot(
                    sincos[good, 2 * hr + 1],
                    np.multiply(times[good], sincos[good, 2 * hc + 1]),
                )

                cov[4 * hr + 3, 4 * hc + 0] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 1]),
                    sincos[good, 2 * hc + 0],
                )
                cov[4 * hr + 3, 4 * hc + 1] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 1]),
                    np.multiply(times[good], sincos[good, 2 * hc + 0]),
                )
                cov[4 * hr + 3, 4 * hc + 2] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 1]),
                    sincos[good, 2 * hc + 1],
                )
                cov[4 * hr + 3, 4 * hc + 3] = np.dot(
                    np.multiply(times[good], sincos[good, 2 * hr + 1]),
                    np.multiply(times[good], sincos[good, 2 * hc + 1]),
                )

        # Fill in lower triangle
        for hr in range(0, 4 * n_harmonics):
            for hc in range(0, hr):
                cov[hr, hc] = cov[hc, hr]
        # Check that condition number is reasonable
        evals = eigvalsh(cov)
        rcond = np.min(evals) / np.max(evals)
        if rcond < 1.0e-8:
            return None
        # LU factorization for later solve
        cov_lu, cov_piv = lu_factor(cov)
        return cov_lu, cov_piv

    @classmethod
    def compute_coeff(cls, detdata, flags, times, sincos, cov_lu, cov_piv):
        n_samp = len(times)
        n_harmonics = sincos.shape[1] // 2
        good = flags == 0
        input = np.copy(detdata)
        dc = np.mean(input[good])
        input[good] -= dc
        rhs = np.zeros(4 * n_harmonics, dtype=np.float64)
        for h in range(n_harmonics):
            rhs[4 * h + 0] = np.dot(input[good], sincos[good, 2 * h])
            rhs[4 * h + 1] = np.dot(
                input[good], np.multiply(sincos[good, 2 * h], times[good])
            )
            rhs[4 * h + 2] = np.dot(input[good], sincos[good, 2 * h + 1])
            rhs[4 * h + 3] = np.dot(
                input[good], np.multiply(sincos[good, 2 * h + 1], times[good])
            )
        coeff = lu_solve((cov_lu, cov_piv), rhs)
        return coeff

    @classmethod
    def build_model(cls, times, flags, sincos, coeff):
        n_samp = len(times)
        n_harmonics = sincos.shape[1] // 2
        good = flags == 0
        model = np.zeros(n_samp, dtype=np.float64)
        for h in range(n_harmonics):
            model[good] += coeff[4 * h + 0] * sincos[good, 2 * h]
            model[good] += coeff[4 * h + 1] * np.multiply(
                sincos[good, 2 * h], times[good]
            )
            model[good] += coeff[4 * h + 2] * sincos[good, 2 * h + 1]
            model[good] += coeff[4 * h + 3] * np.multiply(
                sincos[good, 2 * h + 1], times[good]
            )
        return model

    def _add_to_signal(self, detector, amplitudes, **kwargs):
        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return
        amp_offset = self._det_offset[detector]
        for iob, ob in enumerate(self.data.obs):
            if self.hwp_angle not in ob.shared:
                continue
            if detector not in self._obs_dets[iob]:
                continue
            if self.hwp_flags is None:
                flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
            else:
                flags = ob.shared[self.hwp_flags].data & self.hwp_flag_mask
            if self.view is not None:
                # Flag samples outside the valid intervals
                for vw in self._obs_outview[iob]:
                    vw_slc = slice(vw.first, vw.last + 1, 1)
                    flags[vw_slc] = 1
            coeff = amplitudes.local[amp_offset : amp_offset + self._n_coeff]
            model = self.build_model(
                self._obs_reltime[iob],
                flags,
                self._obs_sincos[iob],
                coeff,
            )
            good = flags == 0
            # Accumulate to timestream
            ob.detdata[self.det_data][detector][good] += model[good]
            amp_offset += self._n_coeff

    def _project_signal(self, detector, amplitudes, **kwargs):
        if detector not in self._all_dets:
            # This must have been cut by per-detector flags during initialization
            return
        amp_offset = self._det_offset[detector]
        for iob, ob in enumerate(self.data.obs):
            if self.hwp_angle not in ob.shared:
                continue
            if detector not in self._obs_dets[iob]:
                continue
            if self.hwp_flags is None:
                flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
            else:
                flags = ob.shared[self.hwp_flags].data & self.hwp_flag_mask
            if self.view is not None:
                # Flag samples outside the valid intervals
                for vw in self._obs_outview[iob]:
                    vw_slc = slice(vw.first, vw.last + 1, 1)
                    flags[vw_slc] = 1
            if self.det_flags is not None:
                flags |= ob.detdata[self.det_flags][detector] & self.det_flag_mask
            if self._obs_cov[iob] is None:
                # Flagged
                amplitudes.local[amp_offset : amp_offset + self._n_coeff] = 0
            else:
                coeff = self.compute_coeff(
                    ob.detdata[self.det_data][detector],
                    flags,
                    self._obs_reltime[iob],
                    self._obs_sincos[iob],
                    self._obs_cov[iob][0],
                    self._obs_cov[iob][1],
                )
                amplitudes.local[amp_offset : amp_offset + self._n_coeff] = coeff
            amp_offset += self._n_coeff

    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        # No prior for this template, nothing to accumulate to output.
        return

    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        # Just the identity matrix
        amplitudes_out.local[:] = amplitudes_in.local
        return

    @function_timer
    def write(self, amplitudes, out):
        """Write out amplitude values.

        This stores the amplitudes to a file for debugging / plotting.
        WARNING: currently this only works for data distributed by
        detector.

        Args:
            amplitudes (Amplitudes):  The amplitude data.
            out (str):  The output file.

        Returns:
            None

        """
        obs_det_amps = dict()
        obs_reltime = dict()
        obs_hwpang = dict()

        for det in self._all_dets:
            amp_offset = self._det_offset[det]
            for iob, ob in enumerate(self.data.obs):
                if self.hwp_angle not in ob.shared:
                    continue
                if det not in self._obs_dets[iob]:
                    continue
                if ob.name not in obs_det_amps:
                    obs_det_amps[ob.name] = dict()
                if ob.comm.group_rank == 0:
                    if ob.name not in obs_reltime:
                        obs_reltime[ob.name] = np.array(self._obs_reltime[iob])
                        if self.hwp_flags is not None:
                            flags = ob.shared[self.hwp_flags].data & self.hwp_flag_mask
                            # Set flagged samples to a negative time value, to
                            # communicate the flags to downstream code.
                            obs_reltime[ob.name][flags != 0] = -1.0
                        obs_hwpang[ob.name] = np.array(
                            ob.shared[self.hwp_angle].data, dtype=np.float32
                        )
                obs_det_amps[ob.name][det] = amplitudes.local[
                    amp_offset : amp_offset + self._n_coeff
                ]
                amp_offset += self._n_coeff

        if self.data.comm.world_size == 1:
            all_obs_dets_amps = [obs_det_amps]
            all_obs_reltime = [obs_reltime]
            all_obs_hwpang = [obs_hwpang]
        else:
            all_obs_dets_amps = self.data.comm.comm_world.gather(obs_det_amps, root=0)
            all_obs_reltime = self.data.comm.comm_world.gather(obs_reltime, root=0)
            all_obs_hwpang = self.data.comm.comm_world.gather(obs_hwpang, root=0)

        if self.data.comm.world_rank == 0:
            obs_det_amps = dict()
            for pdata in all_obs_dets_amps:
                for obname in pdata.keys():
                    if obname not in obs_det_amps:
                        obs_det_amps[obname] = dict()
                    obs_det_amps[obname].update(pdata[obname])
            del all_obs_dets_amps

            obs_reltime = dict()
            for pdata in all_obs_reltime:
                if len(pdata) == 0:
                    continue
                for obname in pdata.keys():
                    if obname not in obs_reltime:
                        obs_reltime[obname] = pdata[obname]
            del all_obs_reltime

            obs_hwpang = dict()
            for pdata in all_obs_hwpang:
                if len(pdata) == 0:
                    continue
                for obname in pdata.keys():
                    if obname not in obs_hwpang:
                        obs_hwpang[obname] = pdata[obname]
            del all_obs_hwpang

            with h5py.File(out, "w") as hf:
                for obname, obamps in obs_det_amps.items():
                    n_det = len(obamps)
                    det_list = list(sorted(obamps.keys()))
                    det_indx = {y: x for x, y in enumerate(det_list)}
                    indx_to_det = {det_indx[x]: x for x in det_list}
                    n_amp = len(obamps[det_list[0]])
                    n_samp = len(obs_reltime[obname])

                    # Create datasets for this observation
                    hg = hf.create_group(obname)
                    hg.attrs["detectors"] = json.dumps(det_list)
                    hamps = hg.create_dataset(
                        "amplitudes",
                        (n_det, n_amp),
                        dtype=np.float64,
                    )
                    htime = hg.create_dataset(
                        "reltime",
                        (n_samp,),
                        dtype=np.float32,
                    )
                    hang = hg.create_dataset(
                        "hwpangle",
                        (n_samp,),
                        dtype=np.float32,
                    )

                    # Write data
                    samp_slice = (slice(0, n_samp, 1),)
                    htime.write_direct(obs_reltime[obname], samp_slice, samp_slice)
                    hang.write_direct(obs_hwpang[obname], samp_slice, samp_slice)
                    for idet in range(n_det):
                        det = indx_to_det[idet]
                        hslice = (slice(idet, idet + 1, 1), slice(0, n_amp, 1))
                        dslice = (slice(0, n_amp, 1),)
                        hamps.write_direct(obamps[det], dslice, hslice)


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

            hamps = obgrp["amplitudes"]
            htime = np.array(obgrp["reltime"])
            hang = np.array(obgrp["hwpangle"])
            n_coeff = hamps.shape[1]
            n_harmonic = n_coeff // 4
            flags = np.zeros(len(htime), dtype=np.uint8)
            bad = htime < 0
            good = np.logical_not(bad)
            flags[bad] = 1
            sincos = Hwpss.sincos_buffer(hang, flags, n_harmonic)

            for idet, det in enumerate(det_list):
                outfile = f"{out_root}_{obname}_{det}.pdf"
                coeff = hamps[idet]
                model = Hwpss.build_model(htime, flags, sincos, coeff)

                fig = plt.figure(dpi=figdpi, figsize=(8, 12))
                ax = fig.add_subplot(2, 1, 1)
                ax.plot(htime[good], model[good], label=f"{det} Model")
                ax.set_xlabel("Time Relative to Start of Observation")
                ax.set_ylabel("Amplitude")
                ax.legend(loc="best")
                ax = fig.add_subplot(2, 1, 2)
                ax.plot(htime[good][:500], model[good][:500], label=f"{det} Model")
                ax.set_xlabel("Time Relative to Start of Observation")
                ax.set_ylabel("Amplitude")
                ax.legend(loc="best")
                if out_root is None:
                    # Interactive
                    plt.show()
                else:
                    plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
                    plt.close()
