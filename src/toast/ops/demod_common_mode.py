# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from time import time

import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Int, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class DemodCommonModeFilter(Operator):
    """Operator that extracts and projects out the Qr/Ur common modes

    The provided data must be demodulated in the radial (Qr/Ur) or
    horizontal (Az/El) frame.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key",
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for optional shared flagging",
    )

    boresight = Unicode(
        defaults.boresight_azel,
        allow_none=True,
        help="Observation shared data key for boresight quaternions for deriving "
        "the boresight roll angle",
    )

    # For now, only support full observation common modes
    # view = Unicode(
    #     None, allow_none=True, help="Use this view of the data in all observations"
    # )

    pol_frame = Unicode(
        "horizontal",
        allow_none=False,
        help="Input Q/U basis. Either 'radial' or 'horizontal'",
    )

    mode = Unicode(
        "IQU", allow_none=False, help="Stokes modes to filter (I, QU or IQU)"
    )

    nmode = Int(3, allow_none=False, help="Number of common modes to extract with PCA")

    rms_cut_low = Float(
        0.05,
        allow_none=False,
        help="Fraction of detectors to cut from the lower end of sorted RMS "
        "distribution before measuring the common modes through PCA.",
    )

    rms_cut_high = Float(
        0.05,
        allow_none=False,
        help="Fraction of detectors to cut from the upper end of sorted RMS "
        "distribution before measuring the common modes through PCA.  PCA will "
        "target detectors with the highest RMS so outliers can degrade the "
        "operator performance.",
    )

    @traitlets.validate("pol_frame")
    def _check_pol_frame(self, proposal):
        check = proposal["value"]
        allowed = ["radial", "horizontal"]
        if check not in allowed:
            raise traitlets.TraitError(f"pol_frame should be one of {allowed}")
        return check

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        check = proposal["value"]
        allowed = ["I", "QU", "IQU"]
        if check not in allowed:
            raise traitlets.TraitError(f"mode should be one of {allowed}")
        return check

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        t0 = time()
        env = Environment.get()
        log = Logger.get()

        for ob in data.obs:
            if ob.dist.comm_row_size != 1:
                raise RuntimeError(
                    "DemodCommonModeFilter only works with observations "
                    "distributed by detector"
                )
            if self.boresight is None:
                roll = 0
            else:
                roll = qa.to_iso_angles(ob.shared[self.boresight])[2]
            good = (ob.shared[self.shared_flags].data & self.shared_flag_mask) == 0
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            tod = self._collect_tod(ob, dets, good, roll)
            templates = self._get_templates(ob, tod)
            self._project_templates(ob, dets, templates, good, roll)

        return

    @function_timer
    def _collect_tod(self, ob, detectors, good, roll):
        """Rotate the TOD to Qr/Ur and send to root process"""
        fp = ob.telescope.focalplane

        Itod = []
        Qtod = []
        Utod = []
        for det in detectors:
            if det.startswith("demod0"):
                Itod.append(ob.detdata[self.det_data][det][good])
                continue
            if det.startswith("demod4r"):
                Qdet = det
            else:
                continue
            Udet = Qdet.replace("demod4r", "demod4i")
            Q = ob.detdata[self.det_data][Qdet][good]
            U = ob.detdata[self.det_data][Udet][good]
            if self.pol_frame == "radial":
                Qr = Q
                Ur = U
            elif self.pol_frame == "horizontal":
                # Rotate from horizontal to radial polarization basis
                phi = qa.to_iso_angles(fp[Qdet]["quat"])[1]
                phi = (phi + roll)[good]
                Qr = Q * np.cos(2 * phi) + U * np.sin(2 * phi)
                Ur = U * np.cos(2 * phi) - Q * np.sin(2 * phi)
            else:
                msg = f"Unknown polarization frame: {self.pol_frame}"
                raise RuntimeError(msg)
            Qtod.append(Qr - np.mean(Qr))
            Utod.append(Ur - np.mean(Ur))

        comm = ob.comm.comm_group
        if comm is None:
            Itod = [Itod]
            Qtod = [Qtod]
            Utod = [Utod]
        else:
            Itod = comm.gather(Itod)
            Qtod = comm.gather(Qtod)
            Utod = comm.gather(Utod)

        if comm is None or comm.rank == 0:
            tod = {}
            for key, tods in ("I", Itod), ("Q", Qtod), ("U", Utod):
                while [] in tods:
                    tods.remove([])
                if len(tods) != 0:
                    tod[key] = np.vstack(tods)
                elif key in self.mode:
                    msg = f"Could not find any demodulated {key} streams in {ob.name}"
                    raise RuntimeError(msg)
        else:
            tod = None

        return tod

    @function_timer
    def _get_templates(self, ob, tod):
        """Use PCA to extract common modes as templates"""

        comm = ob.comm.comm_group
        if comm is None or comm.rank == 0:
            # All TOD has been gathered to the root process for PCA
            templates = {}
            for stokes in self.mode:
                tods = tod[stokes]
                if self.rms_cut_low > 0 or self.rms_cut_high > 0:
                    rms = np.std(tods, 1)
                    sorted_rms = np.sort(rms)
                    ndet = len(rms)
                    lower_limit = sorted_rms[int(ndet * self.rms_cut_low)]
                    upper_limit = sorted_rms[int(ndet * (1 - self.rms_cut_high))]
                    good = np.logical_and(lower_limit <= rms, rms <= upper_limit)
                else:
                    good = slice(len(tods))
                _, S, modes = np.linalg.svd(tods[good], full_matrices=False)
                offset = np.ones(tods[0].size)
                modes = np.vstack([offset, modes[: self.nmode]])
                invcov = np.dot(modes, modes.T)
                cov = np.linalg.inv(invcov)
                templates[stokes] = (modes, cov)
        else:
            templates = None

        if comm is not None:
            templates = comm.bcast(templates)

        return templates

    @function_timer
    def _project_templates(self, ob, detectors, templates, good, roll):
        """Use linear regression to clean the TOD"""

        fp = ob.telescope.focalplane
        for det in detectors:
            if det.startswith("demod0") and "I" in self.mode:
                modes, cov = templates["I"]
                I = ob.detdata[self.det_data][det][good].copy()
                self._regress(modes, cov, I)
                # Save the cleaned data
                ob.detdata[self.det_data][det][good] = I
            elif det.startswith("demod4r") and "QU" in self.mode:
                Qdet = det
                Udet = Qdet.replace("demod4r", "demod4i")
                Q = ob.detdata[self.det_data][Qdet][good].copy()
                U = ob.detdata[self.det_data][Udet][good].copy()
                if self.pol_frame == "radial":
                    Qr = Q.copy()
                    Ur = U.copy()
                else:
                    # Rotate from horizontal to radial polarization basis
                    theta, phi, psi = qa.to_iso_angles(fp[Qdet]["quat"])
                    phi = (phi + roll)[good]
                    Qr = Q * np.cos(2 * phi) + U * np.sin(2 * phi)
                    Ur = U * np.cos(2 * phi) - Q * np.sin(2 * phi)
                # Clean Qr and Ur
                Qmodes, Qcov = templates["Q"]
                Umodes, Ucov = templates["U"]
                self._regress(Qmodes, Qcov, Qr)
                self._regress(Umodes, Ucov, Ur)
                if self.pol_frame == "radial":
                    Q = Qr
                    U = Ur
                else:
                    # Rotate back to horizontal basis
                    Q = Qr * np.cos(2 * phi) - Ur * np.sin(2 * phi)
                    U = Ur * np.cos(2 * phi) + Qr * np.sin(2 * phi)
                # Save the cleaned data
                ob.detdata[self.det_data][Qdet][good] = Q
                ob.detdata[self.det_data][Udet][good] = U

        return

    @function_timer
    def _regress(self, templates, cov, signal):
        """Perform simple linear regression"""

        proj = np.dot(templates, signal)
        coeff = np.dot(cov, proj)
        signal -= np.dot(coeff, templates)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": list(),
            "detdata": [self.det_data],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.boresight is not None:
            req["shared"].append(self.boresight)
        return req

    def _provides(self):
        return dict()
