# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

try:
    from ducc0 import totalconvolve

    ducc_available = True
except (ModuleNotFoundError, ImportError) as e:
    ducc_available = False
import healpy as hp
from pshmem import MPIShared

from .. import qarray as qa
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..timing import Timer, function_timer
from ..traits import (Bool, Instance, Int, List, Quantity, Unicode, Unit,
                      trait_docs)
from ..utils import Logger
from .operator import Operator
from .pipeline import Pipeline
from .pointing import BuildPixelDistribution
from .scan_map import ScanMap, ScanMask


@trait_docs
class ScanAlm(Operator):
    """Operator which reads an a_lm expansion of a sky from disk and
    scans it to a timestream.

    The a_lm file is loaded into node-shared memory.  For each
    observation, the pointing model is used to expand the pointing and
    interpolate values into detector data.

    Since the Stokes weights can carry information beyond the
    polarization angle, this operator scans the sky into separate I/Q/U
    timestreams that are co-added with the appropriate pointing weights.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(
        None,
        allow_none=True,
        help="Path to a_lm FITS file.  Use ';' if providing multiple files.  "
        "If set, `alms` must be empty.",
    )

    alms = List(
        None,
        help="a_lms to scan.  If set, `file` must be None.",
    )

    fwhm = Quantity(
        0 * u.deg,
        help="Additional smoothing to apply to loaded sky",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating output.  Use ';' if different "
        "files are applied to different flavors",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    subtract = Bool(
        False, help="If True, subtract the timestream instead of accumulating"
    )

    zero = Bool(False, help="If True, zero the data before accumulating / subtracting")

    fp_gamma = Unicode(
        "gamma", allow_none=True, help="Focalplane key for detector gamma offset angle"
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a Stokes weights operator",
    )

    save_alm = Bool(False, help="If True, do not delete alm during finalize")

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("stokes_weights")
    def _check_stokes_weights(self, proposal):
        weights = proposal["value"]
        if weights is not None:
            if not isinstance(weights, Operator):
                raise traitlets.TraitError(
                    "stokes_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["weights", "view"]:
                if not weights.has_trait(trt):
                    msg = f"stokes_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return weights

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ("detector_pointing", "stokes_weights"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        if self.file is None:
            if len(self.alms) == 0:
                msg = "You must set either the `file` or `alms` trait"
                raise RuntimeError(msg)
            # Maps are pre-loaded
            nalm = len(self.alms)
            self.file_names = []
        else:
            if len(self.alms) != 0:
                msg = "You cannot set both `file` and `alms` traits"
                raise RuntimeError(msg)
            # Split up the file and map names
            self.file_names = self.file.split(";")
            nalm = len(self.file_names)
            self.alms = []

        self.det_data_keys = self.det_data.split(";")
        nkey = len(self.det_data_keys)
        if nkey != 1 and (nalm != nkey):
            msg = "If multiple detdata keys are provided, each must have its own alm"
            raise RuntimeError(msg)

        filenames = self.file.split(";")
        detdata_keys = self.det_data.split(";")

        # Create our map(s) to scan named after our own operator name.  Generally the
        # files on disk are stored as float32, but even if not there is no real benefit
        # to having higher precision to simulated map signal that is projected into
        # timestreams.

        world_comm = data.comm.comm_world
        if world_comm is None:
            world_rank = 0
        else:
            world_rank = world_comm.rank

        self.lmax = None
        for file_name in self.file_names:
            dtype = complex  # totalconvolve requires dtype=complex
            if world_rank == 0:
                alm = hp.read_alm(file_name, hdu=(1, 2, 3)).astype(dtype)
                alm_shape = alm.shape
                lmax = hp.Alm.getlmax(alm[0].size)
            else:
                alm = None
                alm_shape = None
                lmax = None
            if world_comm is not None:
                alm_shape = world_comm.bcast(alm_shape)
                lmax = world_comm.bcast(lmax)
            if self.lmax is None:
                self.lmax = lmax
            elif lmax != self.lmax:
                msg = f"lmax({file_name}) = {lmax} but "
                msg += f"lmax({self.file_names[0]}) = {self.lmax}"
                raise RuntimeError(msg)
            shared = MPIShared(alm_shape, dtype, world_comm)
            shared.set(alm)
            self.alms.append(shared)

        # Loop over all observations and local detectors, sampling each alm

        for ob in data.obs:
            # Get the detectors we are using for this observation
            focalplane = ob.telescope.focalplane
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            for key in self.det_data_keys:
                # If our output detector data does not yet exist, create it
                exists_data = ob.detdata.ensure(
                    key, detectors=dets, create_units=self.det_data_units
                )
                if self.zero:
                    ob.detdata[key][:] = 0

            if self.stokes_weights.hwp_angle is None:
                hwp_angle = None
            else:
                hwp_angle = ob.shared[self.stokes_weights.hwp_angle].data

            ob_data = data.select(obs_name=ob.name)
            current_ob = ob_data.obs[0]
            for idet, det in enumerate(dets):

                # Convert pointing quaternion into angles

                self.detector_pointing.apply(ob_data, detectors=[det])
                det_quat = current_ob.detdata[self.detector_pointing.quats][det]
                theta, phi, _ = qa.to_iso_angles(det_quat)

                # Get pointing weights

                self.stokes_weights.apply(ob_data, detectors=[det])
                weights = np.atleast_2d(
                    current_ob.detdata[self.stokes_weights.weights][det]
                ).T

                # Interpolate the provided maps and accumulate the
                # appropriate timestreams in the original observation
                for ialm, alm in enumerate(self.alms):
                    if len(self.det_data_keys) == 1:
                        det_data_key = self.det_data_keys[0]
                    else:
                        det_data_key = self.det_data_keys[ialm]
                    ref = ob.detdata[det_data_key][det]

                    separate = False  # Co-add T/E/B
                    epsilon = 1e-5
                    for stokes, stokes_weights in zip(
                        self.stokes_weights.mode, weights
                    ):
                        if np.all(stokes_weights == 0):
                            continue
                        if stokes == "I":
                            kmax = 0
                            # Get an mmax=0 symmetric beam expansion
                            blm = np.atleast_2d(
                                hp.blm_gauss(
                                    self.fwhm.to_value(u.rad),
                                    self.lmax,
                                    pol=False,
                                )
                            )
                            psi = np.zeros_like(theta)
                            alm_ref = np.atleast_2d(alm.data[0])
                        elif stokes in "QU":
                            kmax = 2  # Symmetric, polarized beams
                            # Get an mmax=2 symmetric beam expansion
                            blm = hp.blm_gauss(
                                self.fwhm.to_value(u.rad), self.lmax, pol=True
                            )
                            blm[0] = 0  # Only scan polarization
                            blm *= np.sqrt(2)  # Seems to be required for E/B beam
                            alm_ref = alm.data
                            if stokes == "Q":
                                psi = np.zeros_like(theta) + np.radians(90)
                            else:
                                psi = np.zeros_like(theta) + np.radians(135)
                        else:
                            msg = f"Unsupported Stokes component: {stokes}"
                            raise RuntimeError()
                        interpolator = totalconvolve.Interpolator(
                            alm_ref, blm, separate, lmax, kmax, epsilon
                        )
                        pointing = np.vstack([theta, phi, psi]).T
                        sig = interpolator.interpol(pointing).ravel() * stokes_weights
                        if self.subtract:
                            ref -= sig
                        else:
                            ref += sig

        # Clean up our alm, if needed
        if not self.save_alm:
            self.alms = []

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req.update(self.stokes_weights.requires())
        return req

    def _provides(self):
        prov = {"global": list(), "detdata": [self.det_data]}
        return prov
