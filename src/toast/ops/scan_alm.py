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
        help="Path to a_lm FITS file.  Use ';' if providing multiple files.  "
        "Any focalplane key listed in `focalplane_keys` can be used here and even "
        "formatted. For example: {psi_pol:.0f}"
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

    focalplane_keys = Unicode(
        None,
        allow_none=True,
        help="Comma-separated list of keys to retrieve from the focalplane.  "
        "Used to expand map file names.",
    )

    subtract = Bool(
        False, help="If True, subtract the timestream instead of accumulating"
    )

    zero = Bool(False, help="If True, zero the data before accumulating / subtracting")

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

    def _get_sorted_dets(self, data, detectors=None):
        if self.focalplane_keys is None:
            dets_sorted = [ [], [] ]
            for ob in data.obs:
                dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
                for det in dets:
                    dets_sorted[-2].append(ob.name)
                    dets_sorted[-1].append(det)
            dets_sorted = sorted(zip(*dets_sorted))
        else:
            dets_sorted = [[] for i in range(2 + len(self.focalplane_keys.split(",")))]
            # Loop over all observations and local detectors, sort dets according to focalplane_keys
            for ob in data.obs:
                focalplane = ob.telescope.focalplane
                dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
                for det in dets:
                    for ikey, key in enumerate(self.focalplane_keys.split(",")):
                        if key not in focalplane.detector_data.keys():
                            msg = f"{key} is not in the focalplane during {ob.name}"
                            raise KeyError(msg)
                        dets_sorted[ikey].append(str(focalplane[det][key]))
                    dets_sorted[-2].append(ob.name)
                    dets_sorted[-1].append(det)
            dets_sorted = sorted(zip(*dets_sorted))
        return dets_sorted

    def _parse_alm(self):
        # Split up the file and map names
        self.file_names = self.file.split(";")
        nalm = len(self.file_names)
        self.alms = []
        return nalm

    def _parse_detdata_keys(self):
        self.det_data_keys = self.det_data.split(";")
        nkey = len(self.det_data_keys)
        if nkey != 1 and (len(self.alm) != nkey):
            msg = "If multiple detdata keys are provided, each must have its own alm"
            raise RuntimeError(msg)
        return

    @function_timer
    def _load_alm(self, data, file_name, focalplane_key_value=None):
        """Load, broadcast and save sky a_lm in shared memory"""

        world_comm = data.comm.comm_world
        world_rank = data.comm.world_rank

        #for file_name in self.file_names:
        dtype = complex  # totalconvolve requires dtype=complex
        if self.focalplane_keys is not None:
            # When self.focalplane_keys is set, each group may process
            # differnt input maps. Therefore the alms can't be shared.
            self.alm = hp.read_alm(file_name.format(**focalplane_key_value), hdu=(1, 2, 3)).astype(dtype)
            alm_shape = self.alm.shape
            lmax = hp.Alm.getlmax(self.alm[0].size)
        else:
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

                self.alm = MPIShared(
                    alm_shape,
                    dtype,
                    world_comm,
                    comm_node=data.comm.comm_world_node,
                    comm_node_rank=data.comm.comm_world_node_rank,
                )
                self.alm.set(alm)
            else:
                self.alm = alm

        if self.lmax is None:
            self.lmax = lmax
        elif lmax != self.lmax:
            msg = f"lmax({file_name}) = {lmax} but "
            msg += f"lmax({self.file_names[0]}) = {self.lmax}"
            raise RuntimeError(msg)

        return

    @function_timer
    def _get_pointing(self, ob_data, det):
        """Return detector pointing for one observation"""

        ob = ob_data.obs[0]  # Only one observation in this data object

        # Convert pointing quaternion into angles

        self.detector_pointing.apply(ob_data, detectors=[det])
        det_quat = ob.detdata[self.detector_pointing.quats][det]
        theta, phi, _ = qa.to_iso_angles(det_quat)

        # Get pointing weights

        self.stokes_weights.apply(ob_data, detectors=[det])
        weights = np.atleast_2d(ob.detdata[self.stokes_weights.weights][det]).T

        return theta, phi, weights

    @function_timer
    def _scan_alms(self, interpolators, theta, phi, weights):
        """Scan one a_lm expansion into TOD"""

        signal = np.zeros(theta.size)

        for stokes, stokes_weights in zip(self.stokes_weights.mode, weights):
            if np.all(stokes_weights == 0):
                continue

            if stokes == "I":
                psi = np.zeros_like(theta)
                interpolator = interpolators["I"]
            elif stokes == "Q":
                psi = np.zeros_like(theta) + np.radians(90)
                interpolator = interpolators["QU"]
            elif stokes == "U":
                psi = np.zeros_like(theta) + np.radians(135)
                interpolator = interpolators["QU"]
            else:
                msg = f"Unsupported Stokes component: {stokes}"
                raise RuntimeError(msg)

            phi[phi<0] += 2*np.pi
            pointing = np.vstack([theta, phi, psi]).T
            signal += interpolator.interpol(pointing).ravel() * stokes_weights

        return signal

    @function_timer
    def _cache_blm(self, data):
        """Derive polarized and unpolarized beam expansions"""

        world_comm = data.comm.comm_world
        world_rank = data.comm.world_rank

        if world_rank == 0:
            # Get an mmax=0 symmetric temperature beam expansion
            blm_I = np.atleast_2d(
                hp.blm_gauss(
                    self.fwhm.to_value(u.rad),
                    self.lmax,
                    pol=False,
                )
            )
            blm_I_shape = blm_I.shape
            # Get an mmax=2 symmetric IQU beam expansion
            blm_P = hp.blm_gauss(self.fwhm.to_value(u.rad), self.lmax, pol=True)
            blm_P[0] = 0  # Only scan polarization
            blm_P *= np.sqrt(2)  # Seems to be required for E/B beam
            blm_P_shape = blm_P.shape
        else:
            blm_I = None
            blm_I_shape = None
            blm_P = None
            blm_P_shape = None

        if world_comm is not None:
            blm_I_shape = world_comm.bcast(blm_I_shape)
            blm_P_shape = world_comm.bcast(blm_P_shape)

        if world_comm is not None:
            self._blm_I = MPIShared(
                blm_I_shape,
                complex,
                world_comm,
                comm_node=data.comm.comm_world_node,
                comm_node_rank=data.comm.comm_world_node_rank,
            )
            self._blm_I.set(blm_I)
            self._blm_P = MPIShared(
                blm_P_shape,
                complex,
                world_comm,
                comm_node=data.comm.comm_world_node,
                comm_node_rank=data.comm.comm_world_node_rank,
            )
            self._blm_P.set(blm_P)
        else:
            self._blm_I = blm_I
            self._blm_P = blm_P

        return

    @function_timer
    def _cache_interpolators(self, data, focalplane_key_value=None):
        """Set up the polarized and unpolarized interpolators"""

        separate = False  # Co-add T/E/B
        epsilon = 1e-5

        self.interpolators = []
        for file_name in self.file_names:
            self._load_alm(data, file_name, focalplane_key_value)
            if self._blm_I is None:
                self._cache_blm(data)
            interpolators = {}
            for stokes in "I", "QU":
                if stokes == "I":
                    kmax = 0  # Symmetric, unpolarized beam
                    if focalplane_key_value is None and data.comm.comm_world is not None:
                        alm_ref = np.atleast_2d(self.alm.data[0])
                    else:
                        alm_ref = np.atleast_2d(self.alm[0])
                    if data.comm.comm_world is not None:
                        blm = self._blm_I.data
                    else:
                        blm = self._blm_I
                elif stokes == "QU":
                    kmax = 2  # Symmetric, polarized beams
                    if focalplane_key_value is None and data.comm.comm_world is not None:
                        alm_ref = self.alm.data
                    else:
                        alm_ref = self.alm
                    if data.comm.comm_world is not None:
                        blm = self._blm_P.data
                    else:
                        blm = self._blm_P
                else:
                    msg = f"Unsupported Stokes component: {stokes}"
                    raise RuntimeError(msg)

                interpolators[stokes] = totalconvolve.Interpolator(
                    alm_ref, blm, separate, self.lmax, kmax, epsilon
                )
            self.interpolators.append(interpolators)

            del alm_ref
            if focalplane_key_value is None and data.comm.comm_world is not None:
                self.alm.close()
            del self.alm
        return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        gcomm = data.comm.comm_group

        if not ducc_available:
            msg = "ScanAlm requires ducc0"
            raise RuntimeError(msg)

        for trait in ("detector_pointing", "stokes_weights"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        self._parse_alm()
        self._parse_detdata_keys()
        self._blm_I = None
        self.lmax = None

        dets_sorted = self._get_sorted_dets(data, detectors)
        if self.focalplane_keys is None:
            self._cache_interpolators(data)

        for ob in data.obs:
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            for key in self.det_data_keys:
                # If our output detector data does not yet exist, create it
                exists_data = ob.detdata.ensure(
                    key, detectors=dets, create_units=self.det_data_units
                )
                if self.zero:
                    ob.detdata[key].reset()

        # Loop over all observations and local detectors, sampling each alm
        timer = Timer()
        timer.start()
        ndet = len(dets_sorted)
        prev_fp = None
        for idet, fp_ob_det in enumerate(dets_sorted):
            ob_name = fp_ob_det[-2]
            ob_data = data.select(obs_name=ob_name)
            ob = ob_data.obs[0]
            det = fp_ob_det[-1]
            fp = fp_ob_det[:-2]
            if self.focalplane_keys is not None and fp != prev_fp:
                # Clean up the interpolators
                if prev_fp is not None:
                    log.info(f"Group {data.comm.group:4} : Done detectors with {focalplane_key_value}")
                    del self.interpolators
                focalplane_key_value = {}
                for ikey, key in enumerate(self.focalplane_keys.split(",")):
                    focalplane_key_value[key] = fp_ob_det[ikey]
                self._cache_interpolators(data, focalplane_key_value)
                prev_fp = fp
                log.info(f"Group {data.comm.group:4} : Done building interpolator with {focalplane_key_value}")

            theta, phi, weights = self._get_pointing(ob_data, det)
            for ialm in range(len(self.file_names)):
                if len(self.det_data_keys) == 1:
                    det_data_key = self.det_data_keys[0]
                else:
                    det_data_key = self.det_data_keys[ialm]
                interpolators = self.interpolators[ialm]
                sig = self._scan_alms(interpolators, theta, phi, weights)
                if self.subtract:
                    ob.detdata[det_data_key][det] -= sig
                else:
                    ob.detdata[det_data_key][det] += sig
                del interpolators, sig, theta, phi, weights
            if idet % 1000 == 0:
                log.debug(f"Group {data.comm.group:4} : {idet}/{ndet} detectors finished")

        # Clean up
        del self.interpolators
        if data.comm.comm_world is None:
            del self._blm_I
            del self._blm_P
        else:
            self._blm_I.close()
            del self._blm_I
            self._blm_P.close()
            del self._blm_P

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
