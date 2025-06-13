# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np
import toast.qarray as qa
import traitlets
from astropy import units as u
from pshmem import MPIShared

from ..observation import default_values as defaults
from ..pixels_io_healpix import read_healpix
from ..timing import function_timer
from ..traits import Bool, Instance, Int, List, Unicode, Unit, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class InterpolateHealpixMap(Operator):
    """Operator which reads a HEALPix format map from disk and
    interpolates it to a timestream.

    The map file is loaded and placed in shared memory on every
    participating node.  For each observation, the pointing model is
    used to expand the pointing and bilinearly interpolate the map
    values into detector data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(
        None,
        allow_none=True,
        help="Path to healpix FITS file.  Use ';' if providing multiple files",
    )

    maps = List(
        None,
        allow_none=True,
        help="HEALpix maps to scan.  If set, `file` must be None.",
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
        False, help="If True, subtract the map timestream instead of accumulating"
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

    save_map = Bool(False, help="If True, do not delete map during finalize")

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

        nset = 0
        exclusive = ("file", "maps")
        for trait in exclusive:
            attr = getattr(self, trait)
            if attr is not None and len(attr) > 0:
                nset += 1
        if nset != 1:
            msg = f"You must set exactly one of '{exclusive}' "
            msg += f"traits before calling exec(), not {nset}"
            raise RuntimeError(msg)

        if self.file is None:
            # Maps are pre-loaded
            nmap = len(self.maps)
            self.file_names = []
        else:
            # Split up the file names
            self.file_names = self.file.split(";")
            nmap = len(self.file_names)
            self.maps = []
        self.det_data_keys = self.det_data.split(";")
        nkey = len(self.det_data_keys)
        if nkey != 1 and (nmap != nkey):
            msg = "If multiple detdata keys are provided, each must have its own map"
            raise RuntimeError(msg)

        # Determine the number of non-zeros from the Stokes weights
        nnz = len(self.stokes_weights.mode)

        # Create our map(s) to scan named after our own operator name.  Generally the
        # files on disk are stored as float32, but even if not there is no real benefit
        # to having higher precision to simulated map signal that is projected into
        # timestreams.

        world_comm = data.comm.comm_world
        if world_comm is None:
            world_rank = 0
        else:
            world_rank = world_comm.rank

        for file_name in self.file_names:
            if world_rank == 0:
                m = np.atleast_2d(read_healpix(file_name, None, dtype=np.float32))
                map_shape = m.shape
            else:
                m = None
                map_shape = None
            if world_comm is not None:
                map_shape = world_comm.bcast(map_shape)
            shared = MPIShared(map_shape, np.float32, world_comm)
            shared.set(m)
            self.maps.append(shared)

        # Loop over all observations and local detectors, interpolating each map
        for ob in data.obs:
            # Get the detectors we are using for this observation
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

            ob_data = data.select(obs_name=ob.name)
            current_ob = ob_data.obs[0]
            for idet, det in enumerate(dets):
                self.detector_pointing.apply(ob_data, detectors=[det])
                self.stokes_weights.apply(ob_data, detectors=[det])
                det_quat = current_ob.detdata[self.detector_pointing.quats][det]
                # Convert pointing quaternion into angles
                theta, phi, _ = qa.to_iso_angles(det_quat)
                # Get pointing weights
                weights = np.atleast_2d(
                    current_ob.detdata[self.stokes_weights.weights][det]
                )

                # Interpolate the provided maps and accumulate the
                # appropriate timestreams in the original observation
                for m in self.maps:
                    if len(self.det_data_keys) == 1:
                        det_data_key = self.det_data_keys[0]
                    else:
                        det_data_key = self.det_data_keys[imap]
                    ref = ob.detdata[det_data_key][det]
                    nside = hp.get_nside(m)
                    interp_pix, interp_weight = hp.pixelfunc.get_interp_weights(
                        nside,
                        theta,
                        phi,
                        nest=False,
                        lonlat=False,
                    )
                    sig = np.zeros_like(ref)
                    for inz, map_column in enumerate(m):
                        sig += weights[:, inz] * np.sum(
                            map_column[interp_pix] * interp_weight, 0
                        )
                    if self.subtract:
                        ref -= sig
                    else:
                        ref += sig

        # Clean up our map, if needed
        if not self.save_map:
            self.maps = None

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req.update(self.stokes_weights.requires())
        return req

    def _provides(self):
        prov = {"global": list(), "detdata": [self.det_data]}
        if self.save_map:
            prov["global"] = self.map_names
        return prov
