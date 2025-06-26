# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import toast.qarray as qa
import traitlets
from astropy import units as u
from pshmem import MPIShared

from ..observation import default_values as defaults
from ..pixels_io_healpix import read_healpix
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class ScanHealpixDetectorMap(Operator):
    """Operator which reads a HEALPix format map from disk and scans it
    to a timestream.

    Detectors can be matched to different input maps through the use of
    detector attributes such as pixel, wafer or optics tube.

    Each process loads and discards maps independently which may incur
    significant memory overhead.  At most one map per process is kept
    in memory.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(
        None,
        allow_none=True,
        help="Path to healpix FITS file.  Use ';' if providing multiple files.  "
        "Any focalplane key listed in `focalplane_keys` can be used here and even "
        "formatted. For example: {psi_pol:.0f}",
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
        "pixel",
        help="Comma-separated list of keys to retrieve from the focalplane.  "
        "Used to expand map file names.",
    )

    subtract = Bool(
        False, help="If True, subtract the map timestream instead of accumulating"
    )

    zero = Bool(False, help="If True, zero the data before accumulating / subtracting")

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pixel pointing operator",
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

    @traitlets.validate("pixel_pointing")
    def _check_pixel_pointing(self, proposal):
        pixels = proposal["value"]
        if pixels is not None:
            if not isinstance(pixels, Operator):
                raise traitlets.TraitError(
                    "pixel_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["pixels", "create_dist", "view"]:
                if not pixels.has_trait(trt):
                    msg = f"pixel_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return pixels

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

    def _get_properties(self, ob, det):
        """Find values for all specified focalplane keys and return them
        in a dictionary.
        """
        focalplane = ob.telescope.focalplane
        if det not in focalplane:
            msg = f"{det} is not in the focalplane during {ob.name}"
            raise KeyError(msg)
        props = {}
        for key in self.focalplane_keys.split(","):
            if key not in focalplane.detector_data.keys():
                msg = f"{key} is not in the focalplane during {ob.name}." \
                    + "Valid keys are: {}".format(focalplane.detector_data.keys())
                raise KeyError(msg)
            value = focalplane[det][key]
            props[key] = value
        return props

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ("file", "pixel_pointing", "stokes_weights"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        # Split up the file and map names
        self.file_names = self.file.split(";")
        nmap = len(self.file_names)
        self.det_data_keys = self.det_data.split(";")
        nkey = len(self.det_data_keys)
        if nkey != 1 and (nmap != nkey):
            msg = "If multiple detdata keys are provided, each must have its own map"
            raise RuntimeError(msg)

        # Determine the number of non-zeros from the Stokes weights
        nnz = len(self.stokes_weights.mode)
        if nnz == 2:
            field = (1, 2)
        else:
            field = tuple(np.arange(nnz))

        # Loop over all observations and local detectors, sampling the appropriate maps
        last_file_name = None
        current_map = None
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
                detector_properties = self._get_properties(ob, det)
                # Get pointing
                self.pixel_pointing.apply(ob_data, detectors=[det])
                self.stokes_weights.apply(ob_data, detectors=[det])
                pix = current_ob.detdata[self.pixel_pointing.pixels][det]
                nest = self.pixel_pointing.nest
                nside = self.pixel_pointing.nside
                # Get pointing weights
                weights = current_ob.detdata[self.stokes_weights.weights][det].T.copy()

                # Load and sample the provided maps
                for imap, file_name in enumerate(self.file_names):
                    current_file_name = file_name.format(**detector_properties)
                    if current_file_name != last_file_name:
                        # Load a new map
                        if not os.path.isfile(current_file_name):
                            msg = f"No such file: {current_file_name}"
                            raise FileNotFoundError(msg)
                        current_map = np.atleast_2d(
                            read_healpix(
                                current_file_name,
                                field,
                                nest=nest,
                                dtype=np.float32,
                                verbose=False,
                            )
                        )
                        nside_map = hp.get_nside(current_map)
                        if nside_map != nside:
                            msg = f"Resolution mismatch: "
                            msg += f"NSide({self.pixel_pointing.name})={nside} but "
                            msg += f"NSide({current_file_name})={nside_map}"
                            raise ValueError(msg)
                        last_file_name = current_file_name
                    if len(self.det_data_keys) == 1:
                        det_data_key = self.det_data_keys[0]
                    else:
                        det_data_key = self.det_data_keys[imap]
                    ref = ob.detdata[det_data_key][det]
                    sig = np.zeros_like(ref)
                    # Apply appropriate weights to each stokes component.
                    # Will also work if weights are polarized but map is
                    # intensity-only
                    for i in range(nnz):
                        sig += current_map[i][pix] * weights[i]
                        if len(current_map) == 1:
                            break
                    if self.subtract:
                        ref -= sig
                    else:
                        ref += sig

        # Clean up
        del current_map

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
