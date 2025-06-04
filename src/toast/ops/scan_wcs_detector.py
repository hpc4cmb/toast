# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets
from astropy import units as u

from ..observation import default_values as defaults
from ..pixels_io_wcs import read_wcs
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class ScanWCSDetectorMap(Operator):
    """Operator which reads a WCS format map from disk and scans it to a timestream.

    The map file is loaded and distributed among the processes.  For each observation,
    the pointing model is used to expand the pointing and scan the map values into
    detector data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(
        None,
        allow_none=True,
        help="Path to WCS FITS or HDF5 file.  Use ';' if providing multiple files.  "
        "Any focalplane key listed in `focalplane_keys` can be used here and even "
        "formatted. For example: {psi_pol:.0f}",
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for accumulating output"
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
                msg = f"{key} is not in the focalplane during {ob.name}"
                raise KeyError(msg)
            value = focalplane[det][key]
            props[key] = value
        return props

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.map_name = "{}_map".format(self.name)

    @function_timer
    def _get_map(self, filename, wcs_shape, nnz):
        if filename != self.current_filename:
            # Load a new map
            if not os.path.isfile(filename):
                msg = f"No such file: {filename}"
                raise FileNotFoundError(msg)
            # read_wcs() returns a 3D row-major map
            self.current_map = read_wcs(filename, dtype=np.float32)
            # Check for consistency
            map_nnz = self.current_map.shape[0]
            if map_nnz != nnz:
                msg = f"Component mismatch: "
                msg += f"NNZ({self.stokes_weights.name})={nnz} but "
                msg += f"NNZ({filename})={map_nnz}"
                raise ValueError(msg)
            current_shape = self.current_map.shape[1:]
            if current_shape != wcs_shape:
                msg = f"Pixelization mismatch: "
                msg += f"Shape({self.pixel_pointing.name})={wcs_shape} but "
                msg += f"Shape({filename})={current_shape}"
                raise ValueError(msg)
            # Flatten each component map
            self.current_map = self.current_map.reshape([nnz, -1])
            self.current_filename = filename
        return self.current_map

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in ("file", "pixel_pointing", "stokes_weights"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        nnz = len(self.stokes_weights.mode)
        wcs = None
        wcs_shape = None

        # Loop over all observations and local detectors, sampling the appropriate maps
        self.current_filename = None
        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            # If our output detector data does not yet exist, create it
            key = self.det_data
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
                if wcs is None:
                    # WCS is not available until the pixel pointing
                    # operator is executed at least once
                    wcs = self.pixel_pointing.wcs
                    wcs_shape = self.pixel_pointing.wcs_shape
                self.stokes_weights.apply(ob_data, detectors=[det])
                pix = current_ob.detdata[self.pixel_pointing.pixels][det]
                # Get pointing weights
                weights = current_ob.detdata[self.stokes_weights.weights][det].T.copy()

                # Load and sample the provided maps
                file_name = self.file
                detector_file_name = file_name.format(**detector_properties)
                detector_map = self._get_map(detector_file_name, wcs_shape, nnz)
                ref = ob.detdata[self.det_data][det]
                sig = np.zeros_like(ref)
                # Apply appropriate weights to each stokes component.
                for i in range(nnz):
                    sig += detector_map[i][pix] * weights[i]
                if self.subtract:
                    ref -= sig
                else:
                    ref += sig

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req.update(self.stokes_weights.requires())
        return req

    def _provides(self):
        prov = {"global": list(), "detdata": [self.det_data]}
        if self.save_map:
            prov["global"] = [self.map_name]
        return prov
