# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .operator import Operator
from .pipeline import Pipeline
from .pointing import BuildPixelDistribution
from .scan_map import ScanMap, ScanMask


@trait_docs
class ScanHealpixMap(Operator):
    """Operator which reads a HEALPix format map from disk and scans it to a timestream.

    The map file is loaded and distributed among the processes.  For each observation,
    the pointing model is used to expand the pointing and scan the map values into
    detector data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(
        None,
        allow_none=True,
        help="Path to healpix FITS file.  Use ';' if providing multiple files",
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

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDistribution object is located",
    )

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

    save_map = Bool(False, help="If True, do not delete map during finalize")

    save_pointing = Bool(
        False,
        help="If True, do not clear detector pointing matrices if we "
        "generate the pixel distribution",
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

    def __init__(self, **kwargs):
        self.map_names = []
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the file is set
        if self.file is None:
            raise RuntimeError("You must set the file trait before calling exec()")

        # Split up the file and map names
        self.file_names = self.file.split(";")
        nmap = len(self.file_names)
        self.det_data_keys = self.det_data.split(";")
        nkey = len(self.det_data_keys)
        if nkey != 1 and (nmap != nkey):
            msg = "If multiple detdata keys are provided, each must have its own map"
            raise RuntimeError(msg)
        self.map_names = [f"{self.name}_map{i}" for i in range(nmap)]

        # Construct the pointing distribution if it does not already exist

        if self.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                save_pointing=self.save_pointing,
            )
            pix_dist.apply(data)

        dist = data[self.pixel_dist]
        if not isinstance(dist, PixelDistribution):
            raise RuntimeError("The pixel_dist must be a PixelDistribution instance")

        # Use the pixel distribution and pointing configuration to allocate our
        # map data and read it in.
        nnz = len(self.stokes_weights.mode)

        filenames = self.file.split(";")
        detdata_keys = self.det_data.split(";")

        # Create our map(s) to scan named after our own operator name.  Generally the
        # files on disk are stored as float32, but even if not there is no real benefit
        # to having higher precision to simulated map signal that is projected into
        # timestreams.

        for file_name, map_name in zip(self.file_names, self.map_names):
            if map_name not in data:
                data[map_name] = PixelData(
                    dist, dtype=np.float32, n_value=nnz, units=self.det_data_units
                )
                data[map_name].read(file_name)

        # The pipeline below will run one detector at a time in case we are computing
        # pointing.  Make sure that our full set of requested detector output exists.
        # FIXME:  This seems like a common pattern, maybe move to a "Create" operator?
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

        # Configure the low-level map scanning operator

        scanner = ScanMap(
            det_data=self.det_data_keys[0],
            det_data_units=self.det_data_units,
            det_mask=self.det_mask,
            pixels=self.pixel_pointing.pixels,
            weights=self.stokes_weights.weights,
            map_key=self.map_names[0],
            subtract=self.subtract,
            zero=self.zero,
        )

        # Build and run a pipeline that scans from our map
        scan_pipe = Pipeline(
            detector_sets=["SINGLE"],
            operators=[self.pixel_pointing, self.stokes_weights, scanner],
        )

        for imap, map_name in enumerate(self.map_names):
            if len(self.det_data_keys) == 1:
                det_data_key = self.det_data_keys[0]
            else:
                det_data_key = self.det_data_keys[imap]

            scanner.det_data = det_data_key
            scanner.map_key = map_name
            scan_pipe.apply(data, detectors=detectors, use_accel=False)

            # If we are accumulating on a single key, disable zeroing after first map
            if len(self.det_data_keys) == 1:
                scanner.zero = False

        # Clean up our map, if needed
        if not self.save_map:
            for map_name in self.map_names:
                data[map_name].clear()
                del data[map_name]
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
            prov["global"] = self.map_names
        return prov


@trait_docs
class ScanHealpixMask(Operator):
    """Operator which reads a HEALPix format mask from disk and scans it to a timestream.

    The mask file is loaded and distributed among the processes.  For each observation,
    the pointing model is used to expand the pointing and scan the mask values into
    detector data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(None, allow_none=True, help="Path to healpix FITS file")

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flags_value = Int(
        defaults.det_mask_processing,
        help="The detector flag value to set where the mask result is non-zero",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    mask_bits = Int(
        255, help="The number to bitwise-and with each mask value to form the result"
    )

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDistribution object is located",
    )

    pixel_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pixel pointing operator",
    )

    save_mask = Bool(False, help="If True, do not delete mask during finalize")

    save_pointing = Bool(
        False,
        help="If True, do not clear detector pointing matrices if we "
        "generate the pixel distribution",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_name = f"{self.name}_mask"

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the file is set
        if self.file is None:
            raise RuntimeError("You must set the file trait before calling exec()")

        # Construct the pointing distribution if it does not already exist

        if self.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=self.pixel_dist,
                pixel_pointing=self.pixel_pointing,
                save_pointing=self.save_pointing,
            )
            pix_dist.apply(data)

        dist = data[self.pixel_dist]
        if not isinstance(dist, PixelDistribution):
            raise RuntimeError("The pixel_dist must be a PixelDistribution instance")

        # Create our map to scan named after our own operator name.  Generally the
        # files on disk are stored as float32, but even if not there is no real benefit
        # to having higher precision to simulated map signal that is projected into
        # timestreams.
        if self.mask_name not in data:
            data[self.mask_name] = PixelData(dist, dtype=np.uint8, n_value=1)
            data[self.mask_name].read(self.file)

        # The pipeline below will run one detector at a time in case we are computing
        # pointing.  Make sure that our full set of requested detector output exists.
        # FIXME:  This seems like a common pattern, maybe move to a "Create" operator?
        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            # If our output detector data does not yet exist, create it
            exists_flags = ob.detdata.ensure(
                self.det_flags, dtype=np.uint8, detectors=dets
            )

        # Configure the low-level map scanning operator

        scanner = ScanMask(
            det_flags=self.det_flags,
            det_flags_value=self.det_flags_value,
            det_mask=self.det_mask,
            pixels=self.pixel_pointing.pixels,
            mask_key=self.mask_name,
            mask_bits=self.mask_bits,
        )

        # Build and run a pipeline that scans from our map
        scan_pipe = Pipeline(
            detector_sets=["SINGLE"],
            operators=[self.pixel_pointing, scanner],
        )
        scan_pipe.apply(data, detectors=detectors)

        return

    def _finalize(self, data, **kwargs):
        # Clean up our map, if needed
        if not self.save_mask:
            data[self.mask_name].clear()
            del data[self.mask_name]
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
