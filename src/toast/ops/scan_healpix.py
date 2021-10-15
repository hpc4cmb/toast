# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import (
    read_healpix_fits,
    read_healpix_hdf5,
    filename_is_fits,
    filename_is_hdf5,
)

from ..observation import default_names as obs_names

from .operator import Operator

from .scan_map import ScanMap

from .pointing import BuildPixelDistribution

from .pipeline import Pipeline


@trait_docs
class ScanHealpix(Operator):
    """Operator which reads a HEALPix format map from disk and scans it to a timestream.

    The map file is loaded and distributed among the processes.  For each observation,
    the pointing model is used to expand the pointing and scan the map values into
    detector data.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(None, allow_none=True, help="Path to healpix FITS file")

    det_data = Unicode(
        obs_names.det_data, help="Observation detdata key for accumulating output"
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
        super().__init__(**kwargs)
        self.map_name = "{}_map".format(self.name)

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

        # Use the pixel odistribution and pointing configuration to allocate our
        # map data and read it in.
        nnz = None
        if self.stokes_weights.mode == "I":
            nnz = 1
        elif self.stokes_weights.mode == "IQU":
            nnz = 3
        else:
            msg = "Unknown Stokes weights mode '{}'".format(self.stokes_weights.mode)
            raise RuntimeError(msg)

        # Create our map to scan named after our own operator name.  Generally the
        # files on disk are stored as float32, but even if not there is no real benefit
        # to having higher precision to simulated map signal that is projected into
        # timestreams.
        if self.map_name not in data:
            data[self.map_name] = PixelData(dist, dtype=np.float32, n_value=nnz)
            if filename_is_fits(self.file):
                read_healpix_fits(
                    data[self.map_name], self.file, nest=self.pixel_pointing.nest
                )
            elif filename_is_hdf5(self.file):
                read_healpix_hdf5(
                    data[self.map_name], self.file, nest=self.pixel_pointing.nest
                )
            else:
                f"Could not determine map format (HDF5 or FITS): {self.file}"
                raise RuntimeError(msg)

        # The pipeline below will run one detector at a time in case we are computing
        # pointing.  Make sure that our full set of requested detector output exists.
        # FIXME:  This seems like a common pattern, maybe move to a "Create" operator?
        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            # If our output detector data does not yet exist, create it
            ob.detdata.ensure(self.det_data, detectors=dets)

        # Configure the low-level map scanning operator

        scanner = ScanMap(
            det_data=self.det_data,
            pixels=self.pixel_pointing.pixels,
            weights=self.stokes_weights.weights,
            map_key=self.map_name,
            subtract=self.subtract,
            zero=self.zero,
        )

        # Build and run a pipeline that scans from our map
        scan_pipe = Pipeline(
            detector_sets=["SINGLE"],
            operators=[self.pixel_pointing, self.stokes_weights, scanner],
        )
        scan_pipe.apply(data, detectors=detectors)

        return

    def _finalize(self, data, **kwargs):
        # Clean up our map, if needed
        if not self.save_map:
            data[self.map_name].clear()
            del data[self.map_name]
        return

    def _requires(self):
        req = self.pixel_pointing.requires()
        req.update(self.stokes_weights.requires())
        return req

    def _provides(self):
        prov = {"detdata": [self.det_data]}
        if self.save_map:
            prof["meta"] = [self.map_name]
        return prov

    def _accelerators(self):
        return list()
