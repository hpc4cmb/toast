# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import read_healpix_fits

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

    pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator",
    )

    save_map = Bool(False, help="If True, do not delete map during finalize")

    save_pointing = Bool(
        False,
        help="If True, do not clear detector pointing matrices if we generate the pixel distribution",
    )

    @traitlets.validate("pointing")
    def _check_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError("pointing should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["pixels", "weights", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = "pointing operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return pntg

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
                pointing=self.pointing,
                save_pointing=self.save_pointing,
            )
            pix_dist.apply(data)

        dist = data[self.pixel_dist]
        if not isinstance(dist, PixelDistribution):
            raise RuntimeError("The pixel_dist must be a PixelDistribution instance")

        # Use the pixel distribution and pointing configuration to allocate our
        # map data and read it in.
        nnz = None
        if self.pointing.mode == "I":
            nnz = 1
        elif self.pointing.mode == "IQU":
            nnz = 3
        else:
            msg = "Unknown healpix pointing mode '{}'".format(self.pointing.mode)
            raise RuntimeError(msg)

        # Create our map to scan named after our own operator name.  Generally the
        # files on disk are stored as float32, but even if not there is no real benefit
        # to having higher precision to simulated map signal that is projected into
        # timestreams.
        if self.map_name not in data:
            data[self.map_name] = PixelData(dist, dtype=np.float32, n_value=nnz)
            read_healpix_fits(data[self.map_name], self.file, nest=self.pointing.nest)

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
            pixels=self.pointing.pixels,
            weights=self.pointing.weights,
            map_key=self.map_name,
            subtract=self.subtract,
            zero=self.zero,
        )

        # Build and run a pipeline that scans from our map
        scan_pipe = Pipeline(
            detector_sets=["SINGLE"], operators=[self.pointing, scanner]
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
        req = self.pointing.requires()
        return req

    def _provides(self):
        prov = {"detdata": [self.det_data]}
        if self.save_map:
            prof["meta"] = [self.map_name]
        return prov

    def _accelerators(self):
        return list()
