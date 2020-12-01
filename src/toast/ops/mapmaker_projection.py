# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from .operator import Operator

from .pipeline import Pipeline

from .clear import Clear

from .copy import Copy

from .scan_map import ScanMap


@trait_docs
class Projection(Operator):
    """Operator for map-making projection to template amplitudes.

    This operator performs:

    .. math::
        a = M^T N^{-1} Z d

    Where `d` is a set of timestreams and `a` are the projected amplitudes.  `N` is
    the time domain diagonal noise covariance and `M` is a set of templates.  The `Z`
    matrix is given by:

    .. math::
        Z = I - P (P^T N^{-1} P)^{-1} P^T N^{-1}

    Where `P` is the pointing matrix.  In terms of the binning operation this is:

    .. math::
        Z = I - P B

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    binning = Instance(
        klass=None,
        allow_none=True,
        help="This must be an instance of a binning operator",
    )

    template_matrix = Instance(
        klass=None,
        allow_none=True,
        help="This must be an instance of a template matrix operator",
    )

    @traitlets.validate("binning")
    def _check_binning(self, proposal):
        bin = proposal["value"]
        if bin is not None:
            if not isinstance(bin, Operator):
                raise traitlets.TraitError("binning should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["pointing", "det_data", "binned"]:
                if not bin.has_trait(trt):
                    msg = "binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

    @traitlets.validate("template_matrix")
    def _check_matrix(self, proposal):
        mat = proposal["value"]
        if mat is not None:
            if not isinstance(mat, Operator):
                raise traitlets.TraitError(
                    "template_matrix should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in ["templates", "amplitudes", "det_data", "transpose"]:
                if not mat.has_trait(trt):
                    msg = "template_matrix operator should have a '{}' trait".format(
                        trt
                    )
                    raise traitlets.TraitError(msg)
        return mat

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Set data input for binning
        self.binning.det_data = self.det_data

        # Use the same pointing operator as the binning
        pointing = self.binning.pointing

        # Set up operator for optional clearing of the pointing matrices
        clear_pointing = Clear(detdata=[pointing.pixels, pointing.weights])

        # Name of the temporary detdata created
        det_temp = "temp_projection"

        # Copy data operator
        copy_det = Copy(
            detdata=[
                (self.det_data, det_temp),
            ]
        )

        # Set up map-scanning operator
        scan_map = ScanMap(
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key=self.binning.binned,
            det_data=det_temp,
            subtract=True,
        )

        # Set up noise weighting operator
        noise_weight = NoiseWeight(
            noise_model=self.binning.noise_model, det_data=det_temp
        )

        # Set up template matrix operator

        self.template_matrix.transpose = True
        self.template_matrix.det_data = det_temp

        # Create a pipeline that projects the binned map and applies noise
        # weights and templates.

        proj_pipe = None
        if self.binning.save_pointing:
            # Process all detectors at once
            proj_pipe = Pipeline(detector_sets=["ALL"])
            proj_pipe.operators = [
                copy_det,
                pointing,
                scan_map,
                noise_weight,
                self.template_matrix,
            ]
        else:
            # Process one detector at a time and clear pointing after each one.
            proj_pipe = Pipeline(detector_sets=["SINGLE"])
            proj_pipe.operators = [
                copy_det,
                pointing,
                scan_map,
                clear_pointing,
                noise_weight,
                self.template_matrix,
            ]

        # Compute the binned map.

        self.binning.apply(data, detectors=detectors)

        # Project and apply template matrix.

        proj_pipe.apply(data, detectors=detectors)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator require everything that its sub-operators needs.
        req = self.binning.requires()
        req.update(self.template_matrix.requires())
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": list()}
        if self.save_pointing:
            prov["detdata"].extend([self.pixels, self.weights])
        return prov

    def _accelerators(self):
        return list()
