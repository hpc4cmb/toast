# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Float, Instance

from ..timing import function_timer

from ..pixels import PixelDistribution, PixelData

from .operator import Operator

from .pipeline import Pipeline

from .clear import Clear

from .copy import Copy

from .scan_map import ScanMap


@trait_docs
class MapMaker(Operator):
    """Operator for making maps.

    This operator first solves for a maximum likelihood set of template amplitudes
    that model the timestream contributions from noise, systematics, etc:

    .. math::
        \left[ M^T N^{-1} Z M + M_p \right] a = M^T N^{-1} Z d

    Where `a` are the solved amplitudes and `d` is the input data.  `N` is the
    diagonal time domain noise covariance.  `M` is a matrix of templates that
    project from the amplitudes into the time domain, and the `Z` operator is given
    by:

    .. math::
        Z = I - P (P^T N^{-1} P)^{-1} P^T N^{-1}

    or in terms of the binning operation:

    .. math::
        Z = I - P B

    Where `P` is the pointing matrix.  This operator takes one operator for the
    template matrix `M` and one operator for the binning, `B`.  It then
    uses a conjugate gradient solver to solve for the amplitudes.

    After solving for the template amplitudes, a final map of the signal estimate is
    computed using a simple binning:

    .. math::
        MAP = ({P'}^T N^{-1} P')^{-1} {P'}^T N^{-1} (y - M a)

    Where the "prime" indicates that this final map might be computed using a different
    pointing matrix than the one used to solve for the template amplitudes.

    The template-subtracted detector timestreams are saved either in the input
    `det_data` key of each observation, or (if overwrite == False) in an obs.detdata
    key that matches the name of this class instance.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    convergence = Float(1.0e-12, help="Relative convergence limit")

    iter_max = Int(100, help="Maximum number of iterations")

    overwrite = Bool(
        False, help="Overwrite the input detector data for use as scratch space"
    )

    binning = Instance(
        klass=Operator,
        allow_none=True,
        help="Binning operator used for solving template amplitudes",
    )

    template_matrix = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a template matrix operator",
    )

    map_binning = Instance(
        klass=Operator,
        allow_none=True,
        help="Binning operator for final map making.  Default uses same operator as solver.",
    )

    @traitlets.validate("map_binning")
    def _check_binning(self, proposal):
        bin = proposal["value"]
        if bin is not None:
            if not isinstance(bin, Operator):
                raise traitlets.TraitError("map_binning should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["det_data", "binned"]:
                if not bin.has_trait(trt):
                    msg = "map_binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check map binning
        if self.map_binning is None:
            # Use the same binning as the projection operator used in the solver.
            self.map_binning = self.projection.binning

        # For computing the RHS and also for each iteration of the LHS we will need
        # a full detector-data sized buffer for use as scratch space.  We can either
        # destroy the input data to save memory (useful if this is the last operator
        # processing the data) or we can create a temporary set of timestreams.

        copy_det = None
        clear_temp = None
        detdata_name = self.det_data

        if not self.overwrite:
            # Use a temporary detdata named after this operator
            detdata_name = self.name
            # Copy the original data into place, and then use this copy destructively.
            copy_det = Copy(
                detdata=[
                    (self.det_data, detdata_name),
                ]
            )
            copy_det.apply(data, detectors=detectors)

        # Compute the RHS.  Overwrite inputs, either the original or the copy.

        self.template_matrix.amplitudes = "amplitudes_rhs"
        rhs_calc = SolverRHS(
            det_data=detdata_name,
            overwrite=True,
            binning=self.binning,
            template_matrix=self.template_matrix,
        )
        rhs.apply(data, detectors=detectors)

        # Set up the LHS operator.  Use either the original timestreams or the copy
        # as temp space.

        self.template_matrix.amplitudes = "amplitudes"
        lhs_calc = SolverLHS(
            det_temp=detdata_name,
            binning=self.binning,
            template_matrix=self.template_matrix,
        )

        # Solve for amplitudes.
        solve(
            data,
            detectors,
            lhs_calc,
            data["amplitudes_rhs"],
            convergence=self.convergence,
            n_iter_max=self.iter_max,
        )

        # Reset our timestreams to zero
        for ob in data.obs:
            ob.detdata[detdata_name][:] = 0.0

        # Project our solved amplitudes into timestreams.  We output to either the
        # input det_data or our temp space.

        self.template_matrix.transpose = False
        self.template_matrix.apply(data, detectors=detectors)

        # Make a binned map of these template-subtracted timestreams

        self.map_binning.det_data = detdata_name
        self.map_binning.apply(data, detectors=detectors)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator require everything that its sub-operators needs.
        req = self.binning.requires()
        req.update(self.template_matrix.requires())
        if self.map_binning is not None:
            req.update(self.map_binning.requires())
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        prov = dict()
        if self.map_binning is not None:
            prov["meta"] = [self.map_binning.binned]
        else:
            prov["meta"] = [self.binning.binned]
        return prov

    def _accelerators(self):
        return list()
