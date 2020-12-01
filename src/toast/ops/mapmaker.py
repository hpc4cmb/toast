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

    Where `P` is the pointing matrix.  This operator takes a "Projection" instance
    as one of its traits, and that operator performs:

    .. math::
        PROJ = M^T N^{-1} Z

    This projection operator is then used to compute the right hand side of the
    solver and for each calculation of the left hand side.

    After solving for the template amplitudes, a final map of the signal estimate is
    computed using a simple binning:

    .. math::
        MAP = ({P'}^T N^{-1} P')^{-1} {P'}^T N^{-1} (y - M a)

    Where the "prime" indicates that this final map might be computed using a different
    pointing matrix than the one used to solve for the template amplitudes.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    projection = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a projection operator",
    )

    map_binning = Instance(
        klass=Operator,
        allow_none=True,
        help="Binning operator for final map making.  Default uses same operator as projection.",
    )

    @traitlets.validate("projection")
    def _check_projection(self, proposal):
        proj = proposal["value"]
        if proj is not None:
            if not isinstance(bin, Operator):
                raise traitlets.TraitError("binning should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["templated_matrix", "det_data", "binning"]:
                if not bin.has_trait(trt):
                    msg = "binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

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

        # Check projection
        if self.projection is None:
            raise RuntimeError(
                "You must set the projection trait before calling exec()"
            )

        # Check map binning
        if self.map_binning is None:
            self.map_binning = self.projection.binning

        # Get the template matrix used in the projection
        template_matrix = self.projection.template_matrix

        # Compute the RHS

        if template_matrix.amplitudes in data:
            # Clear any existing amplitudes, so that it will be created.
            del data[template_matrix.amplitudes]

        self.projection.det_data = self.det_data
        self.projection.apply(data)

        # Copy structure of RHS and zero as starting point for the solver

        # Solve for amplitudes

        return

    @function_timer
    def _solve(self):
        """Standard issue PCG solution of A.x = b

        Returns:
            x : the least squares solution
        """
        log = Logger.get()
        timer0 = Timer()
        timer0.start()
        timer = Timer()
        timer.start()
        # Initial guess is zero amplitudes
        guess = self.templates.zero_amplitudes()
        # print("guess:", guess)  # DEBUG
        # print("RHS:", self.rhs)  # DEBUG
        residual = self.rhs.copy()
        # print("residual(1):", residual)  # DEBUG
        residual -= self.apply_lhs(guess)
        # print("residual(2):", residual)  # DEBUG
        precond_residual = self.templates.apply_precond(residual)
        proposal = precond_residual.copy()
        sqsum = precond_residual.dot(residual)
        init_sqsum, best_sqsum, last_best = sqsum, sqsum, sqsum
        if self.rank == 0:
            log.info("Initial residual: {}".format(init_sqsum))
        # Iterate to convergence
        for iiter in range(self.niter_max):
            if not np.isfinite(sqsum):
                raise RuntimeError("Residual is not finite")
            alpha = sqsum
            alpha /= proposal.dot(self.apply_lhs(proposal))
            alpha_proposal = proposal.copy()
            alpha_proposal *= alpha
            guess += alpha_proposal
            residual -= self.apply_lhs(alpha_proposal)
            del alpha_proposal
            # Prepare for next iteration
            precond_residual = self.templates.apply_precond(residual)
            beta = 1 / sqsum
            # Check for convergence
            sqsum = precond_residual.dot(residual)
            if self.rank == 0:
                timer.report_clear(
                    "Iter = {:4} relative residual: {:12.4e}".format(
                        iiter, sqsum / init_sqsum
                    )
                )
            if sqsum < init_sqsum * self.convergence_limit or sqsum < 1e-30:
                if self.rank == 0:
                    timer0.report_clear(
                        "PCG converged after {} iterations".format(iiter)
                    )
                break
            best_sqsum = min(sqsum, best_sqsum)
            if iiter % 10 == 0 and iiter >= self.niter_min:
                if last_best < best_sqsum * 2:
                    if self.rank == 0:
                        timer0.report_clear(
                            "PCG stalled after {} iterations".format(iiter)
                        )
                    break
                last_best = best_sqsum
            # Select the next direction
            beta *= sqsum
            proposal *= beta
            proposal += precond_residual
        # log.info("{} : Solution: {}".format(self.rank, guess))  # DEBUG
        return guess

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator require everything that its sub-operators needs.
        req = self.projection.requires()
        if self.map_binning is not None:
            req.update(self.map_binning.requires())
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        prov = dict()
        if self.map_binning is not None:
            prov["meta"] = [self.map_binning.binned]
        else:
            prov["meta"] = [self.projection.binning.binned]
        return prov

    def _accelerators(self):
        return list()
