# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..timing import function_timer, Timer

from ..pixels import PixelDistribution, PixelData

from .operator import Operator

from .pipeline import Pipeline

from .clear import Clear

from .copy import Copy

from .scan_map import ScanMap

from .noise_weight import NoiseWeight


@trait_docs
class SolverRHS(Operator):
    """Operator for computing the Right Hand Side of the conjugate gradient solver.

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

    overwrite = Bool(
        False, help="Overwrite the input detector data for use as scratch space"
    )

    binning = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a binning operator",
    )

    template_matrix = Instance(
        klass=Operator,
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

        # Check that the inputs are set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")
        if self.binning is None:
            raise RuntimeError("You must set the binning trait before calling exec()")
        if self.template_matrix is None:
            raise RuntimeError(
                "You must set the template_matrix trait before calling exec()"
            )

        # Build a pipeline to make the binned map, optionally one detector at a time.

        self.binning.det_data = self.det_data

        bin_pipe = None
        if self.binning.save_pointing:
            # Process all detectors at once
            bin_pipe = Pipeline(detector_sets=["ALL"])
        else:
            # Process one detector at a time and clear pointing after each one.
            bin_pipe = Pipeline(detector_sets=["SINGLE"])
        bin_pipe.operators = [self.binning]
        bin_pipe.apply(data, detectors=detectors)

        # Build a pipeline for the projection and template matrix application.
        # First create the operators that we will use.

        # Name of the temporary detdata created if we are not overwriting inputs
        det_temp = "temp_RHS"

        # Use the same pointing operator as the binning
        pointing = self.binning.pointing

        # Set up operator for optional clearing of the pointing matrices
        clear_pointing = Clear(detdata=[pointing.pixels, pointing.weights])

        # Optionally Copy data to a temporary location to avoid overwriting the input.
        copy_det = None
        clear_temp = None
        if not self.overwrite:
            copy_det = Copy(
                detdata=[
                    (self.det_data, det_temp),
                ]
            )
            clear_temp = Clear(detdata=[det_temp])

        # The detdata name we will use (either the original or the temp one)
        detdata_name = self.det_data
        if not self.overwrite:
            detdata_name = det_temp

        # Set up map-scanning operator to project the binned map.
        scan_map = ScanMap(
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key=self.binning.binned,
            det_data=detdata_name,
            subtract=True,
        )

        # Set up noise weighting operator
        noise_weight = NoiseWeight(
            noise_model=self.binning.noise_model, det_data=detdata_name
        )

        # Set up template matrix operator.

        self.template_matrix.transpose = True
        self.template_matrix.det_data = detdata_name

        # Create a pipeline that projects the binned map and applies noise
        # weights and templates.

        proj_pipe = None
        if self.binning.save_pointing:
            # Process all detectors at once
            proj_pipe = Pipeline(detector_sets=["ALL"])
            oplist = list()
            if not self.overwrite:
                oplist.append(copy_det)
            oplist.extend(
                [
                    pointing,
                    scan_map,
                    noise_weight,
                    self.template_matrix,
                ]
            )
            if not self.overwrite:
                oplist.append(clear_temp)
            proj_pipe.operators = oplist
        else:
            # Process one detector at a time and clear pointing after each one.
            proj_pipe = Pipeline(detector_sets=["SINGLE"])
            oplist = list()
            if not self.overwrite:
                oplist.append(copy_det)
            oplist.extend(
                [
                    pointing,
                    scan_map,
                    clear_pointing,
                    noise_weight,
                    self.template_matrix,
                ]
            )
            if not self.overwrite:
                oplist.append(clear_temp)
            proj_pipe.operators = oplist

        # Run this projection pipeline.

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
        prov = self.binning.provides()
        prov["meta"].append(self.template_matrix.amplitudes)
        return prov

    def _accelerators(self):
        return list()


@trait_docs
class SolverLHS(Operator):
    """Operator for computing the Left Hand Side of the conjugate gradient solver.

    This operator performs:

    .. math::
        a' = M^T N^{-1} Z M a + M_p a

    Where `a` and `a'` are the input and output template amplitudes.  The template
    amplitudes are stored in the Data object and are updated in place.  `N` is
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

    det_temp = Unicode(
        "temp_LHS", help="Observation detdata key for temporary timestream data"
    )

    binning = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a binning operator",
    )

    template_matrix = Instance(
        klass=Operator,
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

    def _log_debug(self, comm, rank, msg, timer=None):
        """Helper function to log a DEBUG level message from rank zero"""
        log = Logger.get()
        if comm is not None:
            comm.barrier()
        if timer is not None:
            timer.stop()
        if rank == 0:
            if timer is None:
                msg = "MapMaker   LHS {}".format(msg)
            else:
                msg = "MapMaker   LHS {} {:0.2f} s".format(msg, timer.seconds())
            log.debug(msg)
        if timer is not None:
            timer.clear()
            timer.start()

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()

        # The global communicator we are using (or None)
        comm = data.comm.comm_world
        rank = 0
        if comm is not None:
            rank = comm.rank

        # Check that input traits are set
        if self.binning is None:
            raise RuntimeError("You must set the binning trait before calling exec()")
        if self.template_matrix is None:
            raise RuntimeError(
                "You must set the template_matrix trait before calling exec()"
            )

        # Build a pipeline to project amplitudes into timestreams and make a binned
        # map.
        timer.start()
        self._log_debug(comm, rank, "begin project amplitudes and binning")

        self.template_matrix.transpose = False
        self.template_matrix.det_data = self.det_temp
        self.binning.det_data = self.det_temp

        bin_pipe = None
        if self.binning.save_pointing:
            # Process all detectors at once
            bin_pipe = Pipeline(detector_sets=["ALL"])
        else:
            # Process one detector at a time and clear pointing after each one.
            bin_pipe = Pipeline(detector_sets=["SINGLE"])

        bin_pipe.operators = [self.template_matrix, self.binning]

        bin_pipe.apply(data, detectors=detectors)

        self._log_debug(comm, rank, "projection and binning finished in", timer=timer)

        # Build a pipeline for the map scanning and template matrix application.
        # First create the operators that we will use.

        self._log_debug(comm, rank, "begin scan map and accumulate amplitudes")

        # Use the same pointing operator as the binning
        pointing = self.binning.pointing

        # Set up operator for optional clearing of the pointing matrices
        clear_pointing = Clear(detdata=[pointing.pixels, pointing.weights])

        # Set up map-scanning operator to project the binned map.
        scan_map = ScanMap(
            pixels=pointing.pixels,
            weights=pointing.weights,
            map_key=self.binning.binned,
            det_data=self.det_temp,
            subtract=True,
        )

        # Set up noise weighting operator
        noise_weight = NoiseWeight(
            noise_model=self.binning.noise_model, det_data=self.det_temp
        )

        # Same template matrix operator, but now we are applying the transpose.
        self.template_matrix.transpose = True

        # Create a pipeline that projects the binned map and applies noise
        # weights and templates.

        proj_pipe = None
        if self.binning.save_pointing:
            # Process all detectors at once
            proj_pipe = Pipeline(
                detector_sets=["ALL"],
                operators=[
                    pointing,
                    scan_map,
                    noise_weight,
                    self.template_matrix,
                ],
            )
        else:
            # Process one detector at a time and clear pointing after each one.
            proj_pipe = Pipeline(
                detector_sets=["SINGLE"],
                operators=[
                    pointing,
                    scan_map,
                    clear_pointing,
                    noise_weight,
                    self.template_matrix,
                ],
            )

        # Zero out the amplitudes before accumulating the updated values

        data[self.template_matrix.amplitudes].reset()

        # Run the projection pipeline.

        proj_pipe.apply(data, detectors=detectors)

        self._log_debug(
            comm, rank, "map scan and amplitude accumulate finished in", timer=timer
        )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator require everything that its sub-operators needs.
        req = self.binning.requires()
        req.update(self.template_matrix.requires())
        req["meta"].append(self.amplitudes)
        return req

    def _provides(self):
        prov = self.binning.provides()
        return prov

    def _accelerators(self):
        return list()


def solve(
    data,
    detectors,
    lhs,
    rhs_amps,
    guess=None,
    convergence=1.0e-12,
    n_iter_max=100,
    n_iter_min=3,
):
    """Solve for template amplitudes.

    This uses a standard preconditioned conjugate gradient technique (e.g. Shewchuk,
    1994) to solve for the template amplitudes.  The Right Hand Side amplitude values
    are precomputed and passed to this function.  The starting guess of the solver
    can be passed in or else zeros are used.

    Args:
        data (Data):  The distributed data object.
        detectors (list):  The subset of detectors used for the mapmaking.
        lhs (Operator):  The LHS operator.
        rhs_amps (Amplitudes):  The RHS value.
        guess (Amplitudes):  The starting guess.  If None, use all zeros.
        convergence (float):  The convergence limit.
        n_iter_max (int):  The maximum number of iterations.

    Returns:
        None

    """
    log = Logger.get()
    timer_full = Timer()
    timer_full.start()
    timer = Timer()
    timer.start()

    # The global communicator we are using (or None)
    comm = data.comm.comm_world
    rank = 0
    if comm is not None:
        rank = comm.rank

    # Solving A * x = b ...

    # The name of the amplitudes which are updated in place by the LHS operator
    lhs_amps = lhs.template_matrix.amplitudes

    # The starting guess
    if guess is None:
        # Copy structure of the RHS and set to zero
        if lhs_amps in data:
            msg = "LHS amplitudes '{}' already exists in data".format(lhs_amps)
            log.error(msg)
            raise RuntimeError(msg)
        data[lhs_amps] = rhs_amps.duplicate()
        data[lhs_amps].reset()
    else:
        # FIXME:  add a check that the structure of the guess matches the RHS.
        data[lhs_amps] = guess

    # Compute q = A * x (in place)
    lhs.apply(data, detectors=detectors)

    # The initial residual
    # r = b - q
    residual = rhs_amps.duplicate()
    residual -= data[lhs_amps]

    print("RHS ", rhs_amps)
    print("Guess", data[lhs_amps])
    print(residual)

    # The preconditioned residual
    # s = M^-1 * r
    precond_residual = residual.duplicate()
    precond_residual.reset()
    lhs.template_matrix.apply_precond(residual, precond_residual)

    # The proposal
    # d = s
    proposal = precond_residual.duplicate()

    # delta_new = r^T * d
    sqsum = precond_residual.dot(residual)

    init_sqsum = sqsum
    best_sqsum = sqsum
    last_best = sqsum

    sqsum_last = None

    if comm is not None:
        comm.barrier()
    timer.stop()
    if rank == 0:
        msg = "MapMaker initial residual = {}, {:0.2f} s".format(sqsum, timer.seconds())
        log.info(msg)
    timer.clear()
    timer.start()

    for iter in range(n_iter_max):
        if not np.isfinite(sqsum):
            raise RuntimeError("Residual is not finite")

        # Update LHS amplitude inputs
        for k, v in data[lhs_amps].items():
            v.local[:] = proposal[k].local

        # q = A * d (in place)
        lhs.apply(data, detectors=detectors)

        # alpha = delta_new / (d^T * q)
        alpha = sqsum
        alpha /= proposal.dot(data[lhs_amps])

        # r -= alpha * q
        data[lhs_amps] *= alpha
        residual -= data[lhs_amps]

        # The preconditioned residual
        # s = M^-1 * r
        lhs.template_matrix.apply_precond(residual, precond_residual)

        # delta_old = delta_new
        sqsum_last = sqsum

        # delta_new = r^T * s
        sqsum = precond_residual.dot(residual)

        if comm is not None:
            comm.barrier()
        timer.stop()
        if rank == 0:
            msg = "MapMaker iteration {:4d}, relative residual = {}, {:0.2f} s".format(
                iter, sqsum, timer.seconds()
            )
            log.info(msg)
        timer.clear()
        timer.start()

        # beta = delta_new / delta_old
        beta = sqsum / sqsum_last

        # New proposal
        # d = s + beta * d
        proposal *= beta
        proposal += precond_residual

        # Check for convergence
        if sqsum < init_sqsum * convergence or sqsum < 1e-30:
            timer.stop()
            timer_full.stop()
            if rank == 0:
                msg = "MapMaker PCG converged after {:4d} iterations and {:0.2f} seconds".format(
                    iter, timer_full.seconds()
                )
                log.info(msg)
            break

        best_sqsum = min(sqsum, best_sqsum)

        if iter % 10 == 0 and iter >= n_iter_min:
            if last_best < best_sqsum * 2:
                timer.stop()
                timer_full.stop()
                if rank == 0:
                    msg = "MapMaker PCG stalled after {:4d} iterations and {:0.2f} seconds".format(
                        iter, timer_full.seconds()
                    )
                    log.info(msg)
                break
            last_best = best_sqsum
