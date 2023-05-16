# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..observation import default_values as defaults
from ..templates import AmplitudesMap
from ..timing import Timer, function_timer
from ..traits import Instance, Int, Unicode, Unit, trait_docs
from ..utils import Logger
from .copy import Copy
from .delete import Delete
from .noise_weight import NoiseWeight
from .operator import Operator
from .pipeline import Pipeline
from .scan_map import ScanMap


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

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

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
            for trt in [
                "pixel_pointing",
                "stokes_weights",
                "det_data",
                "binned",
                "full_pointing",
            ]:
                if not bin.has_trait(trt):
                    msg = f"binning operator should have a '{trt}' trait"
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
        timer = Timer()

        # The global communicator we are using (or None)
        comm = data.comm.comm_world
        rank = data.comm.world_rank

        # Check that the inputs are set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")
        if self.binning is None:
            raise RuntimeError("You must set the binning trait before calling exec()")
        if self.template_matrix is None:
            raise RuntimeError(
                "You must set the template_matrix trait before calling exec()"
            )

        # Make a binned map

        timer.start()
        log.debug_rank("MapMaker   RHS begin binned map", comm=comm)

        self.binning.det_data = self.det_data
        self.binning.det_data_units = self.det_data_units
        self.binning.apply(data, detectors=detectors)

        log.debug_rank("MapMaker   RHS binned map finished in", comm=comm, timer=timer)

        # Build a pipeline for the projection and template matrix application.
        # First create the operators that we will use.

        log.debug_rank("MapMaker   RHS begin create projection pipeline", comm=comm)

        # Name of the temporary detdata created if we are not overwriting inputs
        det_temp = "temp_RHS"

        # Use the same pointing operator as the binning
        pixels = self.binning.pixel_pointing
        weights = self.binning.stokes_weights

        # Optionally Copy data to a temporary location to avoid overwriting the input.
        copy_det = Copy(
            detdata=[
                (self.det_data, det_temp),
            ]
        )

        # Set up map-scanning operator to project the binned map.
        scan_map = ScanMap(
            pixels=pixels.pixels,
            weights=weights.weights,
            view=pixels.view,
            map_key=self.binning.binned,
            det_data=det_temp,
            det_data_units=self.det_data_units,
            subtract=True,
        )

        # Set up noise weighting operator
        noise_weight = NoiseWeight(
            noise_model=self.binning.noise_model,
            det_data=det_temp,
            view=pixels.view,
        )

        # Set up template matrix operator.
        self.template_matrix.transpose = True
        self.template_matrix.det_data = det_temp
        self.template_matrix.det_data_units = self.det_data_units
        self.template_matrix.view = pixels.view

        # Create a pipeline that projects the binned map and applies noise
        # weights and templates.

        proj_pipe = None
        if self.binning.full_pointing:
            # Process all detectors at once, since we have the pointing already
            proj_pipe = Pipeline(detector_sets=["ALL"])
            oplist = [
                copy_det,
                scan_map,
                noise_weight,
                self.template_matrix,
            ]
            proj_pipe.operators = oplist
        else:
            # Process one detector at a time.
            proj_pipe = Pipeline(detector_sets=["SINGLE"])
            oplist = [
                copy_det,
                pixels,
                weights,
                scan_map,
                noise_weight,
                self.template_matrix,
            ]
            proj_pipe.operators = oplist

        log.debug_rank(
            "MapMaker   RHS projection pipeline created in", comm=comm, timer=timer
        )

        # Run this projection pipeline.

        log.debug_rank("MapMaker   RHS begin run projection pipeline", comm=comm)

        # NOTE: we set `use_accel=False` as templates cannot be initialized on device
        proj_pipe.apply(data, detectors=detectors, use_accel=False)

        log.debug_rank(
            "MapMaker   RHS projection pipeline finished in", comm=comm, timer=timer
        )

        log.debug_rank(
            "MapMaker   RHS begin cleanup temporary detector data", comm=comm
        )

        # Clean up our temp buffer
        delete_temp = Delete(detdata=[det_temp])
        delete_temp.apply(data)

        log.debug_rank("MapMaker   RHS cleanup finished in", comm=comm, timer=timer)

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
        prov["global"].append(self.template_matrix.amplitudes)
        return prov


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

    det_data_units = Unit(defaults.det_data_units, help="Desired timestream units")

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

    out = Unicode(
        None, allow_none=True, help="Output Data key for resulting amplitudes"
    )

    @traitlets.validate("binning")
    def _check_binning(self, proposal):
        bin = proposal["value"]
        if bin is not None:
            if not isinstance(bin, Operator):
                raise traitlets.TraitError("binning should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in [
                "pixel_pointing",
                "stokes_weights",
                "det_data",
                "binned",
                "full_pointing",
            ]:
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
        timer = Timer()

        # The global communicator we are using (or None)
        comm = data.comm.comm_world
        rank = data.comm.world_rank

        # Check that input traits are set
        if self.binning is None:
            raise RuntimeError("You must set the binning trait before calling exec()")
        if self.template_matrix is None:
            raise RuntimeError(
                "You must set the template_matrix trait before calling exec()"
            )
        if self.out is None:
            raise RuntimeError("You must set the 'out' trait before calling exec()")

        # Clear temp detector data if it exists
        for ob in data.obs:
            if self.det_temp in ob.detdata:
                ob.detdata[self.det_temp][:] = 0
                ob.detdata[self.det_temp].update_units(self.det_data_units)
                if ob.detdata[self.det_temp].accel_exists():
                    ob.detdata[self.det_temp].accel_reset()

        # Pointing operator used in the binning
        pixels = self.binning.pixel_pointing
        weights = self.binning.stokes_weights

        # Project amplitudes into timestreams and make a binned map.

        timer.start()
        log.debug_rank("MapMaker   LHS begin project amplitudes and binning", comm=comm)

        self.template_matrix.transpose = False
        self.template_matrix.det_data = self.det_temp
        self.template_matrix.det_data_units = self.det_data_units
        self.template_matrix.view = pixels.view

        self.binning.det_data = self.det_temp
        self.binning.det_data_units = self.det_data_units

        self.binning.pre_process = self.template_matrix

        self.binning.apply(data, detectors=detectors)
        self.binning.pre_process = None

        # nz = data[self.binning.binned].data != 0
        # print(
        #     f"LHS binned:  {self.binning.binned} = {data[self.binning.binned].data[nz]}",
        #     flush=True,
        # )

        log.debug_rank(
            "MapMaker   LHS projection and binning finished in", comm=comm, timer=timer
        )

        # Add noise prior

        log.debug_rank("MapMaker   LHS begin add noise prior", comm=comm)

        # Zero out the amplitudes before accumulating the updated values
        if self.out in data:
            data[self.out].reset()

        self.template_matrix.add_prior(
            data[self.template_matrix.amplitudes], data[self.out]
        )

        log.debug_rank(
            "MapMaker   LHS add noise prior finished in", comm=comm, timer=timer
        )

        # Build a pipeline for the map scanning and template matrix application.
        # First create the operators that we will use.

        log.debug_rank(
            "MapMaker   LHS begin scan map and accumulate amplitudes", comm=comm
        )

        # Clear temp detector data
        for ob in data.obs:
            ob.detdata[self.det_temp][:] = 0
            ob.detdata[self.det_temp].update_units(self.det_data_units)
            if ob.detdata[self.det_temp].accel_exists():
                ob.detdata[self.det_temp].accel_reset()

        # Set up map-scanning operator to project the binned map.
        scan_map = ScanMap(
            pixels=pixels.pixels,
            weights=weights.weights,
            view=pixels.view,
            map_key=self.binning.binned,
            det_data=self.det_temp,
            det_data_units=self.det_data_units,
            subtract=True,
        )

        # Set up noise weighting operator
        noise_weight = NoiseWeight(
            noise_model=self.binning.noise_model,
            det_data=self.det_temp,
            view=pixels.view,
        )

        # Make a copy of the template_matrix operator so that we can apply both the
        # matrix and its transpose in a single pipeline

        template_transpose = self.template_matrix.duplicate()
        template_transpose.amplitudes = self.out
        template_transpose.transpose = True

        # Clear temp detector data
        for ob in data.obs:
            ob.detdata[self.det_temp][:] = 0
            ob.detdata[self.det_temp].update_units(self.det_data_units)
            if ob.detdata[self.det_temp].accel_exists():
                ob.detdata[self.det_temp].accel_reset()

        # Create a pipeline that projects the binned map and applies noise
        # weights and templates.

        proj_pipe = None
        if self.binning.full_pointing:
            # Process all detectors at once
            proj_pipe = Pipeline(
                detector_sets=["ALL"],
                operators=[
                    self.template_matrix,
                    scan_map,
                    noise_weight,
                    template_transpose,
                ],
            )
        else:
            # Process one detector at a time.
            proj_pipe = Pipeline(
                detector_sets=["SINGLE"],
                operators=[
                    self.template_matrix,
                    pixels,
                    weights,
                    scan_map,
                    noise_weight,
                    template_transpose,
                ],
            )

        # Run the projection pipeline.

        proj_pipe.apply(data, detectors=detectors)

        log.debug_rank(
            "MapMaker   LHS map scan and amplitude accumulate finished in",
            comm=comm,
            timer=timer,
        )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator require everything that its sub-operators needs.
        req = self.binning.requires()
        req.update(self.template_matrix.requires())
        return req

    def _provides(self):
        prov = self.binning.provides()
        prov["global"].append(self.out)
        return prov


@function_timer
def solve(
    data,
    detectors,
    lhs_op,
    rhs_key,
    result_key,
    convergence=1.0e-12,
    n_iter_max=100,
    n_iter_min=3,
):
    """Solve for template amplitudes.

    This uses a standard preconditioned conjugate gradient technique (e.g. Shewchuk,
    1994) to solve for the template amplitudes.  The Right Hand Side amplitude values
    are precomputed and stored in the data.  The result key in the Data is either
    created or used as the starting guess.

    Args:
        data (Data):  The distributed data object.
        detectors (list):  The subset of detectors used for the mapmaking.
        lhs_op (Operator):  The LHS operator.
        rhs_key (str):  The Data key containing the RHS value.
        result_key (str):  The Data key containing the output result and
            optionally the starting guess.
        convergence (float):  The convergence limit.
        n_iter_max (int):  The maximum number of iterations.
        n_iter_min (int):  The minimum number of iterations, for detecting a stall.

    Returns:
        (Amplitudes):  The result.

    """
    log = Logger.get()
    timer_full = Timer()
    timer_full.start()
    timer = Timer()
    timer.start()

    # The global communicator we are using (or None)
    comm = data.comm.comm_world
    rank = data.comm.world_rank

    if rhs_key not in data:
        msg = "rhs_key '{}' does not exist in data".format(rhs_key)
        log.error(msg)
        raise RuntimeError(msg)
    rhs = data[rhs_key]

    result = None
    if result_key not in data:
        # Copy structure of the RHS and set to zero
        data[result_key] = rhs.duplicate()
        data[result_key].reset()
        result = data[result_key]
    else:
        result = data[result_key]
        if not isinstance(result, AmplitudesMap):
            raise RuntimeError("starting guess must be an AmplitudesMap instance")
        if result.keys() != rhs.keys():
            raise RuntimeError("starting guess must have same keys as RHS")
        for k, v in result.items():
            if v.n_global != rhs[k].n_global:
                msg = (
                    "starting guess['{}'] has different n_global than rhs['{}']".format(
                        k, k
                    )
                )
                raise RuntimeError(msg)
            if v.n_local != rhs[k].n_local:
                msg = (
                    "starting guess['{}'] has different n_global than rhs['{}']".format(
                        k, k
                    )
                )
                raise RuntimeError(msg)

    # Solving A * x = b ...

    # Temporary variables.  We give things more descriptive names here, but to align
    # with the notation in some literature, we note the mapping between variable
    # names.  Duplicate the structure of the RHS for these when we first assign below.

    # The residual "r"
    residual = None

    # The result of the LHS operator "q"
    lhs_out_key = "{}_out".format(lhs_op.name)
    if lhs_out_key in data:
        data[lhs_out_key].clear()
        del data[lhs_out_key]
    data[lhs_out_key] = rhs.duplicate()
    lhs_out = data[lhs_out_key]

    # The result of the preconditioner "s"
    precond = None

    # The new proposed direction "d"
    proposal_key = "{}_in".format(lhs_op.name)
    if proposal_key in data:
        data[proposal_key].clear()
        del data[proposal_key]
    data[proposal_key] = rhs.duplicate()
    data[proposal_key].reset()
    proposal = data[proposal_key]

    # One additional temp variable.  Allocate this now for use below
    temp = rhs.duplicate()
    temp.reset()

    # Compute q = A * x

    # Input is either the starting guess or zero
    lhs_op.template_matrix.amplitudes = result_key
    lhs_op.out = lhs_out_key
    lhs_op.apply(data, detectors=detectors)

    # The initial residual
    # r = b - q
    residual = rhs.duplicate()
    residual -= lhs_out

    # The preconditioned residual
    # s = M^-1 * r
    precond = rhs.duplicate()
    precond.reset()
    lhs_op.template_matrix.apply_precond(residual, precond)

    # The proposal
    # d = s
    for k, v in proposal.items():
        v.local[:] = precond[k].local

    # Set LHS amplitude inputs to this proposal
    lhs_op.template_matrix.amplitudes = proposal_key

    # Epsilon_0 = r^T * r
    sqsum = rhs.dot(rhs)
    sqsum_init = sqsum
    sqsum_best = sqsum
    last_best = sqsum

    # delta_new = delta_0 = r^T * d
    delta = proposal.dot(residual)
    delta_init = delta

    if comm is not None:
        comm.barrier()
    timer.stop()
    if rank == 0:
        msg = "MapMaker initial residual = {}, {:0.2f} s".format(
            sqsum_init, timer.seconds()
        )
        log.info(msg)
    timer.clear()
    timer.start()

    for iter in range(n_iter_max):
        if not np.isfinite(sqsum):
            raise RuntimeError("Residual is not finite")

        # q = A * d
        lhs_op.apply(data, detectors=detectors)
        # print(f"LHS {iter}:  proposal = {proposal}", flush=True)
        # print(f"LHS {iter}:  lhs_out = {lhs_out}", flush=True)

        # alpha = delta_new / (d^T * q)
        alpha = delta / proposal.dot(lhs_out)
        # print(f"LHS {iter}:  delta = {delta}", flush=True)
        # print(f"LHS {iter}:  alpha = {alpha}", flush=True)

        # Update the result
        # x += alpha * d
        temp.reset()
        for k, v in temp.items():
            v.local[:] = proposal[k].local
        temp *= alpha
        result += temp
        # print(f"LHS {iter}:  new result = {result}", flush=True)

        # Update the residual
        # r -= alpha * q
        temp.reset()
        for k, v in temp.items():
            v.local[:] = lhs_out[k].local
        temp *= alpha
        residual -= temp
        # print(f"LHS {iter}:  new residual = {residual}", flush=True)

        # Epsilon
        sqsum = residual.dot(residual)
        # print(f"LHS {iter}:  sqsum = {sqsum}", flush=True)

        if comm is not None:
            comm.barrier()
        timer.stop()
        if rank == 0:
            msg = "MapMaker iteration {:4d}, relative residual = {:0.6e}, {:0.2f} s".format(
                iter, sqsum / sqsum_init, timer.seconds()
            )
            log.info(msg)
        timer.clear()
        timer.start()

        # Check for convergence
        if (sqsum / sqsum_init) < convergence or sqsum < 1e-30:
            timer.stop()
            timer_full.stop()
            if rank == 0:
                msg = "MapMaker PCG converged after {:4d} iterations and {:0.2f} seconds".format(
                    iter, timer_full.seconds()
                )
                log.info(msg)
            break

        sqsum_best = min(sqsum, sqsum_best)

        # Check for stall / divergence
        if iter % 10 == 0 and iter >= n_iter_min:
            if last_best < sqsum_best * 2:
                timer.stop()
                timer_full.stop()
                if rank == 0:
                    msg = "MapMaker PCG stalled after {:4d} iterations and {:0.2f} seconds".format(
                        iter, timer_full.seconds()
                    )
                    log.info(msg)
                break
            last_best = sqsum_best

        # The preconditioned residual
        # s = M^-1 * r
        lhs_op.template_matrix.apply_precond(residual, precond)

        # delta_old = delta_new
        delta_last = delta

        # delta_new = r^T * s
        delta = precond.dot(residual)

        # beta = delta_new / delta_old
        beta = delta / delta_last

        # New proposal
        # d = s + beta * d
        proposal *= beta
        proposal += precond
