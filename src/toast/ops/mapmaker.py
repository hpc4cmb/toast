# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..mpi import MPI

from ..traits import trait_docs, Int, Unicode, Bool, Float, Instance

from ..timing import function_timer, Timer

from ..pixels import PixelDistribution, PixelData

from .operator import Operator

from .pipeline import Pipeline

from .delete import Delete

from .copy import Copy

from .arithmetic import Subtract

from .scan_map import ScanMap, ScanMask

from .mapmaker_utils import CovarianceAndHits

from .mapmaker_solve import solve, SolverRHS, SolverLHS


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
    key based on the name of this class instance.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    convergence = Float(1.0e-12, help="Relative convergence limit")

    iter_max = Int(100, help="Maximum number of iterations")

    solve_rcond_threshold = Float(
        1.0e-8,
        help="When solving, minimum value for inverse pixel condition number cut.",
    )

    map_rcond_threshold = Float(
        1.0e-8,
        help="For final map, minimum value for inverse pixel condition number cut.",
    )

    mask = Unicode(
        None,
        allow_none=True,
        help="Data key for pixel mask to use in solving.  First bit of pixel values is tested",
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
        help="Binning operator for final map making.  Default is same as solver",
    )

    mc_mode = Bool(False, help="If True, re-use solver flags, sparse covariances, etc")

    save_cleaned = Bool(
        False, help="If True, save the template-subtracted detector timestreams"
    )

    overwrite_cleaned = Bool(
        False, help="If True and save_cleaned is True, overwrite the input data"
    )

    @traitlets.validate("binning")
    def _check_binning(self, proposal):
        bin = proposal["value"]
        if bin is not None:
            if not isinstance(bin, Operator):
                raise traitlets.TraitError("binning should be an Operator instance")
            # Check that this operator has the traits we require
            for trt in [
                "det_data",
                "pixel_dist",
                "pointing",
                "binned",
                "covariance",
                "det_flags",
                "det_flag_mask",
                "shared_flags",
                "shared_flag_mask",
                "noise_model",
                "saved_pointing",
                "sync_type",
            ]:
                if not bin.has_trait(trt):
                    msg = "binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

    @traitlets.validate("map_binning")
    def _check_map_binning(self, proposal):
        bin = proposal["value"]
        if bin is not None:
            if not isinstance(bin, Operator):
                raise traitlets.TraitError("map_binning should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in [
                "det_data",
                "pixel_dist",
                "pointing",
                "binned",
                "covariance",
                "det_flags",
                "det_flag_mask",
                "shared_flags",
                "shared_flag_mask",
                "noise_model",
                "saved_pointing",
                "sync_type",
            ]:
                if not bin.has_trait(trt):
                    msg = "map_binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _log_info(self, comm, rank, msg, timer=None):
        """Helper function to log an INFO level message from rank zero"""
        log = Logger.get()
        if comm is not None:
            comm.barrier()
        if timer is not None:
            timer.stop()
        if rank == 0:
            if timer is None:
                msg = "MapMaker {}".format(msg)
            else:
                msg = "MapMaker {} {:0.2f} s".format(msg, timer.seconds())
            log.info(msg)
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

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check map binning
        if self.map_binning is None:
            # Use the same binning used in the solver.
            self.map_binning = self.binning

        # Binning parameters for the solver.

        # We use the input binning operator to define the flags that the user has
        # specified.  We will save the name / bit mask for these and restore them later.
        # Then we will use the binning operator with our solver flags.  These input
        # flags are combined to the first bit (== 1) of the solver flags.

        save_det_flags = self.binning.det_flags
        save_det_flag_mask = self.binning.det_flag_mask
        save_shared_flags = self.binning.shared_flags
        save_shared_flag_mask = self.binning.shared_flag_mask
        save_covariance = self.binning.covariance

        # Also save the name of the user-requested output binned map.  During the
        # solve we will output to a temporary map and then restore this name, in
        # case we are using the same binning operator for the solve and the final
        # output.
        save_binned = self.binning.binned

        # Data products, prefixed with the name of the operator.

        solver_hits_name = "{}_solve_hits".format(self.name)
        solver_cov_name = "{}_solve_cov".format(self.name)
        solver_rcond_name = "{}_solve_rcond".format(self.name)
        solver_rcond_mask_name = "{}_solve_rcond_mask".format(self.name)
        solver_result = "{}_solve_amplitudes".format(self.name)
        solver_rhs = "{}_solve_rhs".format(self.name)
        solver_bin = "{}_solve_bin".format(self.name)

        hits_name = "{}_hits".format(self.name)
        cov_name = "{}_cov".format(self.name)
        rcond_name = "{}_rcond".format(self.name)

        flagname = "{}_flags".format(self.name)
        clean_name = "{}_cleaned".format(self.name)

        self.binning.covariance = solver_cov_name

        timer.start()

        # Flagging.  We create a new set of data flags for the solver that includes:
        #   - one bit for a bitwise OR of all detector / shared flags
        #   - one bit for any pixel mask, projected to TOD
        #   - one bit for any poorly conditioned pixels, projected to TOD

        if self.mc_mode:
            # Verify that our flags exist
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                for d in dets:
                    if d not in ob.detdata[flagname].detectors:
                        msg = "In MC mode, flags missing for observation {}, det {}".format(
                            ob.name, d
                        )
            self._log_info(comm, rank, "MC mode, reusing flags for solver")
        else:
            self._log_info(comm, rank, "begin building flags for solver")

            # Use the same data view as the pointing operator in binning
            solve_view = self.binning.pointing.view

            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                # Create the new solver flags
                ob.detdata.ensure(flagname, dtype=np.uint8, detectors=detectors)
                # The data views
                views = ob.view[solve_view]
                # For each view...
                for vw in range(len(views)):
                    view_samples = None
                    if views[vw].start is None:
                        # There is one view of the whole obs
                        view_samples = ob.n_local_samples
                    else:
                        view_samples = views[vw].stop - views[vw].start
                    starting_flags = np.zeros(view_samples, dtype=np.uint8)
                    if save_shared_flags is not None:
                        starting_flags[:] = np.where(
                            views.shared[save_shared_flags][vw] & save_shared_flag_mask
                            > 0,
                            1,
                            0,
                        )
                    for d in dets:
                        views.detdata[flagname][vw][d, :] = starting_flags
                        if save_det_flags is not None:
                            views.detdata[flagname][vw][d, :] |= np.where(
                                views.detdata[save_det_flags][vw][d]
                                & save_det_flag_mask
                                > 0,
                                1,
                                0,
                            )

            # Now scan any input mask to this same flag field.  We use the second bit
            # (== 2) for these mask flags.  For the input mask bit we check the first
            # bit of the pixel values.  This is noted in the help string for the mask
            # trait.  Note that we explicitly expand the pointing once here and do not
            # save it.  Even if we are eventually saving the pointing, we want to do
            # that later when building the covariance and the pixel distribution.

            # Use the same pointing operator as the binning
            scan_pointing = self.binning.pointing

            scanner = ScanMask(
                det_flags=flagname,
                pixels=scan_pointing.pixels,
                view=solve_view,
                mask_bits=1,
            )

            scanner.det_flags_value = 2
            scanner.mask_key = self.mask

            scan_pipe = Pipeline(
                detector_sets=["SINGLE"], operators=[scan_pointing, scanner]
            )

            if self.mask is not None:
                # We have a mask.  Scan it.
                scan_pipe.apply(data, detectors=detectors)

            self._log_info(comm, rank, "  finished flag building in", timer=timer)

        # Now construct the noise covariance, hits, and condition number mask for
        # the solver.

        if self.mc_mode:
            # Verify that our covariance and other products exist.
            if self.binning.pixel_dist not in data:
                msg = "MC mode, pixel distribution '{}' does not exist".format(
                    self.binning.pixel_dist
                )
                log.error(msg)
                raise RuntimeError(msg)
            if self.binning.covariance not in data:
                msg = "MC mode, covariance '{}' does not exist".format(
                    self.binning.covariance
                )
                log.error(msg)
                raise RuntimeError(msg)

            self._log_info(comm, rank, "MC mode, reusing covariance for solver")
        else:
            self._log_info(comm, rank, "begin build of solver covariance")

            solver_cov = CovarianceAndHits(
                pixel_dist=self.binning.pixel_dist,
                covariance=self.binning.covariance,
                hits=solver_hits_name,
                rcond=solver_rcond_name,
                det_flags=flagname,
                det_flag_mask=255,
                pointing=self.binning.pointing,
                noise_model=self.binning.noise_model,
                rcond_threshold=self.solve_rcond_threshold,
                sync_type=self.binning.sync_type,
                save_pointing=self.binning.saved_pointing,
            )

            solver_cov.apply(data, detectors=detectors)

            data[solver_rcond_mask_name] = PixelData(
                data[self.binning.pixel_dist], dtype=np.uint8, n_value=1
            )
            data[solver_rcond_mask_name].raw[
                data[solver_rcond_name].raw.array() < self.solve_rcond_threshold
            ] = 1

            # Re-use our mask scanning pipeline, setting third bit (== 4)
            scanner.det_flags_value = 4
            scanner.mask_key = solver_rcond_mask_name
            scan_pipe.apply(data, detectors=detectors)

            self._log_info(
                comm, rank, "  finished build of solver covariance in", timer=timer
            )

            local_total = 0
            local_cut = 0
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                for vw in ob.view[solve_view].detdata[flagname]:
                    for d in dets:
                        local_total += len(vw[d])
                        local_cut += np.count_nonzero(vw[d])
            total = None
            cut = None
            msg = None
            if comm is None:
                total = local_total
                cut = local_cut
            else:
                total = comm.reduce(local_total, op=MPI.SUM, root=0)
                cut = comm.reduce(local_cut, op=MPI.SUM, root=0)
                if comm.rank == 0:
                    msg = "Solver flags cut {} / {} = {:0.2f}% of samples".format(
                        cut, total, 100.0 * (cut / total)
                    )
            self._log_info(comm, rank, msg)

        # Compute the RHS.  Overwrite inputs, either the original or the copy.

        self._log_info(comm, rank, "begin RHS calculation")

        # Set our binning operator to use only our new solver flags
        self.binning.shared_flags = None
        self.binning.shared_flag_mask = 0
        self.binning.det_flags = flagname
        self.binning.det_flag_mask = 255

        # Set the binning operator to output to temporary map.  This will be
        # overwritten on each iteration of the solver.
        self.binning.binned = solver_bin

        self.template_matrix.amplitudes = solver_rhs
        rhs_calc = SolverRHS(
            name="{}_rhs".format(self.name),
            det_data=self.det_data,
            overwrite=False,
            binning=self.binning,
            template_matrix=self.template_matrix,
        )

        rhs_calc.apply(data, detectors=detectors)

        self._log_info(comm, rank, "  finished RHS calculation in", timer=timer)

        # Set up the LHS operator.

        self._log_info(comm, rank, "begin PCG solver")

        lhs_calc = SolverLHS(
            name="{}_lhs".format(self.name),
            binning=self.binning,
            template_matrix=self.template_matrix,
        )

        # If we eventually want to support an input starting guess of the
        # amplitudes, we would need to ensure that data[amplitude_key] is set
        # at this point...

        # Solve for amplitudes.
        solve(
            data,
            detectors,
            lhs_calc,
            solver_rhs,
            solver_result,
            convergence=self.convergence,
            n_iter_max=self.iter_max,
        )

        self._log_info(comm, rank, "  finished solver in", timer=timer)

        # Restore flag names and masks to binning operator, in case it is being used
        # for the final map making or for other external operations.

        self.binning.det_flags = save_det_flags
        self.binning.det_flag_mask = save_det_flag_mask
        self.binning.shared_flags = save_shared_flags
        self.binning.shared_flag_mask = save_shared_flag_mask
        self.binning.binned = save_binned
        self.binning.covariance = save_covariance

        # Now construct the noise covariance, hits, and condition number mask for the
        # final binned map.

        save_covariance = self.map_binning.covariance
        self.map_binning.covariance = cov_name

        if self.mc_mode:
            # Verify that our covariance and other products exist.
            if self.map_binning.pixel_dist not in data:
                msg = "MC mode, pixel distribution '{}' does not exist".format(
                    self.map_binning.pixel_dist
                )
                log.error(msg)
                raise RuntimeError(msg)
            if self.map_binning.covariance not in data:
                msg = "MC mode, covariance '{}' does not exist".format(
                    self.map_binning.covariance
                )
                log.error(msg)
                raise RuntimeError(msg)
            self._log_info(comm, rank, "MC mode, reusing covariance for final binning")
        else:
            self._log_info(comm, rank, "begin build of final binning covariance")

            final_cov = CovarianceAndHits(
                pixel_dist=self.map_binning.pixel_dist,
                covariance=self.map_binning.covariance,
                hits=hits_name,
                rcond=rcond_name,
                det_flags=self.map_binning.det_flags,
                det_flag_mask=self.map_binning.det_flag_mask,
                shared_flags=self.map_binning.shared_flags,
                shared_flag_mask=self.map_binning.shared_flag_mask,
                pointing=self.map_binning.pointing,
                noise_model=self.map_binning.noise_model,
                rcond_threshold=self.map_rcond_threshold,
                sync_type=self.map_binning.sync_type,
                save_pointing=self.map_binning.saved_pointing,
            )

            final_cov.apply(data, detectors=detectors)

            self._log_info(
                comm, rank, "  finished build of final covariance in", timer=timer
            )

        # Project the solved template amplitudes into timestreams and subtract
        # from the original.  Then make a binned map of the result.

        self._log_info(comm, rank, "begin final map binning")

        temp_project = "{}_temp_project".format(self.name)

        # Projecting amplitudes to a temp space
        self.template_matrix.transpose = False
        self.template_matrix.det_data = temp_project
        self.template_matrix.amplitudes = solver_result

        if self.map_binning.binned == "binned":
            # The user did not modify the default name of the output binned map.
            # Set this to something more descriptive, named after our operator
            # instance.
            self.map_binning.binned = "{}_map".format(self.name)

        # Binning the cleaned data
        self.map_binning.det_data = clean_name

        # Operator to copy the input data to the cleaned location
        copy_input = Copy(detdata=[(self.det_data, clean_name)])

        pre_pipe = None
        pre_pipe_dets = ["SINGLE"]
        if self.map_binning.saved_pointing:
            pre_pipe_dets = ["ALL"]
        if self.save_cleaned:
            # We are going to be saving a full copy of the template-subtracted data
            if self.overwrite_cleaned:
                # We are going to modify the input data in place
                sub_cleaned = Subtract(first=self.det_data, second=temp_project)
                pre_pipe = Pipeline(
                    detector_sets=pre_pipe_dets,
                    operators=[
                        self.template_matrix,
                        sub_cleaned,
                    ],
                )
            else:
                # We need to create a new full set of timestreams.  Do this now
                # all at once for all detectors.
                copy_input.apply(data, detectors=detectors)
                # Pipeline to project one detector at a time and subtract.
                sub_cleaned = Subtract(first=clean_name, second=temp_project)
                pre_pipe = Pipeline(
                    detector_sets=pre_pipe_dets,
                    operators=[
                        self.template_matrix,
                        sub_cleaned,
                    ],
                )
        else:
            # Not saving cleaned timestreams.  Use a preprocessing pipeline that
            # just projects and subtracts data one detector at a time.
            sub_cleaned = Subtract(first=clean_name, second=temp_project)
            pre_pipe = Pipeline(
                detector_sets=pre_pipe_dets,
                operators=[
                    self.template_matrix,
                    copy_input,
                    sub_cleaned,
                ],
            )

        # Do the final binning
        self.map_binning.pre_process = pre_pipe
        self.map_binning.apply(data, detectors=detectors)
        self.map_binning.pre_process = None

        self.map_binning.covariance = save_covariance

        self._log_info(comm, rank, "  finished final binning in", timer=timer)

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
