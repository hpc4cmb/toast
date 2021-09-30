# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import traitlets

import numpy as np

from ..utils import Logger

from ..mpi import MPI

from ..traits import trait_docs, Int, Unicode, Bool, Float, Instance

from ..timing import function_timer, Timer

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ..observation import default_names as obs_names

from .operator import Operator

from .pipeline import Pipeline

from .delete import Delete

from .copy import Copy

from .arithmetic import Subtract

from .scan_map import ScanMap, ScanMask

from .mapmaker_utils import CovarianceAndHits

from .mapmaker_solve import solve, SolverRHS, SolverLHS

from .memory_counter import MemoryCounter


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
        obs_names.det_data, help="Observation detdata key for the timestream data"
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

    write_map = Bool(True, help="If True, write the projected map")

    write_noiseweighted_map = Bool(
        False,
        help="If True, write the noise-weighted map",
    )

    write_hits = Bool(True, help="If True, write the hits map")

    write_cov = Bool(True, help="If True, write the white noise covariance matrices.")

    write_invcov = Bool(
        False,
        help="If True, write the inverse white noise covariance matrices.",
    )

    write_rcond = Bool(True, help="If True, write the reciprocal condition numbers.")

    keep_solver_products = Bool(
        False, help="If True, keep the map domain solver products in data"
    )

    keep_final_products = Bool(
        False, help="If True, keep the map domain products in data after write"
    )

    mc_mode = Bool(False, help="If True, re-use solver flags, sparse covariances, etc")

    mc_index = Int(None, allow_none=True, help="The Monte-Carlo index")

    save_cleaned = Bool(
        False, help="If True, save the template-subtracted detector timestreams"
    )

    overwrite_cleaned = Bool(
        False, help="If True and save_cleaned is True, overwrite the input data"
    )

    reset_pix_dist = Bool(
        False,
        help="Clear any existing pixel distribution.  Useful when applying"
        "repeatedly to different data objects.",
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    report_memory = Bool(False, help="Report memory throughout the execution")

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
                "pixel_pointing",
                "stokes_weights",
                "binned",
                "covariance",
                "det_flags",
                "det_flag_mask",
                "shared_flags",
                "shared_flag_mask",
                "noise_model",
                "full_pointing",
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
                "pixel_pointing",
                "stokes_weights",
                "binned",
                "covariance",
                "det_flags",
                "det_flag_mask",
                "shared_flags",
                "shared_flag_mask",
                "noise_model",
                "full_pointing",
                "sync_type",
            ]:
                if not bin.has_trait(trt):
                    msg = "map_binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        log_prefix = "MapMaker"

        memreport = MemoryCounter()
        if not self.report_memory:
            memreport.enabled = False

        memreport.prefix = "Start of mapmaking"
        memreport.apply(data)

        # The global communicator we are using (or None)
        comm = data.comm.comm_world
        rank = data.comm.world_rank

        # Check map binning
        map_binning = self.map_binning
        if self.map_binning is None or not self.map_binning.enabled:
            # Use the same binning used in the solver.
            map_binning = self.binning

        # Optionally destroy existing pixel distributions (useful if calling
        # repeatedly with different data objects)
        if self.reset_pix_dist:
            if self.binning.pixel_dist in data:
                del data[self.binning.pixel_dist]
            if self.map_binning.pixel_dist in data:
                del data[self.map_binning.pixel_dist]

            memreport.prefix = "After resetting pixel distribution"
            memreport.apply(data)

        # We use the input binning operator to define the flags that the user has
        # specified.  We will save the name / bit mask for these and restore them later.
        # Then we will use the binning operator with our solver flags.  These input
        # flags are combined to the first bit (== 1) of the solver flags.

        save_det_flags = self.binning.det_flags
        save_det_flag_mask = self.binning.det_flag_mask
        save_shared_flags = self.binning.shared_flags
        save_shared_flag_mask = self.binning.shared_flag_mask

        # Output data products, prefixed with the name of the operator and optionally
        # the MC index.

        mc_root = None
        if self.mc_mode and self.mc_index is not None:
            mc_root = "{}_{:05d}".format(self.name, mc_index)
        else:
            mc_root = self.name

        self.solver_hits_name = "{}_solve_hits".format(self.name)
        self.solver_cov_name = "{}_solve_cov".format(self.name)
        self.solver_rcond_name = "{}_solve_rcond".format(self.name)
        self.solver_rcond_mask_name = "{}_solve_rcond_mask".format(self.name)
        self.solver_result = "{}_solve_amplitudes".format(mc_root)
        self.solver_rhs = "{}_solve_rhs".format(mc_root)
        self.solver_bin = "{}_solve_bin".format(mc_root)

        self.hits_name = "{}_hits".format(self.name)
        self.cov_name = "{}_cov".format(self.name)
        self.invcov_name = "{}_invcov".format(self.name)
        self.rcond_name = "{}_rcond".format(self.name)
        self.flag_name = "{}_flags".format(self.name)

        self.clean_name = "{}_cleaned".format(mc_root)
        self.map_name = "{}_map".format(mc_root)
        self.noiseweighted_map_name = "{}_noiseweighted_map".format(mc_root)

        timer.start()

        n_enabled_templates = 0
        if self.template_matrix is not None:
            for template in self.template_matrix.templates:
                if template.enabled:
                    n_enabled_templates += 1

        if n_enabled_templates != 0:
            # We are solving for template amplitudes

            self.binning.covariance = self.solver_cov_name

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
                    if self.flag_name not in ob.detdata:
                        msg = "In MC mode, flags missing for observation {}".format(
                            ob.name
                        )
                        log.error(msg)
                        raise RuntimeError(msg)
                    for d in dets:
                        if d not in ob.detdata[self.flag_name].detectors:
                            msg = "In MC mode, flags missing for observation {}, det {}".format(
                                ob.name, d
                            )
                            log.error(msg)
                            raise RuntimeError(msg)
                log.info_rank(
                    f"{log_prefix} MC mode, reusing flags for solver", comm=comm
                )
            else:
                log.info_rank(
                    f"{log_prefix} begin building flags for solver", comm=comm
                )

                # Use the same data view as the pointing operator in binning
                solve_view = self.binning.pixel_pointing.view

                for ob in data.obs:
                    # Get the detectors we are using for this observation
                    dets = ob.select_local_detectors(detectors)
                    if len(dets) == 0:
                        # Nothing to do for this observation
                        continue
                    # Create the new solver flags
                    ob.detdata.ensure(
                        self.flag_name, dtype=np.uint8, detectors=detectors
                    )
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
                                views.shared[save_shared_flags][vw]
                                & save_shared_flag_mask
                                > 0,
                                1,
                                0,
                            )
                        for d in dets:
                            views.detdata[self.flag_name][vw][d, :] = starting_flags
                            if save_det_flags is not None:
                                views.detdata[self.flag_name][vw][d, :] |= np.where(
                                    views.detdata[save_det_flags][vw][d]
                                    & save_det_flag_mask
                                    > 0,
                                    1,
                                    0,
                                ).astype(views.detdata[self.flag_name][vw].dtype)

                # Now scan any input mask to this same flag field.  We use the second
                # bit (== 2) for these mask flags.  For the input mask bit we check the
                # first bit of the pixel values.  This is noted in the help string for
                # the mask trait.  Note that we explicitly expand the pointing once
                # here and do not save it.  Even if we are eventually saving the
                # pointing, we want to do that later when building the covariance and
                # the pixel distribution.

                # Use the same pointing operator as the binning
                scan_pointing = self.binning.pixel_pointing

                scanner = ScanMask(
                    det_flags=self.flag_name,
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

                log.info_rank(
                    f"{log_prefix}  finished flag building in",
                    comm=comm,
                    timer=timer,
                )

                memreport.prefix = "After building flags"
                memreport.apply(data)

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

                log.info_rank(
                    f"{log_prefix} MC mode, reusing covariance for solver",
                    comm=comm,
                )
            else:
                log.info_rank(
                    f"{log_prefix} begin build of solver covariance",
                    comm=comm,
                )

                solver_cov = CovarianceAndHits(
                    pixel_dist=self.binning.pixel_dist,
                    covariance=self.solver_cov_name,
                    hits=self.solver_hits_name,
                    rcond=self.solver_rcond_name,
                    det_flags=self.flag_name,
                    det_flag_mask=255,
                    pixel_pointing=self.binning.pixel_pointing,
                    stokes_weights=self.binning.stokes_weights,
                    noise_model=self.binning.noise_model,
                    rcond_threshold=self.solve_rcond_threshold,
                    sync_type=self.binning.sync_type,
                    save_pointing=self.binning.full_pointing,
                )

                solver_cov.apply(data, detectors=detectors)

                memreport.prefix = "After constructing covariance and hits"
                memreport.apply(data)

                data[self.solver_rcond_mask_name] = PixelData(
                    data[self.binning.pixel_dist], dtype=np.uint8, n_value=1
                )
                data[self.solver_rcond_mask_name].raw[
                    data[self.solver_rcond_name].raw.array()
                    < self.solve_rcond_threshold
                ] = 1

                memreport.prefix = "After constructing rcond mask"
                memreport.apply(data)

                # Re-use our mask scanning pipeline, setting third bit (== 4)
                scanner.det_flags_value = 4
                scanner.mask_key = self.solver_rcond_mask_name
                scan_pipe.apply(data, detectors=detectors)

                log.info_rank(
                    f"{log_prefix}  finished build of solver covariance in",
                    comm=comm,
                    timer=timer,
                )

                local_total = 0
                local_cut = 0
                for ob in data.obs:
                    # Get the detectors we are using for this observation
                    dets = ob.select_local_detectors(detectors)
                    if len(dets) == 0:
                        # Nothing to do for this observation
                        continue
                    for vw in ob.view[solve_view].detdata[self.flag_name]:
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
                log.info_rank(
                    f"{log_prefix} {msg}",
                    comm=comm,
                )

            # Compute the RHS.  Overwrite inputs, either the original or the copy.

            log.info_rank(
                f"{log_prefix} begin RHS calculation",
                comm=comm,
            )

            # Set our binning operator to use only our new solver flags
            self.binning.shared_flags = None
            self.binning.shared_flag_mask = 0
            self.binning.det_flags = self.flag_name
            self.binning.det_flag_mask = 255

            # Set the binning operator to output to temporary map.  This will be
            # overwritten on each iteration of the solver.
            self.binning.binned = self.solver_bin
            self.binning.covariance = self.solver_cov_name

            self.template_matrix.amplitudes = self.solver_rhs
            rhs_calc = SolverRHS(
                name="{}_rhs".format(self.name),
                det_data=self.det_data,
                overwrite=False,
                binning=self.binning,
                template_matrix=self.template_matrix,
            )

            rhs_calc.apply(data, detectors=detectors)

            log.info_rank(
                f"{log_prefix}  finished RHS calculation in",
                comm=comm,
                timer=timer,
            )

            memreport.prefix = "After constructing RHS"
            memreport.apply(data)

            # Set up the LHS operator.

            log.info_rank(
                f"{log_prefix} begin PCG solver",
                comm=comm,
            )

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
                self.solver_rhs,
                self.solver_result,
                convergence=self.convergence,
                n_iter_max=self.iter_max,
            )

            log.info_rank(
                f"{log_prefix}  finished solver in",
                comm=comm,
                timer=timer,
            )

            memreport.prefix = "After solving for amplitudes"
            memreport.apply(data)

        # Delete our solver products to save memory
        if not self.mc_mode and not self.keep_solver_products:
            for prod in [
                self.solver_hits_name,
                self.solver_cov_name,
                self.solver_rcond_name,
                self.solver_rcond_mask_name,
                self.solver_rhs,
                self.solver_bin,
            ]:
                if prod in data:
                    data[prod].clear()
                    del data[prod]

            memreport.prefix = "After deleting solver products"
            memreport.apply(data)

        # Restore flag names and masks to binning operator, in case it is being used
        # for the final map making or for other external operations.

        self.binning.det_flags = save_det_flags
        self.binning.det_flag_mask = save_det_flag_mask
        self.binning.shared_flags = save_shared_flags
        self.binning.shared_flag_mask = save_shared_flag_mask

        # Now construct the noise covariance, hits, and condition number mask for the
        # final binned map.

        map_binning.covariance = self.cov_name

        if self.mc_mode:
            # Verify that our covariance and other products exist.
            if map_binning.pixel_dist not in data:
                msg = "MC mode, pixel distribution '{}' does not exist".format(
                    map_binning.pixel_dist
                )
                log.error(msg)
                raise RuntimeError(msg)
            if map_binning.covariance not in data:
                msg = "MC mode, covariance '{}' does not exist".format(
                    map_binning.covariance
                )
                log.error(msg)
                raise RuntimeError(msg)
            log.info_rank(
                f"{log_prefix} MC mode, reusing covariance for final binning",
                comm=comm,
            )
        else:
            log.info_rank(
                f"{log_prefix} begin build of final binning covariance",
                comm=comm,
            )

            final_cov = CovarianceAndHits(
                pixel_dist=map_binning.pixel_dist,
                covariance=map_binning.covariance,
                inverse_covariance=self.invcov_name,
                hits=self.hits_name,
                rcond=self.rcond_name,
                det_flags=map_binning.det_flags,
                det_flag_mask=map_binning.det_flag_mask,
                shared_flags=map_binning.shared_flags,
                shared_flag_mask=map_binning.shared_flag_mask,
                pixel_pointing=map_binning.pixel_pointing,
                stokes_weights=map_binning.stokes_weights,
                noise_model=map_binning.noise_model,
                rcond_threshold=self.map_rcond_threshold,
                sync_type=map_binning.sync_type,
                save_pointing=map_binning.full_pointing,
            )

            final_cov.apply(data, detectors=detectors)

            log.info_rank(
                f"{log_prefix}  finished build of final covariance in",
                comm=comm,
                timer=timer,
            )

            memreport.prefix = "After constructing final covariance and hits"
            memreport.apply(data)

        # Project the solved template amplitudes into timestreams and subtract
        # from the original.  Then make a binned map of the result.

        log.info_rank(
            f"{log_prefix} begin final map binning",
            comm=comm,
        )

        pre_pipe = None
        if self.write_noiseweighted_map:
            map_binning.noiseweighted = self.noiseweighted_map_name
        map_binning.binned = self.map_name

        if n_enabled_templates != 0:
            # We have some templates to subtract
            temp_project = "{}_temp_project".format(self.name)

            # Projecting amplitudes to a temp space
            self.template_matrix.transpose = False
            self.template_matrix.det_data = temp_project
            self.template_matrix.amplitudes = self.solver_result

            # Binning the cleaned data
            map_binning.det_data = self.clean_name

            # Operator to copy the input data to the cleaned location
            copy_input = Copy(detdata=[(self.det_data, self.clean_name)])

            pre_pipe_dets = ["SINGLE"]
            if map_binning.full_pointing:
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
                    sub_cleaned = Subtract(first=self.clean_name, second=temp_project)
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
                sub_cleaned = Subtract(first=self.clean_name, second=temp_project)
                pre_pipe = Pipeline(
                    detector_sets=pre_pipe_dets,
                    operators=[
                        self.template_matrix,
                        copy_input,
                        sub_cleaned,
                    ],
                )
        else:
            # We have no templates.  This means we are just making a binned map of the
            # input timestreams.
            map_binning.det_data = self.det_data

        # Do the final binning
        map_binning.pre_process = pre_pipe
        map_binning.apply(data, detectors=detectors)
        map_binning.pre_process = None

        log.info_rank(
            f"{log_prefix}  finished final binning in",
            comm=comm,
            timer=timer,
        )

        memreport.prefix = "After binning final map"
        memreport.apply(data)

        # Write and delete the outputs

        # FIXME:  This all assumes the pointing operator is an instance of the
        # PointingHealpix class.  We need to generalize distributed pixel data
        # formats and associate them with the pointing operator.

        write_del = list()
        write_del.append((self.hits_name, self.write_hits))
        write_del.append((self.rcond_name, self.write_rcond))
        write_del.append((self.noiseweighted_map_name, self.write_noiseweighted_map))
        write_del.append((self.map_name, self.write_map))
        write_del.append((self.invcov_name, self.write_invcov))
        write_del.append((self.cov_name, self.write_cov))
        for prod_key, prod_write in write_del:
            if prod_write:
                fname = os.path.join(self.output_dir, "{}.fits".format(prod_key))
                write_healpix_fits(
                    data[prod_key],
                    fname,
                    nest=map_binning.pixel_pointing.nest,
                    report_memory=self.report_memory,
                )
                log.info_rank(f"Wrote {fname} in", comm=comm, timer=timer)
            if not self.keep_final_products:
                if prod_key in data:
                    data[prod_key].clear()
                    del data[prod_key]

            memreport.prefix = f"After writing/deleting {prod_key}"
            memreport.apply(data)

        log.info_rank(
            f"{log_prefix}  finished output write in",
            comm=comm,
            timer=timer,
        )

        memreport.prefix = "End of mapmaking"
        memreport.apply(data)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator requires everything that its sub-operators needs.
        req = self.binning.requires()
        if self.template_matrix is not None:
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
