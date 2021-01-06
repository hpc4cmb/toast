# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Float, Instance

from ..timing import function_timer, Timer

from ..pixels import PixelDistribution, PixelData

from .operator import Operator

from .pipeline import Pipeline

from .clear import Clear

from .copy import Copy

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

    overwrite = Bool(
        False, help="Overwrite the input detector data for use as scratch space"
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
                "save_pointing",
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
                "save_pointing",
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

        # For computing the RHS and also for each iteration of the LHS we will need
        # a full detector-data sized buffer for use as scratch space.  We can either
        # destroy the input data to save memory (useful if this is the last operator
        # processing the data) or we can create a temporary set of timestreams.

        timer.start()

        copy_det = None
        clear_temp = None
        detdata_name = self.det_data

        if not self.overwrite:
            self._log_info(comm, rank, "overwrite is False, making data copy")

            # Use a temporary detdata named after this operator
            detdata_name = "{}_signal".format(self.name)
            # Copy the original data into place, and then use this copy destructively.
            copy_det = Copy(
                detdata=[
                    (self.det_data, detdata_name),
                ]
            )
            copy_det.apply(data, detectors=detectors)
            self._log_info(comm, rank, "  data copy finished in", timer=timer)

        # Flagging.  We create a new set of data flags for the solver that includes:
        #   - one bit for a bitwise OR of all detector / shared flags
        #   - one bit for any pixel mask, projected to TOD
        #   - one bit for any poorly conditioned pixels, projected to TOD

        # We use the input binning operator to define the flags that the user has
        # specified.  We will save the name / bit mask for these and restore them later.
        # Then we will use the binning operator with our solver flags.  These input
        # flags are combined to the first bit (== 1) of the solver flags.

        self._log_info(comm, rank, "begin building flags for solver")

        flagname = "{}_flags".format(self.name)

        save_det_flags = self.binning.det_flags
        save_det_flag_mask = self.binning.det_flag_mask
        save_shared_flags = self.binning.shared_flags
        save_shared_flag_mask = self.binning.shared_flag_mask

        # Use the same data view as the pointing operator in binning
        solve_view = self.binning.pointing.view

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            # Create the new solver flags
            ob.detdata.create(flagname, dtype=np.uint8, detectors=detectors)
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
                        views.shared[save_shared_flags][vw] & save_shared_flag_mask > 0,
                        1,
                        0,
                    )
                for d in dets:
                    views.detdata[flagname][vw][d, :] = starting_flags
                    if save_det_flags is not None:
                        views.detdata[flagname][vw][d, :] |= np.where(
                            views.detdata[save_det_flags][vw][d] & save_det_flag_mask
                            > 0,
                            1,
                            0,
                        )

        # Now scan any input mask to this same flag field.  We use the second bit (== 2)
        # for these mask flags.  For the input mask bit we check the first bit of the
        # pixel values.  This is noted in the help string for the mask trait.

        # Use the same pointing operator as the binning
        scan_pointing = self.binning.pointing

        # Set up operator for optional clearing of the pointing matrices
        clear_pointing = Clear(detdata=[scan_pointing.pixels, scan_pointing.weights])

        scanner = ScanMask(
            det_flags=flagname,
            pixels=scan_pointing.pixels,
            mask_bits=1,
        )

        scanner.det_flags_value = 2
        scanner.mask_key = self.mask

        scan_pipe = None
        if self.binning.save_pointing:
            # Process all detectors at once
            scan_pipe = Pipeline(
                detector_sets=["ALL"], operators=[scan_pointing, scanner]
            )
        else:
            # Process one detector at a time and clear pointing after each one.
            scan_pipe = Pipeline(
                detector_sets=["SINGLE"],
                operators=[scan_pointing, scanner, clear_pointing],
            )

        if self.mask is not None:
            # We actually have an input mask. Scan it.
            scan_pipe.apply(data, detectors=detectors)

        self._log_info(comm, rank, "  finished flag building in", timer=timer)

        # Now construct the noise covariance, hits, and condition number mask

        self._log_info(comm, rank, "begin build of solver covariance")

        solver_hits_name = "{}_solve_hits".format(self.name)
        solver_rcond_name = "{}_solve_rcond".format(self.name)
        solver_rcond_mask_name = "{}_solve_rcond_mask".format(self.name)

        solver_cov = CovarianceAndHits(
            pixel_dist=self.binning.pixel_dist,
            covariance=self.binning.covariance,
            hits=solver_hits_name,
            rcond=solver_rcond_name,
            view=self.binning.pointing.view,
            det_flags=flagname,
            det_flag_mask=255,
            pointing=self.binning.pointing,
            noise_model=self.binning.noise_model,
            rcond_threshold=self.solve_rcond_threshold,
            sync_type=self.binning.sync_type,
            save_pointing=self.binning.save_pointing,
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

        # Compute the RHS.  Overwrite inputs, either the original or the copy.

        self._log_info(comm, rank, "begin RHS calculation")

        # Set our binning operator to use only our new solver flags
        self.binning.shared_flags = None
        self.binning.shared_flag_mask = 0
        self.binning.det_flags = flagname
        self.binning.det_flag_mask = 255

        rhs_amplitude_key = "{}_amplitudes_rhs".format(self.name)

        self.template_matrix.amplitudes = rhs_amplitude_key
        rhs_calc = SolverRHS(
            det_data=detdata_name,
            overwrite=True,
            binning=self.binning,
            template_matrix=self.template_matrix,
        )
        rhs_calc.apply(data, detectors=detectors)

        self._log_info(comm, rank, "  finished RHS calculation in", timer=timer)

        # Set up the LHS operator.  Use either the original timestreams or the copy
        # as temp space.

        self._log_info(comm, rank, "begin PCG solver")

        amplitude_key = "{}_amplitudes".format(self.name)
        self.template_matrix.amplitudes = amplitude_key

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
            data[rhs_amplitude_key],
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

        self._log_info(
            comm, rank, "begin projection of final amplitudes to timestreams"
        )

        # Reset our timestreams to zero
        for ob in data.obs:
            ob.detdata[detdata_name][:] = 0.0

        # Project our solved amplitudes into timestreams.  We output to either the
        # input det_data or our temp space.

        self.template_matrix.transpose = False
        self.template_matrix.apply(data, detectors=detectors)

        self._log_info(comm, rank, "  finished amplitude projection in", timer=timer)

        # Now construct the noise covariance, hits, and condition number mask for the
        # final binned map.

        self._log_info(comm, rank, "begin build of final binning covariance")

        hits_name = "{}_hits".format(self.name)
        rcond_name = "{}_rcond".format(self.name)

        final_cov = CovarianceAndHits(
            pixel_dist=self.map_binning.pixel_dist,
            covariance=self.map_binning.covariance,
            hits=hits_name,
            rcond=rcond_name,
            view=self.map_binning.pointing.view,
            det_flags=self.map_binning.det_flags,
            det_flag_mask=self.map_binning.det_flag_mask,
            shared_flags=self.map_binning.shared_flags,
            shared_flag_mask=self.map_binning.shared_flag_mask,
            pointing=self.map_binning.pointing,
            noise_model=self.map_binning.noise_model,
            rcond_threshold=self.map_rcond_threshold,
            sync_type=self.map_binning.sync_type,
            save_pointing=self.map_binning.save_pointing,
        )

        final_cov.apply(data, detectors=detectors)

        self._log_info(
            comm, rank, "  finished build of final covariance in", timer=timer
        )

        # Make a binned map of these template-subtracted timestreams

        self._log_info(comm, rank, "begin final map binning")

        self.map_binning.det_data = detdata_name
        self.map_binning.apply(data, detectors=detectors)

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
