# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets
from astropy import units as u

from ..accelerator import ImplementationType
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..pixels_io_healpix import write_healpix_fits, write_healpix_hdf5
from ..pixels_io_wcs import write_wcs_fits
from ..templates import AmplitudesMap, Template
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, List, Unicode, Unit, trait_docs
from ..utils import Logger
from .arithmetic import Combine
from .copy import Copy
from .mapmaker_solve import SolverLHS, SolverRHS, solve
from .mapmaker_utils import CovarianceAndHits
from .memory_counter import MemoryCounter
from .operator import Operator
from .pipeline import Pipeline
from .scan_map import ScanMask


@trait_docs
class TemplateMatrix(Operator):
    """Operator for projecting or accumulating template amplitudes."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    templates = List([], help="This should be a list of Template instances")

    amplitudes = Unicode(None, allow_none=True, help="Data key for template amplitudes")

    transpose = Bool(False, help="If True, apply the transpose.")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    @traitlets.validate("templates")
    def _check_templates(self, proposal):
        temps = proposal["value"]
        for tp in temps:
            if not isinstance(tp, Template):
                raise traitlets.TraitError(
                    "templates must be a list of Template instances or None"
                )
        return temps

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False

    def reset(self):
        """Reset templates to allow re-initialization on a new Data object."""
        self._initialized = False

    def duplicate(self):
        """Make a shallow copy which contains the same list of templates.

        This is useful when we want to use both a template matrix and its transpose
        in the same pipeline.

        Returns:
            (TemplateMatrix):  A new instance with the same templates.

        """
        ret = TemplateMatrix(
            API=self.API,
            templates=self.templates,
            amplitudes=self.amplitudes,
            transpose=self.transpose,
            view=self.view,
            det_data=self.det_data,
            det_data_units=self.det_data_units,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
        )
        ret._initialized = self._initialized
        return ret

    def apply_precond(self, amps_in, amps_out, use_accel=None, **kwargs):
        """Apply the preconditioner from all templates to the amplitudes.

        This can only be called after the operator has been used at least once so that
        the templates are initialized.

        Args:
            amps_in (AmplitudesMap):  The input amplitudes.
            amps_out (AmplitudesMap):  The output amplitudes, modified in place.

        Returns:
            None

        """
        if not self._initialized:
            raise RuntimeError(
                "You must call exec() once before applying preconditioners"
            )
        for tmpl in self.templates:
            if tmpl.enabled:
                tmpl.apply_precond(
                    amps_in[tmpl.name],
                    amps_out[tmpl.name],
                    use_accel=use_accel,
                    **kwargs,
                )

    def add_prior(self, amps_in, amps_out, use_accel=None, **kwargs):
        """Apply the noise prior from all templates to the amplitudes.

        This can only be called after the operator has been used at least once so that
        the templates are initialized.

        Args:
            amps_in (AmplitudesMap):  The input amplitudes.
            amps_out (AmplitudesMap):  The output amplitudes, modified in place.

        Returns:
            None

        """
        if not self._initialized:
            raise RuntimeError(
                "You must call exec() once before applying the noise prior"
            )
        for tmpl in self.templates:
            if tmpl.enabled:
                tmpl.add_prior(
                    amps_in[tmpl.name],
                    amps_out[tmpl.name],
                    use_accel=use_accel,
                    **kwargs,
                )

    @property
    def n_enabled_templates(self):
        n_enabled_templates = 0
        for template in self.templates:
            if template.enabled:
                n_enabled_templates += 1
        return n_enabled_templates

    def reset_templates(self):
        """Mark templates to be re-initialized on next call to exec()."""
        self._initialized = False

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check that amplitudes is set
        if self.amplitudes is None:
            raise RuntimeError(
                "You must set the amplitudes trait before calling exec()"
            )

        if len(self.templates) == 0:
            log.debug_rank(
                "No templates in TemplateMatrix, nothing to do",
                comm=data.comm.comm_world,
            )
            return

        # Check that accelerator switch makes sense for this operator
        if use_accel is None:
            use_accel = False
        if use_accel and not self.supports_accel():
            msg = "Template matrix called with use_accel=True, "
            msg += "but does not support accelerators"
            raise RuntimeError(msg)

        # On the first call, we initialize all templates using the Data instance and
        # the fixed options for view, flagging, etc.
        if not self._initialized:
            if use_accel:
                # fail when a user tries to run the initialization pipeline on GPU
                raise RuntimeError(
                    "You cannot currently initialize templates on device (please disable accel for this operator/pipeline)."
                )
            for tmpl in self.templates:
                if tmpl.view is None:
                    tmpl.view = self.view
                tmpl.det_data_units = self.det_data_units
                tmpl.det_flags = self.det_flags
                tmpl.det_flag_mask = self.det_flag_mask
                # This next line will trigger calculation of the number
                # of amplitudes within each template.
                tmpl.data = data
            self._initialized = True

        # Set the data we are using for this execution
        for tmpl in self.templates:
            tmpl.det_data = self.det_data

        # We loop over detectors.  Internally, each template loops over observations
        # and ignores observations where the detector does not exist.
        all_dets = data.all_local_detectors(selection=detectors)

        if self.transpose:
            # Check that the incoming detector data in all observations has the correct
            # units.
            input_units = 1.0 / self.det_data_units
            for ob in data.obs:
                if ob.detdata[self.det_data].units != input_units:
                    msg = f"obs {ob.name} detdata {self.det_data}"
                    msg += f" does not have units of {input_units}"
                    msg += f" before template matrix projection"
                    log.error(msg)
                    raise RuntimeError(msg)

            if self.amplitudes not in data:
                # The output template amplitudes do not yet exist.
                # Create these with all zero values.
                data[self.amplitudes] = AmplitudesMap()
                for tmpl in self.templates:
                    if tmpl.enabled:
                        data[self.amplitudes][tmpl.name] = tmpl.zeros()
                if use_accel:
                    # We are running on the accelerator, so our output data must exist
                    # on the device and will be used there.
                    data[self.amplitudes].accel_create(self.name, zero_out=True)
                    data[self.amplitudes].accel_used(True)
            elif use_accel and not data[self.amplitudes].accel_exists():
                # The output template amplitudes exist on host, but are not yet
                # staged to the accelerator.
                data[self.amplitudes].accel_create(self.name)
                data[self.amplitudes].accel_update_device()

            for d in all_dets:
                for tmpl in self.templates:
                    if tmpl.enabled:
                        log.verbose(
                            f"TemplateMatrix {d} project_signal {tmpl.name} (use_accel={use_accel})"
                        )
                        tmpl.project_signal(
                            d,
                            data[self.amplitudes][tmpl.name],
                            use_accel=use_accel,
                            **kwargs,
                        )
        else:
            if self.amplitudes not in data:
                msg = f"Template amplitudes '{self.amplitudes}' do not exist in data"
                log.error(msg)
                raise RuntimeError(msg)

            # Ensure that our output detector data exists in each observation
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(selection=detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                exists = ob.detdata.ensure(
                    self.det_data,
                    detectors=dets,
                    accel=use_accel,
                    create_units=self.det_data_units,
                )
                if exists:
                    # We need to clear our detector TOD before projecting amplitudes
                    # into timestreams.  Note:  in the accelerator case, the reset call
                    # will clear all detectors, not just the current list.  This is
                    # wasteful if det_data has a very large buffer that has been
                    # restricted to be used for a smaller number of detectors.  We
                    # should deal with that corner case eventually.  If the data was
                    # created, then it was already zeroed out.
                    ob.detdata[self.det_data].reset(dets=dets)

                ob.detdata[self.det_data].update_units(self.det_data_units)

            for d in all_dets:
                for tmpl in self.templates:
                    log.verbose(
                        f"TemplateMatrix {d} add to signal {tmpl.name} (use_accel={use_accel})"
                    )
                    tmpl.add_to_signal(
                        d,
                        data[self.amplitudes][tmpl.name],
                        use_accel=use_accel,
                        **kwargs,
                    )
        return

    def _finalize(self, data, use_accel=None, **kwargs):
        if self.transpose:
            # move amplitudes to host as sync is CPU only
            if use_accel:
                data[self.amplitudes].accel_update_host()
            # Synchronize the result
            for tmpl in self.templates:
                if tmpl.enabled:
                    data[self.amplitudes][tmpl.name].sync()
            # move amplitudes back to GPU as it is NOT finalize's job to move data to host
            if use_accel:
                data[self.amplitudes].accel_update_device()
        # Set the internal initialization to False, so that we are ready to process
        # completely new data sets.
        return

    def _requires(self):
        req = dict()
        req["detdata"] = [self.det_data]
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.transpose:
            if self.det_flags is not None:
                req["detdata"].append(self.det_flags)
        else:
            req["global"] = [self.amplitudes]
        return req

    def _provides(self):
        prov = dict()
        if self.transpose:
            prov["global"] = [self.amplitudes]
        else:
            prov["detdata"] = [self.det_data]
        return prov

    def _implementations(self):
        """
        Find implementations supported by all the templates
        """
        implementations = {
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        }
        for tmpl in self.templates:
            implementations.intersection_update(tmpl.implementations())
        return list(implementations)

    def _supports_accel(self):
        """
        Returns True if all the templates are GPU compatible.
        """
        for tmpl in self.templates:
            if not tmpl.supports_accel():
                log = Logger.get()
                msg = f"{self} does not support accel because of '{tmpl.name}'"
                log.debug(msg)
                return False
        return True


@trait_docs
class SolveAmplitudes(Operator):
    """Solve for template amplitudes.

    This operator solves for a maximum likelihood set of template amplitudes
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

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    amplitudes = Unicode(None, allow_none=True, help="Data key for output amplitudes")

    convergence = Float(1.0e-12, help="Relative convergence limit")

    iter_min = Int(3, help="Minimum number of iterations")

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

    keep_solver_products = Bool(
        False, help="If True, keep the map domain solver products in data"
    )

    write_solver_products = Bool(False, help="If True, write out solver products")

    write_hdf5 = Bool(
        False, help="If True, outputs are in HDF5 rather than FITS format."
    )

    write_hdf5_serial = Bool(
        False, help="If True, force serial HDF5 write of output maps."
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    mc_mode = Bool(False, help="If True, re-use solver flags, sparse covariances, etc")

    mc_index = Int(None, allow_none=True, help="The Monte-Carlo index")

    reset_pix_dist = Bool(
        False,
        help="Clear any existing pixel distribution.  Useful when applying "
        "repeatedly to different data objects.",
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        log_prefix = "SolveAmplitudes"

        # Check if we have any templates
        if (
            self.template_matrix is None
            or self.template_matrix.n_enabled_templates == 0
        ):
            return

        memreport = MemoryCounter()
        if not self.report_memory:
            memreport.enabled = False

        memreport.prefix = "Start of amplitude solve"
        memreport.apply(data)

        # The global communicator we are using (or None)
        comm = data.comm.comm_world
        rank = data.comm.world_rank

        # Optionally destroy existing pixel distributions (useful if calling
        # repeatedly with different data objects)
        if self.reset_pix_dist:
            if self.binning.pixel_dist in data:
                del data[self.binning.pixel_dist]

            memreport.prefix = "After resetting pixel distribution"
            memreport.apply(data)

        # Get the units used across the distributed data for our desired
        # input detector data
        det_data_units = data.detector_units(self.det_data)

        # We use the input binning operator to define the flags that the user has
        # specified.  We will save the name / bit mask for these and restore them later.
        # Then we will use the binning operator with our solver flags.  These input
        # flags are combined to the first bit (== 1) of the solver flags.

        save_det_flags = self.binning.det_flags
        save_det_flag_mask = self.binning.det_flag_mask
        save_shared_flags = self.binning.shared_flags
        save_shared_flag_mask = self.binning.shared_flag_mask
        save_binned = self.binning.binned
        save_covariance = self.binning.covariance

        save_tmpl_flags = self.template_matrix.det_flags
        save_tmpl_mask = self.template_matrix.det_flag_mask

        # Output data products, prefixed with the name of the operator and optionally
        # the MC index.

        mc_root = None
        if self.mc_mode and self.mc_index is not None:
            mc_root = "{}_{:05d}".format(self.name, self.mc_index)
        else:
            mc_root = self.name

        self.solver_flags = "{}_solve_flags".format(self.name)
        self.solver_hits_name = "{}_solve_hits".format(self.name)
        self.solver_cov_name = "{}_solve_cov".format(self.name)
        self.solver_rcond_name = "{}_solve_rcond".format(self.name)
        self.solver_rcond_mask_name = "{}_solve_rcond_mask".format(self.name)
        self.solver_rhs = "{}_solve_rhs".format(mc_root)
        self.solver_bin = "{}_solve_bin".format(mc_root)

        if self.amplitudes is None:
            self.amplitudes = "{}_solve_amplitudes".format(mc_root)

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
                if self.solver_flags not in ob.detdata:
                    msg = "In MC mode, solver flags missing for observation {}".format(
                        ob.name
                    )
                    log.error(msg)
                    raise RuntimeError(msg)
                det_check = set(ob.detdata[self.solver_flags].detectors)
                for d in dets:
                    if d not in det_check:
                        msg = "In MC mode, solver flags missing for observation {}, det {}".format(
                            ob.name, d
                        )
                        log.error(msg)
                        raise RuntimeError(msg)
            log.info_rank(f"{log_prefix} MC mode, reusing flags for solver", comm=comm)
        else:
            log.info_rank(f"{log_prefix} begin building flags for solver", comm=comm)

            # Use the same data view as the pointing operator in binning
            solve_view = self.binning.pixel_pointing.view

            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                # Create the new solver flags
                exists = ob.detdata.ensure(
                    self.solver_flags, dtype=np.uint8, detectors=detectors
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
                            (
                                views.shared[save_shared_flags][vw]
                                & save_shared_flag_mask
                            )
                            > 0,
                            1,
                            0,
                        )
                    for d in dets:
                        views.detdata[self.solver_flags][vw][d, :] = starting_flags
                        if save_det_flags is not None:
                            views.detdata[self.solver_flags][vw][d, :] |= np.where(
                                (
                                    views.detdata[save_det_flags][vw][d]
                                    & save_det_flag_mask
                                )
                                > 0,
                                1,
                                0,
                            ).astype(views.detdata[self.solver_flags][vw].dtype)

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
                det_flags=self.solver_flags,
                pixels=scan_pointing.pixels,
                view=solve_view,
                # mask_bits=1,
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
                msg = f"MC mode, pixel distribution '{self.binning.pixel_dist}' does not exist"
                log.error(msg)
                raise RuntimeError(msg)
            if self.solver_cov_name not in data:
                msg = f"MC mode, covariance '{self.solver_cov_name}' does not exist"
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
                det_data_units=det_data_units,
                det_flags=self.solver_flags,
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
            n_bad = np.count_nonzero(
                data[self.solver_rcond_name].data < self.solve_rcond_threshold
            )
            n_good = data[self.solver_rcond_name].data.size - n_bad
            data[self.solver_rcond_mask_name].data[
                data[self.solver_rcond_name].data < self.solve_rcond_threshold
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
                for vw in ob.view[solve_view].detdata[self.solver_flags]:
                    for d in dets:
                        local_total += len(vw[d])
                        local_cut += np.count_nonzero(vw[d])
            total = None
            cut = None
            msg = None
            if comm is None:
                total = local_total
                cut = local_cut
                msg = "Solver flags cut {} / {} = {:0.2f}% of samples".format(
                    cut, total, 100.0 * (cut / total)
                )
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

        # First application of the template matrix will propagate flags and
        # the flag mask to the templates
        self.template_matrix.det_data_units = det_data_units
        self.template_matrix.det_flags = self.solver_flags
        self.template_matrix.det_flag_mask = 255

        # Set our binning operator to use only our new solver flags
        self.binning.shared_flag_mask = 0
        self.binning.det_flags = self.solver_flags
        self.binning.det_flag_mask = 255
        self.binning.det_data_units = det_data_units

        # Set the binning operator to output to temporary map.  This will be
        # overwritten on each iteration of the solver.
        self.binning.binned = self.solver_bin
        self.binning.covariance = self.solver_cov_name

        self.template_matrix.amplitudes = self.solver_rhs

        rhs_calc = SolverRHS(
            name=f"{self.name}_rhs",
            det_data=self.det_data,
            det_data_units=det_data_units,
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
            det_data_units=det_data_units,
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
            self.amplitudes,
            convergence=self.convergence,
            n_iter_min=self.iter_min,
            n_iter_max=self.iter_max,
        )

        log.info_rank(
            f"{log_prefix}  finished solver in",
            comm=comm,
            timer=timer,
        )

        memreport.prefix = "After solving for amplitudes"
        memreport.apply(data)

        # FIXME:  This I/O technique assumes "known" types of pixel representations.
        # Instead, we should associate read / write functions to a particular pixel
        # class.

        is_pix_wcs = hasattr(self.binning.pixel_pointing, "wcs")
        is_hpix_nest = None
        if not is_pix_wcs:
            is_hpix_nest = self.binning.pixel_pointing.nest

        write_del = [
            self.solver_hits_name,
            self.solver_cov_name,
            self.solver_rcond_name,
            self.solver_rcond_mask_name,
            self.solver_bin,
        ]
        for prod_key in write_del:
            if self.write_solver_products:
                if is_pix_wcs:
                    fname = os.path.join(self.output_dir, "{}.fits".format(prod_key))
                    write_wcs_fits(data[prod_key], fname)
                else:
                    if self.write_hdf5:
                        # Non-standard HDF5 output
                        fname = os.path.join(self.output_dir, "{}.h5".format(prod_key))
                        write_healpix_hdf5(
                            data[prod_key],
                            fname,
                            nest=is_hpix_nest,
                            single_precision=True,
                            force_serial=self.write_hdf5_serial,
                        )
                    else:
                        # Standard FITS output
                        fname = os.path.join(
                            self.output_dir, "{}.fits".format(prod_key)
                        )
                        write_healpix_fits(
                            data[prod_key],
                            fname,
                            nest=is_hpix_nest,
                            report_memory=self.report_memory,
                        )
            if not self.mc_mode and not self.keep_solver_products:
                if prod_key in data:
                    data[prod_key].clear()
                    del data[prod_key]

        if not self.mc_mode and not self.keep_solver_products:
            if self.solver_rhs in data:
                data[self.solver_rhs].clear()
                del data[self.solver_rhs]
            for ob in data.obs:
                del ob.detdata[self.solver_flags]

        # Restore flag names and masks to binning operator, in case it is being used
        # for the final map making or for other external operations.

        self.binning.det_flags = save_det_flags
        self.binning.det_flag_mask = save_det_flag_mask
        self.binning.shared_flags = save_shared_flags
        self.binning.shared_flag_mask = save_shared_flag_mask
        self.binning.binned = save_binned
        self.binning.covariance = save_covariance

        self.template_matrix.det_flags = save_tmpl_flags
        self.template_matrix.det_flag_mask = save_tmpl_mask
        # FIXME: this reset does not seem needed
        # if not self.mc_mode:
        #    self.template_matrix.reset_templates()

        memreport.prefix = "End of amplitude solve"
        memreport.apply(data)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator requires everything that its sub-operators needs.
        req = self.binning.requires()
        if self.template_matrix is not None:
            req.update(self.template_matrix.requires())
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        prov = dict()
        prov["global"] = [self.amplitudes]
        if self.keep_solver_products:
            prov["global"].extend(
                [
                    self.solver_hits_name,
                    self.solver_cov_name,
                    self.solver_rcond_name,
                    self.solver_rcond_mask_name,
                    self.solver_rhs,
                    self.solver_bin,
                ]
            )
            prov["detdata"] = [self.solver_flags]
        return prov


@trait_docs
class ApplyAmplitudes(Operator):
    """Project template amplitudes and do timestream arithmetic.

    This projects amplitudes to the time domain and then adds, subtracts, multiplies,
    or divides the original timestream by this result.

    """

    API = Int(0, help="Internal interface version for this operator")

    op = Unicode(
        "subtract",
        help="Operation on the timestreams: 'subtract', 'add', 'multiply', or 'divide'",
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    amplitudes = Unicode(None, allow_none=True, help="Data key for input amplitudes")

    template_matrix = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a template matrix operator",
    )

    output = Unicode(
        None,
        allow_none=True,
        help="Name of output detdata.  If None, overwrite input.",
    )

    report_memory = Bool(False, help="Report memory throughout the execution")

    @traitlets.validate("op")
    def _check_op(self, proposal):
        val = proposal["value"]
        if val is not None:
            if val not in ["add", "subtract", "multiply", "divide"]:
                raise traitlets.TraitError("op must be one of the 4 allowed strings")
        return val

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()

        # Check if we have any templates
        if self.template_matrix is None:
            return

        n_enabled_templates = 0
        for template in self.template_matrix.templates:
            if template.enabled:
                n_enabled_templates += 1

        if n_enabled_templates == 0:
            # Nothing to do!
            return

        # Get the units used across the distributed data for our desired
        # input detector data
        det_data_units = data.detector_units(self.det_data)

        # Temporary location for single-detector projected template
        # timestreams.
        projected = f"{self.name}_temp"

        # Are we saving the resulting timestream to a new location?  If so,
        # create that now for all detectors.

        if self.output is not None:
            # We just copy the input here, since it will be overwritten
            Copy(detdata=[(self.det_data, self.output)]).apply(
                data, use_accel=use_accel
            )

        # Projecting amplitudes to timestreams
        self.template_matrix.transpose = False
        self.template_matrix.det_data = projected
        self.template_matrix.det_data_units = det_data_units
        self.template_matrix.amplitudes = self.amplitudes

        # Arithmetic operator
        combine = Combine(op=self.op, first=self.det_data, second=projected)
        if self.output is None:
            combine.result = self.det_data
        else:
            combine.result = self.output

        # Project and operate, one detector at a time
        pipe = Pipeline(
            detector_sets=["SINGLE"],
            operators=[
                self.template_matrix,
                combine,
            ],
        )
        pipe.apply(data, use_accel=use_accel)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator requires everything that its sub-operators needs.
        req = dict()
        req["global"] = [self.amplitudes]
        req["detdata"] = list()
        if self.template_matrix is not None:
            req.update(self.template_matrix.requires())
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        prov = dict()
        prov["detdata"] = [self.output]
        return prov
