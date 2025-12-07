# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
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

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

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
            det_mask=self.det_mask,
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
    def initialize(self, data, use_accel=False):
        if not self._initialized:
            if use_accel:
                # fail when a user tries to run the initialization pipeline on GPU
                raise RuntimeError(
                    "You cannot currently initialize templates on device (please disable accel for this operator/pipeline)."
                )
            for tmpl in self.templates:
                if not tmpl.enabled:
                    continue
                if tmpl.view is None:
                    tmpl.view = self.view
                tmpl.det_data_units = self.det_data_units
                tmpl.det_mask = self.det_mask
                tmpl.det_flags = self.det_flags
                tmpl.det_flag_mask = self.det_flag_mask
                # This next line will trigger calculation of the number
                # of amplitudes within each template.
                tmpl.data = data
            self._initialized = True

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

        # Ensure we have initialized templates with the full set of detectors.
        if not self._initialized:
            raise RuntimeError("You must call initialize() before calling exec()")

        # Set the data we are using for this execution
        for tmpl in self.templates:
            if tmpl.enabled:
                tmpl.det_data = self.det_data

        # We loop over detectors.  Internally, each template loops over observations
        # and ignores observations where the detector does not exist.
        all_dets = data.all_local_detectors(selection=detectors, flagmask=self.det_mask)

        if self.transpose:
            # Check that the incoming detector data in all observations has the correct
            # units.
            input_units = 1.0 / self.det_data_units
            for ob in data.obs:
                if self.det_data not in ob.detdata:
                    continue
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
                dets = ob.select_local_detectors(
                    selection=detectors,
                    flagmask=self.det_mask,
                )
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
                    if tmpl.enabled:
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
        \\left[ M^T N^{-1} Z M + M_p \\right] a = M^T N^{-1} Z d

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
        help="Data key for pixel mask to use in solving.  "
        "First bit of pixel values is tested",
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
                "det_mask",
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
    def _write_del(self, prod_key):
        """Write and optionally delete map object"""
        if self.write_solver_products:
            if self.write_hdf5:
                fname = os.path.join(self.output_dir, f"{prod_key}.h5")
            else:
                fname = os.path.join(self.output_dir, f"{prod_key}.fits")
            self._data[prod_key].write(
                fname,
                force_serial=self.write_hdf5_serial,
                single_precision=True,
                report_memory=self.report_memory,
            )

        if not self.mc_mode and not self.keep_solver_products:
            if prod_key in self._data:
                self._data[prod_key].clear()
                del self._data[prod_key]

                self._memreport.prefix = f"After writing/deleting {prod_key}"
                self._memreport.apply(self._data, use_accel=self._use_accel)

        return

    @function_timer
    def _setup(self, data, detectors, use_accel):
        """Set up convenience members used in the _exec() method"""

        self._log = Logger.get()
        self._timer = Timer()
        self._log_prefix = "SolveAmplitudes"

        self._data = data
        self._detectors = detectors
        self._use_accel = use_accel
        self._memreport = MemoryCounter()
        if not self.report_memory:
            self._memreport.enabled = False

        # The global communicator we are using (or None)
        self._comm = data.comm.comm_world
        self._rank = data.comm.world_rank

        # Get the units used across the distributed data for our desired
        # input detector data
        self._det_data_units = data.detector_units(self.det_data)

        # We use the input binning operator to define the flags that the user has
        # specified.  We will save the name / bit mask for these and restore them later.
        # Then we will use the binning operator with our solver flags.  These input
        # flags are combined to the first bit (== 1) of the solver flags.

        self._save_det_flags = self.binning.det_flags
        self._save_det_mask = self.binning.det_mask
        self._save_det_flag_mask = self.binning.det_flag_mask
        self._save_shared_flags = self.binning.shared_flags
        self._save_shared_flag_mask = self.binning.shared_flag_mask
        self._save_binned = self.binning.binned
        self._save_covariance = self.binning.covariance

        self._save_tmpl_flags = self.template_matrix.det_flags
        self._save_tmpl_mask = self.template_matrix.det_mask
        self._save_tmpl_det_mask = self.template_matrix.det_flag_mask

        # Use the same data view as the pointing operator in binning
        self._solve_view = self.binning.pixel_pointing.view

        # Output data products, prefixed with the name of the operator and optionally
        # the MC index.

        if self.mc_mode and self.mc_index is not None:
            self._mc_root = "{self.name}_{self.mc_index:05d}"
        else:
            self._mc_root = self.name

        self.solver_flags = f"{self.name}_solve_flags"
        self.solver_hits_name = f"{self.name}_solve_hits"
        self.solver_cov_name = f"{self.name}_solve_cov"
        self.solver_rcond_name = f"{self.name}_solve_rcond"
        self.solver_rcond_mask_name = f"{self.name}_solve_rcond_mask"
        self.solver_rhs = f"{self._mc_root}_solve_rhs"
        self.solver_bin = f"{self._mc_root}_solve_bin"

        if self.amplitudes is None:
            self.amplitudes = f"{self._mc_root}_solve_amplitudes"

        return

    @function_timer
    def _prepare_pixels(self):
        """Optionally destroy existing pixel distributions (useful if calling
        repeatedly with different data objects)
        """

        if self.reset_pix_dist:
            if self.binning.pixel_dist in self._data:
                del self._data[self.binning.pixel_dist]

        self._memreport.prefix = "After resetting pixel distribution"
        self._memreport.apply(self._data)

        # The pointing matrix used for the solve.  The per-detector flags
        # are normally reset when the binner is run, but here we set them
        # explicitly since we will use these pointing matrix operators for
        # setting up the solver flags below.
        solve_pixels = self.binning.pixel_pointing
        solve_weights = self.binning.stokes_weights
        solve_pixels.detector_pointing.det_mask = self._save_det_mask
        solve_pixels.detector_pointing.det_flag_mask = self._save_det_flag_mask
        if hasattr(solve_weights, "detector_pointing"):
            solve_weights.detector_pointing.det_mask = self._save_det_mask
            solve_weights.detector_pointing.det_flag_mask = self._save_det_flag_mask

        # Set up a pipeline to scan processing and condition number masks
        self._scanner = ScanMask(
            det_flags=self.solver_flags,
            det_mask=self._save_det_mask,
            det_flag_mask=self._save_det_flag_mask,
            pixels=solve_pixels.pixels,
            view=self._solve_view,
        )
        if self.binning.full_pointing:
            # We are caching the pointing anyway- run with all detectors
            scan_pipe = Pipeline(
                detector_sets=["ALL"], operators=[solve_pixels, self._scanner]
            )
        else:
            # Pipeline over detectors
            scan_pipe = Pipeline(
                detector_sets=["SINGLE"], operators=[solve_pixels, self._scanner]
            )

        return solve_pixels, solve_weights, scan_pipe

    @function_timer
    def _prepare_flagging_ob(self, ob):
        """Process a single observation, used by _prepare_flagging

        Copies and masks existing flags
        """

        # Get the detectors we are using for this observation
        dets = ob.select_local_detectors(self._detectors, flagmask=self._save_det_mask)
        if len(dets) == 0:
            # Nothing to do for this observation
            return

        if self.mc_mode:
            # Shortcut, just verify that our flags exist
            if self.solver_flags not in ob.detdata:
                msg = f"In MC mode, solver flags missing for observation {ob.name}"
                self._log.error(msg)
                raise RuntimeError(msg)
            det_check = set(ob.detdata[self.solver_flags].detectors)
            for d in dets:
                if d not in det_check:
                    msg = "In MC mode, solver flags missing for "
                    msg + f"observation {ob.name}, det {d}"
                    self._log.error(msg)
                    raise RuntimeError(msg)
            return

        # Create the new solver flags
        exists = ob.detdata.ensure(self.solver_flags, dtype=np.uint8, detectors=dets)

        # The data views
        views = ob.view[self._solve_view]
        # For each view...
        for vw in range(len(views)):
            view_samples = None
            if views[vw].start is None:
                # There is one view of the whole obs
                view_samples = ob.n_local_samples
            else:
                view_samples = views[vw].stop - views[vw].start
            starting_flags = np.zeros(view_samples, dtype=np.uint8)
            if self._save_shared_flags is not None:
                starting_flags[:] = np.where(
                    (
                        views.shared[self._save_shared_flags][vw]
                        & self._save_shared_flag_mask
                    )
                    > 0,
                    1,
                    0,
                )
            for d in dets:
                views.detdata[self.solver_flags][vw][d, :] = starting_flags
                if self._save_det_flags is not None:
                    views.detdata[self.solver_flags][vw][d, :] |= np.where(
                        (
                            views.detdata[self._save_det_flags][vw][d]
                            & self._save_det_flag_mask
                        )
                        > 0,
                        1,
                        0,
                    ).astype(views.detdata[self.solver_flags][vw].dtype)

        return

    @function_timer
    def _prepare_flagging(self, scan_pipe):
        """Flagging.  We create a new set of data flags for the solver that includes:
        - one bit for a bitwise OR of all detector / shared flags
        - one bit for any pixel mask, projected to TOD
        - one bit for any poorly conditioned pixels, projected to TOD
        """

        if self.mc_mode:
            msg = f"{self._log_prefix} begin verifying flags for solver"
        else:
            msg = f"{self._log_prefix} begin building flags for solver"
        self._log.info_rank(msg, comm=self._comm)

        for ob in self._data.obs:
            self._prepare_flagging_ob(ob)

        if self.mc_mode:
            # Shortcut, just verified that our flags exist
            self._log.info_rank(
                f"{self._log_prefix} MC mode, reusing flags for solver", comm=comm
            )
            return

        # Now scan any input mask to this same flag field.  We use the second
        # bit (== 2) for these mask flags.  For the input mask bit we check the
        # first bit of the pixel values.  This is noted in the help string for
        # the mask trait.  Note that we explicitly expand the pointing once
        # here and do not save it.  Even if we are eventually saving the
        # pointing, we want to do that later when building the covariance and
        # the pixel distribution.

        if self.mask is not None:
            # We have a mask.  Scan it.
            self._scanner.det_flags_value = 2
            self._scanner.mask_key = self.mask
            scan_pipe.apply(self._data, detectors=self._detectors)

        self._log.info_rank(
            f"{self._log_prefix}  finished flag building in",
            comm=self._comm,
            timer=self._timer,
        )

        self._memreport.prefix = "After building flags"
        self._memreport.apply(self._data)

        return

    def _count_cut_data(self):
        """Collect and report statistics about cut data"""
        local_total = 0
        local_cut = 0
        for ob in self._data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(
                self._detectors, flagmask=self._save_det_mask
            )
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            for vw in ob.view[self._solve_view].detdata[self.solver_flags]:
                for d in dets:
                    local_total += len(vw[d])
                    local_cut += np.count_nonzero(vw[d])

        if self._comm is None:
            total = local_total
            cut = local_cut
        else:
            total = self._comm.allreduce(local_total, op=MPI.SUM)
            cut = self._comm.allreduce(local_cut, op=MPI.SUM)

        frac = 100.0 * (cut / total)
        msg = f"Solver flags cut {cut} / {total} = {frac:0.2f}% of samples"
        self._log.info_rank(f"{self._log_prefix} {msg}", comm=self._comm)

        return

    @function_timer
    def _get_pixel_covariance(self, solve_pixels, solve_weights):
        """Construct the noise covariance, hits, and condition number map for
        the solver.
        """

        if self.mc_mode:
            # Shortcut, verify that our covariance and other products exist.
            if self.binning.pixel_dist not in self._data:
                msg = f"MC mode, pixel distribution "
                msg += f"'{self.binning.pixel_dist}' does not exist"
                self._log.error(msg)
                raise RuntimeError(msg)
            if self.solver_cov_name not in self._data:
                msg = f"MC mode, covariance '{self.solver_cov_name}' does not exist"
                self._log.error(msg)
                raise RuntimeError(msg)
            self._log.info_rank(
                f"{self._log_prefix} MC mode, reusing covariance for solver",
                comm=self._comm,
            )
            return

        self._log.info_rank(
            f"{self._log_prefix} begin build of solver covariance",
            comm=self._comm,
        )

        solver_cov = CovarianceAndHits(
            pixel_dist=self.binning.pixel_dist,
            covariance=self.solver_cov_name,
            hits=self.solver_hits_name,
            rcond=self.solver_rcond_name,
            det_data_units=self._det_data_units,
            det_mask=self._save_det_mask,
            det_flags=self.solver_flags,
            det_flag_mask=255,
            pixel_pointing=solve_pixels,
            stokes_weights=solve_weights,
            noise_model=self.binning.noise_model,
            rcond_threshold=self.solve_rcond_threshold,
            sync_type=self.binning.sync_type,
            save_pointing=self.binning.full_pointing,
        )

        solver_cov.apply(self._data, detectors=self._detectors)

        self._memreport.prefix = "After constructing covariance and hits"
        self._memreport.apply(self._data)

        return

    @function_timer
    def _get_rcond_mask(self, scan_pipe):
        """Construct the noise covariance, hits, and condition number mask for
        the solver.
        """

        if self.mc_mode:
            # The flags are already cached
            return

        self._log.info_rank(
            f"{self._log_prefix} begin build of rcond flags",
            comm=self._comm,
        )

        # Translate the rcond map into a mask
        self._data[self.solver_rcond_mask_name] = PixelData(
            self._data[self.binning.pixel_dist], dtype=np.uint8, n_value=1
        )
        rcond = self._data[self.solver_rcond_name].data
        rcond_mask = self._data[self.solver_rcond_mask_name].data
        bad = rcond < self.solve_rcond_threshold
        n_bad = np.count_nonzero(bad)
        n_good = rcond.size - n_bad
        rcond_mask[bad] = 1

        # No more need for the rcond map
        self._write_del(self.solver_rcond_name)

        self._memreport.prefix = "After constructing rcond mask"
        self._memreport.apply(self._data)

        # Re-use our mask scanning pipeline, setting third bit (== 4)
        self._scanner.det_flags_value = 4
        self._scanner.mask_key = self.solver_rcond_mask_name
        scan_pipe.apply(self._data, detectors=self._detectors)

        self._log.info_rank(
            f"{self._log_prefix}  finished build of solver covariance in",
            comm=self._comm,
            timer=self._timer,
        )

        self._count_cut_data()  # Report statistics

        return

    @function_timer
    def _get_rhs(self):
        """Compute the RHS.  Overwrite inputs, either the original or the copy"""

        self._log.info_rank(
            f"{self._log_prefix} begin RHS calculation", comm=self._comm
        )

        # Initialize the template matrix
        self.template_matrix.det_data = self.det_data
        self.template_matrix.det_data_units = self._det_data_units
        self.template_matrix.det_flags = self.solver_flags
        self.template_matrix.det_mask = self._save_det_mask
        self.template_matrix.det_flag_mask = 255
        self.template_matrix.view = self.binning.pixel_pointing.view
        self.template_matrix.initialize(self._data)

        # Set our binning operator to use only our new solver flags
        self.binning.shared_flag_mask = 0
        self.binning.det_flags = self.solver_flags
        self.binning.det_flag_mask = 255
        self.binning.det_data_units = self._det_data_units

        # Set the binning operator to output to temporary map.  This will be
        # overwritten on each iteration of the solver.
        self.binning.binned = self.solver_bin
        self.binning.covariance = self.solver_cov_name

        self.template_matrix.amplitudes = self.solver_rhs

        rhs_calc = SolverRHS(
            name=f"{self.name}_rhs",
            det_data=self.det_data,
            det_data_units=self._det_data_units,
            binning=self.binning,
            template_matrix=self.template_matrix,
        )
        rhs_calc.apply(self._data, detectors=self._detectors)

        self._log.info_rank(
            f"{self._log_prefix}  finished RHS calculation in",
            comm=self._comm,
            timer=self._timer,
        )

        self._memreport.prefix = "After constructing RHS"
        self._memreport.apply(self._data)

        return

    @function_timer
    def _solve_amplitudes(self):
        """Solve the destriping equation"""

        # Set up the LHS operator.

        self._log.info_rank(
            f"{self._log_prefix} begin PCG solver",
            comm=self._comm,
        )

        lhs_calc = SolverLHS(
            name=f"{self.name}_lhs",
            det_data_units=self._det_data_units,
            binning=self.binning,
            template_matrix=self.template_matrix,
        )

        # If we eventually want to support an input starting guess of the
        # amplitudes, we would need to ensure that data[amplitude_key] is set
        # at this point...

        # Solve for amplitudes.
        solve(
            self._data,
            self._detectors,
            lhs_calc,
            self.solver_rhs,
            self.amplitudes,
            convergence=self.convergence,
            n_iter_min=self.iter_min,
            n_iter_max=self.iter_max,
        )

        self._log.info_rank(
            f"{self._log_prefix}  finished solver in",
            comm=self._comm,
            timer=self._timer,
        )

        self._memreport.prefix = "After solving for amplitudes"
        self._memreport.apply(self._data)

        return

    @function_timer
    def _cleanup(self):
        """Clean up convenience members for _exec()"""

        # Restore flag names and masks to binning operator, in case it is being used
        # for the final map making or for other external operations.

        self.binning.det_flags = self._save_det_flags
        self.binning.det_mask = self._save_det_mask
        self.binning.det_flag_mask = self._save_det_flag_mask
        self.binning.shared_flags = self._save_shared_flags
        self.binning.shared_flag_mask = self._save_shared_flag_mask
        self.binning.binned = self._save_binned
        self.binning.covariance = self._save_covariance

        self.template_matrix.det_flags = self._save_tmpl_flags
        self.template_matrix.det_flag_mask = self._save_tmpl_det_mask
        self.template_matrix.det_mask = self._save_tmpl_mask
        # FIXME: this reset does not seem needed
        # if not self.mc_mode:
        #    self.template_matrix.reset_templates()

        del self._solve_view

        # Delete members used by the _exec() method
        del self._log
        del self._timer
        del self._log_prefix

        del self._data
        del self._detectors
        del self._use_accel
        del self._memreport

        del self._comm
        del self._rank

        del self._det_data_units

        del self._mc_root

        del self._scanner

        return

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        # Check if we have any templates
        if (
            self.template_matrix is None
            or self.template_matrix.n_enabled_templates == 0
        ):
            return

        self._setup(data, detectors, use_accel)

        self._memreport.prefix = "Start of amplitude solve"
        self._memreport.apply(self._data)

        solve_pixels, solve_weights, scan_pipe = self._prepare_pixels()

        self._timer.start()

        self._prepare_flagging(scan_pipe)

        self._get_pixel_covariance(solve_pixels, solve_weights)
        self._write_del(self.solver_hits_name)

        self._get_rcond_mask(scan_pipe)
        self._write_del(self.solver_rcond_mask_name)

        self._get_rhs()
        self._solve_amplitudes()

        self._write_del(self.solver_cov_name)
        self._write_del(self.solver_bin)

        if not self.mc_mode and not self.keep_solver_products:
            if self.solver_rhs in self._data:
                self._data[self.solver_rhs].clear()
                del self._data[self.solver_rhs]
            for ob in self._data.obs:
                del ob.detdata[self.solver_flags]

        self._memreport.prefix = "End of amplitude solve"
        self._memreport.apply(self._data)

        self._cleanup()

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator requires everything that its sub-operators need.
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
        # This operator requires everything that its sub-operators need.
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
