# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets

from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Unicode, trait_docs
from ..utils import Logger
from .copy import Copy
from .delete import Delete
from .mapmaker_templates import ApplyAmplitudes, SolveAmplitudes
from .mapmaker_utils import CovarianceAndHits
from .memory_counter import MemoryCounter
from .operator import Operator
from .pipeline import Pipeline
from .pointing import BuildPixelDistribution
from .scan_map import ScanMap, ScanMask


@trait_docs
class MapMaker(Operator):
    r"""Operator for making maps.

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

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

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

    map_binning = Instance(
        klass=Operator,
        allow_none=True,
        help="Binning operator for final map making.  Default is same as solver",
    )

    write_binmap = Bool(
        True, help="If True, write the projected map *before* template subtraction"
    )

    write_map = Bool(True, help="If True, write the projected map")

    write_hdf5 = Bool(
        False, help="If True, outputs are in HDF5 rather than FITS format."
    )

    write_hdf5_serial = Bool(
        False, help="If True, force serial HDF5 write of output maps."
    )

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

    write_solver_products = Bool(
        False, help="If True, write out equivalent solver products."
    )

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
        help="Clear any existing pixel distribution.  Useful when applying "
        "repeatedly to different data objects.",
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    report_memory = Bool(False, help="Report memory throughout the execution")

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
                "det_mask",
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
    def _write_del(self, prod_key, prod_write, force, rootname, extra_header=None):
        """Write data object to file and delete it from cache"""
        log = Logger.get()

        # FIXME:  This I/O technique assumes "known" types of pixel representations.
        # Instead, we should associate read / write functions to a particular pixel
        # class.

        if self.map_binning is not None and self.map_binning.enabled:
            map_binning = self.map_binning
        else:
            map_binning = self.binning

        is_hpix_nest = True
        if not hasattr(map_binning.pixel_pointing, "wcs"):
            is_hpix_nest = map_binning.pixel_pointing.nest

        wtimer = Timer()
        wtimer.start()
        product = prod_key.replace(f"{self.name}_", "")
        if prod_write:
            if self.write_hdf5:
                fname = os.path.join(self.output_dir, f"{rootname}_{product}.h5")
            else:
                fname = os.path.join(self.output_dir, f"{rootname}_{product}.fits")
            if self.mc_mode and not force and os.path.isfile(fname):
                log.info_rank(f"Skipping existing file: {fname}", comm=self._comm)
            else:
                self._data[prod_key].write(
                    fname,
                    force_serial=self.write_hdf5_serial,
                    single_precision=True,
                    report_memory=self.report_memory,
                    extra_header=extra_header,
                )
            log.info_rank(f"Wrote {fname} in", comm=self._comm, timer=wtimer)

        if not self.keep_final_products and not self.mc_mode:
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
        self._log_prefix = "MapMaker"

        self._mc_root = self.name
        if self.mc_mode:
            if self.mc_root is not None:
                self._mc_root += f"_{self.mc_root}"
            if self.mc_index is not None:
                self._mc_root += f"_{self.mc_index:05d}"

        self._data = data
        self._detectors = detectors
        self._use_accel = use_accel
        self._memreport = MemoryCounter()
        if not self.report_memory:
            self._memreport.enabled = False

        # The global communicator we are using (or None)

        self._comm = data.comm.comm_world
        self._rank = data.comm.world_rank

        # Data names of outputs

        self.hits_name = f"{self.name}_hits"
        self.cov_name = f"{self.name}_cov"
        self.invcov_name = f"{self.name}_invcov"
        self.rcond_name = f"{self.name}_rcond"
        self.det_flag_name = f"{self.name}_flags"

        self.clean_name = f"{self.name}_cleaned"
        self.binmap_name = f"{self.name}_binmap"
        self.map_name = f"{self.name}_map"
        self.noiseweighted_map_name = f"{self.name}_noiseweighted_map"

        self._timer.start()

        return

    @function_timer
    def _fit_templates(self):
        """Solve for template amplitudes"""

        amplitudes_solve = SolveAmplitudes(
            name=self.name,
            det_data=self.det_data,
            convergence=self.convergence,
            iter_min=self.iter_min,
            iter_max=self.iter_max,
            solve_rcond_threshold=self.solve_rcond_threshold,
            mask=self.mask,
            binning=self.binning,
            template_matrix=self.template_matrix,
            keep_solver_products=self.keep_solver_products,
            write_solver_products=self.write_solver_products,
            write_hdf5=self.write_hdf5,
            write_hdf5_serial=self.write_hdf5_serial,
            output_dir=self.output_dir,
            mc_mode=self.mc_mode,
            mc_index=self.mc_index,
            reset_pix_dist=self.reset_pix_dist,
            report_memory=self.report_memory,
        )
        amplitudes_solve.apply(
            self._data, detectors=self._detectors, use_accel=self._use_accel
        )
        template_amplitudes = amplitudes_solve.amplitudes

        self._log.info_rank(
            f"{self._log_prefix}  finished template amplitude solve in",
            comm=self._comm,
            timer=self._timer,
        )

        self._memreport.prefix = "After solving amplitudes"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        return template_amplitudes

    @function_timer
    def _prepare_binning(self):
        """Set up the final map binning"""

        # Map binning operator
        if self.map_binning is not None and self.map_binning.enabled:
            map_binning = self.map_binning
        else:
            # Use the same binning used in the solver.
            map_binning = self.binning
        map_binning.pre_process = None
        map_binning.covariance = self.cov_name

        # Pixel distribution
        if self.reset_pix_dist:
            # Purge any stale products from previous runs
            for name in [
                self.hits_name,
                self.cov_name,
                self.invcov_name,
                self.rcond_name,
                self.clean_name,
                self.binmap_name,
                self.map_name,
                self.noiseweighted_map_name,
                map_binning.pixel_dist,
                map_binning.covariance,
            ]:
                if name in self._data:
                    del self._data[name]

        if map_binning.pixel_dist not in self._data:
            self._log.info_rank(
                f"{self._log_prefix} Caching pixel distribution",
                comm=self._comm,
            )
            pix_dist = BuildPixelDistribution(
                pixel_dist=map_binning.pixel_dist,
                pixel_pointing=map_binning.pixel_pointing,
                save_pointing=map_binning.full_pointing,
            )
            pix_dist.apply(self._data, use_accel=self._use_accel)
            self._log.info_rank(
                f"{self._log_prefix}  finished build of pixel distribution in",
                comm=self._comm,
                timer=self._timer,
            )

            self._memreport.prefix = "After pixel distribution"
            self._memreport.apply(self._data, use_accel=self._use_accel)

        return map_binning

    @function_timer
    def _build_pixel_covariance(self, map_binning):
        """Accumulate hits and pixel covariance"""

        if map_binning.covariance in self._data and self.mc_mode:
            # Covariance is already cached
            return

        # Construct the noise covariance, hits, and condition number
        # mask for the final binned map.

        self._log.info_rank(
            f"{self._log_prefix} begin build of final binning covariance",
            comm=self._comm,
        )

        final_cov = CovarianceAndHits(
            pixel_dist=map_binning.pixel_dist,
            covariance=map_binning.covariance,
            inverse_covariance=self.invcov_name,
            hits=self.hits_name,
            rcond=self.rcond_name,
            det_mask=map_binning.det_mask,
            det_flags=map_binning.det_flags,
            det_flag_mask=map_binning.det_flag_mask,
            det_data_units=map_binning.det_data_units,
            shared_flags=map_binning.shared_flags,
            shared_flag_mask=map_binning.shared_flag_mask,
            pixel_pointing=map_binning.pixel_pointing,
            stokes_weights=map_binning.stokes_weights,
            noise_model=map_binning.noise_model,
            rcond_threshold=self.map_rcond_threshold,
            sync_type=map_binning.sync_type,
            save_pointing=map_binning.full_pointing,
        )

        final_cov.apply(
            self._data, detectors=self._detectors, use_accel=self._use_accel
        )

        self._log.info_rank(
            f"{self._log_prefix}  finished build of final covariance in",
            comm=self._comm,
            timer=self._timer,
        )

        self._memreport.prefix = "After constructing final covariance and hits"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        # These data products are not needed later so they can be
        # written out and purged

        self._write_del(self.hits_name, self.write_hits, False, self.name)
        self._write_del(self.rcond_name, self.write_rcond, False, self.name)
        self._write_del(self.invcov_name, self.write_invcov, False, self.name)

        return

    @function_timer
    def _bin_and_write_raw_signal(self, map_binning, extra_header=None):
        """Optionally bin and save an undestriped map"""

        if not self.write_binmap:
            return

        map_binning.det_data = self.det_data
        map_binning.binned = self.binmap_name
        map_binning.noiseweighted = None
        self._log.info_rank(
            f"{self._log_prefix} begin map binning",
            comm=self._comm,
        )
        map_binning.apply(
            self._data, detectors=self._detectors, use_accel=self._use_accel
        )
        self._log.info_rank(
            f"{self._log_prefix}  finished binning in",
            comm=self._comm,
            timer=self._timer,
        )
        self._write_del(
            self.binmap_name,
            self.write_binmap,
            True,
            self._mc_root,
            extra_header=extra_header,
        )

        self._memreport.prefix = "After binning final map"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        return

    @function_timer
    def _clean_signal(self, template_amplitudes):
        if (
            self.template_matrix is None
            or self.template_matrix.n_enabled_templates == 0
        ):
            # No templates to subtract, bin the input signal
            out_cleaned = self.det_data
        else:
            # Apply (subtract) solved amplitudes.

            self._log.info_rank(
                f"{self._log_prefix} begin apply template amplitudes",
                comm=self._comm,
            )

            out_cleaned = self.clean_name
            if self.save_cleaned and self.overwrite_cleaned:
                # Modify data in place
                out_cleaned = None

            amplitudes_apply = ApplyAmplitudes(
                op="subtract",
                det_data=self.det_data,
                amplitudes=template_amplitudes,
                template_matrix=self.template_matrix,
                output=out_cleaned,
            )
            amplitudes_apply.apply(
                self._data, detectors=self._detectors, use_accel=self._use_accel
            )

            if not self.keep_solver_products:
                del self._data[template_amplitudes]

            self._log.info_rank(
                f"{self._log_prefix}  finished apply template amplitudes in",
                comm=self._comm,
                timer=self._timer,
            )

            self._memreport.prefix = "After subtracting templates"
            self._memreport.apply(self._data, use_accel=self._use_accel)

        return out_cleaned

    @function_timer
    def _bin_cleaned_signal(self, map_binning, out_cleaned):
        """Bin and save a map of the destriped signal"""

        self._log.info_rank(
            f"{self._log_prefix} begin final map binning",
            comm=self._comm,
        )

        if out_cleaned is None:
            map_binning.det_data = self.det_data
        else:
            map_binning.det_data = out_cleaned
        if self.write_noiseweighted_map or self.keep_final_products:
            map_binning.noiseweighted = self.noiseweighted_map_name
        map_binning.binned = self.map_name

        # Do the final binning
        map_binning.apply(
            self._data, detectors=self._detectors, use_accel=self._use_accel
        )

        self._log.info_rank(
            f"{self._log_prefix}  finished final binning in",
            comm=self._comm,
            timer=self._timer,
        )

        self._memreport.prefix = "After binning final map"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        return

    @function_timer
    def _purge_cleaned_tod(self):
        """If the cleaned TOD is not being returned, purge it"""

        if self.save_cleaned:
            return

        del_tod = Delete(detdata=[self.clean_name])
        del_tod.apply(self._data, use_accel=self._use_accel)

        self._memreport.prefix = "After purging cleaned TOD"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        return

    @function_timer
    def _write_maps(self, extra_header=None):
        """Write and delete the outputs"""

        self._write_del(
            self.noiseweighted_map_name,
            self.write_noiseweighted_map,
            True,
            self._mc_root,
        )
        self._write_del(
            self.map_name,
            self.write_map,
            True,
            self._mc_root,
            extra_header=extra_header,
        )
        self._write_del(
            self.cov_name, self.write_cov, False, self.name, extra_header=extra_header
        )

        self._log.info_rank(
            f"{self._log_prefix}  finished output write in",
            comm=self._comm,
            timer=self._timer,
        )

        return

    @function_timer
    def _closeout(self):
        """Explicitly delete members used by the _exec() method"""

        del self._log
        del self._timer
        del self._log_prefix
        del self._mc_root
        del self._data
        del self._detectors
        del self._use_accel
        del self._memreport
        del self._comm
        del self._rank

        return

    @function_timer
    def _get_extra_header(self, data, detectors):
        """Extract useful information from the data object to record in
        map headers"""
        extra_header = {}
        start = None
        stop = None
        all_dets = set()
        good_dets = set()
        for ob in data.obs:
            times = ob.shared[self.times].data
            if start is None:
                start = times[0]
            else:
                start = min(start, times[0])
            if stop is None:
                stop = times[-1]
            else:
                stop = max(stop, times[-1])
            all_dets.update(ob.select_local_detectors(detectors))
            good_dets.update(
                ob.select_local_detectors(
                    detectors,
                    flagmask=self.binning.det_mask,
                )
            )
        if self._comm is not None:
            start = self._comm.allreduce(start, op=MPI.MIN)
            stop = self._comm.allreduce(stop, op=MPI.MAX)
            all_dets_list = self._comm.allgather(all_dets)
            good_dets_list = self._comm.allgather(good_dets)
            all_dets.update(*all_dets_list)
            good_dets.update(*good_dets_list)
        extra_header["START"] = (start, "Dataset start time")
        extra_header["STOP"] = (stop, "Dataset stop time")
        extra_header["NDET"] = (len(all_dets), "Total number of detectors")
        extra_header["NGOOD"] = (len(good_dets), "Total number of usable detectors")
        extra_header["OPERATOR"] = ("TOAST MapMaker", "Generating code")

        return extra_header

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        # First confirm that there is at least one valid detector

        if self.map_binning is not None and self.map_binning.enabled:
            map_binning = self.map_binning
        else:
            # Use the same binning used in the solver.
            map_binning = self.binning
        all_local_dets = data.all_local_detectors(
            selection=detectors, flagmask=map_binning.det_mask
        )
        ndet = len(all_local_dets)
        if data.comm.comm_world is not None:
            ndet = data.comm.comm_world.allreduce(ndet, op=MPI.SUM)
        if ndet == 0:
            # No valid detectors, no mapmaking
            return

        # Destripe data and make maps

        self._setup(data, detectors, use_accel)

        extra_header = self._get_extra_header(data, detectors)

        self._memreport.prefix = "Start of mapmaking"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        template_amplitudes = self._fit_templates()

        map_binning = self._prepare_binning()

        self._build_pixel_covariance(map_binning)

        self._bin_and_write_raw_signal(map_binning, extra_header=extra_header)

        out_cleaned = self._clean_signal(template_amplitudes)

        if self.write_noiseweighted_map or self.write_map or self.keep_final_products:
            self._bin_cleaned_signal(map_binning, out_cleaned)

        self._purge_cleaned_tod()  # Potentially frees memory for writing maps

        self._write_maps(extra_header=extra_header)

        self._memreport.prefix = "End of mapmaking"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        self._closeout()

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
            prov["global"] = [self.map_binning.binned]
        else:
            prov["global"] = [self.binning.binned]
        return prov


@trait_docs
class Calibrate(Operator):
    r"""Operator for calibrating timestreams using solved templates.

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

    After solving for the template amplitudes, they are projected into the time
    domain and the input data is element-wise divided by this.

    If the result trait is not set, then the input is overwritten.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    result = Unicode(
        None, allow_none=True, help="Observation detdata key for the output"
    )

    convergence = Float(1.0e-12, help="Relative convergence limit")

    iter_min = Int(3, help="Minimum number of iterations")

    iter_max = Int(100, help="Maximum number of iterations")

    solve_rcond_threshold = Float(
        1.0e-8,
        help="When solving, minimum value for inverse pixel condition number cut.",
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

    mc_mode = Bool(False, help="If True, re-use solver flags, sparse covariances, etc")

    mc_index = Int(None, allow_none=True, help="The Monte-Carlo index")

    mc_root = Unicode(None, allow_node=True, help="Root name for Monte Carlo products")

    reset_pix_dist = Bool(
        False,
        help="Clear any existing pixel distribution.  Useful when applying "
        "repeatedly to different data objects.",
    )

    report_memory = Bool(False, help="Report memory throughout the execution")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        log_prefix = "Calibrate"

        memreport = MemoryCounter()
        if not self.report_memory:
            memreport.enabled = False

        memreport.prefix = "Start of calibration"
        memreport.apply(data, use_accel=use_accel)

        # The global communicator we are using (or None)
        comm = data.comm.comm_world
        rank = data.comm.world_rank

        timer.start()

        # Solve for template amplitudes
        amplitudes_solve = SolveAmplitudes(
            name=self.name,
            det_data=self.det_data,
            convergence=self.convergence,
            iter_min=self.iter_min,
            iter_max=self.iter_max,
            solve_rcond_threshold=self.solve_rcond_threshold,
            mask=self.mask,
            binning=self.binning,
            template_matrix=self.template_matrix,
            keep_solver_products=self.keep_solver_products,
            mc_mode=self.mc_mode,
            mc_index=self.mc_index,
            reset_pix_dist=self.reset_pix_dist,
            report_memory=self.report_memory,
        )
        amplitudes_solve.apply(data, detectors=detectors, use_accel=use_accel)

        log.info_rank(
            f"{log_prefix}  finished template amplitude solve in",
            comm=comm,
            timer=timer,
        )

        # Apply (divide) solved amplitudes.

        log.info_rank(
            f"{log_prefix} begin apply template amplitudes",
            comm=comm,
        )

        out_calib = self.det_data
        if self.result is not None:
            # We are writing out calibrated timestreams to a new set of detector
            # data rather than overwriting the inputs.  Here we create these output
            # timestreams if they do not exist.  We do this by copying the inputs,
            # since the application of the amplitudes below will zero these
            out_calib = self.result
            Copy(detdata=[(self.det_data, self.result)]).apply(
                data, use_accel=use_accel
            )

        amplitudes_apply = ApplyAmplitudes(
            op="divide",
            det_data=self.det_data,
            amplitudes=amplitudes_solve.amplitudes,
            template_matrix=self.template_matrix,
            output=out_calib,
        )
        amplitudes_apply.apply(data, detectors=detectors, use_accel=use_accel)

        log.info_rank(
            f"{log_prefix}  finished apply template amplitudes in",
            comm=comm,
            timer=timer,
        )

        memreport.prefix = "After calibration"
        memreport.apply(data, use_accel=use_accel)

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
        prov["global"] = [self.binning.binned]
        if self.result is not None:
            prov["detdata"] = [self.result]
        return prov
