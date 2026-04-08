# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets

from ..mpi import MPI
from ..footprint import footprint_distribution
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Unicode, trait_docs
from ..utils import Logger
from .arithmetic import Combine
from .copy import Copy
from .delete import Delete
from .mapmaker_templates import SolveAmplitudes
from .memory_counter import MemoryCounter
from .operator import Operator
from .pipeline import Pipeline
from .pointing import BuildPixelDistribution


@trait_docs
class MapMaker(Operator):
    r"""Operator for making maps.

    This operator first solves for a maximum likelihood set of template amplitudes
    that model the timestream contributions from noise, systematics, etc:

    .. math::
        [ M^T N^{-1} Z M + M_p ] a = M^T N^{-1} Z d

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

    A note on the PixelDistribution used for mapmaking:  If defaults are used, the
    pixel_dist specified by the binning operator will be used if it exists.  This
    PixelDistribution will be deleted if `reset_pix_dist` is specified and treated
    as if it did not exist.  If this pixel_dist does not exist, and the footprint
    options are specified, then a fixed sky footprint is used on all processes.  If
    the footprint options are not specified, then the BuildPixelDistribution operator
    will be used to pass through the data and expand all detector pointing to compute
    the distribution.  If loading data from disk, this additional pass through the
    data may be expensive!

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

    write_map = Bool(True, help="If True, write the template-subtracted final map")

    write_template_map = Bool(True, help="If True, write the template map")

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

    write_solver_amplitudes = Bool(
        False, help="If True, write out final solved template amplitudes."
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

    footprint_healpix_file = Unicode(
        None,
        allow_none=True,
        help="The healpix coverage file used to determine solver pixel distribution",
    )

    footprint_wcs_file = Unicode(
        None,
        allow_none=True,
        help="The WCS coverage file used to determine solver pixel distribution",
    )

    footprint_healpix_submap_file = Unicode(
        None,
        allow_none=True,
        help="The healpix coverage file used for the *submap* solver distribution",
    )

    footprint_healpix_nside = Int(
        None,
        allow_none=True,
        help="The healpix NSIDE for a full-sky solver pixel distribution",
    )

    footprint_healpix_submap_nside = Int(
        None,
        allow_none=True,
        help="The healpix submap NSIDE for a full-sky solver pixel distribution",
    )

    map_footprint_healpix_file = Unicode(
        None,
        allow_none=True,
        help="The healpix coverage file used to determine final pixel distribution",
    )

    map_footprint_wcs_file = Unicode(
        None,
        allow_none=True,
        help="The WCS coverage file used to determine final pixel distribution",
    )

    map_footprint_healpix_submap_file = Unicode(
        None,
        allow_none=True,
        help="The healpix coverage file used for the final *submap* distribution",
    )

    map_footprint_healpix_nside = Int(
        None,
        allow_none=True,
        help="The healpix NSIDE for a full-sky final pixel distribution",
    )

    map_footprint_healpix_submap_nside = Int(
        None,
        allow_none=True,
        help="The healpix submap NSIDE for a full-sky final pixel distribution",
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
                "write_binned_path",
            ]:
                if not bin.has_trait(trt):
                    msg = "map_binning operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return bin

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _write_del(
        self, prod_key, prod_write, force, rootname, purge=True, extra_header=None
    ):
        """Write data object to file and optionally delete it from data"""
        log = Logger.get()

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

        if not self.keep_final_products and not self.mc_mode and purge:
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

        # Which binner are we using for the final step?

        if self.map_binning is not None and self.map_binning.enabled:
            self.final_binning = self.map_binning
            self._final_is_solver = False
        else:
            self.final_binning = self.binning
            self._final_is_solver = True

        # Are we actually solving for templates?

        if (
            self.template_matrix is not None
            and self.template_matrix.enabled
            and self.template_matrix.n_enabled_templates > 0
        ):
            self._using_templates = True
        else:
            self._using_templates = False

        # Data names of outputs

        self.pixel_dist = self.binning.pixel_dist
        self.final_pixel_dist = self.final_binning.pixel_dist

        # Time domain
        self.det_flag_name = f"{self.name}_flags"
        self.clean_name = f"{self.name}_cleaned"

        # Map domain
        self.hits_name = f"{self.name}_hits"
        self.cov_name = f"{self.name}_cov"
        self.invcov_name = f"{self.name}_invcov"
        self.rcond_name = f"{self.name}_rcond"
        self.binmap_name = f"{self.name}_binmap"
        self.template_map_name = f"{self.name}_template_map"
        self.map_name = f"{self.name}_map"
        self.noiseweighted_map_name = f"{self.name}_noiseweighted_map"

        if self.reset_pix_dist or self.final_binning.pixel_dist not in data:
            # Purge any stale products from previous runs
            for name in [
                self.hits_name,
                self.cov_name,
                self.invcov_name,
                self.rcond_name,
                self.binmap_name,
                self.map_name,
                self.template_map_name,
                self.noiseweighted_map_name,
                self.binning.pixel_dist,
                self.final_binning.pixel_dist,
            ]:
                if name in self._data:
                    del self._data[name]

        # Check the options that require persistent detector data.  If the det_data
        # does not exist and a loader is being used, raise an error.

        self._using_loader = False
        for ob in data.obs:
            if self.det_data not in ob.detdata:
                if hasattr(ob, "loader"):
                    self._using_loader = True
                    if self.save_cleaned or self.overwrite_cleaned:
                        msg = f"Observation {ob.name} is loading detector data"
                        msg += " on demand.  Cannot use the `save_cleaned` or"
                        msg += " `overwrite_cleaned` options."
                        raise RuntimeError(msg)
                else:
                    msg = f"Detector data {self.det_data} does not exist in "
                    msg += f"observation {ob.name}, and no loader is present."
                    raise RuntimeError(msg)

        self._timer.start()

        return

    @function_timer
    def _create_pixel_dist(
        self,
        binner,
        fp_healpix_file,
        fp_healpix_submap_file,
        fp_healpix_nside,
        fp_healpix_submap_nside,
        fp_wcs_file,
    ):
        """Create a PixelDistribution for use by a binner."""

        if binner.pixel_dist not in self._data:
            self._log.info_rank(
                f"{self._log_prefix} Creating pixel distribution for {binner.name}",
                comm=self._comm,
            )
            # We need to create it.  See if we are using a footprint.
            if fp_healpix_file is not None:
                if fp_healpix_submap_nside is None:
                    msg = "If using a healpix footprint file, you must specify"
                    msg += " the submap NSIDE."
                    raise RuntimeError(msg)
                self._data[binner.pixel_dist] = footprint_distribution(
                    healpix_coverage_file=fp_healpix_file,
                    healpix_nside_submap=fp_healpix_submap_nside,
                    comm=self._comm,
                )
            elif fp_healpix_submap_file is not None:
                if fp_healpix_nside is None:
                    msg = "If using a healpix submap footprint file, you must specify"
                    msg += " the map NSIDE."
                    raise RuntimeError(msg)
                self._data[binner.pixel_dist] = footprint_distribution(
                    healpix_submap_file=fp_healpix_submap_file,
                    healpix_nside=fp_healpix_nside,
                    comm=self._comm,
                )
            elif fp_wcs_file is not None:
                self._data[binner.pixel_dist] = footprint_distribution(
                    wcs_coverage_file=fp_wcs_file,
                    comm=self._comm,
                )
            else:
                # We have to pass through the pointing...
                BuildPixelDistribution(
                    pixel_dist=binner.pixel_dist,
                    pixel_pointing=binner.pixel_pointing,
                    save_pointing=binner.full_pointing,
                ).apply(self._data)
            self._log.info_rank(
                f"{self._log_prefix}  finished pixel distribution for {binner.name} in",
                comm=self._comm,
                timer=self._timer,
            )

            self._memreport.prefix = f"After pixel distribution for {binner.name}"
            self._memreport.apply(self._data, use_accel=self._use_accel)
        else:
            msg = f"{self._log_prefix} Using existing pixel distribution "
            msg += f"for {binner.name}"
            self._log.info_rank(msg, comm=self._comm)

    @function_timer
    def _fit_templates(self):
        """Solve for template amplitudes"""

        print(
            f"DEBUG mapmaker {self.name} call SolveAmplitudes, cache_detdata = {self.binning.cache_detdata}",
            flush=True,
        )

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

        if self.write_solver_amplitudes:
            for tmpl in self.template_matrix.templates:
                tmpl_amps = self._data[template_amplitudes][tmpl.name]
                out_root = os.path.join(self.output_dir, f"{self._mc_root}_{tmpl.name}")
                tmpl.write(tmpl_amps, out_root)
            self._log.info_rank(
                f"{self._log_prefix}  finished template amplitude write in",
                comm=self._comm,
                timer=self._timer,
            )

        self._memreport.prefix = "After solving amplitudes"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        return template_amplitudes

    @function_timer
    def _bin_raw_map(self, extra_header):
        # This bins the original input timestream with the original flagging
        # (which may differ from the solver flags due to masks).  The original
        # data may not be persistent in memory and running this operation may
        # trigger data loading from disk for each observation.
        print("----------- Begin Binned Raw -----------", flush=True)

        # Get the units used across the distributed data for our desired
        # input detector data
        det_data_units = self._data.detector_units(self.det_data)

        # For final binning, we do not need to do any caching.
        self.final_binning.cache_dir = None
        self.final_binning.cache_detdata = False

        # Reset any pre / post processing
        self.final_binning.pre_process = None
        self.final_binning.post_process = None

        self.final_binning.det_data = self.det_data
        self.final_binning.det_data_units = det_data_units
        self.final_binning.binned = self.binmap_name
        self.final_binning.covariance = self.cov_name

        if self.write_hits:
            self.final_binning.hits = self.hits_name
        else:
            self.final_binning.hits = None

        if self.write_invcov:
            self.final_binning.inverse_covariance = self.invcov_name
        else:
            self.final_binning.inverse_covariance = None

        if self.write_noiseweighted_map:
            self.final_binning.noiseweighted = self.noiseweighted_map_name
        else:
            self.final_binning.noiseweighted = None

        if self.write_rcond:
            self.final_binning.rcond = self.rcond_name
        else:
            self.final_binning.rcond = None

        self.final_binning.apply(self._data)

        cdata = self._data[self.cov_name]
        nonz = cdata.data != 0
        print(f"Raw covariance = {cdata.data[nonz]}", flush=True)

        # Write outputs

        self._write_del(
            self.hits_name, self.write_hits, False, self.name, extra_header=extra_header
        )
        self._write_del(
            self.rcond_name,
            self.write_rcond,
            False,
            self.name,
            extra_header=extra_header,
        )
        self._write_del(
            self.invcov_name,
            self.write_invcov,
            False,
            self.name,
            extra_header=extra_header,
        )
        self._write_del(
            self.noiseweighted_map_name,
            self.write_noiseweighted_map,
            True,
            self._mc_root,
            extra_header=extra_header,
        )
        print("-----------   End Binned Raw -----------", flush=True)

    @function_timer
    def _bin_template_map(self, template_amplitudes, extra_header):
        # This bins the projected, solved templates, and produces a "template
        # map" which can be subtracted from the raw binned map to produce
        # the destriped map.  Depending on other options, the "cleaned" timestreams
        # (original - projected templates) are saved.

        print("----------- Begin Binned Templates -----------", flush=True)

        # Get the units used across the distributed data for our desired
        # input detector data
        det_data_units = self._data.detector_units(self.det_data)

        # Template projection timestreams
        projected = f"{self.name}_projected_templates"

        # Re-initialize the template matrix.  The original detector flags (which
        # may be different than the solver flags), have been restored inside the
        # SolveAmplitudes operator.
        self.template_matrix.reset()
        self.template_matrix.det_data = projected
        self.template_matrix.transpose = False
        self.template_matrix.det_data_units = det_data_units
        self.template_matrix.amplitudes = template_amplitudes
        self.template_matrix.view = self.final_binning.pixel_pointing.view
        self.template_matrix.initialize(self._data)

        pre_ops = list()
        pre_ops.append(self.template_matrix)
        do_restore = False

        if self.save_cleaned:
            # We are working with persistent detector data.
            if self.overwrite_cleaned:
                # We are replacing the original data with the template-subtracted
                # timestreams.
                pre_ops.append(
                    Combine(
                        op="subtract",
                        first=self.det_data,
                        second=projected,
                        result=self.det_data,
                    )
                )
            else:
                # We are creating a separate data object for the cleaned timestream.
                pre_ops.append(
                    Combine(
                        op="subtract",
                        first=self.det_data,
                        second=projected,
                        result=self.clean_name,
                    )
                )
        else:
            # We are not saving the cleaned time streams, so just project the templates
            # and then delete them after binning.  Save any loaders in use and restore
            # afterwards.
            saved = self._data.save_loaders()
            do_restore = True

        post_ops = list()
        post_ops.append(Delete(detdata=[projected]))

        pre_pipe = Pipeline(operators=pre_ops)
        post_pipe = Pipeline(operators=post_ops)

        self.final_binning.det_data = projected
        self.final_binning.det_data_units = det_data_units
        self.final_binning.binned = self.template_map_name
        self.final_binning.covariance = self.cov_name

        self.final_binning.hits = None
        self.final_binning.inverse_covariance = None
        self.final_binning.noiseweighted = None
        self.final_binning.rcond = None

        self.final_binning.pre_process = pre_pipe
        self.final_binning.post_process = post_pipe

        cdata = self._data[self.cov_name]
        nonz = cdata.data != 0
        print(f"Template covariance = {cdata.data[nonz]}", flush=True)

        self.final_binning.apply(self._data)

        self.final_binning.pre_process = None
        self.final_binning.post_process = None

        if do_restore:
            self._data.restore_loaders(saved)
        print("-----------   End Binned Templates -----------", flush=True)

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
        start = 1e100
        stop = -1e100
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
        print(f"DEBUG mapmaker {self.name} call _setup", flush=True)
        self._setup(data, detectors, use_accel)

        # Confirm that there is at least one valid detector
        all_local_dets = data.all_local_detectors(
            selection=detectors, flagmask=self.final_binning.det_mask
        )
        ndet = len(all_local_dets)
        if data.comm.comm_world is not None:
            ndet = data.comm.comm_world.allreduce(ndet, op=MPI.SUM)
        if ndet == 0:
            # No valid detectors, no mapmaking
            return

        # Destripe data and make maps

        extra_header = self._get_extra_header(data, detectors)

        self._memreport.prefix = "Start of mapmaking"
        self._memreport.apply(self._data, use_accel=self._use_accel)

        if self._using_templates or self._final_is_solver:
            # We need to create the pixel distribution for the solver binner
            self._create_pixel_dist(
                self.binning,
                self.footprint_healpix_file,
                self.footprint_healpix_submap_file,
                self.footprint_healpix_nside,
                self.footprint_healpix_submap_nside,
                self.footprint_wcs_file,
            )

        if self._using_templates:
            print(f"DEBUG mapmaker {self.name} call _fit_templates", flush=True)
            template_amplitudes = self._fit_templates()

        if not self._final_is_solver:
            # We have a separate binning for the final map.  Compute
            # the pixel distribution for this.
            self._create_pixel_dist(
                self.map_binning,
                self.map_footprint_healpix_file,
                self.map_footprint_healpix_submap_file,
                self.map_footprint_healpix_nside,
                self.map_footprint_healpix_submap_nside,
                self.map_footprint_wcs_file,
            )

        self._bin_raw_map(extra_header)

        if self._using_templates:
            self._bin_template_map(template_amplitudes, extra_header)

            if self.keep_final_products or self.write_map:
                # We need to compute the template-cleaned map
                self._data[self.map_name] = self._data[self.binmap_name].duplicate()
                self._data[self.map_name].data[:] -= self._data[
                    self.template_map_name
                ].data

            self._write_del(
                self.template_map_name,
                self.write_template_map,
                False,
                self.name,
                extra_header=extra_header,
            )
            self._write_del(
                self.map_name,
                self.write_map,
                True,
                self._mc_root,
                extra_header=extra_header,
            )
            self._write_del(
                self.binmap_name,
                self.write_binmap,
                False,
                self.name,
                extra_header=extra_header,
            )
        else:
            # The raw binned map is the final map.
            if self.keep_final_products or self.write_map:
                self._data[self.map_name] = self._data[self.binmap_name]
            self._write_del(
                self.map_name,
                self.write_map,
                True,
                self._mc_root,
                purge=False,
                extra_header=extra_header,
            )
            self._write_del(
                self.binmap_name,
                self.write_binmap,
                False,
                self.name,
                extra_header=extra_header,
            )

        self._write_del(
            self.cov_name, self.write_cov, False, self.name, extra_header=extra_header
        )

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
        [ M^T N^{-1} Z M + M_p ] a = M^T N^{-1} Z d

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

    mc_root = Unicode(None, allow_none=True, help="Root name for Monte Carlo products")

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
