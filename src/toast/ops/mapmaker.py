# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import traitlets

from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels_io_healpix import write_healpix_fits, write_healpix_hdf5
from ..pixels_io_wcs import write_wcs_fits
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
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        log = Logger.get()
        timer = Timer()
        log_prefix = "MapMaker"

        memreport = MemoryCounter()
        if not self.report_memory:
            memreport.enabled = False

        memreport.prefix = "Start of mapmaking"
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
            write_solver_products=self.write_solver_products,
            write_hdf5=self.write_hdf5,
            write_hdf5_serial=self.write_hdf5_serial,
            output_dir=self.output_dir,
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

        # Data names of outputs

        self.hits_name = "{}_hits".format(self.name)
        self.cov_name = "{}_cov".format(self.name)
        self.invcov_name = "{}_invcov".format(self.name)
        self.rcond_name = "{}_rcond".format(self.name)
        self.det_flag_name = "{}_flags".format(self.name)

        self.clean_name = "{}_cleaned".format(self.name)
        self.binmap_name = "{}_binmap".format(self.name)
        self.map_name = "{}_map".format(self.name)
        self.noiseweighted_map_name = "{}_noiseweighted_map".format(self.name)

        # Check map binning

        map_binning = self.map_binning
        if self.map_binning is None or not self.map_binning.enabled:
            # Use the same binning used in the solver.
            map_binning = self.binning
        map_binning.pre_process = None
        map_binning.covariance = self.cov_name

        if self.reset_pix_dist:
            if map_binning.pixel_dist in data:
                del data[map_binning.pixel_dist]
            if map_binning.covariance in data:
                # Cannot trust earlier covariance
                del data[map_binning.covariance]

        if map_binning.pixel_dist not in data:
            log.info_rank(
                f"{log_prefix} Caching pixel distribution",
                comm=comm,
            )
            pix_dist = BuildPixelDistribution(
                pixel_dist=map_binning.pixel_dist,
                pixel_pointing=map_binning.pixel_pointing,
                shared_flags=map_binning.shared_flags,
                shared_flag_mask=map_binning.shared_flag_mask,
            )
            pix_dist.apply(data)
            log.info_rank(
                f"{log_prefix}  finished build of pixel distribution in",
                comm=comm,
                timer=timer,
            )

        if map_binning.covariance not in data:
            # Construct the noise covariance, hits, and condition number
            # mask for the final binned map.

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

            final_cov.apply(data, detectors=detectors, use_accel=use_accel)

            log.info_rank(
                f"{log_prefix}  finished build of final covariance in",
                comm=comm,
                timer=timer,
            )

            memreport.prefix = "After constructing final covariance and hits"
            memreport.apply(data, use_accel=use_accel)

        if self.write_binmap:
            map_binning.det_data = self.det_data
            map_binning.binned = self.binmap_name
            map_binning.noiseweighted = None
            log.info_rank(
                f"{log_prefix} begin map binning",
                comm=comm,
            )
            map_binning.apply(data, detectors=detectors, use_accel=use_accel)
            log.info_rank(
                f"{log_prefix}  finished binning in",
                comm=comm,
                timer=timer,
            )

        if (
            self.template_matrix is None
            or self.template_matrix.n_enabled_templates == 0
        ):
            # No templates to subtract, bin the input signal
            out_cleaned = self.det_data
        else:
            # Apply (subtract) solved amplitudes.

            log.info_rank(
                f"{log_prefix} begin apply template amplitudes",
                comm=comm,
            )

            out_cleaned = self.clean_name
            if self.save_cleaned and self.overwrite_cleaned:
                # Modify data in place
                out_cleaned = None

            amplitudes_apply = ApplyAmplitudes(
                op="subtract",
                det_data=self.det_data,
                amplitudes=amplitudes_solve.amplitudes,
                template_matrix=self.template_matrix,
                output=out_cleaned,
            )
            amplitudes_apply.apply(data, detectors=detectors, use_accel=use_accel)

            log.info_rank(
                f"{log_prefix}  finished apply template amplitudes in",
                comm=comm,
                timer=timer,
            )

        if out_cleaned is None:
            map_binning.det_data = self.det_data
        else:
            map_binning.det_data = out_cleaned
        if self.write_noiseweighted_map:
            map_binning.noiseweighted = self.noiseweighted_map_name
        map_binning.binned = self.map_name

        log.info_rank(
            f"{log_prefix} begin final map binning",
            comm=comm,
        )

        # Do the final binning
        map_binning.apply(data, detectors=detectors, use_accel=use_accel)

        log.info_rank(
            f"{log_prefix}  finished final binning in",
            comm=comm,
            timer=timer,
        )

        memreport.prefix = "After binning final map"
        memreport.apply(data, use_accel=use_accel)

        # Write and delete the outputs

        if not self.save_cleaned:
            Delete(
                detdata=[
                    self.clean_name,
                ]
            ).apply(data, use_accel=use_accel)

        # FIXME:  This I/O technique assumes "known" types of pixel representations.
        # Instead, we should associate read / write functions to a particular pixel
        # class.

        is_pix_wcs = hasattr(map_binning.pixel_pointing, "wcs")
        is_hpix_nest = None
        if not is_pix_wcs:
            is_hpix_nest = map_binning.pixel_pointing.nest

        mc_root = self.name
        if self.mc_mode:
            if self.mc_root is not None:
                mc_root += f"_{self.mc_root}"
            if self.mc_index is not None:
                mc_root += f"_{self.mc_index:05d}"

        write_del = list()
        write_del.append((self.hits_name, self.write_hits, False, self.name))
        write_del.append((self.rcond_name, self.write_rcond, False, self.name))
        write_del.append(
            (self.noiseweighted_map_name, self.write_noiseweighted_map, True, mc_root)
        )
        write_del.append((self.binmap_name, self.write_binmap, True, mc_root))
        write_del.append((self.map_name, self.write_map, True, mc_root))
        write_del.append((self.invcov_name, self.write_invcov, False, self.name))
        write_del.append((self.cov_name, self.write_cov, False, self.name))
        wtimer = Timer()
        wtimer.start()
        for prod_key, prod_write, force, rootname in write_del:
            product = prod_key.replace(f"{self.name}_", "")
            if prod_write:
                if is_pix_wcs:
                    fname = os.path.join(self.output_dir, f"{rootname}_{product}.fits")
                    if self.mc_mode and not force:
                        if os.path.isfile(fname):
                            log.info_rank(f"Skipping existing file: {fname}", comm=comm)
                            continue
                    write_wcs_fits(data[prod_key], fname)
                else:
                    if self.write_hdf5:
                        # Non-standard HDF5 output
                        fname = os.path.join(
                            self.output_dir, f"{rootname}_{product}.h5"
                        )
                        if self.mc_mode and not force:
                            if os.path.isfile(fname):
                                log.info_rank(
                                    f"Skipping existing file: {fname}", comm=comm
                                )
                                continue
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
                            self.output_dir, f"{rootname}_{product}.fits"
                        )
                        if self.mc_mode and not force:
                            if os.path.isfile(fname):
                                log.info_rank(
                                    f"Skipping existing file: {fname}", comm=comm
                                )
                                continue
                        write_healpix_fits(
                            data[prod_key],
                            fname,
                            nest=is_hpix_nest,
                            report_memory=self.report_memory,
                        )
                log.info_rank(f"Wrote {fname} in", comm=comm, timer=wtimer)
            if not self.keep_final_products and not self.mc_mode:
                if prod_key in data:
                    data[prod_key].clear()
                    del data[prod_key]

            memreport.prefix = f"After writing/deleting {prod_key}"
            memreport.apply(data, use_accel=use_accel)

        log.info_rank(
            f"{log_prefix}  finished output write in",
            comm=comm,
            timer=timer,
        )

        memreport.prefix = "End of mapmaking"
        memreport.apply(data, use_accel=use_accel)

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
    """Operator for calibrating timestreams using solved templates.

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
