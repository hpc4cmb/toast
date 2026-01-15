# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import pickle
import re
from glob import glob
from time import time

import astropy.units as u
import numpy as np
import scipy.io
import scipy.sparse
import traitlets

from .._libtoast import (
    accumulate_observation_matrix,
    add_matrix,
    build_template_covariance,
    expand_matrix,
    fourier_templates,
    legendre_templates,
)
from ..mpi import MPI, get_world
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Logger
from .copy import Copy
from .delete import Delete
from .mapmaker_solve import SolverLHS, SolverRHS, solve
from .mapmaker_utils import CovarianceAndHits
from .memory_counter import MemoryCounter
from .operator import Operator
from .pipeline import Pipeline
from .pointing import BuildPixelDistribution
from .scan_map import ScanMap, ScanMask


class SparseTemplates:
    def __init__(self):
        self.starts = []
        self.stops = []
        self.names = []
        self.templates = []
        self.norms = []
        self.template_covariance = None
        self.amplitudes = None

    @property
    def ntemplate(self):
        return len(self.templates)

    def reset(self):
        self.template_covariance = None
        self.amplitudes = None

    @function_timer
    def to_dense(self, nsample):
        dense = np.zeros([self.ntemplate, nsample])
        for itemplate, (start, stop, template) in enumerate(
            zip(self.starts, self.stops, self.templates)
        ):
            dense[itemplate, start:stop] = template
        return dense

    @function_timer
    def fit(self, signal, good):
        proj = self.dot(signal * good)
        self.amplitudes = np.dot(self.template_covariance, proj)
        return

    @function_timer
    def dot(self, signal):
        proj = np.zeros(self.ntemplate)
        for itemplate, (start, stop, template) in enumerate(
            zip(self.starts, self.stops, self.templates)
        ):
            proj[itemplate] = np.dot(template, signal[start:stop])
        return proj

    @function_timer
    def subtract(self, signal):
        for itemplate, (start, stop, template) in enumerate(
            zip(self.starts, self.stops, self.templates)
        ):
            signal[start:stop] -= self.amplitudes[itemplate] * template
        return

    @function_timer
    def trim(self, template):
        first = 0
        last = len(template) - 1
        while first < last and template[first] == 0:
            first += 1
        while last > first and template[last] == 0:
            last -= 1
        self.reset()
        return first, last

    @function_timer
    def append(self, names, templates, start=0, stop=None):
        """Append new sparse template"""
        for name, template in zip(names, templates):
            first, last = self.trim(template)
            if first == last:
                continue
            self.starts.append(start + first)
            self.stops.append(start + last + 1)
            self.names.append(name)
            self.templates.append(template[first : last + 1])
            self.norms.append(1.0)
        self.reset()
        return

    @function_timer
    def mask(self, good):
        """Return a new SparseTemplates instance that complies with the
        provided mask"""
        masked = SparseTemplates()
        failed = []
        for start, stop, name, template in zip(
                self.starts, self.stops, self.names, self.templates
        ):
            nnz = np.sum(template * good[start:stop] != 0)
            if nnz > 0:
                masked.starts.append(start)
                masked.stops.append(stop)
                masked.names.append(name)
                masked.templates.append(template.copy())
                masked.norms.append(1.0)
            else:
                # The masked template is null.  Any samples that the full
                # template spans must be flagged.
                failed.append(slice(start, stop))
        masked.normalize(good)
        return masked, failed

    @function_timer
    def normalize(self, good=None):
        """Normalize templates and discard empty ones"""
        for itemplate, (start, stop, template) in enumerate(
                zip(self.starts, self.stops, self.templates)
        ):
            if good is None:
                norm = np.sum(template**2) ** 0.5
            else:
                norm = np.sum((template * good[start:stop]) ** 2) ** 0.5
            if norm == 0:
                raise RuntimeError("Zero-norm template")
            template /= norm
            self.norms[itemplate] *= norm
        self.reset()
        return

    @property
    def normalized_amplitudes(self):
        normalized = np.array(self.amplitudes) * np.array(self.norms)
        return normalized

    @function_timer
    def build_template_covariance(self, good):
        """Calculate (F^T N^-1_F F)^-1

        Observe that the sample noise weights in N^-1_F need not be the
        same as in binning the filtered signal.  For instance, samples
        falling on point sources may be masked here but included in the
        final map.
        """
        log = Logger.get()
        ntemplate = self.ntemplate
        invcov = np.zeros([ntemplate, ntemplate])
        build_template_covariance(
            self.starts,
            self.stops,
            self.templates,
            good.astype(np.float64),
            invcov,
        )
        try:
            cond = np.linalg.cond(invcov)
            if np.isinf(cond):
                rcond = 0
            else:
                rcond = 1 / cond
        except np.linalg.LinAlgError:
            log.error(
                f"Failed condition number calculation for "
                f"{ntemplate}x{ntemplate} matrix:"
            )
            log.error(f"{invcov}", flush=True)
            log.error(f"Diagonal:")
            for row in range(ntemplate):
                log.error(f"{row:03d} {invcov[row, row]}")
            raise
        log.debug(
            f"SparseTemplates: Template covariance matrix "
            f"rcond = {rcond}",
        )
        if rcond == 0:
            # No covariance for empty templates
            self.cov = None
            return
        if rcond > 1e-10:
            cov = np.linalg.inv(invcov)
        else:
            log.warning(
                f"SparseTemplates: WARNING: template covariance matrix "
                f"is poorly conditioned: "
                f"rcond = {rcond}.  Using matrix pseudoinverse.",
            )
            cov = np.linalg.pinv(invcov, rcond=1e-10, hermitian=True)

        self.template_covariance = cov
        return


def combine_observation_matrix(rootname):
    """Combine slices of the observation matrix into a single
    scipy sparse matrix file

    Args:
        rootname (str) : rootname of the matrix slices.  Typically
            `{filterbin.output_dir}/{filterbin_name}_obs_matrix`.
    Returns:
        filename_matrix (str) : Name of the composed matrix file,
            `{rootname}.npz`.
    """

    log = Logger.get()
    timer0 = Timer()
    timer0.start()
    timer = Timer()
    timer.start()

    datafiles = sorted(glob(f"{rootname}.*.*.*.data.npy"))
    if len(datafiles) == 0:
        msg = f"No files match {rootname}.*.*.*.data.npy"
        raise RuntimeError(msg)

    all_data = []
    all_indices = []
    all_indptr = [0]

    current_row = 0
    current_offset = 0
    shape = None

    log.info(f"Combining observation matrix from {len(datafiles)} input files ...")

    for datafile in datafiles:
        parts = datafile.split(".")
        row_start = int(parts[-5])
        row_stop = int(parts[-4])
        nrow_tot = int(parts[-3])
        if shape is None:
            shape = (nrow_tot, nrow_tot)
        elif shape[0] != nrow_tot:
            raise RuntimeError("Mismatch in shape")
        if current_row != row_start:
            all_indptr.append(np.zeros(row_start - current_row) + current_offset)
            current_row = row_start
        log.info(f"Loading {datafile}")
        data = np.load(datafile)
        indices = np.load(datafile.replace(".data.", ".indices.")).astype(np.int64)
        indptr = np.load(datafile.replace(".data.", ".indptr.")).astype(np.int64)
        all_data.append(data)
        all_indices.append(indices)
        indptr += current_offset
        all_indptr.append(indptr[1:])
        current_row = row_stop
        current_offset = indptr[-1]

    log.info_rank(f"Inputs loaded in", timer=timer, comm=None)

    if current_row != nrow_tot:
        all_indptr.append(np.zeros(nrow_tot - current_row) + current_offset)

    log.info("Constructing CSR matrix ...")

    all_data = np.hstack(all_data)
    all_indices = np.hstack(all_indices)
    all_indptr = np.hstack(all_indptr)
    obs_matrix = scipy.sparse.csr_matrix((all_data, all_indices, all_indptr), shape)
    if obs_matrix.nnz < 0:
        msg = f"Overflow in csr_matrix: nnz = {obs_matrix.nnz}.\n"
        raise RuntimeError(msg)

    log.info_rank(f"Constructed in", timer=timer, comm=None)

    log.info(f"Writing {rootname}.npz ...")
    scipy.sparse.save_npz(rootname, obs_matrix)
    log.info_rank(f"Wrote in", timer=timer, comm=None)

    log.info_rank(f"All done in", timer=timer0, comm=None)

    return f"{rootname}.npz"


@trait_docs
class FilterBin(Operator):
    """FilterBin builds a template matrix and projects out
    compromised modes.  It then bins the signal and optionally
    writes out the sparse observation matrix that matches the
    filtering operations.
    FilterBin supports deprojection templates.

    THIS OPERATOR ASSUMES OBSERVATIONS ARE DISTRIBUTED BY DETECTOR
    WITHIN A GROUP

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
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

    filter_flag_mask = Int(
        defaults.shared_mask_invalid,
        help="Bit mask value for flagging samples that fail filtering",
    )

    det_flag_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for detector sample flagging",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_nonscience,
        help="Bit mask value for optional telescope flagging",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle, allow_none=True, help="Observation shared key for HWP angle"
    )

    deproject_map = Unicode(
        None,
        allow_none=True,
        help="Healpix map containing the deprojection templates: "
        "intensity map and its derivatives",
    )

    deproject_nnz = Int(
        1,
        help="Number of deprojection templates to regress.  Must be less than "
        "or equal to number of columns in `deproject_map`.",
    )

    deproject_pattern = Unicode(
        ".*",
        help="Regular expression to test detector names with.  Only matching "
        "detectors will be deprojected.  Used to identify differenced TOD.",
    )

    binning = Instance(
        klass=Operator,
        allow_none=True,
        help="Binning operator for map making.",
    )

    azimuth = Unicode(
        defaults.azimuth, allow_none=True, help="Observation shared key for Azimuth"
    )

    hwp_filter_order = Int(
        None,
        allow_none=True,
        help="Order of HWP-synchronous signal filter.",
    )

    ground_filter_order = Int(
        None,
        allow_none=True,
        help="Order of a Legendre polynomial to fit as a function of azimuth.",
    )

    ground_filter_bin_width = Quantity(
        None,
        allow_none=True,
        help="Azimuthal bin width of ground filter",
    )

    split_ground_template = Bool(
        False, help="Apply a different template for left and right scans"
    )

    ground_template_expansion_order = Int(
        0, help="Taylor-expand each azimuthal bin in time"
    )

    leftright_interval = Unicode(
        defaults.throw_leftright_interval,
        help="Intervals for left-to-right scans",
    )

    rightleft_interval = Unicode(
        defaults.throw_rightleft_interval,
        help="Intervals for right-to-left scans",
    )

    poly_filter_order = Int(1, allow_none=True, help="Polynomial order")

    poly_filter_view = Unicode(
        "throw", allow_none=True, help="Intervals for polynomial filtering"
    )

    write_obs_matrix = Bool(False, help="Write the observation matrix")

    nskip = Int(
        1,
        help="Only use every n:th detector.  Useful for quick-and-dirty "
        "observation matrix calculation.",
    )

    noiseweight_obs_matrix = Bool(
        False, help="If True, observation matrix should match noise-weighted maps"
    )

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    write_binmap = Bool(False, help="If True, write the unfiltered map")

    write_map = Bool(True, help="If True, write the filtered map")

    write_noiseweighted_binmap = Bool(
        False,
        help="If True, write the noise-weighted unfiltered map",
    )

    write_noiseweighted_map = Bool(
        False,
        help="If True, write the noise-weighted filtered map",
    )

    write_hits = Bool(True, help="If True, write the hits map")

    write_cov = Bool(True, help="If True, write the white noise covariance matrices.")

    write_invcov = Bool(
        False,
        help="If True, write the inverse white noise covariance matrices.",
    )

    write_rcond = Bool(True, help="If True, write the reciprocal condition numbers.")

    keep_final_products = Bool(
        False, help="If True, keep the map domain products in data after write"
    )

    mc_mode = Bool(False, help="If True, re-use solver flags, sparse covariances, etc")

    mc_index = Int(None, allow_none=True, help="The Monte-Carlo index")

    maskfile = Unicode(
        None,
        allow_none=True,
        help="Optional processing mask",
    )

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="Cache directory for additive observation matrix products",
    )

    amplitude_dir = Unicode(
        None,
        allow_none=True,
        help="Write the template amplitudes to this directory",
    )

    rcond_threshold = Float(
        1.0e-3,
        help="Minimum value for inverse pixel condition number cut.",
    )

    deproject_map_name = "deprojection_map"

    write_hdf5 = Bool(
        False, help="If True, output maps are in HDF5 rather than FITS format."
    )

    write_hdf5_serial = Bool(
        False, help="If True, force serial HDF5 write of output maps."
    )

    reset_pix_dist = Bool(
        False,
        help="Clear any existing pixel distribution.  Useful when applying"
        "repeatedly to different data objects.",
    )

    report_memory = Bool(False, help="Report memory throughout the execution")

    precomputed_templates = Unicode(
        None,
        allow_none=True,
        help="Observation key to a dictionary of time domain templates to project out."
        " The dictionary must include a `det_to_key` dictionary which maps each "
        "detector to a key in the dictionary.  That key will then return a list of "
        "templates to fit against that detector.  See `precomputed_template_view`."
        ,
    )

    precomputed_template_view = Unicode(
        "throw",
        allow_none=True,
        help="Intervals for precomputed template filtering. See `precomputed_templates`"
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

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
        timer.start()

        memreport = MemoryCounter()
        if not self.report_memory:
            memreport.enabled = False

        memreport.prefix = "Start of mapmaking"
        memreport.apply(data)

        for trait in ("binning",):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        # Check that samples that fail filtering do not contribute to maps

        if self.filter_flag_mask & self.det_flag_mask == 0:
            msg = f"Filter flag mask does not overlap with det flag mask: "
            msg += f"{self.filter_flag_mask} & {self.det_flag_mask} = 0"
            raise RuntimeError(msg)

        if self.filter_flag_mask & self.shared_flag_mask == 0:
            msg = f"Filter flag mask does not overlap with shared mask: "
            msg += f"{self.filter_flag_mask} & {self.shared_flag_mask} = 0"
            raise RuntimeError(msg)

        if self.filter_flag_mask & self.binning.det_flag_mask == 0:
            msg = f"Filter flag mask does not overlap with det bin mask: "
            msg += f"{self.filter_flag_mask} & {self.binning.det_flag_mask} = 0"
            raise RuntimeError(msg)

        if self.filter_flag_mask & self.binning.shared_flag_mask == 0:
            msg = f"Filter flag mask does not overlap with shared bin mask: "
            msg += f"{self.filter_flag_mask} & {self.binning.shared_flag_mask} = 0"
            raise RuntimeError(msg)

        # Optionally destroy existing pixel distributions (useful if calling
        # repeatedly with different data objects)

        binning = self.binning
        if self.reset_pix_dist:
            if binning.pixel_dist in data:
                del data[binning.pixel_dist]
            if binning.covariance in data:
                # Cannot trust earlier covariance
                del data[binning.covariance]

        if binning.pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=binning.pixel_dist,
                pixel_pointing=binning.pixel_pointing,
                shared_flags=binning.shared_flags,
                shared_flag_mask=binning.shared_flag_mask,
            )
            pix_dist.apply(data)
            log.debug_rank(
                "Cached pixel distribution in", comm=data.comm.comm_world, timer=timer
            )

        self.npix = data[binning.pixel_dist].n_pix
        self.nnz = len(self.binning.stokes_weights.mode)

        self.npixtot = self.npix * self.nnz
        self.ncov = self.nnz * (self.nnz + 1) // 2

        if self.maskfile is not None:
            raise RuntimeError("Filtering mask not yet implemented")

        log.debug_rank(
            f"FilterBin:  Running with self.cache_dir = {self.cache_dir}",
            comm=data.comm.comm_world,
        )

        # Get the units used across the distributed data for our desired
        # input detector data
        self._det_data_units = data.detector_units(self.det_data)

        self._initialize_comm(data)

        extra_header = self._get_extra_header(data, detectors)

        # Filter data

        self._initialize_obs_matrix()
        log.debug_rank(
            "FilterBin: Initialized observation_matrix in",
            comm=self.comm,
            timer=timer,
        )

        self._load_deprojection_map(data)
        log.debug_rank(
            "FilterBin: Loaded deprojection map in", comm=self.comm, timer=timer
        )

        self._bin_map(data, detectors, filtered=False, extra_header=extra_header)
        log.debug_rank(
            "FilterBin: Binned unfiltered map in", comm=self.comm, timer=timer
        )

        log.debug_rank("FilterBin: Filtering signal", comm=self.comm)

        timer1 = Timer()
        timer1.start()
        timer2 = Timer()
        timer2.start()

        memreport.prefix = "Before filtering"
        memreport.apply(data)

        t1 = time()
        for iobs, obs in enumerate(data.obs):
            dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: Processing observation "
                    f"{iobs} / {len(data.obs)}",
                )

            common_templates = self._build_common_templates(obs)
            if self.shared_flags is not None:
                common_flags = obs.shared[self.shared_flags].data
            else:
                common_flags = np.zeros(obs.n_local_samples, dtype=np.uint8)

            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin:   Built common templates in "
                    f"{time() - t1:.2f} s",
                )
                t1 = time()

            memreport.prefix = "After common templates"
            memreport.apply(data)

            last_good_fit = None
            template_amplitudes = {}  # for saving

            for idet, det in enumerate(dets):
                if idet % self.nskip != 0:
                    # Only process every n:th detector
                    continue
                template_amplitudes[det] = None

                t1 = time()
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Processing detector "
                        f"# {idet + 1} / {len(dets)}",
                    )

                signal = obs.detdata[self.det_data][det]
                flags = obs.detdata[self.det_flags][det]
                # `good` is essentially the diagonal noise matrix used in
                # template regression.  All good detector samples have the
                # same noise weight and rest have zero weight.
                good_fit = np.logical_and(
                    (common_flags & self.shared_flag_mask) == 0,
                    (flags & self.det_flag_mask) == 0,
                )
                if np.sum(good_fit) == 0:
                    flags |= self.filter_flag_mask
                    continue

                deproject = (
                    self.deproject_map is not None
                    and self._deproject_pattern.match(det) is not None
                )

                if deproject or self.write_obs_matrix:
                    # We'll need pixel numbers
                    obs_data = data.select(obs_uid=obs.uid)
                    self.binning.pixel_pointing.apply(obs_data, detectors=[det])
                    pixels = obs.detdata[self.binning.pixel_pointing.pixels][det]
                    # and weights
                    self.binning.stokes_weights.apply(obs_data, detectors=[det])
                    weights = obs.detdata[self.binning.stokes_weights.weights][det]
                else:
                    pixels = None
                    weights = None

                det_templates, failed = common_templates.mask(good_fit)
                # Mask samples that cannot be filtered
                if len(failed) != 0:
                    for ind in failed:
                        flags[ind] |= self.filter_flag_mask
                    good_fit = np.logical_and(
                        (common_flags & self.shared_flag_mask) == 0,
                        (flags & self.det_flag_mask) == 0,
                    )
                    if np.sum(good_fit) == 0:
                        flags |= self.filter_flag_mask
                        continue

                if (
                    self.deproject_map is not None
                    and self._deproject_pattern.match(det) is not None
                ):
                    self._add_deprojection_templates(data, obs, pixels, det_templates)
                    # Must re-evaluate the template covariance
                    det_templates.reset()

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built deprojection "
                        f"templates in {time() - t1:.2f} s. "
                        f"ntemplate = {det_templates.ntemplate}",
                    )
                    t1 = time()

                if self.precomputed_templates is not None:
                    self._add_precomputed_templates(obs, det, det_templates)
                    # Must re-evaluate the template covariance
                    det_templates.reset()

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built precomputed "
                        f"templates in {time() - t1:.2f} s. "
                        f"ntemplate = {det_templates.ntemplate}",
                    )
                    t1 = time()

                # Deprojection templates or precomputed templates may also
                # need to be discarded due to flagging

                det_templates, failed = det_templates.mask(good_fit)

                # Mask samples that cannot be filtered

                if len(failed) != 0:
                    for ind in failed:
                        flags[ind] |= self.filter_flag_mask
                    good_fit = np.logical_and(
                        (common_flags & self.shared_flag_mask) == 0,
                        (flags & self.det_flag_mask) == 0,
                    )
                    if np.sum(good_fit) == 0:
                        flags |= self.filter_flag_mask
                        continue

                # Find all samples that remain good after filtering cuts

                good_bin = np.logical_and(
                    (common_flags & self.binning.shared_flag_mask) == 0,
                    (flags & self.binning.det_flag_mask) == 0,
                )

                if det_templates.ntemplate == 0:
                    # No templates to fit
                    continue

                # memreport.prefix = "After detector templates"
                # memreport.apply(data)

                if (
                    det_templates.template_covariance is None
                    or np.any(last_good_fit != good_fit)
                ):
                    det_templates.build_template_covariance(good_fit)
                    last_good_fit = good_fit.copy()

                if self.grank == 0:
                    if det_templates.template_covariance is None:
                        shape = None
                    else:
                        shape = det_templates.template_covariance.shape
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built "
                        f"{shape} template covariance "
                        f"in {time() - t1:.2f} s",
                    )
                    t1 = time()

                if det_templates.template_covariance is None:
                    # template covariance failed to invert. Flag detector data
                    flags |= self.filter_flag_mask
                else:
                    self._regress_templates(
                        det_templates, signal, good_fit
                    )
                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Regressed templates in "
                            f"{time() - t1:.2f} s",
                        )
                        t1 = time()

                    template_amplitudes[det] = dict(zip(
                        det_templates.names, det_templates.normalized_amplitudes
                    ))

                    if self.write_obs_matrix:
                        self._accumulate_observation_matrix(
                            obs,
                            det,
                            pixels,
                            weights,
                            good_fit,
                            good_bin,
                            det_templates,
                            det_templates.template_covariance,
                        )

            if self.amplitude_dir is not None:
                # Collect and save the template amplitudes
                fname_amp = os.path.join(
                    self.amplitude_dir, f"{self.name}_amplitudes_{obs.name}.pck"
                )
                if self.gcomm is None:
                    all_amplitudes = [template_amplitudes]
                else:
                    all_amplitudes = self.gcomm.gather(template_amplitudes)
                if self.grank == 0:
                    os.makedirs(self.amplitude_dir, exist_ok=True)
                    # delete the first amplitudes, they are already in
                    # the local dictionary
                    del all_amplitudes[0]
                    # Put the received amplitudes into the local dictionary
                    for amplitudes in all_amplitudes:
                        template_amplitudes.update(amplitudes)
                    with open(fname_amp, "wb") as f:
                        pickle.dump(template_amplitudes, f)
                    log.info(f"Saved template amplitudes to {fname_amp}")

        log.debug_rank(
            f"{self.group:4} : FilterBin:   Filtered group data in",
            comm=self.gcomm,
            timer=timer1,
        )

        if self.comm is not None:
            self.comm.Barrier()

        log.info_rank(
            f"FilterBin:   Filtered data in",
            comm=self.comm,
            timer=timer2,
        )

        memreport.prefix = "After filtering"
        memreport.apply(data)

        # Bin filtered signal

        self._bin_map(data, detectors, filtered=True, extra_header=extra_header)
        log.debug_rank("FilterBin: Binned filtered map in", comm=self.comm, timer=timer)

        log.info_rank(
            f"FilterBin:   Binned data in",
            comm=self.comm,
            timer=timer2,
        )

        memreport.prefix = "After binning"
        memreport.apply(data)

        if self.write_obs_matrix:
            if not self.noiseweight_obs_matrix:
                log.debug_rank(
                    "FilterBin: De-weighting observation matrix", comm=self.comm
                )
                self._deweight_obs_matrix(data)
                log.debug_rank(
                    "FilterBin: De-weighted observation_matrix in",
                    comm=self.comm,
                    timer=timer2,
                )

            log.info_rank("FilterBin: Collecting observation matrix", comm=self.comm)
            self._collect_obs_matrix()
            log.info_rank(
                "FilterBin: Collected observation_matrix in",
                comm=self.comm,
                timer=timer2,
            )

            memreport.prefix = "After observation matrix"
            memreport.apply(data)

        return

    @function_timer
    def _add_hwp_templates(self, obs, templates):
        if self.hwp_filter_order is None:
            return

        if self.hwp_angle not in obs.shared:
            msg = (
                f"Cannot apply HWP filtering at order = {self.hwp_filter_order}: "
                f"no HWP angle found under key = '{self.hwp_angle}'"
            )
            raise RuntimeError(msg)
        hwp_angle = obs.shared[self.hwp_angle].data
        shared_flags = np.array(obs.shared[self.shared_flags])

        nfilter = 2 * self.hwp_filter_order
        if nfilter < 1:
            return

        ftemplates = np.zeros([nfilter, hwp_angle.size])
        fourier_templates(hwp_angle, ftemplates, 1, self.hwp_filter_order + 1)

        names = []
        for ifilter in range(nfilter):
            order = ifilter // 2 + 1
            if ifilter % 2 == 0:
                names.append(f"HWPSS-cos-{order}")
            else:
                names.append(f"HWPSS-sin-{order}")

        templates.append(names, ftemplates)

        return

    @function_timer
    def _add_ground_poly_templates(self, obs, templates):
        if self.ground_filter_order is None:
            return

        # To avoid template degeneracies, ground filter only includes
        # polynomial orders not present in the polynomial filter

        phase = self._get_phase(obs)
        shared_flags = np.array(obs.shared[self.shared_flags])

        min_order = 0
        if self.poly_filter_order is not None:
            min_order = self.poly_filter_order + 1
        max_order = self.ground_filter_order
        nfilter = max_order - min_order + 1
        if nfilter < 1:
            return

        directionless_templates = np.zeros([nfilter, phase.size])
        legendre_templates(phase, directionless_templates, min_order, max_order + 1)
        directionless_names = []
        for order in range(min_order, max_order + 1):
            directionless_names.append(f"ground-poly-{order}")

        if not self.split_ground_template:
            legendre_filter = directionless_templates
            names = directionless_names
        else:
            # Separate ground filter by scan direction.
            legendre_filter = []
            names = []
            masks = []
            directions = []
            for name in self.leftright_interval, self.rightleft_interval:
                mask = np.zeros(phase.size, dtype=bool)
                for ival in obs.intervals[name]:
                    mask[ival.first : ival.last] = True
                masks.append(mask)
                directions.append(name)
            for name, template in zip(directionless_names, directionless_templates):
                for direction, mask in zip(directions, masks):
                    temp = template.copy()
                    temp[mask] = 0
                    legendre_filter.append(temp)
                    names.append(f"{name}-{direction}")
            legendre_filter = np.vstack(legendre_filter)

        templates.append(names, legendre_filter)

        return

    @function_timer
    def _add_ground_bin_templates(self, ob, templates):
        if self.ground_filter_bin_width is None:
            return

        if self.azimuth is not None:
            az = ob.shared[self.azimuth]
        else:
            quats = ob.shared[self.boresight_azel]
            theta, phi, _ = qa.to_iso_angles(quats)
            az = 2 * np.pi - phi

        # Make sure azimuth is continuous and positive
        az = np.unwrap(az)
        while np.amin(az) < 0:
            az += 2 * np.pi

        # Assign each time stamp to an azimuthal bin
        wbin = self.ground_filter_bin_width.to_value(u.radian)
        ibin = (az // wbin).astype(int)

        # bin numbers are positive by construction.
        # Assign flagged samples to bin = -1
        shared_flags = np.array(ob.shared[self.shared_flags])
        bad = (shared_flags & self.shared_flag_mask) != 0
        ibin[bad] = -1

        # Find the set of hit azimuthal bins
        bins, counts = np.unique(ibin, return_counts=True)
        good = bins >= 0
        bins = bins[good]
        counts = counts[good]
        nhit = len(bins)

        # Discard one bin.  This makes the rest of the templates
        # relative to it and breaks degeneracy with polynomial templates
        cut = np.argmax(counts)
        good = np.ones(len(counts), dtype=bool)
        good[cut] = False
        bins = bins[good]
        counts = counts[good]

        # Each template is just a boolean mask that is true when
        # boresight is in a specific bin
        directionless_templates = []
        directionless_names = []
        for bin_ in bins:
            directionless_templates.append((ibin == bin_).astype(float))
            bin_center = np.degrees((bin_ + 0.5) * wbin)
            directionless_names.append(f"ground-bin-at-{bin_center:.3f}")

        # Optionally separate ground filter by scan direction.
        if not self.split_ground_template:
            ground_templates = directionless_templates
            names = directionless_names
        else:
            ground_templates = []
            names = []
            masks = []
            directions = []
            for name in self.leftright_interval, self.rightleft_interval:
                mask = np.zeros(ibin.size, dtype=bool)
                for ival in ob.intervals[name]:
                    mask[ival.first : ival.last] = True
                masks.append(mask)
                directions.append(name)
            for name, template in zip(directionless_names, directionless_templates):
                for direction, mask in zip(directions, masks):
                    temp = template.copy()
                    temp[mask] = 0
                    ground_templates.append(temp)
                    names.append(f"{name}-{direction}")

        # Optionally add time derivatives of each bin temperature
        norder = self.ground_template_expansion_order
        if norder > 0:
            times = ob.shared[self.times].data
            times = times - times[0]
            times = times / times[-1] * 2 - 1
            new_templates = []
            new_names = []
            for name, template in zip(names, ground_templates):
                for order in range(norder + 1):
                    derivative = template * times**order
                    new_templates.append(derivative)
                    new_names.append(f"{name}-timederiv-{order}")
            ground_templates = new_templates
            names = new_names

        ground_templates = np.vstack(ground_templates)

        templates.append(names, ground_templates)

        return

    @function_timer
    def _add_poly_templates(self, obs, templates):
        if self.poly_filter_order is None:
            return
        nfilter = self.poly_filter_order + 1
        intervals = obs.intervals[self.poly_filter_view]
        if self.shared_flags is None:
            shared_flags = np.zeros(obs.n_local_samples, dtype=np.uint8)
        else:
            shared_flags = np.array(obs.shared[self.shared_flags])
        bad = (shared_flags & self.shared_flag_mask) != 0

        for i, ival in enumerate(intervals):
            istart = ival.first
            istop = ival.last
            # Trim flagged samples from both ends
            while istart < istop and bad[istart]:
                istart += 1
            while istop - 1 > istart and bad[istop - 1]:
                istop -= 1
            if istop - istart < nfilter:
                # Not enough samples to filter, flag this interval
                shared_flags[ival.first : ival.last] |= self.filter_flag_mask
                continue
            wbin = 2 / (istop - istart)
            phase = (np.arange(istop - istart) + 0.5) * wbin - 1
            ltemplates = np.zeros([nfilter, phase.size])
            legendre_templates(phase, ltemplates, 0, nfilter)
            names = []
            for order in range(nfilter):
                names.append(f"poly-{order}-interval-{i}")
            templates.append(names, ltemplates, start=istart, stop=istop)

        if self.shared_flags is not None:
            obs.shared[self.shared_flags].set(shared_flags, offset=(0,), fromrank=0)

        return

    @function_timer
    def _build_common_templates(self, obs):
        templates = SparseTemplates()

        self._add_hwp_templates(obs, templates)
        self._add_ground_poly_templates(obs, templates)
        self._add_ground_bin_templates(obs, templates)
        self._add_poly_templates(obs, templates)

        return templates

    @function_timer
    def _add_deprojection_templates(self, data, obs, pixels, templates):
        deproject_map = data[self.deproject_map_name]
        map_dist = deproject_map.distribution
        local_sm, local_pix = map_dist.global_pixel_to_submap(pixels)

        if deproject_map.dtype.char == "d":
            scan_map = scan_map_float64
        elif deproject_map.dtype.char == "f":
            scan_map = scan_map_float32
        else:
            raise RuntimeError("Deprojection supports only float32 and float64 maps")

        nsample = pixels.size
        nnz = self._deproject_nnz
        weights = np.zeros([nsample, nnz], dtype=np.float64)
        dptemplate_raw = AlignedF64.zeros(nsample)
        dptemplate = dptemplate_raw.array()
        for inz in range(self._deproject_nnz):
            weights[:] = 0
            weights[:, inz] = 1
            scan_map(
                deproject_map.distribution.n_pix_submap,
                deproject_map.n_value,
                local_sm.astype(np.int64),
                local_pix.astype(np.int64),
                deproject_map.raw,
                weights.reshape(-1),
                template,
            )
            name = f"deproject-{inz}"
            templates.append(name, dptemplate)
        return

    @function_timer
    def _add_precomputed_templates(self, obs, det, templates):
        if self.precomputed_templates not in obs:
            raise RuntimeError()

        precomputed = obs[self.precomputed_templates]
        if det not in precomputed["det_to_key"]:
            # This detector does not have precomputed templates
            return

        intervals = obs.intervals[self.precomputed_template_view]
        key = precomputed["det_to_key"][det]
        tod_templates = precomputed[key]
        for i, ival in enumerate(intervals):
            istart = ival.first
            istop = ival.last
            ind = slice(istart, istop)
            slice_templates = []
            names = []
            for name, tod_template in tod_templates.items():
                slice_templates.append(tod_template[ind])
                names.append(f"{name}-interval-{i}")
            slice_templates = np.vstack(slice_templates)
            templates.append(names, slice_templates, start=istart, stop=istop)

        return

    @function_timer
    def _regress_templates(self, templates, signal, good):
        """Calculate Zd = (I - F(F^T N^-1_F F)^-1 F^T N^-1_F)d

        All samples that are not flagged (zero weight in N^-1_F) have
        equal weight.
        """
        templates.fit(signal, good)
        templates.subtract(signal)
        return

    @function_timer
    def _compress_pixels(self, pixels):
        if any(pixels < 0):
            msg = f"Unflagged samples have {np.sum(pixels < 0)} negative pixel numbers"
            raise RuntimeError(msg)
        if any(pixels >= self.npix):
            msg = f"Unflagged samples have {np.sum(pixels >= self.npix)} pixels >= {self.npix}"
            raise RuntimeError(msg)
        local_to_global = np.sort(list(set(pixels)))
        compressed_pixels = np.searchsorted(local_to_global, pixels)
        return compressed_pixels, local_to_global.size, local_to_global

    @function_timer
    def _add_matrix(self, local_obs_matrix, detweight):
        """Add the local (per detector) observation matrix to the full
        matrix
        """
        log = Logger.get()
        t1 = time()
        if False:
            # Use scipy sparse implementation
            self.obs_matrix += local_obs_matrix * detweight
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: Add and construct matrix "
                    f"in {time() - t1:.2f} s",
                )
        else:
            # Use our own compiled kernel
            n = self.obs_matrix.nnz + local_obs_matrix.nnz
            data = np.zeros(n, dtype=np.float64)
            indices = np.zeros(n, dtype=np.int64)
            indptr = np.zeros(self.npixtot + 1, dtype=np.int64)
            add_matrix(
                self.obs_matrix.data,
                self.obs_matrix.indices,
                self.obs_matrix.indptr,
                local_obs_matrix.data * detweight,
                local_obs_matrix.indices,
                local_obs_matrix.indptr,
                data,
                indices,
                indptr,
            )
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: Add matrix in {time() - t1:.2f} s",
                )
            t1 = time()
            n = indptr[-1]
            self.obs_matrix = scipy.sparse.csr_matrix(
                (data[:n], indices[:n], indptr),
                shape=(self.npixtot, self.npixtot),
            )
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: construct CSR matrix "
                    f"in {time() - t1:.2f} s",
                )
        return

    def _expand_matrix(self, compressed_matrix, local_to_global):
        """Expands a dense, compressed matrix into a sparse matrix with
        global indexing
        """
        n = compressed_matrix.size
        values = np.zeros(n, dtype=np.float64)
        indices = np.zeros(n, dtype=np.int64)
        indptr = np.zeros(self.npixtot + 1, dtype=np.int64)
        expand_matrix(
            compressed_matrix,
            local_to_global,
            self.npix,
            self.nnz,
            values,
            indices,
            indptr,
        )
        nnz = indptr[-1]

        sparse_matrix = scipy.sparse.csr_matrix(
            (values[:nnz], indices[:nnz], indptr),
            shape=(self.npixtot, self.npixtot),
        )
        return sparse_matrix

    @function_timer
    def _accumulate_observation_matrix(
        self,
        obs,
        det,
        pixels,
        weights,
        good_fit,
        good_bin,
        det_templates,
        template_covariance,
    ):
        """Calculate P^T N^-1 Z P
        This part of the covariance calculation is cumulative: each observation
        and detector is computed independently and can be cached.

        Observe that `N` in this equation need not be the same used in
        template covariance in `Z`.
        """
        if not self.write_obs_matrix:
            return
        log = Logger.get()
        templates = det_templates.to_dense(good_fit.size)
        fname_cache = None
        local_obs_matrix = None
        t1 = time()
        if self.cache_dir is not None:
            cache_dir = os.path.join(self.cache_dir, obs.name)
            os.makedirs(cache_dir, exist_ok=True)
            fname_cache = os.path.join(cache_dir, det)
            try:
                mm_data = np.load(fname_cache + ".data.npy")
                mm_indices = np.load(fname_cache + ".indices.npy")
                mm_indptr = np.load(fname_cache + ".indptr.npy")
                local_obs_matrix = scipy.sparse.csr_matrix(
                    (mm_data, mm_indices, mm_indptr),
                    self.obs_matrix.shape,
                )
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:     loaded cached matrix from "
                        f"{fname_cache}* in {time() - t1:.2f} s",
                    )
                    t1 = time()
            except:
                local_obs_matrix = None
        else:
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin:     cache_dir = {self.cache_dir}"
                )

        if local_obs_matrix is None:
            templates = templates.T.copy()
            good_any = np.logical_or(good_fit, good_bin)

            # Temporarily compress pixels
            if self.grank == 0:
                log.debug(f"{self.group:4} : FilterBin:     Compressing pixels")
            c_pixels, c_npix, local_to_global = self._compress_pixels(
                pixels[good_any].copy()
            )
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: Compressed in {time() - t1:.2f} s",
                )
                t1 = time()
            c_npixtot = c_npix * self.nnz
            c_obs_matrix = np.zeros([c_npixtot, c_npixtot])
            if self.grank == 0:
                log.debug(f"{self.group:4} : FilterBin:     Accumulating")
            w = weights[good_any].copy()
            if len(w.shape) == 1:
                # Only one Stokes component. We lost an array dimension
                # along the way and our compiled kernel would be unhappy
                w = w[:, np.newaxis]
            accumulate_observation_matrix(
                c_obs_matrix,
                c_pixels,
                w,
                templates[good_any].copy(),
                template_covariance,
                good_fit[good_any].astype(np.uint8),
                good_bin[good_any].astype(np.uint8),
            )
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin:     Accumulated in {time() - t1:.2f} s"
                )
                log.debug(
                    f"{self.group:4} : FilterBin:     Expanding local to global",
                )
                t1 = time()
            # expand to global pixel numbers
            local_obs_matrix = self._expand_matrix(c_obs_matrix, local_to_global)
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin:     Expanded in {time() - t1:.2f} s"
                )
                t1 = time()

            if fname_cache is not None:
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:     Caching to {fname_cache}*",
                    )
                np.save(fname_cache + ".data", local_obs_matrix.data)
                np.save(fname_cache + ".indices", local_obs_matrix.indices)
                np.save(fname_cache + ".indptr", local_obs_matrix.indptr)
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:     cached in {time() - t1:.2f} s",
                    )
                    t1 = time()
            else:
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:     NOT caching detector matrix",
                    )

        if self.grank == 0:
            log.debug(f"{self.group:4} : FilterBin:     Adding to global")
        detweight = obs[self.binning.noise_model].detector_weight(det)
        self._add_matrix(local_obs_matrix, detweight)
        if self.grank == 0:
            log.debug(
                f"{self.group:4} : FilterBin:     Added in {time() - t1:.2f} s",
            )
        return

    @function_timer
    def _get_phase(self, obs):
        if self.ground_filter_order is None:
            return None
        try:
            azmin = obs["scan_min_az"].to_value(u.radian)
            azmax = obs["scan_max_az"].to_value(u.radian)
            if self.azimuth is not None:
                az = obs.shared[self.azimuth]
            else:
                quats = obs.shared[self.boresight_azel]
                theta, phi, _ = qa.to_iso_angles(quats)
                az = 2 * np.pi - phi
        except Exception as e:
            msg = (
                f"Failed to get boresight azimuth from TOD.  "
                f"Perhaps it is not ground TOD? '{e}'"
            )
            raise RuntimeError(msg)
        phase = (np.unwrap(az) - azmin) / (azmax - azmin) * 2 - 1

        return phase

    @function_timer
    def _initialize_comm(self, data):
        """Create convenience aliases to the communicators and properties."""
        self.comm = data.comm.comm_world
        self.rank = data.comm.world_rank
        self.ntask = data.comm.world_size
        self.gcomm = data.comm.comm_group
        self.group = data.comm.group
        self.grank = data.comm.group_rank
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
                ob.select_local_detectors(detectors, flagmask=self.det_mask)
            )
        if self.comm is not None:
            start = self.comm.allreduce(start, op=MPI.MIN)
            stop = self.comm.allreduce(stop, op=MPI.MAX)
            all_dets_list = self.comm.allgather(all_dets)
            good_dets_list = self.comm.allgather(good_dets)
            all_dets.update(*all_dets_list)
            good_dets.update(*good_dets_list)
        extra_header["START"] = (start, "Dataset start time")
        extra_header["STOP"] = (stop, "Dataset stop time")
        extra_header["NDET"] = (len(all_dets), "Total number of detectors")
        extra_header["NGOOD"] = (len(good_dets), "Total number of usable detectors")
        extra_header["OPERATOR"] = ("TOAST FilterBin", "Generating code")

        return extra_header

    @function_timer
    def _initialize_obs_matrix(self):
        if self.write_obs_matrix:
            self.obs_matrix = scipy.sparse.csr_matrix(
                (self.npixtot, self.npixtot), dtype=np.float64
            )
            if self.rank == 0 and self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.obs_matrix = None
        return

    @function_timer
    def _deweight_obs_matrix(self, data):
        """Apply (P^T N^-1 P)^-1 to the cumulative part of the
        observation matrix, P^T N^-1 Z P.
        """
        if not self.write_obs_matrix:
            return
        # Apply the white noise covariance to the observation matrix
        white_noise_cov = data[self.binning.covariance]
        cc = scipy.sparse.dok_matrix((self.npixtot, self.npixtot), dtype=np.float64)
        nsubmap = white_noise_cov.distribution.n_submap
        npix_submap = white_noise_cov.distribution.n_pix_submap
        for isubmap_local, isubmap_global in enumerate(
            white_noise_cov.distribution.local_submaps
        ):
            submap = white_noise_cov.data[isubmap_local]
            offset = isubmap_global * npix_submap
            for pix_local in range(npix_submap):
                if np.all(submap[pix_local] == 0):
                    continue
                pix = pix_local + offset
                icov = 0
                for inz in range(self.nnz):
                    for jnz in range(inz, self.nnz):
                        cc[pix + inz * self.npix, pix + jnz * self.npix] = submap[
                            pix_local, icov
                        ]
                        if inz != jnz:
                            cc[pix + jnz * self.npix, pix + inz * self.npix] = submap[
                                pix_local, icov
                            ]
                        icov += 1
        cc = cc.tocsr()
        self.obs_matrix = cc.dot(self.obs_matrix)
        return

    @function_timer
    def _collect_obs_matrix(self):
        if not self.write_obs_matrix:
            return
        # Combine the observation matrix across processes
        # Reduce the observation matrices.  We use the buffer protocol
        # for better performance, even though it requires more MPI calls
        # than sending the sparse matrix objects directly
        log = Logger.get()
        timer = Timer()
        timer.start()
        nrow_tot = self.npixtot
        nslice = 128
        nrow_write = nrow_tot // nslice
        for islice, row_start in enumerate(range(0, nrow_tot, nrow_write)):
            row_stop = row_start + nrow_write
            obs_matrix_slice = self.obs_matrix[row_start:row_stop]
            nnz = obs_matrix_slice.nnz
            if self.comm is not None:
                nnz = self.comm.allreduce(nnz)
            if nnz == 0:
                log.debug_rank(
                    f"Slice {islice + 1:5} / {nslice}: {row_start:12} - {row_stop:12} "
                    f"is empty.  Skipping.",
                    comm=self.comm,
                )
                continue
            log.debug_rank(
                f"Collecting slice {islice + 1:5} / {nslice} : {row_start:12} - "
                f"{row_stop:12}",
                comm=self.comm,
            )

            factor = 1
            while factor < self.ntask:
                log.debug_rank(
                    f"FilterBin: Collecting {2 * factor} / {self.ntask}",
                    comm=self.comm,
                )
                if self.rank % (factor * 2) == 0:
                    # this task receives
                    receive_from = self.rank + factor
                    if receive_from < self.ntask:
                        size_recv = self.comm.recv(source=receive_from, tag=factor)
                        data_recv = np.zeros(size_recv, dtype=np.float64)
                        self.comm.Recv(
                            data_recv, source=receive_from, tag=factor + self.ntask
                        )
                        indices_recv = np.zeros(size_recv, dtype=np.int64)
                        self.comm.Recv(
                            indices_recv,
                            source=receive_from,
                            tag=factor + 2 * self.ntask,
                        )
                        indptr_recv = np.zeros(
                            obs_matrix_slice.indptr.size, dtype=np.int64
                        )
                        self.comm.Recv(
                            indptr_recv,
                            source=receive_from,
                            tag=factor + 3 * self.ntask,
                        )
                        obs_matrix_slice += scipy.sparse.csr_matrix(
                            (data_recv, indices_recv, indptr_recv),
                            obs_matrix_slice.shape,
                        )
                        del data_recv, indices_recv, indptr_recv
                elif self.rank % (factor * 2) == factor:
                    # this task sends
                    send_to = self.rank - factor
                    self.comm.send(obs_matrix_slice.data.size, dest=send_to, tag=factor)
                    self.comm.Send(
                        obs_matrix_slice.data, dest=send_to, tag=factor + self.ntask
                    )
                    self.comm.Send(
                        obs_matrix_slice.indices.astype(np.int64),
                        dest=send_to,
                        tag=factor + 2 * self.ntask,
                    )
                    self.comm.Send(
                        obs_matrix_slice.indptr.astype(np.int64),
                        dest=send_to,
                        tag=factor + 3 * self.ntask,
                    )

                if self.comm is not None:
                    self.comm.Barrier()
                log.debug_rank("FilterBin: Collected in", comm=self.comm, timer=timer)
                factor *= 2

            # Write out the observation matrix
            if self.noiseweight_obs_matrix:
                fname = os.path.join(
                    self.output_dir, f"{self.name}_noiseweighted_obs_matrix"
                )
            else:
                fname = os.path.join(self.output_dir, f"{self.name}_obs_matrix")
            fname += f".{row_start:012}.{row_stop:012}.{nrow_tot:012}"
            log.debug_rank(
                f"FilterBin: Writing observation matrix to {fname}.npz",
                comm=self.comm,
            )
            if self.rank == 0:
                if True:
                    # Write out the members of the CSR matrix separately because
                    # scipy.sparse.save_npz is so inefficient
                    np.save(f"{fname}.data", obs_matrix_slice.data)
                    np.save(f"{fname}.indices", obs_matrix_slice.indices)
                    np.save(f"{fname}.indptr", obs_matrix_slice.indptr)
                else:
                    scipy.sparse.save_npz(fname, obs_matrix_slice)
            log.info_rank(
                f"FilterBin: Wrote observation matrix to {fname} in",
                comm=self.comm,
                timer=timer,
            )
        # After writing we are done
        del self.obs_matrix
        self.obs_matrix = None
        return

    @function_timer
    def _bin_map(self, data, detectors, filtered, extra_header=None):
        """Bin the signal onto a map.  Optionally write out hits and
        white noise covariance matrices.
        """

        log = Logger.get()
        timer = Timer()
        timer.start()

        hits_name = f"{self.name}_hits"
        invcov_name = f"{self.name}_invcov"
        cov_name = f"{self.name}_cov"
        rcond_name = f"{self.name}_rcond"
        if filtered:
            map_name = f"{self.name}_filtered_map"
            noiseweighted_map_name = f"{self.name}_noiseweighted_filtered_map"
        else:
            map_name = f"{self.name}_unfiltered_map"
            noiseweighted_map_name = f"{self.name}_noiseweighted_unfiltered_map"

        self.binning.noiseweighted = noiseweighted_map_name
        self.binning.binned = map_name
        self.binning.det_data = self.det_data
        self.binning.det_data_units = self._det_data_units
        self.binning.covariance = cov_name

        if self.binning.covariance not in data:
            cov = CovarianceAndHits(
                pixel_dist=self.binning.pixel_dist,
                covariance=self.binning.covariance,
                inverse_covariance=invcov_name,
                hits=hits_name,
                rcond=rcond_name,
                det_mask=self.binning.det_mask,
                det_flags=self.binning.det_flags,
                det_flag_mask=self.binning.det_flag_mask,
                det_data_units=self._det_data_units,
                shared_flags=self.binning.shared_flags,
                shared_flag_mask=self.binning.shared_flag_mask,
                pixel_pointing=self.binning.pixel_pointing,
                stokes_weights=self.binning.stokes_weights,
                noise_model=self.binning.noise_model,
                rcond_threshold=self.rcond_threshold,
                sync_type=self.binning.sync_type,
                save_pointing=self.binning.full_pointing,
            )
            cov.apply(data, detectors=detectors)
            log.info_rank(f"Binned covariance and hits in", comm=self.comm, timer=timer)

        self.binning.apply(data, detectors=detectors)
        log.info_rank(f"Binned signal in", comm=self.comm, timer=timer)

        mc_root = self.name
        if self.mc_mode:
            if self.mc_root is not None:
                mc_root += f"_{self.mc_root}"
            if self.mc_index is not None:
                mc_root += f"_{self.mc_index:05d}"

        binned = not filtered  # only write hits and covariance once
        if binned:
            write_map = self.write_binmap
            write_noiseweighted_map = self.write_noiseweighted_binmap
        else:
            write_map = self.write_map
            write_noiseweighted_map = self.write_noiseweighted_map
        keep_final = self.keep_final_products
        keep_cov = self.keep_final_products or self.write_obs_matrix
        for key, write, keep, force, rootname in [
            (hits_name, self.write_hits and binned, keep_final, False, self.name),
            (rcond_name, self.write_rcond and binned, keep_final, False, self.name),
            (
                noiseweighted_map_name,
                write_noiseweighted_map,
                keep_final,
                True,
                mc_root,
            ),
            (map_name, write_map, keep_final, True, mc_root),
            (invcov_name, self.write_invcov and binned, keep_final, False, self.name),
            (cov_name, self.write_cov and binned, keep_cov, False, self.name),
        ]:
            if write:
                product = key.replace(f"{self.name}_", "")
                if self.write_hdf5:
                    fname_suffix = "h5"
                else:
                    fname_suffix = "fits"
                fname = os.path.join(
                    self.output_dir, f"{rootname}_{product}.{fname_suffix}"
                )
                if self.mc_mode and not force:
                    if os.path.isfile(fname):
                        log.info_rank(
                            f"Skipping existing file: {fname}", comm=self.comm
                        )
                        continue
                data[key].write(
                    fname,
                    force_serial=self.write_hdf5_serial,
                    extra_header=extra_header,
                )
                log.info_rank(f"Wrote {fname} in", comm=self.comm, timer=timer)
            if not keep and not self.mc_mode:
                if key in data:
                    data[key].clear()
                    del data[key]

    @function_timer
    def _load_deprojection_map(self, data):
        if self.deproject_map is None:
            return None
        data[self.deproject_map_name] = PixelData(
            data[self.binning.pixel_dist],
            dtype=np.float32,
            n_value=self.deproject_nnz,
            units=self._det_data_units,
        )
        data[self.deproject_map_name].read(self.deproject_map)
        self._deproject_pattern = re.compile(self.deproject_pattern)
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        # This operator requires everything that its sub-operators needs.
        req = self.binning.requires()
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        prov = dict()
        prov["global"] = [self.binning.binned]
        return prov
