# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
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
    build_template_covariance,
    expand_matrix,
    legendre,
    fourier,
)
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import (
    filename_is_fits,
    filename_is_hdf5,
    read_healpix_fits,
    read_healpix_hdf5,
    write_healpix_fits,
    write_healpix_hdf5,
)
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Unicode, trait_docs
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
        self.templates = []

    @property
    def ntemplate(self):
        return len(self.templates)

    def to_dense(self, nsample):
        dense = np.zeros([self.ntemplate, nsample])
        for itemplate, (start, stop, template) in enumerate(
            zip(self.starts, self.stops, self.templates)
        ):
            dense[itemplate, start:stop] = template
        return dense

    def dot(self, signal):
        proj = np.zeros(self.ntemplate)
        for itemplate, (start, stop, template) in enumerate(
            zip(self.starts, self.stops, self.templates)
        ):
            proj[itemplate] = np.dot(template, signal[start:stop])
        return proj

    def subtract(self, signal, amplitudes):
        for itemplate, (start, stop, template) in enumerate(
            zip(self.starts, self.stops, self.templates)
        ):
            signal[start:stop] -= amplitudes[itemplate] * template
        return

    def trim(self, template):
        first = 0
        last = len(template) - 1
        while first < last and template[first] == 0:
            first += 1
        while last > first and template[last] == 0:
            last -= 1
        return first, last

    def append(self, templates, start=0, stop=None):
        """Append new sparse template"""
        for template in templates:
            first, last = self.trim(template)
            if first == last:
                continue
            self.starts.append(start + first)
            self.stops.append(start + last + 1)
            self.templates.append(template[first : last + 1])
        return

    def mask(self, good):
        """Return a new SparseTemplates instance that complies with the
        provided mask"""
        masked = SparseTemplates()
        for start, stop, template in zip(self.starts, self.stops, self.templates):
            if np.any(good[start:stop]):
                masked.starts.append(start)
                masked.stops.append(stop)
                masked.templates.append(template)
        masked.normalize(good)
        return masked

    def normalize(self, good=None):
        """Normalize templates"""
        for start, stop, template in zip(self.starts, self.stops, self.templates):
            if good is None:
                norm = np.sum(template**2) ** 0.5
            else:
                norm = np.sum((template * good[start:stop]) ** 2) ** 0.5
            template /= norm
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
        indices = np.load(datafile.replace(".data.", ".indices."))
        indptr = np.load(datafile.replace(".data.", ".indptr."))
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

    log.info_rank(f"Constructed in", timer=timer, comm=None)

    log.info(f"Writing {rootname}.npz ...")
    scipy.sparse.save_npz(rootname, obs_matrix)
    log.info_rank(f"Wrote in", timer=timer, comm=None)

    log.info_rank(f"All done in", timer=timer0, comm=None)

    return f"{rootname}.npz"


@trait_docs
class FilterBin(Operator):
    """FilterBin buids a template matrix and projects out
    compromised modes.  It then bins the signal and optionally
    writes out the sparse observation matrix that matches the
    filtering operations.
    FilterBin supports deprojection templates.

    THIS OPERATOR ASSUMES OBSERVATIONS ARE DISTRIBUTED BY DETECTOR
    WITHIN A GROUP

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
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
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid,
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
        5,
        allow_none=True,
        help="Order of a Legendre polynomial to fit as a function of azimuth.",
    )

    split_ground_template = Bool(
        False, help="Apply a different template for left and right scans"
    )

    leftright_mask = Int(
        defaults.scan_leftright, help="Bit mask value for left-to-right scans"
    )

    rightleft_mask = Int(
        defaults.scan_rightleft, help="Bit mask value for right-to-left scans"
    )

    poly_filter_order = Int(1, allow_none=True, help="Polynomial order")

    poly_filter_view = Unicode(
        "throw", allow_none=True, help="Intervals for polynomial filtering"
    )

    write_obs_matrix = Bool(False, help="Write the observation matrix")

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
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

        # Optionally destroy existing pixel distributions (useful if calling
        # repeatedly with different data objects)

        binning = self.binning
        pixel_dist = binning.pixel_dist
        if self.reset_pix_dist:
            if pixel_dist in data:
                del data[pixel_dist]

        if pixel_dist not in data:
            pix_dist = BuildPixelDistribution(
                pixel_dist=pixel_dist,
                pixel_pointing=binning.pixel_pointing,
                shared_flags=binning.shared_flags,
                shared_flag_mask=binning.shared_flag_mask,
            )
            pix_dist.apply(data)
            log.debug_rank(
                "Cached pixel distribution in", comm=data.comm.comm_world, timer=timer
            )

        self.npix = data[pixel_dist].n_pix
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

        self._bin_map(data, detectors, filtered=False)
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
            dets = obs.select_local_detectors(detectors)
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
            template_covariance = None

            for idet, det in enumerate(dets):
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
                good_bin = np.logical_and(
                    (common_flags & self.binning.shared_flag_mask) == 0,
                    (flags & self.binning.det_flag_mask) == 0,
                )

                if np.sum(good_fit) == 0:
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

                det_templates = common_templates.mask(good_fit)

                if (
                    self.deproject_map is not None
                    and self._deproject_pattern.match(det) is not None
                ):
                    self._add_deprojection_templates(data, obs, pixels, det_templates)
                    # Must re-evaluate the template covariance
                    template_covariance = None

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built deprojection "
                        f"templates in {time() - t1:.2f} s. "
                        f"ntemplate = {det_templates.ntemplate}",
                    )
                    t1 = time()

                memreport.prefix = "After detector templates"
                memreport.apply(data)

                if template_covariance is None or np.any(last_good_fit != good_fit):
                    template_covariance = self._build_template_covariance(
                        det_templates, good_fit
                    )
                    last_good_fit = good_fit.copy()

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built template covariance "
                        f"{time() - t1:.2f} s",
                    )
                    t1 = time()

                self._regress_templates(
                    det_templates, template_covariance, signal, good_fit
                )
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Regressed templates in "
                        f"{time() - t1:.2f} s",
                    )
                    t1 = time()

                self._accumulate_observation_matrix(
                    obs,
                    det,
                    pixels,
                    weights,
                    good_fit,
                    good_bin,
                    det_templates,
                    template_covariance,
                )

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

        self._bin_map(data, detectors, filtered=True)
        log.debug_rank("FilterBin: Binned filtered map in", comm=self.comm, timer=timer)

        log.info_rank(
            f"FilterBin:   Binned data in",
            comm=self.comm,
            timer=timer2,
        )

        memreport.prefix = "After binning"
        memreport.apply(data)

        if self.write_obs_matrix:
            log.debug_rank(
                "FilterBin: Noise-weighting observation matrix", comm=self.comm
            )
            self._noiseweight_obs_matrix(data)
            log.debug_rank(
                "FilterBin: Noise-weighted observation_matrix in",
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

        fourier_templates = np.zeros([nfilter, hwp_angle.size])
        fourier(hwp_angle, fourier_templates, 1, self.hwp_filter_order + 1)

        templates.append(fourier_templates)

        return

    @function_timer
    def _add_ground_templates(self, obs, templates):
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

        legendre_templates = np.zeros([nfilter, phase.size])
        legendre(phase, legendre_templates, min_order, max_order + 1)
        if not self.split_ground_template:
            legendre_filter = legendre_templates
        else:
            # Separate ground filter by scan direction.  These bit masks are
            # hard-coded in sim_ground.py
            legendre_filter = []
            mask1 = (shared_flags & self.leftright_mask) != 0
            mask2 = (shared_flags & self.rightleft_mask) != 0
            for template in legendre_templates:
                for mask in mask1, mask2:
                    temp = template.copy()
                    temp[mask] = 0
                    legendre_filter.append(temp)
            legendre_filter = np.vstack(legendre_filter)

        templates.append(legendre_filter)

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

        for ival in intervals:
            istart = ival.first
            istop = ival.last + 1
            # Trim flagged samples from both ends
            while istart < istop and bad[istart]:
                istart += 1
            while istop - 1 > istart and bad[istop - 1]:
                istop -= 1
            if istop - istart < nfilter:
                # Not enough samples to filter, flag this interval
                shared_flags[ival.first : ival.last + 1] |= self.filter_flag_mask
                continue
            wbin = 2 / (istop - istart)
            phase = (np.arange(istop - istart) + 0.5) * wbin - 1
            legendre_templates = np.zeros([nfilter, phase.size])
            legendre(phase, legendre_templates, 0, nfilter)
            templates.append(legendre_templates, start=istart, stop=istop)

        if self.shared_flags is not None:
            obs.shared[self.shared_flags].set(shared_flags, offset=(0,), fromrank=0)

        return

    @function_timer
    def _build_common_templates(self, obs):
        templates = SparseTemplates()

        self._add_hwp_templates(obs, templates)
        self._add_ground_templates(obs, templates)
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
        norm = np.dot(common_templates[0], common_templates[0])
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
            dptemplate *= np.sqrt(norm / np.dot(dptemplate, dptemplate))
            templates.append(dptemplate)
        return

    @function_timer
    def _build_template_covariance(self, templates, good):
        """Calculate (F^T N^-1_F F)^-1

        Observe that the sample noise weights in N^-1_F need not be the
        same as in binning the filtered signal.  For instance, samples
        falling on point sources may be masked here but included in the
        final map.
        """
        log = Logger.get()
        ntemplate = templates.ntemplate
        invcov = np.zeros([ntemplate, ntemplate])
        build_template_covariance(
            templates.starts,
            templates.stops,
            templates.templates,
            good.astype(np.float64),
            invcov,
        )
        rcond = 1 / np.linalg.cond(invcov)
        if self.grank == 0:
            log.debug(
                f"{self.group:4} : FilterBin: Template covariance matrix "
                f"rcond = {rcond}",
            )
        if rcond > 1e-6:
            cov = np.linalg.inv(invcov)
        else:
            log.warning(
                f"{self.group:4} : FilterBin: WARNING: template covariance matrix "
                f"is poorly conditioned: "
                f"rcond = {rcond}.  Using matrix pseudoinverse.",
            )
            cov = np.linalg.pinv(invcov, rcond=1e-10, hermitian=True)

        return cov

    @function_timer
    def _regress_templates(self, templates, template_covariance, signal, good):
        """Calculate Zd = (I - F(F^T N^-1_F F)^-1 F^T N^-1_F)d

        All samples that are not flagged (zero weight in N^-1_F) have
        equal weight.
        """
        proj = templates.dot(signal * good)
        amplitudes = np.dot(template_covariance, proj)
        templates.subtract(signal, amplitudes)
        return

    @function_timer
    def _compress_pixels(self, pixels):
        local_to_global = np.sort(list(set(pixels)))
        compressed_pixels = np.searchsorted(local_to_global, pixels)
        return compressed_pixels, local_to_global.size, local_to_global

    @function_timer
    def _expand_matrix(self, compressed_matrix, local_to_global):
        """Expands a dense, compressed matrix into a sparse matrix with
        global indexing
        """
        n = compressed_matrix.size
        indices = np.zeros(n, dtype=np.int64)
        indptr = np.zeros(self.npixtot + 1, dtype=np.int64)
        expand_matrix(
            compressed_matrix,
            local_to_global,
            self.npix,
            self.nnz,
            indices,
            indptr,
        )

        sparse_matrix = scipy.sparse.csr_matrix(
            (compressed_matrix.ravel(), indices, indptr),
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
            accumulate_observation_matrix(
                c_obs_matrix,
                c_pixels,
                weights[good_any].copy(),
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
        self.obs_matrix += local_obs_matrix * detweight
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
    def _noiseweight_obs_matrix(self, data):
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
                    f"Slice {islice+1:5} / {nslice}: {row_start:12} - {row_stop:12} "
                    f"is empty.  Skipping.",
                    comm=self.comm,
                )
                continue
            log.debug_rank(
                f"Collecting slice {islice+1:5} / {nslice} : {row_start:12} - "
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
                        indices_recv = np.zeros(size_recv, dtype=np.int32)
                        self.comm.Recv(
                            indices_recv,
                            source=receive_from,
                            tag=factor + 2 * self.ntask,
                        )
                        indptr_recv = np.zeros(
                            obs_matrix_slice.indptr.size, dtype=np.int32
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
                        obs_matrix_slice.indices,
                        dest=send_to,
                        tag=factor + 2 * self.ntask,
                    )
                    self.comm.Send(
                        obs_matrix_slice.indptr,
                        dest=send_to,
                        tag=factor + 3 * self.ntask,
                    )

                if self.comm is not None:
                    self.comm.Barrier()
                log.debug_rank("FilterBin: Collected in", comm=self.comm, timer=timer)
                factor *= 2

            # Write out the observation matrix
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
    def _bin_map(self, data, detectors, filtered):
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

        cov = CovarianceAndHits(
            pixel_dist=self.binning.pixel_dist,
            covariance=self.binning.covariance,
            inverse_covariance=invcov_name,
            hits=hits_name,
            rcond=rcond_name,
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

        self.binning.apply(data, detectors=detectors)

        binned = not filtered
        for key, write, keep in [
            (hits_name, self.write_hits and binned, False),
            (rcond_name, self.write_rcond and binned, False),
            (noiseweighted_map_name, self.write_noiseweighted_map, False),
            (map_name, self.write_map, False),
            (invcov_name, self.write_invcov and binned, False),
            (cov_name, self.write_cov and binned, True),
        ]:
            if write:
                try:
                    if self.write_hdf5:
                        # Non-standard HDF5 output
                        fname = os.path.join(self.output_dir, f"{key}.h5")
                        write_healpix_hdf5(
                            data[key],
                            fname,
                            nest=self.binning.pixel_pointing.nest,
                            force_serial=self.write_hdf5_serial,
                        )
                    else:
                        # Standard FITS output
                        fname = os.path.join(self.output_dir, f"{key}.fits")
                        write_healpix_fits(
                            data[key], fname, nest=self.binning.pixel_pointing.nest
                        )
                except Exception as e:
                    msg = f"ERROR: failed to write {fname} : {e}"
                    raise RuntimeError(msg)
                log.info_rank(f"Wrote {fname} in", comm=self.comm, timer=timer)
            if not keep and key in data:
                data[key].clear()
                del data[key]

        return

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
        if filename_is_hdf5(self.deproject_map):
            read_healpix_hdf5(
                data[self.deproject_map_name],
                self.deproject_map,
                nest=self.binning.pixel_pointing.nest,
            )
        elif filename_is_fits(self.deproject_map):
            read_healpix_fits(
                data[self.deproject_map_name],
                self.deproject_map,
                nest=self.binning.pixel_pointing.nest,
            )
        else:
            msg = f"Cannot determine deprojection map type: {self.deproject_map}"
            raise RuntimeError(msg)
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
