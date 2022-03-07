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

from .. import qarray as qa
from .._libtoast import (accumulate_observation_matrix,
                         expand_matrix, legendre)
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelData, PixelDistribution
from ..pixels_io import (filename_is_fits, filename_is_hdf5, read_healpix_fits,
                         read_healpix_hdf5, write_healpix_fits,
                         write_healpix_hdf5)
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


XAXIS, YAXIS, ZAXIS = np.eye(3)


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

    log.info_rank(f"All done in", timer=timer, comm=None)

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

    FOR THE MOMENT, WE ALSO ASSUME THAT JOINTLY FILTERED DETECTORS
    ARE HANDLED ON THE SAME PROCESS.

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

    focalplane_key = Unicode(
        None,
        allow_none=True,
        help="Which focalplane key to match for spatial (2D) filters.  Use 'telescope' "
        "for a universal common mode.",
    )

    poly2d_filter_order = Int(0, allow_none=True, help="Polynomial order")

    poly2d_filter_view = Unicode(
        "throw", allow_none=True, help="Intervals for 2D polynomial filtering"
    )

    filter_in_sequence = Bool(
        True,
        help="Filters are applied in discreet steps rather than in a single fit.",
    )

    @traitlets.validate("poly2d_filter_order")
    def _check_poly2d_filter_order(self, proposal):
        check = proposal["value"]
        if check != 0:
            raise traitlets.TraitError("poly2d_filter_order is currently limited to 0")
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
            # FIXME:  expand to all detectors when adding MPI
            # all_dets = obs.all_detectors
            all_dets = obs.select_local_detectors(detectors)
            local_dets = obs.select_local_detectors(detectors)

            # Prepare for common mode filtering
            focalplane = obs.telescope.focalplane
            detectors_by_value = {}
            if self.focalplane_key is None:
                # Each detector will be filtered independently
                for det in local_dets:
                    detectors_by_value[det] = [det]
            else:
                # Spatial filters will be applied to detectors that share
                # the focalplane key
                for det in all_dets:
                    if self.focalplane_key == "telescope":
                        value = obs.telescope.name
                    else:
                        value = focalplane[det][self.focalplane_key]
                    if value not in detectors_by_value:
                        detectors_by_value[value] = []
                    detectors_by_value[value].append(det)
            values = sorted(detectors_by_value.keys())

            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: Processing observation "
                    f"{iobs} / {len(data.obs)}",
                )

            ground_templates = self._get_ground_templates(obs)
            poly_templates = self._get_poly_templates(obs)

            if self.shared_flags is not None:
                common_flags = obs.shared[self.shared_flags].data
            else:
                common_flags = np.zeros(phase.size, dtype=np.uint8)

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

            for ivalue, value in enumerate(values):
                dets = detectors_by_value[value]
                ndet = len(dets)

                all_templates = []
                spatial_templates = self._get_spatial_templates(obs, dets)
                if spatial_templates is not None:
                    all_templates.append(spatial_templates)
                full_ground_templates = self._expand_templates(ground_templates, ndet)
                if full_ground_templates is not None:
                    all_templates.append(full_ground_templates)
                full_poly_templates = self._expand_templates(poly_templates, ndet)
                if full_poly_templates is not None:
                    all_templates.append(full_poly_templates)
                full_deprojection_templates = []

                full_signal = []
                full_good_fit = []
                full_good_bin = []
                full_pixels = []
                full_weights = []

                for idet, det in enumerate(detectors_by_value[value]):
                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Processing detector "
                            f"# {idet + 1} / {ndet} : key = {value}",
                        )

                    # FIXME: even if jointly-filtered detectors are not on the same
                    # process, every process will need to participate in the collective
                    # operations

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

                    full_signal.append(signal)
                    full_good_fit.append(good_fit)
                    full_good_bin.append(good_bin)

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
                        if weights.ndim == 1:
                            weights = weights.reshape(-1, 1)
                        full_pixels.append(pixels)
                        full_weights.append(weights)
                    else:
                        pixels = None
                        weights = None

                    #det_templates = common_templates.mask(good_fit)

                    if (
                        self.deproject_map is not None
                        and self._deproject_pattern.match(det) is not None
                    ):
                        deprojection_templates = self._get_deprojection_templates(
                            data, obs, pixels, det_templates
                        )
                        # Must re-evaluate the template covariance
                        template_covariance = None
                    else:
                        deprojection_templates = None
                    full_deprojection_templates.append(deprojection_templates)

                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Built deprojection "
                            f"templates in {time() - t1:.2f} s."
                        )
                        t1 = time()

                    memreport.prefix = "After detector templates"
                    memreport.apply(data)

                # Concatenate lists

                full_signal = np.hstack(full_signal)
                full_good_fit = np.hstack(full_good_fit)
                full_good_bin = np.hstack(full_good_bin)
                full_deprojection_templates = self._expand_templates(
                    full_deprojection_templates, ndet
                )
                if full_deprojection_templates is not None:
                    all_templates.append(full_deprojection_templates)

                full_noise_weights = scipy.sparse.diags(full_good_fit, dtype=float)

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built detector templates for "
                        f"key={value} in {time() - t1:.2f} s",
                    )
                    t1 = time()

                # Filter all detectors in the current detector set

                if not self.filter_in_sequence:
                    all_templates = [scipy.sparse.vstack(all_templates, format="csr")]

                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Stacked templates for "
                            f"key={value} in {time() - t1:.2f} s",
                        )
                        t1 = time()

                t2 = time()
                all_covs = []
                for itemplates in range(len(all_templates)):
                    templates = all_templates[itemplates]
                    # Normalize the templates to improve the condition number of the
                    # template covariance matrix

                    ntemplate, nsample_tot = templates.shape
                    good_templates = []
                    for itemplate in range(ntemplate):
                        template = templates[itemplate, :].toarray().ravel()
                        norm = np.sum(template[full_good_fit] ** 2)
                        if norm > 1e-3:
                            templates[itemplate] /= norm ** .5
                            good_templates.append(itemplate)
                    nbad = ntemplate - len(good_templates)
                    if nbad != 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Discarding {nbad} "
                            f"poorly-constrained templates"
                        )
                        templates = templates[good_templates]

                    # Now build the inverse covariance matrix

                    invcov = templates.dot(full_noise_weights.dot(templates.T))
                    invcov = invcov.toarray()

                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Stacked templates for "
                            f"key={value} : {itemplates + 1}/{len(all_templates)} "
                            f"in {time() - t1:.2f} s",
                        )
                        t1 = time()

                    # Invert the sub-matrix that has non-zero diagonal

                    rcond = 1 / np.linalg.cond(invcov)
                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Template covariance matrix "
                            f"rcond = {rcond} in {time() - t1:.2f} s",
                        )
                        t1 = time()

                    if rcond > 1e-6:
                        cov = np.linalg.inv(invcov)
                    else:
                        log.warning(
                            f"{self.group:4} : FilterBin: WARNING: template covariance matrix "
                            f"is poorly conditioned: "
                            f"rcond = {rcond}.  Using matrix pseudoinverse.",
                        )
                        cov = np.linalg.pinv(invcov, rcond=1e-12, hermitian=True)

                    all_templates[itemplates] = templates
                    all_covs.append(cov)

                    if self.grank == 0:
                        log.debug(
                            f"{self.group:4} : FilterBin:   Inverted covariance for "
                            f"key={value} : {itemplates + 1}/{len(all_templates)} "
                            f"in {time() - t1:.2f} s",
                        )
                        t1 = time()

                    import pdb
                    pdb.set_trace()

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Built template covariance "
                        f"{time() - t2:.2f} s",
                    )
                    t1 = time()

                fitted_signal = np.zeros_like(full_signal)
                for itemplates in range(len(all_templates)):
                    templates = all_templates[itemplates]
                    cov = all_covs[itemplates]
                    proj = templates.dot(full_noise_weights.dot(full_signal))
                    coeff = np.dot(cov, proj)

                    full_signal -= np.array(templates.T.dot(coeff.T)).ravel()

                full_signal = full_signal.reshape([ndet, -1])
                for idet, det in enumerate(dets):
                    obs.detdata[self.det_data][det] = full_signal[idet]

                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:   Regressed templates in "
                        f"{time() - t1:.2f} s",
                    )
                    t1 = time()

                self._accumulate_full_observation_matrix(
                    obs,
                    value,
                    detectors_by_value[value],
                    full_pixels,
                    full_weights,
                    full_good_fit,
                    full_good_bin,
                    all_templates,
                    all_covs,
                    full_noise_weights,
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
    def _get_ground_templates(self, obs):
        if self.ground_filter_order is None:
            return None

        phase = self._get_phase(obs)
        shared_flags = np.array(obs.shared[self.shared_flags])

        min_order = 0
        if not self.filter_in_sequence and self.poly_filter_order is not None:
            # To avoid template degeneracies, ground filter only includes
            # polynomial orders not present in the polynomial filter
            min_order = self.poly_filter_order + 1
        max_order = self.ground_filter_order
        nfilter = max_order - min_order + 1
        if nfilter < 1:
            return None

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

        return scipy.sparse.csr_matrix(legendre_filter)

    @function_timer
    def _get_poly_templates(self, obs):
        if self.poly_filter_order is None:
            return None

        norder = self.poly_filter_order + 1
        intervals = obs.intervals[self.poly_filter_view]
        ninterval = len(intervals)
        ntemplate = ninterval * norder
        nsample = obs.n_all_samples
        templates = scipy.sparse.lil_matrix((ntemplate, nsample))

        if self.shared_flags is None:
            shared_flags = np.zeros(obs.n_local_samples, dtype=np.uint8)
        else:
            shared_flags = np.array(obs.shared[self.shared_flags])
        bad = (shared_flags & self.shared_flag_mask) != 0

        template_start = 0
        for ival in intervals:
            template_stop = template_start + norder
            sample_start = ival.first
            sample_stop = ival.last + 1
            # Trim flagged samples from both ends
            while sample_start < sample_stop and bad[sample_start]:
                sample_start += 1
            while sample_stop - 1 > sample_start and bad[sample_stop - 1]:
                sample_stop -= 1
            if sample_stop - sample_start < norder:
                # Not enough samples to filter, flag this interval
                shared_flags[ival.first : ival.last + 1] |= self.filter_flag_mask
                continue
            wbin = 2 / (sample_stop - sample_start)
            phase = (np.arange(sample_stop - sample_start) + 0.5) * wbin - 1
            legendre_templates = np.zeros([norder, phase.size])
            legendre(phase, legendre_templates, 0, norder)
            templates[
                template_start : template_stop, sample_start : sample_stop
            ] = legendre_templates
            template_start = template_stop

        if self.shared_flags is not None:
            obs.shared[self.shared_flags].set(shared_flags, offset=(0,), fromrank=0)

        return templates

    @function_timer
    def _expand_templates(self, templates, ndet):
        """ Take template matrix and tile it to
        filter an entire detector set """
        if templates is None:
            return None

        if isinstance(templates, (list, tuple)):
            # list of per-detector templates
            # This code handles None entries and varying number of templates
            ntemplate_full = 0
            for det_templates in templates:
                if det_templates is not None:
                    ntemplate, nsample = det_templates.shape
                    ntemplate_full += ntemplate
            if ntemplate_full == 0:
                return None
            full_templates = scipy.sparse.lil_matrix((ntemplate_full, nsample * ndet))
            offset = 0
            for idet, det_templates in enumerate(templates):
                ntemplate, nsample = det_templates.shape
                full_templates[
                    offset : offset + ntemplate,
                    idet * nsample : (idet + 1) * nsample
                ] = det_templates
                offset += ntemplate
        else:
            # Only one set of templates to apply to all detectors
            ntemplate, nsample = templates.shape
            full_templates = scipy.sparse.lil_matrix((ntemplate * ndet, nsample * ndet))
            for idet in range(ndet):
                full_templates[
                    idet * ntemplate : (idet + 1) * ntemplate,
                    idet * nsample : (idet + 1) * nsample,
                ] = templates

        return full_templates

    @function_timer
    def _get_spatial_templates(self, obs, detectors):
        if self.focalplane_key is None:
            return None

        ndet = len(detectors)
        nsample = obs.n_all_samples
        norder = (self.poly2d_filter_order + 1) ** 2
        ntemplate = nsample * norder
        # The templates matrix will be transposed before returning it
        templates = scipy.sparse.lil_matrix((nsample * ndet, ntemplate))

        focalplane = obs.telescope.focalplane

        detector_position = {}
        theta_offset = 0
        phi_offset = 0
        thetas = np.zeros(ndet)
        phis = np.zeros(ndet)
        for idet, det in enumerate(detectors):
            det_quat = focalplane[det]["quat"]
            x, y, z = qa.rotate(det_quat, ZAXIS)
            theta, phi = np.arcsin([x, y])
            thetas[idet] = theta
            phis[idet] = phi
        theta_scale = 0.999 / np.ptp(thetas)
        phi_scale = 0.999 / np.ptp(phis)
        theta_offset = 0.5 * (np.amin(thetas) + np.amax(thetas))
        phi_offset = 0.5 * (np.amin(phis) + np.amax(phis))
        thetas = (thetas - theta_offset) * theta_scale
        phis = (phis - phi_offset) * phi_scale

        # Evaluate the templates

        snapshot = np.ones([norder, ndet])
        itemplate = 0
        for xorder in range(norder):
            for yorder in range(norder):
                template = thetas**xorder * phis**yorder
                # normalize the template
                template /= np.dot(template, template) ** .5
                snapshot[itemplate] = template
                itemplate += 1
        snapshot = snapshot.T

        for idet in range(ndet):
            for isample in range(nsample):
                templates[
                    idet * nsample + isample,
                    isample * norder : (isample + 1) * norder
                ] = snapshot[idet]
        templates = templates.T  # Into [ntemplates, nsample * ndet]

        return templates

    @function_timer
    def _build_common_templates(self, obs):
        templates = SparseTemplates()

        self._add_ground_templates(obs, templates)
        self._add_poly_templates(obs, templates)

        return templates

    @function_timer
    def _get_deprojection_templates(self, data, obs, pixels, templates):
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

        ntemplate = nnz
        templates = scipy.sparse.lil_matrix((ntemplate, nsample))
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
            templates[inz] = dptemplate

        return templates

    @function_timer
    def _compress_pixels(self, pixels):
        shape = pixels.shape
        local_to_global = np.sort(list(set(pixels.ravel())))
        compressed_pixels = np.searchsorted(local_to_global, pixels.ravel())
        return compressed_pixels.reshape(shape), local_to_global.size, local_to_global

    @function_timer
    def _expand_matrix(self, compressed_matrix, local_to_global):
        """Expands a dense, compressed matrix into a sparse matrix with
        global indexing
        """
        if np.any(local_to_global < 0):
            # Includes the bad pixel from masking
            nnz = compressed_matrix[0].size // local_to_global.size
            good = local_to_global >= 0
            goodtot = np.tile(good, nnz)
            compressed_matrix = compressed_matrix[goodtot, :][:, goodtot].copy()
            local_to_global = local_to_global[good].copy()

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
    def _accumulate_full_observation_matrix(
        self,
        obs,
        value,
        detectors,
        pixels,
        weights,
        good_fit,
        good_bin,
        all_templates,
        all_covs,
        noise_weights,
    ):
        """Calculate P^T N^-1 Z P
        This part of the covariance calculation is cumulative: each observation
        and detector set is computed independently and can be cached.

        Observe that `N` in this equation need not be the same used in
        template covariance in `Z`.
        """
        if not self.write_obs_matrix:
            return
        log = Logger.get()
        fname_cache = None
        local_obs_matrix = None
        t1 = time()
        if self.cache_dir is not None:
            cache_dir = os.path.join(self.cache_dir, obs.name)
            os.makedirs(cache_dir, exist_ok=True)
            fname_cache = os.path.join(cache_dir, value)
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

        if local_obs_matrix is None:
            good_any = np.logical_or(good_fit, good_bin)

            # Temporarily compress pixels
            if self.grank == 0:
                log.debug(f"{self.group:4} : FilterBin:     Compressing pixels")
            pixels = np.hstack(pixels)
            pixels[np.logical_not(good_any)] = -1
            c_pixels, c_npix, local_to_global = self._compress_pixels(pixels)
            if self.grank == 0:
                log.debug(
                    f"{self.group:4} : FilterBin: Compressed in {time() - t1:.2f} s",
                )
                t1 = time()
            c_npixtot = c_npix * self.nnz
            c_obs_matrix = np.zeros([c_npixtot, c_npixtot])
            if self.grank == 0:
                log.debug(f"{self.group:4} : FilterBin:     Accumulating")

            ############ accumulate spatial observation matrix begin

            ndet = len(detectors)
            nsample = obs.n_all_samples

            Ztot = None
            for templates, cov in zip(all_templates, all_covs):
                cov = scipy.sparse.csr_matrix(cov)
                Z = scipy.sparse.identity(nsample * ndet) \
                    - templates.T.dot(cov.dot(templates.dot(noise_weights)))
                if Ztot is None:
                    Ztot = Z
                else:
                    Ztot = Z.dot(Ztot)

            weights = np.array(weights)
            nnz = self.nnz
            npix = c_npix
            npixtot = c_npixtot

            P = scipy.sparse.lil_matrix((nsample * ndet, npixtot))
            for idet in range(ndet):
                for isample in range(nsample):
                    pix = c_pixels[idet * nsample + isample]
                    for inz in range(nnz):
                        P[idet * nsample + isample, pix + inz * npix] = \
                            weights[idet][isample][inz]

            det_weights = np.zeros(nsample * ndet)
            for idet in range(ndet):
                detweight = obs[self.binning.noise_model].detector_weight(detectors[idet])
                det_weights[idet * nsample : (idet + 1) * nsample] = detweight
            det_weights = scipy.sparse.diags((det_weights))

            c_obs_matrix = P.T.dot(det_weights.dot(Ztot.dot(P))).toarray()
            #import pdb
            #import matplotlib.pyplot as plt
            #pdb.set_trace()

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
                np.save(fname_cache + ".data", local_obs_matrix)
                np.save(fname_cache + ".indices", local_obs_matrix)
                np.save(fname_cache + ".indptr", local_obs_matrix)
                if self.grank == 0:
                    log.debug(
                        f"{self.group:4} : FilterBin:     cached in {time() - t1:.2f} s",
                    )
                    t1 = time()

        if self.grank == 0:
            log.debug(f"{self.group:4} : FilterBin:     Adding to global")

        self.obs_matrix += local_obs_matrix
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
                theta, phi = qa.to_position(quats)
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
        obs_matrix = self.obs_matrix
        nslice_empty = 0
        nslice_hit = 0
        for islice, row_start in enumerate(range(0, nrow_tot, nrow_write)):
            row_stop = row_start + nrow_write
            obs_matrix_slice = obs_matrix[row_start:row_stop]
            nnz = obs_matrix_slice.nnz
            if self.comm is not None:
                nnz = self.comm.allreduce(nnz)
            if nnz == 0:
                nslice_empty += 1
                log.debug_rank(
                    f"Slice {islice+1:5} / {nslice}: {row_start:12} - {row_stop:12} "
                    f"is empty.  Skipping.",
                    comm=self.comm,
                )
                continue
            else:
                nslice_hit += 1
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
        if nslice_hit == 0:
            log.warning_rank(
                f"FilterBin: Temporal observation matrix is completely empty! "
                "Nothing written to file.",
                comm=self.comm,
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
        self.binning.covariance = cov_name

        cov = CovarianceAndHits(
            pixel_dist=self.binning.pixel_dist,
            covariance=self.binning.covariance,
            inverse_covariance=invcov_name,
            hits=hits_name,
            rcond=rcond_name,
            det_flags=self.binning.det_flags,
            det_flag_mask=self.binning.det_flag_mask,
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
