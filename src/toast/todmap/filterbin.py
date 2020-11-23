# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os

import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy.io
import scipy.sparse

from .todmap_math import OpAccumDiag, OpScanScale, OpScanMask
from .._libtoast import (
    bin_templates, add_templates, chebyshev, accumulate_observation_matrix
)
from ..map import covariance_apply, covariance_invert, DistPixels, covariance_rcond
from ..op import Operator
from ..utils import Logger
from ..timing import function_timer


class OpFilterBin(Operator):
    """ OpFilterBin buids a template matrix and projects out
    compromised modes.  It then bins the signal and optionally
    writes out the sparse observation matrix that matches the
    filtering operations.
    OpFilterBin supports deprojection templates.

    THIS OPERATOR ASSUMES OBSERVATIONS ARE DISTRIBUTED BY DETECTOR
    WITHIN A GROUP

    Args:
        name (str):  Name of the output signal cache object will be
            <name_in>_<detector>.  If the object exists, it is used as
            input.  Otherwise signal is read using the tod read method.
        common_flag_name (str):  Cache name of the output common flags.
            If it already exists, it is used.  Otherwise flags
            are read from the tod object and stored in the cache under
            common_flag_name.
        common_flag_mask (byte):  Bitmask to use when flagging data
           based on the common flags.
        flag_name (str):  Cache name of the output detector flags will
            be <flag_name>_<detector>.  If the object exists, it is
            used.  Otherwise flags are read from the tod object.
        flag_mask (byte):  Bitmask to use when flagging data
           based on the detector flags.
        ground_filter_order (int):  Order of a Chebyshev polynomial to
            fit as a function of azimuth
        intervals (str):  Name of the valid intervals in observation
        split_ground_template (bool):  Apply a different template for
             left and right scans
    """

    def __init__(
        self,
        name=None,
        common_flag_name=None,
        common_flag_mask=255,
        flag_name=None,
        flag_mask=255,
        ground_filter_order=None,
        poly_filter_order=None,
        intervals="intervals",
        split_ground_template=False,
        pixels_name="pixels",
        weights_name="weights",
        write_obs_matrix=False,
        nside=256,
        nnz=3,
        rcond_limit=1e-3,
        outdir=".",
        outprefix="filtered_",
        zip_maps=True,
    ):
        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._ground_filter_order = ground_filter_order
        self._poly_filter_order = poly_filter_order
        self._intervals = intervals
        self._split_ground_template = split_ground_template
        self._pixels_name = pixels_name
        self._weights_name = weights_name
        self._write_obs_matrix = write_obs_matrix
        self._nside = nside
        self._npix = 12 * nside ** 2
        self._nnz = nnz
        self._npixtot = self._npix * self._nnz
        self._ncov = nnz * (nnz + 1) // 2
        self._rcond_limit = rcond_limit
        self._outdir = outdir
        self._outprefix = outprefix
        self._zip_maps = zip_maps

        # Call the parent class constructor.
        super().__init__()

    def _build_templates(self, times, phase, intervals, good):
        nsample = times.size
        templates = []

        # Add templates
        templates.append(np.ones(nsample, dtype=np.float))
        templates.append(2 * np.arange(nsample) / (nsample - 1) - 1)

        # Get covariance
        templates = np.vstack(templates)
        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        for row in range(ntemplate):
            itemplate = templates[row] * good
            for col in range(row, ntemplate):
                jtemplate = templates[col]
                invcov[row, col] = np.dot(itemplate, jtemplate)
                invcov[col, row] = invcov[row, col]
        cov = np.linalg.inv(invcov)

        return templates, cov

    def _build_templates_sparse(self, times, phase, intervals, good):
        # Not currently used.
        ntemplate = 2
        nsample = times.size

        data = []
        row_ind = []
        col_ind = []

        # Add templates

        row = 0
        row_ind.append(np.zeros(nsample, dtype=np.int) + row)
        col_ind.append(np.arange(nsample, dtype=np.int))
        data.append(np.ones(nsample, dtype=np.float))

        row += 1
        row_ind.append(np.zeros(nsample, dtype=np.int) + row)
        col_ind.append(np.arange(nsample, dtype=np.int))
        data.append(2 * np.arange(nsample) / (nsample - 1) - 1)

        # Compute covariance

        data = np.hstack(data)
        row_ind = np.hstack(row_ind)
        col_ind = np.hstack(col_ind)

        templates = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(ntemplate, nsample),
            dtype=np.float64,
        )
        templatesT = templates.T.copy()
        invcov = templates.dot(templatesT).toarray()
        cov = np.linalg.inv(invcov)

        return templates, templatesT, cov

    def _regress_templates(self, templates, template_covariance, signal, good):
        proj = np.dot(templates, signal * good)
        amplitudes = np.dot(template_covariance, proj)
        for template, amplitude in zip(templates, amplitudes):
            signal -= amplitude * template
        return

    def _compress_pixels(self, pixels):
        local_to_global = np.sort(list(set(pixels)))
        compressed_pixels = np.searchsorted(local_to_global, pixels)
        return compressed_pixels, local_to_global.size, local_to_global

    def _accumulate_observation_matrix(
            self,
            obs_matrix,
            pixels,
            weights,
            good,
            templates,
            template_covariance,
            detweight,
    ):
        if obs_matrix is None:
            return
        nsample = pixels.size
        npix = self._npix
        nnz = self._nnz
        npixtot = self._npixtot
        cov = template_covariance
        templates = templates.T.copy()
        from time import time
        # Temporarily compress pixels
        c_pixels, c_npix, local_to_global = self._compress_pixels(pixels[good].copy())
        c_npixtot = c_npix * self._nnz
        c_obs_matrix = np.zeros([c_npixtot, c_npixtot])
        t0 = time()
        print("Accumulating", flush=True)
        accumulate_observation_matrix(
            c_obs_matrix,
            c_pixels,
            weights[good].copy(),
            templates[good].copy(),
            template_covariance,
        )
        c_obs_matrix *= detweight
        print("Accumulated in {:.3f}s".format(time() - t0), flush=True)
        # add the compressed observation matrix onto the global one
        t1 = time()
        print("Expanding local to global", flush=True)
        col_indices = []
        for inz in range(nnz):
            col_indices.append(local_to_global + inz * npix)
        col_indices = np.hstack(col_indices)
        nhit = col_indices.size
        indices = []
        indptr = [0]
        irow = 0
        offset = 0
        for inz in range(nnz):
            for ilocal, iglobal in enumerate(local_to_global):
                while irow < iglobal:
                    indptr.append(offset)
                    irow += 1
                offset += nhit
                indices.append(col_indices)
                indptr.append(offset)
                irow += 1
        while irow < npixtot:
            indptr.append(offset)
            irow += 1
        indices = np.hstack(indices)
        indptr = np.hstack(indptr)
        local_obs_matrix = scipy.sparse.csr_matrix(
            (c_obs_matrix.ravel(), indices, indptr), shape=(npixtot, npixtot),
        )
        print("Expanded in {:.1f}s".format(time() - t1), flush=True)
        t1 = time()
        print("Adding to global", flush=True)
        obs_matrix += local_obs_matrix
        print("Added in {:.1f}s".format(time() - t1), flush=True)
        return obs_matrix

    def _accumulate_observation_matrix_sparse(
            self, obs_matrix, pixels, weights, good, templates, template_covariance
    ):
        # Not currently used
        if obs_matrix is None:
            return
        nsample = pixels.size
        #cov = scipy.sparse.csr_matrix(template_covariance)
        cov = template_covariance
        from time import time
        for isample in range(nsample):
            t1 = time()
            print("isample = {:6} / {:6}".format(isample, nsample), flush=True, end="")  # DEBUG
            ipixel = pixels[isample]
            iweights = weights[isample]
            #itemplates = templates.getrow(isample)
            itemplates = templates[isample].toarray().ravel()
            for jsample in range(isample, nsample):
                jpixel = pixels[jsample]
                jweights = weights[jsample]
                # Evaluate the filtering matrix at (isample, jsample)
                #jtemplates = templates.getrow(jsample)
                #jtemplates = templates[jsample].toarray().ravel()
                #filter_matrix = (isample == jsample) - \
                #    itemplates.dot(cov.dot(jtemplates.T))[0, 0]
                filter_matrix = (isample == jsample) - \
                    itemplates.dot(cov.dot(jtemplates))
                #if filter_matrix == 0:
                #    import pdb
                #    pdb.set_trace()
                #obs_matrix[ipixel::self._npix, jpixel::self._npix] \
                #    += np.outer(iweights, jweights) * filter_matrix
                #if ipixel == 3072 and jpixel == 3072:
                #    import pdb
                #    pdb.set_trace()
            print(" {:.1f}s".format(time() - t1), flush=True)
        return

    def _get_phase(self, tod):
        if self._ground_filter_order is None:
            return None
        try:
            (azmin, azmax, _, _) = tod.scan_range
            az = tod.read_boresight_az()
        except Exception as e:
            raise RuntimeError(
                "Failed to get boresight azimuth from TOD.  Perhaps it is "
                'not ground TOD? "{}"'.format(e)
            )
        phase = (az - azmin) / (azmax - azmin) * 2 - 1
        return phase

    def _initialize_obs_matrix(self):
        if self._write_obs_matrix:
            obs_matrix = scipy.sparse.csr_matrix(
                (self._npixtot, self._npixtot), dtype=np.float64
            )
        else:
            obs_matrix = None
        return obs_matrix

    def _collect_obs_matrix(self, obs_matrix, white_noise_cov):
        if obs_matrix is None:
            return
        # Apply the white noise covariance to the observation matrix
        npix = self._npix
        nnz = self._nnz
        npixtot = self._npixtot
        cc = scipy.sparse.dok_matrix((npixtot, npixtot), dtype=np.float64)
        nsubmap = white_noise_cov.nsubmap
        npix_submap = white_noise_cov.npix_submap
        for isubmap_local, isubmap_global in enumerate(
                white_noise_cov.local_submaps
        ):
            submap = white_noise_cov.data[isubmap_local]
            offset = isubmap_global * npix_submap
            for pix_local in range(npix_submap):
                if np.all(submap[pix_local] == 0):
                    continue
                pix = pix_local + offset
                icov = 0
                for inz in range(nnz):
                    for jnz in range(inz, nnz):
                        cc[pix + inz * npix, pix + jnz * npix] \
                            = submap[pix_local, icov]
                        if inz != jnz:
                            cc[pix + jnz * npix, pix + inz * npix] \
                                = submap[pix_local, icov]
                        icov += 1
        cc = cc.tocsr()
        #import pdb
        #pdb.set_trace()
        obs_matrix = cc.dot(obs_matrix)
        # FIXME: Combine the observation matrix across processes
        # Write out the observation matrix
        fname = os.path.join(self._outdir, self._outprefix + "obs_matrix")
        scipy.sparse.save_npz(fname, obs_matrix)
        return obs_matrix

    def _bin_map(self, data, detweights):
        white_noise_cov = DistPixels(
            data, comm=self.comm, nnz=self._ncov, dtype=np.float64
        )
        if white_noise_cov.data is not None:
            white_noise_cov.data.fill(0)

        hits = DistPixels(data, comm=self.comm, nnz=1, dtype=np.int64)
        if hits.data is not None:
            hits.data.fill(0)

        dist_map = DistPixels(data, comm=self.comm, nnz=self._nnz, dtype=np.float64)
        if dist_map.data is not None:
            dist_map.data.fill(0)

        # FIXME: OpAccumDiag should support separate detweights for each observation
        OpAccumDiag(
            invnpp=white_noise_cov,
            hits=hits,
            zmap=dist_map,
            name=self._name,
            detweights=detweights,
            common_flag_mask=self._common_flag_mask,
            flag_mask=self._flag_mask,
        ).exec(data)

        white_noise_cov.allreduce()
        covariance_invert(white_noise_cov, self._rcond_limit)

        dist_map.allreduce()
        covariance_apply(white_noise_cov, dist_map)
        
        hits.allreduce()

        fname = os.path.join(self._outdir, self._outprefix + "wcov.fits")
        if self._zip_maps:
            fname += ".gz"
        white_noise_cov.write_healpix_fits(fname)

        fname = os.path.join(self._outdir, self._outprefix + "hits.fits")
        if self._zip_maps:
            fname += ".gz"
        hits.write_healpix_fits(fname)

        fname = os.path.join(self._outdir, self._outprefix + "binned.fits")
        if self._zip_maps:
            fname += ".gz"
        dist_map.write_healpix_fits(fname)

        return white_noise_cov


    @function_timer
    def exec(self, data, comm=None):
        
        if comm is None:
            self.comm = data.comm.comm_world
        else:
            self.comm = comm
        if self.comm is None:
            self.rank = 0
        else:
            self.rank = self.comm.rank

        # Filter data

        obs_matrix = self._initialize_obs_matrix()
        detweights = {}

        for obs in data.obs:
            tod = obs["tod"]
            if self._intervals in obs:
                intervals = obs[self._intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)
            times = tod.local_times()
            common_flags = tod.local_common_flags(self._common_flag_name)

            phase = self._get_phase(tod)

            for det in tod.local_dets:
                if det not in detweights:
                    detweights[det] = 1e3
                signal = tod.local_signal(det, self._name)
                flags = tod.local_flags(det, self._flag_name)
                good = np.logical_and(
                    (common_flags & self._common_flag_mask) == 0,
                    (flags & self._flag_mask) == 0,
                )

                pixelsname = "{}_{}".format(self._pixels_name, det)
                weightsname = "{}_{}".format(self._weights_name, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                templates, template_covariance = self._build_templates(
                    times, phase, intervals, good
                )
                self._regress_templates(templates, template_covariance, signal, good)
                obs_matrix = self._accumulate_observation_matrix(
                    obs_matrix,
                    pixels,
                    weights,
                    good,
                    templates,
                    template_covariance,
                    detweights[det],
                )

        # Bin filtered signal

        white_noise_cov = self._bin_map(data, detweights)

        obs_matrix = self._collect_obs_matrix(obs_matrix, white_noise_cov)

        return
