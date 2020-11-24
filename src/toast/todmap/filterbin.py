# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os
from time import time

import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy.io
import scipy.sparse

from .todmap_math import OpAccumDiag, OpScanScale, OpScanMask
from .._libtoast import (
    chebyshev,
    accumulate_observation_matrix,
    expand_matrix,
    build_template_covariance,
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
        verbose=True,
        write_hits=True,
        write_wcov_inv=False,
        write_wcov=True,
        write_binned=False,
        maskfile=None,
    ):
        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._ground_filter_order = ground_filter_order
        self._split_ground_template = split_ground_template
        self._poly_filter_order = poly_filter_order
        self._intervals = intervals
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
        self.verbose = verbose
        self._write_hits = write_hits,
        self._write_wcov_inv = write_wcov_inv
        self._write_wcov = write_wcov
        self._write_binned = write_binned
        if maskfile is not None:
            raise RuntimeError("Filtering mask not yet implemented")
        self._maskfile = maskfile

        # Call the parent class constructor.
        super().__init__()

    @function_timer
    def _add_ground_templates(self, templates, phase, common_flags):
        if self._ground_filter_order is None:
            return templates
        # To avoid template degeneracies, ground filter only includes
        # polynomial orders not present in the polynomial filter
        min_order = 0
        if self._poly_filter_order is not None:
            min_order = self._poly_filter_order + 1
        max_order = self._ground_filter_order
        nfilter = max_order - min_order + 1
        if nfilter < 1:
            return
        cheby_templates = np.zeros([nfilter, phase.size])
        chebyshev(phase, cheby_templates, min_order, max_order + 1)
        if not self._split_ground_template:
            cheby_filter = cheby_templates
        else:
            # Separate ground filter by scan direction
            cheby_filter = []
            mask1 = common_flags & tod.LEFTRIGHT_SCAN == 0
            mask2 = common_flags & tod.RIGHTLEFT_SCAN == 0
            for template in cheby_templates:
                for mask in mask1, mask2:
                    temp = template.copy()
                    temp[mask] = 0
                    cheby_filter.append(temp)
            del common_ref
            cheby_filter = np.vstack(cheby_filter)

        if templates is None:
            templates = cheby_filter
        else:
            templates = np.vstack([templates, cheby_filter])

        return templates

    @function_timer
    def _add_poly_templates(self, templates, local_intervals, nsample):
        if self._poly_filter_order is None:
            return templates
        nfilter = self._poly_filter_order + 1
        ninterval = len(local_intervals)
        poly_templates = np.zeros([nfilter * ninterval, nsample])

        offset = 0
        for ival in local_intervals:
            istart = ival.first
            istop = ival.last + 1
            phase = (np.arange(istart, istop) - istart) / (istop - istart - 1)
            cheby_templates = np.zeros([nfilter, phase.size])
            chebyshev(phase, cheby_templates, 0, nfilter)
            poly_templates[offset : offset + nfilter, istart : istop] = cheby_templates
            offset += nfilter

        if templates is None:
            templates = poly_templates
        else:
            templates = np.vstack([templates, poly_templates])

        return templates

    @function_timer
    def _build_common_templates(self, times, phase, local_intervals, common_flags):
        nsample = times.size
        templates = None

        templates = self._add_ground_templates(templates, phase, common_flags)
        templates = self._add_poly_templates(templates, local_intervals, nsample)

        return templates

    @function_timer
    def _build_templates(
            self, times, phase, local_intervals, good, common_flags, common_templates
    ):
        nsample = times.size
        templates = common_templates.copy()

        # FIXME: add deprojection templates here

        # Get covariance
        templates = np.vstack(templates)
        ntemplate = len(templates)
        invcov = np.zeros([ntemplate, ntemplate])
        build_template_covariance(templates, good.astype(np.float64), invcov)
        cov = np.linalg.inv(invcov)

        return templates, cov

    @function_timer
    def _regress_templates(self, templates, template_covariance, signal, good):
        proj = np.dot(templates, signal * good)
        amplitudes = np.dot(template_covariance, proj)
        for template, amplitude in zip(templates, amplitudes):
            signal -= amplitude * template
        return

    @function_timer
    def _compress_pixels(self, pixels):
        local_to_global = np.sort(list(set(pixels)))
        compressed_pixels = np.searchsorted(local_to_global, pixels)
        return compressed_pixels, local_to_global.size, local_to_global

    @function_timer
    def _expand_matrix(self, compressed_matrix, local_to_global):
        """ Expands a dense, compressed matrix into a sparse matrix with
        global indexing
        """
        n = compressed_matrix.size
        indices = np.zeros(n, dtype=np.int64)
        indptr = np.zeros(self._npixtot + 1, dtype=np.int64)
        expand_matrix(
            compressed_matrix,
            local_to_global,
            self._npix,
            self._nnz,
            indices,
            indptr,
        )

        sparse_matrix = scipy.sparse.csr_matrix(
            (compressed_matrix.ravel(), indices, indptr),
            shape=(self._npixtot, self._npixtot),
        )
        return sparse_matrix

    @function_timer
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
        # Temporarily compress pixels
        t1 = time()
        print("Compressing pixels", flush=True)
        c_pixels, c_npix, local_to_global = self._compress_pixels(pixels[good].copy())
        print("Compressed in {:.3f}s".format(time() - t1), flush=True)
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
        local_obs_matrix = self._expand_matrix(c_obs_matrix, local_to_global)
        print("Expanded in {:.3f}s".format(time() - t1), flush=True)
        t1 = time()
        print("Adding to global", flush=True)
        obs_matrix += local_obs_matrix
        print("Added in {:.3f}s".format(time() - t1), flush=True)
        return obs_matrix

    @function_timer
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
        # The azimuth vector is assumed to be arranged so that the
        # azimuth increases monotonously even across the zero meridian.
        phase = (az - azmin) / (azmax - azmin) * 2 - 1
        return phase

    @function_timer
    def _initialize_obs_matrix(self):
        if self._write_obs_matrix:
            obs_matrix = scipy.sparse.csr_matrix(
                (self._npixtot, self._npixtot), dtype=np.float64
            )
        else:
            obs_matrix = None
        return obs_matrix

    @function_timer
    def _noiseweight_obs_matrix(self, obs_matrix, white_noise_cov):
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
        obs_matrix = cc.dot(obs_matrix)
        return obs_matrix

    @function_timer
    def _collect_obs_matrix(self, obs_matrix):
        if obs_matrix is None:
            return
        # Combine the observation matrix across processes
        comm = self.comm
        rank = comm.rank
        ntask = comm.size
        # Reduce the observation matrices.  We use the buffer protocol
        # for better performance, even though it requires more MPI calls
        # than sending the sparse matrix objects directly
        factor = 1
        while factor < ntask:
            if rank % (factor * 2) == 0:
                # this task receives
                receive_from = rank + factor
                if receive_from < ntask:
                    size_recv = comm.recv(source=receive_from, tag=factor)
                    data_recv = np.zeros(size_recv, dtype=np.float64)
                    comm.Recv(data_recv, source=receive_from, tag=factor + ntask)
                    indices_recv = np.zeros(size_recv, dtype=np.int32)
                    comm.Recv(indices_recv, source=receive_from, tag=factor + 2 * ntask)
                    indptr_recv = np.zeros(obs_matrix.indptr.size, dtype=np.int32)
                    comm.Recv(indptr_recv, source=receive_from, tag=factor + 3 * ntask)
                    obs_matrix += scipy.sparse.csr_matrix(
                        (data_recv, indices_recv, indptr_recv), obs_matrix.shape,
                    )
            elif rank % (factor * 2) == factor:
                # this task sends
                send_to = rank - factor
                comm.send(obs_matrix.data.size, dest=send_to, tag=factor)
                comm.Send(obs_matrix.data, dest=send_to, tag=factor + ntask)
                comm.Send(obs_matrix.indices, dest=send_to, tag=factor + 2 * ntask)
                comm.Send(obs_matrix.indptr, dest=send_to, tag=factor + 3 * ntask)
            factor *= 2

        # Write out the observation matrix
        if rank == 0:
            fname = os.path.join(self._outdir, self._outprefix + "obs_matrix")
            scipy.sparse.save_npz(fname, obs_matrix)
        return obs_matrix

    @function_timer
    def _bin_map(self, data, detweights, suffix, white_noise_cov=None):
        """ Bin the signal onto a map.  Optionally write out hits and
        white noise covariance matrices.
        """
        if white_noise_cov is None:
            invnpp = DistPixels(
                data, comm=self.comm, nnz=self._ncov, dtype=np.float64
            )
            if invnpp.data is not None:
                invnpp.data.fill(0)
            hits = DistPixels(data, comm=self.comm, nnz=1, dtype=np.int64)
            if hits.data is not None:
                hits.data.fill(0)
        else:
            invnpp = None
            hits = None

        dist_map = DistPixels(data, comm=self.comm, nnz=self._nnz, dtype=np.float64)
        if dist_map.data is not None:
            dist_map.data.fill(0)

        # FIXME: OpAccumDiag should support separate detweights for each observation
        OpAccumDiag(
            invnpp=invnpp,
            hits=hits,
            zmap=dist_map,
            name=self._name,
            detweights=detweights,
            common_flag_mask=self._common_flag_mask,
            flag_mask=self._flag_mask,
        ).exec(data)

        if white_noise_cov is None:
            if self._write_hits:
                hits.allreduce()
                fname = os.path.join(self._outdir, self._outprefix + "hits.fits")
                if self._zip_maps:
                    fname += ".gz"
                hits.write_healpix_fits(fname)

            invnpp.allreduce()
            if self._write_wcov:
                fname = os.path.join(self._outdir, self._outprefix + "wcov_inv.fits")
                if self._zip_maps:
                    fname += ".gz"
                    invnpp.write_healpix_fits(fname)

            covariance_invert(invnpp, self._rcond_limit)
            white_noise_cov = invnpp
            if self._write_wcov:
                fname = os.path.join(self._outdir, self._outprefix + "wcov.fits")
                if self._zip_maps:
                    fname += ".gz"
                    white_noise_cov.write_healpix_fits(fname)

        dist_map.allreduce()
        covariance_apply(white_noise_cov, dist_map)

        fname = os.path.join(self._outdir, self._outprefix + suffix + ".fits")
        if self._zip_maps:
            fname += ".gz"
        dist_map.write_healpix_fits(fname)

        return white_noise_cov

    @function_timer
    def _get_detweights(self, data):
        detweights = {}
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                if det not in detweights:
                    detweights[det] = 1e3
        return detweights

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
        detweights = self._get_detweights(data)

        white_noise_cov = None
        if self._write_binned:
            white_noise_cov = self._bin_map(data, detweights, "binned")

        if self.verbose and self.rank == 0:
            t0 = time()
            t1 = time()
            print("OpFilterBin: Filtering signal", flush=True)

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
            t1 = time()
            common_templates = self._build_common_templates(
                times, phase, local_intervals, common_flags
            )
            print("Built common templates in {:.3f} s".format(time() - t1), flush=True)

            for det in tod.local_dets:
                signal = tod.local_signal(det, self._name)
                flags = tod.local_flags(det, self._flag_name)
                good = np.logical_and(
                    (common_flags & self._common_flag_mask) == 0,
                    (flags & self._flag_mask) == 0,
                )
                if np.sum(good) == 0:
                    continue

                pixelsname = "{}_{}".format(self._pixels_name, det)
                weightsname = "{}_{}".format(self._weights_name, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                t1 = time()
                templates, template_covariance = self._build_templates(
                    times, phase, local_intervals, good, common_flags, common_templates
                )
                print("Built templates in {:.3f} s".format(time() - t1), flush=True)
                t1 = time()
                self._regress_templates(templates, template_covariance, signal, good)
                print("Regressed templates in {:.3f} s".format(time() - t1), flush=True)
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

        if self.verbose and self.rank == 0:
            print("OpFilterBin: Filtered signal in {:.1f} s".format(time() - t1), flush=True)
            print("OpFilterBin: Binning signal", flush=True)
            t1 = time()

        white_noise_cov = self._bin_map(data, detweights, "filtered", white_noise_cov)

        if self.verbose and self.rank == 0:
            print("OpFilterBin: Binned signal in {:.1f} s".format(time() - t1), flush=True)

        if obs_matrix is not None:
            if self.verbose and self.rank == 0:
                print("OpFilterBin: Noise-weighting observation matrix", flush=True)
                t1 = time()

            obs_matrix = self._noiseweight_obs_matrix(obs_matrix, white_noise_cov)

            if self.verbose and self.rank == 0:
                print("OpFilterBin: Noise-weighted observation matrix in {:.1f} s".format(
                    time() - t1), flush=True)
                print("OpFilterBin: Collecting observation matrix", flush=True)
                t1 = time()

            obs_matrix = self._collect_obs_matrix(obs_matrix)

            if self.verbose and self.rank == 0:
                print("OpFilterBin: Collected observation matrix in {:.1f} s".format(
                    time() - t1), flush=True)

        if self.verbose and self.rank == 0:
            print("OpFilterBin: Completed in {:.1f} s".format(time() - t0), flush=True)

        return
