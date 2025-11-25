# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re

import healpy as hp
import numpy as np
import scipy.io
import scipy.sparse

from ..covariance import covariance_invert
from ..mpi import get_world
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import read_healpix
from ..timing import Timer, function_timer
from ..utils import Logger, memreport


class ObsMat(object):
    """Observation Matrix class"""

    def __init__(self, filename=None, matrix=None):
        self.filename = filename
        self.matrix = matrix
        self.load()
        return

    @function_timer
    def load(self, filename=None):
        if filename is not None:
            self.filename = filename
            self.matrix = None
        if self.matrix is None and self.filename is not None:
            self.matrix = scipy.sparse.load_npz(self.filename)
        if self.matrix is None:
            self.nnz = 0
            self.nrow, self.ncol = 0, 0
            return
        self.nnz = self.matrix.nnz
        if self.nnz < 0:
            msg = f"Overflow in {self.filename}: nnz = {self.nnz}"
            raise RuntimeError(msg)
        self.nrow, self.ncol = self.matrix.shape
        return

    @function_timer
    def apply(self, map_in):
        nmap, npix = np.atleast_2d(map_in).shape
        npixtot = np.prod(map_in.shape)
        if npixtot != self.ncol:
            msg = f"Map is incompatible with the observation matrix. "
            msg += f"shape(matrix) = {self.matrix.shape}, shape(map) = {map_in.shape}"
            raise RuntimeError(msg)
        map_out = self.matrix.dot(map_in.ravel())
        if nmap != 1:
            map_out = map_out.reshape([nmap, -1])
        return map_out

    def sort_indices(self):
        self.matrix.sort_indices()

    @property
    def data(self):
        return self.matrix.data

    @property
    def dtype(self):
        return self.matrix.data.dtype

    def __iadd__(self, other):
        if hasattr(other, "matrix"):
            self.matrix += other.matrix
        else:
            self.matrix += other
        return self

    def __imul__(self, other):
        if hasattr(other, "matrix"):
            self.matrix *= other.matrix
        else:
            self.matrix *= other
        return self


def coadd_observation_matrix(
    inmatrix,
    outmatrix,
    file_mask=None,
    file_invcov=None,
    file_cov=None,
    nside_submap=16,
    rcond_limit=1e-3,
    double_precision=False,
    comm=None,
    save_memory=False,
):
    """Co-add noise-weighted observation matrices

    Args:
        inmatrix(iterable) : One or more noise-weighted observation
            matrix files.  If a matrix is used to model several similar
            observations, append `+N` to the file name to indicate the
            multiplicity.
        outmatrix(string) : Name of output file.  If it includes the
            string 'noiseweighted', the output matrix will be
            noise-weighted like the inputs.
        file_mask(string) : If provided, only pixels with non-zero value
            in the mask will be present in `outmatrix`.
        file_invcov(string) : Name of output inverse covariance file
        file_cov(string) : Name of output covariance file
        nside_submap(int) : Submap size is 12 * nside_submap ** 2.
            Number of submaps is (nside / nside_submap) ** 2
        rcond_limit(float) : "Reciprocal condition number limit
        double_precision(bool) : Output in double precision
        save_memory(bool) : Save memory by loading the indices repeated
    """

    log = Logger.get()
    if comm is None:
        comm, ntask, rank = get_world()
    else:
        ntask = comm.size
        rank = comm.rank
    timer0 = Timer()
    timer1 = Timer()
    timer0.start()
    timer1.start()

    if double_precision:
        dtype = np.float64
    else:
        dtype = np.float32

    if len(inmatrix) == 1:
        # Only one file provided, try interpreting it as a text file with a list
        try:
            with open(inmatrix[0], "r") as listfile:
                infiles = listfile.readlines()
            log.info_rank(f"Loaded {inmatrix[0]} in", timer=timer1, comm=comm)
        except UnicodeDecodeError:
            # Didn't work. Assume that user supplied a single matrix file
            infiles = inmatrix
    else:
        infiles = inmatrix

    if file_mask is None:
        mask = None
    else:
        mask = hp.read_map(file_mask, nest=True)

    obs_matrix_sum = None
    invcov_sum = None
    nnz = None
    npix = None
    if "noiseweighted" in outmatrix:
        deweight = False
        log.info_rank(
            f"Output matrix is labelled 'noiseweighted' and will not be de-weighted"
        )
    else:
        deweight = True

    mem = memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"Before loading any matrices:  {mem}", comm)

    infiles_invcov = []
    infiles_obsmat = {}  # filename : multiplicity

    # Process matrix names and multiplicities
    for infile_matrix in infiles:
        infile_matrix = infile_matrix.strip()
        if "noiseweighted" not in infile_matrix:
            msg = (
                f"Observation matrix does not seem to be "
                f"noise-weighted: '{infile_matrix}'"
            )
            raise RuntimeError(msg)
        if "+" in infile_matrix:
            infile_matrix, N = infile_matrix.split("+")
            N = float(N)
        else:
            N = 1
        if not os.path.isfile(infile_matrix):
            msg = f"Matrix not found: {infile_matrix}"
            raise RuntimeError(msg)
        infiles_obsmat[infile_matrix] = N
        # We'll need the white noise covariance as well.  Synthesize the name here but
        # load the invcov matrices separately
        infile_invcov = infile_matrix.replace("noiseweighted_obs_matrix.npz", "invcov")
        if os.path.isfile(infile_invcov + ".fits"):
            infile_invcov += ".fits"
        elif os.path.isfile(infile_invcov + ".h5"):
            infile_invcov += ".h5"
        else:
            msg = (
                f"Cannot find an inverse covariance matrix to go "
                "with '{infile_matrix}'"
            )
            raise RuntimeError(msg)
        infiles_invcov.append(infile_invcov)

    # Now process the observation matrix
    if save_memory or mask is not None:
        # Assume that we cannot keep three observation matrices in memory at
        # the same time.  We can save memory by loading the matrix indices repeatedly
        indices_by_row = {}
        nrow, ncol = None, None
        for infile_matrix in infiles_obsmat:
            log.info_rank(f"Loading indices from {infile_matrix}", comm)
            with np.load(infile_matrix) as obsmat:
                indices = obsmat["indices"].astype(np.int32)
                indptr = obsmat["indptr"]
                format_ = obsmat["format"]
                shape = obsmat["shape"]
            del obsmat
            log.info_rank(f"Loaded indices from {infile_matrix} in", timer=timer1, comm=None)
            mem = memreport(msg="(whole node)", comm=comm, silent=True)
            log.info_rank(f"After loading indices from {infile_matrix}:  {mem}", comm)
            log.info_rank(f"Processing indices from {infile_matrix}", comm)
            if format_ != b"csr":
                msg = f"{infile_matrix} format is {format_}, not CSR"
                raise RuntimeError(msg)
            if nrow is None:
                nrow, ncol = shape
            elif shape[0] != nrow or shape[1] != ncol:
                msg = f"{infile_matrix} shape is {shape}, not {[nrow, ncol]}"
                raise RuntimeError(msg)
            # Loop over the rows, accumulating hit columns
            for row in range(nrow):
                if mask is not None and mask[row % mask.size] == 0:
                    continue
                istart = indptr[row]
                istop = indptr[row + 1]
                if istop == istart:
                    continue
                print(f"DEBUG : {row} / {nrow}", flush=True)  # DEBUG
                cols = indices[istart : istop]
                if row not in indices_by_row:
                    indices_by_row[row] = cols
                else:
                    # Merge old and new indices but reject duplicates
                    old = indices_by_row[row]
                    new = cols
                    ind = np.searchsorted(old, new)
                    # Isolate the last elements of `new` that are larger
                    # than any element in `old`
                    end = np.searchsorted(ind, old.size)
                    new_start = cols[:end]
                    new_end = cols[end:]
                    ind_start = ind[:end]
                    good = old[ind_start] != new_start
                    indices_by_row[row] = np.sort(np.hstack(
                        [old, new_start[good], new_end]
                    ))
            del indices
            del indptr
            log.info_rank(f"Processed indices from {infile_matrix} in", timer=timer1, comm=None)
            mem = memreport(msg="(whole node)", comm=comm, silent=True)
            log.info_rank(f"After processing indices from {infile_matrix}:  {mem}", comm)
        # Turn sets into integer vectors
        log.info_rank(f"Sorting indices", comm)
        ndata = 0
        for row in indices_by_row:
            if mask is not None and mask[row % mask.size] == 0:
                continue
            ndata += indices_by_row[row].size
        log.info_rank(f"Sorted {ndata} indices in", timer=timer1, comm=None)
        # Now load the observation matrices *again*, accumulating the
        # data directly in the target matrix
        total_data = np.zeros(ndata, np.float64)
        total_indices = np.zeros(ndata, np.int32)
        total_indptr = np.zeros(nrow + 1, np.int64)
        mem = memreport(msg="(whole node)", comm=comm, silent=True)
        log.info_rank(f"After allocating total matrix:  {mem}", comm)
        offset = 0
        for row in range(nrow):
            if row in indices_by_row:
                cols = indices_by_row[row]
                n = cols.size
                total_indices[offset : offset + n] = cols
                offset += n
            total_indptr[row + 1] = offset
        del indices_by_row
        for infile_matrix in infiles_obsmat:
            log.info_rank(f"Loading data from {infile_matrix}", comm)
            with np.load(infile_matrix) as obsmat:
                indices = obsmat["indices"].astype(np.int32)
                indptr = obsmat["indptr"]
                data = obsmat["data"].astype(np.float32)
            del obsmat
            log.info_rank(f"Loaded {infile_matrix} in", timer=timer1, comm=None)
            mem = memreport(msg="(whole node)", comm=comm, silent=True)
            log.info_rank(f"After loading {infile_matrix}:  {mem}", comm)
            # Loop over the rows, accumulating hit columns
            log.info_rank(f"Co-adding data from {infile_matrix}", comm)
            for row in range(nrow):
                if mask is not None and mask[row % mask.size] == 0:
                    continue
                istart = indptr[row]
                istop = indptr[row + 1]
                if istart == istop:
                    continue
                col_ind = indices[istart : istop]
                col_data = data[istart : istop]
                total_slice = slice(total_indptr[row], total_indptr[row + 1])
                total_ind = total_indices[total_slice]
                ind = np.searchsorted(total_ind, col_ind)
                total_data[total_slice][ind] += col_data
            del indices
            del indptr
            del data
            log.info_rank(f"Co-added data from {infile_matrix} in", timer=timer1, comm=None)
            mem = memreport(msg="(whole node)", comm=comm, silent=True)
            log.info_rank(f"After co-adding data from {infile_matrix}:  {mem}", comm)
        log.info_rank(f"Instantiating the observation matrix", comm)
        matrix = scipy.sparse.csr_matrix(
            (total_data, total_indices, total_indptr),
            shape=(nrow, ncol),
            dtype=np.float32,
        )
        del total_data
        del total_indices
        del total_indptr
        obs_matrix_sum = ObsMat(matrix=matrix)
        del matrix
        log.info_rank(f"Instantiated observation matrix in", timer=timer1, comm=None)
    else:
        for infile_matrix, N in infiles_obsmat.items():
            log.info_rank(f"Loading {infile_matrix}", comm)
            obs_matrix = ObsMat(infile_matrix)
            log.info_rank(f"Loaded {infile_matrix} in", timer=timer1, comm=None)
            mem = memreport(msg="(whole node)", comm=comm, silent=True)
            log.info_rank(f"After loading {infile_matrix}:  {mem}", comm)
            if N != 1:
                obs_matrix *= N
            if obs_matrix_sum is None:
                obs_matrix_sum = obs_matrix
            else:
                log.info(f"Co-adding {infile_matrix}")
                obs_matrix_sum += obs_matrix
                log.info_rank(f"Co-added {infile_matrix} in", timer=timer1, comm=None)
            del obs_matrix
            mem = memreport(msg="(whole node)", comm=comm, silent=True)
            log.info_rank(f"After adding {infile_matrix}:  {mem}", comm)

    for infile_invcov in infiles_invcov:
        log.info(f"Loading {infile_invcov}")
        # Always co-add in double precision
        invcov = read_healpix(
            infile_invcov, None, nest=True, dtype=np.float64, verbose=False
        )
        invcov = np.atleast_2d(invcov)
        if N != 1:
            invcov *= N
        if invcov_sum is None:
            invcov_sum = invcov
            nnzcov, npix = invcov.shape
            nnz = 1
            while (nnz * (nnz + 1)) // 2 != nnzcov:
                nnz += 1
            npixtot = npix * nnz
        else:
            invcov_sum += invcov
        log.info_rank(f"Loaded {infile_invcov} in", timer=timer1, comm=None)
        mem = memreport(msg="(whole node)", comm=comm, silent=True)
        log.info_rank(f"After loading {infile_invcov}:  {mem}", comm)

    # Put the inverse white noise covariance in a TOAST pixel object

    npix_submap = 12 * nside_submap**2
    nsubmap = npix // npix_submap
    local_submaps = [submap for submap in range(nsubmap) if submap % ntask == rank]
    dist = PixelDistribution(
        n_pix=npix, n_submap=nsubmap, local_submaps=local_submaps, comm=comm
    )
    dist.nest = True
    dist_cov = PixelData(dist, np.float64, n_value=nnzcov)
    for local_submap, global_submap in enumerate(local_submaps):
        pix_start = global_submap * npix_submap
        pix_stop = pix_start + npix_submap
        dist_cov.data[local_submap] = invcov_sum[:, pix_start:pix_stop].T
    del invcov_sum
    mem = memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After distributing inverse covariance:  {mem}", comm)

    # Optionally write out the inverse white noise covariance

    if file_invcov is not None:
        log.info_rank(f"Writing {file_invcov}", comm=comm)
        dist_cov.write(
            file_invcov,
            single_precision=not double_precision,
        )
        log.info_rank(f"Wrote {file_invcov}", timer=timer1, comm=comm)

    # Invert the white noise covariance

    log.info_rank("Inverting white noise matrices", comm=comm)
    dist_rcond = PixelData(dist, float, n_value=1)
    covariance_invert(dist_cov, rcond_limit, rcond=dist_rcond, use_alltoallv=True)
    log.info_rank("Inverted white noise matrices in", timer=timer1, comm=comm)
    mem = memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After inverting inverse covariance:  {mem}", comm)

    # Optionally write out the white noise covariance

    if file_cov is not None:
        log.info_rank(f"Writing {file_cov}", comm=comm)
        dist_cov.write(
            file_cov,
            single_precision=not double_precision,
        )
        log.info_rank(f"Wrote {file_cov} in", timer=timer1, comm=comm)

    if deweight:
        # De-weight the observation matrix
        log.info_rank("De-weighting obs matrix", comm=comm)
        cc = scipy.sparse.dok_matrix((npixtot, npixtot), dtype=obs_matrix_sum.dtype)
        nsubmap = dist_cov.distribution.n_submap
        npix_submap = dist_cov.distribution.n_pix_submap
        for isubmap_local, isubmap_global in enumerate(
            dist_cov.distribution.local_submaps
        ):
            submap = dist_cov.data[isubmap_local]
            offset = isubmap_global * npix_submap
            for pix_local in range(npix_submap):
                if np.all(submap[pix_local] == 0):
                    continue
                pix = pix_local + offset
                icov = 0
                for inz in range(nnz):
                    for jnz in range(inz, nnz):
                        cc[pix + inz * npix, pix + jnz * npix] = submap[pix_local, icov]
                        if inz != jnz:
                            cc[pix + jnz * npix, pix + inz * npix] = submap[
                                pix_local, icov
                            ]
                        icov += 1
        cc = cc.tocsr()
        mem = memreport(msg="(whole node)", comm=comm, silent=True)
        log.info_rank(f"Before de-weighting obs matrix:  {mem}", comm)
        obs_matrix_sum = cc.dot(obs_matrix_sum.matrix)
        log.info_rank(f"De-weighted obs matrix in", timer=timer1, comm=comm)
        mem = memreport(msg="(whole node)", comm=comm, silent=True)
        log.info_rank(f"After de-weighting obs matrix:  {mem}", comm)
    else:
        # No deweighting, just extract the sparse matrix from the ObsMat object
        obs_matrix_sum = obs_matrix_sum.matrix

    # Write out the co-added and de-weighted matrix

    if not outmatrix.endswith(".npz"):
        outmatrix += ".npz"
    log.info_rank(f"Writing {outmatrix}", comm=comm)
    scipy.sparse.save_npz(outmatrix, obs_matrix_sum.astype(dtype))
    log.info_rank(f"Wrote {outmatrix} in", timer=timer1, comm=comm)

    log.info_rank(f"Co-added and de-weighted obs matrix in", timer=timer0, comm=comm)

    return outmatrix
