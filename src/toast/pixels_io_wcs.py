# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import astropy.io.fits as af
import numpy as np
import pixell
import pixell.enmap

from .mpi import MPI, use_mpi
from .timing import Timer, function_timer
from .utils import Logger, memreport


def submap_to_enmap(dist, submap, sdata, endata):
    """Helper function to unpack our data into pixell format

    This takes a single submap of data with flat pixels x n_values and unpacks
    that into an ndmap with n_values x 2D pixels.  The ndmaps are column
    major, like a FITS image:

        | OOXXXOO
        | OOXXXOO
        | OXXXOOO
        V OXXXOOO

    So a submap covers a range of columns, and can start and end in the middle
    of a column.

    Args:
        dist (PixelDistribution):  The pixel dist
        submap (int):  The global submap index
        sdata (array):  The data for a single submap
        endata (ndmap):  The pixell array

    Returns:
        None

    """
    enshape = endata.shape
    n_value = enshape[0]
    n_cols = enshape[1]
    n_rows = enshape[2]

    # Global pixel range of this submap
    s_offset = submap * dist.n_pix_submap
    s_end = s_offset + dist.n_pix_submap

    # Find which ndmap cols are covered by this submap
    first_col = s_offset // n_rows
    last_col = s_end // n_rows
    if last_col >= n_cols:
        last_col = n_cols - 1

    # Loop over output rows and assign data
    # print(f"endata shape = {endata.shape}")
    for ival in range(n_value):
        for col in range(first_col, last_col + 1):
            pix_offset = col * n_rows
            row_offset = 0
            if s_offset > pix_offset:
                row_offset = s_offset - pix_offset
            n_copy = n_rows - row_offset
            if pix_offset + n_copy > s_end:
                n_copy = s_end - pix_offset
            sbuf_offset = pix_offset + row_offset - s_offset
            # print(
            #     f"endata[{ival}, {col}, {row_offset}:{row_offset+n_copy}] = sdata[{sbuf_offset}:{sbuf_offset+n_copy}, {ival}]",
            #     flush=True,
            # )
            endata[ival, col, row_offset : row_offset + n_copy] = sdata[
                sbuf_offset : sbuf_offset + n_copy, ival
            ]


def enmap_to_submap(dist, endata, submap, sdata):
    """Helper function to fill our data from pixell format

    This takes a single submap of data with flat pixels x n_values and fills
    it from an ndmap with n_values x 2D pixels.

    Args:
        dist (PixelDistribution):  The pixel dist
        endata (ndmap):  The pixell array
        submap (int):  The global submap index
        sdata (array):  The data for a single submap

    Returns:
        None

    """
    enshape = endata.shape
    n_value = enshape[0]
    n_cols = enshape[1]
    n_rows = enshape[2]

    # Global pixel range of this submap
    s_offset = submap * dist.n_pix_submap
    s_end = s_offset + dist.n_pix_submap

    # Find which ndmap rows are covered by this submap
    first_col = s_offset // n_rows
    last_col = s_end // n_rows
    if last_col >= n_cols:
        last_col = n_cols - 1

    # Loop over output rows and assign data
    for ival in range(n_value):
        for col in range(first_col, last_col + 1):
            pix_offset = col * n_rows
            row_offset = 0
            if s_offset > pix_offset:
                row_offset = s_offset - pix_offset
            n_copy = n_rows - row_offset
            if pix_offset + n_copy > s_end:
                n_copy = s_end - pix_offset
            sbuf_offset = pix_offset + row_offset - s_offset
            # print(
            #     f"sdata[{sbuf_offset}:{sbuf_offset+n_copy}, {ival}] = endata[{ival}, {col}, {row_offset}:{row_offset+n_copy}]",
            #     flush=True,
            # )
            sdata[sbuf_offset : sbuf_offset + n_copy, ival] = endata[
                ival, col, row_offset : row_offset + n_copy
            ]


@function_timer
def collect_wcs_submaps(pix, comm_bytes=10000000):
    # The distribution
    dist = pix.distribution

    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    # We will reduce some number of whole submaps at a time.
    # Find the number of submaps that fit into the requested
    # communication size.
    comm_submap = pix.comm_nsubmap(comm_bytes)

    # Determine which processes should send submap data.  We do not use the
    # PixelDistribution.submap_owners here, since that is intended for operations
    # parallelized over submaps, and the submap owners do not necessarily have the
    # owned submaps in local memory.  Instead, we do a buffered allreduce.

    not_owned = None
    allowners = None
    if dist.comm is None:
        not_owned = 1
        allowners = np.zeros(dist.n_submap, dtype=np.int32)
        allowners.fill(not_owned)
        for m in dist.local_submaps:
            allowners[m] = rank
    else:
        not_owned = dist.comm.size
        owners = np.zeros(dist.n_submap, dtype=np.int32)
        owners.fill(not_owned)
        for m in dist.local_submaps:
            owners[m] = dist.comm.rank
        allowners = np.zeros_like(owners)
        dist.comm.Allreduce(owners, allowners, op=MPI.MIN)

    # Create a pixell map structure for the output
    endata = None
    if rank == 0:
        endata = pixell.enmap.zeros((pix.n_value,) + dist.wcs_shape, wcs=dist.wcs)

    n_val_submap = dist.n_pix_submap * pix.n_value

    if dist.comm is None:
        # Just copy our local submaps into the pixell buffer
        for lc, sm in enumerate(dist.local_submaps):
            submap_to_enmap(dist, sm, pix.data[lc], endata)
    else:
        sendbuf = np.zeros(comm_submap * n_val_submap, dtype=pix.dtype)
        sendview = sendbuf.reshape(comm_submap, dist.n_pix_submap, pix.n_value)

        recvbuf = None
        recvview = None
        if rank == 0:
            recvbuf = np.zeros(comm_submap * n_val_submap, dtype=pix.dtype)
            recvview = recvbuf.reshape(comm_submap, dist.n_pix_submap, pix.n_value)

        submap_off = 0
        ncomm = comm_submap
        while submap_off < dist.n_submap:
            if submap_off + ncomm > dist.n_submap:
                ncomm = dist.n_submap - submap_off
            if np.any(allowners[submap_off : submap_off + ncomm] != not_owned):
                # at least one submap has some hits.  reduce.
                for c in range(ncomm):
                    if allowners[submap_off + c] == dist.comm.rank:
                        # print(f"rank {dist.comm.rank}: set sendview[{c}, :, :]")
                        sendview[c, :, :] = pix.data[
                            dist.global_submap_to_local[submap_off + c], :, :
                        ]
                dist.comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)
                if rank == 0:
                    # copy into pixell buffer
                    for c in range(ncomm):
                        submap_to_enmap(dist, (submap_off + c), recvview[c], endata)
                sendbuf.fill(0)
                if rank == 0:
                    recvbuf.fill(0)
            submap_off += ncomm
    return endata


@function_timer
def write_wcs_fits(pix, path, comm_bytes=10000000, report_memory=False):
    """Write pixel data to a FITS image

    The data across all processes is assumed to be synchronized (the data for a given
    submap shared between processes is identical).  The submap data is sent to the root
    process which writes it out.

    Args:
        pix (PixelData): The distributed pixel object.
        path (str): The path to the output FITS file.
        comm_bytes (int): The approximate message size to use.
        report_memory (bool): Report the amount of available memory on the root
            node just before writing out the map.

    Returns:
        None

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    # The distribution
    dist = pix.distribution

    # Check that we have WCS information
    if not hasattr(dist, "wcs"):
        raise RuntimeError("Pixel distribution does not have WCS information")

    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    endata = collect_wcs_submaps(pix, comm_bytes=comm_bytes)

    if rank == 0:
        if os.path.isfile(path):
            os.remove(path)
        # Basic wcs header
        header = endata.wcs.to_header(relax=True)
        # Add map dimensions
        header["NAXIS"] = endata.ndim
        for i, n in enumerate(endata.shape[::-1]):
            header[f"NAXIS{i+1}"] = n
        hdus = af.HDUList([af.PrimaryHDU(endata, header)])
        hdus.writeto(path)

    del endata
    return


@function_timer
def read_wcs_fits(pix, path, ext=0, comm_bytes=10000000):
    """Read and broadcast pixel data stored in a FITS image.

    The root process opens the FITS file and broadcasts the data in units
    of the submap size.

    Args:
        pix (PixelData): The distributed PixelData object.
        path (str): The path to the FITS file.
        ext (int, str): Then index or name of the FITS image extension to load.
        comm_bytes (int): The approximate message size to use in bytes.

    Returns:
        None

    """
    dist = pix.distribution
    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    endata = None
    if rank == 0:
        # Load from disk
        endata = pixell.enmap.read_map(path, fmt="fits")
        # Check dimensions
        enpix = 1
        for s in endata.shape:
            enpix *= s
        tot_pix = dist.n_pix * pix.n_value
        if tot_pix != enpix:
            raise RuntimeError(
                f"Input file has {enpix} pixel values instead of {tot_pix}"
            )

    n_val_submap = dist.n_pix_submap * pix.n_value

    if dist.comm is None:
        # Single process, just copy into place
        for sm in range(dist.n_submap):
            if sm in dist.local_submaps:
                loc = dist.global_submap_to_local[sm]
                enmap_to_submap(dist, endata, sm, pix.data[loc])
    else:
        # One reader broadcasts
        comm_submap = pix.comm_nsubmap(comm_bytes)

        buf = np.zeros(comm_submap * n_val_submap, dtype=pix.dtype)
        view = buf.reshape(comm_submap, dist.n_pix_submap, pix.n_value)
        submap_off = 0
        ncomm = comm_submap
        while submap_off < dist.n_submap:
            if submap_off + ncomm > dist.n_submap:
                ncomm = dist.n_submap - submap_off
            if rank == 0:
                # Fill the bcast buffer
                for c in range(ncomm):
                    enmap_to_submap(dist, endata, (submap_off + c), view[c])
            # Broadcast
            dist.comm.Bcast(buf, root=0)
            # Copy these submaps into local data
            for sm in range(submap_off, submap_off + ncomm):
                if sm in dist.local_submaps:
                    loc = dist.global_submap_to_local[sm]
                    pix.data[loc, :, :] = view[sm - submap_off, :, :]
            submap_off += comm_submap
            buf.fill(0)

    return
