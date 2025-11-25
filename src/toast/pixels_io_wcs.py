# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import astropy.io.fits as af
import h5py
import numpy as np
from astropy import units as u

from .mpi import MPI, use_mpi
from .pixels_io_utils import filename_is_fits, filename_is_hdf5
from .timing import function_timer
from .utils import Logger, memreport


def submap_to_image(dist, submap, sdata, image):
    """Helper function to unpack our data into a 3D image.

    This takes a single submap of data with flat pixels x n_values and
    unpacks that into an ndarray with n_values x 2D pixels.  The images
    are row major:

        ------>
        OOOOOOO
        XXXXXXX
        XXXXOOO
        OOOOOOO

    So a submap covers a range of columns, and can start and end in the middle
    of a row.

    Args:
        dist (PixelDistribution):  The pixel dist
        submap (int):  The global submap index
        sdata (array):  The data for a single submap
        image (array):  The 3D image array

    Returns:
        None

    """
    n_value, n_row, n_col = image.shape

    # Global pixel range of this submap
    s_offset = submap * dist.n_pix_submap
    s_end = s_offset + dist.n_pix_submap

    # Find which ndmap rows are covered by this submap
    first_row = s_offset // n_col
    last_row = min(s_end // n_col + 1, n_row)

    # Loop over output rows and assign data
    for row in range(first_row, last_row):
        row_offset = row * n_col  # First pixel of this row
        col_offset = 0  # Number of columns to skip on this row
        if row_offset < s_offset:
            col_offset = s_offset - row_offset
        n_copy = n_col - col_offset
        if row_offset + col_offset + n_copy > s_end:
            n_copy = s_end - row_offset - col_offset
        sbuf_offset = row_offset + col_offset - s_offset
        image[:, row, col_offset : col_offset + n_copy] = sdata[
            sbuf_offset : sbuf_offset + n_copy :
        ].T


def image_to_submap(dist, image, submap, sdata, scale=1.0):
    """Helper function to fill our data from a 3D image.

    This takes a single submap of data with flat pixels x n_values and fills
    it from an ndarray with n_values x 2D pixels.

    Args:
        dist (PixelDistribution):  The pixel dist
        image (array):  The image array
        submap (int):  The global submap index
        sdata (array):  The data for a single submap
        scale (float):  Scale factor.

    Returns:
        None

    """
    n_value, n_row, n_col = image.shape

    # Global pixel range of this submap
    s_offset = submap * dist.n_pix_submap
    s_end = s_offset + dist.n_pix_submap

    # Find which ndmap rows are covered by this submap
    first_row = s_offset // n_col
    last_row = min(s_end // n_col + 1, n_row)

    # Loop over output rows and assign data
    for row in range(first_row, last_row):
        row_offset = row * n_col  # First pixel of this row
        col_offset = 0  # Number of columns to skip on this row
        if row_offset < s_offset:
            col_offset = s_offset - row_offset
        n_copy = n_col - col_offset  # Number of columns to copy
        if row_offset + col_offset + n_copy > s_end:
            n_copy = s_end - row_offset - col_offset
        sbuf_offset = row_offset + col_offset - s_offset
        sdata[sbuf_offset : sbuf_offset + n_copy, :] = (
            scale * image[:, row, col_offset : col_offset + n_copy].T
        )


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
            owners[m] = rank
        allowners = np.zeros_like(owners)
        dist.comm.Allreduce(owners, allowners, op=MPI.MIN)

    # Create an image array for the output.  WCS shape is (n_row, n_col)
    image = None
    image_shape = (pix.n_value, dist.wcs_shape[0], dist.wcs_shape[1])
    if rank == 0:
        image = np.zeros(image_shape, dtype=pix.dtype)

    n_val_submap = dist.n_pix_submap * pix.n_value

    if dist.comm is None:
        # Just copy our local submaps into the image buffer
        for lc, sm in enumerate(dist.local_submaps):
            submap_to_image(dist, sm, pix.data[lc], image)
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
                        sendview[c, :, :] = pix.data[
                            dist.global_submap_to_local[submap_off + c], :, :
                        ]
                dist.comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)
                if rank == 0:
                    # copy into image buffer
                    for c in range(ncomm):
                        submap_to_image(dist, (submap_off + c), recvview[c], image)
                sendbuf.fill(0)
                if rank == 0:
                    recvbuf.fill(0)
            submap_off += ncomm
    return image


@function_timer
def broadcast_image(image, fscale, pix, comm_bytes):
    """Broadcast image across the distributed pixel data

    Args:
        image (ndarray):  Root process has the image.
            Unused on other processes.
        fscale (float):  Root process has any necessary scaling.
            Unused on other processes.
        pix (PixelData): The distributed PixelData object.
        comm_bytes (int): The approximate message size to use in bytes.
    """
    log = Logger.get()
    dist = pix.distribution
    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    n_val_submap = dist.n_pix_submap * pix.n_value

    if dist.comm is None:
        # Single process, just copy into place
        for sm in range(dist.n_submap):
            if sm in dist.local_submaps:
                loc = dist.global_submap_to_local[sm]
                image_to_submap(dist, image, sm, pix.data[loc], scale=fscale)
    else:
        # One reader broadcasts
        fscale = dist.comm.bcast(fscale, root=0)
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
                    image_to_submap(
                        dist, image, (submap_off + c), view[c], scale=fscale
                    )
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


@function_timer
def write_wcs(filename, image, wcs, units=None, dtype=None, extra_header=None):
    """Write a FITS or HDF5 WCS map on the calling process

    Args:
        filename (str):  The path to the file
        image (ndarray): 2 or 3-dimensional image
        wcs (astropy.WCS):  The World Coordinate System
        units (str):  Image units
        extra_header (dict):  Additional metadata to include with the map

    Returns:
        None
    """

    filename = filename.strip()
    if os.path.isfile(filename):
        os.remove(filename)

    # Basic wcs header
    header = wcs.to_header(relax=True)

    if extra_header is not None:
        header.update(extra_header)

    # Output units
    if dtype is None:
        dtype = image.dtype

    # Row-major to column major (FITS standanrd)
    # image = np.atleast_3d(image).transpose([0, 2, 1]).astype(dtype)
    image = np.atleast_3d(image).astype(dtype)

    if filename_is_fits(filename):
        # Add map dimensions
        header["NAXIS"] = image.ndim
        for i, n in enumerate(image.shape[::-1]):
            header[f"NAXIS{i + 1}"] = n
        if units is not None:
            # Add units
            header["BUNIT"] = str(units)
        hdus = af.HDUList([af.PrimaryHDU(image, header)])
        hdus.writeto(filename)
        del hdus
    elif filename_is_hdf5(filename):
        with h5py.File(filename, "w") as hfile:
            hfile["data"] = image
            for key, value in header.items():
                hfile[f"wcs/{key}"] = value
            if units is not None:
                # Add units
                hfile["bunit"] = str(units)
    else:
        msg = f"Could not ascertain file type for '{filename}'"
        raise RuntimeError(msg)
    return


@function_timer
def read_wcs(filename, units=False, extension=0, dtype=None):
    """Read a FITS or HDF5 WCS map serially.

    This reads the file into simple numpy arrays on the calling process.
    Units in the file are ignored.

    Args:
        filename (str):  The path to the file.
        units (bool):  If True, return the units of the map

    Returns:
        (tuple):  The map data and the appropriate astropy units.

    """

    log = Logger.get()

    filename = filename.strip()
    funits = None
    bunit = ""

    if filename_is_fits(filename):
        # Load a FITS format image
        with af.open(filename, mode="readonly") as hdul:
            hdu = hdul[extension]
            image = np.array(hdu.data)
            if units and "BUNIT" in hdu.header:
                bunit = hdu.header["BUNIT"]
    elif filename_is_hdf5(filename):
        # Load an HDF5 format image
        with h5py.File(filename, "r") as hfile:
            image = np.array(hfile["data"][()])
            if units and "bunit" in hfile:
                # Separately read the units
                bunit = hfile["bunit"][()].decode()
    else:
        msg = f"Could not ascertain file type for '{filename}'"
        raise RuntimeError(msg)

    # Column-major to row major
    # image = np.transpose(image, [0, 2, 1]).astype(dtype)
    image = np.atleast_3d(image).astype(dtype)

    # Optionally parse units
    if units:
        if bunit == "":
            funits = u.dimensionless_unscaled
        else:
            try:
                funits = u.Unit(bunit)
            except ValueError as e:
                log.warning(f"WARNING: failed to parse units in {filename}:\n{e}")
        result = (image, funits)
    else:
        result = image

    return result
