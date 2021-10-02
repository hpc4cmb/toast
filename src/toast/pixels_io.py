# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .timing import function_timer, Timer

from .mpi import MPI

import healpy as hp

from .utils import Logger, memreport


@function_timer
def read_healpix_fits(pix, path, nest=True, comm_bytes=10000000):
    """Read and broadcast a HEALPix FITS table.

    The root process opens the FITS file in memmap mode and iterates over
    chunks of the map in a way to minimize cache misses in the internal
    FITS buffer.  Chunks of submaps are broadcast to all processes, and
    each process copies data to its local submaps.

    Args:
        pix (PixelData): The distributed PixelData object.
        path (str): The path to the FITS file.
        nest (bool): If True, convert input to NESTED ordering, else use RING.
        comm_bytes (int): The approximate message size to use in bytes.

    Returns:
        None

    """
    dist = pix.distribution
    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    comm_submap = pix.comm_nsubmap(comm_bytes)

    # we make the assumption that FITS binary tables are still stored in
    # blocks of 2880 bytes just like always...
    dbytes = pix.dtype.itemsize
    rowbytes = pix.n_value * dbytes
    optrows = 2880 // rowbytes

    # get a tuple of all columns in the table.  We choose memmap here so
    # that we can (hopefully) read through all columns in chunks such that
    # we only ever have a couple FITS blocks in memory.
    fdata = None
    if rank == 0:
        # Check that the file is in expected format
        errors = ""
        h = hp.fitsfunc.pf.open(path, "readonly")
        nside = hp.npix2nside(dist.n_pix)
        nside_map = h[1].header["nside"]
        if nside_map != nside:
            errors += "Wrong NSide: {} has {}, expected {}\n" "".format(
                path, nside_map, nside
            )
        map_nnz = h[1].header["tfields"]
        if map_nnz != pix.n_value:
            errors += "Wrong number of columns: {} has {}, expected {}\n" "".format(
                path, map_nnz, pix.n_value
            )
        h.close()
        if len(errors) != 0:
            raise RuntimeError(errors)
        # Now read the map
        fdata = hp.read_map(
            path,
            field=tuple([x for x in range(pix.n_value)]),
            dtype=[pix.dtype for x in range(pix.n_value)],
            memmap=True,
            nest=nest,
            verbose=False,
        )
        if pix.n_value == 1:
            fdata = (fdata,)

    buf = np.zeros(comm_submap * dist.n_pix_submap * pix.n_value, dtype=pix.dtype)
    view = buf.reshape(comm_submap, dist.n_pix_submap, pix.n_value)

    in_off = 0
    out_off = 0
    submap_off = 0

    rows = optrows
    while in_off < dist.n_pix:
        if in_off + rows > dist.n_pix:
            rows = dist.n_pix - in_off
        # is this the last block for this communication?
        islast = False
        copyrows = rows
        if out_off + rows > (comm_submap * dist.n_pix_submap):
            copyrows = (comm_submap * dist.n_pix_submap) - out_off
            islast = True

        if rank == 0:
            for col in range(pix.n_value):
                coloff = (out_off * pix.n_value) + col
                buf[coloff : coloff + (copyrows * pix.n_value) : pix.n_value] = fdata[
                    col
                ][in_off : in_off + copyrows]

        out_off += copyrows
        in_off += copyrows

        if islast:
            if dist.comm is not None:
                dist.comm.Bcast(buf, root=0)
            # loop over these submaps, and copy any that we are assigned
            for sm in range(submap_off, submap_off + comm_submap):
                if sm in dist.local_submaps:
                    loc = dist.global_submap_to_local[sm]
                    pix.data[loc, :, :] = view[sm - submap_off, :, :]
            out_off = 0
            submap_off += comm_submap
            buf.fill(0)
            islast = False

    # flush the remaining buffer
    if out_off > 0:
        if dist.comm is not None:
            dist.comm.Bcast(buf, root=0)
        # loop over these submaps, and copy any that we are assigned
        for sm in range(submap_off, submap_off + comm_submap):
            if sm in dist.local_submaps:
                loc = dist.global_submap_to_local[sm]
                pix.data[loc, :, :] = view[sm - submap_off, :, :]
    return


@function_timer
def collect_submaps(pix, comm_bytes=10000000):
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
    # owned submaps in local memory.  Instead, we do a buffered allreduce.  For dumping
    # large maps, we should be using HDF5 anyway.

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

    # This function requires lots of RAM, since it accumulates the full map on one
    # process before writing.  We also have to "unpack" the pixel data since the healpy
    # write function requires a list of maps.

    fdata = None
    fview = None
    if rank == 0:
        fdata = list()
        fview = list()
        for col in range(pix.n_value):
            fdata.append(pix.storage_class.zeros(dist.n_pix))
            fview.append(fdata[-1].array())

    if dist.comm is None:
        # Just copy our local submaps into the FITS buffers
        for lc, sm in enumerate(dist.local_submaps):
            global_offset = sm * dist.n_pix_submap
            n_copy = dist.n_pix_submap
            if global_offset + n_copy > dist.n_pix:
                n_copy = dist.n_pix - global_offset
            for col in range(pix.n_value):
                fview[col][global_offset : global_offset + n_copy] = pix.data[
                    lc, 0:n_copy, col
                ]
    else:
        sendbuf = np.zeros(
            comm_submap * dist.n_pix_submap * pix.n_value, dtype=pix.dtype
        )
        sendview = sendbuf.reshape(comm_submap, dist.n_pix_submap, pix.n_value)

        recvbuf = None
        recvview = None
        if rank == 0:
            recvbuf = np.zeros(
                comm_submap * dist.n_pix_submap * pix.n_value, dtype=pix.dtype
            )
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
                    # copy into FITS buffers
                    for c in range(ncomm):
                        global_offset = (submap_off + c) * dist.n_pix_submap
                        n_copy = dist.n_pix_submap
                        if global_offset + n_copy > dist.n_pix:
                            n_copy = dist.n_pix - global_offset
                        for col in range(pix.n_value):
                            fview[col][
                                global_offset : global_offset + n_copy
                            ] = recvview[c, 0:n_copy, col]
                sendbuf.fill(0)
                if rank == 0:
                    recvbuf.fill(0)
            submap_off += ncomm

    return fdata, fview


@function_timer
def write_healpix_fits(pix, path, nest=True, comm_bytes=10000000, report_memory=False):
    """Write pixel data to a HEALPix format FITS table.

    The data across all processes is assumed to be synchronized (the data for a given
    submap shared between processes is identical).  The submap data is sent to the root
    process which writes it out.  For parallel writing, see write_hdf5().

    Args:
        pix (PixelData): The distributed pixel object.
        path (str): The path to the output FITS file.
        nest (bool): If True, data is in NESTED ordering, else data is in RING.
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

    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    fdata, fview = collect_submaps(pix, comm_bytes=comm_bytes)

    log.info_rank(f"Collected submaps in", comm=dist.comm, timer=timer)

    if rank == 0:
        if os.path.isfile(path):
            os.remove(path)
        dtypes = [np.dtype(pix.dtype) for x in range(pix.n_value)]
        if report_memory:
            mem = memreport(msg="(root node)", silent=True)
            log.info_rank(f"About to write {path}:  {mem}")
        hp.write_map(path, fview, dtype=dtypes, fits_IDL=False, nest=nest)
        del fview
        for col in range(pix.n_value):
            fdata[col].clear()
        del fdata

    log.info_rank(f"Wrote map in", comm=dist.comm, timer=timer)

    return


def read_hdf5(pix, path):
    pass


def write_healpix_hdf5(pix, path, nest=True, comm_bytes=10000000, report_memory=False):
    """Write pixel data to a HEALPix format HDF5 dataset.

    The data across all processes is assumed to be synchronized (the data for a given
    submap shared between processes is identical).  The submap data is sent to the root
    process which writes it out.  For parallel writing, see write_hdf5().

    Args:
        pix (PixelData): The distributed pixel object.
        path (str): The path to the output FITS file.
        nest (bool): If True, data is in NESTED ordering, else data is in RING.
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

    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    fdata, fview = collect_submaps(pix, comm_bytes=comm_bytes)

    log.info_rank(f"Collected submaps in", comm=dist.comm, timer=timer)

    if rank == 0:
        if os.path.isfile(path):
            os.remove(path)
        # Write hdf5 as a test
        import h5py
        with h5py.File(path, "w") as f:
            dset = f.create_dataset("map", data=np.vstack(fview))
            dset.attrs["NESTED"] = nest
        del fview
        for col in range(pix.n_value):
            fdata[col].clear()
        del fdata

    log.info_rank(f"Wrote map in", comm=dist.comm, timer=timer)

    return
