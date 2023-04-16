# Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import healpy as hp
import numpy as np
from astropy import units as u

from .io import have_hdf5_parallel
from .mpi import MPI, use_mpi
from .timing import Timer, function_timer
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
    log = Logger.get()
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
    fscale = 1.0
    if rank == 0:
        # Check that the file is in expected format
        errors = ""
        h = hp.fitsfunc.pf.open(path, "readonly")
        nside = hp.npix2nside(dist.n_pix)
        nside_map = h[1].header["nside"]
        if "TUNIT1" in h[1].header:
            if h[1].header["TUNIT1"] == "":
                funits = u.dimensionless_unscaled
            elif h[1].header["TUNIT1"] == "K":
                funits = u.K
            elif h[1].header["TUNIT1"] == "mK":
                funits = u.mK
            elif h[1].header["TUNIT1"] == "uK":
                funits = u.uK
            else:
                funits = u.Unit(h[1].header["TUNIT1"])
        else:
            msg = f"Pixel data in {path} does not have TUNIT1 key.  "
            msg += f"Assuming '{pix.units}'."
            log.info(msg)
            funits = pix.units
        if funits != pix.units:
            scale = 1.0 * funits
            scale.to(pix.units)
            fscale = scale.value

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
        )
        if pix.n_value == 1:
            fdata = (fdata,)

    if dist.comm is not None:
        fscale = dist.comm.bcast(fscale, root=0)
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
                    pix.data[loc, :, :] = fscale * view[sm - submap_off, :, :]
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
                pix.data[loc, :, :] = fscale * view[sm - submap_off, :, :]
    return


@function_timer
def collect_healpix_submaps(pix, comm_bytes=10000000):
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
def write_healpix_fits(
    pix,
    path,
    nest=True,
    comm_bytes=10000000,
    report_memory=False,
    single_precision=False,
):
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
        single_precision (bool): Cast float and integer maps to single precision.

    Returns:
        None

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    # The distribution
    dist = pix.distribution

    # Unit string to write
    if pix.units == u.K:
        funits = "K"
    elif pix.units == u.mK:
        funits = "mK"
    elif pix.units == u.uK:
        funits = "uK"
    else:
        funits = str(pix.units)

    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    fdata, fview = collect_healpix_submaps(pix, comm_bytes=comm_bytes)

    if rank == 0:
        if os.path.isfile(path):
            os.remove(path)
        dtypes = [np.dtype(pix.dtype) for x in range(pix.n_value)]
        if single_precision:
            for i, dtype in enumerate(dtypes):
                if dtype == np.float64:
                    dtypes[i] = np.float32
                elif dtype == np.int64:
                    dtypes[i] = np.int32
        if report_memory:
            mem = memreport(msg="(root node)", silent=True)
            log.info_rank(f"About to write {path}:  {mem}")
        extra = [(f"TUNIT{x}", f"{funits}") for x in range(pix.n_value)]
        hp.write_map(
            path, fview, dtype=dtypes, fits_IDL=False, nest=nest, extra_header=extra
        )
        del fview
        for col in range(pix.n_value):
            fdata[col].clear()
        del fdata

    return


@function_timer
def read_healpix_hdf5(pix, path, nest=True, comm_bytes=10000000):
    """Read and broadcast a HEALPix map from an HDF5 file.

    The root process opens the file and iterates over chunks of the map.
    Chunks of submaps are broadcast to all processes, and each process
    copies data to its local submaps.

    Args:
        pix (PixelData): The distributed PixelData object.
        path (str): The path to the FITS file.
        nest (bool): If True, convert input to NESTED ordering, else use RING.
        comm_bytes (int): The approximate message size to use in bytes.

    Returns:
        None

    """
    log = Logger.get()
    timer = Timer()
    timer.start()

    dist = pix.distribution
    rank = 0
    if dist.comm is not None:
        rank = dist.comm.rank

    comm_submap = pix.comm_nsubmap(comm_bytes)

    fdata = None
    fscale = 1.0
    if rank == 0:
        try:
            f = h5py.File(path, "r")
        except OSError as e:
            msg = f"Failed to open {path} for reading: {e}"
            raise RuntimeError(msg)

        dset = f["map"]
        header = dict(dset.attrs)
        nside_file = header["NSIDE"]
        nside = hp.npix2nside(dist.n_pix)
        if nside_file != nside:
            msg = f"Wrong resolution in {path}: expected {nside} but found {nside_file}"
            raise RuntimeError(msg)
        if header["ORDERING"] == "NESTED":
            file_nested = True
        elif header["ORDERING"] == "RING":
            file_nested = False
        else:
            msg = f"Could not determine {path} pixel ordering."
            raise RuntimeError(msg)
        if "UNITS" in header:
            if header["UNITS"] == "":
                funits = u.dimensionless_unscaled
            else:
                funits = u.Unit(header["UNITS"])
        else:
            msg = f"Pixel data in {path} does not have UNITS.  Assuming {pix.units}."
            log.info(msg)
            funits = pix.units
        if funits != pix.units:
            scale = 1.0 * funits
            scale.to(pix.units)
            fscale = scale.value

        nnz, npix = dset.shape
        if nnz < pix.n_value:
            msg = f"Map in {path} has {nnz} columns but we require {pix.n_value}."
            raise RuntimeError(msg)
        if file_nested != nest:
            log.warning(
                f"{path} has ORDERING={header['ORDERING']}, "
                f"reordering serially upon load."
            )
            if file_nested and not nest:
                mapdata = hp.reorder(dset[:], n2r=True)
            elif not file_nested and nest:
                mapdata = hp.reorder(dset[:], r2n=True)
        else:
            # No reorder, we'll only load what we need
            mapdata = dset

    if dist.comm is not None:
        fscale = dist.comm.bcast(fscale, root=0)

    buf = np.zeros(comm_submap * pix.n_value * dist.n_pix_submap, dtype=pix.dtype)
    view = buf.reshape(comm_submap, pix.n_value, dist.n_pix_submap)

    # Load and broadcast submaps that are used
    hit_submaps = dist.all_hit_submaps
    n_hit_submaps = len(hit_submaps)

    submap_offset = 0
    while submap_offset < n_hit_submaps:
        submap_last = min(submap_offset + comm_submap, n_hit_submaps)
        if rank == 0:
            for i, submap in enumerate(hit_submaps[submap_offset:submap_last]):
                pix_offset = submap * dist.n_pix_submap
                # Healpix submaps are always complete but capping the upper
                # limit helps when users are embedding other pixelizations
                # into Healpix
                pix_last = min(pix_offset + dist.n_pix_submap, dist.n_pix)
                view[i, :, 0 : pix_last - pix_offset] = mapdata[
                    0 : pix.n_value, pix_offset:pix_last
                ]

        if dist.comm is not None:
            dist.comm.Bcast(buf, root=0)

        # loop over these submaps, and copy any that we are assigned
        for i, submap in enumerate(hit_submaps[submap_offset:submap_last]):
            if submap in dist.local_submaps:
                loc = dist.global_submap_to_local[submap]
                pix.data[loc] = fscale * view[i].T

        submap_offset = submap_last

    if rank == 0:
        f.close()

    return


@function_timer
def write_healpix_hdf5(
    pix, path, nest=True, comm_bytes=10000000, single_precision=False, force_serial=True
):
    """Write pixel data to a HEALPix format HDF5 dataset.

    The data across all processes is assumed to be synchronized (the data for a given
    submap shared between processes is identical).

    Args:
        pix (PixelData): The distributed pixel object.
        path (str): The path to the output FITS file.
        nest (bool): If True, data is in NESTED ordering, else data is in RING.
        comm_bytes (int): The approximate message size to use.
        single_precision (bool): If True, write floats and integers in single precision
        force_serial (bool):  If True, use the serial h5py implementation, even if
            parallel support is available.

    Returns:
        None

    """
    log = Logger.get()

    # The distribution
    dist = pix.distribution

    rank = 0
    ntask = 1
    if dist.comm is not None:
        rank = dist.comm.rank
        ntask = dist.comm.size

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

    header = {}
    if nest:
        header["ORDERING"] = "NESTED"
    else:
        header["ORDERING"] = "RING"
    header["NSIDE"] = hp.npix2nside(dist.n_pix)
    header["UNITS"] = str(pix.units)

    dtype = pix.dtype
    if single_precision:
        if dtype == np.float64:
            dtype = np.float32
        elif dtype == np.int64:
            dtype = np.int32

    if have_hdf5_parallel() and not force_serial:
        # Open the file for parallel access.
        with h5py.File(path, "w", driver="mpio", comm=dist.comm) as f:
            # Each process writes their own submaps to the file
            dset = f.create_dataset(
                "map",
                (pix.n_value, dist.n_pix),
                chunks=(pix.n_value, dist.n_pix_submap),
                dtype=dtype,
            )
            for key, value in header.items():
                dset.attrs[key] = value
            for submap in range(dist.n_submap):
                if allowners[submap] == rank:
                    local_submap = dist.global_submap_to_local[submap]
                    # Accommodate submap sizes that do not fit the map cleanly
                    first = submap * dist.n_pix_submap
                    last = min(first + dist.n_pix_submap, dist.n_pix)
                    dset[:, first:last] = pix[local_submap, 0 : last - first].T
    else:
        # No luck, write serially from root process
        if use_mpi:
            # MPI is enabled, but we are not using it.  Warn the user.
            log.warning_rank(
                f"h5py not built with MPI support.  Writing {path} in serial mode.",
                comm=dist.comm,
            )

        # n_send = len(dist.owned_submaps)
        n_send = np.sum(allowners == rank)
        if n_send == 0:
            sendbuffer = None
        else:
            sendbuffer = np.empty(
                [n_send, pix.n_value, dist.n_pix_submap],
                dtype=dtype,
            )
            offset = 0
            # for submap in dist.owned_submaps:
            for submap in range(dist.n_submap):
                if allowners[submap] == rank:
                    local_submap = dist.global_submap_to_local[submap]
                    sendbuffer[offset] = pix.data[local_submap].T
                    offset += 1

        if rank == 0:
            # Root process receives the submaps from other processes and writes
            # them to file
            with h5py.File(path, "w") as f:
                dset = f.create_dataset(
                    "map",
                    (pix.n_value, dist.n_pix),
                    chunks=(pix.n_value, dist.n_pix_submap),
                    dtype=dtype,
                )
                for rank_send in range(ntask):
                    # submaps = np.argwhere(dist.submap_owners == rank_send).ravel()
                    submaps = np.arange(dist.n_submap)[allowners == rank_send]
                    n_receive = len(submaps)
                    if n_receive > 0:
                        if rank_send == rank:
                            recvbuffer = sendbuffer
                        else:
                            recvbuffer = np.empty(
                                [n_receive, pix.n_value, dist.n_pix_submap],
                                dtype=dtype,
                            )
                            dist.comm.Recv(recvbuffer, source=rank_send, tag=rank_send)
                    for i, submap in enumerate(submaps):
                        # Accommodate submap sizes that do not fit the map cleanly
                        first = submap * dist.n_pix_submap
                        last = min(first + dist.n_pix_submap, dist.n_pix)
                        dset[:, first:last] = recvbuffer[i, :, 0 : last - first]

                for key, value in header.items():
                    dset.attrs[key] = value
        else:
            # All others wait for their turn to send
            if sendbuffer is not None:
                dist.comm.Send(sendbuffer, dest=0, tag=rank)

    return


def filename_is_fits(filename):
    return filename.endswith((".fits", ".fit", ".FITS"))


def filename_is_hdf5(filename):
    return filename.endswith((".hdf5", ".h5", ".H5"))


@function_timer
def read_healpix(filename, *args, **kwargs):
    """Read a FITS or HDF5 map serially.

    This reads the file into simple numpy arrays on the calling process.
    Units in the file are ignored.

    Args:
        filename (str):  The path to the file.

    Returns:
        (tuple):  The map data and optionally header.

    """

    filename = filename.strip()

    if filename_is_fits(filename):
        # Load a FITS map with healpy
        result = hp.read_map(filename, *args, **kwargs)

    elif filename_is_hdf5(filename):
        if "verbose" in kwargs and kwargs["verbose"] == False:
            verbose = False
        else:
            # healpy default
            verbose = True

        # Load an HDF5 map
        try:
            f = h5py.File(filename, "r")
        except OSError as e:
            msg = f"Failed to open {filename} for reading: {e}"
            raise RuntimeError(msg)

        dset = f["map"]
        if "field" in kwargs and kwargs["field"] is not None:
            mapdata = []
            for field in kwargs["field"]:
                mapdata.append(dset[field])
            mapdata = np.vstack(mapdata)
        else:
            mapdata = dset[:]

        header = dict(dset.attrs)
        if "ORDERING" not in header or header["ORDERING"] not in ["NESTED", "RING"]:
            raise RuntimeError("Cannot determine pixel ordering")
        if verbose:
            print("")
        if "nest" in kwargs:
            nest = kwargs["nest"]
        else:
            nest = False
        if header["ORDERING"] == "NESTED" and nest == False:
            if verbose:
                print(f"Reordering {filename} to RING")
            mapdata = hp.reorder(mapdata, n2r=True)
        elif header["ORDERING"] == "RING" and nest == True:
            if verbose:
                print(f"Reordering {filename} to NESTED")
            mapdata = hp.reorder(mapdata, r2n=True)
        else:
            if verbose:
                print(f"{filename} is already {header['ORDERING']}")
        f.close()

        if "dtype" in kwargs and kwargs["dtype"] is not None:
            mapdata = mapdata.astype(kwargs["dtype"])

        if "h" in kwargs and kwargs["h"] == True:
            result = mapdata, header
        else:
            result = mapdata
    else:
        msg = f"Could not ascertain file type for '{filename}'"
        raise RuntimeError(msg)

    return result


@function_timer
def write_healpix(filename, mapdata, nside_submap=16, *args, **kwargs):
    """Write a FITS or HDF5 map serially.

    This writes the map data from a simple numpy array on the calling process.
    No units are written to the file.

    Args:
        filename (str):  The path to the file.
        mapdata (array):  The data array.
        nside_submap (int):  The submap NSIDE, used for dataset chunking.

    Returns:
        None

    """

    if filename_is_fits(filename):
        # Write a FITS map with healpy
        return hp.write_map(filename, mapdata, *args, **kwargs)

    elif filename_is_hdf5(filename):
        # Write an HDF5 map
        mapdata = np.atleast_2d(mapdata)
        n_value, n_pix = mapdata.shape
        nside = hp.npix2nside(n_pix)

        nside_submap = min(nside_submap, nside)
        n_pix_submap = 12 * nside_submap**2

        mode = "w-"
        if "overwrite" in kwargs and kwargs["overwrite"] == True:
            mode = "w"

        dtype = mapdata.dtype
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            dtype = kwargs["dtype"]

        with h5py.File(filename, mode) as f:
            dset = f.create_dataset(
                "map",
                (n_value, n_pix),
                chunks=(n_value, n_pix_submap),
                dtype=dtype,
            )
            dset[:] = mapdata

            if "extra_header" in kwargs:
                header = kwargs["extra_header"]
                for key, value in header:
                    dset.attrs[key] = value
            if "nest" in kwargs and kwargs["nest"] == True:
                dset.attrs["ORDERING"] = "NESTED"
            else:
                dset.attrs["ORDERING"] = "RING"
            dset.attrs["NSIDE"] = nside

    return
