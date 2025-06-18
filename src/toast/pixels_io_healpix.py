# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import healpy as hp
import numpy as np
from astropy import units as u

from .io import have_hdf5_parallel
from .mpi import MPI, use_mpi
from .pixels_io_utils import filename_is_fits, filename_is_hdf5
from .timing import Timer, function_timer
from .utils import Logger, memreport, unit_conversion


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
                            fview[col][global_offset : global_offset + n_copy] = (
                                recvview[c, 0:n_copy, col]
                            )
                sendbuf.fill(0)
                if rank == 0:
                    recvbuf.fill(0)
            submap_off += ncomm

    return fdata, fview


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
        # Translate positional arguments to keyword arguments
        if len(args) == 0:
            if "field" not in kwargs:
                kwargs["field"] = (0,)
        else:
            if "field" in kwargs:
                raise ValueError("'field' defined twice")
            field = args[0]
            if field is not None and not hasattr(field, "__len__"):
                field = (field,)
            kwargs["field"] = field

        if len(args) > 1:
            if "dtype" in kwargs:
                raise ValueError("'dtype' defined twice")
            kwargs["dtype"] = args[1]

        if len(args) > 2:
            if "nest" in kwargs:
                raise ValueError("'nest' defined twice")
            kwargs["nest"] = args[2]
        if "nest" in kwargs:
            nest = kwargs["nest"]
        else:
            nest = False

        if len(args) > 3:
            if "partial" in kwargs:
                raise ValueError("'partial' defined twice")
            kwargs["partial"] = args[3]
        if "partial" in kwargs and kwargs["partial"]:
            raise ValueError("HDF5 maps are never explicitly indexed")

        if len(args) > 4:
            if "hdu" in kwargs:
                raise ValueError("'hdu' defined twice")
            kwargs["hdu"] = args[4]
        if "hdu" in kwargs and kwargs["hdu"] != 1:
            raise ValueError("HDF5 maps do not have HDUs")

        if len(args) > 5:
            if "h" in kwargs:
                raise ValueError("'h' defined twice")
            kwargs["h"] = args[5]

        if len(args) > 6:
            if "verbose" in kwargs:
                raise ValueError("'verbose' defined twice")
            kwargs["verbose"] = args[6]
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        else:
            # healpy default
            verbose = True

        if len(args) > 7:
            if "memmap" in kwargs:
                raise ValueError("'memmap' defined twice")
            kwargs["memmap"] = args[7]
        if "memmap" in kwargs and kwargs["memmap"]:
            raise ValueError("HDF5 maps do not have explicit memmap")

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
        if header["ORDERING"] == "NESTED" and nest == False:
            if verbose:
                print(f"\nReordering {filename} to RING")
            mapdata = hp.reorder(mapdata, n2r=True)
        elif header["ORDERING"] == "RING" and nest == True:
            if verbose:
                print(f"\nReordering {filename} to NESTED")
            mapdata = hp.reorder(mapdata, r2n=True)
        else:
            if verbose:
                print(f"\n{filename} is already {header['ORDERING']}")
        f.close()

        if "dtype" in kwargs and kwargs["dtype"] is not None:
            mapdata = mapdata.astype(kwargs["dtype"])

        if mapdata.shape[0] == 1:
            mapdata = mapdata[0]

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
        if len(args) != 0:
            raise ValueError("No positional arguments supported")

        # Write an HDF5 map
        mapdata = np.atleast_2d(mapdata)
        n_value, n_pix = mapdata.shape
        nside = hp.npix2nside(n_pix)

        nside_submap = min(nside_submap, nside)
        n_pix_submap = 12 * nside_submap**2

        mode = "w-"
        if "overwrite" in kwargs and kwargs["overwrite"] == True:
            mode = "w"
        elif os.path.isfile(filename):
            raise FileExistsError(f"'{filename}' exists and `overwrite` is False")

        dtype = mapdata.dtype
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            dtype = kwargs["dtype"]

        if "fits_IDL" in kwargs and kwargs["fits_IDL"]:
            raise ValueError("HDF5 does not support fits_IDL")

        if "partial" in kwargs and kwargs["partial"]:
            raise ValueError("HDF5 does not support partial; map is always chunked.")

        if "column_names" in kwargs and kwargs["column_names"] is not None:
            raise ValueError("HDF5 does not support column_names")

        ordering = "RING"
        if "nest" in kwargs and kwargs["nest"] == True:
            ordering = "NESTED"

        coord = None
        if "coord" in kwargs:
            coord = kwargs["coord"]

        units = None
        if "column_units" in kwargs:
            units = kwargs["column_units"]
            # Only one units attribute is supported
            if not isinstance(units, str):
                msg = f"ERROR: HDF5 map units must be a single string, "
                msg += f"not {units}"
                raise RuntimeError(msg)

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

            dset.attrs["ORDERING"] = ordering
            dset.attrs["NSIDE"] = nside
            if units is not None:
                dset.attrs["UNITS"] = units
            if coord is not None:
                dset.attrs["COORDSYS"] = coord

    return
