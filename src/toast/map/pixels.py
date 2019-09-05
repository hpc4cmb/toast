# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from ..timing import function_timer, Timer

from ..mpi import MPI

import healpy as hp

from ..op import Operator

from ..cache import Cache


class OpLocalPixels(Operator):
    """Operator which computes the set of locally hit pixels.

    Args:
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.

    """

    def __init__(self, pixels="pixels", pixmin=None, pixmax=None, no_hitmap=False, verbose=False):

        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._pixels = pixels
        self._pixmin = pixmin
        self._pixmax = pixmax
        self._no_hitmap = no_hitmap
        self._verbose = verbose

    @function_timer
    def exec(self, data):
        """Iterate over all observations and detectors and compute local pixels.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            (array): An array of the locally hit pixel indices.

        """
        # initialize the local pixel set
        local = None
        timer = Timer()
        timer.start()

        if self._no_hitmap:
            # Avoid allocating extra memory at the cost of slower operation
            for obs in data.obs:
                tod = obs["tod"]
                for det in tod.local_dets:
                    pixelsname = "{}_{}".format(self._pixels, det)
                    pixels = tod.cache.reference(pixelsname)
                    if local is None:
                        local = np.unique(pixels)
                    else:
                        local = np.unique(np.concatenate((local, np.unique(pixels))))
                    del pixels
            if self._verbose:
                timer.report_clear("Identify unique pixels")
        else:
            pixmin = self._pixmin
            pixmax = self._pixmax

            if self._pixmin is None or self._pixmax is None:
                # Find the overall pixel range before allocating the hit map
                pixmin = 2 ** 60
                pixmax = -2 ** 60
                for obs in data.obs:
                    tod = obs["tod"]
                    for det in tod.local_dets:
                        pixelsname = "{}_{}".format(self._pixels, det)
                        pixels = tod.cache.reference(pixelsname)
                        pixmin = min(pixmin, np.amin(pixels))
                        pixmax = max(pixmax, np.amax(pixels))
                        del pixels
                if self._verbose:
                    timer.report_clear("Identify pixel range")

                if pixmin == 2 ** 60 and pixmax == -2 ** 60:
                    # No pixels
                    return np.array([], dtype=np.int64)

            npix = pixmax - pixmin + 1
            hitmap = np.zeros(npix, dtype=np.bool)

            for obs in data.obs:
                tod = obs["tod"]
                for det in tod.local_dets:
                    pixelsname = "{}_{}".format(self._pixels, det)
                    pixels = tod.cache.reference(pixelsname)
                    hitmap[pixels - pixmin] = True
                    del pixels

            if self._verbose:
                timer.report_clear("Build hit map")

            local = np.arange(pixmin, pixmax + 1, dtype=np.int)[hitmap]

            if self._verbose:
                timer.report_clear("hit map to list")

        return local


class DistPixels(object):
    """A distributed map with multiple values per pixel.

    Pixel domain data is distributed across an MPI communicator.  each
    process has a local data stored in one or more "submaps".  The size
    of the submap can be tuned to balance storage (smaller submap size
    means fewer wasted pixels stored) and ease of indexing (larger
    submap means faster global-to-local pixel lookups).

    Although multiple processes may have the same submap of data stored
    locally, the lowest-rank process that has a given submap is the
    "owner" for operations like serialization.

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator containing all
            processes (or None).
        size (int): the total number of pixels.
        nnz (int): the number of values per pixel.
        submap (int): the locally stored data is in units of this size.
        local (array): the list of local submaps (integers).
        localpix (array): the list of local pixels (integers).
        nest (bool): nested pixel order flag

    """

    def __init__(
        self,
        comm=None,
        size=0,
        nnz=1,
        dtype=np.float64,
        submap=None,
        local=None,
        localpix=None,
        nest=True,
    ):
        self._comm = comm
        self._size = size
        self._nnz = nnz
        self._dtype = dtype
        self._submap = submap
        self._nest = nest

        if self._size % self._submap != 0:
            raise RuntimeError(
                "submap size must evenly divide into total " "number of pixels"
            )

        if localpix is not None:
            if local is not None:
                raise RuntimeError("Must not set local with localpix")
            local = np.unique(np.floor_divide(localpix, self._submap))

        self._local = local
        self._nglob = self._size // self._submap
        self._glob2loc = None
        self._cache = Cache()
        self._commsize = 5000000

        # our data is a 3D array of submap, pixel, values
        # we allocate this as a contiguous block

        self.data = None
        self.flatdata = None
        if self._local is None:
            self._nsub = 0
        else:
            if len(self._local) == 0:
                self._nsub = 0
            else:
                self._nsub = len(self._local)
                self._glob2loc = self._cache.create(
                    "glob2loc", np.int64, (self._nglob,)
                )
                self._glob2loc[:] = -1
                for g in enumerate(self._local):
                    self._glob2loc[g[1]] = g[0]
                if (self._submap * self._local.max()) > self._size:
                    raise RuntimeError("local submap indices out of range")
                self.data = self._cache.create(
                    "data", dtype, (self._nsub, self._submap, self._nnz)
                )
                self.flatdata = self.data.view()
                self.flatdata.shape = tuple([self._nsub * self._submap * self._nnz])

    def __del__(self):
        if self._glob2loc is not None:
            del self._glob2loc
        if self.data is not None:
            del self.data
        self._cache.clear()

    @property
    def comm(self):
        """(mpi4py.MPI.Comm): The MPI communicator used (or None)
        """
        return self._comm

    @property
    def size(self):
        """(int): The global number of pixels.
        """
        return self._size

    @property
    def nnz(self):
        """(int): The number of non-zero values per pixel.
        """
        return self._nnz

    @property
    def dtype(self):
        """(numpy.dtype): The data type of the values.
        """
        return self._dtype

    @property
    def local(self):
        """(array): The list of local submaps or None if process has no data.
        """
        return self._local

    @property
    def submap(self):
        """(int): The number of pixels in each submap.
        """
        return self._submap

    @property
    def nsubmap(self):
        """(int): The number of submaps stored on this process.
        """
        return self._nsub

    @property
    def nested(self):
        """(bool): If True, data is HEALPix NESTED ordering.
        """
        return self._nest

    @function_timer
    def global_to_local(self, gl):
        """Convert global pixel indices into the local submap and pixel.

        Args:
            gl (int): The global pixel number.

        Returns:
            (tuple):  A tuple containing the local submap index (int) and the
                pixel index local to that submap (int).

        """
        safe_gl = np.zeros(len(gl), dtype=np.int64)
        good = gl >= 0
        bad = gl < 0
        safe_gl[good] = gl[good]
        sm = np.floor_divide(safe_gl, self._submap)
        pix = np.mod(safe_gl, self._submap)
        pix[bad] = -1
        lsm = self._glob2loc[sm]
        return (lsm, pix)

    @function_timer
    def duplicate(self):
        """Perform a deep copy of the distributed data.

        Returns:
            (DistPixels): A copy of the object.

        """
        ret = DistPixels(
            comm=self._comm,
            size=self._size,
            nnz=self._nnz,
            dtype=self._dtype,
            submap=self._submap,
            local=self._local,
        )
        if self.data is not None:
            ret.data[:, :, :] = self.data
        return ret

    def _comm_nsubmap(self, bytes):
        """Given a buffer size, compute the number of submaps to communicate.

        Args:
            bytes (int):  The number of bytes.

        Returns:
            (int):  The number of submaps in each buffer.

        """
        dbytes = self._dtype(1).itemsize
        nsub = int(bytes / (dbytes * self._submap * self._nnz))
        if nsub == 0:
            nsub = 1
        allsub = int(self._size / self._submap)
        if nsub > allsub:
            nsub = allsub
        return nsub

    @function_timer
    def allreduce(self, comm_bytes=None):
        """Perform a buffered allreduce of the pixel domain data.

        Args:
            comm_bytes (int): The approximate message size to use.

        Returns:
            None.

        """
        if self._comm is None:
            return

        if comm_bytes is None:
            comm_bytes = self._commsize
        comm_submap = self._comm_nsubmap(comm_bytes)
        nsub = int(self._size / self._submap)

        sendbuf = np.zeros(comm_submap * self._submap * self._nnz, dtype=self._dtype)
        sendview = sendbuf.reshape(comm_submap, self._submap, self._nnz)

        recvbuf = np.zeros(comm_submap * self._submap * self._nnz, dtype=self._dtype)
        recvview = recvbuf.reshape(comm_submap, self._submap, self._nnz)

        owners = np.zeros(nsub, dtype=np.int32)
        owners.fill(self._comm.size)
        for m in self._local:
            owners[m] = self._comm.rank
        allowners = np.zeros_like(owners)
        self._comm.Allreduce(owners, allowners, op=MPI.MIN)

        submap_off = 0
        ncomm = comm_submap

        while submap_off < nsub:
            if submap_off + ncomm > nsub:
                ncomm = nsub - submap_off
            if (
                np.sum(allowners[submap_off : submap_off + ncomm])
                != ncomm * self._comm.size
            ):
                # At least one submap has some hits.  Do the allreduce.
                # Otherwise we would skip this buffer to avoid reducing a
                # bunch of zeros.
                for c in range(ncomm):
                    glob = submap_off + c
                    if glob in self._local:
                        # copy our data in.
                        loc = self._glob2loc[glob]
                        sendview[c, :, :] = self.data[loc, :, :]

                self._comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)

                for c in range(ncomm):
                    glob = submap_off + c
                    if glob in self._local:
                        # copy the reduced data
                        loc = self._glob2loc[glob]
                        self.data[loc, :, :] = recvview[c, :, :]

                sendbuf.fill(0)
                recvbuf.fill(0)

            submap_off += ncomm

        return

    @function_timer
    def read_healpix_fits(self, path, comm_bytes=None):
        """Read and broadcast a HEALPix FITS table.

        The root process opens the FITS file in memmap mode and iterates over
        chunks of the map in a way to minimize cache misses in the internal
        FITS buffer.  Chunks of submaps are broadcast to all processes, and
        each process copies data to its local submaps.

        Args:
            path (str): The path to the FITS file.
            comm_bytes (int): The approximate message size to use.

        Returns:
            None

        """
        rank = 0
        if self._comm is not None:
            rank = self._comm.rank

        if comm_bytes is None:
            comm_bytes = self._commsize
        comm_submap = self._comm_nsubmap(comm_bytes)

        # we make the assumption that FITS binary tables are still stored in
        # blocks of 2880 bytes just like always...
        dbytes = self._dtype(1).itemsize
        rowbytes = self._nnz * dbytes
        optrows = int(2880 / rowbytes)

        # get a tuple of all columns in the table.  We choose memmap here so
        # that we can (hopefully) read through all columns in chunks such that
        # we only ever have a couple FITS blocks in memory.
        fdata = None
        if rank == 0:
            # Check that the file is in expected format
            errors = ""
            h = hp.fitsfunc.pf.open(path, "readonly")
            nside = hp.npix2nside(self._size)
            nside_map = h[1].header["nside"]
            if nside_map != nside:
                errors += "Wrong NSide: {} has {}, expected {}\n" "".format(
                    path, nside_map, nside
                )
            map_nested = False
            if "order" in h[1].header and "NEST" in h[1].header["order"].upper():
                map_nested = True
            if "ordering" in h[1].header and "NEST" in h[1].header["ordering"].upper():
                map_nested = True
            if map_nested != self._nest:
                errors += (
                    "Wrong ordering: {} has nest={}, expected nest={}\n"
                    "".format(path, map_nested, self._nest)
                )
            map_nnz = h[1].header["tfields"]
            if map_nnz != self._nnz:
                errors += "Wrong number of columns: {} has {}, expected {}\n" "".format(
                    path, map_nnz, self._nnz
                )
            h.close()
            if len(errors) != 0:
                raise RuntimeError(errors)
            # Now read the map
            fdata = hp.read_map(
                path,
                field=None,
                dtype=self._dtype,
                memmap=True,
                nest=self._nest,
                verbose=False,
            )
            if self._nnz == 1:
                fdata = (fdata,)

        buf = np.zeros(comm_submap * self._submap * self._nnz, dtype=self._dtype)
        view = buf.reshape(comm_submap, self._submap, self._nnz)

        in_off = 0
        out_off = 0
        submap_off = 0

        rows = optrows
        while in_off < self._size:
            if in_off + rows > self._size:
                rows = self._size - in_off
            # is this the last block for this communication?
            islast = False
            copyrows = rows
            if out_off + rows > (comm_submap * self._submap):
                copyrows = (comm_submap * self._submap) - out_off
                islast = True

            if rank == 0:
                for col in range(self._nnz):
                    coloff = (out_off * self._nnz) + col
                    buf[coloff : coloff + (copyrows * self._nnz) : self._nnz] = fdata[
                        col
                    ][in_off : in_off + copyrows]

            out_off += copyrows
            in_off += copyrows

            if islast:
                if self._comm is not None:
                    self._comm.Bcast(buf, root=0)
                # loop over these submaps, and copy any that we are assigned
                for sm in range(submap_off, submap_off + comm_submap):
                    if sm in self._local:
                        loc = self._glob2loc[sm]
                        self.data[loc, :, :] = view[sm - submap_off, :, :]
                out_off = 0
                submap_off += comm_submap
                buf.fill(0)
                islast = False

        # flush the remaining buffer
        if out_off > 0:
            if self._comm is not None:
                self._comm.Bcast(buf, root=0)
            # loop over these submaps, and copy any that we are assigned
            for sm in range(submap_off, submap_off + comm_submap):
                if sm in self._local:
                    loc = self._glob2loc[sm]
                    self.data[loc, :, :] = view[sm - submap_off, :, :]
        return

    @function_timer
    def broadcast_healpix_map(self, fdata, comm_bytes=None):
        """Distribute a map located on a single process.

        The root process takes a map in memory and distributes it.   Chunks of submaps
        are broadcast to all processes, and each process copies data to its local
        submaps.

        Args:
            fdata (array): The input data (only significant on process 0).
            comm_bytes (int): The approximate message size to use.

        Returns:
            None

        """
        rank = 0
        if self._comm is not None:
            rank = self._comm.rank
        if comm_bytes is None:
            comm_bytes = self._commsize
        comm_submap = self._comm_nsubmap(comm_bytes)

        # we make the assumption that FITS binary tables are still stored in
        # blocks of 2880 bytes just like always...
        dbytes = self._dtype(1).itemsize
        rowbytes = self._nnz * dbytes
        optrows = int(2880 / rowbytes)

        # get a tuple of all columns in the table.  We choose memmap here so
        # that we can (hopefully) read through all columns in chunks such that
        # we only ever have a couple FITS blocks in memory.
        if rank == 0:
            if self._nnz == 1:
                fdata = (fdata,)

        buf = np.zeros(comm_submap * self._submap * self._nnz, dtype=self._dtype)
        view = buf.reshape(comm_submap, self._submap, self._nnz)

        in_off = 0
        out_off = 0
        submap_off = 0

        rows = optrows
        while in_off < self._size:
            if in_off + rows > self._size:
                rows = self._size - in_off
            # is this the last block for this communication?
            islast = False
            copyrows = rows
            if out_off + rows > (comm_submap * self._submap):
                copyrows = (comm_submap * self._submap) - out_off
                islast = True

            if rank == 0:
                for col in range(self._nnz):
                    coloff = (out_off * self._nnz) + col
                    buf[coloff : coloff + (copyrows * self._nnz) : self._nnz] = fdata[
                        col
                    ][in_off : in_off + copyrows]

            out_off += copyrows
            in_off += copyrows

            if islast:
                if self._comm is not None:
                    self._comm.Bcast(buf, root=0)
                # loop over these submaps, and copy any that we are assigned
                for sm in range(submap_off, submap_off + comm_submap):
                    if sm in self._local:
                        loc = self._glob2loc[sm]
                        self.data[loc, :, :] = view[sm - submap_off, :, :]
                out_off = 0
                submap_off += comm_submap
                buf.fill(0)
                islast = False

        # flush the remaining buffer

        if out_off > 0:
            if self._comm is not None:
                self._comm.Bcast(buf, root=0)
            # loop over these submaps, and copy any that we are assigned
            for sm in range(submap_off, submap_off + comm_submap):
                if sm in self._local:
                    loc = self._glob2loc[sm]
                    self.data[loc, :, :] = view[sm - submap_off, :, :]
        return

    @function_timer
    def write_healpix_fits(self, path, comm_bytes=None):
        """Write data to a HEALPix format FITS table.

        The data across all processes is assumed to be synchronized (the
        data for a given submap shared between processes is identical).  The
        lowest rank process sharing each submap sends their copy to the root
        process for writing.

        Args:
            path (str): The path to the FITS file.
            comm_bytes (int): The approximate message size to use.

        """
        rank = 0
        if self._comm is not None:
            rank = self._comm.rank

        if comm_bytes is None:
            comm_bytes = self._commsize

        # We will reduce some number of whole submaps at a time.
        # Find the number of submaps that fit into the requested
        # communication size.
        dbytes = self._dtype(1).itemsize
        comm_submap = int(comm_bytes / (dbytes * self._submap * self._nnz))
        if comm_submap == 0:
            comm_submap = 1

        nsubmap = int(self._size / self._submap)
        if nsubmap * self._submap < self._size:
            nsubmap += 1

        # Determine which processes "own" each submap.

        owners = np.zeros(nsubmap, dtype=np.int32)
        allowners = None
        if self._comm is None:
            allowners = owners
        else:
            owners.fill(self._comm.size)
            for m in self._local:
                owners[m] = self._comm.rank
            allowners = np.zeros_like(owners)
            self._comm.Allreduce(owners, allowners, op=MPI.MIN)

        # this function requires lots of RAM, since it accumulates the
        # full map on one process before writing.

        # use a cache to store the local map, so that we can be sure to
        # free the memory afterwards

        fdata = None
        temp = None
        if rank == 0:
            fdata = []
            temp = Cache()
            for col in range(self._nnz):
                name = "col{}".format(col)
                temp.create(name, self._dtype, (self._size,))
                fdata.append(temp.reference(name))

        if self._comm is None:
            dbuf = np.zeros(comm_submap * self._submap * self._nnz, dtype=self._dtype)
            dview = dbuf.reshape(comm_submap, self._submap, self._nnz)

            submap_off = 0
            ncomm = comm_submap
            while submap_off < nsubmap:
                if submap_off + ncomm > nsubmap:
                    ncomm = nsubmap - submap_off
                if np.sum(allowners[submap_off : submap_off + ncomm]) != ncomm:
                    # at least one submap has some hits
                    for c in range(ncomm):
                        dview[c, :, :] = self.data[self._glob2loc[submap_off + c], :, :]
                    # copy into FITS buffers
                    for c in range(ncomm):
                        sampoff = (submap_off + c) * self._submap
                        for col in range(self._nnz):
                            fdata[col][sampoff : sampoff + self._submap] = dview[
                                c, :, col
                            ]
                submap_off += ncomm
        else:
            sendbuf = np.zeros(
                comm_submap * self._submap * self._nnz, dtype=self._dtype
            )
            sendview = sendbuf.reshape(comm_submap, self._submap, self._nnz)

            recvbuf = None
            recvview = None
            if rank == 0:
                recvbuf = np.zeros(
                    comm_submap * self._submap * self._nnz, dtype=self._dtype
                )
                recvview = recvbuf.reshape(comm_submap, self._submap, self._nnz)

            submap_off = 0
            ncomm = comm_submap
            while submap_off < nsubmap:
                if submap_off + ncomm > nsubmap:
                    ncomm = nsubmap - submap_off
                if (
                    np.sum(allowners[submap_off : submap_off + ncomm])
                    != ncomm * self._comm.size
                ):
                    # at least one submap has some hits.  reduce.
                    for c in range(ncomm):
                        if allowners[submap_off + c] == self._comm.rank:
                            sendview[c, :, :] = self.data[
                                self._glob2loc[submap_off + c], :, :
                            ]
                    self._comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)
                    if rank == 0:
                        # copy into FITS buffers
                        for c in range(ncomm):
                            sampoff = (submap_off + c) * self._submap
                            for col in range(self._nnz):
                                fdata[col][sampoff : sampoff + self._submap] = recvview[
                                    c, :, col
                                ]
                    sendbuf.fill(0)
                    if rank == 0:
                        recvbuf.fill(0)
                submap_off += ncomm

        if rank == 0:
            if os.path.isfile(path):
                os.remove(path)
            hp.write_map(
                path, fdata, dtype=self._dtype, fits_IDL=False, nest=self._nest
            )
            del fdata
            del temp

        return
