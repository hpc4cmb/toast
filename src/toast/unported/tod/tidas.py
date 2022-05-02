# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re

import numpy as np

from .. import qarray as qa
from ..data import Data
from ..dist import distribute_discrete
from ..mpi import MPI
from ..operator import Operator
from ..timing import Timer, function_timer
from ..utils import Logger
from .interval import Interval, intervals_to_chunklist
from .tod import TOD

available = True
try:
    import tidas as tds
    from tidas.mpi import MPIVolume

    from . import tidas_utils as tdsutils
except ImportError:
    available = False


# Module-level constants

STR_QUAT = "QUAT"
STR_FLAG = "FLAG"
STR_COMMON = "COMMON"
STR_BORE = "BORE"
STR_BOREAZEL = "BOREAZEL"
STR_POS = "POS"
STR_VEL = "VEL"
STR_DISTINTR = "chunks"
STR_DETGROUP = "detectors"
STR_FPGROUP = "focalplane"
STR_CACHEGROUP = "cache"
STR_OBSGROUP = "observation"


class TODTidas(TOD):
    """This class provides an interface to a single TIDAS data block.

    An instance of this class reads and writes to a single TIDAS block which
    represents a TOAST observation.  The volume and specific block should
    already exist.  Groups and intervals within the observation may already
    exist or may be created with the provided helper methods.  All groups
    and intervals must exist prior to reading or writing from them.

    Detector pointing offsets from the boresight are given as quaternions,
    and are expected to be contained in the dictionary of properties
    found in the TIDAS group containing detector timestreams.

    This class is generic- it can only read / write standard data streams
    and objects in the TOD.cache.  More specialized uses for specific
    experiments should be implemented in their own classes.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which this
            observation data is distributed.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        vol (tidas.MPIVolume):  The volume.
        path (str):  The path to this observation block.
        group_dets (str):  The name of the TIDAS group containing detector
            and other telescope information at the detector sample rate.
        group_fp (str):  The name of the TIDAS group containing detector
            focalplane offsets.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        distintervals (str OR list):  Optional name of the TIDAS intervals that
            determines how the data should be distributed along the time
            axis.  If a list is given use that explicit set of sample chunks.

    """

    # FIXME: currently the data flags are stored in the same group as
    # the data.  Once TIDAS supports per-group backend options like
    # compression, we should move the flags to a separate group:
    #   https://github.com/hpc4cmb/tidas/issues/13

    @function_timer
    def __init__(
        self,
        mpicomm,
        detranks,
        vol,
        path,
        group_dets=STR_DETGROUP,
        group_fp=STR_FPGROUP,
        detbreaks=None,
        distintervals=None,
    ):
        if not available:
            raise RuntimeError("tidas is not available")
        rank = 0
        if mpicomm is not None:
            rank = mpicomm.rank

        tmr = Timer()
        tmr.start()

        # We keep a handle to the volume
        self._vol = vol

        # Get a handle to the observation node
        pfields = path.split("/")
        parent = "/".join(pfields[0:-1])
        self._blockparent = parent
        self._blockname = pfields[-1]
        self._blockpath = "/".join([parent, pfields[-1]])
        self._block = None

        # Temporary handle to the block
        tmpblock = tdsutils.find_obs(self._vol, self._blockparent, self._blockname)

        # Get the existing group and interval names
        grpnames = tmpblock.group_names()

        if mpicomm is not None:
            mpicomm.barrier()
        # if rank == 0:
        #     tmr.report_clear("TODTidas open block {}".format(self._blockname))

        # Read detectors and focalplane offsets

        self._fpgrpname = group_fp
        if self._fpgrpname not in grpnames:
            raise RuntimeError(
                "focalplane group {} does not exist".format(self._fpgrpname)
            )

        fpgrp = tmpblock.group_get(self._fpgrpname)
        if fpgrp.size() != 4:
            raise RuntimeError(
                "focalplane group {} does not have quaternions".format(self._fpgrpname)
            )

        self._detquats = dict()
        schm = fpgrp.schema()
        dets = [x.name for x in schm.fields() if (x.name != "TIDAS_TIME")]
        for d in dets:
            q = fpgrp.read(d, 0, 4)
            self._detquats[d] = np.array(q)
        del schm
        del fpgrp

        self._detlist = sorted(list(self._detquats.keys()))
        if mpicomm is not None:
            mpicomm.barrier()
        # if rank == 0:
        #     tmr.report_clear("TODTidas read fp group {}".format(self._blockname))

        # Read detector data group properties and size.

        self._dgrpname = group_dets
        if self._dgrpname not in grpnames:
            raise RuntimeError(
                "detector group {} does not exist".format(self._dgrpname)
            )
        self._dgrp = None

        # Temporary handle to the group
        tmpdgrp = tmpblock.group_get(self._dgrpname)

        self._samples = tmpdgrp.size()
        meta = tdsutils.to_dict(tmpdgrp.dictionary())
        if mpicomm is not None:
            mpicomm.barrier()
        # if rank == 0:
        #     tmr.report_clear(
        #         "TODTidas read det data group {} meta".format(self._blockname)
        #     )

        # See whether we have ground based data

        self._have_azel = False
        schm = tmpdgrp.schema()
        fields = [x.name for x in schm.fields()]
        if "{}_{}X".format(STR_BOREAZEL, STR_QUAT) in fields:
            self._have_azel = True
        del fields
        del schm
        if mpicomm is not None:
            mpicomm.barrier()
        # if rank == 0:
        #     tmr.report_clear("TODTidas check {} azel".format(self._blockname))

        # We need to assign a unique integer index to each detector.  This
        # is used when seeding the streamed RNG in order to simulate
        # timestreams.  For simplicity, and assuming that detector names
        # are not too long, we can convert the detector name to bytes and
        # then to an integer.

        self._detindx = {}
        for det in self._detlist:
            bdet = det.encode("utf-8")
            uid = None
            try:
                ind = int.from_bytes(bdet, byteorder="little")
                uid = int(ind & 0xFFFFFFFF)
            except:
                raise RuntimeError(
                    "Cannot convert detector name {} to a "
                    "unique integer- maybe it is too long?".format(det)
                )
            self._detindx[det] = uid

        if mpicomm is not None:
            mpicomm.barrier()
        # if rank == 0:
        #     tmr.report_clear("TODTidas make {} detindx".format(self._blockname))

        # Create an MPI lock to use for writing to the TIDAS volume.  We must
        # have only one writing process at a time.  Note that this lock is over
        # the communicator for this single observation (TIDAS block).
        # Processes writing to separate blocks have no restrictions.

        # self._writelock = MPILock(mpicomm)

        # Read intervals and set up distribution chunks.

        sampsizes = None

        if distintervals is not None:
            if isinstance(distintervals, str):
                # This is the name of a TIDAS intervals object.  Check that
                # it exists.
                inames = tmpblock.intervals_names()
                if distintervals not in inames:
                    raise RuntimeError(
                        "distintervals {} does not exist".format(distintervals)
                    )
                distint = tmpblock.intervals_get(distintervals)
                # Rank zero process reads and broadcasts intervals
                intervals = None
                if rank == 0:
                    intervals = distint.read()
                if mpicomm is not None:
                    intervals = mpicomm.bcast(intervals, root=0)
                # Compute the contiguous spans of time for data distribution
                # based on the starting points of all intervals.
                sampsizes = intervals_to_chunklist(intervals, self._samples)
                del intervals
                del distint
            else:
                # This must be an explicit list
                sampsizes = list(distintervals)

        if mpicomm is not None:
            mpicomm.barrier()
        # if rank == 0:
        #     tmr.report_clear("TODTidas read {} chunks".format(self._blockname))

        # Delete our temp handles
        del tmpdgrp
        del tmpblock

        # call base class constructor to distribute data
        super().__init__(
            mpicomm,
            self._detlist,
            self._samples,
            detindx=self._detindx,
            detranks=detranks,
            detbreaks=detbreaks,
            sampsizes=sampsizes,
            meta=meta,
        )

        return

    def __del__(self):
        self._close()
        try:
            del self._vol
        except:
            pass
        return

    def _open(self):
        """Open block and group handles."""
        if self._block is not None:
            raise RuntimeError("block is already open!")
        if self._dgrp is not None:
            raise RuntimeError("data group is already open!")
        self._block = tdsutils.find_obs(self._vol, self._blockparent, self._blockname)
        self._dgrp = self._block.group_get(self._dgrpname)
        return

    def _close(self):
        """Close block and group handles."""
        try:
            del self._dgrp
            self._dgrp = None
        except:
            pass
        try:
            del self._block
            self._block = None
        except:
            pass
        return

    @classmethod
    def create(
        cls,
        vol,
        path,
        detectors,
        samples,
        meta,
        azel,
        group_dets=STR_DETGROUP,
        group_fp=STR_FPGROUP,
        units="none",
    ):
        """Class method to create the underlying TIDAS objects.

        This method should ONLY be called by a single process.  It creates the
        detector data group and the detector focalplane offset group.

        Args:
            vol (tidas.MPIVolume):  The volume.
            path (str):  The path to this observation block.
            detectors (dictionary):  Specify the detector offsets.  Each key is
                the detector name, and each value is a quaternion tuple.
            samples (int):  The number of samples to use.
            meta (dict):  Extra scalar key / value parameters to write.
            azel (bool):  If True, this TOD will have ground-based data.
            group_dets (str):  The name of the TIDAS group containing detector
                and other telescope information at the detector sample rate.
            group_fp (str):  The name of the TIDAS group containing detector
                focalplane offsets.
            units (str):  The units of the detector timestreams.

        """
        tmr = Timer()
        tmr.start()

        # Get a handle to the observation node
        pfields = path.split("/")
        parent = "/".join(pfields[0:-1])
        block = tdsutils.find_obs(vol, parent, pfields[-1])
        blockname = pfields[-1]

        # Get the existing group names
        grpnames = block.group_names()
        if group_dets in grpnames:
            raise RuntimeError(
                "Creating new TODTidas, but detector group {} already exists".format(
                    group_dets
                )
            )

        if (detectors is None) or (samples is None):
            raise RuntimeError("detectors and samples must be specified")

        # Focalplane group.  If this group already exists, verify that its
        # data agrees with what we are passing in.

        if group_fp in grpnames:
            fpgrp = block.group_get(group_fp)
            # tmr.report_clear("create {} fp group get".format(blockname))
            for d in sorted(detectors.keys()):
                check = fpgrp.read(d, 0, 4)
                if not np.allclose(detectors[d], check):
                    raise RuntimeError(
                        "existing focalplane offset for {} " "does not match".format(d)
                    )
            # tmr.report_clear("create {} fp group read".format(blockname))
            del fpgrp
        else:
            # Create the FP schema
            schm = cls._create_fp_schema(list(sorted(detectors.keys())))
            # tmr.report_clear("create {} fp schema".format(blockname))

            # Create the FP group
            g = block.group_add(group_fp, tds.Group(schm, tds.Dictionary(), 4))
            # tmr.report_clear("create {} fp group add".format(blockname))

            # Write the FP offsets
            for d in sorted(detectors.keys()):
                g.write(d, 0, np.ascontiguousarray(detectors[d]))

            # tmr.report_clear("create {} fp group write offsets".format(blockname))

            del schm
            del g

        # Detector data group

        # Detector data group properties
        gprops = tdsutils.from_dict(meta)
        # tmr.report_clear("create {} det gprops".format(blockname))

        # Create the detector data schema
        schm = cls._create_det_schema(
            list(sorted(detectors.keys())), tds.DataType.float64, units, azel
        )
        # tmr.report_clear("create {} det schema".format(blockname))

        # Create the detector data group
        g = block.group_add(group_dets, tds.Group(schm, gprops, samples))
        # tmr.report_clear("create {} det group add".format(blockname))

        del schm
        del gprops
        del g
        del block

        return

    @staticmethod
    def _create_det_schema(detlist, datatype, units, azel=False):
        """Create a schema for a list of detectors.

        All detector timestreams will be set to the same type and units.  A
        flag field will be created for each detector.  Additional built-in
        fields will also be added to the schema.

        Args:
            detlist (list): a list of detector names
            datatype (tidas.DataType): the tidas datatype assigned to all
                detector fields.
            units (str): the units string assigned to all data fields.
            azel (bool): if True, data has Az/El pointing as well.

        Returns (tidas.Schema):
            Schema containing the data and flag fields.

        """
        fields = list()
        for c in ["X", "Y", "Z", "W"]:
            f = "{}_{}{}".format(STR_BORE, STR_QUAT, c)
            fields.append(tds.Field(f, tds.DataType.float64, "NA"))
            if azel:
                f = "{}_{}{}".format(STR_BOREAZEL, STR_QUAT, c)
                fields.append(tds.Field(f, tds.DataType.float64, "NA"))

        for c in ["X", "Y", "Z"]:
            f = "{}{}".format(STR_POS, c)
            fields.append(tds.Field(f, tds.DataType.float64, "NA"))

        for c in ["X", "Y", "Z"]:
            f = "{}{}".format(STR_VEL, c)
            fields.append(tds.Field(f, tds.DataType.float64, "NA"))

        fields.append(
            tds.Field("{}_{}".format(STR_FLAG, STR_COMMON), tds.DataType.uint8, "NA")
        )

        for d in detlist:
            fields.append(tds.Field(d, datatype, units))
            fields.append(
                tds.Field("{}_{}".format(STR_FLAG, d), tds.DataType.uint8, "NA")
            )

        return tds.Schema(fields)

    @staticmethod
    def _create_fp_schema(detlist):
        """Create a schema for the focalplane offsets.

        This schema will include one field per detector to hold the
        double-precision quaternion offsets of each detector from the
        boresight.

        Args:
            detlist (list): a list of detector names.

        Returns (tidas.Schema):
            Schema containing the data and flag fields.

        """
        fields = list()
        for d in detlist:
            fields.append(tds.Field(d, tds.DataType.float64, "NA"))
        return tds.Schema(fields)

    @property
    def volume(self):
        """The open TIDAS volume in use by this TOD."""
        return self._vol

    @property
    def block_path(self):
        """The metadata path to this TOD's observation."""
        return self._blockpath

    @property
    def group_det_name(self):
        """The TIDAS group name for the detector data in this TOD."""
        return self._dgrpname

    @property
    def group_fp_name(self):
        """The TIDAS group name for the focalplane data in this TOD."""
        return self._fpgrpname

    def detoffset(self):
        """
        Return dictionary of detector quaternions.

        This returns a dictionary with the detector names as the keys and the
        values are 4-element numpy arrays containing the quaternion offset
        from the boresight.

        Args:
            None

        Returns (dict):
            the dictionary of quaternions.
        """
        return dict(self._detquats)

    @function_timer
    def _read_cache_helper(self, prefix, comps, start, n, usecache):
        """
        Helper function to read multi-component data, pack into an
        array, optionally cache it, and return.
        """
        # Number of components we have
        ncomp = len(comps)

        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start

        if self.cache.exists(prefix):
            return self.cache.reference(prefix)[start : start + n, :]
        else:
            if usecache:
                # We cache the whole observation, regardless of what sample
                # range we will return.
                data = self.cache.create(
                    prefix, np.float64, (self.local_samples[1], ncomp)
                )
                for c in range(ncomp):
                    field = "{}{}".format(prefix, comps[c])
                    d = self._dgrp.read(field, offset, self.local_samples[1])
                    data[:, c] = d
                # Return just the desired slice
                return data[start : start + n, :]
            else:
                # Read and return just the slice we want
                data = np.zeros((n, ncomp), dtype=np.float64)
                for c in range(ncomp):
                    field = "{}{}".format(prefix, comps[c])
                    d = self._dgrp.read(field, offset, n)
                    data[:, c] = d
                return data

    def _write_helper(self, data, prefix, comps, start):
        """
        Helper function to write multi-component data.
        """
        # Number of components we have
        ncomp = len(comps)

        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start

        tmpdata = np.empty(data.shape[0], dtype=data.dtype, order="C")
        for c in range(ncomp):
            field = "{}{}".format(prefix, comps[c])
            tmpdata[:] = data[:, c]
            self._dgrp.write(field, offset, tmpdata)

        return

    def _get_boresight(self, start, n, usecache=True):
        # Cache name
        cachebore = "{}_{}".format(STR_BORE, STR_QUAT)
        # Read and optionally cache the boresight pointing.
        self._open()
        ret = self._read_cache_helper(
            cachebore, ["X", "Y", "Z", "W"], start, n, usecache
        )
        self._close()
        return ret

    def _put_boresight(self, start, data):
        # Data name
        borename = "{}_{}".format(STR_BORE, STR_QUAT)
        # Write data
        # self._writelock.lock()
        self._open()
        self._write_helper(data, borename, ["X", "Y", "Z", "W"], start)
        self._close()
        # self._writelock.unlock()
        return

    def _get_boresight_azel(self, start, n, usecache=True):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        # Cache name
        cachebore = "{}_{}".format(STR_BOREAZEL, STR_QUAT)
        # Read and optionally cache the boresight pointing.
        self._open()
        ret = self._read_cache_helper(
            cachebore, ["X", "Y", "Z", "W"], start, n, usecache
        )
        self._close()
        return ret

    def _put_boresight_azel(self, start, data):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        # Data name
        borename = "{}_{}".format(STR_BOREAZEL, STR_QUAT)
        # Write data
        # self._writelock.lock()
        self._open()
        self._write_helper(data, borename, ["X", "Y", "Z", "W"], start)
        self._close()
        # self._writelock.unlock()
        return

    def _get(self, detector, start, n):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Read from the data group and return
        self._open()
        ret = self._dgrp.read(detector, offset, n)
        self._close()
        return ret

    def _put(self, detector, start, data):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        # self._writelock.lock()
        self._open()
        self._dgrp.write(detector, offset, np.ascontiguousarray(data))
        self._close()
        # self._writelock.unlock()
        return

    def _get_flags(self, detector, start, n):
        # Field name
        field = "{}_{}".format(STR_FLAG, detector)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        self._open()
        ret = self._dgrp.read(field, offset, n)
        self._close()
        return ret

    def _put_flags(self, detector, start, flags):
        # Field name
        field = "{}_{}".format(STR_FLAG, detector)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        # self._writelock.lock()
        self._open()
        self._dgrp.write(field, offset, np.ascontiguousarray(flags))
        self._close()
        # self._writelock.unlock()
        return

    def _get_common_flags(self, start, n):
        # Field name
        field = "{}_{}".format(STR_FLAG, STR_COMMON)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Read from the data group and return
        self._open()
        ret = self._dgrp.read(field, offset, n)
        self._close()
        return ret

    def _put_common_flags(self, start, flags):
        # Field name
        field = "{}_{}".format(STR_FLAG, STR_COMMON)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        # self._writelock.lock()
        self._open()
        self._dgrp.write(field, offset, np.ascontiguousarray(flags))
        self._close()
        # self._writelock.unlock()
        return

    def _get_times(self, start, n):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        self._open()
        ret = self._dgrp.read_times(offset, n)
        self._close()
        return ret

    def _put_times(self, start, stamps):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        # self._writelock.lock()
        self._open()
        self._dgrp.write_times(offset, np.ascontiguousarray(stamps))
        self._close()
        # self._writelock.unlock()
        return

    def _get_pntg(self, detector, start, n):
        # Get boresight pointing (from disk or cache)
        bore = self._get_boresight(start, n)
        # Apply detector quaternion and return
        self._open()
        ret = qa.mult(bore, self._detquats[detector])
        self._close()
        return ret

    def _put_pntg(self, detector, start, data):
        raise RuntimeError(
            "TODTidas computes detector pointing on the fly."
            " Use the write_boresight() method instead."
        )
        return

    def _get_position(self, start, n, usecache=False):
        # Read and optionally cache the telescope position.
        self._open()
        ret = self._read_cache_helper(STR_POS, ["X", "Y", "Z"], start, n, usecache)
        self._close()
        return ret

    def _put_position(self, start, pos):
        # self._writelock.lock()
        self._open()
        self._write_helper(pos, STR_POS, ["X", "Y", "Z"], start)
        self._close()
        # self._writelock.unlock()
        return

    def _get_velocity(self, start, n, usecache=False):
        # Read and optionally cache the telescope velocity.
        self._open()
        ret = self._read_cache_helper(STR_VEL, ["X", "Y", "Z"], start, n, usecache)
        self._close()
        return ret

    def _put_velocity(self, start, vel):
        # self._writelock.lock()
        self._open()
        self._write_helper(vel, STR_VEL, ["X", "Y", "Z"], start)
        self._close()
        # self._writelock.unlock()
        return


@function_timer
def load_tidas(comm, detranks, path, mode, groupname, todclass, **kwargs):
    """Loads an existing TOAST dataset in TIDAS format.

    This takes a 2-level TOAST communicator and opens an existing TIDAS
    volume using the global communicator.  The opened volume handle is stored
    in the observation dictionary with the "tidas" key.  Similarly, the
    metadata path to the block within the volume is stored in the
    "tidas_block" key.

    The leaf nodes of the hierarchy are assumed to be the "observations".
    the observations are assigned to the process groups in a load-balanced
    way based on the number of samples in each detector group.

    For each observation, the specified TOD class is instantiated, and may
    initialize itself using the TIDAS block however it likes.

    Args:
        comm (toast.Comm): the toast Comm class for distributing the data.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        path (str):  the TIDAS volume path.
        mode (str): whether to open the file in read-only ("r") or
            read-write ("w") mode.  Default is read-only.
        groupname (str): the name of the group to use for load balancing
            the data distribution.
        todclass (TOD): a TIDAS-compatible TOD class, which must have a
            constructor that takes the MPI communicator, the TIDAS volume,
            the block path, and the size of the process grid in the detector
            direction.  All additional arguments should be keyword args.
        kwargs: All additional arguments to this function are passed to the
            TOD constructor for each observation.

    Returns (toast.Data):
        The distributed data object.

    """
    if not available:
        raise RuntimeError("tidas is not available")
        return None
    # the global communicator
    cworld = comm.comm_world
    # the communicator within the group
    cgroup = comm.comm_group
    # the communicator with all processes with
    # the same rank within their group
    crank = comm.comm_rank

    rank = 0
    if cworld is not None:
        rank = cworld.rank

    # Collectively open the volume.  We cannot use a context manager here,
    # since we are keeping a handle to the volume around for future use.
    # This means the volume will remain open for the life of the program,
    # or (hopefully) will get closed if the distributed data object is
    # destroyed.

    tm = None
    if mode == "w":
        tm = tds.AccessMode.write
    else:
        tm = tds.AccessMode.read
    vol = None
    if cworld is None:
        vol = tds.Volume(path, tm)
    else:
        vol = MPIVolume(cworld, path, tm)

    # Traverse the blocks of the volume and get the properties of the
    # observations so we can distribute them.

    obslist = []
    obsweight = dict()
    obspath = {}

    def procblock(pth, nm, current):
        subs = current.block_names()
        pthnm = "{}/{}".format(pth, nm)
        chld = None
        for s in subs:
            chld = current.block_get(s)
            procblock(pthnm, s, chld)
        if len(subs) == 0:
            # this is a leaf node
            obslist.append(nm)
            obspath[nm] = pthnm
            grpnames = current.group_names()
            if groupname not in grpnames:
                raise RuntimeError(
                    "observation {} does not have a group '{}'".format(nm, groupname)
                )
            grp = current.group_get(groupname)
            sch = grp.schema()
            nfield = len(sch.fields())
            del sch
            obsweight[nm] = nfield * grp.size()
            del grp
        del chld
        return

    if rank == 0:
        root = vol.root()
        toplist = root.block_names()
        bk = None
        for b in toplist:
            bk = root.block_get(b)
            procblock("", b, bk)
        del bk
        del root

    if cworld is not None:
        obslist = cworld.bcast(obslist, root=0)
        obspath = cworld.bcast(obspath, root=0)
        obsweight = cworld.bcast(obsweight, root=0)

    # Distribute observations based on number of samples

    dweight = [obsweight[x] for x in obslist]
    distobs = distribute_discrete(dweight, comm.ngroups)

    # Distributed data

    data = Data(comm)

    # Now every group adds its observations to the list

    firstobs = distobs[comm.group][0]
    nobs = distobs[comm.group][1]
    for ob in range(firstobs, firstobs + nobs):
        obs = dict()
        obs["tod"] = todclass(cgroup, detranks, vol, obspath[obslist[ob]], **kwargs)

        # Get the observation properties group
        pfields = obspath[obslist[ob]].split("/")
        parent = "/".join(pfields[0:-1])
        block = tdsutils.find_obs(vol, parent, pfields[-1])
        obsgroup = block.group_get(STR_OBSGROUP)

        # The observation properties
        obsprops = obsgroup.dictionary()

        props = tdsutils.to_dict(obsprops)
        obspat = re.compile("^obs_(.*)")
        for k, v in props.items():
            obsmat = obspat.match(k)
            if obsmat is not None:
                obs[obsmat.group(1)] = v

        del obsprops
        del obsgroup
        del block

        # Verify that the name on disk is what we expect
        if "name" in obs:
            if obs["name"] != obslist[ob]:
                raise RuntimeError(
                    "loading obs {}: name in file set to {}".format(
                        obslist[ob], obs["name"]
                    )
                )
        else:
            obs["name"] = obslist[ob]

        data.obs.append(obs)

    return data


class OpTidasExport(Operator):
    """Operator which writes data to a TIDAS volume.

    The volume is created at construction time, and the full metadata
    path inside the volume can be given for each observation.  If not given,
    all observations are exported to TIDAS blocks under the root block.

    Timestream data, flags, and boresight pointing are read from the
    current TOD for the observation and written to the TIDAS TOD.  Data can
    be read directly or copied from the cache.

    Args:
        path (str): The output TIDAS volume path.  If this already exists,
            ensure that you specify a group name in the TOD constructor
            options which does not already exist.
        todclass (TOD): a TIDAS-compatible TOD class, which must have a
            constructor that takes the MPI communicator, the TIDAS volume,
            the block path, and the size of the process grid in the detector
            direction.  All additional arguments should be keyword args.  This
            class must also have a class method called "create()", which takes
            arguments for the TIDAS volume, the block path, a dictionary of
            detectors, the number of samples, and a dictionary of metadata.
            All additional arguments should be keyword args.
        backend (str): The TIDAS backend type.  If not specified, the volume
            is assumed to already exist and will be opened for appending.  If
            specified, a new volume will be created.
        comp (str): The TIDAS compression type.  Used only when creating a
            new volume.
        backopts (dict): Extra options to the TIDAS backend.  Used only when
            creating a new volume.
        obspath (dict): (optional) each observation has a "name" and these
            should be the keys of this dictionary.  The values of the dict
            should be the metadata parent path of the observation inside
            the volume.  Default is all observations under the root.
        use_todchunks (bool): if True, use the chunks of the original TOD for
            data distribution.
        use_intervals (bool): if True, use the intervals in the observation
            dictionary for data distribution.
        create_opts (dict): dictionary of options to pass as kwargs to the
            TOD.create() classmethod.
        ctor_opts (dict): dictionary of options to pass as kwargs to the
            TOD constructor.
        cache_name (str):  The name of the cache object (<name>_<detector>) in
            the existing TOD to use for the detector timestream.  If None, use
            the read* methods from the existing TOD.
        cache_common (str):  The name of the cache object in the existing TOD
            to use for common flags.  If None, use the read* methods from the
            existing TOD.
        cache_flag_name (str):   The name of the cache object
            (<name>_<detector>) in the existing TOD to use for the flag
            timestream.  If None, use the read* methods from the existing TOD.

    """

    def __init__(
        self,
        path,
        todclass,
        backend=None,
        comp="none",
        backopts=dict(),
        obspath=None,
        use_todchunks=False,
        use_intervals=False,
        create_opts=dict(),
        ctor_opts=dict(),
        cache_common=None,
        cache_name=None,
        cache_flag_name=None,
    ):

        if not available:
            raise RuntimeError("tidas is not available")

        self._path = path.rstrip("/")
        self._todclass = todclass
        self._backend = None
        if backend is not None:
            if backend == "hdf5":
                self._backend = tds.BackendType.hdf5
            self._comp = None
            if comp == "none":
                self._comp = tds.CompressionType.none
            elif comp == "gzip":
                self._comp = tds.CompressionType.gzip
            elif comp == "bzip2":
                self._comp = tds.CompressionType.bzip2
        self._backopts = backopts
        self._obspath = obspath
        if use_todchunks and use_intervals:
            raise RuntimeError("cannot use both TOD chunks and Intervals")
        self._usechunks = use_todchunks
        self._useintervals = use_intervals
        self._create_opts = create_opts
        self._ctor_opts = ctor_opts
        self._cache_common = cache_common
        self._cache_name = cache_name
        self._cache_flag_name = cache_flag_name
        # We call the parent class constructor
        super().__init__()

    @function_timer
    def exec(self, data):
        """Export data to a TIDAS volume.

        Each group will write its list of observations as TIDAS blocks.

        For errors that prevent the export, this function will directly call
        MPI Abort() rather than raise exceptions.  This could be changed in
        the future if additional logic is implemented to ensure that all
        processes raise an exception when one process encounters an error.

        Args:
            data (toast.Data): The distributed data.

        """
        log = Logger.get()
        # the two-level toast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        worldrank = 0
        if cworld is not None:
            worldrank = cworld.rank
        grouprank = 0
        if cgroup is not None:
            grouprank = cgroup.rank

        create = None

        tmr = Timer()
        tmr.start()

        # Handle for the volume.
        vol = None

        # One process checks the path and creates the volume if needed.
        if worldrank == 0:
            if os.path.isdir(self._path):
                # We are appending
                if self._backend is not None:
                    msg = "TIDAS volume {} already exists, but a backend format was specified, which indicates a new volume should be created.".format(
                        self._path
                    )
                    log.error(msg)
                    if cworld is None:
                        raise RuntimeError(msg)
                    else:
                        cworld.Abort()
                create = False
            else:
                # We are creating a new volume.
                if self._backend is None:
                    msg = "TIDAS volume {} does not exist, and a backend format was not specified.".format(
                        self._path
                    )
                    log.error(msg)
                    if cworld is None:
                        raise RuntimeError(msg)
                    else:
                        cworld.Abort()
                create = True
                vol = tds.Volume(self._path, self._backend, self._comp, self._backopts)
                del vol
        if cworld is not None:
            create = cworld.bcast(create, root=0)

        if cworld is not None:
            cworld.barrier()
        if worldrank == 0:
            tmr.report_clear("TIDAS:  Check path / create volume")

        # All processes open the volume.  Note:  we *might* be exporting a new
        # detector group from a toast.Data instance that is already using this
        # volume.  In this case, we need to not modify existing metadata in
        # the volume and only add our new group.

        if cworld is None:
            vol = tds.Volume(self._path, tds.AccessMode.write)
        else:
            vol = MPIVolume(cworld, self._path, tds.AccessMode.write)

        if cworld is not None:
            cworld.barrier()
        if worldrank == 0:
            tmr.report_clear("TIDAS:  world open volume")

        # The rank zero process in each group creates or checks the observation
        # blocks and creates the TOD groups.

        if grouprank == 0:
            for obs in data.obs:
                # The existing TOD
                oldtod = obs["tod"]
                oldcomm = oldtod.mpicomm

                # Sanity check- the group communicator should be the same
                different_comm = False
                if cgroup is None:
                    if oldcomm is not None:
                        different_comm = True
                else:
                    if oldcomm is None:
                        different_comm = True
                    else:
                        comp = MPI.Comm.Compare(oldtod.mpicomm, cgroup)
                        if comp not in (MPI.IDENT, MPI.CONGRUENT):
                            different_comm = True
                if different_comm:
                    msg = "On export, original TOD comm ({}) is different from group comm ({}).  Comm compare = {}".format(
                        oldcomm, cgroup, comp
                    )
                    log.error(msg)
                    if cgroup is None:
                        raise RuntimeError(msg)
                    else:
                        cworld.Abort()

                # Get the name
                if "name" not in obs:
                    log.error("observation does not have a name, cannot export")
                    if cgroup is None:
                        raise RuntimeError()
                    else:
                        cworld.Abort()
                obsname = obs["name"]

                # Get the metadata path
                blockpath = ""
                if self._obspath is not None:
                    blockpath = self._obspath[obsname]

                detranks, sampranks = oldtod.grid_size
                rankdet, ranksamp = oldtod.grid_ranks

                # Either create the observation group, or verify that the
                # contents are consistent.

                # Get any scalar metadata in the existing observation dictionary
                oprops = dict()
                for k, v in obs.items():
                    if isinstance(v, (bool, int, float, str)):
                        oprops["obs_{}".format(k)] = v

                # Create or open the block in the volume that corresponds to
                # this observation.
                tob = tdsutils.find_obs(vol, blockpath, obsname)

                obsprops = None
                obsgroup = None

                if create:
                    # The observation properties
                    obsprops = tdsutils.from_dict(oprops)

                    # Create the observation group
                    obsgroup = tob.group_add(
                        STR_OBSGROUP, tds.Group(tds.Schema(), obsprops, 0)
                    )
                else:
                    # Get the observation properties group
                    obsgroup = tob.group_get(STR_OBSGROUP)

                    # The observation properties
                    obsprops = obsgroup.dictionary()

                    checkprops = tdsutils.to_dict(obsprops)
                    obspat = re.compile("^obs_(.*)")
                    for k, v in checkprops.items():
                        obsmat = obspat.match(k)
                        if obsmat is not None:
                            obskey = obsmat.group(1)
                            if (obskey not in obs) or (obs[obskey] != v):
                                raise RuntimeError(
                                    "Attempting to export to existing observation with missing / changed property '{}'".format(
                                        k
                                    )
                                )

                # Whatever intervals we are using for data distribution (TOD
                # chunks or the input observation intervals), we want to write
                # those to the TIDAS volume.

                ninterval = None
                if self._useintervals:
                    # List of current toast intervals
                    if "intervals" not in obs:
                        raise RuntimeError("Observation does not contain " "intervals")
                    ninterval = len(obs["intervals"])
                else:
                    # We are using TOD chunks
                    ninterval = len(oldtod.total_chunks)

                allintr = tob.intervals_names()
                if STR_DISTINTR in allintr:
                    # Intervals already exist- verify size
                    tintr = tob.intervals_get(STR_DISTINTR)
                    if tintr.size() != ninterval:
                        raise RuntimeError(
                            "Existing intervals have different "
                            "length than ones being exported"
                        )
                    del tintr
                else:
                    # Create distribution intervals
                    tintr = tob.intervals_add(
                        STR_DISTINTR, tds.Intervals(tds.Dictionary(), ninterval)
                    )
                    del tintr

                del obsgroup
                del obsprops
                del tob

                # Do we have ground pointing?
                azel = None
                try:
                    test = oldtod.read_boresight_azel(local_start=0, n=1)
                    azel = True
                    del test
                except:
                    azel = False

                self._todclass.create(
                    vol,
                    "/".join([blockpath, obsname]),
                    oldtod.detoffset(),
                    oldtod.total_samples,
                    oldtod.meta,
                    azel,
                    **self._create_opts
                )

        if cworld is not None:
            cworld.barrier()
        if worldrank == 0:
            tmr.report_clear("TIDAS:  Create obs groups and intervals")

        # print("group {} meta sync".format(comm.group), flush=True)
        if cworld is not None:
            vol.meta_sync()
        del vol

        # Now all metadata is written to disk and synced between processes.
        if cworld is not None:
            cworld.barrier()
        if worldrank == 0:
            tmr.report_clear("TIDAS:  World volume sync")

        # Every process group copies their TOD data for each observation.

        if cworld is None:
            vol = tds.Volume(self._path, tds.AccessMode.write)
        else:
            vol = MPIVolume(cgroup, self._path, tds.AccessMode.write)

        for obs in data.obs:
            # Get the name
            obsname = obs["name"]

            # Get the metadata path
            blockpath = ""
            if self._obspath is not None:
                blockpath = self._obspath[obsname]

            tob = tdsutils.find_obs(vol, blockpath, obsname)

            # The existing TOD
            oldtod = obs["tod"]
            detranks, sampranks = oldtod.grid_size
            rankdet, ranksamp = oldtod.grid_ranks

            # Write out out distribution intervals.  This is either from the
            # old TOD chunks (combined with timestamp information), or using
            # existing intervals from the old observation.

            if self._useintervals:
                # We are using the observation intervals (every process has
                # a copy of these).  The root process writes the data.
                if cgroup.rank == 0:
                    ilist = []
                    for intr in obs["intervals"]:
                        ilist.append(
                            tds.Intrvl(intr.start, intr.stop, intr.first, intr.last)
                        )
                    tintr = tob.intervals_get(STR_DISTINTR)
                    tintr.write(ilist)
                    del tintr
                    del ilist
                if cgroup is not None:
                    cgroup.barrier()
                if grouprank == 0:
                    tmr.report_clear(
                        "TIDAS:  Group write obs {} intervals".format(obsname)
                    )
            else:
                # We are using the TOD chunks and the timestamps to build
                # the intervals.  Gather the timestamps to the root process
                # and do this write on that process.
                if rankdet == 0:
                    psize = oldtod.local_samples[1]
                    pdata = oldtod.read_times()
                    psizes = None
                    if oldtod.grid_comm_row is None:
                        psizes = [psize]
                    else:
                        psizes = oldtod.grid_comm_row.gather(psize, root=0)
                    disp = None
                    stamps = None
                    if ranksamp == 0:
                        # Compute the displacements into the receive buffer.
                        disp = [0]
                        for ps in psizes[:-1]:
                            last = disp[-1]
                            disp.append(last + ps)
                        # allocate receive buffer
                        stamps = np.zeros(np.sum(psizes), dtype=np.float64)

                    if oldtod.grid_comm_row is None:
                        stamps[:] = pdata
                    else:
                        oldtod.grid_comm_row.Gatherv(
                            pdata, [stamps, psizes, disp, MPI.DOUBLE], root=0
                        )

                    if ranksamp == 0:
                        ilist = []
                        ckoff = 0
                        for chk in oldtod.total_chunks:
                            ifirst = ckoff
                            ilast = ckoff + chk - 1
                            istart = stamps[ifirst]
                            istop = stamps[ilast]
                            ilist.append(tds.Intrvl(istart, istop, ifirst, ilast))
                            ckoff += chk
                        tintr = tob.intervals_get(STR_DISTINTR)
                        tintr.write(ilist)
                        del tintr
                        del ilist
                if cgroup is not None:
                    cgroup.barrier()
                if grouprank == 0:
                    tmr.report_clear(
                        "TIDAS:  Group write {} chunk intervals".format(obsname)
                    )

            del tob

            # Determine data distribution chunks
            sampsizes = None
            if self._usechunks:
                sampsizes = oldtod.total_chunks
            elif self._useintervals:
                sampsizes = intervals_to_chunklist(
                    obs["intervals"], oldtod.total_samples
                )

            if cgroup is not None:
                cgroup.barrier()
            if grouprank == 0:
                tmr.report_clear("TIDAS:  Group compute {} sampsizes".format(obsname))

            # The new TIDAS TOD.  Note:  the TOD instance will maintain
            # a handle on the volume for its lifetime.

            obsdir = "/".join([blockpath, obsname])

            # print("export:  construct tod at {}".format(obsdir), flush=True)
            tod = self._todclass(
                cgroup,
                detranks,
                vol,
                obsdir,
                distintervals=sampsizes,
                **self._ctor_opts
            )
            # print("export:  done construct tod at {}".format(obsdir), flush=True)

            if cgroup is not None:
                cgroup.barrier()
            if grouprank == 0:
                tmr.report_clear("TIDAS:  Group instantiate {} TOD".format(obsname))

            # Sanity check that the process grids are the same
            new_detranks, new_sampranks = tod.grid_size
            new_rankdet, new_ranksamp = tod.grid_ranks

            if (
                (new_detranks != detranks)
                or (new_sampranks != sampranks)
                or (new_rankdet != rankdet)
                or (new_ranksamp != ranksamp)
            ):
                msg = "TIDAS: During export to obs ({}), process grid shape mismatch".format(
                    obsname
                )
                log.error(msg)
                if cgroup is None:
                    raise RuntimeError(msg)
                else:
                    cgroup.Abort()

            if cgroup is not None:
                cgroup.barrier()
            if grouprank == 0:
                tmr.report_clear(
                    "TIDAS:  Group check TOD {} process grid".format(obsname)
                )

            # Some data is common across all processes that share the same
            # time span (timestamps, boresight pointing, common flags).
            # Since we only need to write these once, we let the first
            # process row handle that.

            # print("export:  start data copy for {}".format(obsdir), flush=True)

            if rankdet == 0:
                # Only the first row of the process grid does this.
                for p in range(sampranks):
                    # Processes take turns in the sample direction.
                    if p == ranksamp:
                        tod.write_times(stamps=oldtod.read_times())
                        tod.write_boresight(data=oldtod.read_boresight())
                        if azel:
                            tod.write_boresight_azel(data=oldtod.read_boresight_azel())
                        tod.write_position(pos=oldtod.read_position())
                        tod.write_velocity(vel=oldtod.read_velocity())
                        if self._cache_common is None:
                            tod.write_common_flags(flags=oldtod.read_common_flags())
                        else:
                            ref = oldtod.cache.reference(self._cache_common)
                            tod.write_common_flags(flags=ref)
                            del ref
                    if tod.grid_comm_row is not None:
                        tod.grid_comm_row.barrier()

            if cgroup is not None:
                cgroup.barrier()
            if grouprank == 0:
                tmr.report_clear(
                    "TIDAS:  Group write {} timestamps and boresight".format(obsname)
                )

            # print("export:  finish common copy for {}".format(obsdir), flush=True)

            # Now every process takes turns writing their unique data
            groupsize = 1
            if cgroup is not None:
                groupsize = cgroup.size
            for p in range(groupsize):
                if p == grouprank:
                    for d in oldtod.local_dets:
                        if self._cache_name is None:
                            tod.write(detector=d, data=oldtod.read(detector=d))
                        else:
                            ref = oldtod.cache.reference(
                                "{}_{}".format(self._cache_name, d)
                            )
                            tod.write(detector=d, data=ref)
                            del ref
                        if self._cache_flag_name is None:
                            tod.write_flags(
                                detector=d, flags=oldtod.read_flags(detector=d)
                            )
                        else:
                            ref = oldtod.cache.reference(
                                "{}_{}".format(self._cache_flag_name, d)
                            )
                            tod.write_flags(detector=d, flags=ref)
                            del ref
                if cgroup is not None:
                    cgroup.barrier()

            if cgroup is not None:
                cgroup.barrier()
            if grouprank == 0:
                tmr.report_clear("TIDAS:  Group write {} data".format(obsname))

            del tod

        del vol

        return
