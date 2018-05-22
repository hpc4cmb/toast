# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, MPILock

import sys
import os
import re

import numpy as np

from .. import qarray as qa
from .. import timing as timing

from ..dist import Data, distribute_discrete
from ..op import Operator

from .tod import TOD
from .interval import Interval, intervals_to_chunklist

available = True
try:
    import tidas as tds
    from tidas.mpi import MPIVolume
except:
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
STR_CACHEGROUP = "cache"
STR_OBSGROUP = "observation"


def to_dict(td):
    meta = dict()
    for k in td.keys():
        tp = td.get_type(k)
        if (tp == "d"):
            meta[k] = td.get_float64(k)
        elif (tp == "f"):
            meta[k] = td.get_float32(k)
        elif (tp == "l"):
            meta[k] = td.get_int64(k)
        elif (tp == "L"):
            meta[k] = td.get_uint64(k)
        elif (tp == "i"):
            meta[k] = td.get_int32(k)
        elif (tp == "I"):
            meta[k] = td.get_uint32(k)
        elif (tp == "h"):
            meta[k] = td.get_int16(k)
        elif (tp == "H"):
            meta[k] = td.get_uint16(k)
        elif (tp == "b"):
            meta[k] = td.get_int8(k)
        elif (tp == "B"):
            meta[k] = td.get_uint8(k)
        else:
            meta[k] = td.get_string(k)
    return meta


def from_dict(meta):
    td = tds.Dictionary()
    for k, v in meta.items():
        if isinstance(v, float):
            td.put_float64(k, v)
        elif isinstance(v, int):
            td.put_int64(k, v)
        else:
            td.put_string(k, str(v))
    return td


def create_tidas_schema(detlist, datatype, units, azel=False):
    """
    Create a schema for a list of detectors.

    All detector timestreams will be set to the same type and units.  A flag
    field will be created for each detector.  Additional built-in fields will
    also be added to the schema.

    Args:
        detlist (list): a list of detector names
        datatype (tidas.DataType): the tidas datatype assigned to all detector
            fields.
        units (str): the units string assigned to all data fields.
        azel (bool): if True, data has Az/El pointing as well.

    Returns (tidas.Schema):
        Schema containing the data and flag fields.
    """
    fields = list()
    for c in ["X", "Y", "Z", "W"]:
        f = "{}_{}{}".format(STR_BORE, STR_QUAT, c)
        fields.append( tds.Field(f, tds.DataType.float64, "NA") )
        if azel:
            f = "{}_{}{}".format(STR_BOREAZEL, STR_QUAT, c)
            fields.append( tds.Field(f, tds.DataType.float64, "NA") )

    for c in ["X", "Y", "Z"]:
        f = "{}{}".format(STR_POS, c)
        fields.append( tds.Field(f, tds.DataType.float64, "NA") )

    for c in ["X", "Y", "Z"]:
        f = "{}{}".format(STR_VEL, c)
        fields.append( tds.Field(f, tds.DataType.float64, "NA") )

    fields.append( tds.Field("{}_{}".format(STR_FLAG, STR_COMMON),
        tds.DataType.uint8, "NA") )

    for d in detlist:
        fields.append( tds.Field(d, datatype, units) )
        fields.append( tds.Field("{}_{}".format(STR_FLAG, d),
            tds.DataType.uint8, "NA") )

    return tds.Schema(fields)


def tidas_obs(vol, parent, name):
    """Return the TIDAS block representing an observation.

    If the block exists, a handle is returned.  Otherwise it is created.

    Args:
        vol (tidas.MPIVolume):  the volume.
        parent (str):  the path to the parent block.
        name (str):  the name of the observation block.

    Returns:
        (tidas.Block):  the block handle

    """
    # The root block
    root = vol.root()

    par = root
    if parent != "":
        # Descend tree to the parent node
        parentnodes = parent.split("/")
        for pn in parentnodes:
            if pn != "":
                par = par.block_get(pn)
    obs = None
    if name in par.block_names():
        obs = par.block_get(name)
    else:
        # Create the observation block
        obs = par.block_add(name, tds.Block())
    del par
    del root
    return obs


def decode_tidas_quats(props):
    """
    Read detector quaternions from a TIDAS property dictionary.

    This extracts and assembles the quaternion offsets for each detector from
    the metadata properties from a TIDAS group.

    Args:
        props (tidas.Dictionary):  the dictionary of properties associated with
            a TIDAS detector group.

    Returns:
        (tuple): dictionary of detectors and their quaternions, and a separate
            dictionary of the remaining metadata.

    """
    autotimer = timing.auto_timer()

    meta = to_dict(props)

    quatpat = re.compile(r"(.*)_{}([XYZW])".format(STR_QUAT))
    loc = {"X":0, "Y":1, "Z":2, "W":3}
    quats = dict()

    # Extract all quaternions from the properties based on the key names
    for key in props.keys():
        mat = quatpat.match(key)
        if mat is not None:
            det = mat.group(1)
            comp = mat.group(2)
            if det not in quats:
                quats[det] = np.zeros(4, dtype=np.float64)
            quats[det][loc[comp]] = props.get_float64(key)
            del meta[key]

    return quats, meta


def encode_tidas_quats(detquats, props=None):
    """
    Append detector quaternions to a dictionary.

    This takes a dictionary of detector quaternions and encodes the elements
    into named dictionary values suitable for creating a TIDAS group.  The
    entries are merged with the input dictionary and the result is returned.

    This extracts and assembles the quaternion offsets for each detector from
    the metadata of a particular group (the one containing the detector data).

    Args:
        detquats (dict):  a dictionary with keys of detector names and values
            containing 4-element numpy arrays with the quaternion.
        props (tidas.Dictionary):  the starting dictionary of properties.
            The updated properties are returned.

    Returns (tidas.Dictionary):
        a dictionary of detector quaternion values appended to the input.
    """
    autotimer = timing.auto_timer()
    ret = props
    if ret is None:
        ret = tds.Dictionary()

    qname = ["X", "Y", "Z", "W"]
    for det, quat in detquats.items():
        for q in range(4):
            key = "{}_{}{}".format(det, STR_QUAT, qname[q])
            ret.put_float64(key, float(quat[q]))

    return ret


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
        detectors (dictionary):  (Only when creating new groups) Specify the
            detector offsets.  Each key is the detector name, and each value is
            a quaternion tuple.
        samples (int):  (Only when creating new groups) The number of samples
            to use.
        azel (bool):  (Only when creating new groups) If True, this TOD will
            have ground-based data.
        meta (dict):  (Only when creating new groups) Extra scalar key / value
            parameters to write.
        units (str):  (Only when creating new groups) The units of the detector
            timestreams.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        distintervals (str OR list):  Optional name of the TIDAS intervals that
            determines how the data should be distributed along the time
            axis.  If a list is given use that explicit set of sample chunks.
        group_dets (str):  The name of the TIDAS group containing detector
            and other telescope information at the detector sample rate.
        export_name (str):  When exporting data, the name of the cache object
            (<name>_<detector>) to use for the detector timestream.  If None,
            use the TOD read* methods.  If this is specified, then those cache
            objects will NOT be exported to the cachegroup.
        export_common_flag_name (str):  When exporting data, the name of the
            cache object to use for common flags.  If None, use the TOD read*
            methods.  If this is specified, then that cache object will NOT be
            exported to the cachegroup.
        export_flag_name (str):  When exporting data, the name of the cache
            object (<name>_<detector>) to use for the detector flags.  If None,
            use the TOD read* methods.  If this is specified, then those cache
            objects will NOT be exported to the cachegroup.
        export_dist (str):  Export the current data distribution chunks to this
            interval name.

    """
    # FIXME: currently the data flags are stored in the same group as
    # the data.  Once TIDAS supports per-group backend options like
    # compression, we should move the flags to a separate group:
    #   https://github.com/hpc4cmb/tidas/issues/13

    def __init__(self, mpicomm, detranks, vol, path, detectors=None,
        samples=None, azel=False, meta=dict(), units="none", detbreaks=None,
        distintervals=None, group_dets=STR_DETGROUP,
        export_name=None, export_common_flag_name=None, export_flag_name=None,
        export_dist=None):

        if not available:
            raise RuntimeError("tidas is not available")

        self._export_cachename = export_name
        self._export_cachecomm = export_common_flag_name
        self._export_cacheflag = export_flag_name
        self._export_dist = export_dist

        # Get a handle to the observation node
        pfields = path.split("/")
        parent = "/".join(pfields[0:-1])
        self._block = tidas_obs(vol, parent, pfields[-1])

        # Get the existing group names
        grpnames = self._block.group_names()

        # Get the detector group
        self._dgrpname = group_dets

        self._createmode = False
        if (detectors is not None) and (samples is not None):
            self._createmode = True

        self._dgrp = None
        self._have_azel = None

        if self._createmode:
            if self._dgrpname in grpnames:
                raise RuntimeError("Creating new TODTidas, but detector group {} already exists".format(self._dgrpname))
            # We must sync after creating the group so that all processes get
            # the changes locally.  However, if mpicomm is different
            # from the communicator used when opening the volume, then this
            # will cause a hang.
            comp = MPI.Comm.Compare(vol.comm(), mpicomm)
            if comp not in (MPI.IDENT, MPI.CONGRUENT):
                print("When creating a new TOD, the volume must be opened "
                    "with the same communicator used for the TOD.")
                mpicomm.Abort()
            self._create_detgroup(mpicomm, self._dgrpname,
                samples, detectors, azel, meta, units)
            vol.meta_sync()
            # block reference invalidated by sync- fetch it again.
            del self._block
            self._block = tidas_obs(vol, parent, pfields[-1])
            self._have_azel = azel
            self._dgrp = self._block.group_get(self._dgrpname)
        else:
            if self._dgrpname not in grpnames:
                raise RuntimeError("Loading existing TODTidas data, but detector group {} does not exist".format(self._dgrpname))
            self._dgrp = self._block.group_get(self._dgrpname)
            # See whether we have ground based data
            schm = self._dgrp.schema()
            fields = schm.fields()
            if "{}_{}X".format(STR_BOREAZEL, STR_QUAT) in fields:
                self._have_azel = True
            del fields
            del schm

        # Get the detector quaternion offsets
        self._detquats, othermeta = decode_tidas_quats(self._dgrp.dictionary())
        self._detlist = sorted(list(self._detquats.keys()))

        # We need to assign a unique integer index to each detector.  This
        # is used when seeding the streamed RNG in order to simulate
        # timestreams.  For simplicity, and assuming that detector names
        # are not too long, we can convert the detector name to bytes and
        # then to an integer.

        self._detindx = {}
        for det in self._detlist:
            bdet = det.encode("utf-8")
            ind = None
            try:
                ind = int.from_bytes(bdet, byteorder="little")
            except:
                raise RuntimeError("Cannot convert detector name {} to a "
                    "unique integer- maybe it is too long?".format(det))
            self._detindx[det] = ind

        # Create an MPI lock to use for writing to the TIDAS volume.  We must
        # have only one writing process at a time.  Note that this lock is over
        # the communicator for this single observation (TIDAS block).
        # Processes writing to separate blocks have no restrictions.

        self._writelock = MPILock(mpicomm)

        # read intervals and set up distribution chunks.

        sampsizes = None

        if distintervals is not None:
            if isinstance(distintervals, (str)):
                # This is the name of a TIDAS intervals object
                distint = self._block.intervals_get(distintervals)
                # Rank zero process reads and broadcasts intervals
                if mpicomm.rank == 0:
                    intervals = distint.read()
                intervals = mpicomm.bcast(intervals, root=0)
                # Compute the contiguous spans of time for data distribution
                # based on the starting points of all intervals.
                sampsizes = intervals_to_chunklist(intervals, self._dgrp.size())

                del intervals
                del distint
            else:
                # This must be an explicit list
                sampsizes = list(distintervals)

        # call base class constructor to distribute data
        super().__init__(mpicomm, self._detlist, self._dgrp.size(),
            detindx=self._detindx, detranks=detranks, detbreaks=detbreaks,
            sampsizes=sampsizes, meta=othermeta)

        return


    def __del__(self):
        del self._dgrp
        del self._block
        return


    def _create_detgroup(self, mpicomm, groupname, samples, dets, azel, meta,
        units):
        # We must pass in the communicator, since it is not yet stored
        # in the class instance when we first call this function in the
        # constructor.
        if mpicomm.rank == 0:
            # Convert metadata to a tidas dictionary
            gprops = from_dict(meta)

            # Add quaternion offsets to the dictionary
            gprops = encode_tidas_quats(dets, props=gprops)

            # Create the schema
            schm = create_tidas_schema(list(sorted(dets.keys())),
                tds.DataType.float64, units, azel)

            # Create the group
            g = self._block.group_add(groupname, tds.Group(schm, gprops,
                samples))
            del schm
            del gprops
            del g

        mpicomm.barrier()
        return


    @property
    def block(self):
        """
        The TIDAS block for this TOD.

        This can be used for arbitrary access to other groups and intervals
        associated with this observation.
        """
        return self._block


    @property
    def group(self):
        """
        The TIDAS group for the detectors in this TOD.
        """
        return self._dgrp


    @property
    def groupname(self):
        """
        The TIDAS group name for the detectors in this TOD.
        """
        return self._dgrpname


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


    def _read_cache_helper(self, prefix, comps, start, n, usecache):
        """
        Helper function to read multi-component data, pack into an
        array, optionally cache it, and return.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        # Number of components we have
        ncomp = len(comps)

        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start

        if self.cache.exists(prefix):
            return self.cache.reference(prefix)[start:start+n,:]
        else:
            if usecache:
                # We cache the whole observation, regardless of what sample
                # range we will return.
                data = self.cache.create(prefix, np.float64,
                    (self.local_samples[1], ncomp))
                for c in range(ncomp):
                    field = "{}{}".format(prefix, comps[c])
                    d = self._dgrp.read(field, offset, self.local_samples[1])
                    data[:,c] = d
                # Return just the desired slice
                return data[start:start+n,:]
            else:
                # Read and return just the slice we want
                data = np.zeros((n, ncomp), dtype=np.float64)
                for c in range(ncomp):
                    field = "{}{}".format(prefix, comps[c])
                    d = self._dgrp.read(field, offset, n)
                    dat[:,c] = d
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
            tmpdata[:] = data[:,c]
            self._dgrp.write(field, offset, tmpdata)

        return


    def _get_boresight(self, start, n, usecache=True):
        # Cache name
        cachebore = "{}_{}".format(STR_BORE, STR_QUAT)
        # Read and optionally cache the boresight pointing.
        return self._read_cache_helper(cachebore, ["X", "Y", "Z", "W"],
            start, n, usecache)


    def _put_boresight(self, start, data):
        # Data name
        borename = "{}_{}".format(STR_BORE, STR_QUAT)
        # Write data
        #self._writelock.lock()
        self._write_helper(data, borename, ["X", "Y", "Z", "W"],
            start)
        #self._writelock.unlock()
        return


    def _get_boresight_azel(self, start, n, usecache=True):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        # Cache name
        cachebore = "{}_{}".format(STR_BOREAZEL, STR_QUAT)
        # Read and optionally cache the boresight pointing.
        return self._read_cache_helper(cachebore, ["X", "Y", "Z", "W"],
            start, n, usecache)


    def _put_boresight_azel(self, start, data):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        # Data name
        borename = "{}_{}".format(STR_BOREAZEL, STR_QUAT)
        # Write data
        #self._writelock.lock()
        self._write_helper(data, borename, ["X", "Y", "Z", "W"],
            start)
        #self._writelock.unlock()
        return


    def _get(self, detector, start, n):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Read from the data group and return
        return self._dgrp.read(detector, offset, n)


    def _put(self, detector, start, data):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        #self._writelock.lock()
        self._dgrp.write(detector, offset, np.ascontiguousarray(data))
        #self._writelock.unlock()
        return


    def _get_flags(self, detector, start, n):
        # Field name
        field = "{}_{}".format(STR_FLAG, detector)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        return self._dgrp.read(field, offset, n)


    def _put_flags(self, detector, start, flags):
        # Field name
        field = "{}_{}".format(STR_FLAG, detector)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        #self._writelock.lock()
        self._dgrp.write(field, offset, np.ascontiguousarray(flags))
        #self._writelock.unlock()
        return


    def _get_common_flags(self, start, n):
        # Field name
        field = "{}_{}".format(STR_FLAG, STR_COMMON)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Read from the data group and return
        return self._dgrp.read(field, offset, n)


    def _put_common_flags(self, start, flags):
        # Field name
        field = "{}_{}".format(STR_FLAG, STR_COMMON)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        #self._writelock.lock()
        self._dgrp.write(field, offset, np.ascontiguousarray(flags))
        #self._writelock.unlock()
        return


    def _get_times(self, start, n):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        return self._dgrp.read_times(offset, n)


    def _put_times(self, start, stamps):
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        #self._writelock.lock()
        self._dgrp.write_times(offset, np.ascontiguousarray(stamps))
        #self._writelock.unlock()
        return


    def _get_pntg(self, detector, start, n):
        # Get boresight pointing (from disk or cache)
        bore = self._get_boresight(start, n)
        # Apply detector quaternion and return
        return qa.mult(bore, self._detquats[detector])


    def _put_pntg(self, detector, start, data):
        raise RuntimeError("TODTidas computes detector pointing on the fly."
            " Use the write_boresight() method instead.")
        return


    def _get_position(self, start, n, usecache=False):
        # Read and optionally cache the telescope position.
        return self._read_cache(STR_POS, ["X", "Y", "Z"], start, n, usecache)


    def _put_position(self, start, pos):
        #self._writelock.lock()
        self._write_helper(pos, STR_POS, ["X", "Y", "Z"], start)
        #self._writelock.unlock()
        return


    def _get_velocity(self, start, n, usecache=False):
        # Read and optionally cache the telescope velocity.
        return self._read_cache(STR_VEL, ["X", "Y", "Z"], start, n, usecache)


    def _put_velocity(self, start, vel):
        #self._writelock.lock()
        self._write_helper(vel, STR_VEL, ["X", "Y", "Z"], start)
        #self._writelock.unlock()
        return


    def export(self, oldtod):
        # When this is called, the observation block already exists AND
        # the detector group has been created by the constructor.

        # The process grid for this TOD
        detranks, sampranks = self.grid_size
        rankdet, ranksamp = self.grid_ranks

        # Some data is common across all processes that share the same
        # time span (timestamps, boresight pointing, common flags).
        # Since we only need to write these once, we let the first
        # process row handle that.

        # We are going to gather the timestamps to a single process
        # since we need them to convert between the existing TOD
        # chunks and times for the intervals.  The interval list is
        # common between all processes.

        if rankdet == 0:
            # Only the first row of the process grid does this...
            # First process timestamps

            stamps = oldtod.read_times()
            rowdata = self.grid_comm_row.gather(stamps, root=0)

            if ranksamp == 0:
                full = np.empty(self.total_samples, dtype=np.float64, order="C")
                np.concatenate(rowdata, out=full)
                self._dgrp.write_times(0, full)
                if self._export_dist is not None:
                    # Optionally setup intervals for future data distribution.
                    ilist = []
                    off = 0
                    for sz in self.total_chunks:
                        ilist.append(tds.Intrvl(full[off], full[off+sz-1],
                            off, off+sz-1))
                        off += sz
                    intr = self._block.intervals_add(self._export_dist,
                        tds.Intervals(tds.Dictionary(), len(ilist)))
                    intr.write(ilist)
                    del ilist
                    del intr

                del full
            del rowdata

            # Next the boresight data. Serialize the writing
            for rs in range(sampranks):
                if ranksamp == rs:
                    if self._have_azel:
                        self.write_boresight_azel(
                            data=oldtod.read_boresight_azel())
                    self.write_boresight(data=oldtod.read_boresight())
                self.grid_comm_row.barrier()

            # Same with the common flags
            ref = None
            if self._export_cachecomm is not None:
                ref = oldtod.cache.reference(self._export_cachecomm)
            else:
                ref = oldtod.read_common_flags()
            for rs in range(sampranks):
                if ranksamp == rs:
                    self.write_common_flags(flags=ref)
                self.grid_comm_row.barrier()
            del ref

        self.mpicomm.barrier()

        # Now each process can write their unique data slice.

        # FIXME:  Although every write should be guarded by a mutex
        # lock, this does not seem to work in practice- there is a bug
        # in the MPILock class when applied to HDF5 calls (despite
        # extensive unit tests).  For now, we will serialize writes over
        # the process grid.

        for p in range(self.mpicomm.size):
            if self.mpicomm.rank == p:
                for det in self.local_dets:
                    ref = None
                    if self._export_cachename is not None:
                        ref = oldtod.cache.reference("{}_{}"\
                            .format(self._export_cachename, det))
                    else:
                        ref = oldtod.read(detector=det)
                    self.write(detector=det, data=ref)
                    del ref
                    ref = None
                    if self._export_cacheflag is not None:
                        ref = oldtod.cache.reference(
                            "{}_{}".format(self._export_cacheflag, det))
                    else:
                        ref = oldtod.read_flags(detector=det)
                    self.write_flags(detector=det, flags=ref)
                    del ref
            self.mpicomm.barrier()

        return


def load_tidas(comm, detranks, path, mode, todclass, group_dets, **kwargs):
    """
    Loads an existing TOAST dataset in TIDAS format.

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
        mode (string): whether to open the file in read-only ("r") or
            read-write ("w") mode.  Default is read-only.
        todclass (TOD): a TIDAS-compatible TOD class, which must have a
            constructor that takes the MPI communicator, the TIDAS volume,
            the block path, and the size of the process grid in the detector
            direction.  All additional arguments should be keyword args.
        group_dets (str):  The name of the TIDAS group containing detector
            and other telescope information at the detector sample rate.
        kwargs: All additional arguments to this function are passed to the
            TOD constructor for each observation.

    Returns (toast.Data):
        The distributed data object.
    """
    if not available:
        raise RuntimeError("tidas is not available")
        return None
    autotimer = timing.auto_timer()
    # the global communicator
    cworld = comm.comm_world
    # the communicator within the group
    cgroup = comm.comm_group
    # the communicator with all processes with
    # the same rank within their group
    crank = comm.comm_rank

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
    vol = MPIVolume(cworld, path, tm)

    # Traverse the blocks of the volume and get the properties of the
    # observations so we can distribute them.

    obslist = []
    obspath = {}
    obssize = {}

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
            if group_dets not in grpnames:
                raise RuntimeError("observation {} does not have a group '{}'".format(nm, group_dets))
            grp = current.group_get(group_dets)
            obssize[nm] = grp.size()
        del chld
        del grp
        return

    if cworld.rank == 0:
        root = vol.root()
        toplist = root.block_names()
        for b in toplist:
            bk = root.block_get(b)
            procblock("", b, bk)
        del bk
        del root

    obslist = cworld.bcast(obslist, root=0)
    obspath = cworld.bcast(obspath, root=0)
    obssize = cworld.bcast(obssize, root=0)

    # Distribute the observations among groups

    obssizelist = [ obssize[x] for x in obslist ]
    distobs = distribute_discrete(obssizelist, comm.ngroups)

    # Distributed data

    data = Data(comm)

    # Now every group adds its observations to the list

    firstobs = distobs[comm.group][0]
    nobs = distobs[comm.group][1]
    for ob in range(firstobs, firstobs+nobs):
        obs = {}
        obs["name"] = obslist[ob]
        obs["tidas"] = vol
        obs["tidas_block"] = obspath[obslist[ob]]
        obs["tod"] = todclass(cgroup, detranks, vol, obspath[obslist[ob]],
            group_dets=group_dets, **kwargs)

        # Get the observation properties group
        obsgroup = obs["tod"].block.group_get(STR_OBSGROUP)

        # The observation properties
        obsprops = obsgroup.dictionary()

        props = to_dict(obsprops)
        obspat = re.compile("^obs_(.*)")
        for k, v in props.items():
            obsmat = obspat.match(k)
            if obsmat is not None:
                obs[obsmat.group(1)] = v

        del obsprops
        del obsgroup

        data.obs.append(obs)

    return data


class OpTidasExport(Operator):
    """
    Operator which writes data to a TIDAS volume.

    The volume is created at construction time, and the full metadata
    path inside the volume can be given for each observation.  If not given,
    all observations are exported to TIDAS blocks under the root block.

    Timestream data, flags, and boresight pointing are read from the
    current TOD for the observation and written to the TIDAS TOD.  Data can
    be read directly or copied from the cache.

    Args:
        path (str): The output TIDAS volume path (must not exist).
        todclass (TOD): a TIDAS-compatible TOD class, which must have a
            constructor that takes the MPI communicator, the TIDAS volume,
            the block path, and the size of the process grid in the detector
            direction.  All additional arguments should be keyword args.
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
            the volume.
        use_todchunks (bool): if True, use the chunk of the original TOD for
            data distribution.
        use_intervals (bool): if True, use the intervals in the observation
            dictionary for data distribution.
        kwargs: All additional arguments to this constructor are passed to the
            TOD constructor for each observation.

    """
    def __init__(self, path, todclass, backend=None, comp="none",
        backopts=dict(), obspath=None, use_todchunks=False,
        use_intervals=False, **kwargs):

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
        self._kwargs = kwargs
        # We call the parent class constructor
        super().__init__()


    def exec(self, data):
        """
        Export data to a TIDAS volume.

        Each group will write its list of observations as TIDAS blocks.

        For errors that prevent the export, this function will directly call
        MPI Abort() rather than raise exceptions.  This could be changed in
        the future if additional logic is implemented to ensure that all
        processes raise an exception when one process encounters an error.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        # the two-level toast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        create = None

        # One process checks the path
        if cworld.rank == 0:
            if os.path.isdir(self._path):
                # We are appending
                if self._backend is not None:
                    print("TIDAS volume {} already exists, but a backend "
                          "format was specified, which indicates a new volume"
                          " should be created.".format(self._path), flush=True)
                    cworld.Abort()
                create = False
            else:
                # We are creating a new volume.
                if self._backend is None:
                    print("TIDAS volume {} does not exist, and a backend "
                          "format was not specified.".format(self._path), flush=True)
                    cworld.Abort()
                create = True
        create = cworld.bcast(create, root=0)

        # Collectively create the volume

        vol = None
        if create:
            vol = MPIVolume(cworld, self._path, self._backend, self._comp,
                self._backopts)
        else:
            vol = MPIVolume(cworld, self._path, tds.AccessMode.write)

        # First, we go through and add all observations and then sync
        # so that all processes have the full metadata.  Only the root
        # process in each group creates the observations used by that
        # group.  The observation properties are written to a special
        # group, so that they are independent of the groups that might
        # be used by TOD classes.

        if cgroup.rank == 0:
            for obs in data.obs:
                # The existing TOD
                tod = obs["tod"]

                # Sanity check- the group communicator should be the same
                comp = MPI.Comm.Compare(tod.mpicomm, cgroup)
                if comp not in (MPI.IDENT, MPI.CONGRUENT):
                    print("On export, original TOD comm is different from "
                        "group comm")
                    cworld.Abort()

                # Get the name
                if "name" not in obs:
                    print("observation does not have a name, cannot export",
                          flush=True)
                    cworld.Abort()
                obsname = obs["name"]

                # Get the metadata path
                blockpath = ""
                if self._obspath is not None:
                    blockpath = self._obspath[obsname]

                detranks, sampranks = tod.grid_size
                rankdet, ranksamp = tod.grid_ranks

                # Either create the observation group, or verify that the
                # contents are consistent.

                # Get any scalar metadata in the existing observation dictionary
                oprops = dict()
                for k, v in obs.items():
                    if isinstance(v, (bool, int, float, str)):
                        oprops["obs_{}".format(k)] = v

                # Create or open the block in the volume that corresponds to
                # this observation.
                tob = tidas_obs(vol, blockpath, obsname)

                if create:
                    # The observation properties
                    obsprops = from_dict(oprops)

                    # Create the observation group
                    obsgroup = tob.group_add(STR_OBSGROUP,
                        tds.Group(tds.Schema(), obsprops, 0))
                else:
                    # Get the observation properties group
                    obsgroup = tob.group_get(STR_OBSGROUP)

                    # The observation properties
                    obsprops = obsgroup.dictionary()

                    checkprops = to_dict(obsprops)
                    obspat = re.compile("^obs_(.*)")
                    for k, v in checkprops.items():
                        obsmat = obspat.match(k)
                        if obsmat is not None:
                            if (k not in obs) or (obs[k] != v):
                                raise RuntimeError("Attempting to export to existing observation with missing / changed property '{}'".format(k))
                del obsgroup
                del obsprops
                del tob

        # Close the volume.  This syncs the metadata.
        del vol
        cworld.barrier()

        # Reopen the volume separately for each group communicator.  This is
        # fine to do since the groups are working on a disjoint set of
        # observations.

        vol = MPIVolume(cgroup, self._path, tds.AccessMode.write)

        # Now create the detector data groups and sync.
        for obs in data.obs:
            # Get the name
            obsname = obs["name"]

            # Get the metadata path
            blockpath = obsname
            if self._obspath is not None:
                blockpath = "{}/{}".format(self._obspath[obsname], obsname)

            # The existing TOD
            tod = obs["tod"]
            detranks, sampranks = tod.grid_size
            rankdet, ranksamp = tod.grid_ranks

            # Do we have ground pointing?
            azel = None
            if cgroup.rank == 0:
                try:
                    test = tod.read_boresight_az(local_start=0, n=1)
                    azel = True
                except:
                    azel = False
            azel = cgroup.bcast(azel, root=0)

            # The new TIDAS TOD

            tidastod = self._todclass(cgroup, detranks, vol, blockpath,
                detectors=tod.detoffset(), samples=tod.total_samples, azel=azel,
                meta=tod.meta(), **self._kwargs)

            # Sanity check that the process grids are the same
            new_detranks, new_sampranks = tidastod.grid_size
            new_rankdet, new_ranksamp = tidastod.grid_ranks

            if (new_detranks != detranks) or (new_sampranks != sampranks) or \
                (new_rankdet != rankdet) or (new_ranksamp != ranksamp):
                print("During export to obs ({}), process grid shape mismatch"\
                      .format(obsname), flush=True)
                cworld.Abort()

            del tidastod

        vol.meta_sync()
        cgroup.barrier()

        # Now every process group goes through its observations and
        # actually exports the data.

        for obs in data.obs:
            # Get the name
            obsname = obs["name"]

            # Get the metadata path
            blockpath = obsname
            if self._obspath is not None:
                blockpath = "{}/{}".format(self._obspath[obsname], obsname)

            # The existing TOD
            tod = obs["tod"]

            # Construct a TIDAS Tod from disk.
            dist = None
            if self._usechunks:
                dist = tod.total_chunks
            elif self._useintervals:
                if "intervals" not in obs:
                    raise RuntimeError("Observation does not contain intervals"
                        ", cannot distribute using them")
                dist = intervals_to_chunklist(obs["intervals"],
                    tod.total_samples)

            exdist = None
            if dist is not None:
                exdist = STR_DISTINTR

            tidastod = self._todclass(cgroup, detranks, vol, blockpath,
                distintervals=dist, export_dist=exdist, **self._kwargs)

            # Do the data export
            tidastod.export(tod)

            del tidastod

        del vol

        return
