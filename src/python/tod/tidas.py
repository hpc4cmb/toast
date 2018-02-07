# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, MPILock

import sys
import os
import re

import numpy as np

from .. import qarray as qa

from .. import timing

from ..dist import Data, distribute_discrete
from ..op import Operator

from .tod import TOD
from .interval import Interval

available = False
# available = True
# try:
#     import tidas as tds
#     from tidas.mpi_volume import MPIVolume
# except:
#     available = False


# Module-level constants

STR_QUAT = "QUAT"
STR_FLAG = "FLAG"
STR_COMMON = "COMMON"
STR_BORE = "BORE"
STR_POS = "POS"
STR_VEL = "VEL"
STR_DISTINTR = "chunks"
STR_DETGROUP = "detectors"


def create_tidas_schema(detlist, typestr, units):
    """
    Create a schema for a list of detectors.

    All detector timestreams will be set to the same type and units.  A flag
    field will be created for each detector.  Additional built-in fields will
    also be added to the schema.

    Args:
        detlist (list): a list of detector names
        typestr (str): the tidas datatype assigned to all data fields.
        units (str): the units string assigned to all data fields.

    Returns (dict):
        Schema dictionary containing the data and flag fields.
    """
    schm = {}
    for c in ["X", "Y", "Z", "W"]:
        field = "{}_{}{}".format(STR_BORE, STR_QUAT, c)
        schm[field] = ("float64", "NA")

    for c in ["X", "Y", "Z"]:
        field = "{}{}".format(STR_POS, c)
        schm[field] = ("float64", "NA")

    for c in ["X", "Y", "Z"]:
        field = "{}{}".format(STR_VEL, c)
        schm[field] = ("float64", "NA")

    schm["{}_{}".format(STR_FLAG, STR_COMMON)] = ("uint8", "NA")

    for d in detlist:
        schm[d] = (typestr, units)
        schm["{}_{}".format(STR_FLAG, d)] = ("uint8", "NA")
    return schm


@timing.auto_timer
def create_tidas_obs(vol, parent, name, groups=None, intervals=None):
    """
    Create a single TIDAS block that represents an observation.

    This creates a new block to represent an observation, and then creates
    zero or more groups and intervals inside that block.  When writing to a
    new TIDAS volume, this function can be used in the pipeline script
    to set up the "leaf nodes" of the volume, each of which is a single
    observation.

    The detector group will have additional fields added to the schema
    that are expected by the TODTidas class for boresight pointing and
    telescope position / velocity.

    Args:
        vol (tidas.MPIVolume):  the volume.
        parent (str):  the path to the parent block.
        name (str):  the name of the observation block.
        groups (dict):  dictionary where the key is the group name and the
            value is a tuple containing the (schema, size, properties) of
            the group.  The objects in the tuple are exactly the arguments
            to the constructor of the tidas.Group object.
        intervals (dict):  dictionary where the key is the intervals name
            and the value is a tuple containing the (size, properties) of
            the intervals.  The objects in the tuple are exactly the arguments
            to the constructor of the tidas.Intervals object.
        detgroup (str):  The name of the TIDAS group containing detector
            and other telescope information at the detector sample rate.
            If there is only one data group, then this is optional.

    Returns (tidas.Block):
        The newly created block.
    """
    if not available:
        raise RuntimeError("tidas is not available")
        return

    # The root block
    root = vol.root()

    # Only create these objects on one process
    if vol.comm.rank == 0:

        # Descend tree to the parent node
        parentnodes = parent.split("/")
        par = root
        for pn in parentnodes:
            if pn != "":
                par = par.block_get(pn)

        # Create the observation block
        obs = par.block_add(name)

        # Create the groups
        if groups is not None:
            for grp, args in groups.items():
                g = tds.Group(schema=args[0], size=args[1], props=args[2])
                g = obs.group_add(grp, g)

        # Create the intrvals
        if intervals is not None:
            for intrvl, args in intervals.items():
                intr = tds.Intervals(size=args[0], props=args[1])
                intr = obs.intervals_add(intrvl, intr)
    return


@timing.auto_timer
def decode_tidas_quats(props):
    """
    Read detector quaternions from a TIDAS property dictionary.

    This extracts and assembles the quaternion offsets for each detector from
    the metadata properties from a TIDAS group.

    Args:
        props (dict):  the dictionary of properties associated with a TIDAS
            detector group.

    Returns (dict):
        a dictionary of detectors and their quaternions, each stored as a 4
        element numpy array.
    """
    quatpat = re.compile(r"(.*)_{}([XYZW])".format(STR_QUAT))
    loc = {"X":0, "Y":1, "Z":2, "W":3}
    quats = {}

    # Extract all quaternions from the properties based on the key names
    for key, val in props.items():
        mat = quatpat.match(key)
        if mat is not None:
            det = mat.group(1)
            comp = mat.group(2)
            if det not in quats:
                quats[det] = np.zeros(4, dtype=np.float64)
            quats[det][loc[comp]] = float(val)

    return quats


@timing.auto_timer
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
        props (dict):  the starting dictionary of properties.  The updated
            properties are returned.

    Returns (dict):
        a dictionary of detector quaternion values appended to the input.
    """
    ret = props
    if ret is None:
        ret = {}

    qname = ["X", "Y", "Z", "W"]
    for det, quat in detquats.items():
        for q in range(4):
            key = "{}_{}{}".format(det, STR_QUAT, qname[q])
            ret[key] = float(quat[q])

    return ret


class TODTidas(TOD):
    """
    This class provides an interface to a single TIDAS data block.

    An instance of this class reads and writes to a single TIDAS block which
    represents a TOAST observation.  The volume and specific block should
    already exist.  Groups and intervals within the observation may already
    exist or may be created with the provided helper methods.  All groups
    and intervals must exist prior to reading or writing from them.

    Detector pointing offsets from the boresight are given as quaternions,
    and are expected to be contained in the dictionary of properties
    found in the TIDAS group containing detector timestreams.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which this
            observation data is distributed.
        vol (tidas.MPIVolume):  the volume.
        path (str):  the path to this observation block.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        detgroup (str):  The name of the TIDAS group containing detector
            and other telescope information at the detector sample rate.
            If there is only one data group, then this is optional.
        distintervals (str):  Optional name of the TIDAS intervals that
            determines how the data should be distributed along the time
            axis.  Default is to distribute only by detector.

    """

    # FIXME: currently the data flags are stored in the same group as
    # the data.  Once TIDAS supports per-group backend options like
    # compression, we should move the flags to a separate group:
    #   https://github.com/hpc4cmb/tidas/issues/13

    def __init__(self, mpicomm, vol, path, detranks=1, detbreaks=None,
        detgroup=None, distintervals=None):

        if not available:
            raise RuntimeError("tidas is not available")

        # The root block
        root = vol.root()

        # Descend tree to the observation node
        nodes = path.split("/")
        blk = root
        for nd in nodes:
            if nd != "":
                sys.stdout.flush()
                blk = blk.block_get(nd)

        self._block = blk

        # Get the detector group
        self._dgrpname = None
        self._dgrp = None
        grpnames = self._block.group_names()

        if len(grpnames) == 1:
            self._dgrpname = grpnames[0]
            self._dgrp = self._block.group_get(grpnames[0])
        else:
            if detgroup is None:
                raise RuntimeError("You must specify the detector group if "
                    "multiple groups exist")
            else:
                self._dgrpname = detgroup
                self._dgrp = self._block.group_get(detgroup)

        # Get the detector quaternion offsets
        self._detquats = decode_tidas_quats(self._dgrp.props)
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
            self.detindx[det] = ind

        # Create an MPI lock to use for writing to the TIDAS volume.  We must
        # have only one writing process at a time.  Note that this lock is over
        # the communicator for this single observation (TIDAS block).
        # Processes writing to separate blocks have no restrictions.

        self._writelock = MPILock(mpicomm)

        # read intervals and set up distribution chunks.

        sampsizes = None
        self._distintervals = None
        self._intervals = None
        self._distint = None

        if distintervals is not None:
            self._distintervals = distintervals
            self._distint = self._block.intervals_get(distintervals)
            # Rank zero process reads and broadcasts intervals
            if vol.comm.rank == 0:
                self._intervals = self._distint.read()
            self._intervals = vol.comm.bcast(self._intervals, root=0)
            # Compute the contiguous spans of time for data distribution
            # based on the starting points of all intervals.
            sampsizes = [ (x[1].first - x[0].first) for x in
                zip(self._intervals[:-1], self._intervals[1:]) ]
            sampsizes.append( self._dgrp.size -
                self._intervals[-1].first )

        # call base class constructor to distribute data
        super().__init__(vol.comm, self._detlist, self._dgrp.size,
            detindx=self._detindx, detranks=detranks, detbreaks=detbreaks,
            sampsizes=sampsizes, meta=self._dgrp.props)


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


    @timing.auto_timer
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
        self._writelock.lock()
        self._write_helper(data, borename, ["X", "Y", "Z", "W"],
            start)
        self._writelock.unlock()
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
        self._writelock.lock()
        self._dgrp.write(detector, offset, data)
        self._writelock.unlock()
        return


    def _get_flags(self, detector, start, n):
        # Field name
        field = "{}_{}".format(STR_FLAG, detector)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Read from the data group and return
        cflags = self._get_common_flags(start, n)
        return (self._dgrp.read(field, offset, n), cflags)


    def _put_det_flags(self, detector, start, flags):
        # Field name
        field = "{}_{}".format(STR_FLAG, detector)
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        self._writelock.lock()
        self._dgrp.write(field, offset, flags)
        self._writelock.unlock()
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
        self._writelock.lock()
        self._dgrp.write(field, offset, flags)
        self._writelock.unlock()
        return


    def _get_times(self, start, n):
        # FIXME: as soon as the TIDAS group interface supports partial reading
        # of time stamps, change this to use that interface:
        #   https://github.com/hpc4cmb/tidas/issues/17
        # until then, workaround here by reading the timestamps as a normal
        # field.
        field = "TIDAS_TIME"
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Read from the data group and return
        return self._dgrp.read(field, offset, n)


    def _put_times(self, start, stamps):
        # FIXME: as soon as the TIDAS group interface supports partial reading
        # of time stamps, change this to use that interface:
        #   https://github.com/hpc4cmb/tidas/issues/17
        # until then, workaround here by reading the timestamps as a normal
        # field.
        field = "TIDAS_TIME"
        # Compute the sample offset of our local data
        offset = self.local_samples[0] + start
        # Write to the data group
        self._writelock.lock()
        self._dgrp.write(field, offset, stamps)
        self._writelock.unlock()
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
        self._writelock.lock()
        self._write_helper(pos, STR_POS, ["X", "Y", "Z"], start)
        self._writelock.unlock()
        return


    def _get_velocity(self, start, n, usecache=False):
        # Read and optionally cache the telescope velocity.
        return self._read_cache(STR_VEL, ["X", "Y", "Z"], start, n, usecache)


    def _put_velocity(self, start, vel):
        self._writelock.lock()
        self._write_helper(vel, STR_VEL, ["X", "Y", "Z"], start)
        self._writelock.unlock()
        return


@timing.auto_timer
def load_tidas(comm, path, mode="r", detranks=1, detbreaks=None, detgroup=None,
    distintervals=None):
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

    For each observation, the TOD data distribution parameters and the group
    communicator are passed to the TODTidas class.

    Args:
        comm (toast.Comm): the toast Comm class for distributing the data.
        path (str):  the TIDAS volume path.
        mode (string): whether to open the file in read-only ("r") or
                       read-write ("w") mode.  Default is read-only.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI group communicator size must be evenly
            divisible by this number.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        detgroup (str):  The name of the TIDAS group containing detector
            and other telescope information at the detector sample rate.
            If there is only one data group, then this is optional.
        distintervals (str):  Optional name of the TIDAS intervals that
            determines how the data should be distributed along the time
            axis.  Default is to distribute only by detector.

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

    # Collectively open the volume.  We cannot use a context manager here,
    # since we are keeping a handle to the volume around for future use.
    # This means the volume will remain open for the life of the program,
    # or (hopefully) will get closed if the distributed data object is
    # destroyed.

    vol = MPIVolume(cworld, path, mode=mode)

    # Traverse the blocks of the volume and get the properties of the
    # observations so we can distribute them.

    obslist = []
    obspath = {}
    obssize = {}

    def procblock(pth, nm, current):
        subs = current.block_names()
        pthnm = "{}/{}".format(pth, nm)
        for s in subs:
            chld = current.block_get(s)
            procblock(pthnm, s, chld)
        if len(subs) == 0:
            # this is a leaf node
            obslist.append(nm)
            obspath[nm] = pthnm
            grpnames = current.group_names()
            grpnm = detgroup
            if len(grpnames) == 1:
                grpnm = grpnames[0]
            grp = current.group_get(grpnm)
            obssize[nm] = grp.size
        return

    if cworld.rank == 0:
        root = vol.root()
        toplist = root.block_names()
        for b in toplist:
            bk = root.block_get(b)
            procblock("", b, bk)

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
        obs["tod"] = TODTidas(cgroup, vol, obspath[obslist[ob]],
            detranks=detranks, detgroup=detgroup, distintervals=distintervals)
        if "obs_id" in obs["tod"].group.props:
            obs["id"] = obs["tod"].group.props["obs_id"]
        if "obs_telescope_id" in obs["tod"].group.props:
            obs["telescope_id"] = obs["tod"].group.props["obs_telescope_id"]
        if "obs_site_id" in obs["tod"].group.props:
            obs["site_id"] = obs["tod"].group.props["obs_site_id"]

        obs["intervals"] = None
        if distintervals is not None:
            tilist = []
            if cgroup.rank == 0:
                blk = obs["tod"].block
                intr = blk.intervals_get(distintervals)
                ilist = intr.read()
                tilist = [ Interval(start=x.start, stop=x.stop, first=x.first,
                    last=x.last) for x in ilist ]
            tilist = cgroup.bcast(tilist, root=0)
            obs["intervals"] = tilist

        data.obs.append(obs)

    vol.meta_sync()

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
        path (str): the output TIDAS volume path (must not exist).
        backend (str): the TIDAS backend type.
        comp (str): the TIDAS compression type.
        backopts (dict): extra options to the TIDAS backend.
        obspath (dict): (optional) each observation has a "name" and these
            should be the keys of this dictionary.  The values of the dict
            should be the metadata parent path of the observation inside
            the volume.
        name (str): the name of the cache object (<name>_<detector>) to
            use for the detector timestream.  If None, use the TOD.
        common_flag_name (str):  the name of the cache object to use for
            common flags.  If None, use the TOD.
        flag_name (str):  the name of the cache object (<name>_<detector>) to
            use for the detector flags.  If None, use the TOD.
        units (str):  the units of the detector timestreams.
        usedist (bool):  if True, use the TOD total_chunks() method to get
            the chunking used for data distribution and replicate that as a
            set of TIDAS intervals.  Otherwise do not write any intervals
            to the output.
    """
    def __init__(self, path, backend="hdf5", comp="none", backopts=None,
        obspath=None, name=None, common_flag_name=None, flag_name=None,
        units="unknown", usedist=False):

        if not available:
            raise RuntimeError("tidas is not available")

        self._path = path.rstrip("/")
        self._backend = backend
        self._comp = comp
        self._backopts = backopts
        self._obspath = obspath
        self._cachename = name
        self._cachecomm = common_flag_name
        self._cacheflag = flag_name
        self._usedist = usedist
        self._units = units

        # We call the parent class constructor
        super().__init__()


    @timing.auto_timer
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
        # the two-level toast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        # One process checks that the path is OK
        if cworld.rank == 0:
            dname = os.path.dirname(self._path)
            if not os.path.isdir(dname):
                print("Directory for exported TIDAS volume ({}) does not "
                      "exist".format(dname), flush=True)
                cworld.Abort()
            if os.path.exists(self._path):
                print("Path for exported TIDAS volume ({}) already "
                      "exists".format(self._path), flush=True)
                cworld.Abort()
        cworld.barrier()

        # Collectively create the volume

        with MPIVolume(cworld, self._path, backend=self._backend,
            comp=self._comp) as vol:

            # First, we go through and add all observations and then sync
            # so that all processes have the full metadata.

            for obs in data.obs:
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

                # The existing TOD
                tod = obs["tod"]
                detranks, sampranks = tod.grid_size
                rankdet, ranksamp = tod.grid_ranks
                metadata = tod.meta()

                # Get the observation ID (used for RNG)
                obsid = 0
                if "id" in obs:
                    metadata["obs_id"] = obs["id"]

                # Get the telescope ID (used for RNG)
                obstele = 0
                if "telescope_id" in obs:
                    metadata["obs_telescope_id"] = obs["telescope_id"]
                if "site_id" in obs:
                    metadata["obs_site_id"] = obs["site_id"]

                # Optionally setup intervals for future data distribution
                intervals = None
                if self._usedist:
                    # This means that the distribution chunks in the time
                    # direction were intentional (not just the boundaries of
                    # some uniform distribution), and we want to write them
                    # out to tidas intervals so that we can use them for data
                    # distribution when this volume is read in later.
                    intervals = {STR_DISTINTR : (len(tod.total_chunks),
                        dict())}

                # Get detector quaternions and encode them for use in the
                # properties of the tidas group.  Combine this with the
                # existing observation properties into a single dictionary.

                props = encode_tidas_quats(tod.detoffset(), props=metadata)

                # Configure the detector group

                schm = create_tidas_schema(tod.detectors, "float64",
                    self._units)
                groups = {STR_DETGROUP : (schm, tod.total_samples, props)}

                # Create the block in the volume that corresponds to this
                # observation.

                create_tidas_obs(vol, blockpath, obsname, groups=groups,
                    intervals=intervals)

            # Sync metadata so all processes have all metadata.
            vol.meta_sync()

            # Now every process group goes through its observations and
            # actually writes the data.

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

                # The new TIDAS TOD
                distintervals = None
                if self._usedist:
                    distintervals = STR_DISTINTR
                tidastod = TODTidas(tod.mpicomm, vol, blockpath,
                    detranks=detranks, detgroup=STR_DETGROUP,
                    distintervals=distintervals)
                blk = tidastod.block

                # Some data is common across all processes that share the same
                # time span (timestamps, boresight pointing, common flags).
                # Since we only need to write these once, we let the first
                # process row handle that.

                # We are going to gather the timestamps to a single process
                # since we need them to convert between the existing TOD
                # chunks and times for the intervals.  The interval list is
                # common between all processes.

                if rankdet == 0:
                    grp = blk.group_get(STR_DETGROUP)
                    # Only the first row of the process grid does this...
                    # First process timestamps
                    rowdata = tod.grid_comm_row.gather(tod.read_times(),
                        root=0)
                    if ranksamp == 0:
                        full = np.concatenate(rowdata)
                        grp.write_times(full)
                        if self._usedist:
                            ilist = []
                            off = 0
                            for sz in tod.total_chunks:
                                ilist.append(tds.Intrvl(start=full[off],
                                    stop=full[off+sz-1], first=off,
                                    last=(off+sz-1)))
                                off += sz
                            intr = blk.intervals_get(STR_DISTINTR)
                            intr.write(ilist)
                        del full
                    del rowdata

                    # Next the boresight data. Serialize the writing
                    for rs in range(sampranks):
                        if ranksamp == rs:
                            tidastod.write_boresight(data=tod.read_boresight())
                        tod.grid_comm_row.barrier()

                    # Same with the common flags
                    ref = None
                    if self._cachecomm is not None:
                        ref = tod.cache.reference(self._cachecomm)
                    else:
                        ref = tod.read_common_flags()
                    for rs in range(sampranks):
                        if ranksamp == rs:
                            tidastod.write_common_flags(flags=ref)
                        tod.grid_comm_row.barrier()
                    del ref

                tod.mpicomm.barrier()

                # Now each process can write their unique data slice.

                # FIXME:  Although every write should be guarded by a mutex
                # lock, this does not seem to work in practice.  Instead, we
                # will serialize writes over the process grid.

                for p in range(tod.mpicomm.size):
                    if tod.mpicomm.rank == p:
                        for det in tod.local_dets:
                            ref = None
                            if self._cachename is not None:
                                ref = tod.cache.reference("{}_{}"\
                                    .format(self._cachename, det))
                            else:
                                ref = tod.read(detector=det)
                            tidastod.write(detector=det, data=ref)
                            del ref
                            ref = None
                            if self._cacheflag is not None:
                                ref = tod.cache.reference(
                                    "{}_{}".format(self._cacheflag, det))
                            else:
                                ref, cflg = tod.read_flags(detector=det)
                            tidastod.write_det_flags(detector=det, flags=ref)
                            del ref
                    tod.mpicomm.barrier()

        return
