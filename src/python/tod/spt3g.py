# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import sys
import os
import re

import numpy as np

from .. import qarray as qa

from ..dist import Data, distribute_discrete
from ..op import Operator

from .tod import TOD
from .interval import Interval, intervals_to_chunklist

available = True
try:
    from spt3g import core as c3g
    #from spt3g import coordinateutils as c3c
except:
    available = False


# Module-level constants

STR_TIME = "TIMESTAMPS"
CACHE_PREF = "SPT3G"
STR_FLAG = "FLAG"
STR_COMMON = "COMMON"
STR_QUAT = "QUAT"
STR_BORE = "BORE"
STR_BOREAZEL = "BOREAZEL"
STR_POS = "POS"
STR_VEL = "VEL"
STR_DET = "DET"

# Used to determine file breaks when writing new data.  This
# is the desired approximate file size in bytes.
TARGET_FRAMEFILE_SIZE = 500000000
#TARGET_FRAMEFILE_SIZE = 5000000


def from_g3_type(val):
    if isinstance(val, bool):
        return bool(val)
    elif isinstance(val, int):
        return int(val)
    elif isinstance(val, float):
        return float(val)
    else:
        check = str(val)
        if check == "NONE":
            return None
        else:
            return check
    return None


def to_g3_type(val):
    if val is None:
        return c3g.G3String("NONE")
    elif isinstance(val, (bool)):
        return c3g.G3Bool(val)
    elif isinstance(val, (int)):
        return c3g.G3Int(val)
    elif isinstance(val, (float)):
        return c3g.G3Double(val)
    else:
        return c3g.G3String(val)


def read_spt3g_obs(file):
    reader = c3g.G3Reader(file, 1)
    f = list(reader(None))
    obframe = f[0]
    obspat = re.compile("^obs_(.*)")
    detpat = re.compile("{}_{}-(.*)".format(STR_QUAT, STR_DET))

    obs = dict()
    props = dict()
    dets = dict()
    nsamp = None
    for k in obframe.keys():
        if k == "samples":
            nsamp = int(obframe[k])
        else:
            obsmat = obspat.match(k)
            detmat = detpat.match(k)
            if obsmat is not None:
                obs[obsmat.group(1)] = from_g3_type(obframe[k])
            elif detmat is not None:
                dets[detmat.group(1)] = np.array(obframe[k], dtype=np.float64)
            else:
                props[k] = from_g3_type(obframe[k])
    return obs, props, dets, nsamp


def write_spt3g_obs(writer, props, dets, nsamp):
    f = c3g.G3Frame(c3g.G3FrameType.Observation)
    for k, v in props.items():
        f[k] = to_g3_type(v)
    f["samples"] = c3g.G3Int(nsamp)
    for k, v in dets.items():
        f["{}_{}-{}".format(STR_QUAT, STR_DET, k)] = c3g.G3VectorDouble(v)
    writer(f)
    return


def read_spt3g_framesizes(file):
    sizes = list()
    obs = True
    for frame in c3g.G3File(file):
        if obs:
            obs = False
            continue
        sizes.append(len(frame[STR_TIME]))
    return sizes


def bytes_per_sample(ndet, have_azel):
    # For each sample we have:
    #   - 1 x 8 bytes for timestamp
    #   - 1 x 1 bytes for common flag
    #   - 4 x 8 bytes for boresight RA/DEC quats
    #   - 4 x 8 bytes for boresight Az/El quats (optional)
    #   - 3 x 8 bytes for telescope position
    #   - 3 x 8 bytes for telescope velocity
    #   - 1 x 8 bytes x number of dets
    #   - 1 x 1 bytes x number of dets
    azbytes = 0
    if have_azel:
        azbytes = 32
    persample = 8 + 1 + 32 + azbytes + 24 + 24 + 9 * ndet
    return persample


class TOD3G(TOD):
    """This class provides an interface to a directory of spt3g frame files.

    An instance of this class loads a directory of frame files into memory.
    This data may be manipulated in memory (using the write* methods), but
    the modifications will not be stored to disk until the export method is
    called.  If creating a new data set, additional parameters are used to
    set up the initial detector information.

    Observation properties and detector pointing offsets from the boresight are
    given as quaternions, and are stored in an observation frame at the
    start of the sequence of frame files.  The sequence of frame files are
    named with a prefix and a number which represents the first sample of that
    frame relative to the start of the observation.

    This class is generic- it can only read / write standard data streams
    and objects in the TOD.cache.  More specialized uses for specific
    experiments should be implemented in their own classes.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which this
            observation data is distributed.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        path (str):  The path to this observation directory.
        prefix (str):  The prefix string for the frame files.  Different
            prefix values allow grouping sets of frames for different
            frequencies, etc in the same observation directory.
        detectors (dictionary):  (Only when creating new groups) Specify the
            detector offsets.  Each key is the detector name, and each value is
            a quaternion tuple.
        samples (int):  (Only when creating) The number of samples
            to use.
        framesizes (list):  (Only when creating) The list of frame sizes.
            Total must equal samples.
        azel (bool):  (Only when creating) If True, this TOD will
            have ground-based data.
        meta (dict):  (Only when creating) Extra scalar key / value
            parameters to write.
        units (G3TimestreamUnits):  (Only when creating new groups) The units
            of the detector timestreams.
        detbreaks (list):  Optional list of hard breaks in the detector
            distribution.
        export_name (str):  When exporting data, the name of the cache object
            (<name>_<detector>) to use for the detector timestream.  If None,
            use the TOD read* methods.  If this is specified, then those cache
            objects will NOT be exported to the cache frames.
        export_common_flag_name (str):  When exporting data, the name of the
            cache object to use for common flags.  If None, use the TOD read*
            methods.  If this is specified, then that cache object will NOT be
            exported to the cache frames.
        export_flag_name (str):  When exporting data, the name of the cache
            object (<name>_<detector>) to use for the detector flags.  If None,
            use the TOD read* methods.  If this is specified, then those cache
            objects will NOT be exported to the cache frames.

    """
    def __init__(self, mpicomm, detranks, path=None, prefix=None,
        detectors=None, samples=None, framesizes=None, azel=False, meta=dict(),
        units=None, detbreaks=None, export_name=None,
        export_common_flag_name=None, export_flag_name=None):

        if not available:
            raise RuntimeError("spt3g is not available")

        self._export_cachename = export_name
        self._export_cachecomm = export_common_flag_name
        self._export_cacheflag = export_flag_name

        if (path is None) or (prefix is None):
            raise RuntimeError("The path and prefix options must be specified")
        self._path = path
        self._prefix = prefix

        # When constructing an instance of this class, we are either creating
        # a new data set in memory or reading from disk.  Make sure that all
        # the required inputs are given in either case.
        dets = None
        nsamp = None
        props = None
        self._have_azel = None
        self._units = None
        self._frame_sizes = None
        self._frame_sample_offs = None
        self._files = None
        self._file_sample_offs = None
        self._file_frame_offs = None

        self._createmode = False
        if (detectors is not None) and (samples is not None):
            self._createmode = True

        if mpicomm.rank == 0:
            if self._createmode:
                # We are creating a new data set.
                dets = detectors
                nsamp = samples
                props = meta
                self._have_azel = azel
                self._units = units
                self._frame_sizes = framesizes
                if self._frame_sizes is None:
                    self._frame_sizes = [ nsamp ]
                self._frame_sample_offs = list()
                self._files = list()
                self._file_sample_offs = list()
                self._file_frame_offs = list()

                # Compute the future frame file breaks that would be used
                # if exporting the data.  We ignore the observation frame since
                # it is small.
                sampbytes = bytes_per_sample(len(dets), azel)
                filebytes = 0
                filesamps = 0
                fileframes = 0
                fileoff = 0
                fileframeoff = 0
                sampoff = 0
                for fr in self._frame_sizes:
                    frbytes = fr * sampbytes
                    if filebytes + frbytes > TARGET_FRAMEFILE_SIZE:
                        # Start a new file
                        self._file_sample_offs.append(fileoff)
                        self._file_frame_offs.append(fileframeoff)
                        self._files.append(os.path.join(path,
                            "{}_{:08d}.g3".format(prefix, fileoff)))
                        fileoff += filesamps
                        fileframeoff += fileframes
                        filesamps = 0
                        fileframes = 0
                        filebytes = 0
                    # Append frame to current file
                    filesamps += fr
                    fileframes += 1
                    filebytes += frbytes
                    self._frame_sample_offs.append(sampoff)
                    sampoff += fr
                # process the last file
                self._file_sample_offs.append(fileoff)
                self._file_frame_offs.append(fileframeoff)
                self._files.append(os.path.join(path,
                    "{}_{:08d}.g3".format(prefix, fileoff)))
            else:
                # We must have existing data
                self._frame_sizes = list()
                self._frame_sample_offs = list()
                self._files = list()
                self._file_sample_offs = list()
                self._file_frame_offs = list()

                # Parse frame files to determine sample and frame breaks
                fpat = re.compile(r"{}_(\d+).g3".format(prefix))

                frameoff = 0
                checkoff = 0
                for root, dirs, files in os.walk(path, topdown=True):
                    for f in sorted(files):
                        fmat = fpat.match(f)
                        if fmat is not None:
                            ffile = os.path.join(path, f)
                            # Is this the first frame file?  If so parse the
                            # observation frame.
                            if nsamp is None:
                                obs, props, dets, nsamp = read_spt3g_obs(ffile)
                                #print("Read obs:")
                                #print(obs, flush=True)
                                props.update(obs)
                                #print("Read props:")
                                #print(props, flush=True)
                                if "units" in props:
                                    self._units = props["units"]
                                if "have_azel" in props:
                                    self._have_azel = props["have_azel"]

                            fsampoff = int(fmat.group(1))
                            if fsampoff != checkoff:
                                raise RuntimeError("frame file {} is at sample offset {}, are some files missing?".format(ffile, checkoff))

                            self._files.append(ffile)
                            self._file_sample_offs.append(fsampoff)
                            self._file_frame_offs.append(frameoff)

                            fsizes = read_spt3g_framesizes(ffile)
                            fsoff = fsampoff
                            for fs in fsizes:
                                self._frame_sample_offs.append(fsoff)
                                fsoff += fs
                            self._frame_sizes.extend(fsizes)
                            frameoff += len(fsizes)
                            checkoff += np.sum(fsizes)
                    break
                if len(self._files) == 0:
                    raise RuntimeError(
                        "No frames found at '{}' with prefix '{}'".\
                        format(path, prefix))

                # print(self._files)
                # print(self._file_frame_offs)
                # print(self._file_sample_offs)
                # print(self._frame_sizes)
                # print(self._frame_sample_offs)

                # check that the total samples match the obs frame
                if nsamp != checkoff:
                    raise RuntimeError("observation frame specifies {} samples, but sum of frame files is {} samples".format(nsamp, checkoff))

        dets = mpicomm.bcast(dets, root=0)
        self._detquats = dets
        nsamp = mpicomm.bcast(nsamp, root=0)
        props = mpicomm.bcast(props, root=0)
        self._have_azel = mpicomm.bcast(self._have_azel, root=0)
        self._units = mpicomm.bcast(self._units, root=0)
        self._files = mpicomm.bcast(self._files, root=0)
        self._file_sample_offs = mpicomm.bcast(self._file_sample_offs, root=0)
        self._file_frame_offs = mpicomm.bcast(self._file_frame_offs, root=0)
        self._frame_sizes = mpicomm.bcast(self._frame_sizes, root=0)
        self._frame_sample_offs = mpicomm.bcast(self._frame_sample_offs, root=0)

        # We need to assign a unique integer index to each detector.  This
        # is used when seeding the streamed RNG in order to simulate
        # timestreams.  For simplicity, and assuming that detector names
        # are not too long, we can convert the detector name to bytes and
        # then to an integer.

        self._detindx = {}
        for det in dets:
            bdet = det.encode("utf-8")
            ind = None
            try:
                ind = int.from_bytes(bdet, byteorder="little")
            except:
                raise RuntimeError("Cannot convert detector name {} to a "
                    "unique integer- maybe it is too long?".format(det))
            self._detindx[det] = ind

        # call base class constructor to distribute data
        super().__init__(mpicomm, list(sorted(dets.keys())), nsamp,
            detindx=self._detindx, detranks=detranks, detbreaks=detbreaks,
            sampsizes=self._frame_sizes, meta=props)

        # Now that the data distribution is set, read frames into cache
        # if needed.

        self._done_cache_init = False

        if not self._createmode:
            # We are reading existing data into cache.  Create our local pieces
            # of the data.
            self._cache_init()
            self._import_frames()

        return


    def _cache_init(self):
        if not self._done_cache_init:
            # Timestamps
            name = "{}-{}".format(CACHE_PREF, STR_TIME)
            self.cache.create(name, np.float64, (self.local_samples[1],))

            # Boresight quaternions
            name = "{}-{}".format(CACHE_PREF, STR_BORE)
            self.cache.create(name, np.float64, (self.local_samples[1], 4))
            if self._have_azel:
                name = "{}-{}".format(CACHE_PREF, STR_BOREAZEL)
                self.cache.create(name, np.float64, (self.local_samples[1], 4))

            # Common flags
            name = "{}-{}".format(CACHE_PREF, STR_COMMON)
            self.cache.create(name, np.uint8, (self.local_samples[1],))

            # Telescope position and velocity
            name = "{}-{}".format(CACHE_PREF, STR_POS)
            self.cache.create(name, np.float64, (self.local_samples[1], 3))
            name = "{}-{}".format(CACHE_PREF, STR_VEL)
            self.cache.create(name, np.float64, (self.local_samples[1], 3))

            # Detector data and flags
            for det in self.local_dets:
                name = "{}-{}_{}".format(CACHE_PREF, STR_DET, det)
                self.cache.create(name, np.float64, (self.local_samples[1],))
                name = "{}-{}_{}".format(CACHE_PREF, STR_FLAG, det)
                self.cache.create(name, np.uint8, (self.local_samples[1],))

            self._done_cache_init = True

        return


    def _frame_indices(self, frame):
        localfirst = self.local_samples[0]
        locallast = self.local_samples[0] + self.local_samples[1] - 1
        frsize = self._frame_sizes[frame]
        frfirst = self._frame_sample_offs[frame]
        memoff = None
        nmem = None
        froff = None
        nfr = None
        # Does this frame overlap with any of our data?
        if (frfirst <= locallast) and (frfirst + frsize >= localfirst):
            # compute offsets into our local data and the frame
            memoff = frfirst - localfirst
            if memoff < 0:
                memoff = 0
            nmem = frsize
            if memoff + nmem > self.local_samples[1]:
                nmem = self.local_samples[1] - memoff
            froff = localfirst - frfirst
            if froff < 0:
                froff = 0
            nfr = self.local_samples[1]
            if froff + nfr > frsize:
                nfr = frsize - froff
            #print("proc {}:  frame indices {}:{} -> frame local {}:{} == cache local {}:{}".format(self.mpicomm.rank, frfirst, frfirst+frsize-1, froff, froff+nfr-1, memoff, memoff+nmem-1), flush=True)
        return (memoff, nmem, froff, nfr)


    def _import_frames(self):
        for ifile, (ffile, foff) in enumerate(zip(self._files,
            self._file_frame_offs)):
            nf = None
            if ifile == len(self._files) - 1:
                # we are at the last file
                nf = len(self._frame_sizes) - foff
            else:
                # get number of frames in this file
                nf = self._file_frame_offs[ifile+1] - foff

            gfile = [ None for x in range(nf) ]
            if self.mpicomm.rank == 0:
                gfile = c3g.G3File(ffile)

            obs = True
            for reloff, fdata in enumerate(gfile):
                if obs:
                    obs = False
                    continue
                fdata = self.mpicomm.bcast(fdata, root=0)
                memoff, nmem, froff, nfr = self._frame_indices(foff + reloff -1)
                if memoff is not None:
                    # Timestamps
                    name = "{}-{}".format(CACHE_PREF, STR_TIME)
                    ref = self.cache.reference(name)
                    # convert from nanoseconds to floating point seconds
                    ntime = np.array(fdata[STR_TIME][froff:froff+nfr])
                    ref[memoff:memoff+nmem] = 1.0e-9 * ntime.astype(np.float64)
                    del ref
                    del ntime

                    # Boresight quaternions
                    name = "{}-{}".format(CACHE_PREF, STR_BORE)
                    ref = self.cache.reference(name)
                    ref[memoff:memoff+nmem,:] = \
                        np.array(fdata[STR_BORE][4*froff:4*(froff+nfr)]).reshape((-1, 4))
                    del ref
                    if self._have_azel:
                        name = "{}-{}".format(CACHE_PREF, STR_BOREAZEL)
                        ref = self.cache.reference(name)
                        ref[memoff:memoff+nmem,:] = \
                            np.array(fdata[STR_BOREAZEL][4*froff:4*(froff+nfr)]).reshape((-1, 4))
                        del ref

                    # Common flags
                    name = "{}-{}".format(CACHE_PREF, STR_COMMON)
                    ref = self.cache.reference(name)
                    ref[memoff:memoff+nmem] = \
                        np.array(fdata[STR_COMMON][froff:froff+nfr], dtype=np.uint8)
                    del ref

                    # Telescope position / velocity

                    name = "{}-{}".format(CACHE_PREF, STR_POS)
                    ref = self.cache.reference(name)
                    ref[memoff:memoff+nmem,:] = \
                        np.array(fdata[STR_POS][3*froff:3*(froff+nfr)]).reshape((-1, 3))
                    del ref

                    name = "{}-{}".format(CACHE_PREF, STR_VEL)
                    ref = self.cache.reference(name)
                    ref[memoff:memoff+nmem,:] = \
                        np.array(fdata[STR_VEL][3*froff:3*(froff+nfr)]).reshape((-1, 3))
                    del ref

                    # Detector data and flags
                    detmap = fdata[STR_DET]
                    flagmap = fdata[STR_FLAG]
                    for d in self.local_dets:
                        name = "{}-{}_{}".format(CACHE_PREF, STR_DET, d)
                        ref = self.cache.reference(name)
                        ref[memoff:memoff+nmem] = detmap[d][froff:froff+nfr]
                        del ref
                        name = "{}-{}_{}".format(CACHE_PREF, STR_FLAG, d)
                        ref = self.cache.reference(name)
                        ref[memoff:memoff+nmem] = \
                            np.array(flagmap[d][froff:froff+nfr],
                            dtype=np.uint8)
                        del ref
        return


    def _export_frames(self, oldtod):
        # The process grid
        detranks, sampranks = self.grid_size
        rankdet, ranksamp = self.grid_ranks

        for ifile, (ffile, foff) in enumerate(zip(self._files,
            self._file_frame_offs)):
            nf = None
            nsamp = None
            if ifile == len(self._files) - 1:
                # we are at the last file
                nf = len(self._frame_sizes) - foff
                nsamp = self.total_samples - self._file_sample_offs[-1]
            else:
                # get number of frames / samples in this file
                nf = self._file_frame_offs[ifile+1] - foff
                nsamp = self._file_sample_offs[ifile + 1] - \
                    self._file_sample_offs[ifile]

            writer = None
            if self.mpicomm.rank == 0:
                writer = c3g.G3Writer(ffile)
                props = self.meta()
                props["units"] = self._units
                props["have_azel"] = self._have_azel
                #print("Writing props:")
                #print(props, flush=True)
                write_spt3g_obs(writer, props, self._detquats,
                    self.total_samples)

            for fr in range(nf):
                fdata = None
                if self.mpicomm.rank == 0:
                    fdata = c3g.G3Frame(c3g.G3FrameType.Scan)

                memoff, nmem, froff, nfr = self._frame_indices(foff + fr)

                # Gather timestamps

                data = None
                alldata = None
                if rankdet == 0:
                    # Only the first row of the process grid does this.
                    if memoff is not None:
                        data = oldtod.read_times(local_start=memoff, n=nmem)
                alldata = self.mpicomm.gather(data, root=0)

                if self.mpicomm.rank == 0:
                    alldata = np.concatenate([ x for x in alldata if x is not None ])
                    # convert to integer nanoseconds
                    alldata *= 1.0e9
                    idata = alldata.astype(np.int64)
                    # FIXME: there *must* be a better way to instantiate a vector of
                    # G3 timestamps...
                    g3times = list()
                    for t in range(len(idata)):
                        g3times.append(c3g.G3Time(idata[t]))
                    fdata[STR_TIME] = c3g.G3VectorTime(g3times)
                    del g3times
                    del idata
                del data
                del alldata

                # Gather boresight data

                data = None
                alldata = None
                if rankdet == 0:
                    # Only the first row of the process grid does this.
                    if memoff is not None:
                        data = oldtod.read_boresight(local_start=memoff, n=nmem)
                alldata = self.mpicomm.gather(data, root=0)
                if self.mpicomm.rank == 0:
                    alldata = np.concatenate([ x for x in alldata if x is not None ])
                    fdata[STR_BORE] = c3g.G3VectorDouble(alldata.flatten())
                del data
                del alldata

                if self._have_azel:
                    data = None
                    alldata = None
                    if rankdet == 0:
                        # Only the first row of the process grid does this.
                        if memoff is not None:
                            data = oldtod.read_boresight_azel(local_start=memoff, n=nmem)
                    alldata = self.mpicomm.gather(data, root=0)
                    if self.mpicomm.rank == 0:
                        alldata = np.concatenate([ x for x in alldata if x is not None ])
                        fdata[STR_BOREAZEL] = c3g.G3VectorDouble(alldata.flatten())
                    del data
                    del alldata

                # Gather common flags

                data = None
                alldata = None
                if rankdet == 0:
                    # Only the first row of the process grid does this.
                    if memoff is not None:
                        if self._export_cachecomm is not None:
                            data = oldtod.cache.reference(self._export_cachecomm)[memoff:memoff+nmem]
                        else:
                            data = oldtod.read_common_flags(local_start=memoff, n=nmem)
                alldata = self.mpicomm.gather(data, root=0)
                if self.mpicomm.rank == 0:
                    alldata = np.concatenate([ x for x in alldata if x is not None ])
                    # The bindings of G3Vector seem to only work with lists...
                    fdata[STR_COMMON] = \
                        c3g.G3VectorInt(alldata.astype(np.int32).tolist())
                del data
                del alldata

                # Telescope position and velocity

                data = None
                alldata = None
                if rankdet == 0:
                    # Only the first row of the process grid does this.
                    if memoff is not None:
                        data = oldtod.read_position(local_start=memoff, n=nmem)
                alldata = self.mpicomm.gather(data, root=0)
                if self.mpicomm.rank == 0:
                    alldata = np.concatenate([ x for x in alldata if x is not None ])
                    fdata[STR_POS] = c3g.G3VectorDouble(alldata.flatten())
                del data
                del alldata

                data = None
                alldata = None
                if rankdet == 0:
                    # Only the first row of the process grid does this.
                    if memoff is not None:
                        data = oldtod.read_velocity(local_start=memoff, n=nmem)
                alldata = self.mpicomm.gather(data, root=0)
                if self.mpicomm.rank == 0:
                    alldata = np.concatenate([ x for x in alldata if x is not None ])
                    fdata[STR_VEL] = c3g.G3VectorDouble(alldata.flatten())
                del data
                del alldata

                # Now go through all detectors and add the data and flags to the frame.

                detmap = c3g.G3TimestreamMap()
                flagmap = c3g.G3MapVectorInt()

                for det in self.detectors:
                    data = None
                    alldata = None
                    if (det in self.local_dets) and memoff is not None:
                        if self._export_cachename is not None:
                            cname = "{}_{}".format(self._export_cachename, det)
                            data = oldtod.cache.reference(cname)[memoff:memoff+nmem]
                        else:
                            data = oldtod.read(detector=det, local_start=memoff, n=nmem)
                    alldata = self.mpicomm.gather(data, root=0)
                    if self.mpicomm.rank == 0:
                        alldata = [ x for x in alldata if x is not None ]
                        if self._units is None:
                            # We do this, since we can't use G3TimestreamUnits.None
                            # in python ("None" is interpreted as python None).
                            detmap[det] = c3g.G3Timestream(np.concatenate(alldata))
                        else:
                            detmap[det] = \
                                c3g.G3Timestream(np.concatenate(alldata), self._units)
                    del data
                    del alldata
                    data = None
                    alldata = None
                    if (det in self.local_dets) and memoff is not None:
                        if self._export_cacheflag is not None:
                            cname = "{}_{}".format(self._export_cacheflag, det)
                            data = oldtod.cache.reference(cname)[memoff:memoff+nmem]
                        else:
                            data = oldtod.read_flags(detector=det, local_start=memoff, n=nmem)
                    alldata = self.mpicomm.gather(data, root=0)
                    if self.mpicomm.rank == 0:
                        alldata = np.concatenate([ x for x in alldata if x is not None ])
                        # The bindings of G3Vector seem to only work with lists...
                        # Also there is no vectormap for unsigned char, so we have
                        # to use int...
                        flagmap[det] = \
                            c3g.G3VectorInt(alldata.astype(np.int32).tolist())
                    del data
                    del alldata

                if self.mpicomm.rank == 0:
                    fdata[STR_DET] = detmap
                    fdata[STR_FLAG] = flagmap
                    writer(fdata)

        return


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


    def _get_boresight(self, start, n):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_BORE)
        ref = self.cache.reference(name)[start:start+n,:]
        return ref


    def _put_boresight(self, start, data):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_BORE)
        ref = self.cache.reference(name)
        ref[start:(start+data.shape[0]),:] = data
        del ref
        return


    def _get_boresight_azel(self, start, n):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_BOREAZEL)
        ref = self.cache.reference(name)[start:start+n,:]
        return ref


    def _put_boresight_azel(self, start, data):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_BOREAZEL)
        ref = self.cache.reference(name)
        ref[start:(start+data.shape[0]),:] = data
        del ref
        return


    def _get(self, detector, start, n):
        self._cache_init()
        name = "{}-{}_{}".format(CACHE_PREF, STR_DET, detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put(self, detector, start, data):
        self._cache_init()
        name = "{}-{}_{}".format(CACHE_PREF, STR_DET, detector)
        ref = self.cache.reference(name)
        ref[start:(start+data.shape[0])] = data
        del ref
        return


    def _get_flags(self, detector, start, n):
        self._cache_init()
        name = "{}-{}_{}".format(CACHE_PREF, STR_FLAG, detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put_flags(self, detector, start, flags):
        self._cache_init()
        name = "{}-{}_{}".format(CACHE_PREF, STR_FLAG, detector)
        ref = self.cache.reference(name)
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return


    def _get_common_flags(self, start, n):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_COMMON)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put_common_flags(self, start, flags):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_COMMON)
        ref = self.cache.reference(name)
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return


    def _get_times(self, start, n):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_TIME)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put_times(self, start, stamps):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_TIME)
        ref = self.cache.reference(name)
        ref[start:(start+stamps.shape[0])] = stamps
        del ref
        return


    def _get_pntg(self, detector, start, n):
        self._cache_init()
        # Get boresight pointing (from disk or cache)
        bore = self._get_boresight(start, n)
        # Apply detector quaternion and return
        return qa.mult(bore, self._detquats[detector])


    def _put_pntg(self, detector, start, data):
        raise RuntimeError("TOD3G computes detector pointing on the fly."
            " Use the write_boresight() method instead.")
        return


    def _get_position(self, start, n):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_POS)
        ref = self.cache.reference(name)[start:start+n,:]
        return ref


    def _put_position(self, start, pos):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_POS)
        ref = self.cache.reference(name)
        ref[start:(start+pos.shape[0]),:] = pos
        del ref
        return


    def _get_velocity(self, start, n):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_VEL)
        ref = self.cache.reference(name)[start:start+n,:]
        return ref


    def _put_velocity(self, start, vel):
        self._cache_init()
        name = "{}-{}".format(CACHE_PREF, STR_VEL)
        ref = self.cache.reference(name)
        ref[start:(start+vel.shape[0]),:] = vel
        del ref
        return


    def export(self, oldtod):
        # For each frame, gather the data from relevant processes and write
        # out.  In order for this to work, the path and prefix must have been
        # set at construction time.
        self._export_frames(oldtod)
        return


def load_spt3g(comm, detranks, path, prefix, todclass, **kwargs):
    """
    Loads an existing TOAST dataset in SPT3G format.

    This takes a 2-level TOAST communicator and opens existing spt3g
    directories of frame files.  For each observation (directory of frame
    files), the specified TOD class is instantiated.

    Args:
        comm (toast.Comm): the toast Comm class for distributing the data.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        path (str):  The top-level directory that contains subdirectories (one
            per observation).
        prefix (str):  The frame file prefix.
        todclass (TOD): an SPT3G-compatible TOD class, which must have a
            constructor that takes ......  All additional arguments should be keyword args.
        kwargs: All additional arguments to this function are passed to the
            TOD constructor for each observation.

    Returns (toast.Data):
        The distributed data object.
    """
    if not available:
        raise RuntimeError("spt3g is not available")
        return None
    # the global communicator
    cworld = comm.comm_world
    # the communicator within the group
    cgroup = comm.comm_group
    # the communicator with all processes with
    # the same rank within their group
    crank = comm.comm_rank

    # One process gets the list of observation directories
    obslist = list()
    obsweight = dict()

    if cworld.rank == 0:
        for root, dirs, files in os.walk(path, topdown=True):
            for d in dirs:
                # FIXME:  Add some check here to make sure that this is a
                # directory of frame files.
                obslist.append(d)
                # Read the observation frame to find the number of samples.
                fr = os.path.join(path, d, "{}_{:08d}.g3".format(prefix, 0))
                obs, props, dets, nsamp = read_spt3g_obs(fr)
                obsweight[d] = nsamp * len(dets.keys())
            break
        obslist = sorted(obslist)

    obslist = cworld.bcast(obslist, root=0)
    obsweight = cworld.bcast(obsweight, root=0)

    # Distribute observations based on number of samples
    dweight = [ obsweight[x] for x in obslist ]
    distobs = distribute_discrete(dweight, comm.ngroups)

    # Distributed data

    data = Data(comm)

    # Now every group adds its observations to the list

    firstobs = distobs[comm.group][0]
    nobs = distobs[comm.group][1]
    for ob in range(firstobs, firstobs+nobs):
        opath = os.path.join(path, obslist[ob])
        fr = os.path.join(opath, "{}_{:08d}.g3".format(prefix, 0))
        obs, props, dets, nsamp = read_spt3g_obs(fr)
        obs["tod"] = todclass(cgroup, detranks, path=opath, prefix=prefix,
            **kwargs)
        data.obs.append(obs)

    return data


class Op3GExport(Operator):
    """Operator which writes data to a directory tree of frame files.

    The top level directory will contain one subdirectory per observation.
    Each observation directory will contain one frame file per sample chunk.

    Args:
        path (str): The output top-level directory.
        prefix (str): The frame file prefix.
        todclass (TOD): a SPT3G-compatible TOD class, which must have a
            constructor with the proper required arguments and which takes
            additional arguments as keywords.
        use_todchunks (bool): if True, use the chunk of the original TOD for
            data distribution.
        use_intervals (bool): if True, use the intervals in the observation
            dictionary for data distribution.
        kwargs: All additional arguments to this constructor are passed to the
            TOD constructor for each observation.

    """
    def __init__(self, path, prefix, todclass, use_todchunks=False,
        use_intervals=False, **kwargs):

        if not available:
            raise RuntimeError("spt3g is not available")

        self._path = os.path.abspath(path.rstrip("/"))
        self._prefix = prefix
        self._todclass = todclass
        self._kwargs = kwargs
        if use_todchunks and use_intervals:
            raise RuntimeError("cannot use both TOD chunks and Intervals")
        self._usechunks = use_todchunks
        self._useintervals = use_intervals
        # We call the parent class constructor
        super().__init__()


    def exec(self, data):
        """Export data to a directory tree of SPT3G frames.

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

        # One process checks the path
        if cworld.rank == 0:
            if not os.path.isdir(self._path):
                os.makedirs(self._path)
        cworld.barrier()

        for obs in data.obs:
            # Observation information.  Anything here that is a simple data
            # type will get written to the observation frame.
            props = dict()
            for k, v in obs.items():
                if isinstance(v, (int, str, bool, float)):
                    props["obs_{}".format(k)] = v

            # Every observation must have a name...
            obsname = obs["name"]

            # The existing TOD
            oldtod = obs["tod"]
            nsamp = oldtod.total_samples
            dets = oldtod.detoffset()

            # Get any other metadata from the old TOD
            props.update(oldtod.meta())

            # First process in the group makes the output directory
            obsdir = os.path.join(self._path, obsname)
            if cgroup.rank == 0:
                if not os.path.isdir(obsdir):
                    os.makedirs(obsdir)
            cgroup.barrier()

            olddetranks, oldsampranks = oldtod.grid_size

            # Do we have ground pointing?
            azel = None
            if cgroup.rank == 0:
                try:
                    test = oldtod.read_boresight_azel(local_start=0, n=1)
                    azel = True
                    del test
                except:
                    azel = False
            azel = cgroup.bcast(azel, root=0)

            # Determine data distribution chunks
            framesizes = None
            if self._usechunks:
                framesizes = oldtod.total_chunks
            elif self._useintervals:
                if "intervals" not in obs:
                    raise RuntimeError("Observation does not contain intervals"
                        ", cannot distribute using them")
                framesizes = intervals_to_chunklist(obs["intervals"], nsamp)

            # The new TOD to handle exporting
            tod = self._todclass(oldtod.mpicomm, olddetranks, path=obsdir,
                prefix=self._prefix, detectors=dets,
                samples=nsamp, framesizes=framesizes,
                azel=azel, meta=props, **self._kwargs)

            tod.export(oldtod)

        return
