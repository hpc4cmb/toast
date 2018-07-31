# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import sys
import os
import re

import traceback

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
    from . import spt3g_utils as s3utils
except:
    available = False


# Module-level constants

STR_TIME = "timestamps"
STR_FLAG = "flag"
STR_COMMON = "common"
STR_QUAT = "quaternion"
STR_BORE = "boresight"
STR_BOREAZEL = "boresight_azel"
STR_POS = "position"
STR_VEL = "velocity"
STR_DET = "det"

# Used to determine file breaks when writing new data.  This
# is the desired approximate file size in bytes.
TARGET_FRAMEFILE_SIZE = 500000000


def read_spt3g_obs(file):
    reader = c3g.G3Reader(file, 1)
    f = list(reader(None))
    obframe = f[0]
    obspat = re.compile("^obs_(.*)")
    detpat = re.compile("{}_(.*)".format(STR_QUAT))

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
                obs[obsmat.group(1)] = s3utils.from_g3_type(obframe[k])
            elif detmat is not None:
                dets[detmat.group(1)] = np.array(obframe[k], dtype=np.float64)
            else:
                props[k] = s3utils.from_g3_type(obframe[k])
    return obs, props, dets, nsamp


def write_spt3g_obs(writer, props, dets, nsamp):
    f = c3g.G3Frame(c3g.G3FrameType.Observation)
    for k, v in props.items():
        f[k] = s3utils.to_g3_type(v)
    f["samples"] = c3g.G3Int(nsamp)
    for k, v in dets.items():
        f["{}_{}".format(STR_QUAT, k)] = c3g.G3VectorDouble(v)
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

    This class serves as *an example* for how to create a TOD class that
    interfaces with spt3g data files.  Experiments will surely need to
    implement their own classes that use some of the same utilities and also
    use their own "schema" for the frames.  This class just reads / writes
    a minimal set of data required by a TOAST TOD.

    An instance of this class loads a directory of frame files into memory.
    This data may be manipulated in memory (using the write* methods), but
    the modifications will not be stored to disk until the export method is
    called.  If creating a new data set, additional parameters are used to
    set up the initial detector information.

    Observation properties and detector pointing offsets from the boresight are
    given as quaternions, and are stored in an observation frame at the
    start of each frame file.  The sequence of frame files are named with a
    prefix and a number which represents the first sample of that frame
    relative to the start of the observation.

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


    """
    def __init__(self, mpicomm, detranks, path=None, prefix=None,
        detectors=None, samples=None, framesizes=None, azel=False, meta=dict(),
        units=None, detbreaks=None):

        if not available:
            raise RuntimeError("spt3g is not available")

        self._path = path
        self._prefix = prefix

        # When constructing an instance of this class, we are either creating
        # a new data set in memory or reading from disk.  Make sure that all
        # the required inputs are given in either case.
        dets = None
        nsamp = None
        props = None
        self._have_azel = False
        self._units = None
        self._frame_sizes = None
        self._frame_sample_offs = None
        self._files = None
        self._file_sample_offs = None
        self._file_frame_offs = None

        self._createmode = False
        if (detectors is not None) and (samples is not None):
            self._createmode = True
        else:
            if (self._path is None) or (self._prefix is None):
                raise RuntimeError("If reading existing data, path and "
                                   "prefix are required.")

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
                    if frbytes > TARGET_FRAMEFILE_SIZE:
                        msg = "A single frame ({}) is larger than the target"\
                            " frame file size ({}).  Increase the target size."\
                            .format(frbytes, TARGET_FRAMEFILE_SIZE)
                        raise RuntimeError(msg)
                    if filebytes + frbytes > TARGET_FRAMEFILE_SIZE:
                        # Start a new file
                        self._file_sample_offs.append(fileoff)
                        self._file_frame_offs.append(fileframeoff)
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
            uid = None
            try:
                ind = int.from_bytes(bdet, byteorder="little")
                uid = int(ind & 0xFFFFFFFF)
            except:
                raise RuntimeError("Cannot convert detector name {} to a "
                    "unique integer- maybe it is too long?".format(det))
            self._detindx[det] = uid

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
            self.load_frames()

        return


    def _cache_init(self):
        if not self._done_cache_init:
            # Timestamps
            self.cache.create(STR_TIME, np.int64, (self.local_samples[1],))

            # Boresight quaternions
            self.cache.create(STR_BORE, np.float64, (self.local_samples[1], 4))
            if self._have_azel:
                self.cache.create(STR_BOREAZEL, np.float64,
                                  (self.local_samples[1], 4))

            # Common flags
            name = "{}_{}".format(STR_FLAG, STR_COMMON)
            self.cache.create(name, np.uint8, (self.local_samples[1],))

            # Telescope position and velocity
            self.cache.create(STR_POS, np.float64, (self.local_samples[1], 3))
            self.cache.create(STR_VEL, np.float64, (self.local_samples[1], 3))

            # Detector data and flags
            for det in self.local_dets:
                name = "{}_{}".format(STR_DET, det)
                self.cache.create(name, np.float64, (self.local_samples[1],))
                name = "{}_{}".format(STR_FLAG, det)
                self.cache.create(name, np.uint8, (self.local_samples[1],))

            self._done_cache_init = True

        return


    def load_frames(self):
        self._cache_init()
        for ifile, (ffile, foff) in enumerate(zip(self._files,
            self._file_frame_offs)):
            nframes = None
            if ifile == len(self._files) - 1:
                # we are at the last file
                nframes = len(self._frame_sizes) - foff
            else:
                # get number of frames in this file
                nframes = self._file_frame_offs[ifile+1] - foff

            #print("load {}:  proc {} file {} starts at frame {} and has {} frames".format(self._path, self.mpicomm.rank, ifile, foff, nframes), flush=True)

            # nframes includes only the scan frames.  We add one here to
            # get the total including the observation frame.
            gfile = [ None for x in range(nframes + 1) ]
            if self.mpicomm.rank == 0:
                gfile = c3g.G3File(ffile)

            obs = True
            for fileframe, fdata in enumerate(gfile):
                if obs:
                    # Skip the observation frame.
                    obs = False
                    continue
                scanframe = fileframe - 1
                frame = foff + scanframe
                frame_offset = self._frame_sample_offs[frame]
                frame_size = self._frame_sizes[frame]
                # if self.mpicomm.rank == 0:
                #     print("load {}:    proc {} working on frame {} (samples {} - {}) data = {}".format(self._path, self.mpicomm.rank, frame, frame_offset, frame_offset+frame_size, fdata), flush=True)

                s3utils.frame_to_cache(self, frame, frame_offset, frame_size,
                                       frame_data=fdata, detector_map=STR_DET,
                                       flag_map=STR_FLAG, common_prefix=None,
                                       det_prefix="{}_".format(STR_DET),
                                       flag_prefix="{}_".format(STR_FLAG))

                #print("load {}:    proc {} hit barrier for frame {}".format(self._path, self.mpicomm.rank, frame), flush=True)
                self.mpicomm.barrier()
                #if self.mpicomm.rank == 0:
                #print("load {}:    proc {} finished frame {}".format(self._path, self.mpicomm.rank, frame), flush=True)
            del gfile
            #print("load {}:    proc {} finished file {}".format(self._path, self.mpicomm.rank, ifile), flush=True)
        return


    def _export_frames(self, path=None, prefix=None, cache_name=None,
                       cache_common=None, cache_flag_name=None):
        """Export cache data to frames.

        This will either export the cache fields that correspond to the "real"
        data (those manipulated by the read / write methods) or will export
        alternate cache fields specified by the arguments.

        Args:
            path (str):  Override any path specified at construction time.
            prefix (str):  Override any prefix specified at construction time.
            cache_name (str):  When exporting data, the name of the cache
                object (<name>_<detector>) to use for the detector timestream.
                If None, use the TOD read* methods.
            cache_common (str):  When exporting data, the name of the cache
                object to use for common flags.  If None, use the TOD read*
                methods.
            cache_flag_name (str):  When exporting data, the name of the
                cache object (<name>_<detector>) to use for the detector
                flags.  If None, use the TOD read* methods.

        """
        self._cache_init()

        # Create the frame schema we are using when exporting data

        common_fields = [
            (STR_TIME, c3g.G3VectorTime, STR_TIME),
            (STR_BORE, c3g.G3VectorDouble, STR_BORE),
            (STR_BOREAZEL, c3g.G3VectorDouble, STR_BOREAZEL),
            (STR_POS, c3g.G3VectorDouble, STR_POS),
            (STR_VEL, c3g.G3VectorDouble, STR_VEL)
        ]
        if cache_common is None:
            cname = "{}_{}".format(STR_FLAG, STR_COMMON)
            common_fields.append( (cname, c3g.G3VectorUnsignedChar, cname) )
        else:
            common_fields.append( (cache_common, c3g.G3VectorUnsignedChar,
                                   cache_common) )

        det_fields = None
        if cache_name is None:
            det_fields = [ ("{}_{}".format(STR_DET, d), d) \
                           for d in self.detectors ]
        else:
            det_fields = [ ("{}_{}".format(cache_name, d), d) \
                           for d in self.detectors ]

        flag_fields = None
        if cache_flag_name is None:
            flag_fields = [ ("{}_{}".format(STR_FLAG, d), d) \
                           for d in self.detectors ]
        else:
            flag_fields = [ ("{}_{}".format(cache_flag_name, d), d) \
                           for d in self.detectors ]

        ex_path = self._path
        if path is not None:
            ex_path = path

        ex_prefix = self._prefix
        if prefix is not None:
            ex_prefix = prefix

        if (ex_path is None) or (ex_prefix is None):
            raise RuntimeError("You must specify the TOD path and prefix, "
                               "either at construction or export")

        ex_files = [ os.path.join(ex_path,
                    "{}_{:08d}.g3".format(ex_prefix, x)) \
                    for x in self._file_sample_offs ]

        for ifile, (ffile, foff) in enumerate(zip(ex_files,
            self._file_frame_offs)):
            nframes = None
            #print("  ifile = {}, ffile = {}, foff = {}".format(ifile, ffile, foff), flush=True)
            if ifile == len(ex_files) - 1:
                # we are at the last file
                nframes = len(self._frame_sizes) - foff
            else:
                # get number of frames in this file
                nframes = self._file_frame_offs[ifile+1] - foff

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

            # Collect data for all frames in the file in one go.

            frm_offsets = [ self._frame_sample_offs[foff+f] \
                           for f in range(nframes) ]
            frm_sizes = [ self._frame_sizes[foff+f] \
                           for f in range(nframes) ]

            if self.mpicomm.rank == 0:
                print("  {} file {}".format(self._path, ifile), flush=True)
                print("    start frame = {}, nframes = {}".format(foff, nframes), flush=True)
                print("    frame offs = ",frm_offsets, flush=True)
                print("    frame sizes = ",frm_sizes, flush=True)

            fdata = s3utils.cache_to_frames(self, foff, nframes, frm_offsets,
                                           frm_sizes,
                                           common=common_fields,
                                           detector_fields=det_fields,
                                           flag_fields=flag_fields,
                                           detector_map=STR_DET,
                                           flag_map=STR_FLAG,
                                           units=self._units)

            if self.mpicomm.rank == 0:
                for fdt in fdata:
                    writer(fdt)
                del writer
            del fdata

        return


    def detoffset(self):
        return dict(self._detquats)


    def _get_boresight(self, start, n):
        self._cache_init()
        ref = self.cache.reference(STR_BORE)[start:start+n,:]
        return ref


    def _put_boresight(self, start, data):
        self._cache_init()
        ref = self.cache.reference(STR_BORE)
        ref[start:(start+data.shape[0]),:] = data
        del ref
        return


    def _get_boresight_azel(self, start, n):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        self._cache_init()
        ref = self.cache.reference(STR_BOREAZEL)[start:start+n,:]
        return ref


    def _put_boresight_azel(self, start, data):
        if not self._have_azel:
            raise RuntimeError("No Az/El pointing for this TOD")
        self._cache_init()
        ref = self.cache.reference(STR_BOREAZEL)
        ref[start:(start+data.shape[0]),:] = data
        del ref
        return


    def _get(self, detector, start, n):
        self._cache_init()
        name = "{}_{}".format(STR_DET, detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put(self, detector, start, data):
        self._cache_init()
        name = "{}_{}".format(STR_DET, detector)
        ref = self.cache.reference(name)
        ref[start:(start+data.shape[0])] = data
        del ref
        return


    def _get_flags(self, detector, start, n):
        self._cache_init()
        name = "{}_{}".format(STR_FLAG, detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put_flags(self, detector, start, flags):
        self._cache_init()
        name = "{}_{}".format(STR_FLAG, detector)
        ref = self.cache.reference(name)
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return


    def _get_common_flags(self, start, n):
        self._cache_init()
        name = "{}_{}".format(STR_FLAG, STR_COMMON)
        ref = self.cache.reference(name)[start:start+n]
        return ref


    def _put_common_flags(self, start, flags):
        self._cache_init()
        name = "{}_{}".format(STR_FLAG, STR_COMMON)
        ref = self.cache.reference(name)
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return


    def _get_times(self, start, n):
        self._cache_init()
        ref = self.cache.reference(STR_TIME)[start:start+n]
        tm = 1.0e-9 * ref.astype(np.float64)
        del ref
        return tm


    def _put_times(self, start, stamps):
        self._cache_init()
        ref = self.cache.reference(STR_TIME)
        ref[start:(start+stamps.shape[0])] = np.array(1.0e9 * stamps,
                                                      dtype=np.int64)
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
        ref = self.cache.reference(STR_POS)[start:start+n,:]
        return ref


    def _put_position(self, start, pos):
        self._cache_init()
        ref = self.cache.reference(STR_POS)
        ref[start:(start+pos.shape[0]),:] = pos
        del ref
        return


    def _get_velocity(self, start, n):
        self._cache_init()
        ref = self.cache.reference(STR_VEL)[start:start+n,:]
        return ref


    def _put_velocity(self, start, vel):
        self._cache_init()
        ref = self.cache.reference(STR_VEL)
        ref[start:(start+vel.shape[0]),:] = vel
        del ref
        return


    def export(self, **kwargs):
        self._export_frames(**kwargs)
        return


def obsweight_spt3g(framefile):
    """Returns a weight for a framefile.

    This reads the special observation frame at the start of the file and
    returns the product of the number of detectors and the number of
    samples.

    Args:
        framefile (str):  The frame file.  The first frame should be an
            observation frame.

    Returns:
        (float): the weight.

    """
    obs, props, dets, nsamp = read_spt3g_obs(framefile)
    return nsamp * len(dets.keys())


def load_spt3g(comm, detranks, path, prefix, obsweight, todclass, **kwargs):
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
        obsweight (function):  A function that returns the relative weight of
            an observation, given the first frame file of that observation.
            This will only be called on the root process.
        todclass (TOD): an SPT3G-compatible TOD class, which must have a
            constructor that takes the communicator, detranks, path and
            prefix.  All additional arguments should be keyword args.
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
    weight = dict()

    if cworld.rank == 0:
        for root, dirs, files in os.walk(path, topdown=True):
            for d in dirs:
                # FIXME:  Add some check here to make sure that this is a
                # directory of frame files.
                obslist.append(d)
                # Read the observation frame to find the number of samples.
                fr = os.path.join(path, d, "{}_{:08d}.g3".format(prefix, 0))
                weight[d] = obsweight(fr)
            break
        obslist = sorted(obslist)

    obslist = cworld.bcast(obslist, root=0)
    weight = cworld.bcast(weight, root=0)

    # Distribute observations based on number of samples
    dweight = [ weight[x] for x in obslist ]
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
        # if cgroup.rank == 0:
        #     print("group {} creating TOD {}".format(comm.group, opath), flush=True)
        try:
            obs["tod"] = todclass(cgroup, detranks, path=opath, prefix=prefix, **kwargs)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            lines = [ "Proc {}: {}".format(MPI.COMM_WORLD.rank, x) for x in lines ]
            print("".join(lines), flush=True)
            MPI.COMM_WORLD.Abort()
        #print("proc {} hit TOD {} barrier".format(comm.comm_world.rank, opath), flush=True)
        comm.comm_group.barrier()
        data.obs.append(obs)

    return data


class Op3GExport(Operator):
    """Operator which writes data to a directory tree of frame files.

    The top level directory will contain one subdirectory per observation.
    Each observation directory will contain one frame file per sample chunk.

    Args:
        outdir (str): the top-level output directory.
        todclass (TOD): a SPT3G-compatible TOD class, which must have a
            constructor with the proper required arguments and which takes
            additional arguments as keywords.
        use_todchunks (bool): if True, use the chunks of the original TOD for
            data distribution.
        use_intervals (bool): if True, use the intervals in the observation
            dictionary for data distribution.
        ctor_opts (dict): dictionary of options to pass as kwargs to the
            TOD constructor.
        export_opts (dict): dictionary of options to pass to the export method
            of the TOD.
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
    def __init__(self, outdir, todclass, use_todchunks=False,
                 use_intervals=False, ctor_opts={}, export_opts={},
                 cache_common=None, cache_name=None, cache_flag_name=None):

        if not available:
            raise RuntimeError("spt3g is not available")

        self._outdir = outdir
        self._todclass = todclass
        self._ctor_opts = ctor_opts
        self._export_opts = export_opts
        self._cache_common = cache_common
        self._cache_name = cache_name
        self._cache_flag_name = cache_flag_name
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
            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir)
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

            if cworld.rank == 0:
                print("Start spt3g data copy for {}".format(obsname), flush=True)

            # The existing TOD
            oldtod = obs["tod"]
            nsamp = oldtod.total_samples
            dets = oldtod.detoffset()

            #print("rank {}, obs {}:  old tod has {} samps of {}".format(cworld.rank, obsname, nsamp, sorted(dets.keys())), flush=True)

            # Get any other metadata from the old TOD
            props.update(oldtod.meta())

            # First process in the group makes the output directory
            obsdir = os.path.join(self._outdir, obsname)
            if cgroup.rank == 0:
                if not os.path.isdir(obsdir):
                    os.makedirs(obsdir)
            #print("rank {}, obs {}:  hit group barrier".format(cworld.rank, obsname), flush=True)
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
            #print("rank {}, obs {}:  azel = {}".format(cworld.rank, obsname, azel), flush=True)

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
            tod = self._todclass(oldtod.mpicomm, olddetranks, detectors=dets,
                samples=nsamp, framesizes=framesizes,
                azel=azel, meta=props, **self._ctor_opts)

            # Copy data between TODs

            #print("rank {}, obs {}:  start data copy".format(cworld.rank, obsname), flush=True)

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
            for d in oldtod.local_dets:
                if self._cache_name is None:
                    tod.write(detector=d, data=oldtod.read(detector=d))
                else:
                    ref = oldtod.cache.reference("{}_{}"\
                                                 .format(self._cache_name, d))
                    tod.write(detector=d, data=ref)
                    del ref
                if self._cache_flag_name is None:
                    tod.write_flags(detector=d,
                                    flags=oldtod.read_flags(detector=d))
                else:
                    ref = oldtod.cache.reference("{}_{}".format(\
                                                 self._cache_flag_name, d))
                    tod.write_flags(detector=d, flags=ref)
                    del ref

            #print("rank {}, obs {}:  end data copy".format(cworld.rank, obsname), flush=True)

            if cgroup.rank == 0:
                print("Start spt3g data export for {}".format(obsname), flush=True)

            # Export data from cache.
            tod.export(path=obsdir, **self._export_opts)

            if cgroup.rank == 0:
                print("Done spt3g export for {}".format(obsname), flush=True)

        return
