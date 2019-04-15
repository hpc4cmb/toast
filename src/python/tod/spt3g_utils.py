# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import sys
import os
import re
import traceback
import pickle

import numpy as np

from .. import qarray as qa

available = True
try:
    from spt3g import core as c3g
    #from spt3g import coordinateutils as c3c
except:
    available = False


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


def g3_dtype(ar):
    if isinstance(ar, c3g.G3VectorUnsignedChar):
        return np.dtype(np.uint8)
    elif isinstance(ar, c3g.G3VectorInt):
        return np.dtype(np.int32)
    elif isinstance(ar, c3g.G3VectorTime):
        return np.dtype(np.int64)
    else:
        return np.dtype(np.float64)
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


def read_spt3g_framesizes(file):
    sizes = list()
    for frame in c3g.G3File(file):
        if frame.type == c3g.G3FrameType.Scan:
            field = frame.keys()[0]
            sizes.append(len(frame[field]))
    return sizes


def compute_frame_offsets(frame_sizes):
    """Returns a list of the sample offsets for each frame.

    Args:
        frame_sizes (array-like): the size in samples of each frame.

    Returns:
        (array): the starting sample index of each frame.

    """
    ret = list()
    sampoff = 0
    for fr in frame_sizes:
        ret.append(sampoff)
        sampoff += fr
    return np.array(ret, dtype=np.int64)


def compute_file_frames(bytes_per_sample, frame_sizes, file_size=500000000):
    """Divide frames into approximately equal files.

    Given a list of frames and the number of bytes per sample of TOD, compute
    the file boundaries in terms of samples and frames.

    Args:
        bytes_per_sample (int): the number of bytes across all frame vectors
            for a single TOD sample.
        frame_sizes (array-like): the size in samples of each frame.
        file_size (int): the target file size in bytes.

    Returns:
        (tuple): the sample offsets and frame offsets of each file and the
            sample offsets of each frame.

    """
    sample_offs = list()
    frame_offs = list()
    frame_sample_offs = list()

    filebytes = 0
    filesamps = 0
    fileframes = 0
    fileoff = 0
    fileframeoff = 0
    sampoff = 0

    for fr in frame_sizes:
        frbytes = fr * bytes_per_sample
        if frbytes > file_size:
            msg = "A single frame ({}) is larger than the target"\
                " frame file size ({}).  Increase the target"\
                "size.".format(frbytes, file_size)
            raise RuntimeError(msg)
        if filebytes + frbytes > file_size:
            # Start a new file
            sample_offs.append(fileoff)
            frame_offs.append(fileframeoff)
            fileoff += filesamps
            fileframeoff += fileframes
            filesamps = 0
            fileframes = 0
            filebytes = 0
        # Append frame to current file
        filesamps += fr
        fileframes += 1
        filebytes += frbytes
        frame_sample_offs.append(sampoff)
        sampoff += fr

    # process the last file
    sample_offs.append(fileoff)
    frame_offs.append(fileframeoff)

    return (np.array(sample_offs, dtype=np.int64),
            np.array(frame_offs, dtype=np.int64),
            np.array(frame_sample_offs, dtype=np.int64))


def local_frame_indices(local_first, nlocal, frame_offset, frame_size):
    """Compute frame overlap with local data.

    Args:
        local_first (int): the first sample of the local data.
        nlocal (int): the number of local samples.
        frame_offset (int): the first sample of the frame.
        frame_size (int): the number of samples in the frame.

    Returns:
        tuple: the local sample offset (zero == start of local data) of the
            overlap data, the frame sample offset (zero == start of frame) of
            the overlap data, the number of samples in the overlap.

    """
    # The first sample in our local data buffer
    local_off = None

    # The first sample in the frame data buffer
    froff = None

    # The number of samples of overlap
    nsamp = None

    # Does this frame overlap with any of our data?
    if (frame_offset < local_first + nlocal) and \
        (frame_offset + frame_size > local_first):
        # compute offsets into our local data and the frame
        if frame_offset >= local_first:
            # The frame starts in the middle of our local sample range.
            local_off = frame_offset - local_first
            froff = 0
        else:
            # Our local samples start in the middle of the frame.
            local_off = 0
            froff = local_first - frame_offset
        nsamp = frame_size
        if local_off + nsamp > nlocal:
            # The frame extends beyond our local samples
            nsamp = nlocal - local_off

    return (local_off, froff, nsamp)


def frame_to_cache(tod, frame, frame_offset, frame_size, frame_data=None,
                   detector_map="detectors", flag_map="flags",
                   common_prefix=None, det_prefix=None, flag_prefix=None):
    """Distribute a frame from the rank zero process.

    All vectors in the frame dictionary are copied into cache on all processes.
    If the detector_map and / or flag_map options are specified, those are
    frame keys which hold the timestream map and flag vector map.  When
    distributing those, only keys in the maps that contain the local detector
    names will be cached.

    Args:
        tod (toast.TOD): instance of a TOD class.
        frame (int): the frame index.
        frame_offset (int): the first sample of the frame.
        frame_size (int): the number of samples in the the frame.
        frame_data (G3Frame): the input frame (only on rank zero).
        detector_map (str): the name of the frame timestream map.
        flag_map (str): then name of the frame flag map.
        common_prefix (str): a string to prepend to each field name when
            storing in the cache.
        det_prefix (str): a string to prepend to each field name when
            storing in the cache.
        flag_prefix (str): a string to prepend to each field name when
            storing in the cache.

    """
    # First broadcast the frame data.
    #print("proc {}: start frame data bcast for {}".format(tod.mpicomm.rank, frame), flush=True)
    frame_data = tod.mpicomm.bcast(frame_data, root=0)
    #print("proc {}: finished frame data bcast for {}".format(tod.mpicomm.rank, frame), flush=True)

    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # Compute overlap of the frame with the local samples.
    cacheoff, froff, nfr = local_frame_indices(local_first, nlocal,
                                               frame_offset, frame_size)

    # Helper function to actually copy a slice of data into cache.
    def copy_slice(data, fld, cache_prefix):
        cache_fld = fld
        if cache_prefix is not None:
            cache_fld = "{}{}".format(cache_prefix, fld)
        # Check field type and data shape
        ftype = g3_dtype(data[fld])
        flen = len(data[fld])
        nnz = flen // frame_size
        if nnz * frame_size != flen:
            msg = "frame {}, field {} has length {} which is not "\
                "divisible by size {}".format(frame, fld, flen, frame_size)
            raise RuntimeError(msg)
        if not tod.cache.exists(cache_fld):
            # The field does not yet exist in cache, so create it.
            #print("proc {}:  create cache field {}, {}, ({}, {})".format(tod.mpicomm.rank, fld, ftype, tod.local_samples[1], nnz), flush=True)
            if nnz == 1:
                rf = tod.cache.create(cache_fld, ftype,
                                      (tod.local_samples[1],))
            else:
                rf = tod.cache.create(cache_fld, ftype,
                                      (tod.local_samples[1], nnz))
            del rf
        #print("proc {}: get cache ref for {}".format(tod.mpicomm.rank, cache_fld), flush=True)
        rf = tod.cache.reference(cache_fld)
        # Verify that the dimensions of the cache object are what we expect,
        # then copy the data.
        cache_samples = None
        cache_nnz = None
        if (len(rf.shape) > 1) and (rf.shape[1] > 0):
            # We have a packed 2D array
            cache_samples = rf.shape[0]
            cache_nnz = rf.shape[1]
        else:
            cache_nnz = 1
            cache_samples = len(rf)

        if cache_samples != tod.local_samples[1]:
            msg = "frame {}, field {}: cache has {} samples, which is"
            " different from local TOD size {}"\
            .format(frame, fld, cache_samples, tod.local_samples[1])
            raise RuntimeError(msg)

        if cache_nnz != nnz:
            msg = "frame {}, field {}: cache has nnz = {}, which is"\
                " different from frame nnz {}"\
                .format(frame, fld, cache_nnz, nnz)
            raise RuntimeError(msg)

        if cache_nnz > 1:
            slc = \
                np.array(data[fld][nnz*froff:nnz*(froff+nfr)],
                copy=False).reshape((-1, nnz))
            #print("proc {}:  copy_slice field {}[{}:{},:] = frame[{}:{},:]".format(tod.mpicomm.rank, fld, cacheoff, cacheoff+nfr, froff, froff+nfr), flush=True)
            rf[cacheoff:cacheoff+nfr,:] = slc
        else:
            slc = np.array(data[fld][froff:froff+nfr], copy=False)
            #print("proc {}:  copy_slice field {}[{}:{}] = frame[{}:{}]".format(tod.mpicomm.rank, fld, cacheoff, cacheoff+nfr, froff, froff+nfr), flush=True)
            rf[cacheoff:cacheoff+nfr] = slc
        del rf
        return

    if cacheoff is not None:
        #print("proc {} has overlap with frame {}:  {} {} {}".format(tod.mpicomm.rank, frame, cacheoff, froff, nfr), flush=True)

        # This process has some overlap with the frame.
        for field in frame_data.keys():
            # Skip over detector and flags
            if (detector_map is not None) and (field == detector_map):
                continue
            if (flag_map is not None) and (field == flag_map):
                continue
            #print("proc {} copy frame {}, field {}".format(tod.mpicomm.rank, frame, field), flush=True)
            copy_slice(frame_data, field, common_prefix)

        dpats = None
        if (detector_map is not None) or (flag_map is not None):
            # Build our list of regex matches
            dpats = [ re.compile(".*{}.*".format(d)) for d in tod.local_dets ]

        if detector_map is not None:
            # If the field name contains any of our local detectors,
            # then cache it.
            for field in frame_data[detector_map].keys():
                for dp in dpats:
                    if dp.match(field) is not None:
                        #print("proc {} copy frame {}, field {}".format(tod.mpicomm.rank, frame, field), flush=True)
                        copy_slice(frame_data[detector_map], field, det_prefix)
                        break
        if flag_map is not None:
            # If the field name contains any of our local detectors,
            # then cache it.
            for field in frame_data[flag_map].keys():
                for dp in dpats:
                    if dp.match(field) is not None:
                        copy_slice(frame_data[flag_map], field, flag_prefix)
                        break
    return


def cache_to_frames(tod, start_frame, n_frames, frame_offsets, frame_sizes,
                   common=None, detector_fields=None, flag_fields=None,
                   detector_map="detectors", flag_map="flags", units=None):
    """Gather all data from the distributed cache for a single frame.

    Args:
        tod (toast.TOD): instance of a TOD class.
        start_frame (int): the first frame index.
        n_frames (int): the number of frames.
        frame_offsets (list): list of the first samples of all frames.
        frame_sizes (list): list of the number of samples in each frame.
        common (tuple): (cache name, G3 type, frame name) of each common
            field.
        detector_fields (tuple): (cache name, frame name) of each detector
            field.
        flag_fields (tuple): (cache name, frame name) of each flag field.
        detector_map (str): the name of the frame timestream map.
        flag_map (str): then name of the frame flag map.
        units: G3 units of the detector data.

    """
    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # The process grid
    detranks, sampranks = tod.grid_size
    rankdet, ranksamp = tod.grid_ranks

    # Helper function:
    # For a given timestream, the gather is done across the
    # process row which contains the specific detector, or across
    # the first process row for common telescope data.
    def gather_field(prow, fld, indx, cacheoff, ncache):
        gproc = 0
        gdata = None

        # We are going to allreduce this later, so that every process
        # knows the dimensions of the field.
        allnnz = 0

        if rankdet == prow:
            #print("  proc {} doing gather of {}".format(tod.mpicomm.rank, fld), flush=True)
            # This process is in the process row that has this field,
            # participate in the gather operation.
            pdata = None
            # Find the data type and shape from the cache object
            mtype = None
            ref = tod.cache.reference(fld)
            nnz = 1
            if (len(ref.shape) > 1) and (ref.shape[1] > 0):
                nnz = ref.shape[1]
            if ref.dtype == np.dtype(np.float64):
                mtype = MPI.DOUBLE
            elif ref.dtype == np.dtype(np.int64):
                mtype = MPI.INT64_T
            elif ref.dtype == np.dtype(np.int32):
                mtype = MPI.INT32_T
            elif ref.dtype == np.dtype(np.uint8):
                mtype = MPI.UINT8_T
            else:
                msg = "Cannot gather cache field {} of type {}"\
                    .format(fld, ref.dtype)
                raise RuntimeError(msg)
            #print("field {}:  proc {} has nnz = {}".format(fld, tod.mpicomm.rank, nnz), flush=True)
            pz = 0
            if cacheoff is not None:
                pdata = ref.flatten()[nnz*cacheoff:nnz*(cacheoff+ncache)]
                pz = nnz * ncache
            else:
                pdata = np.zeros(0, dtype=ref.dtype)

            psizes = tod.grid_comm_row.gather(pz, root=0)
            disp = None
            totsize = None
            if ranksamp == 0:
                #print("Gathering field {} with type {}".format(fld, mtype), flush=True)
                # We are the process collecting the gathered data.
                gproc = tod.mpicomm.rank
                allnnz = nnz
                # Compute the displacements into the receive buffer.
                disp = [0]
                for ps in psizes[:-1]:
                    last = disp[-1]
                    disp.append(last + ps)
                totsize = np.sum(psizes)
                # allocate receive buffer
                gdata = np.zeros(totsize, dtype=ref.dtype)
                #print("Gatherv psizes = {}, disp = {}".format(psizes, disp), flush=True)

            #print("field {}:  proc {} start Gatherv".format(fld, tod.mpicomm.rank), flush=True)
            tod.grid_comm_row.Gatherv(pdata, [gdata, psizes, disp, mtype],
                                      root=0)
            #print("field {}:  proc {} finish Gatherv".format(fld, tod.mpicomm.rank), flush=True)

            del disp
            del psizes
            del pdata
            del ref

        # Now send this data to the root process of the whole communicator.
        # Only one process (the first one in process row "prow") has data
        # to send.

        # Create a unique message tag
        mtag = 10 * indx

        #print("  proc {} hit allreduce of gproc".format(tod.mpicomm.rank), flush=True)
        # All processes find out which one did the gather
        gproc = tod.mpicomm.allreduce(gproc, MPI.SUM)
        # All processes find out the field dimensions
        allnnz = tod.mpicomm.allreduce(allnnz, MPI.SUM)
        #print("  proc {} for field {}, gproc = {}".format(tod.mpicomm.rank, fld, gproc), flush=True)

        #print("field {}:  proc {}, gatherproc = {}, allnnz = {}".format(fld, tod.mpicomm.rank, gproc, allnnz), flush=True)

        rdata = None
        if gproc == 0:
            if gdata is not None:
                if allnnz == 1:
                    rdata = gdata
                else:
                    rdata = gdata.reshape((-1, allnnz))
        else:
            # Data not yet on rank 0
            if tod.mpicomm.rank == 0:
                # Receive data from the first process in this row
                #print("  proc {} for field {}, recv type".format(tod.mpicomm.rank, fld), flush=True)
                rtype = tod.mpicomm.recv(source=gproc, tag=(mtag+1))

                #print("  proc {} for field {}, recv size".format(tod.mpicomm.rank, fld), flush=True)
                rsize = tod.mpicomm.recv(source=gproc, tag=(mtag+2))

                #print("  proc {} for field {}, recv data".format(tod.mpicomm.rank, fld), flush=True)
                rdata = np.zeros(rsize, dtype=np.dtype(rtype))
                tod.mpicomm.Recv(rdata, source=gproc, tag=mtag)

                # Reshape if needed
                if allnnz > 1:
                    rdata = rdata.reshape((-1, allnnz))

            elif (tod.mpicomm.rank == gproc):
                # Send our data
                #print("  proc {} for field {}, send {} samples of {}".format(tod.mpicomm.rank, fld, len(gdata), gdata.dtype.char), flush=True)

                #print("  proc {} for field {}, send type with tag = {}".format(tod.mpicomm.rank, fld, mtag+1), flush=True)
                tod.mpicomm.send(gdata.dtype.char, dest=0, tag=(mtag+1))

                #print("  proc {} for field {}, send size with tag = {}".format(tod.mpicomm.rank, fld, mtag+2), flush=True)
                tod.mpicomm.send(len(gdata), dest=0, tag=(mtag+2))

                #print("  proc {} for field {}, send data with tag {}".format(tod.mpicomm.rank, fld, mtag), flush=True)
                tod.mpicomm.Send(gdata, 0, tag=mtag)
        return rdata

    # For efficiency, we are going to gather the data for all frames at once.
    # Then we will split those up when doing the write.

    # Frame offsets relative to the memory buffers we are gathering
    fdataoff = [0]
    for f in frame_sizes[:-1]:
        last = fdataoff[-1]
        fdataoff.append(last+f)

    # The list of frames- only on the root process.
    fdata = None
    if tod.mpicomm.rank == 0:
        fdata = [ c3g.G3Frame(c3g.G3FrameType.Scan) for f in range(n_frames) ]
    else:
        fdata = [ None for f in range(n_frames) ]

    # Compute the overlap of all frames with the local process.  We want to
    # to find the full sample range that this process overlaps the total set
    # of frames.

    cacheoff = None
    ncache = 0

    for f in range(n_frames):
        # Compute overlap of the frame with the local samples.
        fcacheoff, froff, nfr = local_frame_indices(local_first, nlocal,
                                                    frame_offsets[f],
                                                    frame_sizes[f])
        #print("proc {}:  frame {} has cache off {}, fr off {}, nfr {}".format(tod.mpicomm.rank, f, fcacheoff, froff, nfr), flush=True)
        if fcacheoff is not None:
            if cacheoff is None:
                cacheoff = fcacheoff
                ncache = nfr
            else:
                ncache += nfr
            #print("proc {}:    cache off now {}, ncache now {}".format(tod.mpicomm.rank, cacheoff, ncache), flush=True)

    # Now gather the full sample data one field at a time.  The root process
    # splits up the results into frames.

    # First gather common fields from the first row of the process grid.

    for findx, (cachefield, g3t, framefield) in enumerate(common):
        #print("proc {} entering gather_field(0, {}, {}, {}, {})".format(tod.mpicomm.rank, cachefield, findx, cacheoff, ncache), flush=True)
        data = gather_field(0, cachefield, findx, cacheoff, ncache)
        if tod.mpicomm.rank == 0:
            #print("Casting field {} to type {}".format(field, g3t), flush=True)
            if g3t == c3g.G3VectorTime:
                # Special case for time values stored as int64_t, but
                # wrapped in a class.
                for f in range(n_frames):
                    dataoff = fdataoff[f]
                    ndata = frame_sizes[f]
                    g3times = list()
                    for t in range(ndata):
                        g3times.append(c3g.G3Time(data[dataoff + t]))
                    fdata[f][framefield] = c3g.G3VectorTime(g3times)
                    del g3times
            else:
                # The bindings of G3Vector seem to only work with
                # lists.  This is probably horribly inefficient.
                for f in range(n_frames):
                    dataoff = fdataoff[f]
                    ndata = frame_sizes[f]
                    if len(data.shape) == 1:
                        fdata[f][framefield] = \
                            g3t(data[dataoff:dataoff+ndata].tolist())
                    else:
                        # We have a 2D quantity
                        fdata[f][framefield] = \
                            g3t(data[dataoff:dataoff+ndata,:].flatten().tolist())
        del data

    # Wait for everyone to catch up...
    tod.mpicomm.barrier()

    # For each detector field, processes which have the detector
    # in their local_dets should be in the same process row.
    # We do the gather over just this process row.

    if (detector_fields is not None) or (flag_fields is not None):
        dpats = { d : re.compile(".*{}.*".format(d)) for d in tod.local_dets }

        detmaps = None
        if detector_fields is not None:
            if tod.mpicomm.rank == 0:
                detmaps = [ c3g.G3TimestreamMap() for f in range(n_frames) ]

            for dindx, (cachefield, framefield) in enumerate(detector_fields):
                pc = -1
                for det, pat in dpats.items():
                    if pat.match(cachefield) is not None:
                        #print("proc {} has field {}".format(tod.mpicomm.rank, field), flush=True)
                        pc = rankdet
                        break
                # As a sanity check, verify that every process which
                # has this field is in the same process row.
                rowcheck = tod.mpicomm.gather(pc, root=0)
                prow = 0
                if tod.mpicomm.rank == 0:
                    rc = np.array([ x for x in rowcheck if (x >= 0) ],
                                  dtype=np.int32)
                    #print(field, rc, flush=True)
                    prow = np.max(rc)
                    if np.min(rc) != prow:
                        msg = "Processes with field {} are not in the "\
                            "same row\n".format(cachefield)
                        sys.stderr.write(msg)
                        tod.mpicomm.abort()

                # Every process finds out which process row is participating.
                prow = tod.mpicomm.bcast(prow, root=0)
                #print("proc {} got prow = {}".format(tod.mpicomm.rank, prow), flush=True)

                # Get the data on rank 0
                data = gather_field(prow, cachefield, dindx, cacheoff, ncache)

                if tod.mpicomm.rank == 0:
                    if units is None:
                        # We do this conditional, since we can't use
                        # G3TimestreamUnits.None in python ("None" is
                        # interpreted as python None).
                        for f in range(n_frames):
                            dataoff = fdataoff[f]
                            ndata = frame_sizes[f]
                            detmaps[f][framefield] = \
                                c3g.G3Timestream(data[dataoff:dataoff+ndata])
                    else:
                        for f in range(n_frames):
                            dataoff = fdataoff[f]
                            ndata = frame_sizes[f]
                            detmaps[f][framefield] = \
                                c3g.G3Timestream(data[dataoff:dataoff+ndata],
                                                 units)

            if tod.mpicomm.rank == 0:
                for f in range(n_frames):
                    fdata[f][detector_map] = detmaps[f]

        flagmaps = None
        if flag_fields is not None:
            if tod.mpicomm.rank == 0:
                flagmaps = [ c3g.G3MapVectorInt() for f in range(n_frames) ]
            for dindx, (cachefield, framefield) in enumerate(flag_fields):
                pc = -1
                for det, pat in dpats.items():
                    if pat.match(cachefield) is not None:
                        pc = rankdet
                        break
                # As a sanity check, verify that every process which
                # has this field is in the same process row.
                rowcheck = tod.mpicomm.gather(pc, root=0)
                prow = 0
                if tod.mpicomm.rank == 0:
                    rc = np.array([ x for x in rowcheck if (x >= 0) ],
                                  dtype=np.int32)
                    prow = np.max(rc)
                    if np.min(rc) != prow:
                        msg = "Processes with field {} are not in the "\
                            "same row\n".format(cachefield)
                        sys.stderr.write(msg)
                        tod.mpicomm.abort()

                # Every process finds out which process row is participating.
                prow = tod.mpicomm.bcast(prow, root=0)

                # Get the data on rank 0
                data = gather_field(prow, cachefield, dindx, cacheoff, ncache)

                if tod.mpicomm.rank == 0:
                    # The bindings of G3Vector seem to only work with
                    # lists...  Also there is no vectormap for unsigned
                    # char, so we have to use int...
                    for f in range(n_frames):
                        dataoff = fdataoff[f]
                        ndata = frame_sizes[f]
                        flagmaps[f][framefield] = \
                            c3g.G3VectorInt(\
                                data[dataoff:dataoff+ndata].astype(np.int32)\
                                .tolist())

            if tod.mpicomm.rank == 0:
                for f in range(n_frames):
                    fdata[f][flag_map] = flagmaps[f]

    return fdata
