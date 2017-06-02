# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.constants import degree

from ..op import Operator
from ..dist import Comm, Data
from .tod import TOD
from ..mpi import MPI


class OpGroundFilter(Operator):
    """
    Operator which applies ground template filtering to constant
    elevation scans.

    Args:
        wbin (float):  Width of an azimuth bin (degrees).
        name (str):  Name of the output signal cache object will be
            <name_in>_<detector>.  If the object exists, it is used as
            input.  Otherwise signal is read using the tod read method.
        common_flag_name (str):  Cache name of the output common flags.
            If it already exists, it is used.  Otherwise flags
            are read from the tod object and stored in the cache under
            common_flag_name.
        common_flag_mask (byte):  Bitmask to use when flagging data
           based on the common flags.
        flag_name (str):  Cache name of the output detector flags will
            be <flag_name>_<detector>.  If the object exists, it is
            used.  Otherwise flags are read from the tod object.
        flag_mask (byte):  Bitmask to use when flagging data
           based on the detector flags.
        ground_flag_mask (byte):  Bitmask to use when adding flags based
           on ground filter failures.
    """

    def __init__(self, wbin=1, name='signal',
                 common_flag_name='common_flags', common_flag_mask=255,
                 flag_name='flags', flag_mask=255, ground_flag_mask=1):

        self._wbin = wbin
        self._name = name
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._ground_flag_mask = ground_flag_mask

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    def exec(self, data):
        """
        Apply the ground filter to the signal.

        Args:
            data (toast.Data): The distributed data.
        """
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        # Each group loops over its own CES:es

        for obs in data.obs:
            tod = obs['tod']

            try:
                (azmin, azmax, elmin, elmax) = tod.scan_range
                az = tod.read_boresight_az()
            except Exception as e:
                raise RuntimeError(
                    'Failed to get boresight azimuth from TOD.  Perhaps it is '
                    'not ground TOD? "{}"'.format(e))

             # Cache the output common flags
            cachename = self._common_flag_name
            if tod.cache.exists(cachename):
                common_ref = tod.cache.reference(cachename)
            else:
                common_flag = tod.read_common_flags()
                common_ref = tod.cache.put(cachename, common_flag)
                del common_flag

            # The azimuth vector is assumed to be arranged so that the
            # azimuth increases monotonously even across the zero meridian.

            wbin = self._wbin * degree
            nbin = int((azmax - azmin) // wbin + 1)
            ibin = ((az - azmin) // wbin).astype(np.int)

            for det in tod.detectors:
                hits = np.zeros(nbin)
                binned = np.zeros(nbin)

                # Bin the local data

                if det in tod.local_dets:
                    # Cache the output signal
                    cachename = '{}_{}'.format(self._name, det)
                    if tod.cache.exists(cachename):
                        ref = tod.cache.reference(cachename)
                    else:
                        signal = tod.read(detector=det)
                        ref = tod.cache.put(cachename, signal)

                    # Cache the output flags
                    cachename = '{}_{}'.format(self._flag_name, det)
                    if tod.cache.exists(cachename):
                        flag_ref = tod.cache.reference(cachename)
                    else:
                        # read_flags always returns both common and detector
                        # flags but we already cached the common flags.
                        flag, dummy = tod.read_flags(detector=det)
                        flag_ref = tod.cache.put(cachename, flag)
                        del flag, dummy

                    good = np.logical_and(
                        common_ref & self._common_flag_mask == 0,
                        flag_ref & self._flag_mask == 0)

                    # If binning ever becomes a bottleneck, it must be
                    # implemented in Cython or compiled code.  The range
                    # checks are very expensive.

                    for i, s in zip(ibin[good], ref[good]):
                        hits[i] += 1
                        binned[i] += s

                    del flag_ref

                # Reduce the binned data

                cgroup.Allreduce(MPI.IN_PLACE, hits, op=MPI.SUM)
                cgroup.Allreduce(MPI.IN_PLACE, binned, op=MPI.SUM)
                good = hits != 0
                binned[good] /= hits[good]

                # Subtract the ground template

                if det in tod.local_dets:
                    ref -= binned[ibin]
                    del ref

            del common_ref

        return

