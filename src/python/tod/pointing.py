# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from .. import healpix as hp

from .. import qarray as qa

from ..op import Operator
from ..dist import Comm, Data
from .tod import TOD



class OpPointingHpix(Operator):
    """
    Operator which generates I/Q/U healpix pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.  An optional dictionary of calibration factors may
    be specified.  Additional options include specifying a constant cross-polar
    response (eps) and a rotating, perfect half-wave plate.  The timestream 
    model is then (see Jones, et al, 2006):

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a} + U \\sin{2a}\\right]\\right]

    Or, if a HWP is included in the response with time varying angle "w", then
    the total response is:

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{4(a+w)} + U \\sin{4(a+w)}\\right]\\right]

    Args:
        pixels (str): write pixels to the cache with name <pixels>_<detector>.
            If the named cache objects do not exist, then they are created.
        weights (str): write pixel weights to the cache with name 
            <weights>_<detector>.  If the named cache objects do not exist, 
            then they are created. 
        nside (int): NSIDE resolution for Healpix NEST ordered intensity map.
        nest (bool): if True, use NESTED ordering.
        mode (string): either "I" or "IQU"
        cal (dict): dictionary of calibration values per detector. A None
            value means a value of 1.0 for all detectors.
        epsilon (dict): dictionary of cross-polar response per detector. A
            None value means epsilon is zero for all detectors.
        hwprpm: if None, a constantly rotating HWP is not included.  Otherwise
            it is the rate (in RPM) of constant rotation.
        hwpstep: if None, then a stepped HWP is not included.  Otherwise, this
            is the step in degrees.
        hwpsteptime: The time in minutes between HWP steps.
        common_flag_name (str): the optional name of a cache object to use for
            the common flags.
        common_flag_mask (byte): the bitmask to use when flagging the pointing
            matrix using the common flags.
        apply_flags (bool): whether to read the TOD common flags, bitwise OR
            with the common_flag_mask, and then flag the pointing matrix.
    """

    def __init__(self, pixels='pixels', weights='weights', nside=64, nest=False, mode='I', cal=None, epsilon=None, hwprpm=None, hwpstep=None, hwpsteptime=None, common_flag_name=None, common_flag_mask=255, apply_flags=False):
        self._pixels = pixels
        self._weights = weights
        self._nside = nside
        self._nest = nest
        self._mode = mode
        self._cal = cal
        self._epsilon = epsilon
        self._common_flag_mask = common_flag_mask
        self._apply_flags = apply_flags
        self._common_flag_name = common_flag_name

        if (hwprpm is not None) and (hwpstep is not None):
            raise RuntimeError("choose either continuously rotating or stepped HWP")

        if (hwpstep is not None) and (hwpsteptime is None):
            raise RuntimeError("for a stepped HWP, you must specify the time between steps")

        if hwprpm is not None:
            # convert to radians / second
            self._hwprate = hwprpm * 2.0 * np.pi / 60.0
        else:
            self._hwprate = None

        if hwpstep is not None:
            # convert to radians and seconds
            self._hwpstep = hwpstep * np.pi / 180.0
            self._hwpsteptime = hwpsteptime * 60.0
        else:
            self._hwpstep = None
            self._hwpsteptime = None

        # initialize the healpix pixels object
        self.hpix = hp.Pixels(self._nside)

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    @property
    def nside(self):
        """
        (int): the HEALPix NSIDE value used.
        """
        return self._nside

    @property
    def nest(self):
        """
        (bool): if True, the pointing is NESTED ordering.
        """
        return self._nest

    @property
    def mode(self):
        """
        (str): the pointing mode "I", "IQU", etc.
        """
        return self._mode


    def exec(self, data):
        """
        Create pixels and weights.

        This iterates over all observations and detectors, and creates
        the pixel and weight arrays representing the pointing matrix.
        This data is stored in the TOD cache.

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

        xaxis = np.array([1,0,0], dtype=np.float64)
        yaxis = np.array([0,1,0], dtype=np.float64)
        zaxis = np.array([0,0,1], dtype=np.float64)
        nullquat = np.array([0,0,0,1], dtype=np.float64)

        for obs in data.obs:
            tod = obs['tod']

            # compute effective sample rate

            times = tod.read_times(local_start=0, n=tod.local_samples[1])
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt

            # generate HWP angles

            nsamp = tod.local_samples[1]
            first = tod.local_samples[0]
            hwpang = None

            if self._hwprate is not None:
                # continuous HWP
                # HWP increment per sample is: 
                # (hwprate / samplerate)
                hwpincr = self._hwprate / rate
                startang = np.fmod(first * hwpincr, 2*np.pi)
                hwpang = hwpincr * np.arange(nsamp, dtype=np.float64)
                hwpang += startang
            elif self._hwpstep is not None:
                # stepped HWP
                hwpang = np.ones(nsamp, dtype=np.float64)
                stepsamples = int(self._hwpsteptime * rate)
                wholesteps = int(first / stepsamples)
                remsamples = first - wholesteps * stepsamples
                curang = np.fmod(wholesteps * self._hwpstep, 2*np.pi)
                curoff = 0
                fill = remsamples
                while (curoff < nsamp):
                    if curoff + fill > nsamp:
                        fill = nsamp - curoff
                    hwpang[curoff:fill] *= curang
                    curang += self._hwpstep
                    curoff += fill
                    fill = stepsamples

            # read the common flags and apply bitmask

            common = None
            if self._apply_flags:
                if self._common_flag_name is not None:
                    common = np.copy(tod.cache.reference(self._common_flag_name))
                else:
                    common = tod.read_common_flags()
                common &= self._common_flag_mask

            for det in tod.local_dets:

                eps = 0.0
                if self._epsilon is not None:
                    eps = self._epsilon[det]

                cal = 1.0
                if self._cal is not None:
                    cal = self._cal[det]

                eta = (1 - eps) / ( 1 + eps )

                pdata = np.copy(tod.read_pntg(detector=det))

                if self._apply_flags:
                    pdata[common != 0,:] = nullquat

                dir = qa.rotate(pdata, np.tile(zaxis, nsamp).reshape(-1,3))

                #pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=self._nest)
                pixels = None
                if self._nest:
                    pixels = self.hpix.vec2nest(dir)
                else:
                    pixels = self.hpix.vec2ring(dir)

                if self._apply_flags:
                    pixels[common != 0] = -1

                if self._mode == 'I':
                    
                    weights = np.ones((nsamp,1), dtype=np.float64)
                    weights *= cal

                elif self._mode == 'IQU':

                    orient = qa.rotate(pdata.reshape(-1, 4), np.tile(xaxis, nsamp).reshape(-1,3))

                    by = orient[:,0] * dir[:,1] - orient[:,1] * dir[:,0]
                    bx = orient[:,0] * (-dir[:,2] * dir[:,0]) + orient[:,1] * (-dir[:,2] * dir[:,1]) + orient[:,2] * (dir[:,0] * dir[:,0] + dir[:,1] * dir[:,1])
                        
                    detang = np.arctan2(by, bx)

                    if hwpang is not None:
                        detang += 2.0*hwpang
                    detang *= 2.0

                    cang = np.cos(detang)
                    sang = np.sin(detang)
                     
                    Ival = np.ones_like(cang)
                    Ival *= cal
                    Qval = cang
                    Qval *= (eta * cal)
                    Uval = sang
                    Uval *= (eta * cal)

                    weights = np.ravel(np.column_stack((Ival, Qval, Uval))).reshape(-1,3)

                else:
                    raise RuntimeError("invalid mode for healpix pointing")

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                
                pixelsref = None
                weightsref = None

                if tod.cache.exists(pixelsname):
                    pixelsref = tod.cache.reference(pixelsname)
                else:
                    pixelsref = tod.cache.create(pixelsname, np.int64, (tod.local_samples[1],))

                if tod.cache.exists(weightsname):
                    weightsref = tod.cache.reference(weightsname)
                else:
                    weightsref = tod.cache.create(weightsname, np.float64, (tod.local_samples[1],weights.shape[1]))

                pixelsref[:] = pixels
                weightsref[:,:] = weights

                del pixelsref
                del weightsref

            del common

        return

