# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_det_samples



class TOD(object):
    """
    Base class for an object that provides detector pointing and a 
    collection of streams for a single observation.

    Each TOD class has one or more detectors, and this class provides 
    pointing quaternions and flags for each detector.  Each detector
    might also have different flavors of detector data and flags.

    Attributes:
        DEFAULT_FLAVOR (string): the name of the default detector data
        flavor which always exists.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
            detector.
        detectors (list): list of names to use for the detectors.
        flavors (list): list of *EXTRA* flavors to use (beyond the default).
        samples (int): pre-initialize the storage with this number of samples.
        sizes (list): specify the indivisible chunks in which to split the samples.
    """

    DEFAULT_FLAVOR = 'default'

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, flavors=None, samples=0, sizes=None):

        self._mpicomm = mpicomm
        self._timedist = timedist
        self._dets = []
        if detectors is not None:
            self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        if flavors is not None:
            self._flavors.extend(flavors)
        self._nsamp = samples
        self._npntg = 4 * self._nsamp
        self._sizes = sizes

        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp, sizes=self._sizes)

        self._dist_npntg = (4*self._dist_samples[0], 4*self._dist_samples[1])

        self.stamps = np.zeros(self._dist_samples[1], dtype=np.float64)

        self.data = {}
        for det in self._dist_dets:
            self.data[det] = {}
            for flv in self._flavors:
                self.data[det][flv] = np.zeros(self._dist_samples[1], dtype=np.float64)

        self.flags = {}
        for det in self._dist_dets:
            self.flags[det] = {}
            for flv in self._flavors:
                self.flags[det][flv] = np.zeros(self._dist_samples[1], dtype=np.uint8)

        self.pntg = {}
        for det in self._dist_dets:
            self.pntg[det] = np.zeros(self._dist_npntg[1], dtype=np.float64)

        self.pflags = {}
        for det in self._dist_dets:
            self.pflags[det] = np.zeros(self._dist_npntg[1], dtype=np.uint8)

        self.pmat = {}


    @property
    def detectors(self):
        return self._dets

    @property
    def flavors(self):
        return self._flavors

    @property
    def timedist(self):
        return self._timedist

    @property
    def total_samples(self):
        return self._nsamp

    @property
    def local_samples(self):
        return self._dist_samples

    @property
    def local_dets(self):
        return self._dist_dets

    @property
    def mpicomm(self):
        return self._mpicomm


    def _get(self, detector, flavor, start, n):
        return (self.data[detector][flavor][start:start+n], self.flags[detector][flavor][start:start+n])


    def _put(self, detector, flavor, start, data, flags):
        n = data.shape[0]
        self.data[detector][flavor][start:start+n] = np.copy(data)
        self.flags[detector][flavor][start:start+n] = np.copy(flags)
        return


    def _get_pntg(self, detector, start, n):
        return (self.pntg[detector][4*start:4*(start+n)], self.pflags[detector][start:start+n])


    def _put_pntg(self, detector, start, data, flags):
        n = flags.shape[0]
        self.pntg[detector][4*start:4*(start+n)] = np.copy(data)
        self.pflags[detector][start:(start+n)] = np.copy(flags)
        return


    def _get_times(self, start, n):
        return (self.stamps[start:start+n])


    def _put_times(self, start, stamps):
        n = stamps.shape[0]
        self.stamps[start:start+n] = np.copy(stamps)
        return


    def read(self, detector=None, flavor='default', local_start=0, n=0):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.flavors:
            raise ValueError('flavor {} not found'.format(flavor))
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get(detector, flavor, local_start, n)


    def write(self, detector=None, flavor='default', local_start=0, data=None, flags=None):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.flavors:
            raise ValueError('flavor {} not found'.format(flavor))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape != flags.shape:
            raise ValueError('data and flags arrays must be the same length')
        if (local_start < 0) or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put(detector, flavor, local_start, data, flags)
        return


    def read_times(self, local_start=0, n=0):
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_times(local_start, n)


    def write_times(self, local_start=0, stamps=None):
        if (local_start < 0) or (local_start + stamps.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put_times(local_start, stamps)
        return


    def read_pntg(self, detector=None, local_start=0, n=0):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_pntg(detector, local_start, n)


    def write_pntg(self, detector=None, local_start=0, data=None, flags=None):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape[0] != 4 * flags.shape[0]:
            raise ValueError('data and flags arrays must be the same number of samples')
        if (local_start < 0) or (local_start + flags.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+flags.shape[0]-1))
        self._put_pntg(detector, local_start, data, flags)
        return


    def read_pmat(self, name=None, detector=None, local_start=0, n=0):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if name not in self.pmat.keys():
            raise ValueError('pointing matrix {} not found'.format(name))
        if detector not in self.pmat[name].keys():
            raise RuntimeError('detector {} not found in pointing matrix {}'.format(detector))
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        if 'pixels' not in self.pmat[name][detector]:
            raise RuntimeError('detector {} in pointing matrix {} does not have pixel vector'.format(detector, name))
        if 'weights' not in self.pmat[name][detector]:
            raise RuntimeError('detector {} in pointing matrix {} does not have weights vector'.format(detector, name))
        nnz = int(len(self.pmat[name][detector]['weights']) / self.pmat[name][detector]['pixels'])
        return (self.pmat[name][detector]['pixels'][local_start:local_start+n], self.pmat[name][detector]['weights'][nnz*local_start:nnz*(local_start+n)])


    def write_pmat(self, name=None, detector=None, local_start=0, pixels=None, weights=None):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (pixels is None) or (weights is None):
            raise ValueError('both pixels and weights must be specified')
        npix = pixels.shape[0]
        nw = weights.shape[0]
        nnz = int(nw / npix)
        if nnz * npix != nw:
            raise ValueError('number of pointing weights {} is not a multiple of pixels length {}'.format(nw, npix))
        if (local_start < 0) or (local_start + npix > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+npix-1))
        if name not in self.pmat.keys():
            self.pmat[name] = {}
        if detector not in self.pmat[name].keys():
            self.pmat[name][detector] = {}
            self.pmat[name][detector]['pixels'] = np.zeros(self.local_samples[1], dtype=np.int64)
            self.pmat[name][detector]['weights'] = np.zeros(nnz*self.local_samples[1], dtype=np.float64)
        self.pmat[name][detector]['pixels'][local_start:local_start+npix] = np.copy(pixels)
        self.pmat[name][detector]['weights'][nnz*local_start:nnz*(local_start+npix)] = np.copy(weights)
        return


    def pmat_nnz(self, name=None):
        if name not in self.pmat.keys():
            raise ValueError('pointing matrix {} not found'.format(name))
        nnz = int(len(self.pmat[name][detector]['weights']) / self.pmat[name][detector]['pixels'])
        return nnz



class TODFake(TOD):
    """
    Provide a simple generator of fake detector data.

    This provides timestreams for a specified number of detectors.  The
    sky signal is a linear gradient across healpix pixels in NEST ordering.
    The focalplane geometry is specified
    and a focalplane geometry provided by a specifying the quaternion rotation for
    each detector from the boresight.  The boresight pointing is just wrapping 
    around the healpix sphere in ring order

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
                  detector.
        detectors (dictionary): each key is the detector name, and each value
                  is a quaternion.
        rms (float): RMS of the white noise.
        min (float): minimum of signal gradient.
        max (float): maximum of signal gradient.
        samples (int): maximum allowed samples.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, rms=1.0, samples=0, rngstream=0, firsttime=0.0, rate=100.0):
        
        # We call the parent class constructor to set the MPI communicator and
        # distribution type, but we do NOT pass the detector list, as this 
        # would allocate memory for the data buffer of the base class.
        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)

        if detectors is None:
            # in this case, we will have a single detector at the boresight
            self._dets = {'boresight' : (0.0, 0.0, 0.0, 1.0)}
        else:
            self._dets = detectors

        self._flavors = [self.DEFAULT_FLAVOR]
        self._nsamp = samples
        
        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp)

        self._dist_npntg = (4*self._dist_samples[0], 4*self._dist_samples[1])

        self._theta_steps = 180
        self._phi_steps = 360

        self.rngstream = rngstream
        self.seeds = {}
        for det in enumerate(self._dets.keys()):
            self.seeds[det[1]] = det[0] 
        self.rms = rms
        self.firsttime = firsttime
        self.rate = rate


    def _get(self, detector, flavor, start, n):
        # Setting the seed like this does NOT guarantee uncorrelated
        # results from the generator.  This is just a place holder until
        # the streamed rng is implemented.
        np.random.seed(self.seeds[detector])
        trash = np.random.normal(loc=0.0, scale=self.rms, size=(n-start))
        return ( np.random.normal(loc=0.0, scale=self.rms, size=n), np.zeros(n, dtype=np.uint8) )


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('cannot write data to simulated data streams')
        return


    def _get_times(self, start, n):
        start_abs = self._dist_samples[0] + start
        start_time = self.firsttime + float(start_abs) / self.rate
        stop_time = start_time + float(n) / self.rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False, dtype=np.float64)
        return stamps


    def _put_times(self, start, stamps):
        raise RuntimeError('cannot write timestamps to simulated data streams')
        return


    def _get_pntg(self, detector, start, n):
        # compute the absolute sample offset
        start_abs = self._dist_samples[0] + start

        # compute the position on the sphere, spiralling from North
        # pole to South.  obviously the pointings will be densest at
        # the poles.
        start_theta = int(start_abs / self._theta_steps)
        start_phi = int(start_abs / self._phi_steps)

        # ... in progress ...

        # convert to quaternions
        # FIXME: use our quaternion library once implemented

        data = np.zeros(4*n, dtype=np.float64)
        flags = np.zeros(n, dtype=np.uint8)

        return (data, flags)


    def _put_pntg(self, detector, start, data, flags):
        raise RuntimeError('cannot write data to simulated pointing')
        return


