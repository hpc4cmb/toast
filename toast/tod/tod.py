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

    The "schema" of the TOD object (detector list and possible flavors)
    is fixed at construction time.

    Attributes:
        DEFAULT_FLAVOR (string): the name of the default detector data
        flavor which always exists.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
            detector.
        detectors (list): list of names to use for the detectors.
        flavors (list): list of *EXTRA* flavors to use (beyond the default).
        samples (int): the number of global samples represented by this TOD object.
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
        self._sizes = sizes

        # if sizes is specified, it must be consistent with
        # the total number of samples.
        if sizes is not None:
            test = np.sum(sizes)
            if samples != test:
                raise RuntimeError("Sum of sizes ({}) does not equal total samples ({})".format(test, samples))

        (self._dist_dets, self._dist_samples, self._dist_sizes) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp, sizes=self._sizes)

        self.stamps = None
        self.data = {}
        self.flags = {}
        self.pntg = {}
        self.pflags = {} 
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
    def chunks(self):
        return self._sizes

    @property
    def local_chunks(self):
        return self._dist_sizes

    @property
    def total_samples(self):
        return self._nsamp

    @property
    def local_samples(self):
        return self._dist_samples[1]

    @property
    def local_offset(self):
        return self._dist_samples[0]

    @property
    def local_dets(self):
        return self._dist_dets

    @property
    def mpicomm(self):
        return self._mpicomm

    @property
    def pointings(self):
        return sorted(list(self.pmat.keys()))


    def _get(self, detector, flavor, start, n):
        if detector not in self.data.keys():
            raise ValueError('detector {} data not yet written'.format(detector))
        if flavor not in self.data[detector].keys():
            raise ValueError('detector {} flavor {} data not yet written'.format(detector, flavor))
        return (self.data[detector][flavor][start:start+n], self.flags[detector][flavor][start:start+n])


    def _put(self, detector, flavor, start, data, flags):
        if detector not in self.data.keys():
            self.data[detector] = {}
            self.flags[detector] = {}
        if flavor not in self.data[detector].keys():
            self.data[detector][flavor] = np.zeros(self._dist_samples[1], dtype=np.float64)
            self.flags[detector][flavor] = np.zeros(self._dist_samples[1], dtype=np.uint8)
        n = data.shape[0]
        self.data[detector][flavor][start:start+n] = np.copy(data)
        self.flags[detector][flavor][start:start+n] = np.copy(flags)
        return


    def _get_pntg(self, detector, start, n):
        if detector not in self.pntg.keys():
            raise ValueError('detector {} pointing not yet written'.format(detector))
        return (self.pntg[detector][4*start:4*(start+n)], self.pflags[detector][start:start+n])


    def _put_pntg(self, detector, start, data, flags):
        if detector not in self.pntg.keys():
            self.pntg[detector] = np.zeros(4*self._dist_samples[1], dtype=np.float64)
            self.pflags[detector] = np.zeros(self._dist_samples[1], dtype=np.uint8)
        n = flags.shape[0]
        self.pntg[detector][4*start:4*(start+n)] = np.copy(data)
        self.pflags[detector][start:(start+n)] = np.copy(flags)
        return


    def _get_times(self, start, n):
        if self.stamps is None:
            raise RuntimeError('cannot read timestamps before writing them')
        return (self.stamps[start:start+n])


    def _put_times(self, start, stamps):
        if self.stamps is None:
            self.stamps = np.zeros(self._dist_samples[1], dtype=np.float64)
        n = stamps.shape[0]
        self.stamps[start:start+n] = np.copy(stamps)
        return


    def read(self, detector=None, flavor=None, local_start=0, n=0):
        if flavor is None:
            flavor = self.DEFAULT_FLAVOR
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.flavors:
            raise ValueError('flavor {} not found'.format(flavor))
        if (local_start < 0) or (local_start + n > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get(detector, flavor, local_start, n)


    def write(self, detector=None, flavor=None, local_start=0, data=None, flags=None):
        if flavor is None:
            flavor = self.DEFAULT_FLAVOR
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.flavors:
            raise ValueError('flavor {} not found'.format(flavor))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape != flags.shape:
            raise ValueError('data and flags arrays must be the same length')
        if (local_start < 0) or (local_start + data.shape[0] > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put(detector, flavor, local_start, data, flags)
        return


    def read_times(self, local_start=0, n=0):
        if (local_start < 0) or (local_start + n > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_times(local_start, n)


    def write_times(self, local_start=0, stamps=None):
        if stamps is None:
            raise ValueError('you must specify the vector of time stamps')
        if (local_start < 0) or (local_start + stamps.shape[0] > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+stamps.shape[0]-1))
        self._put_times(local_start, stamps)
        return


    def read_pntg(self, detector=None, local_start=0, n=0):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (local_start < 0) or (local_start + n > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_pntg(detector, local_start, n)


    def write_pntg(self, detector=None, local_start=0, data=None, flags=None):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape[0] != 4 * flags.shape[0]:
            raise ValueError('data and flags arrays must represent the same number of samples')
        if (local_start < 0) or (local_start + flags.shape[0] > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+flags.shape[0]-1))
        self._put_pntg(detector, local_start, data, flags)
        return


    def read_pmat(self, name=None, detector=None, local_start=0, n=0):
        if name is None:
            name = self.DEFAULT_FLAVOR
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if name not in self.pmat.keys():
            raise ValueError('pointing matrix {} not found'.format(name))
        if detector not in self.pmat[name].keys():
            raise RuntimeError('detector {} not found in pointing matrix {}'.format(detector, name))
        if (local_start < 0) or (local_start + n > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        if 'pixels' not in self.pmat[name][detector]:
            raise RuntimeError('detector {} in pointing matrix {} not yet written'.format(detector, name))
        nnz = int(len(self.pmat[name][detector]['weights']) / len(self.pmat[name][detector]['pixels']))
        return (self.pmat[name][detector]['pixels'][local_start:local_start+n], self.pmat[name][detector]['weights'][nnz*local_start:nnz*(local_start+n)])


    def write_pmat(self, name=None, detector=None, local_start=0, pixels=None, weights=None):
        if name is None:
            name = self.DEFAULT_FLAVOR
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (pixels is None) or (weights is None):
            raise ValueError('both pixels and weights must be specified')
        npix = pixels.shape[0]
        nw = weights.shape[0]
        nnz = int(nw / npix)
        if nnz * npix != nw:
            raise ValueError('number of pointing weights {} is not a multiple of pixels length {}'.format(nw, npix))
        if (local_start < 0) or (local_start + npix > self.local_samples):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+npix-1))
        if name not in self.pmat.keys():
            self.pmat[name] = {}
        if detector not in self.pmat[name].keys():
            self.pmat[name][detector] = {}
            self.pmat[name][detector]['pixels'] = np.zeros(self.local_samples, dtype=np.int64)
            self.pmat[name][detector]['weights'] = np.zeros(nnz*self.local_samples, dtype=np.float64)
        self.pmat[name][detector]['pixels'][local_start:local_start+npix] = np.copy(pixels)
        self.pmat[name][detector]['weights'][nnz*local_start:nnz*(local_start+npix)] = np.copy(weights)
        return


    def pmat_nnz(self, name=None, detector=None):
        if name is None:
            name = self.DEFAULT_FLAVOR
        if detector is None:
            raise ValueError('you must specify the detector')
        if name not in self.pmat.keys():
            raise ValueError('pointing matrix {} not found'.format(name))
        nnz = int(len(self.pmat[name][detector]['weights']) / len(self.pmat[name][detector]['pixels']))
        return nnz


