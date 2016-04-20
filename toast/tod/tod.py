# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_samples

from .cache import Cache


class TOD(object):
    """
    Base class for an object that provides detector pointing and
    timestreams for a single observation.

    Each TOD class has one or more detectors, and this class provides 
    pointing quaternions and flags for each detector.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
            detector.
        detectors (list): list of names to use for the detectors.
        samples (int): the number of global samples represented by this TOD object.
        sizes (list): specify the indivisible chunks in which to split the samples.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, samples=0, sizes=None):

        self._mpicomm = mpicomm
        self._timedist = timedist
        self._dets = []
        if detectors is not None:
            self._dets = detectors
        self._nsamp = samples
        self._sizes = sizes

        # if sizes is specified, it must be consistent with
        # the total number of samples.
        if sizes is not None:
            test = np.sum(sizes)
            if samples != test:
                raise RuntimeError("Sum of sizes ({}) does not equal total samples ({})".format(test, samples))

        (self._dist_dets, self._dist_samples, self._dist_sizes) = distribute_samples(self._mpicomm, self._timedist, self._dets, self._nsamp, sizes=self._sizes)

        if self._mpicomm.rank == 0:
            # check that all processes have some data, otherwise print warning
            for r in range(self._mpicomm.size):
                if len(self._dist_samples[r]) == 0:
                    print("WARNING: process {} has no data assigned in TOD.  Use fewer processes.".format(r))

        self._pref_detdata = 'toast_tod_detdata_'
        self._pref_detflags = 'toast_tod_detflags_'
        self._pref_detpntg = 'toast_tod_detpntg_'
        self._pref_detpflags = 'toast_tod_detpflags_'
        self._stamps = 'toast_tod_stamps'
        
        self.cache = Cache()


    @property
    def detectors(self):
        """
        The total list of detectors.
        """
        return self._dets

    @property
    def local_dets(self):
        """
        The detectors assigned to this process.
        """
        return self._dist_dets

    @property
    def timedist(self):
        return self._timedist

    @property
    def total_chunks(self):
        return self._sizes

    @property
    def dist_chunks(self):
        return self._dist_sizes

    @property
    def local_chunks(self):
        if self._dist_sizes is None:
            return None
        else:
            mysizes = self._dist_sizes[self._mpicomm.rank]
            if len(mysizes) == 0:
                return [(-1, -1)]
            else:
                return mysizes

    @property
    def total_samples(self):
        return self._nsamp

    @property
    def dist_samples(self):
        return self._dist_samples

    @property
    def local_samples(self):
        mysamples = self._dist_samples[self._mpicomm.rank]
        if len(mysamples) == 0:
            return [(-1, -1)]
        else:
            return mysamples

    @property
    def mpicomm(self):
        return self._mpicomm

    # The base class methods that get and put just use the cache.

    def _get(self, detector, start, n):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cachedata = "{}{}".format(self._pref_detdata, detector)
        cacheflags = "{}{}".format(self._pref_detflags, detector)
        if not self.cache.exists(cachedata):
            raise ValueError('detector {} data not yet written'.format(detector))
        if not self.cache.exists(cacheflags):
            raise ValueError('detector {} flags not yet written'.format(detector))
        dataref = self.cache.reference(cachedata)[start:start+n]
        flagsref = self.cache.reference(cacheflags)[start:start+n]
        return dataref, flagsref


    def _get_flags(self, detector, start, n):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cacheflags = "{}{}".format(self._pref_detflags, detector)
        if not self.cache.exists(cacheflags):
            raise ValueError('detector {} flags not yet written'.format(detector))
        flagsref = self.cache.reference(cachename)[start:start+n]
        return flagsref


    def _put(self, detector, start, data, flags):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cachedata = "{}{}".format(self._pref_detdata, detector)
        cacheflags = "{}{}".format(self._pref_detflags, detector)
        
        if not self.cache.exists(cachedata):
            self.cache.create(cachedata, np.float64, (self.local_samples[1],))
        if not self.cache.exists(cacheflags):
            self.cache.create(cacheflags, np.uint8, (self.local_samples[1],))
        
        n = data.shape[0]
        refdata = self.cache.reference(cachedata)[start:start+n]
        refflags = self.cache.reference(cacheflags)[start:start+n]

        refdata[:] = data
        refflags[:] = flags
        return


    def _put_flags(self, detector, start, flags):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cacheflags = "{}{}".format(self._pref_detflags, detector)
        
        if not self.cache.exists(cacheflags):
            self.cache.create(cacheflags, np.uint8, (self.local_samples[1],))
        
        n = data.shape[0]
        refflags = self.cache.reference(cacheflags)[start:start+n]

        refflags[:] = flags
        return


    def _get_pntg(self, detector, start, n):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cachepntg = "{}{}".format(self._pref_detpntg, detector)
        cachepflags = "{}{}".format(self._pref_detpflags, detector)
        if not self.cache.exists(cachepntg):
            raise ValueError('detector {} pointing data not yet written'.format(detector))
        if not self.cache.exists(cachepflags):
            raise ValueError('detector {} pointing flags not yet written'.format(detector))

        pntgref = self.cache.reference(cachepntg)[4*start:4*(start+n)]
        pflagsref = self.cache.reference(cachepflags)[start:start+n]
        
        return pntgref, pflagsref


    def _get_pntg_flags(self, detector, start, n):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cachepflags = "{}{}".format(self._pref_detpflags, detector)
        if not self.cache.exists(cachepflags):
            raise ValueError('detector {} pointing flags not yet written'.format(detector))
        pflagsref = self.cache.reference(cachepflags)[start:start+n]
        return pflagsref


    def _put_pntg(self, detector, start, data, flags):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cachepntg = "{}{}".format(self._pref_detpntg, detector)
        cachepflags = "{}{}".format(self._pref_detpflags, detector)
        if not self.cache.exists(cachepntg):
            self.cache.create(cachepntg, np.float64, (4*self.local_samples[1],))
        if not self.cache.exists(cachepflags):
            self.cache.create(cachepflags, np.uint8, (self.local_samples[1],))

        n = flags.shape[0]
        pntgref = self.cache.reference(cachepntg)[4*start:4*(start+n)]
        pflagsref = self.cache.reference(cachepflags)[start:start+n]

        pntgref[:] = data
        pflagsref[:] = flags
        return


    def _put_pntg_flags(self, detector, start, flags):
        if detector not in self.local_dets:
            raise ValueError('detector {} not assigned to local process'.format(detector))
        cachepflags = "{}{}".format(self._pref_detpflags, detector)
        if not self.cache.exists(cachepflags):
            self.cache.create(cachepflags, np.uint8, (self.local_samples[1],))

        n = flags.shape[0]
        pflagsref = self.cache.reference(cachepflags)[start:start+n]

        pflagsref[:] = flags
        return


    def _get_times(self, start, n):
        if not self.cache.exists(self._stamps):
            raise ValueError('timestamps not yet written')
        ref = self.cache.reference(self._stamps)[start:start+n]
        return ref


    def _put_times(self, start, stamps):
        if not self.cache.exists(self._stamps):
            self.cache.create(self._stamps, np.float64, (self.local_samples[1],))
        n = stamps.shape[0]
        ref = self.cache.reference(self._stamps)[start:start+n]
        ref[:] = stamps
        return


    def read(self, detector=None, local_start=0, n=0):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot read- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get(detector, local_start, n)


    def read_flags(self, detector=None, local_start=0, n=0):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot read flags- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_flags(detector, local_start, n)


    def write(self, detector=None, local_start=0, data=None, flags=None):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape != flags.shape:
            raise ValueError('data and flags arrays must be the same length')
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot write- process has no assigned local samples')
        if (local_start < 0) or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put(detector, local_start, data, flags)
        return


    def write_flags(self, detector=None, local_start=0, flags=None):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape != flags.shape:
            raise ValueError('data and flags arrays must be the same length')
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot write flags- process has no assigned local samples')
        if (local_start < 0) or (local_start + data.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put_flags(detector, local_start, flags)
        return


    def read_times(self, local_start=0, n=0):
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot read times- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_times(local_start, n)


    def write_times(self, local_start=0, stamps=None):
        if stamps is None:
            raise ValueError('you must specify the vector of time stamps')
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot write times- process has no assigned local samples')
        if (local_start < 0) or (local_start + stamps.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+stamps.shape[0]-1))
        self._put_times(local_start, stamps)
        return


    def read_pntg(self, detector=None, local_start=0, n=0):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot read pntg- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_pntg(detector, local_start, n)


    def read_pntg_flags(self, detector=None, local_start=0, n=0):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot read pntg flags- process has no assigned local samples')
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get_pntg_flags(detector, local_start, n)


    def write_pntg(self, detector=None, local_start=0, data=None, flags=None):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape[0] != 4 * flags.shape[0]:
            raise ValueError('data and flags arrays must represent the same number of samples')
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot write pntg- process has no assigned local samples')
        if (local_start < 0) or (local_start + flags.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+flags.shape[0]-1))
        self._put_pntg(detector, local_start, data, flags)
        return


    def write_pntg_flags(self, detector=None, local_start=0, flags=None):
        if detector is None:
            raise ValueError('you must specify the detector')
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape[0] != 4 * flags.shape[0]:
            raise ValueError('data and flags arrays must represent the same number of samples')
        if self.local_samples[1] <= 0:
            raise RuntimeError('cannot write pntg flags- process has no assigned local samples')
        if (local_start < 0) or (local_start + flags.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+flags.shape[0]-1))
        self._put_pntg_flags(detector, local_start, flags)
        return


    # def read_pmat(self, name=None, detector=None, local_start=0, n=0):
    #     if name is None:
    #         name = self.DEFAULT_FLAVOR
    #     if detector is None:
    #         raise ValueError('you must specify the detector')
    #     if detector not in self.local_dets:
    #         raise ValueError('detector {} not found'.format(detector))
    #     if name not in self.pmat.keys():
    #         raise ValueError('pointing matrix {} not found'.format(name))
    #     if detector not in self.pmat[name].keys():
    #         raise RuntimeError('detector {} not found in pointing matrix {}'.format(detector, name))
    #     if self.local_samples[1] <= 0:
    #         raise RuntimeError('cannot read pmat- process has no assigned local samples')
    #     if (local_start < 0) or (local_start + n > self.local_samples[1]):
    #         raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
    #     if 'pixels' not in self.pmat[name][detector]:
    #         raise RuntimeError('detector {} in pointing matrix {} not yet written'.format(detector, name))
    #     nnz = int(len(self.pmat[name][detector]['weights']) / len(self.pmat[name][detector]['pixels']))
    #     return (self.pmat[name][detector]['pixels'][local_start:local_start+n], self.pmat[name][detector]['weights'][nnz*local_start:nnz*(local_start+n)])


    # def write_pmat(self, name=None, detector=None, local_start=0, pixels=None, weights=None):
    #     if name is None:
    #         name = self.DEFAULT_FLAVOR
    #     if detector is None:
    #         raise ValueError('you must specify the detector')
    #     if detector not in self.local_dets:
    #         raise ValueError('detector {} not found'.format(detector))
    #     if (pixels is None) or (weights is None):
    #         raise ValueError('both pixels and weights must be specified')
    #     npix = pixels.shape[0]
    #     nw = weights.shape[0]
    #     nnz = int(nw / npix)
    #     if nnz * npix != nw:
    #         raise ValueError('number of pointing weights {} is not a multiple of pixels length {}'.format(nw, npix))
    #     if self.local_samples[1] <= 0:
    #         raise RuntimeError('cannot write pmat- process has no assigned local samples')
    #     if (local_start < 0) or (local_start + npix > self.local_samples[1]):
    #         raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+npix-1))
    #     if name not in self.pmat.keys():
    #         self.pmat[name] = {}
    #     if detector not in self.pmat[name].keys():
    #         self.pmat[name][detector] = {}
    #         self.pmat[name][detector]['pixels'] = np.zeros(self.local_samples[1], dtype=np.int64)
    #         self.pmat[name][detector]['weights'] = np.zeros(nnz*self.local_samples[1], dtype=np.float64)
    #     self.pmat[name][detector]['pixels'][local_start:local_start+npix] = pixels
    #     self.pmat[name][detector]['weights'][nnz*local_start:nnz*(local_start+npix)] = weights
    #     return



    # def pmat_nnz(self, name=None, detector=None):
    #     if name is None:
    #         name = self.DEFAULT_FLAVOR
    #     if detector is None:
    #         raise ValueError('you must specify the detector')
    #     if name not in self.pmat.keys():
    #         raise ValueError('pointing matrix {} not found'.format(name))
    #     nnz = int(len(self.pmat[name][detector]['weights']) / len(self.pmat[name][detector]['pixels']))
    #     return nnz

