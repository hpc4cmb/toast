# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_det_samples


class Streams(object):
    """
    Base class for an object that provides a collection of streams.

    Each Streams class has one or more detectors, and each detector
    might have different flavors of data and flags.

    Attributes:
        DEFAULT_FLAVOR (string): the name of the default flavor which always exists.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
            detector.
        detectors (list): list of names to use for the detectors.
        flavors (list): list of *EXTRA* flavors to use (beyond the default).
        samples (int): pre-initialize the storage with this number of samples.
    """

    DEFAULT_FLAVOR = 'default'

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, flavors=None, samples=0):

        self._mpicomm = mpicomm
        self._timedist = timedist

        self._dets = []
        if detectors is not None:
            self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        if flavors is not None:
            self._flavors.extend(flavors)
        self._nsamp = samples
        
        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp)

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



class StreamsWhiteNoise(Streams):
    """
    Provide white noise streams.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
                  detector.
        detectors (list): list of names to use for the detectors.
        rms (float): RMS of the white noise.
        samples (int): maximum allowed samples.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, rms=1.0, samples=0, rngstream=0):
        
        # We call the parent class constructor to set the MPI communicator and
        # distribution type, but we do NOT pass the detector list, as this 
        # would allocate memory for the data buffer of the base class.
        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)

        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        self._nsamp = samples
        
        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp)

        self.rngstream = rngstream
        self.seeds = {}
        for det in enumerate(self._dets):
            self.seeds[det[1]] = det[0] 
        self.rms = rms


    def _get(self, detector, flavor, start, n):
        # Setting the seed like this does NOT guarantee uncorrelated
        # results from the generator.  This is just a place holder until
        # the streamed rng is implemented.
        np.random.seed(self.seeds[detector])
        trash = np.random.normal(loc=0.0, scale=self.rms, size=(n-start))
        return ( np.random.normal(loc=0.0, scale=self.rms, size=n), np.zeros(n, dtype=np.uint8) )


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('cannot write data to simulated white noise streams')
        return


class StreamsPlanckEFF(Streams):
    """
    Provide Planck Exchange File Format streams

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
                  detector.
        detectors (list): list of names to use for the detectors. Must match the names in the FITS HDUs.
        samples (int): maximum allowed samples.
        ringdb: Path to an SQLite database defining ring boundaries.
        effdir: directory containing the exchange files
        obt_range: data span in TAI seconds, overrides ring_range
        ring_range: data span in pointing periods, overrides od_range
        od_range: data span in operational days
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, fn_ringdb=None, effdir=None, obt_range=None, ring_range=None, od_range=None, obtmask=0, flagmask=0, freq=None):
        
        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        if fn_ringdb is None:
            raise ValueError('You must provide a path to the ring database')

        if effdir is None:
            raise ValueError('You must provide a path to the exchange files')
        
        if freq is None:
            raise ValueError('You must set specify the frequency to run on')

        if obt_range is None and ring_range is None and od_range is None:
            raise ValueError('Cannot initialize EFF streams without one of obt_range, ring_range or od_range')

        self._load_ringdb( fn_ringdb, mpicomm )

        self.freq = freq

        samples = self._count_samples( obt_range, ring_range, od_range )

        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=detectors, flavors=None, samples=0)

        self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        self._nsamp = samples

        self.effdir = effdir
        self.obtmask = obtmask
        self.flagmask = flagmask
        

    def _get(self, detector, flavor, start, n):
        raise RuntimeError('Reading from EFF not implemented yet.')
        return


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('Writing to EFF not implemented yet.')
        return


    def _load_ringdb(self, fn_ringdb, mpicomm):
        """
        Load and broadcast the ring database.
        """

        itask = mpicomm.Get_rank()
        ntask = mpicomm.Get_size()
        
        # Read database to tempfile and broadcast

        if itask == 0:

            conn = sqlite3.connect(fn_ringdb)
            tempfile = StringIO.StringIO()
            for line in conn.iterdump():
                tempfile.write('{}\n'.format(line))
            conn.close()
            tempfile.seek( 0 )

        tempfile = mpicomm.bcast(tempfile, root=0)

        # Create a database in memory and import from tempfile

        self.ringdb = sqlite3.connect(':memory:')
        self.ringdb.cursor().executescript(tempfile.read())
        self.ringdb.commit()


    def _count_samples(self, obt_range, ring_range, od_range):
        """
        Query the ring database to determine which global samples the requested sample indices point to.
        """

        if self.freq < 100:
            self.ringtable = 'ring_times_{}'.format(self.freq)
        else:
            self.ringtable = 'ring_times_hfi'

        if obt_range is not None:
            start, stop = obt_range

            cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format( self.ringtable, start )
            start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = self.conn.execute( cmd ).fetchall()[-1]

            cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format( self.ringtable, stop )
            start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = self.conn.execute( cmd ).fetchall()[0]

            self.offset = start_index1
            
            return stop_index2 - start_index1
        elif ring_range is not None:
            raise Exception('Ring span selection not yet implemented')
        elif od_range is not None:
            raise Exception('OD span selection not yet implemented')

        raise Exception('No data span selected')
