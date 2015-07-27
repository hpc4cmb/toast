# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import sqlite3

import io

import glob

import astropy.io.fits as pf

from ..dist import distribute_det_samples

from ..tod.streams import Streams

from .utilities import load_ringdb, count_samples, read_eff, write_eff


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

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, ringdb=None, effdir=None, obt_range=None, ring_range=None, od_range=None, obtmask=0, flagmask=0, freq=None):
        
        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        if ringdb is None:
            raise ValueError('You must provide a path to the ring database')

        if effdir is None:
            raise ValueError('You must provide a path to the exchange files')
        
        if freq is None:
            raise ValueError('You must specify the frequency to run on')

        self.ringdb_path = ringdb
        self.ringdb = load_ringdb( self.ringdb_path, mpicomm )

        self.freq = freq
        if self.freq < 100:
            self.ringtable = 'ring_times_{}'.format(self.freq)
        else:
            self.ringtable = 'ring_times_hfi'

        self._offset,self._nsamp, self._sizes = count_samples( self.ringdb, self.ringtable, obt_range, ring_range, od_range )

        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=detectors, flavors=None, samples=0, sizes=self._sizes)        

        self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]

        self.effdir = effdir
        self.obtmask = obtmask
        self.flagmask = flagmask
        

    def _get(self, detector, flavor, local_start, n):

        data, flag = read_eff( detector, local_start, n, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, detector.lower(), self.obtmask, self.flagmask )

        return (data, flag)


    def _put(self, detector, flavor, local_start, data, flags):

        result = write_eff( detector, local_start, data, flags, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, detector.lower(), self.flagmask )

        return result

    @property
    def sizes(self):
        return self._sizes
