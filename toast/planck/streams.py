# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_det_samples

from ..tod.streams import Streams


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

        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)

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
