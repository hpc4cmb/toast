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

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, ringdb=None, effdir=None, obt_range=None, ring_range=None, od_range=None, obtmask=0, flagmask=0, freq=None):
        
        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        if ringdb is None:
            raise ValueError('You must provide a path to the ring database')

        if effdir is None:
            raise ValueError('You must provide a path to the exchange files')
        
        if freq is None:
            raise ValueError('You must set specify the frequency to run on')

        self._load_ringdb( ringdb, mpicomm )

        self.freq = freq

        samples = self._count_samples( obt_range, ring_range, od_range )

        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)

        self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        self._nsamp = samples

        self.effdir = effdir
        self.obtmask = obtmask
        self.flagmask = flagmask
        

    def _get(self, detector, flavor, local_start, n):

        # Convert to global indices

        start = self.offset + self.local_samples[0] + local_start
        stop = start + n

        # Start, stop interval follows Python conventions. (stop-start) samples are read starting at "start". Sample at "stop" is omitted

        # Determine first and last ring to read

        ntry = 10
        try:
            for itry in range(ntry):
                try:
                    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format(self.ringtable, start)
                    start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = self.ringdb.execute( cmd ).fetchall()[-1]
                except Exception as e:
                    if itry < ntry: continue
                    raise Exception('Failed to query ring start and stop times from {} (1/2). Failed query: {}. Error: {}'.format(self.ringdb_path, cmd, e))
                try:
                    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format(self.ringtable, stop)
                    start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = self.ringdb.execute( cmd ).fetchall()[0]
                except Exception as e:
                    if itry < ntry: continue
                    raise Exception('Failed to query ring start and stop times from {} (2/2). Failed query: {}. Error: '.format(self.ringdb_path, cmd, e))
        except Exception as e:
            raise Exception( 'StreamsPlanckEFF._get: Giving up after {} tries to get source ODs: {}'.format(ntry,e) )

        # Determine the list of ODs that contain the rings

        ods = []
        if self.freq < 100:
            query = self.conn.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == {}'.format(self.freq, start_time1, stop_time2) )
        else:
            query = self.conn.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == 100'.format(start_time1, stop_time2) )
        for q in query:
            ods.append( [int(q[0]), int(q[1])] )

        data = []
        flag = []
        first_row = start_row1 + ( start - start_index1 )
        nleft = stop - start
        nread = 0

        while len(ods) > 0:
            od, nrow = ods[0]

            if nrow < first_row:
                ods = ods[1:]
                first_row -= nrow
                continue

            pattern = self.effdir + '/{:04}/?{:03}*fits'.format(od, self.freq)
            try:
                fn = sorted( glob.glob( pattern ) )[-1]
            except:
                raise Exception( 'Error: failed to find a file to read matching: ' + pattern )
            h = pf.open( fn, 'readonly' )
            hdu = 1
            while detector.lower() not in h[hdu].header['extname'].strip().lower():
                hdu += 1
                if hdu == len(h): raise Exception('No HDU matching extname = {} in {}'.format(self.extname, fn))

            if nrow - first_row > nleft:
                last_row = first_row + nleft
            else:
                last_row = nrow

            nbuff = last_row - first_row

            ncol = len( h[hdu].columns )

            if ncol == 2:
                dat = np.array( h[hdu].data.field(0)[first_row:last_row] )
            else:
                dat = np.array( [ h[hdu].data.field(col)[first_row:last_row].ravel() for col in range(ncol-1) ] )
            data.append( dat )

            flg = np.zeros( last_row-first_row, dtype=np.byte )
            if self.obtmask != 0:
                if self.obtmask > 0:
                    flg += np.array(h[1].data.field(1)[first_row:last_row], dtype=np.byte) & self.obtmask
                else:
                    flg += np.logical_not( np.array(h[1].data.field(1)[first_row:last_row], dtype=np.byte) & -self.obtmask )
            if self.flagmask != 0:
                if self.flagmask > 0:
                    flg += np.array(h[hdu].data.field(ncol-1)[first_row:last_row], dtype=np.byte) & self.flagmask
                else:
                    flg += np.logical_not( np.array(h[hdu].data.field(ncol-1)[first_row:last_row], dtype=np.byte) & -self.flagmask )
            flag.append( flg )

            ods = ods[1:]
            first_row = 0
            h.close()

            nread += nbuff
            nleft -= nbuff

            if nleft == 0: break

        if len(data) > 0: data = np.hstack(data).T
        if len(flag) > 0: flag = np.hstack(flag).T

        data = np.array( data )
        flag = np.array( flag != 0 )

        if np.shape( flag )[-1] != stop - start:
            raise Exception('ERROR: inconsistent dimensions: shape(data) = ', np.shape(data), ', shape(flag) = ', np.shape(flag), ', stop-start = ', stop-start)

        if self.shape[0] > 1: data = data.T # Conform to expected geometry            

        return (data, flag)


    def _put(self, detector, flavor, local_start, data, flags):

        # Convert to global indices

        start = self.offset + self.local_samples[0] + local_start
        stop = start + len(data)

        # Promote all input data to 2 dimensions to support multicolumn writing

        data2d = np.atleast_2d( data )

        if len(data2d[0]) != stop - start:
            raise Exception( 'streamsPlanckEFF: stop-start = {} but len(data) = {}'.format( stop-start, len(data2d[0]) ) )

        ntry = 10
        try:
            for itry in range(10):
                try:
                    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format(self.ringtable, start)
                    start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = self.ringdb.execute( cmd ).fetchall()[-1]
                except Exception as e:
                    if itry < ntry: continue
                    raise Exception('Failed to query ring start and stop times from {} (1/2). Failed query: {}. Error: {}'.format(self.ringdb_path, cmd, e))

                try:
                    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format(self.ringtable, stop)
                    start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = self.ringdb.execute( cmd ).fetchall()[0]
                except Exception as e:
                    if itry < ntry: continue
                    raise Exception('Failed to query ring start and stop times from {} (2/2). Failed query: {}. Error: '.format(self.ringdb_path, cmd, e))
        except Exception as e:
            raise Exception( 'streamsPlanckEFF._put: Giving up after {} tries to get target ODs: {}'.format(ntry,e) )

        ods = []
        if self.freq < 100:
            query = self.conn.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == {}'.format(self.freq, start_time1, stop_time2) )
        else:
            query = self.conn.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == 100'.format(start_time1, stop_time2) )
        for q in query:
            ods.append( [int(q[0]), int(q[1])] )

        nleft = stop - start
        nwrote = 0

        first_row = start_row1 + ( start - start_index1 )
        offset = 0

        while len(ods) > 0:
            od, nrow = ods[0]

            if nrow < first_row:
                ods = ods[1:]
                first_row -= nrow
                continue

            pattern = self.effdir + '/{:04}/?{:03}*fits'.format(od, self.freq)
            try:
                fn = sorted( glob.glob( pattern ) )[-1]
            except:
                raise Exception( 'Error: failed to find a file to update matching: ' + pattern )
            h = pf.open( fn, 'update' )
            hdu = 1
            while detector.lower() not in h[hdu].header['extname'].strip().lower():
                hdu += 1
                if hdu == len(h): raise Exception('No HDU matching extname = {} in {}'.format(self.extname, fn))

            if nrow - first_row > nleft:
                last_row = first_row + nleft
            else:
                last_row = nrow

            nwrite = last_row - first_row
            ncol = len( h[hdu].columns )
            ncol_data = len(data2d)

            if ncol-1 != ncol_data: raise Exception( 'streamsPlanckEFF._put: Expected {} columns to write data but got {}.'.format(ncol-1,ncol_data) )

            if nwrite > 0:
                try:
                    for col in range(len(data2d)):
                        h[hdu].data.field(col)[first_row:last_row] = data2d[col][offset:offset+nwrite]
                    if self.flagmask >= 0:
                        h[hdu].data.field(ncol-1)[first_row:last_row] = np.array(flags[offset:offset+nwrite] != 0, dtype=np.byte) * self.flagmask
                    else:
                        h[hdu].data.field(ncol-1)[first_row:last_row] = np.array(flags[offset:offset+nwrite] == 0, dtype=np.byte) * (-self.flagmask)
                except:
                    raise Exception( 'Indexing error: fn = {}, start = {}, stop = {}, first_row = {}, last_row = {}, offset = {}, nwrite = {}, len(h[hdu].data.field(1)) = {}, len(data) = {}, self.flagmask = {}'.format( fn, start, stop, first_row, last_row, offset, nwrite, len(h[hdu].data.field(1)), len(data2d[0]), self.flagmask ) )
                offset += nwrite

            ods = ods[1:]
            first_row = 0
            result = h.flush()
            h.close()

            nwrote += nwrite
            nleft -= nwrite

            if nleft == 0: break

        return result


    def _load_ringdb(self, path, mpicomm):
        """
        Load and broadcast the ring database.
        """

        itask = mpicomm.Get_rank()
        ntask = mpicomm.Get_size()
        
        # Read database to tempfile and broadcast

        if itask == 0:

            conn = sqlite3.connect(path)
            tempfile = StringIO.StringIO()
            for line in conn.iterdump():
                tempfile.write('{}\n'.format(line))
            conn.close()
            tempfile.seek( 0 )

        tempfile = mpicomm.bcast(tempfile, root=0)

        # Create a database in memory and import from tempfile

        self.ringdb_path = path

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
            start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = self.ringdb.execute( cmd ).fetchall()[-1]

            cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format( self.ringtable, stop )
            start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = self.ringdb.execute( cmd ).fetchall()[0]

            self.offset = start_index1
            
            return stop_index2 - start_index1
        elif ring_range is not None:
            raise Exception('Ring span selection not yet implemented')
        elif od_range is not None:
            raise Exception('OD span selection not yet implemented')
        else:
            # no span specified, use all available data

            cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} order by start_index'.format( self.ringtable )

            start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = self.ringdb.execute( cmd ).fetchall()[-1]

            start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = self.ringdb.execute( cmd ).fetchall()[0]

            self.offset = start_index1
            
            return stop_index2 - start_index1
