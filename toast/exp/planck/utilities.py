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

from toast.dist import distribute_det_samples

from toast.tod.interval import Interval

from collections import namedtuple

from scipy.constants import degree, arcmin, arcsec, c

import quaternionarray as qa

xaxis, yaxis, zaxis = np.eye( 3 )

spinangle = 85.0 * degree

spinrot = qa.rotation( yaxis, np.pi/2 - spinangle )

# define a named tuples to contain detector data and RIMO in a standardized format

DetectorData = namedtuple( 'DetectorData', 'detector phi_uv theta_uv psi_uv psi_pol epsilon fsample fknee alpha net quat' )

def read_eff(local_start, n, globalfirst, local_offset, ringdb, ringdb_path, freq, effdir, extname, obtmask, flagmask, eff_cache=None, debug=0 ):

    if n < 1: raise Exception( 'ERROR: cannot read negative number of samples: {}'.format(n) )

    ringtable = ringdb_table_name(freq)

    # Convert to global indices

    start = globalfirst + local_offset + local_start
    stop = start + n

    # Start, stop interval follows Python conventions. (stop-start) samples are read starting at "start". Sample at "stop" is omitted

    # Determine first and last ring to read

    start_time1 = None
    start_time2 = None
    stop_time1 = None
    stop_time2 = None
    start_index1 = None
    start_index2 = None
    stop_index1 = None
    stop_index2 = None
    start_row1 = None
    start_row2 = None
    stop_row1 = None
    stop_row2 = None

    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format(ringtable, start)
    start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = ringdb.execute( cmd ).fetchall()[-1]

    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format(ringtable, stop)
    start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = ringdb.execute( cmd ).fetchall()[0]

    # Determine the list of ODs that contain the rings

    ods = []
    if int(freq) < 100:
        query = ringdb.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == {}'.format(start_time1, stop_time2, freq) )
    else:
        query = ringdb.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == 100'.format(start_time1, stop_time2) )
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

        if nrow - first_row > nleft:
            last_row = first_row + nleft
        else:
            last_row = nrow

        nbuff = last_row - first_row

        if nbuff < 1: raise Exception( 'Empty read on OD {}: indices {} - {}, rows {} - {}, timestamps {} - {}'.format( od, start, stop, first_row, last_row, start_time1, stop_time2 ) )

        if eff_cache is None or effdir not in eff_cache or od not in eff_cache[ effdir ] or extname not in eff_cache[ effdir ][ od ]:

            if eff_cache is not None and effdir not in eff_cache: eff_cache[ effdir ] = {}
            if eff_cache is not None and od not in eff_cache[ effdir ]: eff_cache[ effdir ][ od ] = {}

            if extname.lower() in ['attitude', 'velocity']:
                pattern = effdir + '/{:04}/pointing*fits'.format(od, freq)
            else:
                pattern = effdir + '/{:04}/?{:03}*fits'.format(od, freq)
            try:
                fn = sorted( glob.glob( pattern ) )[-1]
            except:
                raise Exception( 'Error: failed to find a file to read matching: ' + pattern )

            h = pf.open( fn, 'readonly' )
            hdu = 1
            while extname not in h[hdu].header['extname'].strip().lower():
                hdu += 1
                if hdu == len(h): raise Exception('No HDU matching extname = {} in {}'.format(extname, fn))

            ncol = len( h[hdu].columns )

            if eff_cache is not None:
                # Cache the entire column
                if extname not in eff_cache[ effdir ][ od ]:
                    if ncol == 2:
                        temp = np.array( h[hdu].data.field(0) )
                    else:
                        temp = np.array( [ h[hdu].data.field(col).ravel() for col in range(ncol-1) ] )
                    if (debug > 3): print('Storing {:.2f} MB to eff_cache:{}:{}:{}'.format(temp.nbytes/2.0**20,effdir,od,extname)) # DEBUG
                    eff_cache[ effdir ][ od ][ extname ] = temp.copy()
                if ncol == 2:
                    dat = eff_cache[ effdir ][ od ][ extname ][ first_row:last_row ].copy()
                else:
                    dat = eff_cache[ effdir ][ od ][ extname ][ :, first_row:last_row ].copy()
                if (debug > 3): print('Retrieved {:.2f} MB from eff_cache:{}:{}:{}'.format(dat.nbytes/2.0**20,effdir,od,extname)) # DEBUG
            else:
                if ncol == 2:
                    dat = np.array( h[hdu].data.field(0)[first_row:last_row] )
                else:
                    dat = np.array( [ h[hdu].data.field(col)[first_row:last_row].ravel() for col in range(ncol-1) ] )

            flg = np.zeros( last_row-first_row, dtype=np.byte )

            if obtmask != 0:
                if eff_cache is not None:
                    if 'obtflag' not in eff_cache[ effdir ][ od ]:
                        # Cache the OBT flags if they are not already cached
                        temp = np.array(h[1].data.field(1), dtype=np.byte)
                        if (debug > 3): print('Storing {:.2f} MB to eff_cache:{}:{}:{}'.format(temp.nbytes/2.0**20,effdir,od,'obtflag')) # DEBUG
                        eff_cache[ effdir ][ od ][ 'obtflag' ] = temp.copy()
                    obtflg = eff_cache[ effdir ][ od ][ 'obtflag' ][first_row:last_row].copy()
                    if (debug > 3): print('Retrieved {:.2f} MB from eff_cache:{}:{}:{}'.format(obtflg.nbytes/2.0**20,effdir,od,'obtflag')) # DEBUG
                else:
                    obtflg = np.array(h[1].data.field(1)[first_row:last_row], dtype=np.byte)
                if obtmask > 0:
                    flg += obtflg & obtmask
                else:
                    flg += np.logical_not( obtflg & -obtmask )

            if flagmask != 0:
                if eff_cache is not None:
                    # Cache the detector flags
                    if extname+'flag' not in eff_cache[ effdir ][ od ]:
                        temp = np.array( h[hdu].data.field(ncol-1), dtype=np.byte )
                        if (debug > 3): print('Storing {:.2f} MB to eff_cache:{}:{}:{}'.format(temp.nbytes/2.0**20,effdir,od,extname+'flag')) # DEBUG
                        eff_cache[ effdir ][ od ][ extname+'flag' ] = temp.copy()
                    detflg = eff_cache[ effdir ][ od ][ extname+'flag' ][first_row:last_row].copy()
                    if (debug > 3): print('Retrieved {:.2f} MB from eff_cache:{}:{}:{}'.format(detflg.nbytes/2.0**20,effdir,od,extname+'flag')) # DEBUG
                else:
                    detflg = np.array(h[hdu].data.field(ncol-1)[first_row:last_row], dtype=np.byte)
                if flagmask > 0:
                    flg += detflg & flagmask
                else:
                    flg += np.logical_not( detflg & -flagmask )

            h.close()
        else:
            # get the requested TOI from the cache

            if len( np.shape( eff_cache[ effdir ][ od ][ extname ] ) ) == 1:
                ncol = 2
            else:
                ncol = np.shape( eff_cache[ effdir ][ od ][ extname ] )[0] + 1

            if ncol == 2:
                dat = eff_cache[ effdir ][ od ][ extname ][ first_row:last_row ].copy()
            else:
                dat = eff_cache[ effdir ][ od ][ extname ][ :, first_row:last_row ].copy()

            if (debug > 3): print('Retrieved {:.2f} MB from eff_cache:{}:{}:{}'.format(dat.nbytes/2.0**20,effdir,od,extname)) # DEBUG
            
            flg = np.zeros( last_row-first_row, dtype=np.byte )

            if obtmask != 0:
                obtflg = eff_cache[ effdir ][ od ][ 'obtflag' ][first_row:last_row].copy()

                if (debug > 3): print('Retrieved {:.2f} MB from eff_cache:{}:{}:{}'.format(obtflg.nbytes/2.0**20,effdir,od,'obtflag')) # DEBUG
                              
                if obtmask > 0:
                    flg += obtflg & obtmask
                else:
                    flg += np.logical_not( obtflg & -obtmask )

            if flagmask != 0:
                detflg = eff_cache[ effdir ][ od ][ extname+'flag' ][first_row:last_row].copy()

                if (debug > 3): print('Retrieved {:.2f} MB from eff_cache:{}:{}:{}'.format(detflg.nbytes/2.0**20,effdir,od,extname+'flag')) # DEBUG
                              
                if flagmask > 0:
                    flg += detflg & flagmask
                else:
                    flg += np.logical_not( detflg & -flagmask )


        data.append( dat )
        flag.append( flg )

        ods = ods[1:]
        first_row = 0

        nread += nbuff
        nleft -= nbuff

        if nleft == 0: break


    if len(data) > 0: data = np.hstack(data).T
    if len(flag) > 0: flag = np.hstack(flag).T

    data = np.array( data )
    flag = np.array( flag != 0 )

    if np.shape( flag )[-1] != stop - start:
        raise Exception('ERROR: inconsistent dimensions: shape(data) = ', np.shape(data), ', shape(flag) = ', np.shape(flag), ', stop-start = ', stop-start)

    if len(np.shape(data)) > 1: data = data.T # Conform to expected geometry

    if (debug > 3): print('Read {} samples from extension {}. Number of flagged samples = {} ({:.2f}%)'.format(n,extname,np.sum(flag),np.sum(flag)*100./n)) # DEBUG

    return (data, flag)


def write_eff(local_start, data, flags, globalfirst, local_offset, ringdb, ringdb_path, freq, effdir, extname, flagmask ):

    ringtable = ringdb_table_name(freq)

    # Convert to global indices

    start = globalfirst + local_offset + local_start
    stop = start + len(data)

    # Promote all input data to 2 dimensions to support multicolumn writing

    data2d = np.atleast_2d( data )

    if len(data2d[0]) != stop - start:
        raise Exception( 'stop-start = {} but len(data) = {}'.format( stop-start, len(data2d[0]) ) )

    start_time1 = None
    start_time2 = None
    stop_time1 = None
    stop_time2 = None
    start_index1 = None
    start_index2 = None
    stop_index1 = None
    stop_index2 = None
    start_row1 = None
    start_row2 = None
    stop_row1 = None
    stop_row2 = None

    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format(ringtable, start)
    start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = ringdb.execute( cmd ).fetchall()[-1]

    cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format(ringtable, stop)
    start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = ringdb.execute( cmd ).fetchall()[0]

    ods = []
    if freq < 100:
        query = ringdb.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == {}'.format(freq, start_time1, stop_time2) )
    else:
        query = ringdb.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == 100'.format(start_time1, stop_time2) )
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

        pattern = effdir + '/{:04}/?{:03}*fits'.format(od, freq)
        try:
            fn = sorted( glob.glob( pattern ) )[-1]
        except:
            raise Exception( 'Error: failed to find a file to update matching: ' + pattern )
        h = pf.open( fn, 'update' )
        hdu = 1
        while extname not in h[hdu].header['extname'].strip().lower():
            hdu += 1
            if hdu == len(h): raise Exception('No HDU matching extname = {} in {}'.format(extname, fn))

        if nrow - first_row > nleft:
            last_row = first_row + nleft
        else:
            last_row = nrow

        nwrite = last_row - first_row
        ncol = len( h[hdu].columns )
        ncol_data = len(data2d)

        if ncol-1 != ncol_data: raise Exception( 'Expected {} columns to write data but got {}.'.format(ncol-1,ncol_data) )

        if nwrite > 0:
            try:
                for col in range(len(data2d)):
                    h[hdu].data.field(col)[first_row:last_row] = data2d[col][offset:offset+nwrite]
                if flagmask >= 0:
                    h[hdu].data.field(ncol-1)[first_row:last_row] = np.array(flags[offset:offset+nwrite] != 0, dtype=np.byte) * flagmask
                else:
                    h[hdu].data.field(ncol-1)[first_row:last_row] = np.array(flags[offset:offset+nwrite] == 0, dtype=np.byte) * (-flagmask)
            except:
                raise Exception( 'Indexing error: fn = {}, start = {}, stop = {}, first_row = {}, last_row = {}, offset = {}, nwrite = {}, len(h[hdu].data.field(1)) = {}, len(data) = {}, flagmask = {}'.format( fn, start, stop, first_row, last_row, offset, nwrite, len(h[hdu].data.field(1)), len(data2d[0]), flagmask ) )
            offset += nwrite

        ods = ods[1:]
        first_row = 0
        result = h.flush()
        h.close()

        nwrote += nwrite
        nleft -= nwrite

        if nleft == 0: break

    return result


def ringdb_table_name(freq):
    if int(freq) < 100:
        return 'ring_times_{}'.format(self.freq)
    else:
        return 'ring_times_hfi'


def load_ringdb(path, mpicomm):
    """
    Load and broadcast the ring database.
    """

    itask = mpicomm.Get_rank()
    ntask = mpicomm.Get_size()

    # Read database to tempfile and broadcast

    tempfile = ''

    if itask == 0:
        conn = sqlite3.connect(path)
        tempfile = io.StringIO()
        for line in conn.iterdump():
            tempfile.write('{}\n'.format(line))
        conn.close()
        tempfile.seek( 0 )

    tempfile = mpicomm.bcast(tempfile, root=0)

    # Create a database in memory and import from tempfile

    ringdb = sqlite3.connect(':memory:')
    ringdb.cursor().executescript(tempfile.read())
    ringdb.commit()

    return ringdb


def load_RIMO(path, mpicomm):
    """
    Load and broadcast the reduced instrument model, a.k.a. focal plane database.
    """

    itask = mpicomm.Get_rank()
    ntask = mpicomm.Get_size()

    # Read database, parse and broadcast

    RIMO = {}

    if itask == 0:
        hdulist = pf.open( path, 'readonly' )
        detectors = hdulist[1].data.field('detector').ravel()
        phi_uvs = hdulist[1].data.field('phi_uv').ravel()
        theta_uvs = hdulist[1].data.field('theta_uv').ravel()
        psi_uvs = hdulist[1].data.field('psi_uv').ravel()
        psi_pols = hdulist[1].data.field('psi_pol').ravel()
        epsilons = hdulist[1].data.field('epsilon').ravel()
        fsamples = hdulist[1].data.field('f_samp').ravel()
        fknees = hdulist[1].data.field('f_knee').ravel()
        alphas = hdulist[1].data.field('alpha').ravel()
        nets = hdulist[1].data.field('net').ravel()

        for i in range(len(detectors)):
            quat1 = np.zeros(4)
            quat2 = np.zeros(4)
            phi = (phi_uvs[i])*degree
            theta = theta_uvs[i]*degree
            psi = (psi_uvs[i] + psi_pols[i])*degree - phi # Make sure we don't double count psi rotation already included in phi
            quat = np.zeros(4)
            # ZYZ conversion from http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
            # Note: The above document has the scalar part of the quaternion at first position but
            #       quaternionarray module has it at the end, we use the quaternionarray convention
            quat[3] =  np.cos(.5*theta) * np.cos(.5*(phi+psi)) # This is the scalar part
            quat[0] = -np.sin(.5*theta) * np.sin(.5*(phi-psi))
            quat[1] =  np.sin(.5*theta) * np.cos(.5*(phi-psi))
            quat[2] =  np.cos(.5*theta) * np.sin(.5*(phi+psi))
            # apply the bore sight rotation to the detector quaternion
            quat = qa.mult( spinrot, quat )
            RIMO[ detectors[i] ] = DetectorData( detectors[i], phi_uvs[i], theta_uvs[i], psi_uvs[i], psi_pols[i], epsilons[i], fsamples[i], fknees[i], alphas[i], nets[i], quat )

    RIMO = mpicomm.bcast(RIMO, root=0)

    return RIMO


def count_samples(ringdb, freq, obt_range, ring_range, od_range):
    """
    Query the ring database to determine which global samples the requested time ranges
    point to.  The ranges are checked in order:  OD, ring, then OBT.  If none are
    specified, the full data is selected.
    """

    ringtable = ringdb_table_name(freq)

    rings = []
    samples = 0
    global_start = 0.0
    global_first = 0

    select_range = None

    if od_range is not None:
        cmd = 'select od, start_time, stop_time from ods where od >= {} and od <= {} order by od'.format(od_range[0], od_range[1])
        ods = ringdb.execute( cmd ).fetchall()
        od1, start1, stop1 = ods[0]
        od2, start2, stop2 = ods[-1]
        select_range = (start1, stop2)

    if ring_range is not None:
        # Ring numbers are not recorded into the database. Simply get a list of the repointing maneuvers at the start of each pointing period
        cmd = 'select start_time, stop_time from {} where pointID_unique like "%-H%" order by start_time'.format( ringtable )
        ringlist = ringdb.execute( cmd ).fetchall()
        try:
            start1, stop1 = ringlist[ ring_range[0] ]
            start2, stop2 = ringlist[ ring_range[1]+1 ]
        except:
            raise Exception( 'Failed to determine the ring span {} from the datababase'.format( ring_range ) )
        select_range = (start1, start2 - 1.0 )

    if obt_range is not None:
        select_range = obt_range

    cmd = ''

    if select_range is not None:
        start, stop = select_range
        cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_time <= {} and stop_time >= {} order by start_index'.format( ringtable, stop, start )
    else:
        # no span specified, use all available data
        # This first version of the query will include the repointing maneuvers in the definitions of the science scans immediately before them.
        #cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where pointID_unique like "%S%" or pointID_unique like "%O%" order by start_index'.format( ringtable )
        # This second version of the query will list the repointing maneuvers as separate intervals
        cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} order by start_index'.format( ringtable )
        # FIXME: a third option would be to include the repointing maneuvers in the following science scans (MOC definition) but this would require extra processing of the query results. 
    
    intervals = ringdb.execute( cmd ).fetchall()

    if len(intervals) < 1: raise Exception( 'Warning: failed to find any intervals with the query: {}'.format( cmd ) )

    start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = intervals[0]
    start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = intervals[-1]

    global_start = start_time1
    global_first = start_index1
    samples = stop_index2 - start_index1 + 1

    for interval in intervals:
        start_time, stop_time, start_index, stop_index, start_row, stop_row = interval
        rings.append( Interval(start=start_time, stop=stop_time, first=start_index, last=stop_index) )

    return global_start, global_first, samples, rings


def bolos_by_type(type):
    vals = {
        'P100' : ["100-1a","100-1b","100-2a","100-2b","100-3a","100-3b","100-4a","100-4b"],
        'P143' : ["143-1a","143-1b","143-2a","143-2b","143-3a","143-3b","143-4a","143-4b"],
        'S143' : ["143-5","143-6","143-7","143-8"],
        'P217' : ["217-5a","217-5b","217-6a","217-6b","217-7a","217-7b","217-8a","217-8b"],
        'S217' : ["217-1","217-2","217-3","217-4"],
        'P353' : ["353-3a","353-3b","353-4a","353-4b","353-5a","353-5b","353-6a","353-6b"],
        'S353' : ["353-1","353-2","353-7","353-8"],
        'S545' : ["545-1","545-2","545-3","545-4"],
        'S857' : ["857-1","857-2","857-3","857-4"],
        'SDRK' : ["Dark-1","Dark-2"]
    }
    return vals[type]


def bolos_by_p(type):
    vals = {
    'P' : ["100-1a","100-1b","100-2a","100-2b","100-3a","100-3b","100-4a","100-4b","143-1a","143-1b","143-2a","143-2b","143-3a","143-3b","143-4a","143-4b","217-5a","217-5b","217-6a","217-6b","217-7a","217-7b","217-8a","217-8b","353-3a","353-3b","353-4a","353-4b","353-5a","353-5b","353-6a","353-6b"],
    'S' : ["143-5","143-6","143-7","143-8","217-1","217-2","217-3","217-4","353-1","353-2","353-7","353-8","545-1","545-2","545-3","545-4","857-1","857-2","857-3","857-4","Dark-1","Dark-2"]
    }
    return vals[type]


def bolo_types():
    return ["P100","P143","S143","P217","S217","P353","S353","S545","S857","SDRK"]


def bolos_by_freq(freq):
    ret = []
    for t in ['P', 'S']:
        key = '{}{}'.format(t, freq)
        if key in bolo_types():
            ret.extend(bolos_by_type(key))
    return ret


def bolos():
    return [
        '100-1a',
        '100-1b',
        '143-1a',
        '143-1b',
        '217-1' ,
        '353-1' ,
        '143-5' ,
        '217-5a',
        '217-5b',
        '353-2' ,
        '545-1' ,
        'Dark-1',
        '100-2a',
        '100-2b',
        '217-2' ,
        '353-3a',
        '353-3b',
        '857-1' ,
        '143-2a',
        '143-2b',
        '353-4a',
        '353-4b',
        '545-2' ,
        '857-2' ,
        '100-3a',
        '100-3b',
        '143-6',
        '217-6a',
        '217-6b',
        '353-7',
        '143-3a',
        '143-3b',
        '217-3',
        '353-5a',
        '353-5b',
        '545-3',
        '143-7' ,
        '217-7a',
        '217-7b',
        '353-6a',
        '353-6b',
        '857-3' ,
        '143-8' ,
        '217-8a',
        '217-8b',
        '545-4' ,
        '857-4' ,
        'Dark-2',
        '100-4a',
        '100-4b',
        '143-4a',
        '143-4b',
        '217-4' ,
        '353-8'
    ]


def bolo_to_BC(bolo):
    vals = {
        '100-1a' : "00",
        '100-1b' : "01",
        '143-1a' : "02",
        '143-1b' : "03",
        '217-1' : "04",
        '353-1' : "05",
        '143-5' : "10",
        '217-5a' : "11",
        '217-5b' : "12",
        '353-2' : "13",
        '545-1' : "14",
        'Dark-1' : "15",
        '100-2a' : "20",
        '100-2b' : "21",
        '217-2' : "22",
        '353-3a' : "23",
        '353-3b' : "24",
        '857-1' : "25",
        '143-2a' : "30",
        '143-2b' : "31",
        '353-4a' : "32",
        '353-4b' : "33",
        '545-2' : "34",
        '857-2' : "35",
        '100-3a' : "40",
        '100-3b' : "41",
        '143-6' : "42",
        '217-6a' : "43",
        '217-6b' : "44",
        '353-7' : "45",
        '143-3a' : "50",
        '143-3b' : "51",
        '217-3' : "52",
        '353-5a' : "53",
        '353-5b' : "54",
        '545-3' : "55",
        '143-7' : "60",
        '217-7a' : "61",
        '217-7b' : "62",
        '353-6a' : "63",
        '353-6b' : "64",
        '857-3' : "65",
        '143-8' : "70",
        '217-8a': "71",
        '217-8b': "72",
        '545-4' : "73",
        '857-4' : "74",
        'Dark-2' : "75",
        '100-4a' : "80",
        '100-4b' : "81",
        '143-4a' : "82",
        '143-4b' : "83",
        '217-4' : "84",
        '353-8' : "85"
    }
    return vals[bolo]


def bolo_to_pnt(bolo):
    vals = {
        '100-1a' : "00_100_1a",
        '100-1b' : "01_100_1b",
        '143-1a' : "02_143_1a",
        '143-1b' : "03_143_1b",
        '217-1' : "04_217_1",
        '353-1' : "05_353_1",
        '143-5' : "10_143_5",
        '217-5a' : "11_217_5a",
        '217-5b' : "12_217_5b",
        '353-2' : "13_353_2",
        '545-1' : "14_545_1",
        'Dark-1' : "15_Dark1",
        '100-2a' : "20_100_2a",
        '100-2b' : "21_100_2b",
        '217-2' : "22_217_2",
        '353-3a' : "23_353_3a",
        '353-3b' : "24_353_3b",
        '857-1' : "25_857_1",
        '143-2a' : "30_143_2a",
        '143-2b' : "31_143_2b",
        '353-4a' : "32_353_4a",
        '353-4b' : "33_353_4b",
        '545-2' : "34_545_2",
        '857-2' : "35_857_2",
        '100-3a' : "40_100_3a",
        '100-3b' : "41_100_3b",
        '143-6' : "42_143_6",
        '217-6a' : "43_217_6a",
        '217-6b' : "44_217_6b",
        '353-7' : "45_353_7",
        '143-3a' : "50_143_3a",
        '143-3b' : "51_143_3b",
        '217-3' : "52_217_3",
        '353-5a' : "53_353_5a",
        '353-5b' : "54_353_5b",
        '545-3' : "55_545_3",
        '143-7' : "60_143_7",
        '217-7a' : "61_217_7a",
        '217-7b' : "62_217_7b",
        '353-6a' : "63_353_6a",
        '353-6b' : "64_353_6b",
        '857-3' : "65_857_3",
        '143-8' : "70_143_8",
        '217-8a': "71_217_8a",
        '217-8b': "72_217_8b",
        '545-4' : "73_545_4",
        '857-4' : "74_857_4",
        'Dark-2' : "75_Dark2",
        '100-4a' : "80_100_4a",
        '100-4b' : "81_100_4b",
        '143-4a' : "82_143_4a",
        '143-4b' : "83_143_4b",
        '217-4' : "84_217_4",
        '353-8' : "85_353_8"
    }
    return vals[bolo]


def bolo_to_ADU(bolo):
    return "HFI_" + bolo_to_BC(bolo) + "_C"

