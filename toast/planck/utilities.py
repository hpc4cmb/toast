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

#from ..tod import TOD

from collections import namedtuple

from scipy.constants import degree, arcmin, arcsec, c

import quaternionarray as qa

xaxis, yaxis, zaxis = np.eye( 3 )

spinangle = 85.0 * degree

spinrot = qa.rotation( yaxis, np.pi/2 - spinangle )

# define a named tuples to contain detector data and RIMO in a standardized format

DetectorData = namedtuple( 'DetectorData', 'detector phi_uv theta_uv psi_uv psi_pol epsilon fsample fknee alpha net quat' )

RIMOstore = namedtuple( 'RIMOstore', 'path RIMO' )

storedRIMO = None

def read_eff(detector, local_start, n, offset, local_samples, ringtable, ringdb, ringdb_path, freq, effdir, extname, obtmask, flagmask ):

    # Convert to global indices

    start = offset + local_samples[0] + local_start
    stop = start + n

    # Start, stop interval follows Python conventions. (stop-start) samples are read starting at "start". Sample at "stop" is omitted

    # Determine first and last ring to read

    ntry = 10
    try:
        for itry in range(ntry):
            try:
                cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format(ringtable, start)
                start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = ringdb.execute( cmd ).fetchall()[-1]
            except Exception as e:
                if itry < ntry: continue
                raise Exception('Failed to query ring start and stop times from {} (1/2). Failed query: {}. Error: {}'.format(ringdb_path, cmd, e))
            try:
                cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format(ringtable, stop)
                start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = ringdb.execute( cmd ).fetchall()[0]
            except Exception as e:
                if itry < ntry: continue
                raise Exception('Failed to query ring start and stop times from {} (2/2). Failed query: {}. Error: '.format(ringdb_path, cmd, e))
    except Exception as e:
        raise Exception( 'TODPlanckEFF._get: Giving up after {} tries to get source ODs: {}'.format(ntry,e) )

    # Determine the list of ODs that contain the rings

    ods = []
    if freq < 100:
        query = ringdb.execute( 'select eff_od, nrow from eff_files where stop_time >= {} and start_time <= {} and freq == {}'.format(freq, start_time1, stop_time2) )
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

        if nrow - first_row > nleft:
            last_row = first_row + nleft
        else:
            last_row = nrow

        nbuff = last_row - first_row

        ncol = len( h[hdu].columns )

        #tme = np.array( h[1].data.field(0)[first_row:last_row] ) # DEBUG
        #print('Reading EFF for:') # DEBUG
        #for t in tme: print('{:.8f}'.format(t*1e-9)) # DEBUG
        
        if ncol == 2:
            dat = np.array( h[hdu].data.field(0)[first_row:last_row] )
        else:
            dat = np.array( [ h[hdu].data.field(col)[first_row:last_row].ravel() for col in range(ncol-1) ] )
        data.append( dat )

        flg = np.zeros( last_row-first_row, dtype=np.byte )
        if obtmask != 0:
            if obtmask > 0:
                flg += np.array(h[1].data.field(1)[first_row:last_row], dtype=np.byte) & obtmask
            else:
                flg += np.logical_not( np.array(h[1].data.field(1)[first_row:last_row], dtype=np.byte) & -obtmask )
        if flagmask != 0:
            if flagmask > 0:
                flg += np.array(h[hdu].data.field(ncol-1)[first_row:last_row], dtype=np.byte) & flagmask
            else:
                flg += np.logical_not( np.array(h[hdu].data.field(ncol-1)[first_row:last_row], dtype=np.byte) & -flagmask )
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

    if len(np.shape(data)) > 1: data = data.T # Conform to expected geometry            

    return (data, flag)


def write_eff( detector, local_start, data, flags, offset, local_samples, ringtable, ringdb, ringdb_path, freq, effdir, extname, flagmask ):

    # Convert to global indices

    start = offset + local_samples[0] + local_start
    stop = start + len(data)

    # Promote all input data to 2 dimensions to support multicolumn writing

    data2d = np.atleast_2d( data )

    if len(data2d[0]) != stop - start:
        raise Exception( 'TODPlanckEFF: stop-start = {} but len(data) = {}'.format( stop-start, len(data2d[0]) ) )

    ntry = 10
    try:
        for itry in range(10):
            try:
                cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} order by start_index'.format(ringtable, start)
                start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = ringdb.execute( cmd ).fetchall()[-1]
            except Exception as e:
                if itry < ntry: continue
                raise Exception('Failed to query ring start and stop times from {} (1/2). Failed query: {}. Error: {}'.format(ringdb_path, cmd, e))

            try:
                cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where stop_index >= {} order by start_index'.format(ringtable, stop)
                start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = ringdb.execute( cmd ).fetchall()[0]
            except Exception as e:
                if itry < ntry: continue
                raise Exception('Failed to query ring start and stop times from {} (2/2). Failed query: {}. Error: '.format(ringdb_path, cmd, e))
    except Exception as e:
        raise Exception( 'TODPlanckEFF._put: Giving up after {} tries to get target ODs: {}'.format(ntry,e) )

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

        if ncol-1 != ncol_data: raise Exception( 'TODPlanckEFF._put: Expected {} columns to write data but got {}.'.format(ncol-1,ncol_data) )

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


def load_ringdb(path, mpicomm):
    """
    Load and broadcast the ring database.
    """

    itask = mpicomm.Get_rank()
    ntask = mpicomm.Get_size()

    # Read database to tempfile and broadcast

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

    global storedRIMO

    if storedRIMO != None and storedRIMO.path == path:
        return storedRIMO.RIMO

    itask = mpicomm.Get_rank()
    ntask = mpicomm.Get_size()

    # Read database, parse and broadcast

    if itask == 0:

        RIMO = {}

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

    storedRIMO = RIMOstore( path, RIMO )

    return RIMO


def count_samples(ringdb, ringtable, obt_range, ring_range, od_range):
    """
    Query the ring database to determine which global samples the requested sample indices point to.
    """

    if obt_range is not None:
        start, stop = obt_range

        cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where start_index <= {} and stop_index >= {} order by start_index'.format( ringtable, stop, start )

        intervals = ringdb.execute( cmd ).fetchall()
        
        start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = intervals[0]

        start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = intervals[-1]
        
        sizes = []
        for interval in intervals:
            start_time, stop_time, start_index, stop_index, start_row, stop_row = interval
            sizes.append( stop_index - start_index + 1 )

        offset = start_index1
        samples = stop_index2 - start_index1

        return offset, samples, sizes
    elif ring_range is not None:
        raise Exception('Ring span selection not yet implemented')
    elif od_range is not None:
        raise Exception('OD span selection not yet implemented')
    else:
        # no span specified, use all available data

        # This first version of the query will include the repointing maneuvers in the definitions of the science scans immediately before them.
        #cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} where pointID_unique like "%S%" or pointID_unique like "%O%" order by start_index'.format( ringtable )
        # This second version of the query will list the repointing maneuvers as separate intervals
        cmd = 'select start_time, stop_time, start_index, stop_index, start_row, stop_row from {} order by start_index'.format( ringtable )
        # FIXME: a third option would be to include the repointing maneuvers in the following science scans (MOC definition) but this would require extra processing of the query results. 

        intervals = ringdb.execute( cmd ).fetchall()

        start_time1, stop_time1, start_index1, stop_index1, start_row1, stop_row1 = intervals[0]

        start_time2, stop_time2, start_index2, stop_index2, start_row2, stop_row2 = intervals[-1]

        sizes = []
        for interval in intervals:
            start_time, stop_time, start_index, stop_index, start_row, stop_row = interval
            sizes.append( stop_index - start_index + 1 )

        offset = start_index1
        samples = stop_index2 - start_index1

        return offset, samples, sizes
