# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from scipy.constants import degree, arcmin, arcsec, c

import quaternionarray as qa

import healpy as hp

from toast.dist import distribute_det_samples

from toast.tod import TOD

from .utilities import load_ringdb, count_samples, read_eff, write_eff, load_RIMO


xaxis = np.array( [1,0,0], dtype=np.float64 )
yaxis = np.array( [0,1,0], dtype=np.float64 )
zaxis = np.array( [0,0,1], dtype=np.float64 )

spinangle = 85.0 * degree

spinrot = qa.rotation( yaxis, np.pi/2 - spinangle )

cinv = 1e3 / c # Inverse light speed in km / s ( the assumed unit for velocity )


class Exchange(TOD):
    """
    Provide pointing and detector timestreams for Planck Exchange File Format data.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
                  detector.
        detectors (list): list of names to use for the detectors. Must match the names in the FITS HDUs.
        ringdb: Path to an SQLite database defining ring boundaries.
        effdir: directory containing the exchange files
        obt_range: data span in TAI seconds, overrides ring_range
        ring_range: data span in pointing periods, overrides od_range
        od_range: data span in operational days
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, fn_ringdb=None, effdir=None, obt_range=None, ring_range=None, od_range=None, freq=None, RIMO=None, coord='G', mode='THETAPHIPSI', deaberrate=True, order='RING', nside=2048, obtmask=0, flagmask=0, bufsize=100000):

        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        if fn_ringdb is None:
            raise ValueError('You must provide a path to the ring database')

        if effdir is None:
            raise ValueError('You must provide a path to the exchange files')
        
        if freq is None:
            raise ValueError('You must specify the frequency to run on')

        if RIMO is None:
            raise ValueError('You must specify which RIMO to use')

        self.ringdb_path = fn_ringdb
        self.ringdb = load_ringdb( self.ringdb_path, mpicomm )

        self.RIMO_path = RIMO
        self.RIMO = load_RIMO( self.RIMO_path, mpicomm )

        self.bufsize = bufsize

        self.freq = freq
        if self.freq < 100:
            self.ringtable = 'ring_times_{}'.format(self.freq)
        else:
            self.ringtable = 'ring_times_hfi'

        self.coord = coord
        if mode == 'QUATERNION':
            self.ncol = 4
        elif mode == 'THETAPHIPSI':
            self.ncol = 3
        elif mode == 'IQU':
            self.ncol = 3
        self.mode = mode
        self.deaberrate = deaberrate
        self.order = order
        self.nside = nside

        self._offset, self._nsamp, self._sizes = count_samples( self.ringdb, self.ringtable, obt_range, ring_range, od_range )
        
        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=detectors, samples=self._nsamp, sizes=self._sizes)
        

        self._dets = detectors

        self.effdir = effdir
        self.obtmask = obtmask
        self.flagmask = flagmask

        self.satobtmask = 1
        self.satquatmask = 1
        self.satvelmask = 1

        if self.coord == 'E':
            self.coordmatrix = None
            self.coordquat = None
        else:
            self.coordmatrix, do_conv, normcoord = hp.rotator.get_coordconv_matrix( ['E', self.coord] )
            self.coordquat = qa.from_rotmat( self.coordmatrix )
    

    def _get(self, detector, flavor, local_start, n):

        data, flag = read_eff( detector, local_start, n, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, detector.lower(), self.obtmask, self.flagmask )

        return (data, flag)


    def _put(self, detector, flavor, local_start, data, flags):

        result = write_eff( detector, local_start, data, flags, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, detector.lower(), self.flagmask )
        # FIXME: should we check the result here?
        # We should NOT return result, since the return needs to be empty
        return


    def _get_pntg(self, detector, local_start, n):

        epsilon = self.RIMO[ detector ].epsilon
        eta = (1 - epsilon) / (1 + epsilon)
        detquat = self.RIMO[ detector ].quat        
        #detvec = qa.rotate( detquat, zaxis ) # DEBUG
        #orivec = qa.rotate( detquat, xaxis ) # DEBUG

        #print('detvec = {}'.format( detvec )) # DEBUG
        #print('orivec = {}'.format( orivec )) # DEBUG

        # Get the satellite attitude

        satquat, flag = read_eff( detector, local_start, n, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, 'attitude', self.satobtmask, self.satquatmask )
        satquat = satquat.T.copy()

        #satquat = qa.mult( satquat, spinrot )

        #print('satquat = {}'.format(satquat))

        # Get the satellite velocity

        if self.deaberrate:        
            satvel, flag2 = read_eff( detector, local_start, n, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, 'velocity', self.satobtmask, self.satvelmask )
            flag |= flag2
            satvel = satvel.T.copy()
            
            #print('satvel = {}'.format(satvel))

        # Initialize output

        out = np.zeros( [n,self.ncol], dtype=np.float64 )

        # Rotate into detector frame and convert to desired format

        istart = 0

        while istart < n:
            
            istop = min( istart+self.bufsize, n )
            ind = slice( istart, istop )
            quats = qa.mult( qa.norm( satquat[ind] ), detquat )

            istart = istop

            if self.deaberrate:
                # Correct for aberration
                vec = qa.rotate( quats[ind], zaxis )
                abvec = np.cross( vec, satvel[ind] )
                lens = np.linalg.norm( abvec, axis=1 )
                ang = lens * cinv
                abvec /= np.tile( lens, (3,1) ).T # Normalize for direction
                abquat = qa.rotation( abvec, -ang )
                quats = qa.mult( abquat, quats )

            if self.coordquat is not None:
                quats = qa.mult( self.coordquat, quats )

            if self.mode == 'QUATERNION':
                out[ind] = quats
            elif self.mode == 'THETAPHIPSI' or self.mode == 'IQU':
                vec_dir = qa.rotate( quats, zaxis )
                theta, phi = hp.vec2ang( vec_dir )

                vec_dir = vec_dir.T.copy()
                vec_orient = qa.rotate( quats, xaxis ).T                
                
                ypa = vec_orient[0]*vec_dir[1] - vec_orient[1]*vec_dir[0]
                xpa = -vec_orient[0]*vec_dir[2]*vec_dir[0] - vec_orient[1]*vec_dir[2]*vec_dir[1] + vec_orient[2]*(vec_dir[0]**2 + vec_dir[1]**2)

                psi = np.arctan2( ypa, xpa )

                if self.mode == 'THETAPHIPSI':                    
                    out[ind,0] = theta
                    out[ind,1] = phi
                    out[ind,2] = psi
                elif self.mode == 'IQU':
                    out[ind,0] = hp.vec2pix( self.nside, *(vec.T), nest=(self.order=='NEST') )
                    out[ind,1] = eta * np.cos( 2 * psi )
                    out[ind,2] = eta * np.sin( 2 * psi )

        # Return
        return (out, flag)
    

    def _put_pntg(self, detector, start, data, flags):

        result = write_eff( detector, local_start, data, flags, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, 'attitude', self.flagmask )
        # FIXME: should we check the result here?
        # We should NOT return result, since the return needs to be empty
        return


    def _get_times(self, start, n):
        data, flag = read_eff( detector, local_start, n, self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, 'OBT', 0, 0 )
        return data


    def _put_times(self, start, stamps):
        result = write_eff( detector, local_start, stamps, np.zeros(stamps.shape[0], dtype=np.uint8), self._offset, self.local_samples, self.ringtable, self.ringdb, self.ringdb_path, self.freq, self.effdir, 'OBT', 0 )
        return

