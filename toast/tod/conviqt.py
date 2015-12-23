# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import ctypes as ct
from ctypes.util import find_library

import unittest

import numpy as np
import numpy.ctypeslib as npc

import quaternionarray as qa

from ..dist import Comm, Data
from ..operator import Operator
from ..tod import TOD
from ..tod import Interval



libconviqt = ct.CDLL('libconviqt')

# Beam functions

libconviqt.conviqt_beam_new.restype = ct.c_void_p
libconviqt.conviqt_beam_new.argtypes = []

libconviqt.conviqt_beam_del.restype = ct.c_int
libconviqt.conviqt_beam_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_beam_read.restype = ct.c_int
libconviqt.conviqt_beam_read.argtypes = [
    ct.c_void_p,
    ct.c_long,
    ct.c_long,
    ct.c_byte,
    ct.c_char_p,
    ct.c_int
]

# Sky functions

libconviqt.conviqt_sky_new.restype = ct.c_void_p
libconviqt.conviqt_sky_new.argtypes = []

libconviqt.conviqt_sky_del.restype = ct.c_int
libconviqt.conviqt_sky_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_sky_read.restype = ct.c_int
libconviqt.conviqt_sky_read.argtypes = [
    ct.c_void_p,
    ct.c_long,
    ct.c_byte,
    ct.c_char_p,
    ct.c_double,
    ct.c_int
]

# Detector functions

libconviqt.conviqt_detector_new.restype = ct.c_void_p
libconviqt.conviqt_detector_new.argtypes = []

libconviqt.conviqt_detector_new_with_id.restype = ct.c_void_p
libconviqt.conviqt_detector_new_with_id.argtypes = [ct.c_char_p]

libconviqt.conviqt_detector_del.restype = ct.c_int
libconviqt.conviqt_detector_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_detector_set_epsilon.restype = ct.c_int
libconviqt.conviqt_detector_set_epsilon.argtypes = [
    ct.c_void_p,
    ct.c_double
]

libconviqt.conviqt_detector_get_epsilon.restype = ct.c_int
libconviqt.conviqt_detector_get_epsilon.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_double)
]

libconviqt.conviqt_detector_get_id.restype = ct.c_int
libconviqt.conviqt_detector_get_id.argtypes = [
    ct.c_void_p,
    ct.c_char_p
]

# Pointing functions

libconviqt.conviqt_pointing_new.restype = ct.c_void_p
libconviqt.conviqt_pointing_new.argtypes = []

libconviqt.conviqt_pointing_del.restype = ct.c_int
libconviqt.conviqt_pointing_del.argtypes = [ct.c_void_p]

libconviqt.conviqt_pointing_alloc.restype = ct.c_int
libconviqt.conviqt_pointing_alloc.argtypes = [
    ct.c_void_p,
    ct.c_long
]

libconviqt.conviqt_pointing_data.restype = ct.POINTER(ct.c_double)
libconviqt.conviqt_pointing_data.argtypes = [ct.c_void_p]



class OpSimConviqt(Operator):
    """
    Operator which uses libconviqt to generate beam-convolved timestreams.

    This passes through each observation and ...

    Args:
        
    """

    def __init__(self, lmax, beamlmax, beammmax, skyfile, detectordata, pol=True, fwhm=4.0, nbetafac=10, mcsamples=0, lmaxout=32, order=3, comm=MPI.COMM_WORLD, flavor=None):
        """
        Set up on-the-fly signal convolution. Inputs:
        lmax : maximum ell to do the convolution to
        beamlmax : maximum ell beams are provided in
        beammmax : maximum m to convolve to
        skyfile : path to spherical harmonic expansion of the sky to convolve [Healpix FITS]
        detectordata : list of (detector_name, detector_beam_file, epsilon) tuples
        pol(True) : boolean to determine if polarized simulation is needed
        fwhm(5.0) : width of a symmetric gaussian beam [in arcmin] already present in the skyfile (will be deconvolved away).
        nbetafac(10) : conviqt resolution parameter (expert mode)
        mcsamples(0) : reserved input for future Monte Carlo mode
        lmaxout(32) : 
        order(3) : 
        comm(MPI.COMM_WORLD) : communicator to use in loading the inputs
        """
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._lmax = lmax
        self._beamlmax = beamlmax
        self._beammmax = beammax
        self._skyfile = skyfile
        self._detectordata = detectordata
        self._pol = pol
        self._fwhm = fwhm
        self._nbetafac = nbetafac
        self._mcsamples = mcsamples
        self._lmaxout = lmaxout
        self._order = order

        self._flavor = flavor
        
        self._sky = libconviqt.conviqt_sky_new()
        # FIXME : check the latest on passing the communicator to C++
        err = libconviqt.conviqt_sky_read( self._sky, self._lmax, self._pol, self._skyfile, self._fwhm, comm.py2f())
        if err != 0: raise Exception('Failed to load ' + self._skyfile)

        self._beams = {}
        self._detectors = {}
        for det_id, beamfile, epsilon in self._detectordata:
            beam = libconviqt.conviqt_beam_new()
            err = libconviqt.conviqt_beam_read(beam, self._beamlmax, self._beammmax, self._pol, beamfile, comm.py2f())
            if err != 0: raise Exception('Failed to load ' + beamfile)
            self._beams[ det_id ] = beam
            detector = libconviqt.conviqt_detector_new_with_id(det_id)
            libconviqt.conviqt_detector_set_epsilon(detector, epsilon)
            self._detectors[ det_id ] = detector


    def __del__(self):

        try:
            libconviqt.conviqt_sky_del( self._sky )
        except:
            pass

        for beam in self._beams.values():
            try:
                libconviqt.conviqt_beam_del( beam )
            except:
                pass

        for detector in self._detectors.values():
            try:
                libconviqt.conviqt_detector_del( detector )
            except:
                pass


        

    def exec(self, data):
        """
        Calling exec will perform the convolution over the communicator, one detector at a time.
        All MPI tasks must have the same list of detectors.
        """
        # the two-level pytoast communicator
        #comm = data.comm
        # the global communicator
        #cworld = comm.comm_world
        # the communicator within the group
        #cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        #crank = comm.comm_rank
        comm = data.comm.mpicomm # Check how to pass the communicator correctly

        zaxis = np.array([0,0,1], dtype=np.float64)

        for obs in data.obs:
            tod = obs['tod']
            intrvl = obs['intervals']

            for det in tod.local_dets:
                # We need the three pointing angles to describe the pointing. read_pntg returns the attitude quaternions.
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)

                pdata = pdata.reshape(-1,4).copy()
                pdata[ pflags != 0 ] = nullquat
                
                vec_dir = qa.rotate( pdata, zaxis ).T.copy()
                
                theta, phi = hp.vec2dir(*vec_dir)
                theta[ pflags != 0 ] = 0
                phi[ pflags != 0 ] = 0

                vec_orient = qa.rotate( pdata, xaxis ).T.copy()

                ypa = vec_orient[0]*vec_dir[1] - vec_orient[1]*vec_dir[0]
                xpa = -vec_dir[2]*(vec_orient[0]*vec_dir[0] + vec_orient[1]*vec_dir[1]) + vec_orient[2]*(vec_dir[0]**2 + vec_dir[1]**2)

                psi = np.arctan2(ypa, xpa)

                # Is the psi angle in Pxx or Dxx? Pxx will include the detector polarization angle, Dxx will not.

                pnt = libconviqt.conviqt_pointing_new()

                err = libconviqt.conviqt_pointing_alloc( pnt, tod.local_samples*5)
                if err != 0: raise Exception('Failed to allocate pointing array')

                ppnt = libconviqt.conviqt_pointing_data(pnt)

                for row in range(tod.local_samples):
                    ppnt[row*5 + 0] = phi[row]
                    ppnt[row*5 + 1] = theta[row]
                    ppnt[row*5 + 2] = psi[row]
                    ppnt[row*5 + 3] = 0
                    ppnt[row*5 + 4] = row

                convolver = libconviqt.conviqt_convolver_new(self._sky, self._beams[det], self._detectors[det], self._pol, self._lmax, self._beammmax, self._nbetafac, self._mcsamples, self._lmaxout, self._order, comm )
                if convolver == 0: raise Exception( "Failed to instantiate convolver" )

                err = libconviqt.conviqt_convolver_convolve( convolver, pnt )
                if err != 0: raise Exception( 'Convolution FAILED!' )

                convolved_data = np.zeros(tod.local_samples)
                for row in range(tod.local_samples):
                    convolved_data[row] = ppnt[row*5+3]

                libconviqt.conviqt_convolver_del(convolver)

                tod.write(detector=det, flavor=self._flavor, local_start=0, data=convolved_data, flags=pflags)

                libconviqt.conviqt_pointing_del(pnt)

        return



# This is copied from the libconviqt python test code,
# as a temporary usage example while implementing the operator
# above...


# lmax = 32
# beamlmax = lmax
# beammmax = 4
# pol = 0
# fwhm = 4.
# beamfile = '../data/mb_lfi_30_27_x_rescaled.alm'
# skyfile = '../data/slm.fits'
# det_id = 'LFITEST'

# ntheta = 3
# nphi = 3
# npsi = 3
# nsamp = ntheta*nphi*npsi

# nbetafac = 10
# mcsamples = 0
# lmaxout = 32
# order = 3


# beam = libconviqt.conviqt_beam_new()
# err = libconviqt.conviqt_beam_read( beam, ctypes.c_long(beamlmax), ctypes.c_long(beammmax), ctypes.c_byte(pol), beamfile, comm.py2f()  )
# if err != 0: raise Exception( 'Failed to load ' + beamfile )

# sky = libconviqt.conviqt_sky_new()
# err = libconviqt.conviqt_sky_read( sky, ctypes.c_long(lmax), ctypes.c_byte(pol), skyfile, ctypes.c_double(fwhm), comm.py2f()  )
# if err != 0: raise Exception( 'Failed to load ' + skyfile )

# detector = libconviqt.conviqt_detector_new_with_id( det_id )

# s = ctypes.create_string_buffer( 255 )
# libconviqt.conviqt_detector_get_id( detector, s )
# print 'detector name = ' + s.value
# epsilon = 1.32495160e-04
# libconviqt.conviqt_detector_set_epsilon( detector, ctypes.c_double(epsilon) )
# eps = ctypes.c_double( 0. )
# libconviqt.conviqt_detector_get_epsilon( detector, ctypes.byref(eps) )
# print 'epsilon = ',eps.value

# print 'Allocating pointing array'

# pnt = libconviqt.conviqt_pointing_new()
# err = libconviqt.conviqt_pointing_alloc( pnt, ctypes.c_long(nsamp*5) )
# if err != 0: raise Exception( 'Failed to allocate pointing array' )
# libconviqt.conviqt_pointing_data.restype = ctypes.POINTER( ctypes.c_double )
# ppnt = libconviqt.conviqt_pointing_data( pnt )

# print 'Populating pointing array'

# row = 0
# for theta in np.arange(ntheta) * np.pi / ntheta:
#     for phi in np.arange(nphi) * 2 * np.pi / nphi:
#         for psi in np.arange(npsi) * np.pi / npsi:
#             ppnt[ row*5 + 0 ] = phi
#             ppnt[ row*5 + 1 ] = theta
#             ppnt[ row*5 + 2 ] = psi
#             ppnt[ row*5 + 3 ] = 0
#             ppnt[ row*5 + 4 ] = row
#             row += 1

# for i in range(10):
#     print 'ppnt[',i,'] = ',ppnt[i]

# print 'Creating convolver'

# convolver = libconviqt.conviqt_convolver_new( sky, beam, detector, ctypes.c_byte(pol), ctypes.c_long(lmax), ctypes.c_long(beammmax), ctypes.c_long(nbetafac), ctypes.c_long(mcsamples), ctypes.c_long(lmaxout), ctypes.c_long(order), comm.py2f() )
# if convolver == 0: raise Exception( "Failed to instantiate convolver" );

# print 'Convolving data'

# err = libconviqt.conviqt_convolver_convolve( convolver, pnt )
# if err != 0: raise Exception( 'Convolution FAILED!' )

# # The pointer to the data will have changed during the convolution call ...

# ppnt = libconviqt.conviqt_pointing_data( pnt )

# if itask == 0:

#     print 'Convolved TOD: '
#     for row in np.arange( nsamp ):
#         print ppnt[row*5+4], ' ', ppnt[row*5+3]

#     if pol:
#         if np.abs( ppnt[ 0*5+3] -  0.8546349819096275 ) > 1e-6: raise Exception( "Row 0 should be 0.8546349819096275, not " + str(ppnt[ 0*5+3]) )
#         if np.abs( ppnt[10*5+3] +  25.53734467183137 )  > 1e-6: raise Exception( "Row 10 should be -25.53734467183137, not " + str(ppnt[10*5+3]) )
#         if np.abs( ppnt[15*5+3] +  76.04945574990082 )  > 1e-6: raise Exception( "Row 15 should be -76.04945574990082, not " + str(ppnt[15*5+3]) )
#     else:
#         if np.abs( ppnt[ 0*5+3] -  0.8545846415739397 ) > 1e-6: raise Exception( "Row 0 should be 0.8545846415739397, not " + str(ppnt[ 0*5+3]) )
#         if np.abs( ppnt[10*5+3] +  25.20150061107036 )  > 1e-6: raise Exception( "Row 10 should be -25.20150061107036, not " + str(ppnt[ 10*5+3]) )
#         if np.abs( ppnt[15*5+3] +  76.14723911261254 )  > 1e-6: raise Exception( "Row 15 should be -76.14723911261254, not " + str(ppnt[ 15*5+3]) )

# libconviqt.conviqt_convolver_del( convolver )
# libconviqt.conviqt_pointing_del( pnt )
# libconviqt.conviqt_detector_del( detector )
# libconviqt.conviqt_sky_del( sky )
# libconviqt.conviqt_beam_del( beam )

