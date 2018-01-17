# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI

import os

import numpy as np
from scipy.constants import degree

import healpy as hp

try:
    import ephem
except:
    ephem = None

from .. import qarray as qa
from .. import timing as timing

from ..healpix import ang2vec, vec2ang
from .tod import TOD
from .interval import Interval
from .noise import Noise
from .pointing_math import quat_equ2ecl, quat_equ2gal

from ..op import Operator


XAXIS, YAXIS, ZAXIS = np.eye(3)


def slew_precession_axis(nsim=1000, firstsamp=0, samplerate=100.0, degday=1.0):
    """
    Generate quaternions for constantly slewing precession axis.

    This constructs quaternions which rotates the Z coordinate axis
    to the X/Y plane, and then slowly rotates this.  This can be used
    to generate quaternions for the precession axis used in satellite
    scanning simulations.

    Args:
        nsim (int): The number of samples to simulate.
        firstsamp (int): The offset in samples from the start
            of rotation.
        samplerate (float): The sampling rate in Hz.
        degday (float): The rotation rate in degrees per day.

    Returns:
        Array of quaternions stored as an ndarray of
        shape (nsim, 4).
    """
    autotimer = timing.auto_timer()
    # this is the increment in radians per sample
    angincr = degday * (np.pi / 180.0) / (24.0 * 3600.0 * samplerate)

    # Compute the time-varying quaternions representing the rotation
    # from the coordinate frame to the precession axis frame.  The
    # angle of rotation is fixed (PI/2), but the axis starts at the Y
    # coordinate axis and sweeps.

    satang = np.arange(nsim, dtype=np.float64)
    satang *= angincr
    satang += angincr * firstsamp + (np.pi / 2)

    cang = np.cos(satang)
    sang = np.sin(satang)

    # this is the time-varying rotation axis
    sataxis = np.concatenate((cang.reshape(-1, 1), sang.reshape(-1, 1),
                              np.zeros((nsim, 1))), axis=1)

    # the rotation about the axis is always pi/2
    csatrot = np.cos(0.25 * np.pi)
    ssatrot = np.sin(0.25 * np.pi)

    # now construct the axis-angle quaternions for the precession
    # axis
    sataxis = np.multiply(np.repeat(ssatrot, nsim).reshape(-1,1), sataxis)
    satquat = np.concatenate((sataxis, np.repeat(csatrot, nsim).reshape(-1,1)),
                             axis=1)

    return satquat


def satellite_scanning(nsim=1000, firstsamp=0, samplerate=100.0, qprec=None,
                       spinperiod=1.0, spinangle=85.0, precperiod=0.0,
                       precangle=0.0):
    """
    Generate boresight quaternions for a generic satellite.

    Given scan strategy parameters and the relevant angles
    and rates, generate an array of quaternions representing
    the rotation of the ecliptic coordinate axes to the
    boresight.

    Args:
        nsim (int) : The number of samples to simulate.
        qprec (ndarray): If None (the default), then the
            precession axis will be fixed along the
            X axis.  If a 1D array of size 4 is given,
            This will be the fixed quaternion used
            to rotate the Z coordinate axis to the
            precession axis.  If a 2D array of shape
            (nsim, 4) is given, this is the time-varying
            rotation of the Z axis to the precession axis.
        samplerate (float): The sampling rate in Hz.
        spinperiod (float): The period (in minutes) of the
            rotation about the spin axis.
        spinangle (float): The opening angle (in degrees)
            of the boresight from the spin axis.
        precperiod (float): The period (in minutes) of the
            rotation about the precession axis.
        precangle (float): The opening angle (in degrees)
            of the spin axis from the precession axis.

    Returns:
        Array of quaternions stored as an ndarray of
        shape (nsim, 4).
    """
    autotimer = timing.auto_timer()
    if spinperiod > 0.0:
        spinrate = 1.0 / (60.0 * spinperiod)
    else:
        spinrate = 0.0
    spinangle = spinangle * np.pi / 180.0

    if precperiod > 0.0:
        precrate = 1.0 / (60.0 * precperiod)
    else:
        precrate = 0.0
    precangle = precangle * np.pi / 180.0

    xaxis = np.array([1,0,0], dtype=np.float64)
    yaxis = np.array([0,1,0], dtype=np.float64)
    zaxis = np.array([0,0,1], dtype=np.float64)

    satrot = None
    if qprec is None:
        # in this case, we just have a fixed precession axis, pointing
        # along the ecliptic X axis.
        satrot = np.tile(
            qa.rotation(np.array([0.0, 1.0, 0.0]), np.pi/2),
            nsim).reshape(-1, 4)
    elif qprec.flatten().shape[0] == 4:
        # we have a fixed precession axis.
        satrot = np.tile(qprec, nsim).reshape(-1,4)
    elif qprec.shape == (nsim, 4):
        # we have full vector of quaternions
        satrot = qprec
    else:
        raise RuntimeError("qprec has wrong dimensions")

    # Time-varying rotation about precession axis.
    # Increment per sample is
    # (2pi radians) X (precrate) / (samplerate)
    # Construct quaternion from axis / angle form.

    #print("satrot = ", satrot[-1])

    precang = np.arange(nsim, dtype=np.float64)
    precang += float(firstsamp)
    precang *= 2.0 * np.pi * precrate / samplerate
    #print("precang = ", precang[-1])

    cang = np.cos(0.5 * precang)
    sang = np.sin(0.5 * precang)

    precaxis = np.multiply(sang.reshape(-1, 1),
                           np.tile(zaxis, nsim).reshape(-1, 3))
    #print("precaxis = ", precaxis[-1])

    precrot = np.concatenate((precaxis, cang.reshape(-1, 1)), axis=1)
    #print("precrot = ", precrot[-1])

    # Rotation which performs the precession opening angle

    precopen = qa.rotation(np.array([1.0, 0.0, 0.0]), precangle)
    #print("precopen = ", precopen)

    # Time-varying rotation about spin axis.  Increment
    # per sample is
    # (2pi radians) X (spinrate) / (samplerate)
    # Construct quaternion from axis / angle form.

    spinang = np.arange(nsim, dtype=np.float64)
    spinang += float(firstsamp)
    spinang *= 2.0 * np.pi * spinrate / samplerate
    #print("spinang = ", spinang[-1])

    cang = np.cos(0.5 * spinang)
    sang = np.sin(0.5 * spinang)

    spinaxis = np.multiply(sang.reshape(-1, 1),
                           np.tile(zaxis, nsim).reshape(-1, 3))
    #print("spinaxis = ", spinaxis[-1])

    spinrot = np.concatenate((spinaxis, cang.reshape(-1, 1)), axis=1)
    #print("spinrot = ", spinrot[-1])

    # Rotation which performs the spin axis opening angle

    spinopen = qa.rotation(np.array([1.0, 0.0, 0.0]), spinangle)
    #print("spinopen = ", spinopen)

    # compose final rotation

    boresight = qa.mult(satrot,
                        qa.mult(precrot,
                                qa.mult(precopen,
                                        qa.mult(spinrot, spinopen))))
    #print("boresight = ", boresight[-1])

    return boresight


class TODHpixSpiral(TOD):
    """
    Provide a simple generator of fake detector pointing.

    Detector focalplane offsets are specified as a dictionary of 4-element
    ndarrays.  The boresight pointing is a simple looping over HealPix
    ring ordered pixel centers.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the
            data are distributed.
        detectors (dictionary): each key is the detector name, and each value
            is a quaternion tuple.
        samples (int):  The total number of samples.
        firsttime (float): starting time of data.
        rate (float): sample rate in Hz.
        nside (int): sky NSIDE to use.
        Other keyword arguments are passed to the parent class constructor.

    """
    def __init__(self, mpicomm, detectors, samples, firsttime=0.0,
                 rate=100.0, nside=512, **kwargs):

        self._fp = detectors
        self._detlist = sorted(list(self._fp.keys()))

        props = {
            "nside": nside,
        }
        super().__init__(mpicomm, self._detlist, samples, meta=props, **kwargs)

        self._firsttime = firsttime
        self._rate = rate
        self._nside = nside
        self._npix = 12 * self._nside * self._nside

    def detoffset(self):
        return { d : np.asarray(self._fp[d]) for d in self._detlist }

    def _get(self, detector, start, n):
        # This class just returns data streams of zeros
        return np.zeros(n, dtype=np.float64)

    def _put(self, detector, start, data, flags):
        raise RuntimeError("cannot write data to simulated data streams")
        return

    def _get_flags(self, detector, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_flags(self, detector, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_common_flags(self, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_common_flags(self, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_times(self, start, n):
        start_abs = self.local_samples[0] + start
        start_time = self._firsttime + float(start_abs) / self._rate
        stop_time = start_time + float(n) / self._rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False,
                             dtype=np.float64)
        return stamps

    def _put_times(self, start, stamps):
        raise RuntimeError("cannot write timestamps to simulated data streams")
        return

    def _get_boresight(self, start, n):
        autotimer = timing.auto_timer(type(self).__name__)
        # compute the absolute sample offset
        start_abs = self.local_samples[0] + start

        # pixel offset
        start_pix = int(start_abs % self._npix)
        pixels = np.linspace(start_pix, start_pix + n, num=n, endpoint=False)
        pixels = np.mod(pixels,
                        self._npix*np.ones(n, dtype=np.int64)).astype(np.int64)

        # the result of this is normalized
        x, y, z = hp.pix2vec(self._nside, pixels, nest=False)

        # z axis is obviously normalized
        zaxis = np.array([0, 0, 1], dtype=np.float64)
        ztiled = np.tile(zaxis, x.shape[0]).reshape((-1, 3))

        # ... so dir is already normalized
        dir = np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

        # get the rotation axis
        v = np.cross(ztiled, dir)
        v = v / np.sqrt(np.sum(v * v, axis=1)).reshape((-1, 1))

        # this is the vector-wise dot product
        zdot = np.sum(ztiled * dir, axis=1).reshape((-1, 1))
        ang = 0.5 * np.arccos(zdot)

        # angle element
        s = np.cos(ang)

        # axis
        v *= np.sin(ang)

        # build the un-normalized quaternion
        boresight = np.concatenate((v, s), axis=1)

        return qa.norm(boresight)

    def _put_boresight(self, start, data):
        raise RuntimeError("cannot write boresight to simulated data streams")
        return

    def _get_pntg(self, detector, start, n):
        autotimer = timing.auto_timer(type(self).__name__)
        detquat = np.asarray(self._fp[detector])
        boresight = self._get_boresight(start, n)
        data = qa.mult(boresight, detquat)
        return data

    def _put_pntg(self, detector, start, data):
        raise RuntimeError("cannot write data to simulated pointing")
        return

    def _get_position(self, start, n):
        return np.zeros((n, 3), dtype=np.float64)

    def _put_position(self, start, pos):
        raise RuntimeError("cannot write data to simulated position")
        return

    def _get_velocity(self, start, n):
        return np.zeros((n, 3), dtype=np.float64)

    def _put_velocity(self, start, vel):
        raise RuntimeError("cannot write data to simulated velocity")
        return


class TODSatellite(TOD):
    """
    Provide a simple generator of satellite detector pointing.

    Detector focalplane offsets are specified as a dictionary of 4-element
    ndarrays.  The boresight pointing is a generic 2-angle model.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the
            data are distributed.
        detectors (dictionary): each key is the detector name, and each value
            is a quaternion tuple.
        samples (int):  The total number of samples.
        firsttime (float): starting time of data.
        rate (float): sample rate in Hz.
        spinperiod (float): The period (in minutes) of the
            rotation about the spin axis.
        spinangle (float): The opening angle (in degrees)
            of the boresight from the spin axis.
        precperiod (float): The period (in minutes) of the
            rotation about the precession axis.
        precangle (float): The opening angle (in degrees)
            of the spin axis from the precession axis.
        All other keyword arguments are passed to the parent constructor.

    """
    def __init__(self, mpicomm, detectors, samples, firsttime=0.0, rate=100.0,
                 spinperiod=1.0, spinangle=85.0, precperiod=0.0, precangle=0.0,
                 **kwargs):

        self._fp = detectors
        self._detlist = sorted(list(self._fp.keys()))

        self._firsttime = firsttime
        self._rate = rate
        self._spinperiod = spinperiod
        self._spinangle = spinangle
        self._precperiod = precperiod
        self._precangle = precangle
        self._boresight = None

        props = {
            "spinperiod": spinperiod,
            "spinangle": spinangle,
            "precperiod": precperiod,
            "precangle": precangle
        }

        # call base class constructor to distribute data
        super().__init__(mpicomm, self._detlist, samples, meta=props, **kwargs)

        self._AU = 149597870.7
        self._radperday = 0.01720209895
        self._radpersec = self._radperday / 86400.0
        self._radinc = self._radpersec / self._rate
        self._earthspeed = self._radpersec * self._AU

    def set_prec_axis(self, qprec=None):
        """
        Set the fixed or time-varying precession axis.

        This function sets the precession axis for the locally assigned samples.
        It also triggers the generation and caching of the boresight pointing.

        Args:
            qprec (ndarray): If None (the default), then the
                precession axis will be fixed along the
                X axis.  If a 1D array of size 4 is given,
                This will be the fixed quaternion used
                to rotate the Z coordinate axis to the
                precession axis.  If a 2D array of shape
                (local samples, 4) is given, this is the time-varying
                rotation of the Z axis to the precession axis.
        """
        if qprec is not None:
            if (qprec.shape != (4,)) and \
               (qprec.shape != (self.local_samples[1], 4)):
                raise RuntimeError(
                    "precession quaternion has incorrect dimensions")

        # generate and cache the boresight pointing
        self._boresight = satellite_scanning(
            nsim=self.local_samples[1], firstsamp=self.local_samples[0],
            qprec=qprec, samplerate=self._rate, spinperiod=self._spinperiod,
            spinangle=self._spinangle, precperiod=self._precperiod,
            precangle=self._precangle)

    def detoffset(self):
        return {d : np.asarray(self._fp[d]) for d in self._detlist}

    def _get_boresight(self, start, n):
        if self._boresight is None:
            raise RuntimeError("you must set the precession axis before "
                               "reading pointing")
        return self._boresight[start:start+n]

    def _put_boresight(self, start, data):
        raise RuntimeError("cannot write boresight to simulated data streams")
        return

    def _get(self, detector, start, n):
        # This class just returns data streams of zeros
        return np.zeros(n, dtype=np.float64)

    def _put(self, detector, start, data):
        raise RuntimeError("cannot write data to simulated data streams")
        return

    def _get_flags(self, detector, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_det_flags(self, detector, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_common_flags(self, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_common_flags(self, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_times(self, start, n):
        start_abs = self.local_samples[0] + start
        start_time = self._firsttime + float(start_abs) / self._rate
        stop_time = start_time + float(n) / self._rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False,
                             dtype=np.float64)
        return stamps

    def _put_times(self, start, stamps):
        raise RuntimeError("cannot write timestamps to simulated data streams")
        return

    def _get_pntg(self, detector, start, n):
        boresight = self._get_boresight(start, n)
        detquat = self._fp[detector]
        data = qa.mult(boresight, detquat)
        return data

    def _put_pntg(self, detector, start, data):
        raise RuntimeError("cannot write data to simulated pointing")
        return

    def _get_position(self, start, n):
        # For this simple class, assume that the Earth is located
        # along the X axis at time == 0.0s.  We also just use the
        # mean values for distance and angular speed.  Classes for
        # real experiments should obviously use ephemeris data.
        rad = np.fmod((start - self._firsttime) * self._radpersec, 2.0 * np.pi)
        ang = self._radinc * np.arange(n, dtype=np.float64) + rad
        x = self._AU * np.cos(ang)
        y = self._AU * np.sin(ang)
        z = np.zeros_like(x)
        return np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

    def _put_position(self, start, pos):
        raise RuntimeError("cannot write data to simulated position")
        return

    def _get_velocity(self, start, n):
        # For this simple class, assume that the Earth is located
        # along the X axis at time == 0.0s.  We also just use the
        # mean values for distance and angular speed.  Classes for
        # real experiments should obviously use ephemeris data.
        rad = np.fmod((start - self._firsttime) * self._radpersec, 2.0 * np.pi)
        ang = self._radinc * np.arange(n, dtype=np.float64) + rad + (0.5*np.pi)
        x = self._earthspeed * np.cos(ang)
        y = self._earthspeed * np.sin(ang)
        z = np.zeros_like(x)
        return np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

    def _put_velocity(self, start, vel):
        raise RuntimeError("cannot write data to simulated velocity")
        return


class TODGround(TOD):
    """
    Provide a simple generator of ground-based detector pointing.

    Detector focalplane offsets are specified as a dictionary of
    4-element ndarrays.  The boresight pointing is a generic
    2-angle model.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the
            data is distributed.
        detectors (dictionary): each key is the detector name, and each
            value is a quaternion tuple.
        samples (int):  The total number of samples.
        firsttime (float): starting time of data.
        rate (float): sample rate in Hz.
        site_lon (float/str): Observing site Earth longitude in radians
            or a pyEphem string.
        site_lat (float/str): Observing site Earth latitude in radians
            or a pyEphem string.
        site_alt (float/str): Observing site Earth altitude in meters.
        scanrate (float): Sky scanning rate in degrees / second.
        scan_accel (float): Sky scanning rate acceleration in
            degrees / second^2 for the turnarounds.
        CES_start (float): Start time of the constant elevation scan
        CES_stop (float): Stop time of the constant elevation scan
        sun_angle_min (float): Minimum angular distance for the scan and
            the Sun [degrees].
        sampsizes (list):  Optional list of sample chunk sizes which
            cannot be split.
        sampbreaks (list):  Optional list of hard breaks in the sample
            distribution.
        coord (str):  Sky coordinate system.  One of
            C (Equatorial), E (Ecliptic) or G (Galactic)
        report_timing (bool):  Report the time spent simulating the scan
            and translating the pointing.
        All other keyword arguments are passed to the parent constructor.
    """

    TURNAROUND = 1
    LEFTRIGHT_SCAN = 2
    RIGHTLEFT_SCAN = 4
    LEFTRIGHT_TURNAROUND = LEFTRIGHT_SCAN + TURNAROUND
    RIGHTLEFT_TURNAROUND = RIGHTLEFT_SCAN + TURNAROUND
    SUN_UP = 8
    SUN_CLOSE = 16

    def __init__(self, mpicomm, detectors, samples, firsttime=0.0, rate=100.0,
                 site_lon=0, site_lat=0, site_alt=0, azmin=0, azmax=0, el=0,
                 scanrate=1, scan_accel=0.1,
                 CES_start=None, CES_stop=None, el_min=0, sun_angle_min=90,
                 sampsizes=None, sampbreaks=None, coord="C",
                 report_timing=True, **kwargs):
        autotimer = timing.auto_timer(type(self).__name__)

        if samples < 1:
            raise RuntimeError(
                "TODGround must be instantiated with a positive number of "
                "samples, not samples == {}".format(samples))

        if ephem is None:
            raise RuntimeError("Cannot instantiate a TODGround object "
                               "without pyephem.")

        if sampsizes is not None or sampbreaks is not None:
            raise RuntimeError("TODGround will synthesize the sizes to match "
                               "the subscans.")

        if CES_start is None:
            CES_start = firsttime
        elif firsttime < CES_start:
            raise RuntimeError("TODGround: firsttime < CES_start: {} < {}"
                               "".format(firsttime, CES_start))
        lasttime = firsttime + samples / rate
        if CES_stop is None:
            CES_stop = lasttime
        elif lasttime > CES_stop:
            raise RuntimeError("TODGround: lasttime > CES_stop: {} > {}"
                               "".format(lasttime, CES_stop))

        self._firsttime = firsttime
        self._lasttime = lasttime
        self._rate = rate
        self._site_lon = site_lon
        self._site_lat = site_lat
        self._site_alt = site_alt
        self._azmin = azmin * degree
        self._azmax = azmax * degree
        if el < 1 or el > 89:
            raise RuntimeError(
                "Impossible CES at {:.2f} degrees".format(el))
        self._el = el * degree
        self._scanrate = scanrate * degree
        self._scan_accel = scan_accel * degree
        self._CES_start = CES_start
        self._CES_stop = CES_stop
        self._el_min = el_min
        self._sun_angle_min = sun_angle_min
        if coord not in "CEG":
            raise RuntimeError("Unknown coordinate system: {}".format(coord))
        self._coord = coord
        self._report_timing = report_timing

        self._observer = ephem.Observer()
        self._observer.lon = self._site_lon
        self._observer.lat = self._site_lat
        self._observer.elevation = self._site_alt # In meters
        self._observer.epoch = ephem.J2000 # "2000"
        #self._observer.epoch = -9786 # EOD
        self._observer.compute_pressure()

        self._min_az = None
        self._max_az = None
        self._min_el = None
        self._min_el = None

        self._az = None
        self._commonflags = None
        self._boresight_azel = None
        self._boresight = None

        # Set the boresight pointing based on the given scan parameters

        if self._report_timing:
            mpicomm.Barrier()
            tstart = MPI.Wtime()

        sizes, starts = self.simulate_scan(samples)

        if self._report_timing:
            mpicomm.Barrier()
            tstop = MPI.Wtime()
            if mpicomm.rank == 0 and tstop-tstart > 1:
                print("TODGround: Simulated scan in {:.2f} s"
                      "".format(tstop - tstart), flush=True)
            tstart = tstop

        # Create a list of subscans that excludes the turnarounds.
        # All processes in the group still have all samples.

        self.subscans = []
        self._subscan_min_length = 10 # in samples
        for istart, istop in zip(self._stable_starts, self._stable_stops):
            if istop-istart < self._subscan_min_length:
                self._commonflags[istart:istop] |= self.TURNAROUND
                continue
            start = self._firsttime + istart / self._rate
            stop = self._firsttime + istop / self._rate
            self.subscans.append(
                Interval(start=start, stop=stop, first=istart, last=istop-1))

        self._commonflags[istop:] |= self.TURNAROUND

        if np.sum((self._commonflags & self.TURNAROUND) == 0) == 0:
            raise RuntimeError(
                'The entire TOD is flagged as turnaround. Samplerate too low '
                '({} Hz) or scanrate too high ({} deg/s)?'
                ''.format(rate, scanrate))

        if self._report_timing:
            mpicomm.Barrier()
            tstop = MPI.Wtime()
            if mpicomm.rank == 0 and tstop-tstart > 1:
                print("TODGround: Listed valid intervals in {:.2f} s"
                      "".format(tstop - tstart), flush=True)
            tstart = tstop

        self._fp = detectors
        self._detlist = sorted(list(self._fp.keys()))

        # call base class constructor to distribute data

        props = {
            "site_lon": site_lon,
            "site_lat": site_lat,
            "site_alt": site_alt,
            "azmin": azmin,
            "azmax": azmax,
            "el": el,
            "scanrate": scanrate,
            "scan_accel": scan_accel,
            "el_min": el_min,
            "sun_angle_min": sun_angle_min
        }
        super().__init__(mpicomm, self._detlist, samples, sampsizes=[samples],
                         sampbreaks=None, meta=props, **kwargs)

        if self._report_timing:
            mpicomm.Barrier()
            tstop = MPI.Wtime()
            if mpicomm.rank == 0 and tstop-tstart > 1:
                print("TODGround: Called parent constructor in {:.2f} s"
                      "".format(tstop - tstart), flush=True)
            tstart = tstop

        self.translate_pointing()

        self.crop_vectors()

        if self._report_timing:
            mpicomm.Barrier()
            tstop = MPI.Wtime()
            if mpicomm.rank == 0 and tstop-tstart > 1:
                print("TODGround: Translated scan pointing in {:.2f} s"
                      "".format(tstop - tstart), flush=True)

    def __del__(self):
        autotimer = timing.auto_timer(type(self).__name__)

        try:
            del self._boresight_azel
        except:
            pass
        try:
            del self._boresight
        except:
            pass
        try:
            del self._az
        except:
            pass
        try:
            del self._commonflags
        except:
            pass
        try:
            self.cache.clear()
        except:
            pass

    def to_JD(self, t):
        """
        Convert TOAST UTC time stamp to Julian date
        """
        return t / 86400. + 2440587.5

    def to_DJD(self, t):
        """
        Convert TOAST UTC time stamp to Dublin Julian date used
        by pyEphem.
        """
        return self.to_JD(t) - 2415020

    @property
    def scan_range(self):
        """
        (tuple):  The extent of the boresight pointing as (min_az, max_az,
            min_el, max_el) in radians.  Includes turnarounds.
        """
        return self._min_az, self._max_az, self._min_el, self._max_el

    def simulate_scan(self, samples):
        # simulate the scanning with turnarounds. Regardless of firsttime,
        # we must simulate from the beginning of the CES.
        # Generate matching common flags.
        # Sets self._boresight.
        autotimer = timing.auto_timer(type(self).__name__)

        self._az = np.zeros(samples)
        self._commonflags = np.zeros(samples, dtype=np.uint8)
        # Scan starts from the left edge of the patch at the fixed scan rate
        lim_left = self._azmin
        lim_right = self._azmax
        if lim_right < lim_left:
            # We are scanning across the zero meridian
            lim_right += 2*np.pi
        az_last = lim_left
        scanrate = self._scanrate / self._rate # per sample, not per second
        # Modulate scan rate so that the rate on sky is constant
        scanrate /= np.cos(self._el)
        scan_accel = self._scan_accel / self._rate # per sample, not per second
        scan_accel /= np.cos(self._el)
        tol = self._rate / 10
        # the index, i, is relative to the start of the tod object.
        # If CES begun before the TOD, first values of i are negative.
        i = int((self._CES_start - self._firsttime - tol) * self._rate)
        starts = [0] # Subscan start indices
        self._stable_starts = []
        self._stable_stops = []
        while True:
            #
            # Left to right, fixed rate
            #
            self._stable_starts.append(i)
            dazdt = scanrate
            nstep = min(int((lim_right-az_last) // dazdt) + 1, samples-i)
            offset_in = max(0, -i)
            offset_out = max(0, i)
            ngood = nstep - offset_in
            if ngood > 0:
                self._commonflags[offset_out:offset_out+ngood] \
                    |= self.LEFTRIGHT_SCAN
                self._az[offset_out:offset_out+ngood] \
                    = az_last + np.arange(offset_in, offset_in+ngood)*dazdt
            i += nstep
            self._stable_stops.append(i)
            if i == samples:
                break
            az_last += dazdt*nstep
            #
            # Left to right, turnaround
            #
            nstep_full = int((2*scanrate) // scan_accel) + 1
            nstep = min(int(nstep_full), samples-i)
            offset_in = max(0, -i)
            offset_out = max(0, i)
            ngood = nstep - offset_in
            if ngood > 0:
                self._commonflags[offset_out:offset_out+ngood] \
                    |= self.LEFTRIGHT_TURNAROUND
                ii = np.arange(offset_in, offset_in+ngood)
                self._az[offset_out:offset_out+ngood] \
                    = az_last + ii*dazdt - 0.5*scan_accel*ii**2
                halfway = i + nstep_full//2
                if halfway > 0 and halfway < samples:
                    starts.append(halfway)
            i += nstep
            if i == samples:
                break
            az_last += dazdt*nstep - .5*scan_accel*nstep**2
            #
            # Right to left, fixed rate
            #
            self._stable_starts.append(i)
            dazdt = -scanrate
            nstep = min(int((lim_left-az_last) // dazdt) + 1, samples-i)
            offset_in = max(0, -i)
            offset_out = max(0, i)
            ngood = nstep - offset_in
            if ngood > 0:
                self._commonflags[offset_out:offset_out+ngood] \
                    |= self.RIGHTLEFT_SCAN
                self._az[offset_out:offset_out+ngood] \
                    = az_last + np.arange(offset_in, offset_in+ngood)*dazdt
            i += nstep
            self._stable_stops.append(i)
            if i == samples: break
            az_last += dazdt*nstep
            #
            # Right to left, turnaround
            #
            nstep_full = int((2*scanrate) // scan_accel) + 1
            nstep = min(int(nstep_full), samples-i)
            offset_in = max(0, -i)
            offset_out = max(0, i)
            ngood = nstep - offset_in
            if ngood > 0:
                self._commonflags[offset_out:offset_out+ngood] \
                    |= self.RIGHTLEFT_TURNAROUND
                ii = np.arange(offset_in, offset_in+ngood)
                self._az[offset_out:offset_out+ngood] \
                    = az_last + ii*dazdt + 0.5*scan_accel*ii**2
                halfway = i + nstep_full//2
                if halfway > 0 and halfway < samples:
                    starts.append(halfway)
            i += nstep
            if i == samples:
                break
            az_last += dazdt*nstep + .5*scan_accel*nstep**2

        starts.append(samples)
        sizes = np.diff(starts)
        if np.sum(sizes) != samples:
            raise RuntimeError("Subscans do not match samples")

        # Store the scan range before discarding samples not assigned
        # to this process

        self._az %= 2*np.pi
        if np.ptp(self._az) < np.pi:
            self._min_az = np.amin(self._az)
            self._max_az = np.amax(self._az)
        else:
            # Scanning across the zero azimuth.
            self._min_az = np.amin(self._az[self._az > np.pi]) - 2*np.pi
            self._max_az = np.amax(self._az[self._az < np.pi])
        self._min_el = self._el
        self._max_el = self._el

        return sizes, starts[:-1]

    def translate_pointing(self):
        """ Translate Az/El into bore sight quaternions

        Translate the azimuth and elevation into bore sight quaternions.

        """
        autotimer = timing.auto_timer(type(self).__name__)

        # At this point, all processes still have all of the scan

        nsamp = len(self._az)
        rank = self._mpicomm.rank
        ntask = self._mpicomm.size
        nsamp_task = nsamp // ntask + 1
        my_start = rank * nsamp_task
        my_stop = min(my_start+nsamp_task, nsamp)
        my_nsamp = max(0, my_stop-my_start)
        my_ind = slice(my_start, my_stop)

        # Remember that the azimuth is measured clockwise and the
        # longitude counter-clockwise
        my_azelquats = qa.from_angles(
            np.pi/2 - np.ones(my_nsamp)*self._el,
            -self._az[my_ind], np.zeros(my_nsamp), IAU=False)
        azelquats = np.vstack(self._mpicomm.allgather(my_azelquats))
        self._boresight_azel = azelquats

        my_times = self.local_times()[my_ind]
        azel2radec_times, azel2radec_quats = self._get_azel2radec_quats()
        my_azel2radec_quats = qa.slerp(my_times, azel2radec_times, azel2radec_quats)
        my_quats = qa.mult(my_azel2radec_quats, my_azelquats)
        del my_azelquats

        quats = np.vstack(self._mpicomm.allgather(my_quats))
        self._boresight = quats
        del my_quats

        return

    def crop_vectors(self):
        """ Crop the TOD vectors.

        Crop the TOD vectors to match the sample range assigned to this task.

        """
        autotimer = timing.auto_timer(type(self).__name__)
        offset, n = self.local_samples
        ind = slice(offset, offset+n)

        self._az = self.cache.put("az", self._az[ind])
        self._commonflags = self.cache.put(
            "common_flags", self._commonflags[ind], replace=True)
        self._boresight_azel = self.cache.put("boresight_azel",
                                              self._boresight_azel[ind])
        self._boresight = self.cache.put("boresight_radec",
                                         self._boresight[ind])

        return

    def _get_azel2radec_quats(self):
        """ Construct a sparsely sampled vector of Az/El->Ra/Dec quaternions.

        The interpolation times must be tied to the total observation so
        that the results do not change when data is distributed in time
        domain.

        """
        # One control point at least every 10 minutes.  Overkill but
        # costs nothing.
        autotimer = timing.auto_timer(type(self).__name__)
        n = max(2, 1 + int((self._lasttime-self._firsttime) / 600))
        times = np.linspace(self._firsttime, self._lasttime, n)
        quats = np.zeros([n, 4])
        for i, t in enumerate(times):
            quats[i] = self._get_coord_quat(t)

        return times, quats

    def _get_coord_quat(self, t):
        """ Get the Az/El -> Ra/Dec conversion quaternion for boresight.

        We will apply atmospheric refraction and stellar aberration in
        the detector frame.

        """
        autotimer = timing.auto_timer(type(self).__name__)
        self._observer.date = self.to_DJD(t)
        # Set pressure to zero to disable atmospheric refraction.
        pressure = self._observer.pressure
        self._observer.pressure = 0
        # Rotate the X, Y and Z axes from horizontal to equatorial frame.
        # Strictly speaking, two coordinate axes would suffice but the
        # math is cleaner with three axes.
        #
        # PyEphem measures the azimuth East (clockwise) from North.
        # The direction is standard but opposite to ISO spherical coordinates.
        try:
            xra, xdec = self._observer.radec_of(       0,       0, fixed=False)
            yra, ydec = self._observer.radec_of(-np.pi/2,       0, fixed=False)
            zra, zdec = self._observer.radec_of(       0, np.pi/2, fixed=False)
        except:
            # Modified pyephem not available.
            # Translated pointing will include stellar aberration.
            xra, xdec = self._observer.radec_of(       0,       0)
            yra, ydec = self._observer.radec_of(-np.pi/2,       0)
            zra, zdec = self._observer.radec_of(       0, np.pi/2)
        self._observer.pressure = pressure
        xvec, yvec, zvec = ang2vec(np.pi/2-np.array([xdec, ydec, zdec]),
                                   np.array([xra, yra, zra]))
        # Solve for the quaternions from the transformed axes.
        X = (xvec[1] + yvec[0]) / 4
        Y = (xvec[2] + zvec[0]) / 4
        Z = (yvec[2] + zvec[1]) / 4
        d = np.sqrt(Y * Z / X) # Choose positive root
        c = d * X / Y
        b = X / c
        a = (xvec[1]/2 - b*c) / d
        # qarray has the scalar part as the last index
        quat = np.array([b, c, d, a])
        return quat

    def free_azel_quats(self):
        autotimer = timing.auto_timer(type(self).__name__)
        self._boresight_azel = None
        self.cache.destroy("boresight_azel")

    def free_radec_quats(self):
        autotimer = timing.auto_timer(type(self).__name__)
        self._boresight = None
        self.cache.destroy("boresight_radec")

    def radec2quat(self, ra, dec, pa):
        autotimer = timing.auto_timer(type(self).__name__)

        qR = qa.rotation(ZAXIS, ra+np.pi/2)
        qD = qa.rotation(XAXIS, np.pi/2-dec)
        qP = qa.rotation(ZAXIS, pa) # FIXME: double-check this
        q = qa.mult(qR, qa.mult(qD, qP))

        if self._coord != "C":
            # Add the coordinate system rotation
            if self._coord == "G":
                q = qa.mult(quat_equ2gal, q)
            elif self._coord == "E":
                q = qa.mult(quat_equ2ecl, q)
            else:
                raise RuntimeError(
                    "Unknown coordinate system: {}".format(self._coord))

        return q

    def detoffset(self):
        return { d : np.asarray(self._fp[d]) for d in self._detlist }

    def _get(self, detector, start, n):
        # This class just returns data streams of zeros
        return np.zeros(n, dtype=np.float64)

    def _put(self, detector, start, data):
        raise RuntimeError("cannot write data to simulated data streams")
        return

    def _get_flags(self, detector, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_flags(self, detector, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_common_flags(self, start, n):
        return self._commonflags[start:start+n]

    def _put_common_flags(self, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_times(self, start, n):
        autotimer = timing.auto_timer(type(self).__name__)
        start_abs = self.local_samples[0] + start
        start_time = self._firsttime + float(start_abs) / self._rate
        return start_time + np.arange(n) / self._rate

    def _put_times(self, start, stamps):
        raise RuntimeError("cannot write timestamps to simulated data streams")
        return

    def _get_boresight(self, start, n, azel=False):
        if azel:
            if self._boresight_azel is None:
                raise RuntimeError("Boresight azel pointing was purged.")
            return self._boresight_azel[start:start+n]
        else:
            if self._boresight is None:
                raise RuntimeError("Boresight radec pointing was purged.")
            return self._boresight[start:start+n]

    def _put_boresight(self, start, data):
        raise RuntimeError("cannot write boresight to simulated data streams")
        return

    def read_boresight_az(self, local_start=0, n=0):
        """
        Read the boresight azimuth.

        Args:
            local_start (int): the sample offset relative to the first locally
                assigned sample.
            n (int): the number of samples to read.  If zero, read to end.

        Returns:
            (array): a numpy array containing the timestamps.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError("cannot read boresight azimuth - process "
                "has no assigned local samples")
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError("local sample range {} - {} is invalid".format(
                local_start, local_start+n-1))
        return self._az[local_start:local_start+n]

    def _get_pntg(self, detector, start, n, azel=False):
        # FIXME: this is where we will apply atmospheric refraction and
        # stellar aberration corrections in the detector frame.  For
        # simulations they will only matter if we want to simulate the
        # error coming from ignoring them.
        autotimer = timing.auto_timer(type(self).__name__)
        boresight = self._get_boresight(start, n, azel=azel)
        detquat = self._fp[detector]
        return qa.mult(boresight, detquat)

    def _put_pntg(self, detector, start, data):
        raise RuntimeError("cannot write data to simulated pointing")
        return

    def _get_position(self, start, n):
        # For this simple class, assume that the Earth is located
        # along the X axis at time == 0.0s.  We also just use the
        # mean values for distance and angular speed.  Classes for
        # real experiments should obviously use ephemeris data.
        autotimer = timing.auto_timer(type(self).__name__)
        rad = np.fmod((start - self._firsttime) * self._radpersec, 2.0 * np.pi)
        ang = self._radinc * np.arange(n, dtype=np.float64) + rad
        x = self._AU * np.cos(ang)
        y = self._AU * np.sin(ang)
        z = np.zeros_like(x)
        return np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

    def _put_position(self, start, pos):
        raise RuntimeError("cannot write data to simulated position")
        return

    def _get_velocity(self, start, n):
        # For this simple class, assume that the Earth is located
        # along the X axis at time == 0.0s.  We also just use the
        # mean values for distance and angular speed.  Classes for
        # real experiments should obviously use ephemeris data.
        autotimer = timing.auto_timer(type(self).__name__)
        rad = np.fmod((start - self._firsttime) * self._radpersec, 2.0 * np.pi)
        ang = self._radinc * np.arange(n, dtype=np.float64) + rad + (0.5*np.pi)
        x = self._earthspeed * np.cos(ang)
        y = self._earthspeed * np.sin(ang)
        z = np.zeros_like(x)
        return np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

    def _put_velocity(self, start, vel):
        raise RuntimeError("cannot write data to simulated velocity")
        return
