#!/usr/bin/env python3

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast.mpi import MPI

import argparse
import copy
from datetime import datetime
import dateutil.parser
import os
import pickle
import re
import sys
import traceback

import numpy as np
import healpy as hp

import toast
import toast.tod as tt
import toast.map as tm
import toast.todmap as ttm
import toast.qarray as qa
import toast.timing as timing
from toast import Weather

#import warnings
#warnings.filterwarnings('error')
#warnings.simplefilter('ignore', ImportWarning)
#warnings.simplefilter('ignore', ResourceWarning)
#warnings.simplefilter('ignore', DeprecationWarning)
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


XAXIS, YAXIS, ZAXIS = np.eye(3)

def parse_arguments(comm):

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "atmosphere and make maps for some number of noise Monte Carlos.",
        fromfile_prefix_chars='@')
    parser.add_argument('--groupsize',
                        required=False, type=np.int,
                        help='Size of a process group assigned to a CES')

    parser.add_argument('--timezone', required=False, type=np.int, default=0,
                        help='Offset to apply to MJD to separate days [hours]')
    parser.add_argument('--coord', required=False, default='C',
                        help='Sky coordinate system [C,E,G]')
    parser.add_argument('--schedule', required=True,
                        help='Comma-separated list CES schedule files '
                        '(from toast_ground_schedule.py)')
    parser.add_argument('--weather',
                        required=False,
                        help='Comma-separated list of TOAST weather files for '
                        'every schedule.  Repeat the same file if the '
                        'schedules share observing site.')
    parser.add_argument('--samplerate',
                        required=False, default=100.0, type=np.float,
                        help='Detector sample rate (Hz)')
    parser.add_argument('--scanrate',
                        required=False, default=1.0, type=np.float,
                        help='Scanning rate [deg / s]')
    parser.add_argument('--scan_accel',
                        required=False, default=1.0, type=np.float,
                        help='Scanning rate change [deg / s^2]')
    parser.add_argument('--sun_angle_min',
                        required=False, default=30.0, type=np.float,
                        help='Minimum azimuthal distance between the Sun and '
                        'the bore sight [deg]')

    parser.add_argument('--polyorder',
                        required=False, type=np.int,
                        help='Polynomial order for the polyfilter')

    parser.add_argument('--wbin_ground',
                        required=False, type=np.float,
                        help='Ground template bin width [degrees]')

    parser.add_argument('--gain_sigma',
                        required=False, type=np.float,
                        help='Gain error distribution')

    parser.add_argument('--hwprpm',
                        required=False, default=0.0, type=np.float,
                        help='The rate (in RPM) of the HWP rotation')
    parser.add_argument('--hwpstep', required=False, default=None,
                        help='For stepped HWP, the angle in degrees '
                        'of each step')
    parser.add_argument('--hwpsteptime',
                        required=False, default=0.0, type=np.float,
                        help='For stepped HWP, the the time in seconds '
                        'between steps')

    parser.add_argument('--input_map', required=False,
                        help='Input map for signal')
    parser.add_argument('--input_pysm_model', required=False,
                        help='Comma separated models for on-the-fly PySM '
                        'simulation, e.g. s3,d6,f1,a2"')
    parser.add_argument('--apply_beam', required=False, action='store_true',
                        help='Apply beam convolution to input map with gaussian '
                        'beam parameters defined in focalplane')

    parser.add_argument('--skip_atmosphere',
                        required=False, default=False, action='store_true',
                        help='Disable simulating the atmosphere.')
    parser.add_argument('--skip_noise',
                        required=False, default=False, action='store_true',
                        help='Disable simulating detector noise.')
    parser.add_argument('--skip_bin',
                        required=False, default=False, action='store_true',
                        help='Disable binning the map.')
    parser.add_argument('--skip_hits',
                        required=False, default=False, action='store_true',
                        help='Do not save the 3x3 matrices and hitmaps')
    parser.add_argument('--skip_destripe',
                        required=False, default=False, action='store_true',
                        help='Do not destripe the data')
    parser.add_argument('--skip_daymaps',
                        required=False, default=False, action='store_true',
                        help='Do not bin daily maps')

    parser.add_argument('--atm_lmin_center',
                        required=False, default=0.01, type=np.float,
                        help='Kolmogorov turbulence dissipation scale center')
    parser.add_argument('--atm_lmin_sigma',
                        required=False, default=0.001, type=np.float,
                        help='Kolmogorov turbulence dissipation scale sigma')
    parser.add_argument('--atm_lmax_center',
                        required=False, default=10.0, type=np.float,
                        help='Kolmogorov turbulence injection scale center')
    parser.add_argument('--atm_lmax_sigma',
                        required=False, default=10.0, type=np.float,
                        help='Kolmogorov turbulence injection scale sigma')
    parser.add_argument('--atm_gain',
                        required=False, default=1e-4, type=np.float,
                        help='Atmospheric gain factor.')
    parser.add_argument('--atm_zatm',
                        required=False, default=40000.0, type=np.float,
                        help='atmosphere extent for temperature profile')
    parser.add_argument('--atm_zmax',
                        required=False, default=200.0, type=np.float,
                        help='atmosphere extent for water vapor integration')
    parser.add_argument('--atm_xstep',
                        required=False, default=10.0, type=np.float,
                        help='size of volume elements in X direction')
    parser.add_argument('--atm_ystep',
                        required=False, default=10.0, type=np.float,
                        help='size of volume elements in Y direction')
    parser.add_argument('--atm_zstep',
                        required=False, default=10.0, type=np.float,
                        help='size of volume elements in Z direction')
    parser.add_argument('--atm_nelem_sim_max',
                        required=False, default=1000, type=np.int,
                        help='controls the size of the simulation slices')
    parser.add_argument('--atm_gangsize',
                        required=False, default=1, type=np.int,
                        help='size of the gangs that create slices')
    parser.add_argument('--atm_wind_time',
                        required=False, default=36000.0, type=np.float,
                        help='Maximum time to simulate without discontinuity')
    parser.add_argument('--atm_z0_center',
                        required=False, default=2000.0, type=np.float,
                        help='central value of the water vapor distribution')
    parser.add_argument('--atm_z0_sigma',
                        required=False, default=0.0, type=np.float,
                        help='sigma of the water vapor distribution')
    parser.add_argument('--atm_T0_center',
                        required=False, default=280.0, type=np.float,
                        help='central value of the temperature distribution')
    parser.add_argument('--atm_T0_sigma',
                        required=False, default=10.0, type=np.float,
                        help='sigma of the temperature distribution')
    parser.add_argument('--atm_cache',
                        required=False, default='atm_cache',
                        help='Atmosphere cache directory')

    parser.add_argument('--outdir',
                        required=False, default='out',
                        help='Output directory')
    parser.add_argument('--zip',
                        required=False, default=False, action='store_true',
                        help='Compress the output fits files')
    parser.add_argument('--debug',
                        required=False, default=False, action='store_true',
                        help='Write diagnostics')
    parser.add_argument('--flush',
                        required=False, default=False, action='store_true',
                        help='Flush every print statement.')
    parser.add_argument('--nside',
                        required=False, default=512, type=np.int,
                        help='Healpix NSIDE')
    parser.add_argument('--madam_prefix',
                        required=False, default='toast',
                        help='Output map prefix')
    parser.add_argument('--madam_iter_max',
                        required=False, default=1000, type=np.int,
                        help='Maximum number of CG iterations in Madam')
    parser.add_argument('--madam_baseline_length',
                        required=False, default=10000.0, type=np.float,
                        help='Destriping baseline length (seconds)')
    parser.add_argument('--madam_baseline_order',
                        required=False, default=0, type=np.int,
                        help='Destriping baseline polynomial order')
    parser.add_argument('--madam_noisefilter',
                        required=False, default=False, action='store_true',
                        help='Destripe with the noise filter enabled')
    parser.add_argument('--madampar',
                        required=False, default=None,
                        help='Madam parameter file')
    parser.add_argument('--no_madam_allreduce',
                        required=False, default=False, action='store_true',
                        help='Do not use allreduce communication in Madam')
    parser.add_argument('--common_flag_mask',
                        required=False, default=1, type=np.uint8,
                        help='Common flag mask')
    parser.add_argument('--MC_start',
                        required=False, default=0, type=np.int,
                        help='First Monte Carlo noise realization')
    parser.add_argument('--MC_count',
                        required=False, default=1, type=np.int,
                        help='Number of Monte Carlo noise realizations')
    parser.add_argument('--fp',
                        required=False, default=None,
                        help='Pickle file containing a dictionary of detector '
                        'properties.  The keys of this dict are the detector '
                        'names, and each value is also a dictionary with keys '
                        '"quat" (4 element ndarray), "fwhm" (float, arcmin), '
                        '"fknee" (float, Hz), "alpha" (float), and '
                        '"NET" (float).')
    parser.add_argument('--focalplane_radius',
                        required=False, type=np.float,
                        help='Override focal plane radius [deg]')
    parser.add_argument('--freq',
                        required=True,
                        help='Comma-separated list of frequencies with '
                        'identical focal planes')
    parser.add_argument('--tidas',
                        required=False, default=None,
                        help='Output TIDAS export path')

    args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))

    if len(args.freq.split(',')) != 1:
        # Multi frequency run.  We don't support multiple copies of
        # scanned signal.
        if args.input_map:
            raise RuntimeError('Multiple frequencies are not supported when '
                               'scanning from a map')

    if not args.skip_atmosphere and args.weather is None:
        raise RuntimeError('Cannot simulate atmosphere without a TOAST '
                           'weather file')

    if args.tidas is not None:
        if not tt.tidas_available:
            raise RuntimeError("TIDAS not found- cannot export")

    if comm.comm_world.rank == 0:
        print('\nAll parameters:')
        print(args, flush=args.flush)
        print('')

    if args.groupsize:
        comm = toast.Comm(groupsize=args.groupsize)

    if comm.comm_world.rank == 0:
        if not os.path.isdir(args.outdir):
            try:
                os.makedirs(args.outdir)
            except FileExistsError:
                pass

    return args, comm


def name2id(name, maxval=2**16):
    """ Map a name into an index.

    """
    value = 0
    for c in name:
        value += ord(c)
    return value % maxval


def load_weather(args, comm, schedules):
    """ Load TOAST weather file(s) and attach them to the schedules.

    """
    if args.weather is None:
        return

    start = MPI.Wtime()
    autotimer = timing.auto_timer()
    if comm.comm_world.rank == 0:
        weathers = []
        weatherdict = {}
        for fname in args.weather.split(','):
            if fname not in weatherdict:
                if not os.path.isfile(fname):
                    raise RuntimeError('No such weather file: {}'.format(fname))
                start1 = MPI.Wtime()
                weatherdict[fname] = Weather(fname)
                stop1 = MPI.Wtime()
                print('Load {}: {:.2f} seconds'.format(fname, stop1-start1),
                      flush=args.flush)
            weathers.append(weatherdict[fname])
    else:
        weathers = None

    weathers = comm.comm_world.bcast(weathers)
    if len(weathers) == 1 and len(schedules) > 1:
        weathers *= len(schedules)
    if len(weathers) != len(schedules):
        raise RuntimeError(
            'Number of weathers must equal number of schedules or be 1.')

    for schedule, weather in zip(schedules, weathers):
        schedule.append(weather)

    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Loading weather {:.3f} s'.format(stop-start), flush=args.flush)


def load_schedule(args, comm):
    """ Load the observing schedule(s).

    """
    start = MPI.Wtime()
    autotimer = timing.auto_timer()
    schedules = []

    if comm.comm_world.rank == 0:
        for ischedule, fn in enumerate(args.schedule.split(',')):
            if not os.path.isfile(fn):
                raise RuntimeError('No such schedule file: {}'.format(fn))
            start1 = MPI.Wtime()
            with open(fn, 'r') as f:
                while True:
                    line = f.readline()
                    if line.startswith('#'):
                        continue
                    (site_name, telescope, site_lat, site_lon,
                     site_alt) = line.split()
                    site_alt = float(site_alt)
                    site = (site_name, telescope, site_lat, site_lon, site_alt)
                    break
                all_ces = []
                for line in f:
                    if line.startswith('#'):
                        continue
                    (start_date, start_time, stop_date, stop_time, mjdstart,
                     mjdstop, name, azmin, azmax, el, rs, sun_el1, sun_az1,
                     sun_el2, sun_az2, moon_el1, moon_az1, moon_el2, moon_az2,
                     moon_phase, scan, subscan) = line.split()
                    start_time = start_date + ' ' + start_time
                    stop_time = stop_date + ' ' + stop_time
                    # Define season as a calendar year.  This can be
                    # changed later and could even be in the schedule file.
                    season = int(start_date.split('-')[0])
                    try:
                        start_time = dateutil.parser.parse(start_time+' +0000')
                        stop_time = dateutil.parser.parse(stop_time+' +0000')
                    except:
                        start_time = dateutil.parser.parse(start_time)
                        stop_time = dateutil.parser.parse(stop_time)
                    start_timestamp = start_time.timestamp()
                    stop_timestamp = stop_time.timestamp()
                    all_ces.append([
                        start_timestamp, stop_timestamp, name, float(mjdstart),
                        int(scan), int(subscan), float(azmin), float(azmax),
                        float(el), season, start_date])
            schedules.append([site, all_ces])
            stop1 = MPI.Wtime()
            print('Load {}: {:.2f} seconds'.format(fn, stop1-start1),
                  flush=args.flush)

    schedules = comm.comm_world.bcast(schedules)

    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Loading schedule {:.3f} s'.format(stop-start), flush=args.flush)

    return schedules


def get_focalplane_radius(args, focalplane, rmin=1.0):
    """ Find the furthest angular distance from the boresight

    """
    if args.focalplane_radius:
        return args.focalplane_radius

    autotimer = timing.auto_timer()
    xaxis, yxis, zaxis = np.eye(3)
    cosangs = []
    for det in focalplane:
        quat = focalplane[det]['quat']
        vec = qa.rotate(quat, zaxis)
        cosangs.append(np.dot(zaxis, vec))
    mincos = np.amin(cosangs)
    maxdist = max(np.degrees(np.arccos(mincos)), rmin)

    return maxdist*1.001


def load_focalplanes(args, comm, schedules):
    """ Attach a focalplane to each of the schedules.

    """
    start = MPI.Wtime()
    autotimer = timing.auto_timer()

    # Load focalplane information

    focalplanes = []
    if comm.comm_world.rank == 0:
        for fpfile in args.fp.split(','):
            start1 = MPI.Wtime()
            with open(fpfile, 'rb') as picklefile:
                focalplane = pickle.load(picklefile)
                stop1 = MPI.Wtime()
                print('Load {}:  {:.2f} seconds'.format(fpfile, stop1-start1),
                      flush=args.flush)
                focalplanes.append(focalplane)
                start1 = stop1
    focalplanes = comm.comm_world.bcast(focalplanes)

    if len(focalplanes) == 1 and len(schedules) > 1:
        focalplanes *= len(schedules)
    if len(focalplanes) != len(schedules):
        raise RuntimeError(
            'Number of focalplanes must equal number of schedules or be 1.')

    detweights = {}
    for schedule, focalplane in zip(schedules, focalplanes):
        schedule.append(focalplane)
        for detname, det in focalplane.items():
            net = det['NET']
            detweight = 1.0 / (args.samplerate * net * net)
            if detname in detweights and detweights[detname] != detweight:
                raise RuntimeError(
                    'Detector weight for {} changes'.format(detname))
            detweights[detname] = detweight

    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Load focalplane(s):  {:.2f} seconds'.format(stop-start),
              flush=args.flush)

    return detweights


def get_analytic_noise(args, focalplane):
    """ Create a TOAST noise object.

    Create a noise object from the 1/f noise parameters contained in the
    focalplane database.

    """
    autotimer = timing.auto_timer()
    detectors = sorted(focalplane.keys())
    fmin = {}
    fknee = {}
    alpha = {}
    NET = {}
    rates = {}
    for d in detectors:
        rates[d] = args.samplerate
        fmin[d] = focalplane[d]['fmin']
        fknee[d] = focalplane[d]['fknee']
        alpha[d] = focalplane[d]['alpha']
        NET[d] = focalplane[d]['NET']

    return tt.AnalyticNoise(rate=rates, fmin=fmin, detectors=detectors,
                            fknee=fknee, alpha=alpha, NET=NET)


def get_breaks(comm, all_ces, nces, args):
    """ List operational day limits in the list of CES:s.

    """
    autotimer = timing.auto_timer()
    breaks = []
    do_break = False
    for i in range(nces-1):
        # If current and next CES are on different days, insert a break
        tz = args.timezone / 24
        start1 = all_ces[i][3] # MJD start
        start2 = all_ces[i+1][3] # MJD start
        scan1 = all_ces[i][4]
        scan2 = all_ces[i+1][4]
        if scan1 != scan2 and do_break:
            breaks.append(nces + i + 1)
            do_break = False
            continue
        day1 = int(start1 + tz)
        day2 = int(start2 + tz)
        if day1 != day2:
            if scan1 == scan2:
                # We want an entire CES, even if it crosses the day bound.
                # Wait until the scan number changes.
                do_break = True
            else:
                breaks.append(nces + i + 1)

    nbreak = len(breaks)
    if nbreak != comm.ngroups-1:
        raise RuntimeError(
            'Number of observing days ({}) does not match number of process '
            'groups ({}).'.format(nbreak+1, comm.ngroups))

    return breaks


def create_observation(args, comm, all_ces_tot, ices, noise):
    """ Create a TOAST observation.

    Create an observation for the CES scan defined by all_ces_tot[ices].

    """
    autotimer = timing.auto_timer()
    ces, site, fp, fpradius, detquats, weather = all_ces_tot[ices]

    (CES_start, CES_stop, CES_name, mjdstart, scan, subscan,
     azmin, azmax, el, season, date) = ces

    site_name, telescope, site_lat, site_lon, site_alt = site

    totsamples = int((CES_stop - CES_start) * args.samplerate)

    # create the TOD for this observation

    try:
        tod = tt.TODGround(
            comm.comm_group, detquats, totsamples,
            detranks=comm.comm_group.size, firsttime=CES_start,
            rate=args.samplerate, site_lon=site_lon, site_lat=site_lat,
            site_alt=site_alt, azmin=azmin, azmax=azmax, el=el,
            scanrate=args.scanrate, scan_accel=args.scan_accel,
            CES_start=None, CES_stop=None, sun_angle_min=args.sun_angle_min,
            coord=args.coord, sampsizes=None)
    except RuntimeError as e:
        raise RuntimeError('Failed to create TOD for {}-{}-{}: "{}"'
                           ''.format(CES_name, scan, subscan, e))

    # Create the observation

    site_name = site[0]
    telescope_name = site[1]
    site_id = name2id(site_name)
    telescope_id = name2id(telescope_name)

    obs = {}
    obs['name'] = 'CES-{}-{}-{}-{}-{}'.format(site_name, telescope_name,
                                              CES_name, scan, subscan)
    obs['tod'] = tod
    obs['baselines'] = None
    obs['noise'] = noise
    obs['id'] = int(mjdstart * 10000)
    obs['intervals'] = tod.subscans
    obs['site'] = site_name
    obs['telescope'] = telescope_name
    obs['site_id'] = site_id
    obs['telescope_id'] = telescope_id
    obs['fpradius'] = fpradius
    obs['weather'] = weather
    obs['start_time'] = CES_start
    obs['altitude'] = site_alt
    obs['season'] = season
    obs['date'] = date
    obs['MJD'] = mjdstart
    obs['focalplane'] = fp

    return obs


def create_observations(args, comm, schedules, counter):
    """ Create and distribute TOAST observations for every CES in schedules.

    """
    start = MPI.Wtime()
    autotimer = timing.auto_timer()

    data = toast.Data(comm)

    # Loop over the schedules, distributing each schedule evenly across
    # the process groups.  For now, we'll assume that each schedule has
    # the same number of operational days and the number of process groups
    # matches the number of operational days.  Relaxing these constraints
    # will cause the season break to occur on different process groups
    # for different schedules and prevent splitting the communicator.

    for schedule in schedules:

        if args.weather is None:
            site, all_ces, focalplane = schedule
            weather = None
        else:
            site, all_ces, weather, focalplane = schedule

        fpradius = get_focalplane_radius(args, focalplane)

        # Focalplane information for this schedule
        detectors = sorted(focalplane.keys())
        detquats = {}
        for d in detectors:
            detquats[d] = focalplane[d]['quat']

        # Noise model for this schedule
        noise = get_analytic_noise(args, focalplane)

        all_ces_tot = []
        nces = len(all_ces)
        for ces in all_ces:
            all_ces_tot.append((ces, site, focalplane, fpradius,
                                detquats, weather))

        breaks = get_breaks(comm, all_ces, nces, args)

        groupdist = toast.distribute_uniform(nces, comm.ngroups, breaks=breaks)
        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ices in range(group_firstobs, group_firstobs + group_numobs):
            obs = create_observation(args, comm, all_ces_tot, ices, noise)
            data.obs.append(obs)

    if args.skip_atmosphere:
        for ob in data.obs:
            tod = ob['tod']
            tod.free_azel_quats()

    if comm.comm_group.rank == 0:
        print('Group # {:4} has {} observations.'.format(
            comm.group, len(data.obs)), flush=args.flush)

    if len(data.obs) == 0:
        raise RuntimeError('Too many tasks. Every MPI task must '
                           'be assigned to at least one observation.')

    counter.exec(data)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Simulated scans in {:.2f} seconds'
              ''.format(stop-start), flush=args.flush)

    # Split the data object for each telescope for separate mapmaking.
    # We could also split by site.

    if len(schedules) > 1:
        telescope_data = data.split('telescope')
        if len(telescope_data) == 1:
            # Only one telescope available
            telescope_data = []
    else:
        telescope_data = []
    telescope_data.insert(0, ('all', data))

    return data, telescope_data


def expand_pointing(args, comm, data, counter):
    """ Expand boresight pointing to every detector.

    """
    start = MPI.Wtime()
    autotimer = timing.auto_timer()

    hwprpm = args.hwprpm
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = args.hwpsteptime

    npix = 12 * args.nside**2

    if comm.comm_world.rank == 0:
        print('Expanding pointing', flush=args.flush)

    pointing = tt.OpPointingHpix(
        nside=args.nside, nest=True, mode='IQU',
        hwprpm=hwprpm, hwpstep=hwpstep, hwpsteptime=hwpsteptime)

    pointing.exec(data)

    # Only purge the pointing if we are NOT going to export the
    # data to a TIDAS volume
    if args.tidas is None:
        for ob in data.obs:
            tod = ob['tod']
            tod.free_radec_quats()

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Pointing generation took {:.3f} s'.format(stop-start),
              flush=args.flush)

    counter.exec(data)
    return


def get_submaps(args, comm, data):
    """ Get a list of locally hit pixels and submaps on every process.

    """
    if not args.skip_bin or args.input_map:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Scanning local pixels', flush=args.flush)
        start = MPI.Wtime()

        # Prepare for using distpixels objects
        nside = args.nside
        subnside = 16
        if subnside > nside:
            subnside = nside
        subnpix = 12 * subnside * subnside

        # get locally hit pixels
        lc = tm.OpLocalPixels()
        localpix = lc.exec(data)
        if localpix is None:
            raise RuntimeError(
                'Process {} has no hit pixels. Perhaps there are fewer '
                'detectors than processes in the group?'.format(
                    comm.comm_world.rank))

        # find the locally hit submaps.
        localsm = np.unique(np.floor_divide(localpix, subnpix))

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Local submaps identified in {:.3f} s'.format(elapsed),
                  flush=args.flush)
    else:
        localpix, localsm = None, None

    return localpix, localsm, subnpix


def add_sky_signal(args, comm, data, totalname_freq, signalname):
    """ Add previously simulated sky signal to the atmospheric noise.

    """
    if signalname is not None:
        autotimer = timing.auto_timer()
        for obs in data.obs:
            tod = obs['tod']
            for det in tod.local_dets:
                cachename_in = '{}_{}'.format(signalname, det)
                cachename_out = '{}_{}'.format(totalname_freq, det)
                ref_in = tod.cache.reference(cachename_in)
                if tod.cache.exists(cachename_out):
                    ref_out = tod.cache.reference(cachename_out)
                    ref_out += ref_in
                else:
                    ref_out = tod.cache.put(cachename_out, ref_in)
                del ref_in, ref_out

    return


def simulate_sky_signal(args, comm, data, counter, schedules, subnpix, localsm):
    """ Use PySM to simulate smoothed sky signal.

    """
    # Convolve a signal TOD from PySM
    start = MPI.Wtime()
    signalname = 'signal'
    op_sim_pysm = ttm.OpSimPySM(comm=comm.comm_rank,
                               out=signalname,
                               pysm_model=args.input_pysm_model,
                               focalplanes=[s[3] for s in schedules],
                               nside=args.nside,
                               subnpix=subnpix, localsm=localsm,
                               apply_beam=args.apply_beam)
    op_sim_pysm.exec(data)
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('PySM took {:.2f} seconds'.format(stop-start),
              flush=args.flush)

    counter.exec(data)

    return signalname


def scan_sky_signal(args, comm, data, counter, localsm, subnpix):
    """ Scan sky signal from a map.

    """
    signalname = None

    if args.input_map:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Scanning input map', flush=args.flush)
        start = MPI.Wtime()

        npix = 12*args.nside**2

        # Scan the sky signal
        if  comm.comm_world.rank == 0 and not os.path.isfile(args.input_map):
            raise RuntimeError(
                'Input map does not exist: {}'.format(args.input_map))
        distmap = tm.DistPixels(
            comm=comm.comm_world, size=npix, nnz=3,
            dtype=np.float32, submap=subnpix, local=localsm)
        counter._objects.append(distmap)
        distmap.read_healpix_fits(args.input_map)
        scansim = tt.OpSimScan(distmap=distmap, out='signal')
        scansim.exec(data)

        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Read and sampled input map:  {:.2f} seconds'
                  ''.format(stop-start), flush=args.flush)
        signalname = 'signal'

        counter.exec(data)

    return signalname


def setup_sigcopy(args, comm):
    """ Determine if an extra copy of the atmospheric signal is needed.

    When we simulate multichroic focal planes, the frequency-independent
    part of the atmospheric noise is simulated first and then the
    frequency scaling is applied to a copy of the atmospheric noise.
    """
    if len(args.freq.split(',')) == 1:
        totalname = 'total'
        totalname_freq = 'total'
    else:
        totalname = 'total'
        totalname_freq = 'total_freq'

    return totalname, totalname_freq


def setup_madam(args, comm):
    """ Create a Madam parameter dictionary.

    Initialize the Madam parameters from the command line arguments.

    """
    autotimer = timing.auto_timer()
    pars = {}

    cross = args.nside // 2
    submap = 16
    if submap > args.nside:
        submap = args.nside

    pars['temperature_only'] = False
    pars['force_pol'] = True
    pars['kfirst'] = not args.skip_destripe
    pars['write_map'] = not args.skip_destripe
    pars['write_binmap'] = not args.skip_bin
    pars['write_matrix'] = not args.skip_hits
    pars['write_wcov'] = not args.skip_hits
    pars['write_hits'] = not args.skip_hits
    pars['nside_cross'] = cross
    pars['nside_submap'] = submap
    pars['allreduce'] = not args.no_madam_allreduce
    pars['pixlim_cross'] = 1e-3
    pars['pixmode_cross'] = 2
    pars['pixlim_map'] = 1e-2
    pars['pixmode_map'] = 2
    # Instead of fixed detector weights, we'll want to use scaled noise
    # PSD:s that include the atmospheric noise
    pars['radiometers'] = True
    pars['noise_weights_from_psd'] = True

    if args.madampar is not None:
        pat = re.compile(r'\s*(\S+)\s*=\s*(\S+(\s+\S+)*)\s*')
        comment = re.compile(r'^#.*')
        with open(args.madampar, 'r') as f:
            for line in f:
                if comment.match(line) is None:
                    result = pat.match(line)
                    if result is not None:
                        key, value = result.group(1), result.group(2)
                        pars[key] = value

    pars['base_first'] = args.madam_baseline_length
    pars['basis_order'] = args.madam_baseline_order
    pars['nside_map'] = args.nside
    if args.madam_noisefilter:
        if args.madam_baseline_order != 0:
            raise RuntimeError('Madam cannot build a noise filter when baseline'
                               'order is higher than zero.')
        pars['kfilter'] = True
    else:
        pars['kfilter'] = False
    pars['precond_width'] = 1
    pars['fsample'] = args.samplerate
    pars['iter_max'] = args.madam_iter_max
    pars['file_root'] = args.madam_prefix

    return pars


def scale_atmosphere_by_frequency(args, comm, data, freq, totalname_freq, mc):
    """ Scale atmospheric fluctuations by frequency.

    Assume that cached signal under totalname_freq is pure atmosphere
    and scale the absorption coefficient according to the frequency.

    If the focalplane is included in the observation and defines
    bandpasses for the detectors, the scaling is computed for each
    detector separately.

    """
    if args.skip_atmosphere:
        return

    autotimer = timing.auto_timer()
    start = MPI.Wtime()
    for obs in data.obs:
        tod = obs['tod']
        todcomm = tod.mpicomm
        site_id = obs['site_id']
        weather = obs['weather']
        if 'focalplane' in obs:
            focalplane = obs['focalplane']
        else:
            focalplane = None
        start_time = obs['start_time']
        weather.set(site_id, mc, start_time)
        altitude = obs['altitude']
        air_temperature = weather.air_temperature
        surface_pressure = weather.surface_pressure
        pwv = weather.pwv
        # Use the entire processing group to sample the absorption
        # coefficient as a function of frequency
        ntask = todcomm.size
        freqmin = 0
        freqmax = 2 * freq
        nfreq = 1001
        freqstep = (freqmax - freqmin) / (nfreq - 1)
        nfreq_task = int(nfreq // todcomm.size) + 1
        my_ifreq_min = nfreq_task * todcomm.rank
        my_ifreq_max = min(nfreq, nfreq_task*(todcomm.rank+1))
        my_nfreq = my_ifreq_max - my_ifreq_min
        if my_nfreq > 0:
            my_freqs = freqmin + np.arange(my_ifreq_min, my_ifreq_max)*freqstep
            my_absorption = np.zeros(my_nfreq)
            err = toast.ctoast.atm_get_absorption_coefficient_vec(
                altitude, air_temperature, surface_pressure, pwv,
                my_freqs[0], my_freqs[-1], my_nfreq, my_absorption)
            if err != 0:
                raise RuntimeError(
                    'Failed to get absorption coefficient vector')
        else:
            my_freqs = np.array([])
            my_absorption = np.array([])
        freqs = np.hstack(todcomm.allgather(my_freqs))
        absorption = np.hstack(todcomm.allgather(my_absorption))
        #loading = toast.ctoast.atm_get_atmospheric_loading(
        #    altitude, pwv, freq)
        for det in tod.local_dets:
            try:
                # Use detector bandpass from the focalplane
                center = focalplane[det]['bandcenter_ghz']
                width = focalplane[det]['bandwidth_ghz']
            except:
                # Use default values for the entire focalplane
                center = freq
                width = .2 * freq
            nstep = 101
            # Interpolate the absorption coefficient to do a top hat
            # integral across the bandpass
            det_freqs = np.linspace(center-width/2, center+width/2, nstep)
            absorption_det = np.mean(np.interp(det_freqs, freqs, absorption))
            cachename = '{}_{}'.format(totalname_freq, det)
            ref = tod.cache.reference(cachename)
            ref *= absorption_det
            del ref

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Atmosphere scaling took {:.3f} s'.format(stop-start),
              flush=args.flush)

    return


def update_atmospheric_noise_weights(args, comm, data, freq, mc):
    """ Update atmospheric noise weights.

    Estimate the atmospheric noise level from weather parameters and
    encode it as a noise_scale in the observation.  Madam will apply
    the noise_scale to the detector weights.  This approach assumes
    that the atmospheric noise dominates over detector noise.  To be
    more precise, we would have to add the squared noise weights but
    we do not have their relative calibration.

    """
    if args.weather:
        autotimer = timing.auto_timer()
        start = MPI.Wtime()
        for obs in data.obs:
            tod = obs['tod']
            site_id = obs['site_id']
            weather = obs['weather']
            start_time = obs['start_time']
            weather.set(site_id, mc, start_time)
            altitude = obs['altitude']
            absorption = toast.ctoast.atm_get_absorption_coefficient(
                altitude, weather.air_temperature, weather.surface_pressure,
                weather.pwv, freq)
            obs['noise_scale'] = absorption * weather.air_temperature
        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Atmosphere weighting took {:.3f} s'.format(stop-start),
                  flush=args.flush)

    else:
        for obs in data.obs:
            obs['noise_scale'] = 1.

    return


def simulate_atmosphere(args, comm, data, mc, counter,
                        totalname):
    if not args.skip_atmosphere:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Simulating atmosphere', flush=args.flush)
            if args.atm_cache and not os.path.isdir(args.atm_cache):
                try:
                    os.makedirs(args.atm_cache)
                except FileExistsError:
                    pass
        start = MPI.Wtime()

        # Simulate the atmosphere signal
        atm = tt.OpSimAtmosphere(
            out=totalname, realization=mc,
            lmin_center=args.atm_lmin_center,
            lmin_sigma=args.atm_lmin_sigma,
            lmax_center=args.atm_lmax_center, gain=args.atm_gain,
            lmax_sigma=args.atm_lmax_sigma, zatm=args.atm_zatm,
            zmax=args.atm_zmax, xstep=args.atm_xstep,
            ystep=args.atm_ystep, zstep=args.atm_zstep,
            nelem_sim_max=args.atm_nelem_sim_max,
            verbosity=int(args.debug), gangsize=args.atm_gangsize,
            z0_center=args.atm_z0_center, z0_sigma=args.atm_z0_sigma,
            apply_flags=False, common_flag_mask=args.common_flag_mask,
            cachedir=args.atm_cache, flush=args.flush,
            wind_time=args.atm_wind_time)

        atm.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Atmosphere simulation took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)

    return


def copy_atmosphere(args, comm, data, counter, totalname, totalname_freq):
    """ Copy the atmospheric signal.

    Make a copy of the atmosphere so we can scramble the gains and apply
    frequency-dependent scaling.

    """
    if totalname != totalname_freq:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Copying atmosphere from {} to {}'.format(
                totalname, totalname_freq), flush=args.flush)
        cachecopy = tt.OpCacheCopy(totalname, totalname_freq, force=True)
        cachecopy.exec(data)
        counter.exec(data)
    return


def simulate_noise(args, comm, data, mc, counter, totalname_freq):
    if not args.skip_noise:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Simulating noise', flush=args.flush)
        start = MPI.Wtime()

        nse = tt.OpSimNoise(out=totalname_freq, realization=mc)
        nse.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Noise simulation took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)
    return


def scramble_gains(args, comm, data, mc, counter, totalname_freq):
    if args.gain_sigma:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Scrambling gains', flush=args.flush)
        start = MPI.Wtime()

        scrambler = tt.OpGainScrambler(
            sigma=args.gain_sigma, name=totalname_freq, realization=mc)
        scrambler.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Gain scrambling took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)
    return


def setup_output(args, comm, mc, freq):
    outpath = '{}/{:08}/{:03}'.format(args.outdir, mc, int(freq))
    if comm.comm_world.rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


def apply_polyfilter(args, comm, data, counter, totalname_freq):
    if args.polyorder:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Polyfiltering signal', flush=args.flush)
        start = MPI.Wtime()
        polyfilter = tt.OpPolyFilter(
            order=args.polyorder, name=totalname_freq,
            common_flag_mask=args.common_flag_mask)
        polyfilter.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Polynomial filtering took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)
    return


def apply_groundfilter(args, comm, data, counter, totalname_freq):
    if args.wbin_ground:
        autotimer = timing.auto_timer()
        if comm.comm_world.rank == 0:
            print('Ground filtering signal', flush=args.flush)
        start = MPI.Wtime()
        groundfilter = tt.OpGroundFilter(
            wbin=args.wbin_ground, name=totalname_freq,
            common_flag_mask=args.common_flag_mask)
        groundfilter.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Ground filtering took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)
    return


def output_tidas(args, comm, data, totalname):
    if args.tidas is None:
        return
    autotimer = timing.auto_timer()
    from toast.tod.tidas import OpTidasExport
    tidas_path = os.path.abspath(args.tidas)
    comm.comm_world.Barrier()
    if comm.comm_world.rank == 0:
        print('Exporting TOD to a TIDAS volume at {}'.format(tidas_path),
              flush=args.flush)
    start = MPI.Wtime()

    export = OpTidasExport(tidas_path, name=totalname, usedist=True)
    export.exec(data)

    comm.comm_world.Barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Wrote simulated TOD to {}:{} in {:.2f} s'
              ''.format(tidas_path, totalname,
                        stop-start), flush=args.flush)
    return


def get_time_communicators(comm, data):
    """ Split the world communicator by time.

    """
    autotimer = timing.auto_timer()
    time_comms = [('all', comm.comm_world)]

    # A process will only have data for one season and one day.  If more
    # than one season is observed, we split the communicator to make
    # season maps.

    my_season = data.obs[0]['season']
    seasons = np.array(comm.comm_world.allgather(my_season))
    do_seasons = np.any(seasons != my_season)
    if do_seasons:
        season_comm = comm.comm_world.Split(my_season, comm.comm_world.rank)
        time_comms.append((str(my_season), season_comm))

    # Split the communicator to make daily maps.  We could easily split
    # by month as well

    my_day = int(data.obs[0]['MJD'])
    my_date = data.obs[0]['date']
    days = np.array(comm.comm_world.allgather(my_day))
    do_days = np.any(days != my_day)
    if do_days:
        day_comm = comm.comm_world.Split(my_day, comm.comm_world.rank)
        time_comms.append((my_date, day_comm))

    return time_comms


def apply_madam(args, comm, time_comms, data, telescope_data, freq, madampars,
                counter, mc, firstmc, outpath, detweights, totalname_madam,
                first_call=True, extra_prefix=None):
    """ Use libmadam to bin and optionally destripe data.

    Bin and optionally destripe all conceivable subsets of the data.

    """
    if comm.comm_world.rank == 0:
        print('Making maps', flush=args.flush)
    start = MPI.Wtime()
    autotimer = timing.auto_timer()

    pars = copy.deepcopy(madampars)
    pars['path_output'] = outpath
    file_root = pars['file_root']
    if len(file_root) > 0 and not file_root.endswith('_'):
        file_root += '_'
    if extra_prefix is not None:
        file_root += '{}_'.format(extra_prefix)
    file_root += '{:03}'.format(int(freq))

    if first_call:
        if mc != firstmc:
            pars['write_matrix'] = False
            pars['write_wcov'] = False
            pars['write_hits'] = False
    else:
        pars['kfirst'] = False
        pars['write_map'] = False
        pars['write_binmap'] = True
        pars['write_matrix'] = False
        pars['write_wcov'] = False
        pars['write_hits'] = False

    outputs = [pars['write_map'], pars['write_binmap'], pars['write_hits'],
               pars['write_wcov'], pars['write_matrix']]
    if not np.any(outputs):
        if comm.comm_world.rank == 0:
            print('No Madam outputs requested.  Skipping.', flush=args.flush)
        return

    madam = tm.OpMadam(
        params=pars, detweights=detweights,
        name=totalname_madam,
        common_flag_mask=args.common_flag_mask,
        purge_tod=False)

    if 'info' in madam.params:
        info = madam.params['info']
    else:
        info = 3

    for time_name, time_comm in time_comms:
        for tele_name, tele_data in telescope_data:
            if len(time_name.split('-')) == 3:
                # Special rules for daily maps
                if args.skip_daymaps:
                    continue
                if ((len(telescope_data) > 1) and (tele_name == 'all')):
                    # Skip daily maps over multiple telescopes
                    continue
                if first_call:
                    # Do not destripe daily maps
                    kfirst_save = pars['kfirst']
                    write_map_save = pars['write_map']
                    write_binmap_save = pars['write_binmap']
                    pars['kfirst'] = False
                    pars['write_map'] = False
                    pars['write_binmap'] = True

            start1 = MPI.Wtime()
            madam.params['file_root'] = '{}_telescope_{}_time_{}'.format(
                file_root, tele_name, time_name)
            if time_comm == comm.comm_world:
                madam.params['info'] = info
            else:
                # Cannot have verbose output from concurrent mapmaking
                madam.params['info'] = 0
            if time_comm.rank == 0:
                print('Mapping {}'.format(madam.params['file_root']),
                      flush=args.flush)
            madam.exec(tele_data, time_comm)
            time_comm.barrier()
            stop1 = MPI.Wtime()
            if time_comm.rank == 0:
                print('Mapping {} took {:.3f} s'.format(
                    madam.params['file_root'], stop1-start1), flush=args.flush)
            if len(time_name.split('-')) == 3 and first_call:
                # Restore destriping parameters
                pars['kfirst'] = kfirst_save
                pars['write_map'] = write_map_save
                pars['write_binmap'] = write_binmap_save

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Madam took {:.3f} s'.format(stop-start), flush=args.flush)

    counter.exec(data)

    return


def main():

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print('Running with {} processes at {}'.format(
            comm.comm_world.size, str(datetime.now())), flush=True)

    global_timer = timing.simple_timer("Total time")
    global_timer.start()

    args, comm = parse_arguments(comm)

    autotimer = timing.auto_timer("@{}".format(timing.FILE()))

    # Initialize madam parameters

    madampars = setup_madam(args, comm)

    # Load and broadcast the schedule file

    schedules = load_schedule(args, comm)

    # Load the weather and append to schedules

    load_weather(args, comm, schedules)

    # load or simulate the focalplane

    detweights = load_focalplanes(args, comm, schedules)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    counter = tt.OpMemoryCounter()

    data, telescope_data = create_observations(args, comm, schedules, counter)

    # Split the communicator for day and season mapmaking

    time_comms = get_time_communicators(comm, data)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data, counter)

    # Prepare auxiliary information for distributed map objects

    localpix, localsm, subnpix = get_submaps(args, comm, data)

    if args.input_pysm_model:
        signalname = simulate_sky_signal(args, comm, data, counter,
                                         schedules, subnpix, localsm)
    else:
        signalname = scan_sky_signal(args, comm, data, counter, localsm,
                                     subnpix)

    # Set up objects to take copies of the TOD at appropriate times

    totalname, totalname_freq = setup_sigcopy(args, comm)

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    freqs = [float(freq) for freq in args.freq.split(',')]
    nfreq = len(freqs)

    for mc in range(firstmc, firstmc+nmc):

        simulate_atmosphere(args, comm, data, mc, counter, totalname)

        # Loop over frequencies with identical focal planes and identical
        # atmospheric noise.

        for ifreq, freq in enumerate(freqs):

            if comm.comm_world.rank == 0:
                print('Processing frequency {}GHz {} / {}, MC = {}'
                      ''.format(freq, ifreq+1, nfreq, mc), flush=args.flush)

            copy_atmosphere(args, comm, data, counter, totalname, totalname_freq)

            scale_atmosphere_by_frequency(args, comm, data, freq,
                                          totalname_freq, mc)

            update_atmospheric_noise_weights(args, comm, data, freq, mc)

            add_sky_signal(args, comm, data, totalname_freq, signalname)

            mcoffset = ifreq * 1000000

            simulate_noise(args, comm, data, mc+mcoffset, counter,
                           totalname_freq)

            scramble_gains(args, comm, data, mc+mcoffset, counter,
                           totalname_freq)

            if (mc == firstmc) and (ifreq == 0):
                # For the first realization and frequency, optionally
                # export the timestream data to a TIDAS volume.
                output_tidas(args, comm, data, totalname)

            outpath = setup_output(args, comm, mc, freq)

            # Bin and destripe maps

            apply_madam(args, comm, time_comms, data, telescope_data, freq,
                        madampars, counter, mc+mcoffset, firstmc, outpath,
                        detweights, totalname_freq,
                        first_call=True)

            if args.polyorder or args.wbin_ground:

                # Filter signal

                apply_polyfilter(args, comm, data, counter, totalname_freq)

                apply_groundfilter(args, comm, data, counter, totalname_freq)

                # Bin maps

                apply_madam(args, comm, time_comms, data, telescope_data, freq,
                            madampars, counter, mc+mcoffset, firstmc, outpath,
                            detweights, totalname_freq, first_call=False,
                            extra_prefix='filtered')

    counter.exec(data)

    comm.comm_world.barrier()
    global_timer.stop()
    if comm.comm_world.rank == 0:
        global_timer.report()


if __name__ == '__main__':
    try:
        main()
        tman = timing.timing_manager()
        tman.report()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        if MPI.COMM_WORLD.size == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('*** print_tb:')
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print('*** print_exception:')
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=5, file=sys.stdout)
        print('*** print_exc:')
        traceback.print_exc()
        print('*** format_exc, first and last line:')
        formatted_lines = traceback.format_exc().splitlines()
        print(formatted_lines[0])
        print(formatted_lines[-1])
        print('*** format_exception:')
        print(repr(traceback.format_exception(exc_type, exc_value,
                                              exc_traceback)))
        print('*** extract_tb:')
        print(repr(traceback.extract_tb(exc_traceback)))
        print('*** format_tb:')
        print(repr(traceback.format_tb(exc_traceback)))
        print('*** tb_lineno:', exc_traceback.tb_lineno, flush=True)
        toast.raise_error(6) # typical error code for SIGABRT
        MPI.COMM_WORLD.Abort(6)
