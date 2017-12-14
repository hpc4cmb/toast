#!/usr/bin/env python3

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from toast.mpi import MPI

import os
import sys
import re
import argparse
import pickle
from datetime import datetime
import dateutil.parser
import traceback

import numpy as np
from scipy.constants import degree

import toast
import toast.tod as tt
import toast.map as tm
import toast.qarray as qa
from toast import Weather


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
                        required=True,
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

    parser.add_argument('--fp_radius',
                        required=False, default=1, type=np.float,
                        help='Focal plane radius assumed in the atmospheric '
                        'simulation.')
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
                        help='Minimum time to simulate without discontinuity')
    parser.add_argument('--atm_w_center',
                        required=False, default=1.0, type=np.float,
                        help='central value of the wind speed distribution')
    parser.add_argument('--atm_w_sigma',
                        required=False, default=0.1, type=np.float,
                        help='sigma of the wind speed distribution')
    parser.add_argument('--atm_wdir_center',
                        required=False, default=0.0, type=np.float,
                        help='central value of the wind direction distribution')
    parser.add_argument('--atm_wdir_sigma',
                        required=False, default=100.0, type=np.float,
                        help='sigma of the wind direction distribution')
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
    parser.add_argument('--madam',
                        required=False, default=False, action='store_true',
                        help='If specified, use libmadam for map-making')
    parser.add_argument('--madampar',
                        required=False, default=None,
                        help='Madam parameter file')
    parser.add_argument('--madam_allreduce',
                        required=False, default=False, action='store_true',
                        help='Use allreduce communication in Madam')
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
    parser.add_argument('--freq',
                        required=True,
                        help='Comma-separated list of frequencies with '
                        'identical focal planes')
    parser.add_argument('--tidas',
                        required=False, default=None,
                        help='Output TIDAS export path')

    args = parser.parse_args()

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
                        float(el)])
            schedules.append([site, all_ces])
            stop1 = MPI.Wtime()
            print('Load {}: {:.2f} seconds'.format(fn, stop1-start1),
                  flush=args.flush)

    schedules = comm.comm_world.bcast(schedules)

    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Loading schedule {:.3f} s'.format(stop-start), flush=args.flush)

    return schedules


def load_fp(args, comm, schedules):
    """ Attach a focalplane to each of the schedules.

    """
    start = MPI.Wtime()

    # Load focalplane information

    focalplanes = []
    if comm.comm_world.rank == 0:
        for fpfile in args.fp.split(','):
            start1 = MPI.Wtime()
            with open(fpfile, 'rb') as p:
                fp = pickle.load(p)
                stop1 = MPI.Wtime()
                print('Load {}:  {:.2f} seconds'.format(fpfile, stop1-start1),
                      flush=args.flush)
                focalplanes.append(fp)
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


def create_observations(args, comm, schedules, counter):
    start = MPI.Wtime()

    data = toast.Data(comm)

    nces_tot = 0
    breaks = []
    all_ces_tot = []

    for schedule in schedules:
        if args.weather is None:
            site, all_ces, fp = schedule
            weather = None
        else:
            site, all_ces, weather, fp = schedule
            
        if nces_tot != 0:
            breaks.append(nces_tot)

        # Focalplane information for this schedule
        detectors = sorted(fp.keys())
        detquats = {}
        for d in detectors:
            detquats[d] = fp[d]['quat']

        # Noise model for this schedule
        fmin = {}
        fknee = {}
        alpha = {}
        NET = {}
        rates = {}
        for d in detectors:
            rates[d] = args.samplerate
            fmin[d] = fp[d]['fmin']
            fknee[d] = fp[d]['fknee']
            alpha[d] = fp[d]['alpha']
            NET[d] = fp[d]['NET']
        noise = tt.AnalyticNoise(rate=rates, fmin=fmin, detectors=detectors,
                                 fknee=fknee, alpha=alpha, NET=NET)

        nces = len(all_ces)
        for ces in all_ces:
            all_ces_tot.append((ces, site, fp, detquats, weather))

        do_break = False
        for i in range(nces-1):
            # If current and next CES are on different days, insert a break
            tz = args.timezone / 24
            start1 = all_ces[i][3] # MJD start
            start2 = all_ces[i+1][3] # MJD start
            scan1 = all_ces[i][4]
            scan2 = all_ces[i+1][4]
            if scan1 != scan2 and do_break:
                breaks.append(nces_tot + i + 1)
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
                    breaks.append(nces_tot + i + 1)

        nces_tot += nces

    nbreak = len(breaks)
    if nbreak != comm.ngroups-1:
        raise RuntimeError(
            'Number of observing days ({}) does not match number of process '
            'groups ({}).'.format(nbreak+1, comm.ngroups))

    groupdist = toast.distribute_uniform(nces_tot, comm.ngroups, breaks=breaks)
    group_firstobs = groupdist[comm.group][0]
    group_numobs = groupdist[comm.group][1]

    for ices in range(group_firstobs, group_firstobs + group_numobs):
        ces, site, fp, detquats, weather = all_ces_tot[ices]

        (CES_start, CES_stop, CES_name, mjdstart, scan, subscan,
         azmin, azmax, el) = ces

        site_name, telescope, site_lat, site_lon, site_alt = site

        totsamples = int((CES_stop - CES_start) * args.samplerate)

        # create the single TOD for this observation

        # FIXME: TOD must know the PWV distribution and set
        # tod.meta['pwv_center'] and tod.meta['pwv_sigma']

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
            print('Failed to create the CES scan: {}'.format(e),
                  flush=args.flush)
            continue

        # Create the (single) observation

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
        obs['site'] = site_id
        obs['telescope'] = telescope_id
        obs['weather'] = weather
        obs['start_time'] = CES_start
        obs['altitude'] = site_alt

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

    # Report the memory allocated for the TOAST caches.


    return data


def expand_pointing(args, comm, data, counter):
    start = MPI.Wtime()

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
    if not args.skip_bin or args.input_map:
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


def scan_signal(args, comm, data, counter, localsm, subnpix):
    signalname = None

    if args.input_map:
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


def setup_sigcopy(args, comm, signalname):
    # Operator for signal copying, used in each MC iteration

    if len(args.freq.split(',')) == 1:
        totalname = 'total'
        totalname_freq = 'total'
    else:
        totalname = 'total'
        totalname_freq = 'total_freq'

    if args.skip_bin:
        totalname_madam = totalname_freq
    else:
        totalname_madam = 'total_madam'

    if signalname is not None:
        sigcopy = tt.OpCacheCopy(signalname, totalname)
    else:
        sigcopy = None

    if totalname != totalname_freq:
        sigcopy_freq = tt.OpCacheCopy(totalname, totalname_freq, force=True)
    else:
        sigcopy_freq = None

    if args.madam and totalname_freq != totalname_madam:
        sigcopy_madam = tt.OpCacheCopy(totalname_freq, totalname_madam)
        sigclear = tt.OpCacheClear(totalname_freq)
    else:
        sigcopy_madam = None
        sigclear = None

    return sigcopy, sigcopy_freq, sigcopy_madam, sigclear, \
        totalname, totalname_freq, totalname_madam


def build_npp(args, comm, data, counter, localsm, subnpix, detweights,
              flag_name, common_flag_name):

    if not args.skip_bin:

        if comm.comm_world.rank == 0:
            print('Preparing distributed map', flush=args.flush)
        start0 = MPI.Wtime()
        start = start0

        npix = 12*args.nside**2

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        invnpp = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=6,
                               dtype=np.float64, submap=subnpix, local=localsm)
        counter._objects.append(invnpp)
        invnpp.data.fill(0.0)

        hits = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=1,
                             dtype=np.int64, submap=subnpix, local=localsm)
        counter._objects.append(hits)
        hits.data.fill(0)

        zmap = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=3,
                             dtype=np.float64, submap=subnpix, local=localsm)
        counter._objects.append(zmap)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - distobjects initialized in {:.3f} s'
                  ''.format(stop-start), flush=args.flush)
        start = stop

        invnpp_group = None
        hits_group = None
        zmap_group = None
        if comm.comm_group.size < comm.comm_world.size:
            invnpp_group = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=6,
                                         dtype=np.float64, submap=subnpix,
                                         local=localsm)
            counter._objects.append(invnpp_group)
            invnpp_group.data.fill(0.0)

            hits_group = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=1,
                                       dtype=np.int64, submap=subnpix,
                                       local=localsm)
            counter._objects.append(hits_group)
            hits_group.data.fill(0)

            zmap_group = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=3,
                                       dtype=np.float64, submap=subnpix,
                                       local=localsm)
            counter._objects.append(zmap_group)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - group distobjects initialized in {:.3f} s'
                      ''.format(stop-start), flush=args.flush)
            start = stop

        # compute the hits and covariance once, since the pointing and noise
        # weights are fixed.

        build_invnpp = tm.OpAccumDiag(
            detweights=detweights, invnpp=invnpp, hits=hits,
            flag_name=flag_name, common_flag_name=common_flag_name,
            common_flag_mask=args.common_flag_mask)

        build_invnpp.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - distobjects accumulated in {:.3f} s'
                  ''.format(stop-start), flush=args.flush)
        start = stop

        invnpp.allreduce()
        if not args.skip_hits:
            hits.allreduce()

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - distobjects reduced in {:.3f} s'.format(stop-start),
                  flush=args.flush)
        start = stop

        if invnpp_group is not None:
            build_invnpp_group = tm.OpAccumDiag(
                detweights=detweights, invnpp=invnpp_group, hits=hits_group,
                flag_name=flag_name, common_flag_name=common_flag_name,
                common_flag_mask=args.common_flag_mask)

            build_invnpp_group.exec(data)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - group distobjects accumulated in {:.3f} s'
                      ''.format(stop-start), flush=args.flush)
            start = stop

            invnpp_group.allreduce()
            if not args.skip_hits:
                hits_group.allreduce()

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - group distobjects reduced in {:.3f} s'
                      ''.format(stop-start), flush=args.flush)
            start = stop

        if not args.skip_hits:
            fn = '{}/hits.fits'.format(args.outdir)
            if args.zip:
                fn += '.gz'
            hits.write_healpix_fits(fn)
            comm.comm_world.barrier()
            stop = MPI.Wtime()
            if comm.comm_world.rank == 0:
                print(' - Writing hit map to {} took {:.3f} s'
                      ''.format(fn, stop-start), flush=args.flush)
            start = stop
        counter._objects.remove(hits)
        del hits

        if hits_group is not None:
            if not args.skip_hits:
                fn = '{}/hits_group_{:04}.fits'.format(args.outdir, comm.group)
                if args.zip:
                    fn += '.gz'
                hits_group.write_healpix_fits(fn)
                comm.comm_group.barrier()
                stop = MPI.Wtime()
                if comm.comm_group.rank == 0:
                    print(' - Writing group hit map to {} took {:.3f} s'
                          ''.format(fn, stop-start), flush=args.flush)
                start = stop
            counter._objects.remove(hits_group)
            del hits_group

        if not args.skip_hits:
            fn = '{}/invnpp.fits'.format(args.outdir)
            if args.zip:
                fn += '.gz'
            invnpp.write_healpix_fits(fn)
            comm.comm_world.barrier()
            stop = MPI.Wtime()
            if comm.comm_world.rank == 0:
                print(' - Writing N_pp^-1 to {} took {:.3f} s'
                      ''.format(fn, stop-start), flush=args.flush)
            start = stop

        if not args.skip_hits:
            if invnpp_group is not None:
                fn = '{}/invnpp_group_{:04}.fits'.format(args.outdir,
                                                         comm.group)
                if args.zip:
                    fn += '.gz'
                invnpp_group.write_healpix_fits(fn)
                comm.comm_group.barrier()
                stop = MPI.Wtime()
                if comm.comm_group.rank == 0:
                    print(' - Writing group N_pp^-1 to {} took {:.3f} s'
                          ''.format(fn, stop-start), flush=args.flush)
                start = stop

        # invert it
        tm.covariance_invert(invnpp, 1.0e-3)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - Inverting N_pp^-1 took {:.3f} s'.format(stop-start),
                  flush=args.flush)
        start = stop

        if not args.skip_hits:
            fn = '{}/npp.fits'.format(args.outdir)
            if args.zip:
                fn += '.gz'
            invnpp.write_healpix_fits(fn)
            comm.comm_world.barrier()
            stop = MPI.Wtime()
            if comm.comm_world.rank == 0:
                print(' - Writing N_pp to {} took {:.3f} s'
                      ''.format(fn, stop-start), flush=args.flush)
            start = stop

        if invnpp_group is not None:
            tm.covariance_invert(invnpp_group, 1.0e-3)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - Inverting group N_pp^-1 took {:.3f} s'
                ''.format(stop-start), flush=args.flush)
            start = stop

            if not args.skip_hits:
                fn = '{}/npp_group_{:04}.fits'.format(args.outdir, comm.group)
                if args.zip:
                    fn += '.gz'
                invnpp_group.write_healpix_fits(fn)
                comm.comm_group.barrier()
                stop = MPI.Wtime()
                if comm.comm_group.rank == 0:
                    print(' - Writing group N_pp to {} took {:.3f} s'.format(
                        fn, stop-start), flush=args.flush)
                start = stop

        stop = MPI.Wtime()
        if comm.comm_group.rank == 0:
            print('Building Npp took {:.3f} s'.format(
                stop-start0), flush=args.flush)

        counter.exec(data)

    return invnpp, zmap, invnpp_group, zmap_group, flag_name, common_flag_name


def setup_madam(args, comm):

    pars = None

    if args.madam:

        # Set up MADAM map making.

        pars = {}

        cross = args.nside // 2
        submap = 16
        if submap > args.nside:
            submap = args.nside

        pars['temperature_only'] = False
        pars['force_pol'] = True
        pars['kfirst'] = True
        pars['write_map'] = True
        pars['write_binmap'] = True
        pars['write_matrix'] = True
        pars['write_wcov'] = True
        pars['write_hits'] = True
        pars['nside_cross'] = cross
        pars['nside_submap'] = submap
        pars['allreduce'] = args.madam_allreduce

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
            pars['kfilter'] = True
        else:
            pars['kfilter'] = False
        pars['precond_width'] = 1
        pars['fsample'] = args.samplerate
        pars['iter_max'] = args.madam_iter_max

    return pars


def copy_signal(args, comm, data, sigcopy, counter):
    if sigcopy is not None:
        if comm.comm_world.rank == 0:
            print('Making a copy of the signal TOD', flush=args.flush)
        sigcopy.exec(data)
        counter.exec(data)
    return


def scale_atmosphere_by_frequency(args, comm, data, freq, totalname_freq, mc):
    """ Scale atmospheric fluctuations by frequency.

    Assume that cached signal under totalname_freq is pure atmosphere
    and scale the absorption coefficient according to the frequency.

    """
    if not args.skip_atmosphere:
        for obs in data.obs:
            tod = obs['tod']
            site = obs['site']
            weather = obs['weather']
            start_time = obs['start_time']
            weather.set(site, mc, start_time)
            altitude = obs['altitude']
            absorption = toast.ctoast.atm_get_absorption_coefficient(
                altitude, weather.air_temperature, weather.surface_pressure,
                weather.pwv, freq)
            #loading = toast.ctoast.atm_get_atmospheric_loading(
            #    altitude, pwv, freq)
            for det in tod.local_dets:
                cachename = '{}_{}'.format(totalname_freq, det)
                ref = tod.cache.reference(cachename)
                ref *= absorption
                del ref

    return


def simulate_atmosphere(args, comm, data, mc, counter,
                        flag_name, common_flag_name, totalname):
    if not args.skip_atmosphere:
        if comm.comm_world.rank == 0:
            print('Simulating atmosphere', flush=args.flush)
            if args.atm_cache and not os.path.isdir(args.atm_cache):
                try:
                    os.makedirs(args.atm_cache)
                except FileExistsError:
                    pass
        start = MPI.Wtime()

        if common_flag_name is None:
            common_flag_name = 'common_flags'
        if flag_name is None:
            flag_name = 'flags'

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
            fp_radius=args.fp_radius, apply_flags=True,
            common_flag_name=common_flag_name,
            common_flag_mask=args.common_flag_mask, flag_name=flag_name,
            cachedir=args.atm_cache, flush=args.flush)

        atm.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Atmosphere simulation took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)

    return flag_name, common_flag_name


def copy_signal_freq(args, comm, data, sigcopy_freq, counter):
    if sigcopy_freq is not None:
        # Make a copy of the atmosphere so we can scramble the gains
        # repeatedly
        if comm.comm_world.rank == 0:
            print('Making a copy of the TOD for multifrequency',
                  flush=args.flush)
        sigcopy_freq.exec(data)
        counter.exec(data)
    return


def simulate_noise(args, comm, data, mc, counter, totalname_freq):
    if not args.skip_noise:
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


def setup_output(args, comm, mc):
    outpath = '{}/{:08d}'.format(args.outdir, mc)
    if comm.comm_world.rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


def copy_signal_madam(args, comm, data, sigcopy_madam, counter):
    if sigcopy_madam is not None:
        # Make a copy of the timeline for Madam
        if comm.comm_world.rank == 0:
            print('Making a copy of the TOD for Madam', flush=args.flush)
        sigcopy_madam.exec(data)

        counter.exec(data)
    return


def bin_maps(args, comm, data, rootname, counter,
             zmap, invnpp, zmap_group, invnpp_group, detweights, totalname_freq,
             flag_name, common_flag_name, mc, outpath):
    if not args.skip_bin:
        if comm.comm_world.rank == 0:
            print('Binning unfiltered maps', flush=args.flush)
        start0 = MPI.Wtime()
        start = start0

        # Bin a map using the toast facilities

        zmap.data.fill(0.0)
        build_zmap = tm.OpAccumDiag(
            detweights=detweights, zmap=zmap, name=totalname_freq,
            flag_name=flag_name, common_flag_name=common_flag_name,
            common_flag_mask=args.common_flag_mask)
        build_zmap.exec(data)
        zmap.allreduce()

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - Building noise weighted map {:04d} took {:.3f} s'
                  ''.format(mc, stop-start), flush=args.flush)
        start = stop

        tm.covariance_apply(invnpp, zmap)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - Computing {} map {:04d} took {:.3f} s'
                  ''.format(rootname, mc, stop-start), flush=args.flush)
        start = stop

        fn = os.path.join(outpath, rootname+'.fits')
        if args.zip:
            fn += '.gz'
        zmap.write_healpix_fits(fn)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - Writing {} map {:04d} to {} took {:.3f} s'
                  ''.format(rootname, mc, fn, stop-start), flush=args.flush)

        if zmap_group is not None:

            zmap_group.data.fill(0.0)
            build_zmap_group = tm.OpAccumDiag(
                detweights=detweights, zmap=zmap_group,
                name=totalname_freq,
                flag_name=flag_name, common_flag_name=common_flag_name,
                common_flag_mask=args.common_flag_mask)
            build_zmap_group.exec(data)
            zmap_group.allreduce()

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - Building group noise weighted map {:04d} took '
                      '{:.3f} s'.format(mc, stop-start), flush=args.flush)
            start = stop

            tm.covariance_apply(invnpp_group, zmap_group)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_group.rank == 0:
                print(' - Computing {} map {:04d} took {:.3f} s'
                      ''.format(rootname, mc, stop-start), flush=args.flush)
            start = stop

            fn = os.path.join(outpath, '{}_group_{:04}.fits'
                              ''.format(rootname, comm.group))
            if args.zip:
                fn += '.gz'
            zmap_group.write_healpix_fits(fn)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - Writing group {} map {:04d} to {} took '
                      '{:.3f} s'.format(rootname, mc, fn, stop-start),
                      flush=args.flush)

        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Mapmaking {:04d} took {:.3f} s'
                  ''.format(mc, stop-start0), flush=args.flush)

        counter.exec(data)

    return


def apply_polyfilter(args, comm, data, counter, totalname_freq):
    if args.polyorder:
        if comm.comm_world.rank == 0:
            print('Polyfiltering signal', flush=args.flush)
        start = MPI.Wtime()
        common_flag_name = 'common_flags'
        flag_name = 'flags'
        polyfilter = tt.OpPolyFilter(
            order=args.polyorder, name=totalname_freq,
            common_flag_name=common_flag_name,
            common_flag_mask=args.common_flag_mask,
            flag_name=flag_name)
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
        if comm.comm_world.rank == 0:
            print('Ground filtering signal', flush=args.flush)
        start = MPI.Wtime()
        common_flag_name = 'common_flags'
        flag_name = 'flags'
        groundfilter = tt.OpGroundFilter(
            wbin=args.wbin_ground, name=totalname_freq,
            common_flag_name=common_flag_name,
            common_flag_mask=args.common_flag_mask,
            flag_name=flag_name)
        groundfilter.exec(data)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Ground filtering took {:.3f} s'.format(stop-start),
                  flush=args.flush)

        counter.exec(data)
    return


def clear_signal(args, comm, data, sigclear, counter):
    if sigclear is not None:
        if comm.comm_world.rank == 0:
            print('Clearing filtered signal')
        sigclear.exec(data)
        counter.exec(data)
    return


def output_tidas(args, comm, data, totalname, common_flag_name, flag_name):
    if args.tidas is None:
        return
    from toast.tod.tidas import OpTidasExport
    tidas_path = os.path.abspath(args.tidas)
    comm.comm_world.Barrier()
    if comm.comm_world.rank == 0:
        print('Exporting TOD to a TIDAS volume at {}'.format(tidas_path),
              flush=args.flush)
    start = MPI.Wtime()

    export = OpTidasExport(tidas_path, name=totalname,
        common_flag_name=common_flag_name,
        flag_name=flag_name, usedist=True)
    export.exec(data)

    comm.comm_world.Barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Wrote simulated TOD to {}:{} in {:.2f} s'
              ''.format(tidas_path, totalname,
                        stop-start), flush=args.flush)
    return


def apply_madam(args, comm, data, madampars, counter, mc, firstmc, outpath,
                detweights, totalname_madam, flag_name, common_flag_name):
    if args.madam:
        if comm.comm_world.rank == 0:
            print('Destriping signal', flush=args.flush)
        start = MPI.Wtime()

        # create output directory for this realization
        madampars['path_output'] = outpath
        if mc != firstmc:
            madampars['write_matrix'] = False
            madampars['write_wcov'] = False
            madampars['write_hits'] = False

        madam = tm.OpMadam(
            params=madampars, detweights=detweights,
            name=totalname_madam,
            common_flag_name=common_flag_name, flag_name=flag_name,
            common_flag_mask=args.common_flag_mask,
            purge_tod=True)

        madam.exec(data)

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

    global_start = MPI.Wtime()

    args, comm = parse_arguments(comm)

    # Load and broadcast the schedule file

    schedules = load_schedule(args, comm)

    # Load the weather and append to schedules

    load_weather(args, comm, schedules)

    # load or simulate the focalplane

    detweights = load_fp(args, comm, schedules)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    counter = tt.OpMemoryCounter()

    data = create_observations(args, comm, schedules, counter)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data, counter)

    # Prepare auxiliary information for distributed map objects

    localpix, localsm, subnpix = get_submaps(args, comm, data)

    # Scan input map

    signalname = scan_signal(args, comm, data, counter, localsm, subnpix)

    # Set up objects to take copies of the TOD at appropriate times

    sigcopy, sigcopy_freq, sigcopy_madam, sigclear, \
        totalname, totalname_freq, totalname_madam \
        = setup_sigcopy(args, comm, signalname)

    common_flag_name = None
    flag_name = None

    invnpp, zmap, invnpp_group, zmap_group, flag_name, common_flag_name \
        = build_npp(args, comm, data, counter, localsm, subnpix, detweights,
                    flag_name, common_flag_name)

    madampars = setup_madam(args, comm)

    # Loop over Monte Carlos

    firstmc = int(args.MC_start)
    nmc = int(args.MC_count)

    freqs = [float(freq) for freq in args.freq.split(',')]
    nfreq = len(freqs)

    for mc in range(firstmc, firstmc+nmc):

        flag_name, common_flag_name = simulate_atmosphere(
            args, comm, data, mc, counter,
            flag_name, common_flag_name, totalname)

        # Loop over frequencies with identical focal planes and identical
        # atmospheric noise.

        for ifreq, freq in enumerate(freqs):

            if comm.comm_world.rank == 0:
                print('Processing frequency {}GHz {} / {}, MC = {}'
                      ''.format(freq, ifreq+1, nfreq, mc), flush=args.flush)

            copy_signal_freq(args, comm, data, sigcopy_freq, counter)

            scale_atmosphere_by_frequency(args, comm, data, freq,
                                          totalname_freq, mc)

            add_sky_signal(args, comm, data, totalname_freq, signalname)

            mcoffset = ifreq * 1000000

            simulate_noise(args, comm, data, mc+mcoffset, counter,
                           totalname_freq)

            scramble_gains(args, comm, data, mc+mcoffset, counter,
                           totalname_freq)

            if (mc == firstmc) and (ifreq == 0):
                # For the first realization and frequency, optionally
                # export the timestream data to a TIDAS volume.
                output_tidas(args, comm, data, totalname, common_flag_name,
                             flag_name)

            outpath = setup_output(args, comm, mc+mcoffset)

            copy_signal_madam(args, comm, data, sigcopy_madam, counter)

            bin_maps(args, comm, data, 'binned', counter,
                     zmap, invnpp, zmap_group, invnpp_group, detweights,
                     totalname_freq, flag_name, common_flag_name,
                     mc+mcoffset, outpath)

            apply_polyfilter(args, comm, data, counter, totalname_freq)

            apply_groundfilter(args, comm, data, counter, totalname_freq)

            if args.polyorder or args.wbin_ground:
                bin_maps(args, comm, data, 'filtered', counter,
                         zmap, invnpp, zmap_group, invnpp_group, detweights,
                         totalname_freq, flag_name, common_flag_name,
                         mc+mcoffset, outpath)

            clear_signal(args, comm, data, sigclear, counter)

            apply_madam(args, comm, data, madampars, counter,
                        mc+mcoffset, firstmc, outpath, detweights,
                        totalname_madam, flag_name, common_flag_name)

    counter.exec(data)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - global_start
    if comm.comm_world.rank == 0:
        print('Total Time:  {:.2f} seconds'.format(elapsed), flush=True)

    #raise Exception('Done!')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Exception occurred: "{}"'.format(e), flush=True)
        if MPI.COMM_WORLD.size == 1:
            raise
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('*** print_tb:')
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print('*** print_exception:')
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
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
        MPI.COMM_WORLD.Abort()
