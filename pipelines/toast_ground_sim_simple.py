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
import toast.timing as timing

XAXIS, YAXIS, ZAXIS = np.eye(3)

def parse_arguments(comm):

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "and map astrophysical signal.", fromfile_prefix_chars='@')
    parser.add_argument('--groupsize',
                        required=False, type=np.int,
                        help='Size of a process group assigned to a CES')

    parser.add_argument('--timezone', required=False, type=np.int, default=0,
                        help='Offset to apply to MJD to separate days [hours]')
    parser.add_argument('--coord', required=False, default='C',
                        help='Sky coordinate system [C,E,G]')
    parser.add_argument('--schedule', required=True,
                        help='CES schedule file from toast_ground_schedule.py')
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
    parser.add_argument('--fp',
                        required=False, default=None,
                        help='Pickle file containing a dictionary of detector '
                        'properties.  The keys of this dict are the detector '
                        'names, and each value is also a dictionary with keys '
                        '"quat" (4 element ndarray), "fwhm" (float, arcmin), '
                        '"fknee" (float, Hz), "alpha" (float), and '
                        '"NET" (float).  For optional plotting, the key "color"'
                        ' can specify a valid matplotlib color string.')
    parser.add_argument('--tidas',
                        required=False, default=None,
                        help='Output TIDAS export path')

    args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))

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


def load_schedule(args, comm):
    start = MPI.Wtime()
    autotimer = timing.auto_timer()
    if comm.comm_world.rank == 0:
        fn = args.schedule
        if not os.path.isfile(fn):
            raise RuntimeError('No such schedule file: {}'.format(fn))
        start = MPI.Wtime()
        f = open(fn, 'r')
        while True:
            line = f.readline()
            if line.startswith('#'):
                continue
            site_name, site_lat, site_lon, site_alt = line.split()
            site_alt = float(site_alt)
            site = [site_name, site_lat, site_lon, site_alt]
            break
        all_ces = []
        for line in f:
            if line.startswith('#'):
                continue
            start_date, start_time, stop_date, stop_time, mjdstart, mjdstop, \
                name, azmin, azmax, el, rs, \
                sun_el1, sun_az1, sun_el2, sun_az2, \
                moon_el1, moon_az1, moon_el2, moon_az2, moon_phase, \
                scan, subscan = line.split()
            start_time = start_date + ' ' + start_time
            stop_time = stop_date + ' ' + stop_time
            try:
                start_time = dateutil.parser.parse(start_time + ' +0000')
                stop_time = dateutil.parser.parse(stop_time + ' +0000')
            except:
                start_time = dateutil.parser.parse(start_time)
                stop_time = dateutil.parser.parse(stop_time)

            start_timestamp = start_time.timestamp()
            stop_timestamp = stop_time.timestamp()

            all_ces.append([
                start_timestamp, stop_timestamp, name, float(mjdstart),
                int(scan), int(subscan), float(azmin), float(azmax), float(el)])
        f.close()
        stop = MPI.Wtime()
        elapsed = stop - start
        print('Load schedule:  {:.2f} seconds'.format(stop-start),
              flush=args.flush)
    else:
        site = None
        all_ces = None

    site = comm.comm_world.bcast(site)
    all_ces = comm.comm_world.bcast(all_ces)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Loading schedule {:.3f} s'.format(stop-start), flush=args.flush)

    return site, all_ces


def load_fp(args, comm):
    start = MPI.Wtime()
    autotimer = timing.auto_timer()

    fp = None

    # Load focalplane information

    nullquat = np.array([0,0,0,1], dtype=np.float64)

    if comm.comm_world.rank == 0:
        if args.fp is None:
            # in this case, create a fake detector at the boresight
            # with a pure white noise spectrum.
            fake = {}
            fake['quat'] = nullquat
            fake['fwhm'] = 30.0
            fake['fknee'] = 0.0
            fake['fmin'] = 1e-9
            fake['alpha'] = 1.0
            fake['NET'] = 1.0
            fake['color'] = 'r'
            fp = {}
            # Second detector at 22.5 degree polarization angle
            fp['bore1'] = fake
            fake2 = {}
            zrot = qa.rotation(ZAXIS, 22.5*degree)
            fake2['quat'] = qa.mult(fake['quat'], zrot)
            fake2['fwhm'] = 30.0
            fake2['fknee'] = 0.0
            fake2['fmin'] = 1e-9
            fake2['alpha'] = 1.0
            fake2['NET'] = 1.0
            fake2['color'] = 'r'
            fp['bore2'] = fake2
            # Third detector at 45 degree polarization angle
            fake3 = {}
            zrot = qa.rotation(ZAXIS, 45*degree)
            fake3['quat'] = qa.mult(fake['quat'], zrot)
            fake3['fwhm'] = 30.0
            fake3['fknee'] = 0.0
            fake3['fmin'] = 1e-9
            fake3['alpha'] = 1.0
            fake3['NET'] = 1.0
            fake3['color'] = 'r'
            fp['bore3'] = fake3
            # Fourth detector at 67.5 degree polarization angle
            fake4 = {}
            zrot = qa.rotation(ZAXIS, 67.5*degree)
            fake4['quat'] = qa.mult(fake['quat'], zrot)
            fake4['fwhm'] = 30.0
            fake4['fknee'] = 0.0
            fake4['fmin'] = 1e-9
            fake4['alpha'] = 1.0
            fake4['NET'] = 1.0
            fake4['color'] = 'r'
            fp['bore4'] = fake4
        else:
            with open(args.fp, 'rb') as p:
                fp = pickle.load(p)
    fp = comm.comm_world.bcast(fp, root=0)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print('Create focalplane:  {:.2f} seconds'.format(stop-start),
              flush=args.flush)
    start = stop

    if args.debug:
        if comm.comm_world.rank == 0:
            outfile = '{}/focalplane.png'.format(args.outdir)
            tt.plot_focalplane(fp, 6, 6, outfile)

    detectors = sorted(fp.keys())
    detweights = {}
    for d in detectors:
        net = fp[d]['NET']
        detweights[d] = 1.0 / (args.samplerate * net * net)

    return fp, detweights


def create_observations(args, comm, fp, all_ces, site):
    """ Simulate constant elevation scans.

    Simulate constant elevation scans at "site" matching entries in
    "all_ces".  Each operational day is assigned to a different
    process group to allow making day maps.

    """
    start = MPI.Wtime()
    autotimer = timing.auto_timer()

    data = toast.Data(comm)

    site_name, site_lat, site_lon, site_alt = site

    detectors = sorted(fp.keys())
    detquats = {}
    for d in detectors:
        detquats[d] = fp[d]['quat']

    nces = len(all_ces)

    breaks = []
    do_break = False
    for i in range(nces-1):
        # If current and next CES are on different days, insert a break
        tz = args.timezone / 24.
        start1 = all_ces[i][3] # MJD start
        start2 = all_ces[i+1][3] # MJD start
        scan1 = all_ces[i][4]
        scan2 = all_ces[i+1][4]
        if scan1 != scan2 and do_break:
            breaks.append(i + 1)
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
                breaks.append(i + 1)

    nbreak = len(breaks)
    if nbreak != comm.ngroups-1:
        raise RuntimeError(
            'Number of observing days ({}) does not match number of process '
            'groups ({}).'.format(nbreak+1, comm.ngroups))

    groupdist = toast.distribute_uniform(nces, comm.ngroups, breaks=breaks)
    group_firstobs = groupdist[comm.group][0]
    group_numobs = groupdist[comm.group][1]

    # Create the noise model used by all observations

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

    for ices in range(group_firstobs, group_firstobs + group_numobs):
        ces = all_ces[ices]

        CES_start, CES_stop, name, mjdstart, scan, subscan, azmin, azmax, \
            el = ces

        totsamples = int((CES_stop - CES_start) * args.samplerate)

        # create the single TOD for this observation

        try:
            tod = tt.TODGround(
                comm.comm_group,
                detquats,
                totsamples,
                detranks=comm.comm_group.size,
                firsttime=CES_start,
                rate=args.samplerate,
                site_lon=site_lon,
                site_lat=site_lat,
                site_alt=site_alt,
                azmin=azmin,
                azmax=azmax,
                el=el,
                scanrate=args.scanrate,
                scan_accel=args.scan_accel,
                CES_start=None,
                CES_stop=None,
                sun_angle_min=args.sun_angle_min,
                coord=args.coord,
                sampsizes=None)
        except RuntimeError as e:
            print('Failed to create the CES scan: {}'.format(e),
                  flush=args.flush)
            return

        # Create the (single) observation

        ob = {}
        ob['name'] = 'CES-{}-{}-{}'.format(name, scan, subscan)
        ob['tod'] = tod
        if len(tod.subscans) > 0:
            ob['intervals'] = tod.subscans
        else:
            raise RuntimeError('{} has no valid intervals'.format(ob['name']))
        ob['baselines'] = None
        ob['noise'] = noise
        ob['id'] = int(mjdstart * 10000)

        data.obs.append(ob)

    for ob in data.obs:
        tod = ob['tod']
        tod.free_azel_quats()

    if comm.comm_group.rank == 0:
        print('Group # {:4} has {} observations.'.format(
            comm.group, len(data.obs)), flush=args.flush)

    if len(data.obs) == 0:
        raise RuntimeError('Too many tasks. Every MPI task must '
                           'be assigned to at least one observation.')

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    if comm.comm_world.rank == 0:
        print('Simulated scans in {:.2f} seconds'
              ''.format(stop-start), flush=args.flush)

    return data


def expand_pointing(args, comm, data):
    """ Expand the bore sight pointing to every detector.

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

    return


def get_submaps(args, comm, data):
    """ Determine submap distribution

    """
    if not args.skip_bin or args.input_map:
        if comm.comm_world.rank == 0:
            print('Scanning local pixels', flush=args.flush)
        start = MPI.Wtime()
        autotimer = timing.auto_timer()

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


def scan_signal(args, comm, data, localsm, subnpix):
    """ Scan time-ordered signal from a map.

    """
    signalname = None

    if args.input_map:
        if comm.comm_world.rank == 0:
            print('Scanning input map', flush=args.flush)
        start = MPI.Wtime()
        autotimer = timing.auto_timer()

        npix = 12*args.nside**2

        # Scan the sky signal
        if  comm.comm_world.rank == 0 and not os.path.isfile(args.input_map):
            raise RuntimeError(
                'Input map does not exist: {}'.format(args.input_map))
        distmap = tm.DistPixels(
            comm=comm.comm_world, size=npix, nnz=3,
            dtype=np.float32, submap=subnpix, local=localsm)
        distmap.read_healpix_fits(args.input_map)
        scansim = tt.OpSimScan(distmap=distmap, out='signal')
        scansim.exec(data)

        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Read and sampled input map:  {:.2f} seconds'
                  ''.format(stop-start), flush=args.flush)
        signalname = 'signal'

    return signalname


def setup_sigcopy(args, comm, signalname):
    """ Setup for copying the signal so we can run filter+bin and Madam.

    """
    if args.skip_bin:
        signalname_madam = signalname
    else:
        signalname_madam = 'signal_madam'

    if args.madam:
        sigcopy_madam = tt.OpCacheCopy(signalname, signalname_madam)
        sigclear = tt.OpCacheClear(signalname)
    else:
        sigcopy_madam = None
        sigclear = None

    return signalname_madam, sigcopy_madam, sigclear


def build_npp(args, comm, data, localsm, subnpix, detweights,
              flag_name, common_flag_name):
    """ Build pixel-pixel noise covariance matrices.

    """
    if not args.skip_bin:

        if comm.comm_world.rank == 0:
            print('Preparing distributed map', flush=args.flush)
        start0 = MPI.Wtime()
        start = start0
        autotimer = timing.auto_timer()

        npix = 12*args.nside**2

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        invnpp = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=6,
                               dtype=np.float64, submap=subnpix, local=localsm)
        invnpp.data.fill(0.0)

        hits = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=1,
                             dtype=np.int64, submap=subnpix, local=localsm)
        hits.data.fill(0)

        zmap = tm.DistPixels(comm=comm.comm_world, size=npix, nnz=3,
                             dtype=np.float64, submap=subnpix, local=localsm)

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
            invnpp_group.data.fill(0.0)

            hits_group = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=1,
                                       dtype=np.int64, submap=subnpix,
                                       local=localsm)
            hits_group.data.fill(0)

            zmap_group = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=3,
                                       dtype=np.float64, submap=subnpix,
                                       local=localsm)

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

    return invnpp, zmap, invnpp_group, zmap_group, flag_name, common_flag_name


def setup_madam(args, comm):
    """ Prepare to run Madam on the stored TOD.

    """
    pars = None

    if args.madam:
        autotimer = timing.auto_timer()

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


def setup_output(args, comm):
    outpath = '{}'.format(args.outdir)
    if comm.comm_world.rank == 0:
        if not os.path.isdir(outpath):
            try:
                os.makedirs(outpath)
            except FileExistsError:
                pass
    return outpath


def copy_signal_madam(args, comm, data, sigcopy_madam):
    """ Make a copy of the TOD for Madam.

    """
    if sigcopy_madam is not None:
        if comm.comm_world.rank == 0:
            print('Making a copy of the TOD for Madam', flush=args.flush)
        sigcopy_madam.exec(data)

    return


def bin_maps(args, comm, data, rootname,
             zmap, invnpp, zmap_group, invnpp_group, detweights, totalname_freq,
             flag_name, common_flag_name, outpath):
    """ Use TOAST facilities to bin stored signal.

    """
    if not args.skip_bin:
        if comm.comm_world.rank == 0:
            print('Binning unfiltered maps', flush=args.flush)
        start0 = MPI.Wtime()
        start = start0
        autotimer = timing.auto_timer()

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
            print(' - Building noise weighted map took {:.3f} s'
                  ''.format(stop-start), flush=args.flush)
        start = stop

        tm.covariance_apply(invnpp, zmap)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - Computing {} map took {:.3f} s'
                  ''.format(rootname, stop-start), flush=args.flush)
        start = stop

        fn = os.path.join(outpath, rootname+'.fits')
        if args.zip:
            fn += '.gz'
        zmap.write_healpix_fits(fn)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print(' - Writing {} map to {} took {:.3f} s'
                  ''.format(rootname, fn, stop-start), flush=args.flush)

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
                print(' - Building group noise weighted map took '
                      '{:.3f} s'.format(stop-start), flush=args.flush)
            start = stop

            tm.covariance_apply(invnpp_group, zmap_group)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_group.rank == 0:
                print(' - Computing {} map took {:.3f} s'
                      ''.format(rootname, stop-start), flush=args.flush)
            start = stop

            fn = os.path.join(outpath, '{}_group_{:04}.fits'
                              ''.format(rootname, comm.group))
            if args.zip:
                fn += '.gz'
            zmap_group.write_healpix_fits(fn)

            comm.comm_group.barrier()
            stop = MPI.Wtime()
            if comm.comm_group.rank == 0:
                print(' - Writing group {} map to {} took '
                      '{:.3f} s'.format(rootname, fn, stop-start),
                      flush=args.flush)

        stop = MPI.Wtime()
        if comm.comm_world.rank == 0:
            print('Mapmaking took {:.3f} s'
                  ''.format(stop-start0), flush=args.flush)

    return


def apply_polyfilter(args, comm, data, totalname_freq):
    if args.polyorder:
        if comm.comm_world.rank == 0:
            print('Polyfiltering signal', flush=args.flush)
        start = MPI.Wtime()
        autotimer = timing.auto_timer()
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

    return


def apply_groundfilter(args, comm, data, totalname_freq):
    if args.wbin_ground:
        if comm.comm_world.rank == 0:
            print('Ground filtering signal', flush=args.flush)
        start = MPI.Wtime()
        autotimer = timing.auto_timer()
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

    return


def clear_signal(args, comm, data, sigclear):
    if sigclear is not None:
        if comm.comm_world.rank == 0:
            print('Clearing filtered signal')
        sigclear.exec(data)
    return


def output_tidas(args, comm, data, totalname, common_flag_name, flag_name):
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


def apply_madam(args, comm, data, madampars, outpath,
                detweights, totalname_madam, flag_name, common_flag_name):
    if args.madam:
        if comm.comm_world.rank == 0:
            print('Destriping signal', flush=args.flush)
        start = MPI.Wtime()
        autotimer = timing.auto_timer()

        # create output directory for this realization
        madampars['path_output'] = outpath

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

    return


def main():

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print('Running with {} processes at {}'.format(
            comm.comm_world.size, str(datetime.now())), flush=True)

    global_start = MPI.Wtime()
    autotimer = timing.auto_timer("@{}".format(timing.FILE()))

    args, comm = parse_arguments(comm)

    # Load and broadcast the schedule file

    site, all_ces = load_schedule(args, comm)

    # load or simulate the focalplane

    fp, detweights = load_fp(args, comm)

    # Create the TOAST data object to match the schedule.  This will
    # include simulating the boresight pointing.

    data = create_observations(args, comm, fp, all_ces, site)

    # Expand boresight quaternions into detector pointing weights and
    # pixel numbers

    expand_pointing(args, comm, data)

    # Prepare auxiliary information for distributed map objects

    localpix, localsm, subnpix = get_submaps(args, comm, data)

    # Scan input map

    signalname = scan_signal(args, comm, data, localsm, subnpix)

    # Set up objects to take copies of the TOD at appropriate times

    signalname_madam, sigcopy_madam, sigclear \
        = setup_sigcopy(args, comm, signalname)

    common_flag_name = None
    flag_name = None

    invnpp, zmap, invnpp_group, zmap_group, flag_name, common_flag_name \
        = build_npp(args, comm, data, localsm, subnpix, detweights,
                    flag_name, common_flag_name)

    madampars = setup_madam(args, comm)

    output_tidas(args, comm, data, signalname, common_flag_name, flag_name)

    outpath = setup_output(args, comm)

    # Make a copy of the signal for Madam

    copy_signal_madam(args, comm, data, sigcopy_madam)

    # Bin unprocessed signal for reference

    bin_maps(args, comm, data, 'binned', zmap, invnpp, zmap_group, invnpp_group,
             detweights, signalname, flag_name, common_flag_name, outpath)

    # Filter signal

    apply_polyfilter(args, comm, data, signalname)

    apply_groundfilter(args, comm, data, signalname)

    # Bin the filtered signal

    if args.polyorder or args.wbin_ground:
        bin_maps(args, comm, data, 'filtered', zmap, invnpp, zmap_group,
                 invnpp_group, detweights, signalname, flag_name,
                 common_flag_name, outpath)

    clear_signal(args, comm, data, sigclear)

    # Now run Madam on the unprocessed copy of the signal

    apply_madam(args, comm, data, madampars, outpath, detweights,
                signalname_madam, flag_name, common_flag_name)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - global_start
    if comm.comm_world.rank == 0:
        print('Total Time:  {:.2f} seconds'.format(elapsed), flush=True)

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
