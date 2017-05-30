#!/usr/bin/env python

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from toast.mpi import MPI

import os
import re
import argparse
import pickle
from datetime import datetime
import dateutil.parser

import numpy as np
from scipy.constants import degree

import toast
import toast.tod as tt
import toast.map as tm
import toast.qarray as qa


XAXIS, YAXIS, ZAXIS = np.eye(3)


def view_focalplane(fp, outfile):
    # To avoid python overhead in large MPI jobs, place the
    # matplotlib import inside this function, which is only called
    # when the --debug option is specified.
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # field of view, in degrees
    width = 10.0
    height = 10.0
            
    fig = plt.figure( figsize=(12,12), dpi=100 )
    ax = fig.add_subplot(1, 1, 1)

    half_width = 0.5 * width
    half_height = 0.5 * height
    ax.set_xlabel('Degrees', fontsize='large')
    ax.set_ylabel('Degrees', fontsize='large')
    ax.set_xlim([-half_width, half_width])
    ax.set_ylim([-half_height, half_height])

    for det in sorted(fp.keys()):

        # radius in degrees
        detradius = 0.5 * fp[det]['fwhm'] / 60.0

        # rotation from boresight
        dir = qa.rotate(fp[det]['quat'], ZAXIS).flatten()
        ang = np.arctan2(dir[1], dir[0])

        orient = qa.rotate(fp[det]['quat'], XAXIS).flatten()
        polang = np.arctan2(orient[1], orient[0])

        mag = np.arccos(dir[2]) * 180.0 / np.pi
        xpos = mag * np.cos(ang)
        ypos = mag * np.sin(ang)

        circ = plt.Circle((xpos, ypos), radius=detradius, fc='white', ec='k')
        ax.add_artist(circ)

        xtail = xpos - detradius * np.cos(polang)
        ytail = ypos - detradius * np.sin(polang)
        dx = 2.0 * detradius * np.cos(polang)
        dy = 2.0 * detradius * np.sin(polang)    

        detcolor = None
        if 'color' in fp[det].keys():
            detcolor = fp[det]['color']
        ax.arrow(xtail, ytail, dx, dy, width=0.1*detradius,
                 head_width=0.3*detradius, head_length=0.3*detradius,
                 fc=detcolor, ec=detcolor, length_includes_head=True)

    plt.savefig(outfile)
    return


def main():

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print("Running with {} processes".format(comm.comm_world.size),
              flush=True)

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser(
        description="Simulate ground-based boresight pointing.  Simulate "
        "atmosphere and make maps for some number of noise Monte Carlos.",
        fromfile_prefix_chars='@')
    parser.add_argument('--groupsize',
                        required=False, type=np.int,
                        help='Size of a process group assigned to a CES')

    parser.add_argument('--coord',
                        required=False, default='C',
                        help='Sky coordinate system [C,E,G]')
    parser.add_argument('--schedule',
                        required=True,
                        help='CES schedule file from toast_ground_schedule.py')
    parser.add_argument('--samplerate',
                        required=False, default=100.0, type=np.float,
                        help='Detector sample rate (Hz)')
    parser.add_argument('--scanrate',
                        required=False, default=1.0, type=np.float,
                        help='Scanning rate [deg / s]')
    parser.add_argument('--scan_accel',
                        required=False, default=0.5, type=np.float,
                        help='Scanning rate change [deg / s^2]')
    parser.add_argument('--sun_angle_min',
                        required=False, default=90.0, type=np.float,
                        help='Minimum azimuthal distance between the Sun and '
                        'the bore sight [deg]')

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
    parser.add_argument('--atm_fnear',
                        required=False, default=0.3, type=np.float,
                        help='multiplier for the near field simulation')
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

    parser.add_argument('--outdir',
                        required=False, default='out',
                        help='Output directory')
    parser.add_argument('--debug',
                        required=False, default=False, action='store_true',
                        help='Write diagnostics')
    parser.add_argument('--nside',
                        required=False, default=512, type=np.int,
                        help='Healpix NSIDE')
    parser.add_argument('--baseline',
                        required=False, default=60.0, type=np.float,
                        help='Destriping baseline length (seconds)')
    parser.add_argument('--noisefilter',
                        required=False, default=False, action='store_true',
                        help='Destripe with the noise filter enabled')
    parser.add_argument('--madam',
                        required=False, default=False, action='store_true',
                        help='If specified, use libmadam for map-making')
    parser.add_argument('--madampar',
                        required=False, default=None,
                        help='Madam parameter file')
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
                        '"NET" (float).  For optional plotting, the key "color"'
                        ' can specify a valid matplotlib color string.')

    args = parser.parse_args()

    if args.groupsize:
        comm = toast.Comm(groupsize=args.groupsize)

    # Load the schedule
    fn = args.schedule
    if not os.path.isfile(fn):
        raise RuntimeError('No such schedule file: {}'.format(fn))
    if comm.comm_world.rank == 0:
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
                scan = line.split()
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

            all_ces.append([start_timestamp, stop_timestamp, name, int(scan),
                            float(azmin), float(azmax), float(el)])
        f.close()
        stop = MPI.Wtime()
        elapsed = stop - start
        print('Load schedule:  {:.2f} seconds'.format(stop-start),
              flush=True)
    else:
        site = None
        all_ces = None

    site_name, site_lat, site_lon, site_alt = comm.comm_world.bcast(site)
    all_ces = comm.comm_world.bcast(all_ces)

    # get options

    hwprpm = args.hwprpm
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = args.hwpsteptime

    nside = args.nside
    npix = 12 * nside * nside

    start = MPI.Wtime()

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
        else:
            with open(args.fp, 'rb') as p:
                fp = pickle.load(p)
    fp = comm.comm_world.bcast(fp, root=0)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print('Create focalplane:  {:.2f} seconds'.format(stop-start),
              flush=True)
    start = stop

    if args.debug:
        if comm.comm_world.rank == 0:
            outfile = '{}_focalplane.png'.format(args.outdir)
            view_focalplane(fp, outfile)

    # Build observations out of the CES:es

    data = toast.Data(comm)

    detectors = sorted(fp.keys())
    detquats = {}
    for d in detectors:
        detquats[d] = fp[d]['quat']

    for ices, ces in enumerate(all_ces):

        # Assign the CES:es to process groups in a round-robin schedule
        if ices % comm.ngroups != comm.group:
            continue

        CES_start, CES_stop, name, scan, azmin, azmax, el = ces

        totsamples = int((CES_stop - CES_start) * args.samplerate)

        # create the single TOD for this observation

        try:
            tod = tt.TODGround(
                comm.comm_group,
                detquats,
                totsamples,
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
            print('Failed to create the CES scan: {}'.format(e), flush=True)
            return

        # Create the noise model for this observation

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

        # Create the (single) observation

        ob = {}
        ob['name'] = 'CES-{}-{}'.format(name, scan)
        ob['tod'] = tod
        ob['intervals'] = None
        ob['baselines'] = None
        ob['noise'] = noise
        ob['id'] = 0

        data.obs.append(ob)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print('Read parameters, compute data distribution:  {:.2f} seconds'
              ''.format(stop-start), flush=True)
    start = stop

    # Simulate the atmosphere signal

    """
    atm = tt.OpSimAtmosphere(
        out='signal', lmin_center=args.atm_lmin_center,
        lmin_sigma=args.atm_lmin_sigma, lmax_center=args.atm_lmax_center,
        lmax_sigma=args.atm_lmax_sigma, zatm=args.atm_zatm, zmax=args.atm_zmax,
        xstep=args.atm_xstep, ystep=args.atm_ystep, zstep=args.atm_zstep,
        nelem_sim_max=args.atm_nelem_sim_max, verbosity=int(args.debug),
        gangsize=args.atm_gangsize, fnear=args.atm_fnear,
        w_center=args.atm_w_center,
        w_sigma=args.atm_w_sigma, wdir_center=args.atm_wdir_center,
        wdir_sigma=args.atm_wdir_sigma, z0_center=args.atm_z0_center,
        z0_sigma=args.atm_z0_sigma, T0_center=args.atm_T0_center,
        T0_sigma=args.atm_T0_sigma)

    atm.exec(data)
    """

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print('Atmosphere simulation took {:.3f} s'.format(elapsed), flush=True)
    start = stop

    # We could also scan from a map and accumulate to 'signal' here...

    # make a Healpix pointing matrix.

    pointing = tt.OpPointingHpix(
        nside=nside, nest=True, mode='IQU', hwprpm=hwprpm, hwpstep=hwpstep,
        hwpsteptime=hwpsteptime)

    pointing.exec(data)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print('Pointing generation took {:.3f} s'.format(elapsed), flush=True)
    start = stop

    # Operator for signal copying, used in each MC iteration

    sigcopy = tt.OpCacheCopy("signal", "total")

    # Mapmaking.  For purposes of this simulation, we use detector noise
    # weights based on the NET (white noise level).  If the destriping
    # baseline is too long, this will not be the best choice.

    detweights = {}
    for d in detectors:
        net = fp[d]['NET']
        detweights[d] = 1.0 / (args.samplerate * net * net)

    if not args.madam:
        if comm.comm_world.rank == 0:
            print('Not using Madam, will only make a binned map!', flush=True)

        subnside = 16
        if subnside > nside:
            subnside = nside
        subnpix = 12 * subnside * subnside

        # get locally hit pixels
        lc = tm.OpLocalPixels()
        localpix = lc.exec(data)

        # find the locally hit submaps.
        allsm = np.floor_divide(localpix, subnpix)
        sm = set(allsm)
        localsm = np.array(sorted(sm), dtype=np.int64)

        # construct distributed maps to store the covariance,
        # noise weighted map, and hits

        invnpp = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=6,
                               dtype=np.float64, submap=subnpix, local=localsm)
        hits = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=1,
                             dtype=np.int64, submap=subnpix, local=localsm)
        zmap = tm.DistPixels(comm=comm.comm_group, size=npix, nnz=3,
                             dtype=np.float64, submap=subnpix, local=localsm)

        # compute the hits and covariance once, since the pointing and noise
        # weights are fixed.

        invnpp.data.fill(0.0)
        hits.data.fill(0)

        build_invnpp = tm.OpAccumDiag(detweights=detweights, invnpp=invnpp,
                                      hits=hits)
        build_invnpp.exec(data)

        invnpp.allreduce()
        hits.allreduce()

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Building hits and N_pp^-1 took {:.3f} s'.format(elapsed),
                  flush=True)
        start = stop

        hits.write_healpix_fits('{}_hits.fits'.format(args.outdir))
        invnpp.write_healpix_fits('{}_invnpp.fits'.format(args.outdir))

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Writing hits and N_pp^-1 took {:.3f} s'.format(elapsed),
                  flush=True)
        start = stop

        # invert it
        tm.covariance_invert(invnpp, 1.0e-3)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Inverting N_pp^-1 took {:.3f} s'.format(elapsed),
                  flush=True)
        start = stop

        invnpp.write_healpix_fits('{}_npp.fits'.format(args.outdir))

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Writing N_pp took {:.3f} s'.format(elapsed),
                  flush=True)
        start = stop

        """
        # in debug mode, print out data distribution information
        if args.debug:
            handle = None
            if comm.comm_world.rank == 0:
                handle = open('{}_distdata.txt'.format(args.outdir), 'w')
            data.info(handle)
            if comm.comm_world.rank == 0:
                handle.close()
            
            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('Dumping debug data distribution took {:.3f} s'
                      ''.format(elapsed))
            start = stop
        """

        mcstart = start

        # Loop over Monte Carlos

        firstmc = int(args.MC_start)
        nmc = int(args.MC_count)

        for mc in range(firstmc, firstmc+nmc):
            # create output directory for this realization
            outpath = '{}_{:03d}'.format(args.outdir, mc)
            if comm.comm_world.rank == 0:
                if not os.path.isdir(outpath):
                    os.makedirs(outpath)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('Creating output dir {:04d} took {:.3f} s'
                      ''.format(mc, elapsed), flush=True)
            start = stop

            # Copy the signal timestreams to the total ones before
            # accumulating the noise.

            sigcopy.exec(data)

            # simulate noise

            nse = tt.OpSimNoise(out='total', realization=mc)
            nse.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Noise simulation {:04d} took {:.3f} s'
                      ''.format(mc, elapsed), flush=True)
            start = stop

            zmap.data.fill(0.0)
            build_zmap = tm.OpAccumDiag(zmap=zmap, name='total')
            build_zmap.exec(data)
            zmap.allreduce()

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Building noise weighted map {:04d} took {:.3f} s'
                      ''.format(mc, elapsed), flush=True)
            start = stop

            tm.covariance_apply(invnpp, zmap)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Computing binned map {:04d} took {:.3f} s'
                      ''.format(mc, elapsed), flush=True)
            start = stop
        
            zmap.write_healpix_fits(os.path.join(outpath, 'binned.fits'))

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Writing binned map {:04d} took {:.3f} s'
                      ''.format(mc, elapsed), flush=True)
            elapsed = stop - mcstart
            if comm.comm_world.rank == 0:
                print('  Mapmaking {:04d} took {:.3f} s'.format(mc, elapsed),
                      flush=True)
            start = stop

    else:

        # Set up MADAM map making.

        pars = {}

        cross = int(nside / 2)
        submap = 16
        if submap > nside:
            submap = nside

        pars[ 'temperature_only' ] = 'F'
        pars[ 'force_pol' ] = 'T'
        pars[ 'kfirst' ] = 'F' # 'T' DEBUG
        pars[ 'concatenate_messages' ] = 'T'
        pars[ 'write_map' ] = 'T'
        pars[ 'write_binmap' ] = 'T'
        pars[ 'write_matrix' ] = 'T'
        pars[ 'write_wcov' ] = 'T'
        pars[ 'write_hits' ] = 'T'
        pars[ 'run_submap_test' ] = 'T'
        pars[ 'nside_cross' ] = cross
        pars[ 'nside_submap' ] = submap

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

        pars[ 'base_first' ] = args.baseline
        pars[ 'nside_map' ] = nside
        if args.noisefilter:
            pars[ 'kfilter' ] = 'T'
        else:
            pars[ 'kfilter' ] = 'F'
        pars[ 'fsample' ] = args.samplerate

        # Loop over Monte Carlos

        firstmc = int(args.MC_start)
        nmc = int(args.MC_count)

        for mc in range(firstmc, firstmc+nmc):

            # Copy the signal timestreams to the total ones before
            # accumulating the noise.

            sigcopy.exec(data)

            # simulate noise

            nse = tt.OpSimNoise(out='total', realization=mc)
            nse.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('Noise simulation took {:.3f} s'.format(elapsed),
                      flush=True)
            start = stop

            # create output directory for this realization
            pars[ 'path_output' ] = '{}_{:03d}'.format(args.outdir, mc)
            if comm.comm_world.rank == 0:
                if not os.path.isdir(pars['path_output']):
                    os.makedirs(pars['path_output'])

            """
            # in debug mode, print out data distribution information
            if args.debug:
                handle = None
                if comm.comm_world.rank == 0:
                    handle = open(
                        os.path.join(pars['path_output'], 'distdata.txt'), 'w')
                data.info(handle)
                if comm.comm_world.rank == 0:
                    handle.close()
            """

            madam = tm.OpMadam(params=pars, detweights=detweights, name='total',
                               common_flag_mask=args.common_flag_mask)
            madam.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('Mapmaking took {:.3f} s'.format(elapsed), flush=True)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - global_start
    if comm.comm_world.rank == 0:
        print('Total Time:  {:.2f} seconds'.format(elapsed), flush=True)


if __name__ == '__main__':
    main()
