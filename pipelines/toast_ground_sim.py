#!/usr/bin/env python

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

from toast.mpi import MPI

import os
import re
import argparse
import pickle

import numpy as np
from scipy.constants import degree

import toast
import toast.tod as tt
import toast.map as tm
import toast.tod.qarray as qa


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
        print("Running with {} processes".format(comm.comm_world.size))

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser(description='Simulate ground-based '
                                     'boresight pointing and make a noise map.')
    parser.add_argument('--samplerate',
                        required=False, default=40.0, type=np.float,
                        help='Detector sample rate (Hz)')
    parser.add_argument('--site_lon',
                        required=False, default=10.0,
                        help='Observing site longitude [pyEphem string]')
    parser.add_argument('--site_lat',
                        required=False, default=10.0,
                        help='Observing site latitude [pyEphem string]')
    parser.add_argument('--site_alt',
                        required=False, default=10.0, type=np.float,
                        help='Observing site altitude [meters]')
    parser.add_argument('--patch_lon',
                        required=False, default=30.0,
                        help='Sky patch longitude [pyEphem string]')
    parser.add_argument('--patch_lat',
                        required=False, default=30.0,
                        help='Sky patch latitude [pyEphem string]')
    parser.add_argument('--patch_coord',
                        required=False, default='C',
                        help='Sky patch coordinate system [C,E,G]')
    parser.add_argument('--throw',
                        required=False, default=5.0, type=np.float,
                        help='Sky patch width in azimuth [degrees]')
    parser.add_argument('--scanrate',
                        required=False, default=6.0, type=np.float,
                        help='Scanning rate [deg / s]')
    parser.add_argument('--scan_accel',
                        required=False, default=6.0, type=np.float,
                        help='Scanning rate change [deg / s^2]')
    parser.add_argument('--el_min',
                        required=False, default=0.0, type=np.float,
                        help='Minimum elevation for a CES')
    parser.add_argument('--sun_angle_min',
                        required=False, default=90.0, type=np.float,
                        help='Minimum angular distance between the Sun and '
                        'the sky patch')
    parser.add_argument('--allow_sun_up',
                        required=False, default=False, action='store_true',
                        help='If specified, allow day time scans.')
    parser.add_argument('--hwprpm',
                        required=False, default=0.0, type=np.float,
                        help='The rate (in RPM) of the HWP rotation')
    parser.add_argument('--hwpstep', required=False, default=None,
                        help='For stepped HWP, the angle in degrees of each step')
    parser.add_argument('--hwpsteptime',
                        required=False, default=0.0, type=np.float,
                        help='For stepped HWP, the the time in seconds between steps')
    parser.add_argument('--CES_start',
                        required=False, default=0, type=np.float,
                        help='Start time of the CES')
    parser.add_argument('--CES_stop',
                        required=False, default=1000, type=np.float,
                        help='Stop time of the CES')
    parser.add_argument('--outdir',
                        required=False, default='out',
                        help='Output directory')
    parser.add_argument('--debug',
                        required=False, default=False, action='store_true',
                        help='Write diagnostics')
    parser.add_argument('--nside',
                        required=False, default=1024, type=np.int,
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

    if comm.comm_world.rank == 0:
        if args.fp is None:
            # in this case, create a fake detector at the boresight
            # with a pure white noise spectrum.
            fake = {}
            fake['quat'] = np.array([0.0, 0.0, 1.0, 0.0])
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
        print('Create focalplane:  {:.2f} seconds'.format(stop-start))
    start = stop

    if args.debug:
        if comm.comm_world.rank == 0:
            outfile = '{}_focalplane.png'.format(args.outdir)
            view_focalplane(fp, outfile)

    # The distributed timestream data

    data = toast.Data(comm)

    # FIXME: what format will we assume the CES times to be?

    CES_start = args.CES_start
    CES_stop = args.CES_stop

    totsamples = int((CES_stop - CES_start) * args.samplerate)

    # create the single TOD for this observation

    detectors = sorted(fp.keys())
    detquats = {}
    for d in detectors:
        detquats[d] = fp[d]['quat']

    try:
        tod = tt.TODGround(
            comm.comm_group, 
            detquats,
            totsamples,
            firsttime=CES_start,
            rate=args.samplerate,
            site_lon=args.site_lon,
            site_lat=args.site_lat,
            site_alt=args.site_alt,
            patch_lon=args.patch_lon,
            patch_lat=args.patch_lat,
            patch_coord=args.patch_coord,
            throw=args.throw,
            scanrate=args.scanrate,
            scan_accel=args.scan_accel,
            CES_start=None,
            CES_stop=None,
            el_min=args.el_min,
            sun_angle_min=args.sun_angle_min,
            allow_sun_up=args.allow_sun_up,
            sampsizes=None)
    except RuntimeError as e:
        print('Failed to create the CES scan: {}'.format(e))
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
    ob['name'] = 'CES'
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
              ''.format(stop-start))
    start = stop

    # make a Healpix pointing matrix.

    pointing = tt.OpPointingHpix(
        nside=nside, nest=True, mode='IQU', hwprpm=hwprpm, hwpstep=hwpstep,
        hwpsteptime=hwpsteptime)

    pointing.exec(data)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print('Pointing generation took {:.3f} s'.format(elapsed))
    start = stop

    # Mapmaking.  For purposes of this simulation, we use detector noise
    # weights based on the NET (white noise level).  If the destriping
    # baseline is too long, this will not be the best choice.

    detweights = {}
    for d in detectors:
        net = fp[d]['NET']
        detweights[d] = 1.0 / (args.samplerate * net * net)

    if not args.madam:
        if comm.comm_world.rank == 0:
            print('Not using Madam, will only make a binned map!')

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
            print('Building hits and N_pp^-1 took {:.3f} s'.format(elapsed))
        start = stop

        hits.write_healpix_fits('{}_hits.fits'.format(args.outdir))
        invnpp.write_healpix_fits('{}_invnpp.fits'.format(args.outdir))

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Writing hits and N_pp^-1 took {:.3f} s'.format(elapsed))
        start = stop

        # invert it
        tm.covariance_invert(invnpp, 1.0e-3)

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Inverting N_pp^-1 took {:.3f} s'.format(elapsed))
        start = stop

        invnpp.write_healpix_fits('{}_npp.fits'.format(args.outdir))

        comm.comm_world.barrier()
        stop = MPI.Wtime()
        elapsed = stop - start
        if comm.comm_world.rank == 0:
            print('Writing N_pp took {:.3f} s'.format(elapsed))
        start = stop

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
                      ''.format(mc, elapsed))
            start = stop

            # clear all noise data from the cache, so that we can generate
            # new noise timestreams.
            tod.cache.clear('noise_.*')

            # simulate noise

            nse = tt.OpSimNoise(out='noise', realization=mc)
            nse.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Noise simulation {:04d} took {:.3f} s'
                      ''.format(mc, elapsed))
            start = stop

            zmap.data.fill(0.0)
            build_zmap = tm.OpAccumDiag(zmap=zmap, name='noise')
            build_zmap.exec(data)
            zmap.allreduce()

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Building noise weighted map {:04d} took {:.3f} s'
                      ''.format(mc, elapsed))
            start = stop

            tm.covariance_apply(invnpp, zmap)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Computing binned map {:04d} took {:.3f} s'
                      ''.format(mc, elapsed))
            start = stop
        
            zmap.write_healpix_fits(os.path.join(outpath, 'binned.fits'))

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('  Writing binned map {:04d} took {:.3f} s'
                      ''.format(mc, elapsed))
            elapsed = stop - mcstart
            if comm.comm_world.rank == 0:
                print('  Mapmaking {:04d} took {:.3f} s'.format(mc, elapsed))
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
        pars[ 'kfirst' ] = 'T'
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
            # clear all noise data from the cache, so that we can generate
            # new noise timestreams.
            tod.cache.clear('noise_.*')

            # simulate noise

            nse = tt.OpSimNoise(out='noise', realization=mc)
            nse.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('Noise simulation took {:.3f} s'.format(elapsed))
            start = stop

            # create output directory for this realization
            pars[ 'path_output' ] = '{}_{:03d}'.format(args.outdir, mc)
            if comm.comm_world.rank == 0:
                if not os.path.isdir(pars['path_output']):
                    os.makedirs(pars['path_output'])

            # in debug mode, print out data distribution information
            if args.debug:
                handle = None
                if comm.comm_world.rank == 0:
                    handle = open(
                        os.path.join(pars['path_output'], 'distdata.txt'), 'w')
                data.info(handle)
                if comm.comm_world.rank == 0:
                    handle.close()

            madam = tm.OpMadam(params=pars, detweights=detweights, name='noise',
                               common_flag_mask=args.common_flag_mask)
            madam.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print('Mapmaking took {:.3f} s'.format(elapsed))

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - global_start
    if comm.comm_world.rank == 0:
        print('Total Time:  {:.2f} seconds'.format(elapsed))


if __name__ == '__main__':
    main()
