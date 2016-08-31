#!/usr/bin/env python

import os
if 'TOAST_NO_MPI' in os.environ.keys():
    from toast import fakempi as MPI
else:
    from mpi4py import MPI

import re
import argparse

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np

import quaternionarray as qa

import toast
import toast.tod as tt
import toast.map as tm


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

    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    yaxis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for det in sorted(fp.keys()):

        # radius in degrees
        detradius = 0.5 * fp[det]['fwhm'] / 60.0

        # rotation from boresight
        dir = qa.rotate(fp[det]['quat'], zaxis).flatten()
        ang = np.arctan2(dir[1], dir[0])

        orient = qa.rotate(fp[det]['quat'], xaxis).flatten()
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
        ax.arrow(xtail, ytail, dx, dy, width=0.1*detradius, head_width=0.3*detradius, head_length=0.3*detradius, fc=detcolor, ec=detcolor, length_includes_head=True)

    plt.savefig(outfile)
    return


def main():

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    if comm.comm_world.rank == 0:
        print("Running with {} processes".format(comm.comm_world.size))

    global_start = MPI.Wtime()

    parser = argparse.ArgumentParser( description='Simulate satellite boresight pointing and make a noise map.' )
    parser.add_argument( '--samplerate', required=False, default=40.0, help='Detector sample rate (Hz)' )
    parser.add_argument( '--spinperiod', required=False, default=10.0, help='The period (in minutes) of the rotation about the spin axis' )
    parser.add_argument( '--spinangle', required=False, default=30.0, help='The opening angle (in degrees) of the boresight from the spin axis' )
    parser.add_argument( '--precperiod', required=False, default=50.0, help='The period (in minutes) of the rotation about the precession axis' )
    parser.add_argument( '--precangle', required=False, default=65.0, help='The opening angle (in degrees) of the spin axis from the precession axis' )
    parser.add_argument( '--hwprpm', required=False, default=0.0, help='The rate (in RPM) of the HWP rotation' )
    parser.add_argument( '--hwpstep', required=False, default=None, help='For stepped HWP, the angle in degrees of each step' )
    parser.add_argument( '--hwpsteptime', required=False, default=0.0, help='For stepped HWP, the the time in seconds between steps' )

    parser.add_argument( '--obs', required=False, default=1.0, help='Number of hours in one science observation' )
    parser.add_argument( '--gap', required=False, default=0.0, help='Cooler cycle time in hours between science obs' )
    parser.add_argument( '--numobs', required=False, default=1, help='Number of complete observations' )
    parser.add_argument( '--obschunks', required=False, default=1, help='Number of chunks to subdivide each observation into for data distribution' )
    parser.add_argument( '--outdir', required=False, default='.', help='Output directory' )
    parser.add_argument( '--debug', required=False, default=False, action='store_true', help='Write diagnostics' )

    parser.add_argument( '--nside', required=False, default=64, help='Healpix NSIDE' )
    parser.add_argument( '--baseline', required=False, default=60.0, help='Destriping baseline length (seconds)' )
    parser.add_argument( '--noisefilter', required=False, default=False, action='store_true', help='Destripe with the noise filter enabled' )

    parser.add_argument( '--madam', required=False, default=False, action='store_true', help='If specified, use libmadam for map-making' )
    parser.add_argument( '--madampar', required=False, default=None, help='Madam parameter file' )

    parser.add_argument( '--MC_start', required=False, default=0, help='First Monte Carlo noise realization' )
    parser.add_argument( '--MC_count', required=False, default=1, help='Number of Monte Carlo noise realizations' )
    
    parser.add_argument( '--fp', required=False, default=None, help='Pickle file containing a dictionary of detector properties.  The keys of this dict are the detector names, and each value is also a dictionary with keys "quat" (4 element ndarray), "fwhm" (float, arcmin), "fknee" (float, Hz), "alpha" (float), and "NET" (float).  For optional plotting, the key "color" can specify a valid matplotlib color string.' )
    
    args = parser.parse_args()

    # get options

    samplerate = float(args.samplerate)
    spinperiod = float(args.spinperiod)
    spinangle = float(args.spinangle)
    precperiod = float(args.precperiod)
    precangle = float(args.precangle)

    hwprpm = float(args.hwprpm)
    hwpstep = None
    if args.hwpstep is not None:
        hwpstep = float(args.hwpstep)
    hwpsteptime = float(args.hwpsteptime)

    nside = int(args.nside)
    baseline = float(args.baseline)

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
            fake['alpha'] = 1.0
            fake['NET'] = 1.0
            fake['color'] = 'r'
            fp = {}
            fp['bore'] = fake
        else:
            with open(args.fp, "rb") as p:
                fp = pickle.load(p)
    fp = comm.comm_world.bcast(fp, root=0)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Create focalplane:  {:.2f} seconds".format(stop-start))
    start = stop

    if args.debug:
        if comm.comm_world.rank == 0:
            outfile = "{}_focalplane.png".format(args.outdir)
            view_focalplane(fp, outfile)

    # Since madam only supports a single observation, we use
    # that here so that we can use the same data distribution whether
    # or not we are using libmadam.  Normally we would have multiple 
    # observations with some subset assigned to each process group.

    # The distributed timestream data

    data = toast.Data(comm)

    # construct the list of intervals

    obschunks = int(args.obschunks)

    intervals = tt.regular_intervals(int(args.numobs), 0.0, 0, samplerate, 3600*float(args.obs), 3600*float(args.gap), chunks=obschunks)

    distsizes = []
    for it in intervals:
        distsizes.append(it.last - it.first + 1)

    totsamples = np.sum(distsizes)

    # create the single TOD for this observation

    detectors = sorted(fp.keys())
    detquats = {}
    for d in detectors:
        detquats[d] = fp[d]['quat']

    tod = tt.TODSatellite(
        mpicomm=comm.comm_group, 
        detectors=detquats,
        samples=totsamples,
        firsttime=0.0,
        rate=samplerate,
        spinperiod=spinperiod,
        spinangle=spinangle,
        precperiod=precperiod,
        precangle=precangle,
        sizes=distsizes
    )

    # Create the noise model for this observation

    fmin = 2.0 / samplerate
    fknee = {}
    alpha = {}
    NET = {}
    for d in detectors:
        fknee[d] = fp[d]['fknee']
        alpha[d] = fp[d]['alpha']
        NET[d] = fp[d]['NET']

    noise = tt.AnalyticNoise(rate=samplerate, fmin=fmin, detectors=detectors, fknee=fknee, alpha=alpha, NET=NET)

    # Create the (single) observation

    ob = {}
    ob['id'] = 'mission'
    ob['tod'] = tod
    ob['intervals'] = intervals
    ob['baselines'] = None
    ob['noise'] = noise

    data.obs.append(ob)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Read parameters, compute data distribution:  {:.2f} seconds".format(stop-start))
    start = stop

    # Constantly slewing precession axis

    degday = 360.0 / 365.25

    precquat = tt.slew_precession_axis(nsim=tod.local_samples[1], firstsamp=tod.local_samples[0], samplerate=samplerate, degday=degday)

    # we set the precession axis now, which will trigger calculation
    # of the boresight pointing.
    tod.set_prec_axis(qprec=precquat)

    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Construct boresight pointing:  {:.2f} seconds".format(stop-start))
    start = stop

    # make a Healpix pointing matrix.

    pointing = tt.OpPointingHpix(nside=nside, nest=True, mode='IQU', hwprpm=hwprpm, hwpstep=hwpstep, hwpsteptime=hwpsteptime)
    pointing.exec(data)

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - start
    if comm.comm_world.rank == 0:
        print("Pointing generation took {:.3f} s".format(elapsed))
    start = stop

    # Mapmaking.  For purposes of this simulation, we use detector noise
    # weights based on the NET (white noise level).  If the destriping
    # baseline is too long, this will not be the best choice.

    detweights = {}
    for d in detectors:
        net = fp[d]['NET']
        detweights[d] = 1.0 / (samplerate * net * net)

    if not args.madam:
        raise RuntimeError("Internal pytoast mapmaking not yet implemented, use the --madam option for now.")
    else:

        # Set up MADAM map making.  By setting purge=True, we will
        # purge all data after copying it into the madam
        # buffers.  This is ok, as long as madam is the last step of 
        # the pipeline.

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

        pars[ 'base_first' ] = baseline
        pars[ 'nside_map' ] = nside
        if args.noisefilter:
            pars[ 'kfilter' ] = 'T'
        else:
            pars[ 'kfilter' ] = 'F'
        pars[ 'fsample' ] = samplerate

        # Loop over Monte Carlos

        firstmc = int(args.MC_start)
        nmc = int(args.MC_count)

        for mc in range(firstmc, firstmc+nmc):
            # clear all noise data from the cache, so that we can generate
            # new noise timestreams.
            tod.cache.clear("noise_.*")

            # simulate noise

            nse = tt.OpSimNoise(out="noise", stream=0, realization=mc)
            nse.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Noise simulation took {:.3f} s".format(elapsed))
            start = stop

            # create output directory for this realization
            pars[ 'path_output' ] = "{}_{:03d}".format(args.outdir, mc)
            if comm.comm_world.rank == 0:
                if not os.path.isdir(pars['path_output']):
                    os.makedirs(pars['path_output'])

            # in debug mode, print out data distribution information
            if args.debug:
                handle = None
                if comm.comm_world.rank == 0:
                    handle = open(os.path.join(pars['path_output'],"distdata.txt"), "w")
                data.info(handle)
                if comm.comm_world.rank == 0:
                    handle.close()

            madam = tm.OpMadam(params=pars, detweights=detweights, name='noise')
            madam.exec(data)

            comm.comm_world.barrier()
            stop = MPI.Wtime()
            elapsed = stop - start
            if comm.comm_world.rank == 0:
                print("Mapmaking took {:.3f} s".format(elapsed))

    comm.comm_world.barrier()
    stop = MPI.Wtime()
    elapsed = stop - global_start
    if comm.comm_world.rank == 0:
        print("Total Time:  {:.2f} seconds".format(elapsed))


if __name__ == "__main__":
    main()

