#!/usr/bin/env python3

import mpi4py.MPI as MPI

import os
import re

import numpy as np

import quaternionarray as qa

import toast
import toast.tod as tt
import toast.map as tm


# scanning parameters

samplerate = 25.0 # Hz
spinperiod = 1.0 # minutes
spinangle = 85.0 # degrees
precperiod = 0 # minutes
precangle = 0 # degrees

# we add a HWP here, since we have one detector for just a couple days

hwprpm = 60.0

# map making

nside = 256
baseline = 60.0

# make a fake focalplane

fake = {}
fake['quat'] = np.array([0.0, 0.0, 1.0, 0.0])
fake['fwhm'] = 30.0
fake['fknee'] = 0.1
fake['alpha'] = 1.5
fake['NET'] = 0.0002
fp = {}
fp['bore'] = fake

# Since madam only supports a single observation, we use
# that here so that we can use the same data distribution whether
# or not we are using libmadam.  Normally we would have multiple 
# observations with some subset assigned to each process group.

# This is the 2-level toast communicator.  By default,
# there is just one group which spans MPI_COMM_WORLD.

comm = toast.Comm()

# construct the list of intervals.  We'll use 55 minute intervals
# with a 5 minute gap. and observe for 4 days.

numobs = 4 * 24
science = 55 * 60
gap = 5 * 60

intervals = tt.regular_intervals(numobs, 0.0, 0, samplerate, science, gap)

# how many samples in one interval (plus the gap)?

interval_samples = intervals[1].first - intervals[0].first

# when we distribute the data, we don't want to split these intervals
# between processes.  So make a list of the samples in each interval
# and pass that list when constructing the TOD class.  This way it 
# distributes the data as evenly as possible among the processes.

distsizes = []
for it in intervals:
    distsizes.append(interval_samples)

# how many total samples do we have now?

totsamples = np.sum(distsizes)

# create the single TOD for this observation

detectors = sorted(fp.keys())
detquats = {}
for d in detectors:
    detquats[d] = fp[d]['quat']

tod = tt.TODSatellite(
    comm.comm_group, 
    detquats,
    totsamples,
    firsttime=0.0,
    rate=samplerate,
    spinperiod=spinperiod,
    spinangle=spinangle,
    precperiod=precperiod,
    precangle=precangle,
    sampsizes=distsizes
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

# The distributed timestream data

data = toast.Data(comm)

# Create the (single) observation

ob = {}
ob['name'] = 'mission'
ob['tod'] = tod
ob['intervals'] = intervals
ob['baselines'] = None
ob['noise'] = noise
ob['id'] = 0

data.obs.append(ob)

# Constantly slewing precession axis, so that it makes a circle in one year

degday = 360.0 / 365.25

precquat = tt.slew_precession_axis(nsim=tod.local_samples[1], 
    firstsamp=tod.local_samples[0], samplerate=samplerate, degday=degday)

# we set the precession axis now, which will trigger calculation
# of the boresight pointing.

tod.set_prec_axis(qprec=precquat)

# simulate noise

nse = tt.OpSimNoise(out='simdata')
nse.exec(data)

# make a Healpix pointing matrix.

pointing = tt.OpPointingHpix(nside=nside, nest=True, mode='IQU', hwprpm=hwprpm)
pointing.exec(data)

# Simulate a gradient signal and accumulate it to the same output
# as the noise simulation.

grad = tt.OpSimGradient(out='simdata', nside=nside, min=-1.0, max=1.0, nest=True)
grad.exec(data)

# Mapmaking.  For purposes of this simulation, we use detector noise
# weights based on the NET (white noise level).  If the destriping
# baseline is too long, this will not be the best choice.

detweights = {}
for d in detectors:
    net = fp[d]['NET']
    detweights[d] = 1.0 / (samplerate * net * net)

# Set up MADAM map making.  By setting purge=True, we will
# purge all data after copying it into the madam
# buffers.  This is ok, as long as madam is the last step of 
# the pipeline.

outdir = 'out_example_customize'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

pars = {}

cross = int(nside / 2)
submap = int(nside / 8)

pars[ 'temperature_only' ] = 'F'
pars[ 'force_pol' ] = 'T'
pars[ 'kfirst' ] = 'T'
pars[ 'base_first' ] = baseline
pars[ 'nside_map' ] = nside
pars[ 'nside_cross' ] = cross
pars[ 'nside_submap' ] = submap
pars[ 'write_map' ] = 'T'
pars[ 'write_binmap' ] = 'T'
pars[ 'write_matrix' ] = 'T'
pars[ 'write_wcov' ] = 'T'
pars[ 'write_hits' ] = 'T'
pars[ 'kfilter' ] = 'F'
pars[ 'fsample' ] = samplerate
pars[ 'path_output' ] = outdir

madam = tm.OpMadam(params=pars, detweights=detweights, name='simdata', purge=True)
madam.exec(data)

