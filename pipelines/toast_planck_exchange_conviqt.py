# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import mpi4py.MPI as MPI

import toast
import toast.tod
import toast.map
import toast.exp.planck as tp

import re
import argparse

import time

import numpy as np

parser = argparse.ArgumentParser( description='Simple on-the-fly signal convolution + MADAM Mapmaking' )
parser.add_argument( '--lmax', required=True, help='Simulation lmax' )
parser.add_argument( '--skylmax', required=True, help='Sky lmax' )
parser.add_argument( '--beamlmax', required=True, help='Beam lmax' )
parser.add_argument( '--beammmax', required=True, help='Beam mmax' )
parser.add_argument( '--skyfile', required=True, help='Path to sky alm files. Tag DETECTOR will be replaced with detector name.' )
parser.add_argument( '--beamfile', required=True, help='Path to beam alm files. Tag DETECTOR will be replaced with detector name.' )
parser.add_argument( '--rimo', required=True, help='RIMO file' )
parser.add_argument( '--freq', required=True, help='Frequency' )
parser.add_argument( '--debug', dest='debug', default=False, action='store_true', help='Write data distribution info to file')
parser.add_argument( '--highmem', dest='highmem', default=False, action='store_true', help='Do not purge intermediate data products from memory')
parser.add_argument( '--dets', required=False, default=None, help='Detector list (comma separated)' )
parser.add_argument( '--effdir', required=True, help='Input Exchange Format File directory' )
parser.add_argument( '--ringdb', required=True, help='Ring DB file' )
parser.add_argument( '--odfirst', required=False, default=None, help='First OD to use' )
parser.add_argument( '--odlast', required=False, default=None, help='Last OD to use' )
parser.add_argument( '--ringfirst', required=False, default=None, help='First ring to use' )
parser.add_argument( '--ringlast', required=False, default=None, help='Last ring to use' )
parser.add_argument( '--obtfirst', required=False, default=None, help='First OBT to use' )
parser.add_argument( '--obtlast', required=False, default=None, help='Last OBT to use' )
parser.add_argument( '--madampar', required=False, default=None, help='Madam parameter file' )
parser.add_argument( '--out', required=False, default='.', help='Output directory' )

args = parser.parse_args()

start = MPI.Wtime()

odrange = None
if args.odfirst is not None and args.odlast is not None:
    odrange = (int(args.odfirst), int(args.odlast))

ringrange = None
if args.ringfirst is not None and args.ringlast is not None:
    ringrange = (int(args.ringfirst), int(args.ringlast))

obtrange = None
if args.obtfirst is not None and args.obtlast is not None:
    obtrange = (float(args.obtfirst), float(args.obtlast))

detectors = None
if args.dets is not None:
    detectors = re.split(',', args.dets)

# This is the 2-level toast communicator.  By default,
# there is just one group which spans MPI_COMM_WORLD.
comm = toast.Comm()

# This is the distributed data, consisting of one or
# more observations, each distributed over a communicator.
data = toast.Data(comm)

# Read in madam parameter file
pars = {}

if comm.comm_world.rank == 0:
    if args.madampar is not None:
        pat = re.compile(r'\s*(\S+)\s*=\s*(\S+)\s*')
        comment = re.compile(r'^#.*')
        with open(args.madampar, 'r') as f:
            for line in f:
                if not comment.match(line):
                    result = pat.match(line)
                    if result:
                        pars[result.group(1)] = result.group(2)
    else:
        pars[ 'kfirst' ] = False
        pars[ 'temperature_only' ] = False
        pars[ 'base_first' ] = 60.0
        pars[ 'fsample' ] = 180.35
        pars[ 'nside_map' ] = 1024
        pars[ 'nside_cross' ] = 1024
        pars[ 'nside_submap' ] = 16
        pars[ 'write_map' ] = False
        pars[ 'write_binmap' ] = True
        pars[ 'write_matrix' ] = False
        pars[ 'write_wcov' ] = False
        pars[ 'write_hits' ] = True
        pars[ 'kfilter' ] = False
        pars[ 'run_submap_test' ] = False
        pars[ 'path_output' ] = './'
        pars[ 'info' ] = 3

pars = comm.comm_world.bcast(pars, root=0)

# madam only supports a single observation.  Normally
# we would have multiple observations with some subset
# assigned to each process group.

# create the TOD for this observation

tod = tp.Exchange(
    mpicomm=comm.comm_group, 
    detectors=detectors,
    ringdb=args.ringdb,
    effdir=args.effdir,
    obt_range=obtrange,
    ring_range=ringrange,
    od_range=odrange,
    freq=int(args.freq),
    RIMO=args.rimo,
    obtmask=1,
    flagmask=1,
)

RIMO = tod.rimo

# normally we would get the intervals from somewhere else, but since
# the Exchange TOD already had to get that information, we can
# get it from there.

ob = {}
ob['id'] = 'mission'
ob['tod'] = tod
ob['intervals'] = tod.valid_intervals
ob['baselines'] = None
ob['noise'] = None

#times = tod.read_times( n=tod.local_samples )
#print( '{:4} : Processing {} chunks : {}. Local offset : {}, samples {} / {}. Between {} and {}.'.format( comm.comm_world.rank, len(tod.local_chunks), tod.local_chunks, tod.local_offset, tod.local_samples, tod.total_samples, times[0], times[-1] ) )

data.obs.append(ob)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Metadata queries took {:.3f} s".format(elapsed))
start = stop

# cache the data in memory
cache = toast.tod.OpCopy()
cache.exec(data)

tod.purge_eff_cache()
#del tod # Also deleting "tod" will purge the cache but you cannot reference other useful "tod" member later on

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Data read and cache took {:.3f} s".format(elapsed))
start = stop

# simulate the TOD by convolving the sky with the beams

detectordata = []
for det in tod.detectors:
    skyfile = args.skyfile.replace('DETECTOR',det)
    beamfile = args.beamfile.replace('DETECTOR',det)
    epsilon = RIMO[det].epsilon
    # Getting the right polarization angle can be a sensitive matter. Dxx beams are always defined
    # without psi_uv or psi_pol rotation but some Pxx beams may require psi_pol to be removed and psi_uv left in.
    psipol = np.radians(RIMO[det].psi_uv + RIMO[det].psi_pol)
    #psipol = np.radians(RIMO[det].psi_pol) # This would work if a Pxx beam that is defined in the psi_uv frame, provided that the convolver has Dxx=True...
    detectordata.append((det, skyfile, beamfile, epsilon, psipol))

conviqt = toast.tod.OpSimConviqt(int(args.skylmax), int(args.beamlmax), int(args.beammmax), detectordata, lmaxout=int(args.lmax))
conviqt.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Convolution took {:.3f} s".format(elapsed))
start = stop

# make a planck Healpix pointing matrix
mode = 'IQU'
if pars['temperature_only'] == 'T':
    mode = 'I'
pointing = tp.OpPointingPlanck(nside=int(pars['nside_map']), mode=mode, RIMO=RIMO, highmem=args.highmem)
pointing.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Pointing Matrix took {:.3f} s, mode = {}".format(elapsed,mode))
start = stop

# for now, we pass in the noise weights from the RIMO.
# once the noise class is implemented, then the madam
# operator can use that.
detweights = {}
for d in tod.detectors:
    net = tod.rimo[d].net
    detweights[d] = 1.0 / (net * net)

if args.debug:
    with open("debug_planck_exchange_madam.txt", "w") as f:
        data.info(f)

madam = toast.map.OpMadam(params=pars, detweights=detweights, highmem=args.highmem)
madam.exec(data)

comm.comm_world.barrier()
stop = MPI.Wtime()
elapsed = stop - start
if comm.comm_world.rank == 0:
    print("Madam took {:.3f} s".format(elapsed))

