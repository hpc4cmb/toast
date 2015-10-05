# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import toast

import argparse


parser = argparse.ArgumentParser( description='Simple MADAM Mapmaking' )
parser.add_argument( '--rimo', required=True, help='RIMO file' )
parser.add_argument( '--freq', required=True, help='Frequency' )
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
    odrange = (args.odfirst, args.odlast)

ringrange = None
if args.ringfirst is not None and args.ringlast is not None:
    ringrange = (args.ringfirst, args.ringlast)

obtrange = None
if args.obtfirst is not None and args.obtlast is not None:
    obtrange = (args.obtfirst, args.obtlast)

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
        pars[ 'base_first' ] = 60.0
        pars[ 'fsample' ] = 180.35
        pars[ 'nside_map' ] = 1024
        pars[ 'nside_cross' ] = 1024
        pars[ 'nside_submap' ] = 1024
        pars[ 'write_map' ] = False
        pars[ 'write_binmap' ] = True
        pars[ 'write_matrix' ] = False
        pars[ 'write_wcov' ] = False
        pars[ 'write_hits' ] = True
        pars[ 'kfilter' ] = False
        pars[ 'run_submap_test' ] = False
        pars[ 'path_output' ] = './'

comm.comm_world.bcast(pars, root=0)

# madam only supports a single observation.  Normally
# we would have multiple observations with some subset
# assigned to each process group.

# create the TOD for this observation

tod = planck.Exchange(
    mpicomm=comm.comm_group, 
    detectors=args.dets,
    ringdb=args.ringdb,
    effdir=args.effdir,
    obt_range=obtrange,
    ring_range=ringrange,
    od_range=odrange,
    freq=args.freq,
    RIMO=args.rimo
)

# normally we would get the intervals from somewhere else, but since
# the Exchange TOD already had to get that information, we can
# get it from there.

data.obs.append( 
    Obs( 
        tod = tod,
        intervals = tod.valid_intervals(),
        baselines = None, 
        noise = None
    )
)

stop = MPI.Wtime()
elapsed = stop - start
if mpicomm.rank == 0:
    print("Metadata queries took {:.3f} s".format(elapsed))
start = stop

# cache the data in memory
cache = OpCopy()
data = cache.exec(data)

stop = MPI.Wtime()
elapsed = stop - start
if mpicomm.rank == 0:
    print("Data read and cache took {:.3f} s".format(elapsed))
start = stop

# make a planck Healpix pointing matrix
# FIXME: get mode from madam parameter if T-only
pointing = OpPointingPlanck(nside=pars['nside_map'])
pointing.exec(data)

stop = MPI.Wtime()
elapsed = stop - start
if mpicomm.rank == 0:
    print("Pointing Matrix took {:.3f} s".format(elapsed))
start = stop

# for now, we pass in the noise weights from the RIMO.
# once the noise class is implemented, then the madam
# operator can use that.
detweights = {}
for d in tod.detectors:
    net = tod.rimo()[d]['net']
    detweights[d] = net * net

madam = OpMadam(params=pars, detweights=detweights)
madam.exec(data)

stop = MPI.Wtime()

elapsed = stop - start
if mpicomm.rank == 0:
    print("Madam took {:.3f} s".format(elapsed))

