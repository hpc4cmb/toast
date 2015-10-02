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
parser.add_argument( '--odfirst', required=False, default=91, help='First OD to use' )
parser.add_argument( '--odlast', required=False, default=110, help='Last OD to use' )
parser.add_argument( '--madampar', required=False, default=None, help='Madam parameter file' )
parser.add_argument( '--out', required=False, default='.', help='Output directory' )
args = parser.parse_args()


# This is the 2-level toast communicator.  By default,
# there is just one group which spans MPI_COMM_WORLD.
comm = toast.Comm()

data = toast.Data(comm)

# madam only supports a single observation.  Normally
# we would have multiple observations with some subset
# assigned to each process group.



# create the TOD for this observation

tod = planck.Exchange(
    mpicomm=comm.comm_group, 
    detectors=args.dets,
    fn_ringdb=args.ringdb,
    effdir=args.effdir,

    samples=self.totsamp,
    rate=self.rate,
    nside=self.sim_nside
)

mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, fn_ringdb=None, effdir=None, obt_range=None, ring_range=None, od_range=None, freq=None, RIMO=None, coord='G', mode='THETAPHIPSI', deaberrate=True, order='RING', nside=2048, obtmask=0, flagmask=0, bufsize=100000)

data.obs.append( 
    Obs( 
        tod = tod,
        intervals = [],
        baselines = None, 
        noise = None
    )
)




        start = MPI.Wtime()

        # cache the data in memory
        cache = OpCopy()
        data = cache.exec(self.data)

        # add simple sky gradient signal
        grad = OpSimGradient(nside=self.sim_nside)
        grad.exec(data)

        # make a simple pointing matrix
        pointing = OpPointingFake(nside=self.map_nside, nest=True)
        pointing.exec(data)

        pars = {}
        pars[ 'kfirst' ] = False
        pars[ 'base_first' ] = 1.0
        pars[ 'fsample' ] = self.rate
        pars[ 'nside_map' ] = self.map_nside
        pars[ 'nside_cross' ] = self.map_nside
        pars[ 'nside_submap' ] = self.map_nside
        pars[ 'write_map' ] = False
        pars[ 'write_binmap' ] = True
        pars[ 'write_matrix' ] = False
        pars[ 'write_wcov' ] = False
        pars[ 'write_hits' ] = True
        pars[ 'kfilter' ] = False
        pars[ 'run_submap_test' ] = False
        pars[ 'path_output' ] = './'

        madam = OpMadam(params=pars)
        madam.exec(data)

        stop = MPI.Wtime()
        elapsed = stop - start
        self.print_in_turns("Madam test took {:.3f} s".format(elapsed))

