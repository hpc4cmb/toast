#!/usr/bin/env python

import mpi4py.MPI as MPI

import toast
import toast.tod as tt

# Split COMM_WORLD into groups of 4 processes each
cm = toast.Comm(world=MPI.COMM_WORLD, groupsize=4)

# Create the distributed data object
dd = toast.Data(comm=cm)

# Each process group appends some observations.
# For this example, each observation is going to have the same
# number of samples, and the same list of detectors.  We just
# use the base TOD class, which contains the data in memory.

obs_samples = 100
obs_dets = ['detA', 'detB', 'detC']

for i in range(10):
    tod = tt.TOD(cm.comm_group, obs_dets, obs_samples)
    indx = cm.group * 10 + i
    ob = {}
    ob['name'] = '{}'.format(indx)
    ob['tod'] = tod
    ob['intervals'] = None
    ob['noise'] = None
    ob['id'] = indx
    dd.obs.append(ob)

# Now at the end we have 4 process groups, each of which is assigned
# 10 observations.  Each of these observations has 3 detectors and 100
# samples.  So the Data object contains a total of 40 observations and
# 12000 samples.

