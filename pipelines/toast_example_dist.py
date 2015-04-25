
import toast

# Split COMM_WORLD into groups of 4 processes each
cm = toast.Comm(world=MPI.COMM_WORLD, groupsize=4)

# Create the distributed data object
dd = toast.Dist(comm=cm)

# Each process group appends some observations.
# For this example, each observation is going to have the same
# number of samples, and the same list of detectors.  We just
# use the base Streams and Pointing classes, which contains
# the data directly as numpy arrays.

obs_samples = 100
obs_dets = ['detA', 'detB', 'detC']

for i in range(10):
    # Streams object, with just the default flavor and the
    # default time slicing
    strms = toast.Streams(mpicomm=cm.comm_group, detectors=obs_dets, samples=obs_samples)

    # Pointing object
    pntg = toast.Pointing(mpicomm=cm.comm_group, detectors=obs_dets, samples=obs_samples)

    # Observation
    obs = toast.Obs(mpicomm=cm.comm_group, streams=strms, pointing=pntg)

    dd.obs.append(obs)

# Now at the end we have 4 process groups, each of which is assigned
# 10 observations.  Each of these observations has 3 detectors and 100
# samples.  So the Dist object contains a total of 40 observations and
# 12000 samples.

