#!/usr/bin/env python

# Always import this first, to make sure MPI is initialized as soon
# as possible.
import toast

import os
import sys
import traceback

import numpy as np

from toast.mpi import MPI
import toast.qarray as qa

import toast.tod as tt
import toast.map as tm

from toast.vis import set_backend


def fake_focalplane():
    """
    Make a fake focalplane geometry.

    This function returns a fake hexagonal focalplane with 19 pixels
    and 2 detectors at each position.  The spacing is set to fill the
    field of view.
    """
    npix = 19
    # 5' beam
    fwhm = 5.0 / 60.0
    # 5 degree FOV...
    fov = 5.0
    # ...converted to size of hexagon
    angwidth = fov * np.cos(30.0 * np.pi / 180.0)
    
    # Alternating polarization orientations
    Apol = tt.hex_pol_angles_qu(npix, offset=0.0)
    Bpol = tt.hex_pol_angles_qu(npix, offset=90.0)
    
    # Build a simple hexagon layout
    Adets = tt.hex_layout(npix, 100.0, angwidth, fwhm, "fake_", "A", Apol)
    Bdets = tt.hex_layout(npix, 100.0, angwidth, fwhm, "fake_", "B", Bpol)
    
    # Combine into a single dictionary
    dets = Adets.copy()
    dets.update(Bdets)

    # Give each detector the same fake noise properties
    for indx, d in enumerate(sorted(dets.keys())):
        # 50mHz knee frequency
        dets[d]["fknee"] = 0.050
        # High pass "plateau" to avoid blow-up at very low f
        dets[d]["fmin"] = 1.0e-5
        # Exponent
        dets[d]["alpha"] = 1.0
        # Sensitivity
        dets[d]["NET"] = 20.0e-6
        # Unique index for reproducibility of simulations
        dets[d]["index"] = indx

    return dets


class TextTOD(toast.tod.TOD):
    """
    Read boresight quaternions from a text file.

    This is just a demonstration.  Please do not actually store timestream data
    in a text file...  This reads the boresight data from a text file, and gets
    the focalplane layout from the constructor as a dictionary.

    This class does not read or write any detector data- it is only useful for
    providing pointing data for simulations.

    Args:
        path (str): the path to the text file containing 4 columns.
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is
            distributed.
        detectors (dictionary): each key is the detector name, and each value
            is a quaternion tuple.
        samples (int):  The total number of samples.
        rate (float): sample rate in Hz.
        detindx (dict): the detector indices for use in simulations.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.
        meta (dict): Some extra metadata
    """
    def __init__(self, path, mpicomm, detectors, rate=100.0, detranks=1, 
        meta=None):
        self._path = path
        self._fp = detectors
        self._detlist = sorted(list(self._fp.keys()))
        self._firsttime = 0.0
        self._rate = rate
        # Now go read the file and broadcast the boresight pointing to all
        # processes.
        self._boresight = None
        if mpicomm.rank == 0:
            self._boresight = np.loadtxt(self._path, 
                dtype=np.float64).reshape((-1,4))
        self._boresight = mpicomm.bcast(self._boresight, root=0)
        detindx = {}
        for d in detectors.keys():
            detindx[d] = detectors[d]["index"]
        # Call base class constructor, which computes the data distribution.
        super().__init__(mpicomm, self._detlist, self._boresight.shape[0],
            detindx=detindx, detranks=detranks, meta=meta)
        
    def detoffset(self):
        """
        Return the quaternion offsets for all detectors.
        """
        return { d : np.asarray(self._fp[d]) for d in self._detlist }

    # The rest of the overrides for these TOD base class methods mostly
    # just return zeros or throw exceptions.  For a TOD class that reads
    # real data, these would actually return something meaningful.

    def _get(self, detector, start, n):
        return np.zeros(n, dtype=np.float64)

    def _put(self, detector, start, data, flags):
        raise RuntimeError("cannot write data to simulated data streams")
        return

    def _get_flags(self, detector, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_det_flags(self, detector, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_common_flags(self, start, n):
        return np.zeros(n, dtype=np.uint8)

    def _put_common_flags(self, start, flags):
        raise RuntimeError("cannot write flags to simulated data streams")
        return

    def _get_times(self, start, n):
        start_abs = self.local_samples[0] + start
        start_time = self._firsttime + float(start_abs) / self._rate
        stop_time = start_time + float(n) / self._rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False, dtype=np.float64)
        return stamps

    def _put_times(self, start, stamps):
        raise RuntimeError("cannot write timestamps to simulated data streams")
        return

    def _get_boresight(self, start, n):
        return self._boresight[start:start+n,:]

    def _put_boresight(self, start, data):
        raise RuntimeError("cannot write boresight to simulated data streams")
        return

    def _get_pntg(self, detector, start, n):
        detquat = np.asarray(self._fp[detector]["quat"])
        boresight = self._boresight[start:start+n,:]
        data = qa.mult(boresight, detquat)
        return data

    def _put_pntg(self, detector, start, data):
        raise RuntimeError("cannot write data to simulated pointing")
        return

    def _get_position(self, start, n):
        return np.zeros((n,3), dtype=np.float64)

    def _put_position(self, start, pos):
        raise RuntimeError("cannot write data to simulated position")
        return

    def _get_velocity(self, start, n):
        return np.zeros((n,3), dtype=np.float64)

    def _put_velocity(self, start, vel):
        raise RuntimeError("cannot write data to simulated velocity")
        return


def create_observations(comm, rate, fp, borefiles):
    """
    Helper function to create the distributed data
    """
    # The distributed timestream data.  This is a container for holding our
    # observations.
    data = toast.Data(comm)
    
    # Every process group creates its observations.  In this case, we have
    # only one process group.

    nobs = len(borefiles)

    obindx = 0
    for bf in borefiles:
        # Some contrived metadata
        meta = {
            "boresightfile" : bf
        }
        # The TOD class
        tod = TextTOD(bf, comm.comm_group, fp, rate=rate, meta=meta)
        
        # Let's add a noise model too, so we can do simulations
        # Create the noise model used for all observations
        fmin = {}
        fknee = {}
        alpha = {}
        NET = {}
        rates = {}
        for d in fp.keys():
            rates[d] = tod._rate
            fmin[d] = fp[d]["fmin"]
            fknee[d] = fp[d]["fknee"]
            alpha[d] = fp[d]["alpha"]
            NET[d] = fp[d]["NET"]

        noise = tt.AnalyticNoise(rate=rates, fmin=fmin, 
            detectors=list(fp.keys()), fknee=fknee, alpha=alpha, NET=NET)

        obs = {
            "name": "observation_{:05d}".format(obindx),
            "tod": tod,
            "noise": noise,
            "id": obindx,
        }
        data.obs.append(obs)
        obindx += 1

    return data


def pixel_dist(data, submap):
    """
    Compute the locally hit submaps for distributed pixel objects.
    """
    # get locally hit pixels
    loc = tm.OpLocalPixels(pixels="pixels")
    localpix = loc.exec(data)
    # find the locally hit submaps.
    localsm = np.unique(np.floor_divide(localpix, submap))
    return localsm


def main():
    # We are going to group our processes in a single group.  This is fine
    # if we have fewer processes than detectors.  Otherwise we should group
    # them in a reasonable size that is smaller than the number of detectors
    # and which divides evenly into the total number of processes.

    comm = toast.Comm(world=MPI.COMM_WORLD, groupsize=MPI.COMM_WORLD.size)

    # Make a fake focalplane.  Plot it just for fun (don't waste time on this
    # for very large runs though).
    fp = fake_focalplane()
    if comm.comm_world.rank == 0:
        outfile = "custom_example_focalplane.png"
        set_backend()
        tt.plot_focalplane(fp, 6.0, 6.0, outfile)

    # Read in 2 boresight files
    borefiles = [
        "../data/custom_example_boresight_1.txt",
        "../data/custom_example_boresight_2.txt"
    ]

    # Set up the distributed data
    rate = 100.0
    data = create_observations(comm, rate, fp, borefiles)

    # Configure the healpix pixelization we will use for map-making and
    # also the "submap" resolution, which sets granularity of the locally
    # stored pieces of the sky.
    map_nside = 512
    map_npix = 12 * map_nside**2
    sub_nside = 4
    sub_npix = 12 * sub_nside**2

    # Compute a pointing matrix with healpix pixels and weights.
    pointing = tt.OpPointingHpix(nside=map_nside, nest=True, mode="IQU",
        pixels="pixels", weights="weights")
    pointing.exec(data)

    # Compute the locally hit submaps
    local_submaps = pixel_dist(data, sub_npix)

    # Sources of simulated data:  scan from a symmetric beam convolved sky
    # and then add some simulated noise.

    signalmap = tm.DistPixels(comm=comm.comm_world, size=map_npix, nnz=3, 
        dtype=np.float64, submap=sub_npix, local=local_submaps)
    signalmap.read_healpix_fits("../data/custom_example_sky.fits")

    scanmap = tt.OpSimScan(distmap=signalmap, pixels='pixels', 
        weights='weights', out="sim")
    scanmap.exec(data)

    nse = tt.OpSimNoise(out="sim", realization=0)
    nse.exec(data)

    # Accumulate the hits and inverse diagonal pixel covariance, as well as the
    # noise weighted map.  Here we simply use inverse noise weighting.

    detweights = {}
    for d in fp.keys():
        net = fp[d]["NET"]
        detweights[d] = 1.0 / (rate * net * net)
        
    invnpp = tm.DistPixels(comm=comm.comm_world, size=map_npix, nnz=6, 
        dtype=np.float64, submap=sub_npix, local=local_submaps)

    hits = tm.DistPixels(comm=comm.comm_world, size=map_npix, nnz=1, 
        dtype=np.int64, submap=sub_npix, local=local_submaps)
    
    zmap = tm.DistPixels(comm=comm.comm_world, size=map_npix, nnz=3, 
        dtype=np.float64, submap=sub_npix, local=local_submaps)

    invnpp.data.fill(0.0)
    hits.data.fill(0)
    zmap.data.fill(0.0)

    build_invnpp = tm.OpAccumDiag(detweights=detweights, invnpp=invnpp, 
        hits=hits, zmap=zmap, name="sim")
    build_invnpp.exec(data)

    invnpp.allreduce()
    hits.allreduce()
    zmap.allreduce()

    # Write these products out

    hits.write_healpix_fits("custom_example_hits.fits")
    invnpp.write_healpix_fits("custom_example_invnpp.fits")
    zmap.write_healpix_fits("custom_example_zmap.fits")

    # Invert the covariance and write
    
    tm.covariance_invert(invnpp, 1.0e-3)
    invnpp.write_healpix_fits("custom_example_npp.fits")

    # Apply covariance to make the binned map

    tm.covariance_apply(invnpp, zmap)
    zmap.write_healpix_fits("custom_example_binned.fits")

    MPI.Finalize()
    return



if __name__ == "__main__":
    try:
        main()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for ln in lines:
            print(ln, flush=True)
        MPI.COMM_WORLD.Abort()
