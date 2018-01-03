# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import healpy as hp

from .. import qarray as qa
from .. import timing as timing
from .tod import TOD
from ..op import Operator
from ..map import DistRings, PySMSky, LibSharpSmooth, DistPixels
from ..mpi import MPI
from ..ctoast import sim_map_scan_map


class OpSimGradient(Operator):
    """
    Generate a fake sky signal as a gradient between the poles.

    This passes through each observation and creates a fake signal timestream
    based on the cartesian Z coordinate of the HEALPix pixel containing the
    detector pointing.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        nside (int): the HEALPix NSIDE value to use.
        min (float): the minimum value to use at the South Pole.
        max (float): the maximum value to use at the North Pole.
        nest (bool): whether to use NESTED ordering.
    """

    def __init__(self, out='grad', nside=512, min=-100.0, max=100.0, nest=False,
                 flag_mask=255, common_flag_mask=255, keep_quats=False):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._out = out
        self._min = min
        self._max = max
        self._nest = nest
        self._flag_mask = flag_mask
        self._common_flag_mask = common_flag_mask
        self._keep_quats = keep_quats

    def exec(self, data):
        """
        Create the gradient timestreams.

        This pixelizes each detector's pointing and then assigns a
        timestream value based on the cartesian Z coordinate of the pixel
        center.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        comm = data.comm

        zaxis = np.array([0, 0, 1], dtype=np.float64)
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        range = self._max - self._min

        for obs in data.obs:
            tod = obs['tod']
            base = obs['baselines']
            nse = obs['noise']

            offset, nsamp = tod.local_samples

            common = tod.local_common_flags() & self._common_flag_mask

            for det in tod.local_dets:
                flags = tod.local_flags(det) & self._flag_mask
                totflags = (flags | common)
                del flags

                pdata = tod.local_pointing(det).copy()
                pdata[totflags != 0, :] = nullquat

                dir = qa.rotate(pdata, zaxis)
                pixels = hp.vec2pix(self._nside, dir[:, 0], dir[:, 1], dir[:, 2],
                                    nest=self._nest)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=self._nest)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                z[totflags != 0] = 0.0

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (nsamp, ))
                ref = tod.cache.reference(cachename)
                ref[:] += z
                del ref

                if not self._keep_quats:
                    cachename = 'quat_{}'.format(det)
                    tod.cache.destroy(cachename)

            del common
        return

    def sigmap(self):
        """
        (array): Return the underlying signal map (full map on all processes).
        """
        autotimer = timing.auto_timer(type(self).__name__)
        range = self._max - self._min
        pix = np.arange(0, 12*self._nside*self._nside, dtype=np.int64)
        x, y, z = hp.pix2vec(self._nside, pix, nest=self._nest)
        z += 1.0
        z *= 0.5
        z *= range
        z += self._min
        return z


class OpSimScan(Operator):
    """
    Operator which generates sky signal by scanning from a map.

    The signal to use should already be in a distributed pixel structure,
    and local pointing should already exist.

    Args:
        distmap (DistPixels): the distributed map domain data.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
    """
    def __init__(self, distmap=None, pixels='pixels', weights='weights',
                 out='scan', dets=None):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._map = distmap
        self._pixels = pixels
        self._weights = weights
        self._out = out
        self._dets = dets

    def exec(self, data):
        """
        Create the timestreams by scanning from the map.

        This loops over all observations and detectors and uses the pointing
        matrix to project the distributed map into a timestream.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        for obs in data.obs:
            tod = obs['tod']

            dets = tod.local_dets if self._dets is None else self._dets

            for det in dets:

                # get the pixels and weights from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                weightsname = "{}_{}".format(self._weights, det)
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                nsamp, nnz = weights.shape

                sm, lpix = self._map.global_to_local(pixels)

                #f = (np.dot(weights[x], self._map.data[sm[x], lpix[x]])
                #     if (lpix[x] >= 0) else 0
                #     for x in range(tod.local_samples[1]))
                #maptod = np.fromiter(f, np.float64, count=tod.local_samples[1])
                maptod = np.zeros(nsamp)
                sim_map_scan_map(sm, weights, lpix, self._map.data, maptod)

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (nsamp, ))
                ref = tod.cache.reference(cachename)
                ref[:] += maptod

                del ref
                del pixels
                del weights

        return


def extract_local_dets(data):
    """Extracts the local detectors from the TOD objects

    Some detectors could only appear in some observations, so we need
    to loop through all observations and accumulate all detectors in
    a set
    """
    autotimer = timing.auto_timer()
    local_dets = set()
    for obs in data.obs:
        tod = obs['tod']
        local_dets.update(tod.local_dets)
    return local_dets


def assemble_map_on_rank0(comm, local_map, pixel_indices, n_components, npix):
    autotimer = timing.auto_timer()
    full_maps_rank0 = np.zeros((n_components, npix),
                               dtype=np.float64) if comm.rank == 0 else None
    local_map_buffer = np.zeros((n_components, npix),
                                   dtype=np.float64)
    local_map_buffer[:,pixel_indices] = local_map
    comm.Reduce(local_map_buffer, full_maps_rank0, root=0, op=MPI.SUM)
    return full_maps_rank0


def extract_detector_parameters(det, focalplanes):
    autotimer = timing.auto_timer()
    for fp in focalplanes:
        if det in fp:
            if "fwhm" in fp[det]:
                return fp[det]["bandcenter_ghz"], fp[det]["bandwidth_ghz"], \
                    fp[det]["fwhm"]
            else:
                return fp[det]["bandcenter_ghz"], fp[det]["bandwidth_ghz"], -1
    raise RuntimeError("Cannot find detector {} in any focalplane")


class OpSimPySM(Operator):
    """
    Operator which generates sky signal by scanning from a map.

    The signal to use should already be in a distributed pixel structure,
    and local pointing should already exist.

    Args:
        distmap (DistPixels): the distributed map domain data.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        units(str): Output units.
        debug(bool):  Verbose progress reports.
    """
    def __init__(self, comm=None,
                 out='signal', pysm_model='', focalplanes=None, nside=None,
                 subnpix=None, localsm=None, apply_beam=False, nest=True,
                 units='K_CMB', debug=False):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._out = out
        self._nest = nest
        self.comm = comm
        self._debug = debug
        self.dist_rings = DistRings(comm,
                            nside = nside,
                            nnz = 3)

        pysm_sky_components = [
            'synchrotron',
            'dust',
            'freefree',
            'cmb',
            'ame',
        ]
        pysm_sky_config = dict()
        for component_model in pysm_model.split(','):
            full_component_name = [
                each for each in pysm_sky_components
                if each.startswith(component_model[0])][0]
            pysm_sky_config[full_component_name] = component_model
        self.pysm_sky = PySMSky(comm=self.comm,
                                local_pixels=self.dist_rings.local_pixels,
                                nside=nside, pysm_sky_config=pysm_sky_config,
                                units=units)

        self.nside = nside
        self.focalplanes = focalplanes
        self.npix = hp.nside2npix(nside)
        self.distmap = DistPixels(
            comm=comm, size=self.npix, nnz=3,
            dtype=np.float32, submap=subnpix, local=localsm)
        self.apply_beam = apply_beam

    def __del__(self):
        # Ensure that the PySMSky member is destroyed first because
        # it contains a reference to self.dist_rings.local_pixels
        del self.pysm_sky
        del self.dist_rings
        del self.distmap

    def exec(self, data):
        autotimer = timing.auto_timer(type(self).__name__)
        local_dets = extract_local_dets(data)

        bandpasses = {}
        fwhm_deg = {}
        N_POINTS_BANDPASS = 10  # possibly take as parameter
        for det in local_dets:
            bandcenter, bandwidth, fwhm_deg[det] = \
                    extract_detector_parameters(det, self.focalplanes)
            bandpasses[det] = \
                (np.linspace(bandcenter-bandwidth/2,
                             bandcenter+bandwidth/2,
                             N_POINTS_BANDPASS),
                 np.ones(N_POINTS_BANDPASS))

        lmax = 3*self.nside -1

        if self.comm.rank == 0:
            print('Collecting, Broadcasting map', flush=True)
        start = MPI.Wtime()
        local_maps = dict()  # FIXME use Cache instead
        for det in local_dets:
            self.comm.Barrier()
            if self.comm.rank == 0 and self._debug:
                print('Running PySM on {}'.format(det), flush=True)
            self.pysm_sky.exec(local_maps, out="sky",
                               bandpasses={"": bandpasses[det]})

            if self.apply_beam:
                if fwhm_deg[det] == -1:
                    raise RuntimeError(
                        "OpSimPySM: apply beam is True but focalplane doesn't "
                        "have fwhm")
                # LibSharp also supports transforming multiple channels
                # together each with own beam
                self.comm.Barrier()
                if self.comm.rank == 0 and self._debug:
                    print('Initializing LibSharpSmooth on {}'.format(det),
                          flush=True)
                smooth = LibSharpSmooth(
                    self.comm, signal_map="sky", out="sky",
                    lmax=lmax, grid=self.dist_rings.libsharp_grid,
                    fwhm_deg=fwhm_deg[det], beam=None)
                self.comm.Barrier()
                if self.comm.rank == 0 and self._debug:
                    print('Executing LibSharpSmooth on {}'.format(det),
                          flush=True)
                smooth.exec(local_maps)
                self.comm.Barrier()
                if self.comm.rank == 0 and self._debug:
                    print('LibSharpSmooth completed on {}'.format(det),
                          flush=True)

            n_components = 3

            self.comm.Barrier()
            if self.comm.rank == 0 and self._debug:
                print('Assemble PySM map on rank0, shape of local map is {}'
                      ''.format(local_maps["sky"].shape), flush=True)
            full_map_rank0 = assemble_map_on_rank0(
                self.comm, local_maps["sky"], self.dist_rings.local_pixels,
                n_components, self.npix)

            self.comm.Barrier()
            if self.comm.rank == 0 and self._debug:
                print('Communication completed', flush=True)
            if self.comm.rank == 0 and self._nest:
                # PySM is RING, toast is NEST
                full_map_rank0 = hp.reorder(full_map_rank0, r2n=True)
            # full_map_rank0 dict contains on rank 0 the smoothed PySM map

            self.comm.Barrier()
            if self.comm.rank == 0 and self._debug:
                print('Broadcasting the map to other processes', flush=True)
            self.distmap.broadcast_healpix_map(full_map_rank0)
            self.comm.Barrier()
            if self.comm.rank == 0 and self._debug:
                print('Running OpSimScan', flush=True)
            scansim = OpSimScan(distmap=self.distmap, out=self._out, dets=[det])
            scansim.exec(data)

        stop = MPI.Wtime()
        if self.comm.rank == 0:
            print('PySM Operator completed:  {:.2f} seconds'
                  ''.format(stop-start), flush=True)
