# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import healpy as hp

import pysm
import pysm.units as u

from ..mpi import MPI

from ..timing import function_timer

from ..utils import Logger, Timer

from ..op import Operator

from ..map import DistRings, PySMSky, DistPixels

from .sim_det_map import OpSimScan


def extract_local_dets(data):
    """Extracts the local detectors from the TOD objects

    Some detectors could only appear in some observations, so we need
    to loop through all observations and accumulate all detectors in
    a set
    """
    local_dets = set()
    for obs in data.obs:
        tod = obs["tod"]
        local_dets.update(tod.local_dets)
    return local_dets


@function_timer
def assemble_map_on_rank0(comm, local_map, pixel_indices, n_components, npix):
    full_maps_rank0 = (
        np.zeros((n_components, npix), dtype=np.float64) if comm.rank == 0 else None
    )
    local_map_buffer = np.zeros((n_components, npix), dtype=np.float64)
    local_map_buffer[:, pixel_indices] = local_map
    comm.Reduce(local_map_buffer, full_maps_rank0, root=0, op=MPI.SUM)
    return full_maps_rank0


@function_timer
def extract_detector_parameters(det, focalplanes):
    for fp in focalplanes:
        if det in fp:
            if "fwhm" in fp[det]:
                return (
                    fp[det]["bandcenter_ghz"],
                    fp[det]["bandwidth_ghz"],
                    fp[det]["fwhm"] / 60,
                )
            else:
                return fp[det]["bandcenter_ghz"], fp[det]["bandwidth_ghz"], -1
    raise RuntimeError("Cannot find detector {} in any focalplane")


class OpSimPySM(Operator):
    """Operator which generates sky signal by scanning from a map.

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

    @function_timer
    def __init__(
        self,
        comm=None,
        out="signal",
        pysm_model="",
        pysm_precomputed_cmb_K_CMB=None,
        focalplanes=None,
        nside=None,
        subnpix=None,
        localsm=None,
        apply_beam=False,
        nest=True,
        units="K_CMB",
        debug=False,
        coord="G",
    ):
        # Call the parent class constructor.
        super().__init__()
        self._out = out
        self._nest = nest
        self.comm = comm
        self._debug = debug
        self.pysm_precomputed_cmb_K_CMB = pysm_precomputed_cmb_K_CMB
        self.dist_rings = DistRings(comm, nside=nside, nnz=3)
        self.coord = coord

        pysm_sky_components = ["synchrotron", "dust", "freefree", "cmb", "ame"]
        pysm_sky_config = dict()
        for component_model in pysm_model.split(","):
            full_component_name = [
                each
                for each in pysm_sky_components
                if each.startswith(component_model[0])
            ][0]
            pysm_sky_config[full_component_name] = component_model
        self.pysm_sky = PySMSky(
            comm=self.comm,
            pixel_indices=None,
            nside=nside,
            pysm_sky_config=pysm_sky_config,
            pysm_precomputed_cmb_K_CMB=self.pysm_precomputed_cmb_K_CMB,
            units=units,
        )

        self.nside = nside
        self.focalplanes = focalplanes
        self.npix = hp.nside2npix(nside)
        self.distmap = DistPixels(
            comm=comm,
            size=self.npix,
            nnz=3,
            dtype=np.float32,
            submap=subnpix,
            local=localsm,
        )
        self.apply_beam = apply_beam

    def __del__(self):
        # Ensure that the PySMSky member is destroyed first because
        # it contains a reference to self.dist_rings.local_pixels
        del self.pysm_sky
        del self.dist_rings
        del self.distmap

    @function_timer
    def exec(self, data):
        log = Logger.get()
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        local_dets = extract_local_dets(data)
        bandpasses = {}
        fwhm_deg = {}
        N_POINTS_BANDPASS = 10  # possibly take as parameter
        for det in local_dets:
            bandcenter, bandwidth, fwhm_deg[det] = extract_detector_parameters(
                det, self.focalplanes
            )
            bandpasses[det] = (
                np.linspace(
                    bandcenter - bandwidth / 2,
                    bandcenter + bandwidth / 2,
                    N_POINTS_BANDPASS,
                ),
                np.ones(N_POINTS_BANDPASS),
            )

        lmax = 3 * self.nside - 1

        if rank == 0:
            log.debug("Collecting, Broadcasting map")
        tm = Timer()
        tm.start()

        for det in local_dets:
            # FIXME: this used to be outside this loop, but there is no need for that.
            # Actually I don't think we even need a dictionary, just the map that is
            # output from PySMSky which we can then feed to libsharp smoothing.
            local_maps = dict()
            if self.comm is not None:
                self.comm.Barrier()
            if rank == 0:
                log.debug("Running PySM on {}".format(det))
            self.pysm_sky.exec(local_maps, out="sky", bandpasses={det: bandpasses[det]})

            if self.apply_beam:
                if fwhm_deg[det] == -1:
                    raise RuntimeError(
                        "OpSimPySM: apply beam is True but focalplane doesn't "
                        "have fwhm"
                    )
                # LibSharp also supports transforming multiple channels
                # together each with own beam
                if self.comm is not None:
                    self.comm.Barrier()
                if rank == 0:
                    log.debug("Executing Smoothing with libsharp on {}".format(det))
                local_maps["sky_{}".format(det)] = pysm.apply_smoothing_and_coord_transform(
                    local_maps["sky_{}".format(det)], fwhm=fwhm_deg[det]*u.deg, map_dist=self.pysm_sky.sky.map_dist
                )
                if self.comm is not None:
                    self.comm.Barrier()
                if rank == 0:
                    log.debug("Smoothing completed on {}".format(det))

            n_components = 3

            if self.comm is not None:
                self.comm.Barrier()
            if rank == 0:
                log.debug(
                    "Assemble PySM map on rank0, shape of local map is {}".format(
                        local_maps["sky_{}".format(det)].shape
                    )
                )
            full_map_rank0 = assemble_map_on_rank0(
                self.comm,
                local_maps["sky_{}".format(det)],
                np.arange(len(local_maps["sky_{}".format(det)])) if self.comm is None else self.pysm_sky.sky.map_dist.pixel_indices,
                n_components,
                self.npix,
            )

            if self.comm is not None:
                self.comm.Barrier()
            if rank == 0:
                log.debug("Communication completed")
            if rank == 0 and self.coord != "G":
                # PySM is always in Galactic, make rotation to Ecliptic or Equatorial
                rot = hp.Rotator(coord=["G", self.coord])
                # this requires healpy 1.12.8
                try:
                    full_map_rank0 = rot.rotate_map_alms(
                        full_map_rank0, use_pixel_weights=True
                    )
                except AttributeError:
                    print(
                        "PySM coordinate conversion from G to another reference frame requires"
                        "healpy.Rotator.rotate_map_alms available since healpy 1.12.8"
                    )
                    raise
            if rank == 0 and self._nest:
                # PySM is RING, convert to NEST if desired.
                full_map_rank0 = hp.reorder(full_map_rank0, r2n=True)
            # full_map_rank0 dict contains on rank 0 the smoothed PySM map

            if self.comm is not None:
                self.comm.Barrier()
            if rank == 0:
                log.debug(
                    "PySM map min / max pixel value = {} / {}".format(
                        hp.ma(full_map_rank0).min(), hp.ma(full_map_rank0).max()
                    )
                )
                log.debug("Broadcasting the map to other processes")
            self.distmap.broadcast_healpix_map(full_map_rank0)
            if rank == 0:
                log.debug("Running OpSimScan")
            scansim = OpSimScan(distmap=self.distmap, out=self._out, dets=[det])
            scansim.exec(data)
            if rank == 0:
                tod = data.obs[0]["tod"]
                sig = tod.cache.reference(self._out + "_" + det)
                log.debug(
                    "Rank 0 timeline min / max after smoothing = {} / {}".format(
                        sig.min(), sig.max()
                    )
                )

        tm.stop()
        if rank == 0:
            tm.report("PySM Operator")

        return
