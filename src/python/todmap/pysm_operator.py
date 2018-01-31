# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from .. import timing as timing
from ..map import DistRings, PySMSky, LibSharpSmooth, DistPixels
from ..mpi import MPI
from ..op import Operator

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
        autotimer = timing.auto_timer(type(self).__name__)
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._out = out
        self._nest = nest
        self.comm = comm
        self._debug = debug
        self.dist_rings = DistRings(comm,
                                    nside=nside,
                                    nnz=3)

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
                (np.linspace(bandcenter - bandwidth / 2,
                             bandcenter + bandwidth / 2,
                             N_POINTS_BANDPASS),
                 np.ones(N_POINTS_BANDPASS))

        lmax = 3 * self.nside - 1

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
                  ''.format(stop - start), flush=True)
