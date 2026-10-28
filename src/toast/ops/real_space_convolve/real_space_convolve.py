# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np
from pshmem import MPIShared

from ...observation import default_values as defaults
from ...pixels_io_healpix import read_healpix
from ...timing import function_timer
from ...traits import Instance, Int, List, Unicode, Unit, trait_docs
from ..operator import Operator
from .beam import RealSpaceBeam
from .kernels import real_space_convolve


@trait_docs
class SimRealSpaceConvolve(Operator):
    """Operator which convolves a HEALPix map with a beam in real space.

    The beam is a set of nodes around the line of sight, each with a weight for I
    and a weight shared by Q and U (see `RealSpaceBeam`).  At each sample, the map
    is read at every node with bilinear interpolation, Q and U are parallel
    transported to the line of sight, and the weighted sums give the beam-convolved
    I, Q and U in the meridian basis of the line of sight.  These are contracted
    with the Stokes weights and added to the detector data.

    Because a single weight acts on both Q and U, it commutes with the rotation of
    the polarization basis, so the contraction with the Stokes weights is exact for
    any detector angle and half-wave plate.

    The beam weights are used as given, so a beam with unit gain has weights that
    sum to one.  The map must be in the units of the detector data, since no unit
    conversion is applied.  The full-sky map is loaded in RING ordering, in double
    precision, and placed in shared memory on every node.  Only the COSMO
    polarization convention is supported.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    file = Unicode(
        None,
        allow_none=True,
        help="Path to the HEALPix FITS map, with I or I, Q, U columns",
    )

    maps = List(
        None,
        allow_none=True,
        help="The RING map components (I or I, Q, U).  If set, `file` must be None.",
    )

    beam_file = Unicode(
        None, allow_none=True, help="Path to the beam file (see RealSpaceBeam)"
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for accumulating output"
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    stokes_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="Stokes weights operator, which also provides the detector pointing",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._beam = None
        self._maps = None
        self._shared = None

    def _load(self, data):
        self._beam = RealSpaceBeam.load(self.beam_file)
        if self.file is None:
            self._maps = np.atleast_2d(np.array(self.maps, dtype=np.float64))
            return
        world_comm = data.comm.comm_world
        if world_comm is None or world_comm.rank == 0:
            m = np.atleast_2d(read_healpix(self.file, None, dtype=np.float64))
            map_shape = m.shape
        else:
            m = None
            map_shape = None
        if world_comm is not None:
            map_shape = world_comm.bcast(map_shape)
        self._shared = MPIShared(map_shape, np.float64, world_comm)
        self._shared.set(m)
        self._maps = self._shared.data

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        for trait in ("beam_file", "stokes_weights"):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)
        if (self.file is None) == (self.maps is None or len(self.maps) == 0):
            msg = "You must set exactly one of the 'file' and 'maps' traits "
            msg += "before calling exec()"
            raise RuntimeError(msg)
        if self.stokes_weights.IAU:
            raise RuntimeError("The IAU polarization convention is not supported")

        # The beam and map are loaded once and kept until finalize, because a
        # pipeline may call exec once per detector.
        if self._maps is None:
            self._load(data)
        beam = self._beam
        maps = self._maps

        n_map = maps.shape[0]
        nnz = len(self.stokes_weights.mode)
        if n_map not in (1, 3) or n_map != nnz:
            msg = f"The map has {n_map} components and the Stokes weights have "
            msg += f"{nnz}, expected 1 and 1 (I) or 3 and 3 (IQU)"
            raise RuntimeError(msg)
        if n_map == 3 and beam.weights_P is None:
            raise RuntimeError("An IQU map needs a beam with Q and U weights")
        nside = hp.npix2nside(maps.shape[1])
        weights_P = beam.weights_P if n_map == 3 else np.zeros(0)

        pointing = self.stokes_weights.detector_pointing
        view = self.stokes_weights.view
        if view is None:
            view = pointing.view

        for ob in data.obs:
            # Create the output for all detectors at once: a pipeline may call exec
            # one detector at a time, and adding detectors later zeroes the data.
            ob.detdata.ensure(
                self.det_data,
                detectors=ob.select_local_detectors(flagmask=self.det_mask),
                create_units=self.det_data_units,
            )
            dets = ob.select_local_detectors(detectors, flagmask=self.det_mask)
            ob_data = data.select(obs_name=ob.name)
            intervals = ob.intervals[view].data
            for det in dets:
                # Expand the pointing one detector at a time to bound the memory.
                self.stokes_weights.apply(ob_data, detectors=[det])
                quats = ob.detdata[pointing.quats]
                weights = ob.detdata[self.stokes_weights.weights]
                det_data = ob.detdata[self.det_data]
                real_space_convolve(
                    quats.indices([det]),
                    quats.data,
                    weights.indices([det]),
                    weights.data,
                    det_data.indices([det]),
                    det_data.data,
                    intervals,
                    nside,
                    maps,
                    beam.quats,
                    beam.weights_I,
                    weights_P,
                    impl=self.kernel_implementation,
                )

    def _finalize(self, data, **kwargs):
        if self._shared is not None:
            self._shared.close()
        self._beam = None
        self._maps = None
        self._shared = None

    def _requires(self):
        req = self.stokes_weights.requires()
        req["detdata"].append(self.det_data)
        return req

    def _provides(self):
        return {"detdata": [self.det_data]}
