# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Environment, Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance

from ..healpix import HealpixPixels

from ..timing import function_timer

from .. import qarray as qa

from ..pixels import PixelDistribution

from ..observation import default_names as obs_names

from .._libtoast import healpix_pixels

from .operator import Operator

from .delete import Delete


@trait_docs
class PixelsHealpix(Operator):
    """Operator which generates healpix pixel numbers.

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    nside = Int(64, help="The NSIDE resolution")

    nside_submap = Int(16, help="The NSIDE of the submap resolution")

    nest = Bool(False, help="If True, use NESTED ordering instead of RING")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode(
        obs_names.pixels, help="Observation detdata key for output pixel indices"
    )

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output quaternions",
    )

    create_dist = Unicode(
        None,
        allow_none=True,
        help="Create the submap distribution for all detectors and store in the "
        "Data key specified",
    )

    single_precision = Bool(False, help="If True, use 32bit int in output")

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("nside")
    def _check_nside(self, proposal):
        check = proposal["value"]
        if ~check & (check - 1) != check - 1:
            raise traitlets.TraitError("Invalid NSIDE value")
        if check < self.nside_submap:
            raise traitlets.TraitError("NSIDE value is less than nside_submap")
        return check

    @traitlets.validate("nside_submap")
    def _check_nside_submap(self, proposal):
        check = proposal["value"]
        if ~check & (check - 1) != check - 1:
            raise traitlets.TraitError("Invalid NSIDE submap value")
        if check > self.nside:
            newval = 16
            if newval > self.nside:
                newval = 1
            log = Logger.get()
            log.warning(
                "nside_submap greater than NSIDE.  Setting to {} instead".format(newval)
            )
            check = newval
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check that healpix pixels are set up.  If the nside is left as
        # default, then the 'observe' function will not have baen called yet.
        if not hasattr(self, "_local_submaps"):
            self._set_hpix(self.nside, self.nside_submap)

    @traitlets.observe("nside", "nside_submap")
    def _reset_hpix(self, change):
        # (Re-)initialize the healpix pixels object when one of these traits change.
        # Current values:
        nside = self.nside
        nside_submap = self.nside_submap
        # Update to the trait that changed
        if change["name"] == "nside":
            nside = change["new"]
        if change["name"] == "nside_submap":
            nside_submap = change["new"]
        self._set_hpix(nside, nside_submap)

    def _set_hpix(self, nside, nside_submap):
        self.hpix = HealpixPixels(nside)
        self._n_pix = 12 * nside ** 2
        self._n_pix_submap = 12 * nside_submap ** 2
        self._n_submap = (nside // nside_submap) ** 2
        self._local_submaps = None

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        if self._local_submaps is None and self.create_dist is not None:
            self._local_submaps = np.zeros(self._n_submap, dtype=np.bool)

        # Expand detector pointing
        if self.quats is not None:
            quats_name = self.quats
        else:
            if self.detector_pointing.quats is not None:
                quats_name = self.detector_pointing.quats
            else:
                quats_name = "quats"

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Expand detector pointing
        self.detector_pointing.quats = quats_name
        self.detector_pointing.apply(data, detectors=detectors)

        # We do the calculation over buffers of timestream samples to reduce memory
        # overhead from temporary arrays.
        tod_buffer_length = env.tod_buffer_length()

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Check that our view is fully covered by detector pointing.  If the
            # detector_pointing view is None, then it has all samples.  If our own
            # view was None, then it would have been set to the detector_pointing
            # view above.
            if (view is not None) and (self.detector_pointing.view is not None):
                if ob.intervals[view] != ob.intervals[self.detector_pointing.view]:
                    # We need to check intersection
                    intervals = ob.intervals[self.view]
                    detector_intervals = ob.intervals[self.detector_pointing.view]
                    intersection = detector_intervals & intervals
                    if intersection != intervals:
                        msg = (
                            f"view {self.view} is not fully covered by valid "
                            "detector pointing"
                        )
                        raise RuntimeError(msg)

            # Do we already have pointing for all requested detectors?
            if self.pixels in ob.detdata:
                pix_dets = ob.detdata[self.pixels].detectors
                for d in dets:
                    if d not in pix_dets:
                        break
                else:  # no break
                    # We already have pointing for all specified detectors
                    if self.create_dist is not None:
                        # but the caller wants the pixel distribution
                        for ob in data.obs:
                            views = ob.view[self.view]
                            for det in ob.select_local_detectors(detectors):
                                for view in range(len(views)):
                                    self._local_submaps[
                                        views.detdata[self.pixels][view][det]
                                        // self._n_pix_submap
                                    ] = True

                    if data.comm.group_rank == 0:
                        msg = (
                            f"Group {data.comm.group}, ob {ob.name}, pointing "
                            f"already computed for {dets}"
                        )
                        log.verbose(msg)
                    continue

            # Create (or re-use) output data for the pixels, weights and optionally the
            # detector quaternions.

            if self.single_precision:
                ob.detdata.ensure(
                    self.pixels, sample_shape=(), dtype=np.int32, detectors=dets
                )
            else:
                ob.detdata.ensure(
                    self.pixels, sample_shape=(), dtype=np.int64, detectors=dets
                )

            # Focalplane for this observation
            focalplane = ob.telescope.focalplane

            # Loop over views
            views = ob.view[view]
            for vw in range(len(views)):
                # Get the flags if needed.  Use the same flags as
                # detector pointing.
                flags = None
                if self.detector_pointing.shared_flags is not None:
                    flags = np.array(
                        views.shared[self.detector_pointing.shared_flags][vw]
                    )
                    flags &= self.detector_pointing.shared_flag_mask

                for det in dets:
                    # Timestream of detector quaternions
                    quats = views.detdata[quats_name][vw][det]
                    view_samples = len(quats)

                    # Buffered pointing calculation
                    buf_off = 0
                    buf_n = tod_buffer_length
                    while buf_off < view_samples:
                        if buf_off + buf_n > view_samples:
                            buf_n = view_samples - buf_off

                        bslice = slice(buf_off, buf_off + buf_n)

                        # This buffer of detector quaternions
                        detp = quats[bslice, :].reshape(-1)

                        # Buffer of flags
                        fslice = None
                        if flags is not None:
                            fslice = flags[bslice].reshape(-1)

                        # Pixel buffer
                        pxslice = views.detdata[self.pixels][vw][det, bslice].reshape(
                            -1
                        )

                        if self.single_precision:
                            pbuf = np.zeros(len(pxslice), dtype=np.int64)
                        else:
                            pbuf = pxslice

                        healpix_pixels(
                            self.hpix,
                            self.nest,
                            detp,
                            fslice,
                            pbuf,
                        )

                        if self.single_precision:
                            pxslice[:] = pbuf.astype(np.int32)

                        buf_off += buf_n

                    if self.create_dist is not None:
                        self._local_submaps[
                            views.detdata[self.pixels][vw][det] // self._n_pix_submap
                        ] = True

        return

    def _finalize(self, data, **kwargs):
        if self.create_dist is not None:
            submaps = None
            if self.single_precision:
                submaps = np.arange(self._n_submap, dtype=np.int32)[self._local_submaps]
            else:
                submaps = np.arange(self._n_submap, dtype=np.int64)[self._local_submaps]
            data[self.create_dist] = PixelDistribution(
                n_pix=self._n_pix,
                n_submap=self._n_submap,
                local_submaps=submaps,
                comm=data.comm.comm_world,
            )
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        if self.cal is not None:
            req["meta"].append(self.cal)
        if self.hwp_angle is not None:
            req["shared"].append(self.hwp_angle)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.pixels],
        }
        if self.create_dist is not None:
            prov["meta"].append(self.create_dist)
        if self.quats is not None:
            prov["detdata"].append(self.quats)
        return prov

    def _accelerators(self):
        return list()
