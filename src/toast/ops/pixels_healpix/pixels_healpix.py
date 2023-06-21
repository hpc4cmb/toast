# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets

from ...accelerator import ImplementationType
from ...healpix import Pixels
from ...observation import default_values as defaults
from ...pixels import PixelDistribution
from ...timing import function_timer
from ...traits import Bool, Instance, Int, Unicode, UseEnum, trait_docs
from ...utils import Environment, Logger
from ..operator import Operator
from .kernels import pixels_healpix


@trait_docs
class PixelsHealpix(Operator):
    """Operator which generates healpix pixel numbers.

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels.

    Any samples with "bad" pointing should have already been set to a "safe" quaternion
    value by the detector pointing operator.

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

    nest = Bool(True, help="If True, use NESTED ordering instead of RING")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode(
        defaults.pixels, help="Observation detdata key for output pixel indices"
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
        self.hpix = Pixels(nside)
        self._n_pix = 12 * nside**2
        self._n_pix_submap = 12 * nside_submap**2
        self._n_submap = (nside // nside_submap) ** 2
        self._local_submaps = None

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        if self._local_submaps is None and self.create_dist is not None:
            self._local_submaps = np.zeros(self._n_submap, dtype=np.uint8)

        # Expand detector pointing
        quats_name = self.detector_pointing.quats

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Expand detector pointing
        self.detector_pointing.apply(data, detectors=detectors, use_accel=use_accel)

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

            # Create (or re-use) output data for the pixels.
            if self.single_precision:
                exists = ob.detdata.ensure(
                    self.pixels,
                    sample_shape=(),
                    dtype=np.int32,
                    detectors=dets,
                    accel=use_accel,
                )
            else:
                exists = ob.detdata.ensure(
                    self.pixels,
                    sample_shape=(),
                    dtype=np.int64,
                    detectors=dets,
                    accel=use_accel,
                )

            hit_submaps = self._local_submaps
            if hit_submaps is None:
                hit_submaps = np.zeros(self._n_submap, dtype=np.uint8)

            quat_indx = ob.detdata[quats_name].indices(dets)
            pix_indx = ob.detdata[self.pixels].indices(dets)

            view_slices = [slice(x.first, x.last + 1, 1) for x in ob.intervals[view]]

            # Do we already have pointing for all requested detectors?
            if exists:
                # Yes...
                if self.create_dist is not None:
                    # but the caller wants the pixel distribution
                    restore_dev = False
                    if ob.detdata[self.pixels].accel_in_use():
                        # The data is on the accelerator- copy back to host for
                        # this calculation.  This could eventually be a kernel.
                        ob.detdata[self.pixels].accel_update_host()
                        restore_dev = True
                    for det in ob.select_local_detectors(detectors):
                        for vslice in view_slices:
                            good = ob.detdata[self.pixels][det, vslice] >= 0
                            self._local_submaps[
                                ob.detdata[self.pixels][det, vslice][good]
                                // self._n_pix_submap
                            ] = 1
                    if restore_dev:
                        ob.detdata[self.pixels].accel_update_device()

                if data.comm.group_rank == 0:
                    msg = (
                        f"Group {data.comm.group}, ob {ob.name}, healpix pixels "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            # Get the flags if needed.  Use the same flags as
            # detector pointing.
            if self.detector_pointing.shared_flags is None:
                flags = np.zeros(1, dtype=np.uint8)
            else:
                flags = ob.shared[self.detector_pointing.shared_flags].data

            pixels_healpix(
                quat_indx,
                ob.detdata[quats_name].data,
                flags,
                self.detector_pointing.shared_flag_mask,
                pix_indx,
                ob.detdata[self.pixels].data,
                ob.intervals[self.view].data,
                hit_submaps,
                self._n_pix_submap,
                self.nside,
                self.nest,
                impl=implementation,
                use_accel=use_accel,
            )

            if self._local_submaps is not None:
                self._local_submaps[:] |= hit_submaps

        return

    def _finalize(self, data, use_accel=None, **kwargs):
        if self.create_dist is not None:
            submaps = None
            if self.single_precision:
                submaps = np.arange(self._n_submap, dtype=np.int32)[
                    self._local_submaps == 1
                ]
            else:
                submaps = np.arange(self._n_submap, dtype=np.int64)[
                    self._local_submaps == 1
                ]

            data[self.create_dist] = PixelDistribution(
                n_pix=self._n_pix,
                n_submap=self._n_submap,
                local_submaps=submaps,
                comm=data.comm.comm_world,
            )
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        if "detdata" not in req:
            req["detdata"] = list()
        req["detdata"].append(self.pixels)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.pixels)
        if self.create_dist is not None:
            prov["global"].append(self.create_dist)
        return prov

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        if (self.detector_pointing is not None) and (
            self.detector_pointing.supports_accel()
        ):
            return True
        else:
            return False
