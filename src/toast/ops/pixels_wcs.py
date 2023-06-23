# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import warnings

import numpy as np
import traitlets
from astropy import units as u
from astropy.wcs import WCS

from .. import qarray as qa
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelDistribution
from ..pointing_utils import scan_range_lonlat
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Tuple, Unicode, trait_docs
from ..utils import Environment, Logger
from .delete import Delete
from .operator import Operator


@trait_docs
class PixelsWCS(Operator):
    """Operator which generates detector pixel indices defined on a flat projection.

    When placing the projection on the sky, either the `center` or `bounds`
    traits must be specified, but not both.

    When determining the pixel density in the projection, exactly two traits from the
    set of `bounds`, `resolution` and `dimensions` must be specified.

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels.

    This uses the astropy wcs utilities to build the projection parameters.  By
    default, the world to pixel conversion is performed with internal, optimized code
    unless use_astropy is set to True.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    projection = Unicode("CAR", help="Supported values are CAR, CEA, MER, ZEA, TAN")

    center = Tuple(
        (180 * u.degree, 0 * u.degree),
        allow_none=True,
        help="The center Lon/Lat coordinates (Quantities) of the projection",
    )

    center_offset = Unicode(
        None,
        allow_none=True,
        help="Optional name of shared field with lon, lat offset in degrees",
    )

    bounds = Tuple(
        tuple(),
        help="The (lon_min, lon_max, lat_min, lat_max) values (Quantities)",
    )

    auto_bounds = Bool(
        False,
        help="If True, set the bounding box based on boresight and field of view",
    )

    dimensions = Tuple(
        (710, 350),
        help="The Lon/Lat pixel dimensions of the projection",
    )

    resolution = Tuple(
        (0.5 * u.degree, 0.5 * u.degree),
        help="The Lon/Lat projection resolution (Quantities) along the 2 axes",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode("pixels", help="Observation detdata key for output pixel indices")

    submaps = Int(10, help="Number of submaps to use")

    create_dist = Unicode(
        None,
        allow_none=True,
        help="Create the submap distribution for all detectors and store in the Data key specified",
    )

    single_precision = Bool(False, help="If True, use 32bit int in output")

    use_astropy = Bool(True, help="If True, use astropy for world to pix conversion")

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

    @traitlets.validate("wcs_projection")
    def _check_wcs_projection(self, proposal):
        check = proposal["value"]
        if check not in ["CAR", "CEA", "MER", "ZEA", "TAN"]:
            raise traitlets.TraitError("Invalid WCS projection name")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # If running with all default values, the 'observe' function will not
        # have been called yet.
        if not hasattr(self, "_local_submaps"):
            self._set_wcs(
                self.projection,
                self.center,
                self.bounds,
                self.dimensions,
                self.resolution,
            )
        self._done_auto = False

    @traitlets.observe("auto_bounds")
    def _reset_auto_bounds(self, change):
        # Track whether we need to recompute the bounds.
        if change["new"]:
            # enabling
            self._done_auto = False
        else:
            self._done_auto = True

    @traitlets.observe("center_offset")
    def _reset_auto_center(self, change):
        # Track whether we need to recompute the bounds.
        if change["new"] is not None:
            if self.auto_bounds:
                self._done_auto = False

    @traitlets.observe("projection", "center", "bounds", "dimensions", "resolution")
    def _reset_wcs(self, change):
        # (Re-)initialize the WCS projection when one of these traits change.
        # Current values:
        proj = str(self.projection)
        center = self.center
        if len(center) > 0:
            center = tuple(self.center)
        bounds = self.bounds
        if len(bounds) > 0:
            bounds = tuple(self.bounds)
        dims = self.dimensions
        if len(dims) > 0:
            dims = tuple(self.dimensions)
        res = self.resolution
        if len(res) > 0:
            res = tuple(self.resolution)

        # Update to the trait that changed
        if change["name"] == "projection":
            proj = change["new"]
        if change["name"] == "center":
            center = change["new"]
            if len(center) > 0:
                bounds = tuple()
        if change["name"] == "bounds":
            bounds = change["new"]
            if len(bounds) > 0:
                center = tuple()
                if len(dims) > 0 and len(res) > 0:
                    # Most likely the user cares about the resolution more...
                    dims = tuple()
        if change["name"] == "dimensions":
            dims = change["new"]
            if len(dims) > 0 and len(bounds) > 0:
                res = tuple()
        if change["name"] == "resolution":
            res = change["new"]
            if len(res) > 0 and len(bounds) > 0:
                dims = tuple()
        self._set_wcs(proj, center, bounds, dims, res)
        self.projection = proj
        self.center = center
        self.bounds = bounds
        self.dimensions = dims
        self.resolution = res

    def _set_wcs(self, proj, center, bounds, dims, res):
        log = Logger.get()
        log.verbose(f"PixelsWCS: set_wcs {proj}, {center}, {bounds}, {dims}, {res}")
        if len(res) > 0:
            res = np.array(
                [
                    res[0].to_value(u.degree),
                    res[1].to_value(u.degree),
                ]
            )
        if len(dims) > 0:
            dims = np.array([self.dimensions[0], self.dimensions[1]])

        if len(bounds) == 0:
            # Using center, need both resolution and dimensions
            if len(center) == 0:
                # Cannot calculate yet
                return
            if len(res) == 0 or len(dims) == 0:
                # Cannot calculate yet
                return
            pos = np.array(
                [
                    center[0].to_value(u.degree),
                    center[1].to_value(u.degree),
                ]
            )
            mid = pos
        else:
            # Using bounds, exactly one of resolution or dimensions specified
            if len(res) > 0 and len(dims) > 0:
                # Cannot calculate yet
                return

            # Max Longitude
            lower_left_lon = bounds[0].to_value(u.degree)
            # Min Latitude
            lower_left_lat = bounds[2].to_value(u.degree)
            # Min Longitude
            upper_right_lon = bounds[1].to_value(u.degree)
            # Max Latitude
            upper_right_lat = bounds[3].to_value(u.degree)

            pos = np.array(
                [[lower_left_lon, lower_left_lat], [upper_right_lon, upper_right_lat]]
            )
            mid = np.mean(pos, axis=0)

        def _wcs_ref_res(w, p, r, d):
            w.wcs.crpix = [1, 1]
            if len(r) == 0:
                w.wcs.cdelt = [1, 1]
                corners = w.wcs_world2pix(p, 1)
                w.wcs.cdelt *= (corners[1] - corners[0]) / d
            else:
                w.wcs.cdelt = r
                if p.ndim == 2:
                    w.wcs.cdelt[p[1] < p[0]] *= -1
            if p.ndim == 1:
                if len(dims) > 0:
                    off = w.wcs_world2pix(p[None], 0)[0]
                    w.wcs.crpix = np.array(d) / 2.0 + 0.5 - off
            else:
                off = w.wcs_world2pix(p[0, None], 0)[0] + 0.5
                w.wcs.crpix -= off

        self.wcs = WCS(naxis=2)
        if proj == "CAR":
            self.wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
            self.wcs.wcs.crval = np.array([mid[0], 0])
        elif proj == "CEA":
            self.wcs.wcs.ctype = ["RA---CEA", "DEC--CEA"]
            self.wcs.wcs.crval = np.array([mid[0], 0])
            lam = np.cos(np.deg2rad(mid[1])) ** 2
            self.wcs.wcs.set_pv([(2, 1, lam)])
        elif proj == "MER":
            self.wcs.wcs.ctype = ["RA---MER", "DEC--MER"]
            self.wcs.wcs.crval = np.array([mid[0], 0])
        elif proj == "ZEA":
            self.wcs.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
            self.wcs.wcs.crval = mid
        elif proj == "TAN":
            self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            self.wcs.wcs.crval = mid
        else:
            msg = f"Invalid WCS projection name '{proj}'"
            raise ValueError(msg)
        _wcs_ref_res(self.wcs, pos, res, dims)

        if len(dims) == 0:
            # Compute from the bounding box corners
            lower_left = self.wcs.wcs_world2pix(np.array([[pos[0, 0], pos[0, 1]]]), 0)[
                0
            ]
            upper_right = self.wcs.wcs_world2pix(np.array([[pos[1, 0], pos[1, 1]]]), 0)[
                0
            ]
            self.wcs_shape = tuple(
                np.round(np.abs(upper_right - lower_left)).astype(int)
            )
        else:
            self.wcs_shape = tuple(dims)
        log.verbose(f"PixelsWCS: wcs_shape = {self.wcs_shape}")

        self.pix_ra = self.wcs_shape[0]
        self.pix_dec = self.wcs_shape[1]
        self._n_pix = self.pix_ra * self.pix_dec
        self._n_pix_submap = self._n_pix // self.submaps
        if self._n_pix_submap * self.submaps < self._n_pix:
            self._n_pix_submap += 1
        self._local_submaps = None
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        if not self.use_astropy:
            raise NotImplementedError("Only astropy conversion is currently supported")

        if self.auto_bounds and not self._done_auto:
            # Pass through the boresight pointing for every observation and build
            # the maximum extent of the detector field of view.
            lonmax = -2 * np.pi * u.radian
            lonmin = 2 * np.pi * u.radian
            latmax = (-np.pi / 2) * u.radian
            latmin = (np.pi / 2) * u.radian
            for ob in data.obs:
                # The scan range is computed collectively among the group.
                lnmin, lnmax, ltmin, ltmax = scan_range_lonlat(
                    ob,
                    self.detector_pointing.boresight,
                    flags=self.detector_pointing.shared_flags,
                    flag_mask=self.detector_pointing.shared_flag_mask,
                    field_of_view=None,
                    center_offset=self.center_offset,
                )
                lonmin = min(lonmin, lnmin)
                lonmax = max(lonmax, lnmax)
                latmin = min(latmin, ltmin)
                latmax = max(latmax, ltmax)
            if data.comm.comm_world is not None:
                lonlatmin = np.zeros(2, dtype=np.float64)
                lonlatmax = np.zeros(2, dtype=np.float64)
                lonlatmin[0] = lonmin.to_value(u.radian)
                lonlatmin[1] = latmin.to_value(u.radian)
                lonlatmax[0] = lonmax.to_value(u.radian)
                lonlatmax[1] = latmax.to_value(u.radian)
                all_lonlatmin = np.zeros(2, dtype=np.float64)
                all_lonlatmax = np.zeros(2, dtype=np.float64)
                data.comm.comm_world.Allreduce(lonlatmin, all_lonlatmin, op=MPI.MIN)
                data.comm.comm_world.Allreduce(lonlatmax, all_lonlatmax, op=MPI.MAX)
                lonmin = all_lonlatmin[0] * u.radian
                latmin = all_lonlatmin[1] * u.radian
                lonmax = all_lonlatmax[0] * u.radian
                latmax = all_lonlatmax[1] * u.radian
            new_bounds = (
                lonmin.to(u.degree),
                lonmax.to(u.degree),
                latmin.to(u.degree),
                latmax.to(u.degree),
            )
            log.verbose(f"PixelsWCS auto_bounds set to {new_bounds}")
            self.bounds = new_bounds
            self._done_auto = True

        if self._local_submaps is None and self.create_dist is not None:
            self._local_submaps = np.zeros(self.submaps, dtype=np.uint8)

        # Expand detector pointing
        quats_name = self.detector_pointing.quats

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Once this supports accelerator, pass that instead of False
        self.detector_pointing.apply(data, detectors=detectors, use_accel=False)

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

            # Create (or re-use) output data for the pixels, weights and optionally the
            # detector quaternions.

            if self.single_precision:
                exists = ob.detdata.ensure(
                    self.pixels, sample_shape=(), dtype=np.int32, detectors=dets
                )
            else:
                exists = ob.detdata.ensure(
                    self.pixels, sample_shape=(), dtype=np.int64, detectors=dets
                )

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
                        f"Group {data.comm.group}, ob {ob.name}, WCS pixels "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            # Focalplane for this observation
            focalplane = ob.telescope.focalplane

            # Get the flags if needed.  Use the same flags as
            # detector pointing.
            flags = None
            if self.detector_pointing.shared_flags is not None:
                flags = ob.shared[self.detector_pointing.shared_flags].data
                flags &= self.detector_pointing.shared_flag_mask

            center_lonlat = None
            if self.center_offset is not None:
                center_lonlat = ob.shared[self.center_offset].data

            # Process all detectors
            for det in ob.select_local_detectors(detectors):
                for vslice in view_slices:
                    # Timestream of detector quaternions
                    quats = ob.detdata[quats_name][det][vslice]

                    view_samples = len(quats)
                    theta, phi, _ = qa.to_iso_angles(quats)
                    to_deg = 180.0 / np.pi
                    theta *= to_deg
                    phi *= to_deg
                    shift = phi >= 360.0
                    phi[shift] -= 360.0
                    shift = phi < 0.0
                    phi[shift] += 360.0

                    world_in = np.column_stack([phi, 90.0 - theta])

                    if center_lonlat is not None:
                        world_in[:, 0] -= center_lonlat[vslice, 0]
                        world_in[:, 1] -= center_lonlat[vslice, 1]

                    rdpix = self.wcs.wcs_world2pix(world_in, 0)
                    if flags is not None:
                        # Set bad pointing to pixel -1
                        bad_pointing = flags[vslice] != 0
                        rdpix[bad_pointing] = -1
                    rdpix = np.array(np.around(rdpix), dtype=np.int64)

                    ob.detdata[self.pixels][det, vslice] = (
                        rdpix[:, 0] * self.pix_dec + rdpix[:, 1]
                    )
                    bad_pointing = ob.detdata[self.pixels][det, vslice] >= self._n_pix
                    (ob.detdata[self.pixels][det, vslice])[bad_pointing] = -1

                    if self.create_dist is not None:
                        good = ob.detdata[self.pixels][det][vslice] >= 0
                        self._local_submaps[
                            (ob.detdata[self.pixels][det, vslice])[good]
                            // self._n_pix_submap
                        ] = 1

    def _finalize(self, data, **kwargs):
        if self.create_dist is not None:
            submaps = None
            if self.single_precision:
                submaps = np.arange(self.submaps, dtype=np.int32)[
                    self._local_submaps == 1
                ]
            else:
                submaps = np.arange(self.submaps, dtype=np.int64)[
                    self._local_submaps == 1
                ]

            data[self.create_dist] = PixelDistribution(
                n_pix=self._n_pix,
                n_submap=self.submaps,
                local_submaps=submaps,
                comm=data.comm.comm_world,
            )
            # Store a copy of the WCS information in the distribution object
            data[self.create_dist].wcs = self.wcs.deepcopy()
            data[self.create_dist].wcs_shape = tuple(self.wcs_shape)
        return

    def _requires(self):
        req = self.detector_pointing.requires()
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
