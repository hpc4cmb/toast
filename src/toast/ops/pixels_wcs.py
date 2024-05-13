# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import warnings

import astropy.io.fits as af
import numpy as np
import traitlets
from astropy import units as u
from astropy.wcs import WCS

from .. import qarray as qa
from ..instrument_coords import quat_to_xieta
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelDistribution
from ..pointing_utils import center_offset_lonlat, scan_range_lonlat
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

    This uses the astropy wcs utilities to build the projection parameters.  Eventually
    this operator will use internal kernels for the projection unless `use_astropy`
    is set to True.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    fits_header = Unicode(
        None,
        allow_none=True,
        help="FITS file containing header to use with pre-existing WCS parameters",
    )

    coord_frame = Unicode("EQU", help="Supported values are AZEL, EQU, GAL, ECL")

    projection = Unicode(
        "CAR", help="Supported values are CAR, CEA, MER, ZEA, TAN, SFL"
    )

    center = Tuple(
        tuple(),
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
        True,
        help="If True, set the bounding box based on boresight and field of view",
    )

    dimensions = Tuple(
        (1000, 1000),
        help="The Lon/Lat pixel dimensions of the projection",
    )

    resolution = Tuple(
        tuple(),
        help="The Lon/Lat projection resolution (Quantities) along the 2 axes",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode("pixels", help="Observation detdata key for output pixel indices")

    submaps = Int(1, help="Number of submaps to use")

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
        if check not in ["CAR", "CEA", "MER", "ZEA", "TAN", "SFL"]:
            raise traitlets.TraitError("Invalid WCS projection name")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Track whether we need to recompute autobounds
        self._done_auto = False
        # Track whether we need to recompute the WCS projection
        self._done_wcs = False

    @traitlets.observe("auto_bounds")
    def _reset_auto_bounds(self, change):
        # Track whether we need to recompute the bounds.
        old_val = change["old"]
        new_val = change["new"]
        if new_val != old_val:
            self._done_auto = False
            self._done_wcs = False

    @traitlets.observe("center_offset")
    def _reset_auto_center(self, change):
        old_val = change["old"]
        new_val = change["new"]
        # Track whether we need to recompute the projection
        if new_val != old_val:
            self._done_wcs = False
            self._done_auto = False

    @traitlets.observe("projection", "center", "bounds", "dimensions", "resolution")
    def _reset_wcs(self, change):
        # (Re-)initialize the WCS projection when one of these traits change.
        old_val = change["old"]
        new_val = change["new"]
        if old_val != new_val:
            self._done_wcs = False
            self._done_auto = False

    @classmethod
    def create_wcs(
        cls,
        coord="EQU",
        proj="CAR",
        center_deg=None,
        bounds_deg=None,
        res_deg=None,
        dims=None,
    ):
        """Create a WCS object given projection parameters.

        Either the `center_deg` or `bounds_deg` parameters must be specified,
        but not both.

        When determining the pixel density in the projection, exactly two
        parameters from the set of `bounds_deg`, `res_deg` and `dims` must be
        specified.

        Args:
            coord (str):  The coordinate frame name.
            proj (str):  The projection type.
            center_deg (tuple):  The (lon, lat) projection center in degrees.
            bounds_deg (tuple):  The (lon_min, lon_max, lat_min, lat_max)
                values in degrees.
            res_deg (tuple):  The (lon, lat) resolution in degrees.
            dims (tuple):  The (lon, lat) projection size in pixels.

        Returns:
            (WCS, shape): The instantiated WCS object and final shape.

        """
        log = Logger.get()

        # Compute projection center
        if center_deg is not None:
            # We are specifying the center.  Bounds should not be set and we should
            # have both resolution and dimensions
            if bounds_deg is not None:
                msg = f"PixelsWCS: only one of center and bounds should be set."
                log.error(msg)
                raise RuntimeError(msg)
            if res_deg is None or dims is None:
                msg = f"PixelsWCS: when center is set, both resolution and dimensions"
                msg += f" are required."
                log.error(msg)
                raise RuntimeError(msg)
            crval = np.array(center_deg, dtype=np.float64)
        else:
            # Not using center, bounds is required
            if bounds_deg is None:
                msg = f"PixelsWCS: when center is not specified, bounds required."
                log.error(msg)
                raise RuntimeError(msg)
            mid_lon = 0.5 * (bounds_deg[1] + bounds_deg[0])
            mid_lat = 0.5 * (bounds_deg[3] + bounds_deg[2])
            crval = np.array([mid_lon, mid_lat], dtype=np.float64)
            # Either resolution or dimensions should be specified
            if res_deg is not None:
                # Using resolution
                if dims is not None:
                    msg = f"PixelsWCS: when using bounds, only one of resolution or"
                    msg += f" dimensions must be specified."
                    log.error(msg)
                    raise RuntimeError(msg)
            else:
                # Using dimensions
                if res_deg is not None:
                    msg = f"PixelsWCS: when using bounds, only one of resolution or"
                    msg += f" dimensions must be specified."
                    log.error(msg)
                    raise RuntimeError(msg)

        # Create the WCS object.
        # CTYPE1 = Longitude
        # CTYPE2 = Latitude
        wcs = WCS(naxis=2)

        if coord == "AZEL":
            # For local Azimuth and Elevation coordinate frame, we
            # use the generic longitude and latitude string.
            coordstr = ("TLON", "TLAT")
        elif coord == "EQU":
            coordstr = ("RA--", "DEC-")
        elif coord == "GAL":
            coordstr = ("GLON", "GLAT")
        elif coord == "ECL":
            coordstr = ("ELON", "ELAT")
        else:
            msg = f"Unsupported coordinate frame '{coord}'"
            raise RuntimeError(msg)

        if proj == "CAR":
            wcs.wcs.ctype = [f"{coordstr[0]}-CAR", f"{coordstr[1]}-CAR"]
            wcs.wcs.crval = crval
        elif proj == "CEA":
            wcs.wcs.ctype = [f"{coordstr[0]}-CEA", f"{coordstr[1]}-CEA"]
            wcs.wcs.crval = crval
            lam = np.cos(np.deg2rad(crval[1])) ** 2
            wcs.wcs.set_pv([(2, 1, lam)])
        elif proj == "MER":
            wcs.wcs.ctype = [f"{coordstr[0]}-MER", f"{coordstr[1]}-MER"]
            wcs.wcs.crval = crval
        elif proj == "ZEA":
            wcs.wcs.ctype = [f"{coordstr[0]}-ZEA", f"{coordstr[1]}-ZEA"]
            wcs.wcs.crval = crval
        elif proj == "TAN":
            wcs.wcs.ctype = [f"{coordstr[0]}-TAN", f"{coordstr[1]}-TAN"]
            wcs.wcs.crval = crval
        elif proj == "SFL":
            wcs.wcs.ctype = [f"{coordstr[0]}-SFL", f"{coordstr[1]}-SFL"]
            wcs.wcs.crval = crval
        else:
            msg = f"Invalid WCS projection name '{proj}'"
            raise ValueError(msg)

        # Compute resolution.  Note that we negate the longitudinal
        # coordinate so that the resulting projections match expectations
        # for plotting, etc.
        if center_deg is not None:
            wcs.wcs.cdelt = np.array([-res_deg[0], res_deg[1]])
        else:
            if res_deg is not None:
                wcs.wcs.cdelt = np.array([-res_deg[0], res_deg[1]])
            else:
                # Compute CDELT from the bounding box and image size.
                wcs.wcs.cdelt = np.array(
                    [
                        -(bounds_deg[1] - bounds_deg[0]) / dims[0],
                        (bounds_deg[3] - bounds_deg[2]) / dims[1],
                    ]
                )

        # Compute shape of the projection
        if dims is not None:
            wcs_shape = tuple(dims)
        else:
            # Compute from the bounding box corners
            lower_left = wcs.wcs_world2pix(
                np.array([[bounds_deg[0], bounds_deg[2]]]), 0
            )[0]
            upper_right = wcs.wcs_world2pix(
                np.array([[bounds_deg[1], bounds_deg[3]]]), 0
            )[0]
            wcs_shape = tuple(np.round(np.abs(upper_right - lower_left)).astype(int))

        # Set the reference pixel to the center of the projection
        off = wcs.wcs_world2pix(crval.reshape((1, 2)), 0)[0]
        wcs.wcs.crpix = 0.5 * np.array(wcs_shape, dtype=np.float64) + 0.5 + off

        return wcs, wcs_shape

    def set_wcs(self):
        if self._done_wcs:
            return

        log = Logger.get()
        msg = f"PixelsWCS: set_wcs coord={self.coord_frame}, "
        msg += f"proj={self.projection}, center={self.center}, bounds={self.bounds}"
        msg += f", dims={self.dimensions}, res={self.resolution}"
        log.verbose(msg)

        center_deg = None
        if len(self.center) > 0:
            if self.center_offset is None:
                center_deg = (
                    self.center[0].to_value(u.degree),
                    self.center[1].to_value(u.degree),
                )
            else:
                center_deg = (0.0, 0.0)
        bounds_deg = None
        if len(self.bounds) > 0:
            bounds_deg = tuple([x.to_value(u.degree) for x in self.bounds])
        res_deg = None
        if len(self.resolution) > 0:
            res_deg = tuple([x.to_value(u.degree) for x in self.resolution])
        if len(self.dimensions) > 0:
            dims = tuple(self.dimensions)
        else:
            dims = None

        self.wcs, self.wcs_shape = self.create_wcs(
            coord=self.coord_frame,
            proj=self.projection,
            center_deg=center_deg,
            bounds_deg=bounds_deg,
            res_deg=res_deg,
            dims=dims,
        )

        self.pix_lon = self.wcs_shape[0]
        self.pix_lat = self.wcs_shape[1]
        self._n_pix = self.pix_lon * self.pix_lat
        self._n_pix_submap = self._n_pix // self.submaps
        if self._n_pix_submap * self.submaps < self._n_pix:
            self._n_pix_submap += 1
        self._local_submaps = np.zeros(self.submaps, dtype=np.uint8)
        self._done_wcs = True
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        if not self.use_astropy:
            raise NotImplementedError("Only astropy conversion is currently supported")

        if self.fits_header is not None:
            # with open(self.fits_header, "rb") as f:
            #     header = af.Header.fromfile(f)
            raise NotImplementedError(
                "Initialization from a FITS header not yet finished"
            )

        if self.coord_frame == "AZEL":
            is_azimuth = True
        else:
            is_azimuth = False

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
                    is_azimuth=is_azimuth,
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
            self.bounds = (
                lonmin.to(u.degree),
                lonmax.to(u.degree),
                latmin.to(u.degree),
                latmax.to(u.degree),
            )
            log.verbose(f"PixelsWCS: auto_bounds set to {self.bounds}")
            self._done_auto = True

        # Compute the projection if needed
        self.set_wcs()

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
            dets = ob.select_local_detectors(
                detectors, flagmask=self.detector_pointing.det_mask
            )
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
                    for det in dets:
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
                center_lonlat = np.radians(ob.shared[self.center_offset].data)

            # Process all detectors
            for det in dets:
                for vslice in view_slices:
                    # Timestream of detector quaternions
                    quats = ob.detdata[quats_name][det][vslice]
                    view_samples = len(quats)

                    if center_lonlat is None:
                        center_offset = None
                    else:
                        center_offset = center_lonlat[vslice]

                    rel_lon, rel_lat = center_offset_lonlat(
                        quats,
                        center_offset=center_offset,
                        degrees=True,
                        is_azimuth=is_azimuth,
                    )

                    world_in = np.column_stack([rel_lon, rel_lat])

                    rdpix = self.wcs.wcs_world2pix(world_in, 0)
                    rdpix = np.array(np.around(rdpix), dtype=np.int64)

                    ob.detdata[self.pixels][det, vslice] = (
                        rdpix[:, 0] * self.pix_lat + rdpix[:, 1]
                    )
                    bad_pointing = ob.detdata[self.pixels][det, vslice] >= self._n_pix
                    if flags is not None:
                        bad_pointing = np.logical_or(bad_pointing, flags[vslice] != 0)
                    (ob.detdata[self.pixels][det, vslice])[bad_pointing] = -1

                    if self.create_dist is not None:
                        good = ob.detdata[self.pixels][det][vslice] >= 0
                        self._local_submaps[
                            (ob.detdata[self.pixels][det, vslice])[good]
                            // self._n_pix_submap
                        ] = 1

    def _finalize(self, data, **kwargs):
        if self.create_dist is not None:
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
            # Reset the local submaps
            self._local_submaps[:] = 0
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
