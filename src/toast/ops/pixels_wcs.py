# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import warnings

import numpy as np
import pixell
import pixell.enmap
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..mpi import MPI
from ..observation import default_values as defaults
from ..pixels import PixelDistribution
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

    This uses the pixell package to construct the WCS projection parameters.  By
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
        None,
        allow_none=True,
        help="The lower left and upper right Lon/Lat corners (Quantities)",
    )

    auto_bounds = Bool(
        False,
        help="If True, set the bounding box based on boresight and field of view",
    )

    dimensions = Tuple(
        (710, 350),
        allow_none=True,
        help="The Lon/Lat pixel dimensions of the projection",
    )

    resolution = Tuple(
        (0.5 * u.degree, 0.5 * u.degree),
        allow_none=True,
        help="The Lon/Lat projection resolution (Quantities) along the 2 axes",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels = Unicode("pixels", help="Observation detdata key for output pixel indices")

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output quaternions",
    )

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
    def _reset_auto_bounds(self, change):
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
        if center is not None:
            center = tuple(self.center)
        bounds = self.bounds
        if bounds is not None:
            bounds = tuple(self.bounds)
        dims = self.dimensions
        if dims is not None:
            dims = tuple(self.dimensions)
        res = self.resolution
        if res is not None:
            res = tuple(self.resolution)

        # Update to the trait that changed
        if change["name"] == "projection":
            proj = change["new"]
        if change["name"] == "center":
            center = change["new"]
            if center is not None:
                bounds = None
        if change["name"] == "bounds":
            bounds = change["new"]
            if bounds is not None:
                center = None
                if dims is not None and res is not None:
                    # Most likely the user cares about the resolution more...
                    dims = None
        if change["name"] == "dimensions":
            dims = change["new"]
            if dims is not None and bounds is not None:
                res = None
        if change["name"] == "resolution":
            res = change["new"]
            if res is not None and bounds is not None:
                dims = None
        self._set_wcs(proj, center, bounds, dims, res)
        self.projection = proj
        self.center = center
        self.bounds = bounds
        self.dimensions = dims
        self.resolution = res

    def _set_wcs(self, proj, center, bounds, dims, res):
        log = Logger.get()
        log.verbose(f"PixelsWCS: set_wcs {proj}, {center}, {bounds}, {dims}, {res}")
        if res is not None:
            res = np.array(
                [
                    res[0].to_value(u.degree),
                    res[1].to_value(u.degree),
                ]
            )
        if dims is not None:
            dims = np.array([self.dimensions[0], self.dimensions[1]])

        if bounds is None:
            # Using center, need both resolution and dimensions
            if center is None:
                # Cannot calculate yet
                return
            if res is None or dims is None:
                # Cannot calculate yet
                return
            pos = np.array(
                [
                    center[0].to_value(u.degree),
                    center[1].to_value(u.degree),
                ]
            )
        else:
            # Using bounds, exactly one of resolution or dimensions specified
            if res is not None and dims is not None:
                # Cannot calculate yet
                return
            pos = np.array(
                [
                    [
                        bounds[0][0].to_value(u.degree),
                        bounds[0][1].to_value(u.degree),
                    ],
                    [
                        bounds[1][0].to_value(u.degree),
                        bounds[1][1].to_value(u.degree),
                    ],
                ]
            )

        self.wcs = pixell.wcsutils.build(pos, res, dims, rowmajor=False, system=proj)
        if dims is None:
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
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                lnmin, lnmax, ltmin, ltmax = self._get_scan_range(ob)
                lonmin = min(lonmin, lnmin)
                lonmax = max(lonmax, lnmax)
                latmin = min(latmin, ltmin)
                latmax = max(latmax, ltmax)
            if ob.comm.comm_group_rank is not None:
                # Synchronize between groups
                if ob.comm.group_rank == 0:
                    lonmin = ob.comm.comm_group_rank.allreduce(lonmin, op=MPI.MIN)
                    latmin = ob.comm.comm_group_rank.allreduce(latmin, op=MPI.MIN)
                    lonmax = ob.comm.comm_group_rank.allreduce(lonmax, op=MPI.MAX)
                    latmax = ob.comm.comm_group_rank.allreduce(latmax, op=MPI.MAX)
            # Broadcast result within the group
            if ob.comm.comm_group is not None:
                lonmin = ob.comm.comm_group.bcast(lonmin, root=0)
                lonmax = ob.comm.comm_group.bcast(lonmax, root=0)
                latmin = ob.comm.comm_group.bcast(latmin, root=0)
                latmax = ob.comm.comm_group.bcast(latmax, root=0)
            new_bounds = (
                (lonmax.to(u.degree), latmin.to(u.degree)),
                (lonmin.to(u.degree), latmax.to(u.degree)),
            )
            # print(
            #     f"WCS auto now set to lon: {lonmin.to(u.degree)} .. {lonmax.to(u.degree)}, lat: {latmin.to(u.degree)} .. {latmax.to(u.degree)}"
            # )
            log.verbose(f"PixelsWCS auto_bounds set to {new_bounds}")
            self.bounds = new_bounds
            self._done_auto = True

        if self._local_submaps is None and self.create_dist is not None:
            # print("WCS reset _local_submaps to zeros")
            self._local_submaps = np.zeros(self.submaps, dtype=np.bool)

        # Expand detector pointing
        if self.quats is not None:
            quats_name = self.quats
        else:
            if self.detector_pointing.quats is not None:
                quats_name = self.detector_pointing.quats
            else:
                quats_name = defaults.quats

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        self.detector_pointing.quats = quats_name
        self.detector_pointing.apply(data, detectors=detectors)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # FIXME:  remove this workaround after #557 is merged
            if view is None:
                view_slices = [slice(None)]
            else:
                view_slices = [
                    slice(x.first, x.last + 1, 1) for x in ob.intervals[view]
                ]

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

            # Do we already have pointing for all requested detectors?
            if exists:
                # Yes...
                if self.create_dist is not None:
                    # but the caller wants the pixel distribution
                    for det in ob.select_local_detectors(detectors):
                        for vslice in view_slices:
                            good = ob.detdata[self.pixels][det][vslice] >= 0
                            self._local_submaps[
                                ob.detdata[self.pixels][det][vslice][good]
                                // self._n_pix_submap
                            ] = True

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
                flags = np.array(ob.shared[self.detector_pointing.shared_flags])
                fvals, fcounts = np.unique(flags, return_counts=True)
                # print(f"WCS flag counts = {fvals}, {fcounts}")
                flags &= self.detector_pointing.shared_flag_mask
                n_good = np.sum(flags == 0)
                n_bad = np.sum(flags != 0)
                # print(
                #     f"WCS flag mask {int(self.detector_pointing.shared_flag_mask)} has {n_good} good and {n_bad} bad samples"
                # )

            center_lonlat = None
            if self.center_offset is not None:
                center_lonlat = ob.shared[self.center_offset].data

            # Process all detectors
            for det in ob.select_local_detectors(detectors):
                for vslice in view_slices:
                    # Timestream of detector quaternions
                    quats = ob.detdata[quats_name][det][vslice]
                    # print(f"WCS det {det}, quats = {quats}")

                    view_samples = len(quats)
                    theta, phi, _ = qa.to_iso_angles(quats)
                    # print(f"WCS det {det}, theta rad = {theta}, phi rad = {phi}")
                    to_deg = 180.0 / np.pi
                    theta *= to_deg
                    phi *= to_deg

                    # print(f"WCS det {det}, theta deg = {theta}, phi deg = {phi}")

                    world_in = np.column_stack([phi, 90.0 - theta])
                    # print(f"WCS det {det}, world_in = {world_in}")

                    if center_lonlat is not None:
                        world_in[:, 0] -= center_lonlat[vslice, 0]
                        world_in[:, 1] -= center_lonlat[vslice, 1]

                    # print(f"WCS det {det}, world_in after center = {world_in}")

                    rdpix = self.wcs.wcs_world2pix(world_in, 0)
                    # print(
                    #     f"WCS det {det}, {view_samples} samps, {np.count_nonzero(rdpix >= 0)} pix >= 0, {np.count_nonzero(rdpix > 0)} pix > 0"
                    # )
                    if flags is not None:
                        # Set bad pointing to pixel -1
                        bad_pointing = flags[vslice] != 0
                        rdpix[bad_pointing] = -1
                    rdpix = np.array(np.around(rdpix), dtype=np.int64)

                    ob.detdata[self.pixels][det][vslice] = (
                        rdpix[:, 0] * self.pix_dec + rdpix[:, 1]
                    )

                    if self.create_dist is not None:
                        good = ob.detdata[self.pixels][det][vslice] >= 0
                        self._local_submaps[
                            (ob.detdata[self.pixels][det][vslice])[good]
                            // self._n_pix_submap
                        ] = 1

    @function_timer
    def _get_scan_range(self, obs):
        # FIXME: mostly copied from the atmosphere simulation code- we should
        # extract this into a more general helper routine somewhere.
        fov = obs.telescope.focalplane.field_of_view
        fp_radius = 0.5 * fov.to_value(u.radian)

        # Get the flags if needed.  Use the same flags as
        # detector pointing.
        flags = None
        if self.detector_pointing.shared_flags is not None:
            flags = np.array(obs.shared[self.detector_pointing.shared_flags])
            flags &= self.detector_pointing.shared_flag_mask

        # work in parallel
        rank = obs.comm.group_rank
        ntask = obs.comm.group_size

        # Create a fake focalplane of detectors in a circle around the boresight
        xaxis, yaxis, zaxis = np.eye(3)
        ndet = 64
        phidet = np.linspace(0, 2 * np.pi, ndet, endpoint=False)
        detquats = []
        thetarot = qa.rotation(yaxis, fp_radius)
        for phi in phidet:
            phirot = qa.rotation(zaxis, phi)
            detquat = qa.mult(phirot, thetarot)
            detquats.append(detquat)

        # Get fake detector pointing

        center_lonlat = None
        if self.center_offset is not None:
            center_lonlat = np.array(obs.shared[self.center_offset].data)
            center_lonlat[:, :] *= np.pi / 180.0

        lon = []
        lat = []
        quats = obs.shared[self.detector_pointing.boresight][rank::ntask].copy()
        rank_good = slice(None)
        if self.detector_pointing.shared_flags is not None:
            rank_good = flags[rank::ntask] == 0

        for idet, detquat in enumerate(detquats):
            theta, phi, _ = qa.to_iso_angles(qa.mult(quats, detquat))
            if center_lonlat is None:
                lon.append(phi[rank_good])
                lat.append(np.pi / 2 - theta[rank_good])
            else:
                lon.append(phi[rank_good] - center_lonlat[rank::ntask, 0][rank_good])
                lat.append(
                    (np.pi / 2 - theta[rank_good])
                    - center_lonlat[rank::ntask, 1][rank_good]
                )
        lon = np.unwrap(np.hstack(lon))
        lat = np.hstack(lat)

        # find the extremes
        lonmin = np.amin(lon)
        lonmax = np.amax(lon)
        latmin = np.amin(lat)
        latmax = np.amax(lat)

        if lonmin < -2 * np.pi:
            lonmin += 2 * np.pi
            lonmax += 2 * np.pi
        elif lonmax > 2 * np.pi:
            lonmin -= 2 * np.pi
            lonmax -= 2 * np.pi

        # Combine results
        if obs.comm.comm_group is not None:
            lonmin = obs.comm.comm_group.allreduce(lonmin, op=MPI.MIN)
            lonmax = obs.comm.comm_group.allreduce(lonmax, op=MPI.MAX)
            latmin = obs.comm.comm_group.allreduce(latmin, op=MPI.MIN)
            latmax = obs.comm.comm_group.allreduce(latmax, op=MPI.MAX)

        return (
            lonmin * u.radian,
            lonmax * u.radian,
            latmin * u.radian,
            latmax * u.radian,
        )

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
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.pixels)
        if self.create_dist is not None:
            prov["global"].append(self.create_dist)
        return prov
