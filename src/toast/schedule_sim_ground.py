#!/usr/bin/env python3

# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
This script creates a CES schedule file that can be used as input
to toast_ground_sim.py
"""

import argparse
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import dateutil.parser
import ephem
import healpy as hp
import numpy as np
from matplotlib import cm
from scipy.constants import degree

from . import qarray as qa
from .coordinates import DJDtoUNIX, to_DJD, to_MJD, to_UTC
from .timing import function_timer
from .utils import Logger

XAXIS, YAXIS, ZAXIS = np.eye(3)


class TooClose(Exception):
    pass


class SunTooClose(TooClose):
    pass


class MoonTooClose(TooClose):
    pass


class Patch(object):
    hits = 0
    partial_hits = 0
    rising_hits = 0
    setting_hits = 0
    time = 0
    rising_time = 0
    setting_time = 0
    step = -1
    az_min = 0
    az_max = 2 * np.pi
    _area = None
    current_el_min = 0
    current_el_max = 0
    el_min0 = 0
    el_max0 = np.pi / 2
    el_min = el_min0
    el_max = el_max0
    el_step = 0
    alternate = False
    ra_amplitude = None
    ra_period = 10
    dec_amplitude = None
    dec_period = 10
    corners = []
    preferred_el = None

    def __init__(
        self,
        name,
        weight,
        corners,
        el_min=0,
        el_max=np.pi / 2,
        el_step=0,
        alternate=False,
        site_lat=0,
        area=None,
        ra_period=10,
        ra_amplitude=None,
        dec_period=10,
        dec_amplitude=None,
        elevations=None,
    ):
        self.name = name
        self.weight = weight
        self.corners = corners
        self.el_min0 = el_min
        self.el_min = el_min
        self.el_max0 = el_max
        self.el_step = np.abs(el_step)
        self.alternate = alternate
        self._area = area
        self.site_lat = site_lat
        self.ra_period = ra_period
        self.ra_amplitude = np.radians(ra_amplitude)
        self.dec_period = dec_period
        self.dec_amplitude = np.radians(dec_amplitude)
        # Use the site latitude to infer the lowest elevation that all
        # corners cross.
        site_el_max = np.pi / 2
        for corner in corners:
            el_max = np.pi / 2 - np.abs(corner._dec - self.site_lat)
            if el_max < site_el_max:
                site_el_max = el_max
        self.parse_elevations(elevations, site_el_max)
        if el_step != 0:
            self.nstep_el = int((self.el_max0 - self.el_min0 + 1e-3) // el_step) + 1
        self.el_max = self.el_max0
        self.el_lim = self.el_min0
        self.step_azel()
        return

    def parse_elevations(self, elevations, site_el_max=np.pi / 2):
        if elevations is None:
            if site_el_max < self.el_max0:
                self.el_max0 = site_el_max
            self.elevations = None
        else:
            # Parse the allowed elevations
            try:
                # Try parsing as a string
                self.elevations = [
                    np.radians(float(el)) for el in elevations.split(",")
                ]
            except AttributeError:
                # Try parsing as an iterable
                self.elevations = [np.radians(el) for el in elevations]
            self.elevations = np.sort(np.array(self.elevations))
            # Check if any of the allowed elevations is above the highest
            # observable elevation
            bad = self.elevations > site_el_max
            if np.any(bad):
                good = np.logical_not(bad)
                if np.any(good):
                    print(
                        "WARNING: {} of the observing elevations are too high "
                        "for '{}': {} > {:.2f} deg".format(
                            np.sum(bad),
                            self.name,
                            np.degrees(self.elevations[bad]),
                            np.degrees(site_el_max),
                        ),
                        flush=True,
                    )
                    self.elevations = self.elevations[good]
                else:
                    print(
                        "ERROR: all of the observing elevations are too high for {}.  "
                        "Maximum observing elevation is {} deg".format(
                            self.name, np.degrees(site_el_max)
                        ),
                        flush=True,
                    )
                    sys.exit()
            self.el_min0 = np.amin(self.elevations)
            self.el_max0 = np.amax(self.elevations)
        self.elevations0 = self.elevations
        return

    def oscillate(self):
        if self.ra_amplitude:
            # Oscillate RA
            halfperiod = self.ra_period // 2
            old_phase = np.fmod(self.hits - 1 + halfperiod, self.ra_period) - halfperiod
            new_phase = np.fmod(self.hits + halfperiod, self.ra_period) - halfperiod
            old_offset = old_phase / halfperiod * self.ra_amplitude
            new_offset = new_phase / halfperiod * self.ra_amplitude
            offset = new_offset - old_offset
            for corner in self.corners:
                corner._ra += offset
        if self.dec_amplitude:
            # Oscillate DEC
            halfperiod = self.dec_period // 2
            old_phase = (
                np.fmod(self.hits - 1 + halfperiod, self.dec_period) - halfperiod
            )
            new_phase = np.fmod(self.hits + halfperiod, self.dec_period) - halfperiod
            old_offset = old_phase / halfperiod * self.dec_amplitude
            new_offset = new_phase / halfperiod * self.dec_amplitude
            offset = new_offset - old_offset
            for corner in self.corners:
                corner._dec += offset
        return

    @function_timer
    def get_area(self, observer, nside=32, equalize=False):
        self.update(observer)
        if self._area is None:
            npix = 12 * nside**2
            hitmap = np.zeros(npix)
            for corner in self.corners:
                corner.compute(observer)
            for pix in range(npix):
                lon, lat = hp.pix2ang(nside, pix, lonlat=True)
                center = ephem.FixedBody()
                center._ra = np.radians(lon)
                center._dec = np.radians(lat)
                center.compute(observer)
                hitmap[pix] = self.in_patch(center)
            self._area = np.sum(hitmap) / hitmap.size
        if self._area == 0:
            raise RuntimeError("Patch has zero area!")
        if equalize:
            self.weight /= self._area
        return self._area

    @function_timer
    def corner_coordinates(self, observer=None, unwind=False):
        """Return the corner coordinates in horizontal frame.

        PyEphem measures the azimuth East (clockwise) from North.
        """
        azs = []
        els = []
        az0 = None
        for corner in self.corners:
            if observer is not None:
                corner.compute(observer)
            if unwind:
                if az0 is None:
                    az0 = corner.az
                azs.append(unwind_angle(az0, corner.az))
            else:
                azs.append(corner.az)
            els.append(corner.alt)
        return np.array(azs), np.array(els)

    @function_timer
    def in_patch(self, obj):
        """
        Determine if the object (e.g. Sun or Moon) is inside the patch
        by using a ray casting algorithm.  The ray is cast along a
        constant meridian to follow a great circle.
        """
        az0 = obj.az
        # Get corner coordinates, assuming they were already computed
        azs, els = self.corner_coordinates()
        els_cross = []
        for i in range(len(self.corners)):
            az1 = azs[i]
            el1 = els[i]
            j = (i + 1) % len(self.corners)
            az2 = unwind_angle(az1, azs[j])
            el2 = els[j]
            azmean = 0.5 * (az1 + az2)
            az0 = unwind_angle(azmean, float(obj.az), np.pi)
            if (az1 - az0) * (az2 - az0) > 0:
                # the constant meridian is not between the two corners
                continue
            el_cross = el1 + (az1 - az0) * (el2 - el1) / (az1 - az2)
            if np.abs(obj.az - (az0 % (2 * np.pi))) < 1e-3:
                els_cross.append(el_cross)
            elif el_cross > 0:
                els_cross.append(np.pi - el_cross)
            else:
                els_cross.append(-np.pi - el_cross)

        els_cross = np.array(els_cross)
        if els_cross.size < 2:
            return False

        # Unwind the crossing elevations to minimize the scatter
        els_cross = np.sort(els_cross)
        if els_cross.size > 1:
            ptps = []
            for i in range(els_cross.size):
                els_cross_alt = els_cross.copy()
                els_cross_alt[:i] += 2 * np.pi
                ptps.append(np.ptp(els_cross_alt))
            i = np.argmin(ptps)
            if i > 0:
                els_cross[:i] += 2 * np.pi
                els_cross = np.sort(els_cross)
        el_mean = np.mean(els_cross)
        el0 = unwind_angle(el_mean, float(obj.alt))

        ncross = np.sum(els_cross > el0)

        if ncross % 2 == 0:
            # Even number of crossings means that the object is outside
            # of the patch
            return False
        return True

    @function_timer
    def step_azel(self):
        self.step += 1
        if self.el_step != 0 and self.alternate:
            # alternate between rising and setting scans
            if self.rising_hits < self.setting_hits:
                # Schedule a rising scan
                istep = self.rising_hits % self.nstep_el
                self.el_min = min(self.el_max0, self.el_min0 + istep * self.el_step)
                self.el_max = self.el_max0
                self.az_min = 0
                self.az_max = np.pi
            else:
                # Schedule a setting scan
                istep = self.setting_hits % self.nstep_el
                self.el_min = self.el_min0
                self.el_max = max(self.el_min0, self.el_max0 - istep * self.el_step)
                self.az_min = np.pi
                self.az_max = 2 * np.pi
        else:
            if self.alternate:
                self.az_min = (self.az_min + np.pi) % (2 * np.pi)
                self.az_max = self.az_min + np.pi
            else:
                self.el_min += self.el_step
                if self.el_min > self.el_max0:
                    self.el_min = self.el_min0
        if self.el_step != 0 and self.elevations is not None:
            tol = np.radians(0.1)
            self.elevations = np.array(
                [
                    el
                    for el in self.elevations0
                    if (el + tol >= self.el_min and el - tol <= self.el_max)
                ]
            )
        return

    def reset(self):
        self.step += 1
        self.el_min = self.el_min0
        self.el_max = self.el_max0
        try:
            self.elevations = self.elevations0
            self.az_min = 0
            if self.alternate:
                self.az_max = np.pi
            else:
                self.az_max = 2 * np.pi
        except AttributeError:
            # HorizontalPatch does not maintain a list of elevations
            pass
        return

    def el_range(self, observer, rising):
        """Return the minimum and maximum elevation"""
        self.update(observer)
        patch_el_max = -1000
        patch_el_min = 1000
        in_view = False
        for i, corner in enumerate(self.corners):
            corner.compute(observer)
            if rising and corner.az > np.pi:
                continue
            if not rising and corner.az < np.pi:
                continue
            patch_el_min = min(patch_el_min, corner.alt)
            patch_el_max = max(patch_el_max, corner.alt)
        return patch_el_min, patch_el_max

    def visible(
        self,
        el_min,
        observer,
        sun,
        moon,
        sun_avoidance_angle,
        moon_avoidance_angle,
        check_sso,
    ):
        self.update(observer)
        patch_el_max = -1000
        patch_el_min = 1000
        in_view = False
        for i, corner in enumerate(self.corners):
            corner.compute(observer)
            patch_el_min = min(patch_el_min, corner.alt)
            patch_el_max = max(patch_el_max, corner.alt)
            if corner.alt > el_min:
                # At least one corner is visible
                in_view = True
            if check_sso:
                if sun_avoidance_angle > 0:
                    angle = np.degrees(ephem.separation(sun, corner))
                    if angle < sun_avoidance_angle:
                        # Patch is too close to the Sun
                        return False, f"Too close to Sun {angle:.2f}"
                if moon_avoidance_angle > 0:
                    angle = np.degrees(ephem.separation(moon, corner))
                    if angle < moon_avoidance_angle:
                        # Patch is too close to the Moon
                        return False, f"Too close to Moon {angle:.2f}"
        if not in_view:
            msg = "Below el_min = {:.2f} at el = {:.2f}..{:.2f}.".format(
                np.degrees(el_min), np.degrees(patch_el_min), np.degrees(patch_el_max)
            )
        else:
            msg = "in view"
            self.current_el_min = patch_el_min
            self.current_el_max = patch_el_max

        return in_view, msg

    def update(self, *args, **kwargs):
        """
        A virtual method that is implemented by moving targets
        """
        pass


class SSOPatch(Patch):
    def __init__(
        self,
        name,
        weight,
        radius,
        el_min=0,
        el_max=np.pi / 2,
        elevations=None,
    ):
        self.name = name
        self.weight = weight
        self.radius = radius
        self._area = np.pi * radius**2 / (4 * np.pi)
        self.el_min0 = el_min
        self.el_min = el_min
        self.el_max = el_max
        self.el_max0 = el_max
        self.parse_elevations(elevations)
        try:
            self.body = getattr(ephem, name)()
        except:
            raise RuntimeError("Failed to initialize {} from pyEphem".format(name))
        self.corners = None
        return

    def update(self, observer):
        """
        Calculate the relative position of the SSO at a given time
        """
        self.body.compute(observer)
        ra, dec = self.body.ra, self.body.dec
        # Synthesize 8 corners around the center
        phi = ra
        theta = dec
        r = self.radius
        ncorner = 8
        angstep = 2 * np.pi / ncorner
        self.corners = []
        for icorner in range(ncorner):
            ang = angstep * icorner
            delta_theta = np.cos(ang) * r
            delta_phi = np.sin(ang) * r / np.cos(theta + delta_theta)
            patch_corner = ephem.FixedBody()
            patch_corner._ra = phi + delta_phi
            patch_corner._dec = theta + delta_theta
            self.corners.append(patch_corner)
        return


class CoolerCyclePatch(Patch):
    def __init__(
        self,
        name,
        weight,
        power,
        hold_time_min,
        hold_time_max,
        cycle_time,
        az,
        el,
        last_cycle_end,
    ):
        # Standardized name for cooler cycles
        self.name = name
        self.hold_time_min = hold_time_min * 3600
        self.hold_time_max = hold_time_max * 3600
        self.cycle_time = cycle_time * 3600
        self.az = az
        self.el = el
        self.last_cycle_end = last_cycle_end
        self.weight0 = weight
        self.weight = weight
        self.power = power
        return

    def get_area(self, *args, **kwargs):
        if self._area is None:
            self._area = 0
        return self._area

    def corner_coordinates(self, *args, **kwargs):
        return None

    def in_patch(self, *args, **kwargs):
        return False

    def step_azel(self, *args, **kwargs):
        return

    def reset(self, *args, **kwargs):
        return

    def get_current_hold_time(self, observer):
        tlast = to_DJD(self.last_cycle_end)
        tnow = float(observer.date)  # In Dublin Julian date
        hold_time = (tnow - tlast) * 86400  # in seconds
        return hold_time

    def visible(
        self,
        el_min,
        observer,
        sun,
        moon,
        sun_avoidance_angle,
        moon_avoidance_angle,
        check_sso,
    ):
        self.update(observer)
        hold_time = self.get_current_hold_time(observer)
        if hold_time > self.hold_time_min:
            visible = True
            msg = "minimum hold time exceeded"
        else:
            visible = False
            msg = "minimum hold time not met"
        return visible, msg

    def update(self, observer):
        hold_time = self.get_current_hold_time(observer)
        if hold_time < self.hold_time_min:
            self.weight = np.inf
        else:
            weight = (self.hold_time_max - hold_time) / (
                self.hold_time_max - self.hold_time_min
            )
            self.weight = self.weight0 * weight**self.power
        return


class HorizontalPatch(Patch):
    elevations = None

    def __init__(self, name, weight, azmin, azmax, el, scantime):
        self.name = name
        self.weight = weight
        if azmin <= np.pi and azmax <= np.pi:
            self.rising = True
        elif azmin >= np.pi and azmax >= np.pi:
            self.rising = False
        else:
            # This patch is being observed across the meridian
            self.rising = None
        self.az_min = azmin % (2 * np.pi)
        self.az_max = azmax % (2 * np.pi)
        self.el = el
        # scan time is the maximum time spent on this scan before targeting again
        self.scantime = scantime  # in minutes.

        self.el_min0 = el
        self.el_min = el
        self.el_max0 = el
        self.el_step = 0
        self.alternate = False
        self._area = 0
        self.el_max = self.el_max0
        self.el_lim = self.el_min0
        self.time = 0
        self.hits = 0
        return

    # The HorizontalPatch objects make no distinction between rising and setting scans

    @property
    def rising_time(self):
        return self.time

    @rising_time.setter
    def rising_time(self, value):
        # self.time += value
        pass

    @property
    def setting_time(self):
        return self.time

    @setting_time.setter
    def setting_time(self, value):
        # self.time += value
        pass

    @property
    def rising_hits(self):
        return self.hits

    @rising_hits.setter
    def rising_hits(self, value):
        # self.hits += value
        pass

    @property
    def setting_hits(self):
        return self.hits

    @setting_hits.setter
    def setting_hits(self, value):
        # self.hits += value
        pass

    def get_area(self, observer, nside=32, equalize=False):
        return 1

    def corner_coordinates(self, observer=None, unwind=False):
        azs = [self.az_min, self.az_max]
        els = [self.el_min, self.el_max]
        return np.array(azs), np.array(els)

    def in_patch(self, obj, angle=0, el_min=-90, observer=None):
        if angle == 0:
            return False
        azmin = obj.az
        azmax = obj.az
        elmin = obj.alt
        elmax = obj.alt
        if observer is not None:
            observer2 = observer.copy()
            obj2 = obj.copy()
            observer2.date += self.scantime / 1440
            obj2.compute(observer2)
            azmin = min(azmin, obj2.az)
            azmax = max(azmax, obj2.az)
            elmin = min(elmin, obj2.alt)
            elmax = max(elmax, obj2.alt)
        azmin -= angle / np.sin(obj.alt)
        azmax += angle / np.sin(obj.alt)
        elmin -= angle
        elmax += angle
        if (
            azmin > self.az_min
            and azmax < self.az_max
            and elmin > self.el_min
            and elmax < self.el_max
        ):
            return True
        return False

    def step_azel(self):
        return

    def visible(
        self,
        el_min,
        observer,
        sun,
        moon,
        sun_avoidance_angle,
        moon_avoidance_angle,
        check_sso,
    ):
        in_view = True
        msg = ""
        if check_sso:
            for sso, angle, name in [
                (sun, sun_avoidance_angle, "Sun"),
                (moon, moon_avoidance_angle, "Moon"),
            ]:
                if self.in_patch(sso, angle=angle, observer=observer):
                    in_view = False
                    msg += f"{name} too close;"

        if in_view:
            msg = "in view"
            self.current_el_min = self.el_min
            self.current_el_max = self.el_max
        return in_view, msg


def patch_is_rising(patch):
    try:
        # Horizontal patch definition
        rising = patch.rising
    except:
        rising = True
        for corner in patch.corners:
            if corner.alt > 0 and corner.az > np.pi:
                # The patch is setting
                rising = False
                break
    return rising


@function_timer
def prioritize(args, observer, visible, last_el):
    """Order visible targets by priority and number of scans."""
    log = Logger.get()
    for i in range(len(visible)):
        for j in range(len(visible) - i - 1):
            # If either of the patches is a cooler cycle, we don't modulate
            # the priorities with hit counts, observing time or elevation
            if isinstance(visible[j], CoolerCyclePatch) or isinstance(
                visible[j + 1], CoolerCyclePatch
            ):
                weight1 = visible[j].weight
                weight2 = visible[j + 1].weight
            else:
                rising1 = patch_is_rising(visible[j])
                if rising1:
                    if args.equalize_time:
                        hits1 = visible[j].rising_time
                    else:
                        hits1 = visible[j].rising_hits
                    el1 = np.degrees(visible[j].current_el_max)
                else:
                    if args.equalize_time:
                        hits1 = visible[j].setting_time
                    else:
                        hits1 = visible[j].setting_hits
                    el1 = np.degrees(visible[j].current_el_min)
                rising2 = patch_is_rising(visible[j + 1])
                if rising2:
                    if args.equalize_time:
                        hits2 = visible[j + 1].rising_time
                    else:
                        hits2 = visible[j + 1].rising_hits
                    el2 = np.degrees(visible[j + 1].current_el_max)
                else:
                    if args.equalize_time:
                        hits2 = visible[j + 1].setting_time
                    else:
                        hits2 = visible[j + 1].setting_hits
                    el2 = np.degrees(visible[j + 1].current_el_min)
                # Patch with the lower weight goes first.  Having more
                # earlier observing time and lower observing elevation
                # will increase the weight.
                weight1 = (hits1 + 1) * visible[j].weight
                weight2 = (hits2 + 1) * visible[j + 1].weight
                # Optional elevation penalty
                if args.elevation_penalty_limit > 0:
                    lim = args.elevation_penalty_limit
                    if el1 < lim:
                        weight1 *= (lim / el1) ** args.elevation_penalty_power
                    if el2 < lim:
                        weight2 *= (lim / el2) ** args.elevation_penalty_power
                # Optional elevation change penalty
                if (
                    last_el is not None
                    and args.elevation_change_limit_deg > 0
                    and args.elevation_change_penalty != 1
                ):
                    lim = args.elevation_change_limit_deg
                    last_el_deg = np.degrees(last_el)
                    # How much will the corners drift during stabilization?
                    future_observer = observer.copy()
                    future_observer.date += args.elevation_change_time_s / 86400
                    if rising1:
                        el1_drift = visible[j].el_range(future_observer, rising1)[1]
                    else:
                        el1_drift = visible[j].el_range(future_observer, rising1)[0]
                    if rising2:
                        el2_drift = visible[j + 1].el_range(future_observer, rising2)[1]
                    else:
                        el2_drift = visible[j + 1].el_range(future_observer, rising2)[0]
                    el1_drift = np.degrees(el1_drift)
                    el2_drift = np.degrees(el2_drift)
                    # No penalty is applied if it is more expedient to just
                    # wait for the patch to reach current elevation
                    if np.abs(last_el_deg - el1) > lim:
                        if (el1 - last_el_deg) * (el1_drift - last_el_deg) > 0:
                            weight1 *= args.elevation_change_penalty
                    if np.abs(last_el_deg - el2) > lim:
                        if (el2 - last_el_deg) * (el2_drift - last_el_deg) > 0:
                            weight2 *= args.elevation_change_penalty
            if weight1 > weight2:
                visible[j], visible[j + 1] = visible[j + 1], visible[j]
    names = []
    for patch in visible:
        names.append(patch.name)
    log.debug(f"Prioritized list of viewable patches: {names}")
    return


@function_timer
def attempt_scan(
    args,
    observer,
    visible,
    not_visible,
    t,
    fp_radius,
    stop_timestamp,
    tstop_cooler,
    sun,
    moon,
    sun_el_max,
    fout,
    fout_fmt,
    ods,
    boresight_angle,
    last_t,
    last_el,
):
    """Attempt scanning the visible patches in order until success."""
    log = Logger.get()
    success = False
    # Always begin by attempting full scans.  If none can be completed
    # and user allowed partials scans, try them next.
    for allow_partial_scans in False, True:
        if allow_partial_scans and not args.allow_partial_scans:
            break
        for patch in visible:
            if isinstance(patch, CoolerCyclePatch):
                # Cycle the cooler
                t = add_cooler_cycle(
                    args,
                    t,
                    stop_timestamp,
                    observer,
                    sun,
                    moon,
                    fout,
                    fout_fmt,
                    patch,
                    boresight_angle,
                )
                success = True
                break
            # All on-sky targets
            for rising in [True, False]:
                if last_el is not None and args.elevation_change_limit_deg != 0:
                    # See if we need to add extra stabilization time before
                    # observing the target due to change in elevation
                    observer.date = to_DJD(t)
                    el_min, el_max = patch.el_range(observer, rising)
                    if rising:
                        delta_el = np.degrees(np.abs(last_el - el_max))
                    else:
                        delta_el = np.degrees(np.abs(last_el - el_min))
                    if delta_el > args.elevation_change_limit_deg:
                        t_try = max(t, last_t + args.elevation_change_time_s)
                    else:
                        t_try = t
                else:
                    t_try = t
                observer.date = to_DJD(t_try)
                el = get_constant_elevation(
                    args,
                    observer,
                    patch,
                    rising,
                    fp_radius,
                    not_visible,
                    partial_scan=allow_partial_scans,
                )
                if el is None:
                    continue
                if t_try != t:
                    # Decide if we should wait at the present observing elevation
                    # instead of changing elevation and waiting for the system to
                    # stabilize

                    observer.date = to_DJD(t)
                    el_range_then = patch.el_range(observer, rising)
                    observer.date = to_DJD(t_try)
                    el_range_now = patch.el_range(observer, rising)
                    # drift is 15 degrees / hour (upper limit)
                    t_drift = delta_el / 15 * 3600
                    if rising and (
                        (el_range_then[1] - last_el) * (el_range_now[1] - last_el) < 0
                    ):
                        log.debug(
                            f"Will wait at {np.degrees(last_el)} rather than move "
                            f"to {np.degrees(el)} and stabilize"
                        )
                        el = last_el
                        t_try = t
                        observer.date = to_DJD(t_try)
                    elif not rising and (
                        (el_range_then[0] - last_el) * (el_range_now[0] - last_el) < 0
                    ):
                        log.debug(
                            f"Will wait at {np.degrees(last_el)} rather than move "
                            f"to {np.degrees(el)} and stabilize"
                        )
                        el = last_el
                        t_try = t
                    observer.date = to_DJD(t_try)
                success, azmins, azmaxs, aztimes, tstop = scan_patch(
                    args,
                    el,
                    patch,
                    t_try,
                    fp_radius,
                    observer,
                    sun,
                    not_visible,
                    tstop_cooler,
                    sun_el_max,
                    rising,
                )
                if success:
                    try:
                        t, _ = add_scan(
                            args,
                            t_try,
                            tstop,
                            aztimes,
                            azmins,
                            azmaxs,
                            rising,
                            fp_radius,
                            observer,
                            sun,
                            moon,
                            fout,
                            fout_fmt,
                            patch,
                            el,
                            ods,
                            boresight_angle,
                            partial_scan=allow_partial_scans,
                        )
                        patch.step_azel()
                        break
                    except TooClose:
                        success = False
                        break
            if success:
                break
        if success:
            break
    return success, t, el


def from_angles(az, el):
    elquat = qa.rotation(YAXIS, np.radians(90 - el))
    azquat = qa.rotation(ZAXIS, np.radians(az))
    return qa.mult(azquat, elquat)


def unwind_quat(quat1, quat2):
    if np.sum(np.abs(quat1 - quat2)) > np.sum(np.abs(quat1 + quat2)):
        return -quat2
    else:
        return quat2


@function_timer
def check_sso(observer, az1, az2, el, sso, angle, el_min, tstart, tstop):
    """
    Check if a solar system object (SSO) enters within "angle" of
    the constant elevation scan.
    """
    if az2 < az1:
        az2 += 360
    naz = max(3, int(0.25 * (az2 - az1) * np.cos(np.radians(el))))
    quats = []
    for az in np.linspace(az1, az2, naz):
        quats.append(from_angles(az % 360, el))
    vecs = qa.rotate(quats, ZAXIS)

    tstart = to_DJD(tstart)
    tstop = to_DJD(tstop)
    t1 = tstart
    # Test every ten minutes
    tstep = 10 / 1440
    while t1 < tstop:
        t2 = min(tstop, t1 + tstep)
        observer.date = t1
        sso.compute(observer)
        az1, el1 = np.degrees(sso.az), np.degrees(sso.alt)
        observer.date = t2
        sso.compute(observer)
        az2, el2 = np.degrees(sso.az), np.degrees(sso.alt)
        # Only test distance if the SSO is high enough
        if el1 > el_min and el2 > el_min:
            quat1 = from_angles(az1, el1)
            quat2 = from_angles(az2, el2)
            quat2 = unwind_quat(quat1, quat2)
            t = np.linspace(0, 1, 10)
            quats = qa.slerp(t, [0, 1], [quat1, quat2])
            sso_vecs = qa.rotate(quats, ZAXIS).T
            dpmax = np.amax(np.dot(vecs, sso_vecs))
            min_dist = np.degrees(np.arccos(dpmax))
            if min_dist < angle:
                return True, DJDtoUNIX(t1)
        t1 = t2
    return False, DJDtoUNIX(t2)


@function_timer
def attempt_scan_pole(
    args,
    observer,
    visible,
    not_visible,
    tstart,
    fp_radius,
    el_max,
    el_min,
    stop_timestamp,
    tstop_cooler,
    sun,
    moon,
    sun_el_max,
    fout,
    fout_fmt,
    ods,
    boresight_angle,
):
    """Attempt scanning the visible patches in order until success."""
    if args.one_scan_per_day and stop_timestamp > tstop_cooler:
        raise RuntimeError("one_scan_per_day is incompatible with cooler cycles")
    success = False
    for patch in visible:
        observer.date = to_DJD(tstart)
        if isinstance(patch, CoolerCyclePatch):
            # Cycle the cooler
            t = add_cooler_cycle(
                args,
                tstart,
                stop_timestamp,
                observer,
                sun,
                moon,
                fout,
                fout_fmt,
                patch,
                boresight_angle,
            )
            success = True
            break
        # In pole scheduling, first elevation is just below the patch
        el = get_constant_elevation_pole(
            args, observer, patch, fp_radius, el_min, el_max, not_visible
        )
        if el is None:
            continue
        pole_success = True
        subscan = -1
        t = tstart
        while pole_success:
            (pole_success, azmins, azmaxs, aztimes, tstop) = scan_patch_pole(
                args,
                el,
                patch,
                t,
                fp_radius,
                observer,
                sun,
                not_visible,
                tstop_cooler,
                sun_el_max,
            )
            if pole_success:
                if success:
                    # Still the same scan
                    patch.hits -= 1
                    patch.rising_hits -= 1
                    patch.setting_hits -= 1
                try:
                    t, subscan = add_scan(
                        args,
                        t,
                        tstop,
                        aztimes,
                        azmins,
                        azmaxs,
                        False,
                        fp_radius,
                        observer,
                        sun,
                        moon,
                        fout,
                        fout_fmt,
                        patch,
                        el,
                        ods,
                        boresight_angle,
                        subscan=subscan,
                    )
                    el += np.radians(args.pole_el_step_deg)
                    success = True
                except TooClose:
                    success = False
                    pole_success = False
        if success:
            el -= np.radians(args.pole_el_step_deg)
            break
    tstop = t
    if args.one_scan_per_day:
        day1 = int(to_MJD(tstart))
        while int(to_MJD(tstop)) == day1:
            tstop += 60.0
    return success, tstop, el


@function_timer
def get_constant_elevation(
    args, observer, patch, rising, fp_radius, not_visible, partial_scan=False
):
    """Determine the elevation at which to scan."""
    log = Logger.get()

    azs, els = patch.corner_coordinates(observer)

    ind_rising = azs < np.pi
    ind_setting = azs > np.pi

    el = None
    if rising:
        if np.sum(ind_rising) == 0:
            not_visible.append((patch.name, "No rising corners"))
        else:
            el = np.amax(els[ind_rising]) + fp_radius
    else:
        if np.sum(ind_setting) == 0:
            not_visible.append((patch.name, "No setting corners"))
        else:
            el = np.amin(els[ind_setting]) - fp_radius

    # Check that the elevation is valid

    if el is not None and patch.elevations is not None:
        # Fixed elevation mode.  Find the first allowed observing elevation.
        if rising:
            ind = patch.elevations >= el
            if np.any(ind):
                el = np.amin(patch.elevations[ind])
            elif partial_scan:
                # None of the elevations allow a full rising scan,
                # Observe at the highest allowed elevation
                el = np.amax(patch.elevations)
                if el < np.amin(els[ind_rising]) + fp_radius:
                    not_visible.append(
                        (patch.name, "Rising patch above maximum elevation")
                    )
                    el = None
            else:
                not_visible.append((patch.name, "Only partial rising scans available"))
                el = None
        else:
            ind = patch.elevations <= el
            if np.any(ind):
                el = np.amax(patch.elevations[ind])
            elif partial_scan:
                # None of the elevations allow a full setting scan,
                # Observe at the lowest allowed elevation
                el = np.amin(patch.elevations)
                if el > np.amax(els[ind_setting]) + fp_radius:
                    not_visible.append(
                        (patch.name, "Setting patch above below elevation")
                    )
                    el = None
            else:
                not_visible.append((patch.name, "Only partial setting scans available"))
                el = None
    elif el is not None:
        if el < patch.el_min:
            if partial_scan and np.any(patch.el_min < els[ind_setting] - fp_radius):
                # Partial setting scan
                el = patch.el_min
            else:
                not_visible.append(
                    (
                        patch.name,
                        "el < el_min ({:.2f} < {:.2f}) rising = {}, partial = {}".format(
                            el / degree, patch.el_min / degree, rising, partial_scan
                        ),
                    )
                )
                el = None
        elif el > patch.el_max:
            if partial_scan and np.any(patch.el_max > els[ind_rising] + fp_radius):
                # Partial rising scan
                el = patch.el_max
            else:
                not_visible.append(
                    (
                        patch.name,
                        "el > el_max ({:.2f} > {:.2f}) rising = {}, partial = {}".format(
                            el / degree, patch.el_max / degree, rising, partial_scan
                        ),
                    )
                )
                el = None
    if el is None:
        log.debug(f"NO ELEVATION: {not_visible[-1]}")
    else:
        log.debug(
            "{} : ELEVATION = {}, rising = {}, partial = {}".format(
                patch.name, el / degree, rising, partial_scan
            )
        )
    return el


@function_timer
def get_constant_elevation_pole(
    args, observer, patch, fp_radius, el_min, el_max, not_visible
):
    """Determine the elevation at which to scan."""
    log = Logger.get()
    _, els = patch.corner_coordinates(observer)
    el = np.amin(els) - fp_radius

    if el < el_min:
        not_visible.append(
            (
                patch.name,
                "el < el_min ({:.2f} < {:.2f})".format(el / degree, el_min / degree),
            )
        )
        el = None
    elif el > el_max:
        not_visible.append(
            (
                patch.name,
                "el > el_max ({:.2f} > {:.2f})".format(el / degree, el_max / degree),
            )
        )
        el = None
    if el is None:
        log.debug(f"NOT VISIBLE: {not_visible[-1]}")
    return el


def check_sun_el(t, observer, sun, sun_el_max, args, not_visible):
    log = Logger.get()
    observer.date = to_DJD(t)
    if sun_el_max < np.pi / 2:
        sun.compute(observer)
        if sun.alt > sun_el_max:
            not_visible.append(
                (
                    patch.name,
                    "Sun too high {:.2f} rising = {}"
                    "".format(np.degrees(sun.alt), rising),
                )
            )
            log.debug(f"NOT VISIBLE: {not_visible[-1]}")
            return True
    return False


@function_timer
def scan_patch(
    args,
    el,
    patch,
    t,
    fp_radius,
    observer,
    sun,
    not_visible,
    stop_timestamp,
    sun_el_max,
    rising,
):
    """Attempt scanning the patch specified by corners at elevation el."""
    log = Logger.get()
    azmins, azmaxs, aztimes = [], [], []
    if isinstance(patch, HorizontalPatch):
        # No corners.  Simply scan for the requested time
        if rising and not patch.rising:
            return False, azmins, azmaxs, aztimes, t
        if check_sun_el(t, observer, sun, sun_el_max, args, not_visible):
            return False, azmins, azmaxs, aztimes, t
        azmins = [patch.az_min]
        azmaxs = [patch.az_max]
        aztimes = [t]
        return True, azmins, azmaxs, aztimes, t + patch.scantime * 60
    # Traditional patch, track each corner
    success = False
    # and now track when all corners are past the elevation
    tstop = t
    tstep = 60
    to_cross = np.ones(len(patch.corners), dtype=bool)
    scan_started = False
    while True:
        if tstop > stop_timestamp or tstop - t > 86400:
            not_visible.append((patch.name, f"Ran out of time rising = {rising}"))
            log.debug(f"NOT VISIBLE: {not_visible[-1]}")
            break
        if check_sun_el(tstop, observer, sun, sun_el_max, args, not_visible):
            break
        azs, els = patch.corner_coordinates(observer)
        has_extent = current_extent(
            azmins,
            azmaxs,
            aztimes,
            patch.corners,
            fp_radius,
            el,
            azs,
            els,
            rising,
            tstop,
            to_cross,
        )
        if has_extent:
            scan_started = True

        if rising:
            good = azs <= np.pi
            to_cross[np.logical_and(els > el + fp_radius, good)] = False
        else:
            good = azs >= np.pi
            to_cross[np.logical_and(els < el - fp_radius, good)] = False

        # If we are alternating rising and setting scans, reject patches
        # that appear on the wrong side of the sky.
        if np.any((np.array(azmins) % (2 * np.pi)) < patch.az_min) or np.any(
            (np.array(azmaxs) % (2 * np.pi)) > patch.az_max
        ):
            success = False
            break

        if len(aztimes) > 0 and not np.any(to_cross):
            # All corners made it across the CES line.
            success = True
            # Begin the scan before the patch is at the CES line
            if aztimes[0] > t:
                aztimes[0] -= tstep
            break

        if scan_started and not has_extent:
            # The patch went out of view before all corners
            # could cross the elevation line.
            success = False
            break
        tstop += tstep

    return success, azmins, azmaxs, aztimes, tstop


def unwind_angle(alpha, beta, multiple=2 * np.pi):
    """Minimize absolute difference between alpha and beta.

    Minimize the absolute difference by adding a multiple of
    2*pi to beta to match alpha.
    """
    while np.abs(alpha - beta - multiple) < np.abs(alpha - beta):
        beta += multiple
    while np.abs(alpha - beta + multiple) < np.abs(alpha - beta):
        beta -= multiple
    return beta


@function_timer
def get_pole_raster_scan(
    args,
    el_start,
    t_start,
    patch,
    fp_radius,
    observer,
):
    """Determine the time it takes to perform Az-locked scanning with
    elevation steps after each half scan pair.
    """
    el_stop = el_start + np.radians(args.pole_el_step_deg)
    el_step = np.radians(args.pole_raster_el_step_deg)
    az_rate_sky = np.radians(args.az_rate_sky_deg)
    az_accel_mount = np.radians(args.az_accel_mount_deg)
    el_rate = np.radians(args.el_rate_deg)
    el_accel = np.radians(args.el_accel_deg)

    # Time it takes to perform one elevation step
    t_accel_el = el_rate / el_accel  # acceleration time
    if el_accel * t_accel_el**2 > el_step:
        # Telescope does not reach constant el_rate during the step.
        # The step is made of acceleration and deceleration
        t_el_step = 2 * np.sqrt(el_step / el_accel)
    else:
        # length of constant elevation rate scan
        el_scan = el_step - el_accel * t_accel**2
        # The elevation step is made of acceleration,
        # constant scan and deceleration
        t_el_step = 2 * t_accel_el + el_scan / el_rate

    nstep = int((el_stop - el_start) / el_step)
    el = el_start
    t_stop = t_start
    radius = max(np.radians(1), fp_radius)
    azmins, azmaxs, aztimes = [], [], []
    elevations = []
    az_ranges = []
    scan_started = False
    for istep in range(nstep):
        observer.date = to_DJD(t_stop)
        azs, els = patch.corner_coordinates(observer)
        has_extent = current_extent_pole(
            azmins, azmaxs, aztimes, patch.corners, radius, el, azs, els, t_stop
        )
        if has_extent:
            scan_started = True
        elif scan_started:
            nstep = istep + 1
            break
        # Get time to scan the half scan pair, including turnarounds
        if has_extent:
            az_range = azmaxs[-1] - azmins[-1]
        else:
            az_range = 0
        az_ranges.append(az_range)
        elevations.append(el)
        scan_time = np.cos(el) * az_range / az_rate_sky
        az_rate_mount = az_rate_sky / np.cos(el)
        turnaround_time = az_rate_mount / az_accel_mount * 2
        t_stop += 2 * scan_time + 2 * turnaround_time
        if istep < nstep - 1:
            el += el_step
            t_stop += t_el_step

    # Now extend the half scans so they are azimuth-locked
    azmin = np.amin(azmins)
    azmax = np.amax(azmaxs)
    az_range_full = azmax - azmin
    # t_old = t_stop
    for az_range, el in zip(az_ranges, elevations):
        delta = az_range_full - az_range
        t_stop += np.cos(el) * delta / az_rate_sky * 2

    # FIXME : the additional time needed to extend the scans
    # adds a small amount of sky rotation.  We could extend the
    # azimuthal range to counter the sky rotation and increase
    # the scanning time accordingly but the effect is small.

    return t_stop - t_start, azmin, azmax


@function_timer
def scan_patch_pole(
    args,
    el,
    patch,
    t,
    fp_radius,
    observer,
    sun,
    not_visible,
    stop_timestamp,
    sun_el_max,
):
    """Attempt scanning the patch specified by corners at elevation el.

    The pole scheduling mode will not wait for the patch to drift across.
    It simply attempts to scan for the required time: args.pole_ces_time.
    """
    log = Logger.get()
    success = False
    tstop = t
    tstep = 60
    azmins, azmaxs, aztimes = [], [], []
    observer.date = to_DJD(t)
    azs, els = patch.corner_coordinates(observer)
    # Check if el is above the target.  If so, there is nothing to do.
    if np.amax(els) + fp_radius < el:
        not_visible.append((patch.name, "Patch below {:.2f}".format(el / degree)))
        log.debug(f"NOT VISIBLE: {not_visible[-1]}")
    else:
        if args.pole_raster_scan:
            scan_time, azmin_raster, azmax_raster = get_pole_raster_scan(
                args, el, t, patch, fp_radius, observer
            )
        else:
            scan_time = args.pole_ces_time_s
        while True:
            if tstop - t > scan_time - 1:
                # Succesfully scanned the maximum time
                if len(azmins) > 0:
                    success = True
                else:
                    not_visible.append(
                        (patch.name, "No overlap at {:.2f}".format(el / degree))
                    )
                    log.debug(f"NOT VISIBLE: {not_visible[-1]}")
                break
            if tstop > stop_timestamp or tstop - t > 86400:
                not_visible.append((patch.name, "Ran out of time"))
                log.debug(f"NOT VISIBLE: {not_visible[-1]}")
                break
            observer.date = to_DJD(tstop)
            sun.compute(observer)
            if sun.alt > sun_el_max:
                not_visible.append(
                    (patch.name, "Sun too high {:.2f}".format(sun.alt / degree))
                )
                log.debug(f"NOT VISIBLE: {not_visible[-1]}")
                break
            azs, els = patch.corner_coordinates(observer)
            radius = max(np.radians(1), fp_radius)
            current_extent_pole(
                azmins, azmaxs, aztimes, patch.corners, radius, el, azs, els, tstop
            )
            tstop += tstep
        if args.pole_raster_scan:
            azmins = np.ones(len(aztimes)) * azmin_raster
            azmaxs = np.ones(len(aztimes)) * azmax_raster
    return success, azmins, azmaxs, aztimes, tstop


@function_timer
def current_extent_pole(
    azmins, azmaxs, aztimes, corners, fp_radius, el, azs, els, tstop
):
    """Get the azimuthal extent of the patch along elevation el.

    Pole scheduling does not care if the patch is "rising" or "setting".
    """
    azs_cross = []
    for i in range(len(corners)):
        if np.abs(els[i] - el) < fp_radius:
            azs_cross.append(azs[i])
        j = (i + 1) % len(corners)
        if np.abs(els[j] - el) < fp_radius:
            azs_cross.append(azs[j])
        if np.abs(els[i] - el) < fp_radius or np.abs(els[j] - el) < fp_radius:
            continue
        elif (els[i] - el) * (els[j] - el) < 0:
            # Record the location where a line between the corners
            # crosses el.
            az1 = azs[i]
            az2 = azs[j]
            el1 = els[i] - el
            el2 = els[j] - el
            if az2 - az1 > np.pi:
                az1 += 2 * np.pi
            if az1 - az2 > np.pi:
                az2 += 2 * np.pi
            az_cross = (az1 + el1 * (az2 - az1) / (el1 - el2)) % (2 * np.pi)
            azs_cross.append(az_cross)

    # Translate the azimuths at multiples of 2pi so they are in a
    # compact cluster

    for i in range(1, len(azs_cross)):
        azs_cross[i] = unwind_angle(azs_cross[0], azs_cross[i])

    if len(azs_cross) > 0:
        azs_cross = np.sort(azs_cross)
        azmin = azs_cross[0]
        azmax = azs_cross[-1]
        azmax = unwind_angle(azmin, azmax)
        if azmax - azmin > np.pi:
            # Patch crosses the zero meridian
            azmin, azmax = azmax, azmin
        if len(azmins) > 0:
            azmin = unwind_angle(azmins[-1], azmin)
            azmax = unwind_angle(azmaxs[-1], azmax)
        azmins.append(azmin)
        azmaxs.append(azmax)
        aztimes.append(tstop)
        has_extent = True
    else:
        has_extent = False
    return has_extent


@function_timer
def current_extent(
    azmins, azmaxs, aztimes, corners, fp_radius, el, azs, els, rising, t, to_cross
):
    """Get the azimuthal extent of the patch along elevation el.

    Find the pairs of corners that are on opposite sides
    of the CES line.  Record the crossing azimuth of a
    line between the corners.

    """
    azs_cross = []
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        if not to_cross[i] and not to_cross[j]:
            # Both of the corners already crossed the elevation line
            continue
        for el0 in [el - fp_radius, el, el + fp_radius]:
            if (els[i] - el0) * (els[j] - el0) < 0:
                # The corners are on opposite sides of the elevation line
                az1 = azs[i]
                az2 = azs[j]
                el1 = els[i] - el0
                el2 = els[j] - el0
                az2 = unwind_angle(az1, az2)
                az_cross = (az1 + el1 * (az2 - az1) / (el1 - el2)) % (2 * np.pi)
                if rising and az_cross > np.pi:
                    continue
                if not rising and az_cross < np.pi:
                    continue
                azs_cross.append(az_cross)
            if fp_radius == 0:
                break
    if len(azs_cross) == 0:
        return False

    azs_cross = np.array(azs_cross)
    if rising:
        good = azs_cross < np.pi
    else:
        good = azs_cross > np.pi
    ngood = np.sum(good)
    if ngood == 0:
        return False
    elif ngood > 1:
        azs_cross = azs_cross[good]

    # Unwind the crossing azimuths to minimize the scatter
    azs_cross = np.sort(azs_cross)
    if azs_cross.size > 1:
        ptp0 = azs_cross[-1] - azs_cross[0]
        ptps = azs_cross[:-1] + 2 * np.pi - azs_cross[1:]
        ptps = np.hstack([ptp0, ptps])
        i = np.argmin(ptps)
        azs_cross[:i] += 2 * np.pi
        np.roll(azs_cross, i)

    if len(azs_cross) > 1:
        azmin = azs_cross[0] % (2 * np.pi)
        azmax = azs_cross[-1] % (2 * np.pi)
        if azmax - azmin > np.pi:
            # Patch crosses the zero meridian
            azmin, azmax = azmax, azmin
        azmins.append(azmin)
        azmaxs.append(azmax)
        aztimes.append(t)
        return True
    return False


@function_timer
def add_scan(
    args,
    tstart,
    tstop,
    aztimes,
    azmins,
    azmaxs,
    rising,
    fp_radius,
    observer,
    sun,
    moon,
    fout,
    fout_fmt,
    patch,
    el,
    ods,
    boresight_angle,
    subscan=-1,
    partial_scan=False,
):
    """Make an entry for a CES in the schedule file."""
    log = Logger.get()
    ces_time = tstop - tstart
    if ces_time > args.ces_max_time_s:  # and not args.pole_mode:
        nsub = int(np.ceil(ces_time / args.ces_max_time_s))
        ces_time /= nsub
    aztimes = np.array(aztimes)
    azmins = np.array(azmins)
    azmaxs = np.array(azmaxs)
    azmaxs[0] = unwind_angle(azmins[0], azmaxs[0])
    for i in range(1, azmins.size):
        azmins[i] = unwind_angle(azmins[0], azmins[i])
        azmaxs[i] = unwind_angle(azmaxs[0], azmaxs[i])
        azmaxs[i] = unwind_angle(azmins[i], azmaxs[i])
    if args.lock_az_range:
        # Hold the azimuth range fixed across split observations
        azmins[:] = np.amin(azmins)
        azmaxs[:] = np.amax(azmaxs)
    # for i in range(azmins.size-1):
    #    if azmins[i+1] - azmins[i] > np.pi:
    #        azmins[i+1], azmaxs[i+1] = azmins[i+1]-2*np.pi, azmaxs[i+1]-2*np.pi
    #    if azmins[i+1] - azmins[i] < np.pi:
    #        azmins[i+1], azmaxs[i+1] = azmins[i+1]+2*np.pi, azmaxs[i+1]+2*np.pi
    rising_string = "R" if rising else "S"
    t1 = np.amin(aztimes)
    entries = []
    obs_time = 0  # Actual integration time
    while t1 < tstop - 1:
        subscan += 1
        if args.operational_days:
            # See if adding this scan would exceed the number of desired
            # operational days
            if subscan == 0:
                tz = args.timezone / 24
                od = int(to_MJD(tstart) + tz)
                ods.add(od)
            if len(ods) > args.operational_days:
                # Prevent adding further entries to the schedule once
                # the number of operational days is full
                break
        t2 = min(t1 + ces_time, tstop)
        if tstop - t2 < ces_time / 10:
            # Append leftover scan to the last full subscan
            t2 = tstop
        ind = np.logical_and(aztimes >= t1, aztimes <= t2)
        if np.all(aztimes > t2):
            ind[0] = True
        if np.all(aztimes < t1):
            ind[-1] = True
        if azmins[ind][0] < azmaxs[ind][0]:
            azmin = np.amin(azmins[ind])
            azmax = np.amax(azmaxs[ind])
        else:
            # we are, scan from the maximum to the minimum
            azmin = np.amax(azmins[ind])
            azmax = np.amin(azmaxs[ind])
        if args.scan_margin > 0:
            # Add a random error to the scan parameters to smooth out
            # caustics in the hit map
            delta_az = azmax - unwind_angle(azmax, azmin)
            sub_az = delta_az * np.abs(np.random.randn()) * args.scan_margin * 0.5
            add_az = delta_az * np.abs(np.random.randn()) * args.scan_margin * 0.5
            azmin = (azmin - sub_az) % (2 * np.pi)
            azmax = (azmax + add_az) % (2 * np.pi)
            if t2 == tstop:
                delta_t = t2 - t1  # tstop - tstart
                add_t = delta_t * np.abs(np.random.randn()) * args.scan_margin
                t2 += add_t
        # Add the focal plane radius to the scan width
        fp_radius_eff = fp_radius / np.cos(el)
        azmin = np.degrees((azmin - fp_radius_eff) % (2 * np.pi))
        azmax = np.degrees((azmax + fp_radius_eff) % (2 * np.pi))
        # Get the Sun and Moon locations at the beginning and end
        observer.date = to_DJD(t1)
        sun.compute(observer)
        moon.compute(observer)
        sun_az1, sun_el1 = np.degrees(sun.az), np.degrees(sun.alt)
        moon_az1, moon_el1 = np.degrees(moon.az), np.degrees(moon.alt)
        moon_phase1 = moon.phase
        # It is possible that the Sun or the Moon gets too close to the
        # scan, even if they are far enough from the actual patch.
        sun_too_close, sun_time = check_sso(
            observer,
            azmin,
            azmax,
            np.degrees(el),
            sun,
            args.sun_avoidance_angle_deg,
            args.sun_avoidance_altitude_deg,
            t1,
            t2,
        )
        moon_too_close, moon_time = check_sso(
            observer,
            azmin,
            azmax,
            np.degrees(el),
            moon,
            args.moon_avoidance_angle_deg,
            args.moon_avoidance_altitude_deg,
            t1,
            t2,
        )

        if (
            (isinstance(patch, HorizontalPatch) or partial_scan)
            and sun_time > tstart + 1
            and moon_time > tstart + 1
        ):
            # Simply terminate the scan when the Sun or the Moon is too close
            t2 = min(sun_time, moon_time)
            if sun_too_close or moon_too_close:
                tstop = t2
                if t1 == t2:
                    break
        else:
            # For regular patches, this is a failure condition
            if sun_too_close:
                log.debug("Sun too close")
                raise SunTooClose
            if moon_too_close:
                log.debug("Moon too close")
                raise MoonTooClose

        # Do not schedule observations shorter than a second
        too_short = t2 - t1 < 1

        if not too_short:
            observer.date = to_DJD(t2)
            sun.compute(observer)
            moon.compute(observer)
            sun_az2, sun_el2 = sun.az / degree, sun.alt / degree
            moon_az2, moon_el2 = moon.az / degree, moon.alt / degree
            moon_phase2 = moon.phase
            # optionally offset scan
            if args.boresight_offset_az_deg != 0 or args.boresight_offset_el_deg != 0:
                az_offset = np.radians(args.boresight_offset_az_deg)
                el_offset = np.radians(args.boresight_offset_el_deg)

                az_offset_rot = qa.rotation(ZAXIS, az_offset)
                el_offset_rot = qa.rotation(YAXIS, el_offset)
                offset_rot = qa.mult(az_offset_rot, el_offset_rot)
                offset_vec = qa.rotate(offset_rot, XAXIS)

                az_min_rot = qa.rotation(ZAXIS, np.radians(azmin))
                az_max_rot = qa.rotation(ZAXIS, np.radians(azmax))
                el_rot = qa.rotation(YAXIS, -el)
                min_rot = qa.mult(az_min_rot, el_rot)

                vec_min = qa.rotate(min_rot, offset_vec)

                az_min, el_min = hp.vec2dir(vec_min, lonlat=True)
                # Choose the right branch of Azimuth
                if az_min < 0:
                    az_min += 360

                el_offset = np.degrees(el) - el_min
                el_observe = np.degrees(el) + el_offset

                az_offset = (
                    (az_min - azmin) * np.cos(el) / np.cos(np.radians(el_observe))
                )
                azmin += az_offset
                azmax += az_offset
            else:
                el_observe = np.degrees(el)
            # ensure azimuth is increasing
            azmin = azmin % 360
            azmax = azmax % 360
            if azmax < azmin:
                azmax += 360
            # Accumulate observing time (will not include gaps)
            obs_time += t2 - t1
            # Create an entry in the schedule
            if args.verbose_schedule:
                entry = fout_fmt.format(
                    to_UTC(t1),
                    to_UTC(t2),
                    to_MJD(t1),
                    to_MJD(t2),
                    boresight_angle,
                    patch.name,
                    azmin,
                    azmax,
                    el_observe,
                    rising_string,
                    sun_el1,
                    sun_az1,
                    sun_el2,
                    sun_az2,
                    moon_el1,
                    moon_az1,
                    moon_el2,
                    moon_az2,
                    0.005 * (moon_phase1 + moon_phase2),
                    -1 - patch.partial_hits if partial_scan else patch.hits,
                    subscan,
                    (patch.time + obs_time) / 86400,
                )
            else:
                entry = fout_fmt.format(
                    to_UTC(t1),
                    to_UTC(t2),
                    boresight_angle,
                    patch.name,
                    azmin,
                    azmax,
                    el_observe,
                    -1 - patch.partial_hits if partial_scan else patch.hits,
                    subscan,
                )
            entries.append(entry)

        if too_short or sun_too_close or moon_too_close or partial_scan:
            # Never append more than one partial scan before
            # checking if full scans are again available
            tstop = t2
            break

        t1 = t2 + args.gap_small_s

    # Write the entries
    for entry in entries:
        log.debug(entry)
        fout.write(entry)
    fout.flush()

    if not partial_scan:
        # Only update the patch counters when performing full scans
        patch.hits += 1
        patch.time += obs_time
        if rising or args.pole_mode:
            patch.rising_hits += 1
            patch.rising_time += obs_time
        if not rising or args.pole_mode:
            patch.setting_hits += 1
            patch.setting_time += obs_time
        # The oscillate method will slightly shift the patch to
        # blur the boundaries
        patch.oscillate()
        # Advance the time
        tstop += args.gap_s
    else:
        patch.partial_hits += 1
        # Advance the time
        tstop += args.gap_small_s

    return tstop, subscan


@function_timer
def add_cooler_cycle(
    args, tstart, tstop, observer, sun, moon, fout, fout_fmt, patch, boresight_angle
):
    """Make an entry for a cooler cycle in the schedule file."""
    log = Logger.get()
    az = patch.az
    el = patch.el
    t1 = tstart
    t2 = t1 + patch.cycle_time

    observer.date = to_DJD(t1)
    sun.compute(observer)
    moon.compute(observer)
    sun_az1, sun_el1 = sun.az / degree, sun.alt / degree
    moon_az1, moon_el1 = moon.az / degree, moon.alt / degree
    moon_phase1 = moon.phase

    observer.date = to_DJD(t2)
    sun.compute(observer)
    moon.compute(observer)
    sun_az2, sun_el2 = sun.az / degree, sun.alt / degree
    moon_az2, moon_el2 = moon.az / degree, moon.alt / degree
    moon_phase2 = moon.phase

    # Create an entry in the schedule
    entry = fout_fmt.format(
        to_UTC(t1),
        to_UTC(t2),
        to_MJD(t1),
        to_MJD(t2),
        boresight_angle,
        patch.name,
        az,
        az,
        el,
        "R",
        sun_el1,
        sun_az1,
        sun_el2,
        sun_az2,
        moon_el1,
        moon_az1,
        moon_el2,
        moon_az2,
        0.005 * (moon_phase1 + moon_phase2),
        patch.hits,
        0,
    )

    # Write the entry
    log.debug(entry)
    fout.write(entry)
    fout.flush()

    patch.last_cycle_end = t2
    patch.hits += 1
    patch.time += t2 - t1
    patch.rising_hits += 1
    patch.rising_time += t2 - t1
    patch.setting_hits += 1
    patch.setting_time += t2 - t1

    return t2


@function_timer
def get_visible(args, observer, sun, moon, patches, el_min):
    """Determine which patches are visible."""
    log = Logger.get()
    visible = []
    not_visible = []
    check_sun = np.degrees(sun.alt) >= args.sun_avoidance_altitude_deg
    check_moon = np.degrees(moon.alt) >= args.moon_avoidance_altitude_deg
    for patch in patches:
        # Reject all patches that have even one corner too close
        # to the Sun or the Moon and patches that are completely
        # below the horizon
        in_view, msg = patch.visible(
            el_min,
            observer,
            sun,
            moon,
            args.sun_avoidance_angle_deg * check_sun,
            args.moon_avoidance_angle_deg * check_moon,
            not (args.allow_partial_scans or args.delay_sso_check),
        )
        if not in_view:
            not_visible.append((patch.name, msg))

        if in_view:
            if not (args.allow_partial_scans or args.delay_sso_check):
                # Finally, check that the Sun or the Moon are not
                # inside the patch
                if (
                    args.sun_avoidance_angle_deg >= 0
                    and np.degrees(sun.alt) >= args.sun_avoidance_altitude_deg
                    and patch.in_patch(sun)
                ):
                    not_visible.append((patch.name, "Sun in patch"))
                    in_view = False
                if (
                    args.moon_avoidance_angle_deg >= 0
                    and np.degrees(moon.alt) >= args.moon_avoidance_altitude_deg
                    and patch.in_patch(moon)
                ):
                    not_visible.append((patch.name, "Moon in patch"))
                    in_view = False
        if in_view:
            visible.append(patch)
            log.debug(
                "In view: {}. el = {:.2f}..{:.2f}".format(
                    patch.name, np.degrees(patch.el_min), np.degrees(patch.el_max)
                )
            )
        else:
            log.debug(f"NOT VISIBLE: {not_visible[-1]}")
    return visible, not_visible


@function_timer
def get_boresight_angle(args, t, t0=0):
    """Return the scheduled boresight angle at time t."""
    if args.boresight_angle_step_deg == 0 or args.boresight_angle_time_min == 0:
        return 0

    nstep = int(
        np.round(
            (args.boresight_angle_max_deg - args.boresight_angle_min_deg)
            / args.boresight_angle_step_deg
        )
    )
    if (args.boresight_angle_min_deg % 360) != (args.boresight_angle_max_deg % 360):
        # The range does not wrap around.
        # Include both ends of the range as separate steps
        nstep += 1
    istep = int((t - t0) / 60 / args.boresight_angle_time_min) % nstep
    angle = args.boresight_angle_min_deg + istep * args.boresight_angle_step_deg
    return angle


@function_timer
def apply_blockouts(args, t_in):
    """Check if `t` is inside a blockout period.
    If so, advance it to the next unblocked time.

    Returns:  The (new) time and a boolean flag indicating if
        the time was blocked and subsequently advanced.
    """
    if not args.block_out:
        return t_in, False
    log = Logger.get()
    t = t_in
    blocked = False
    for block_out in args.block_out:
        current = datetime.fromtimestamp(t, timezone.utc)
        start, stop = block_out.split("-")
        try:
            # If the block out specifies the year then no extra logic is needed
            start_year, start_month, start_day = start.split("/")
            start = datetime(
                int(start_year),
                int(start_month),
                int(start_day),
                0,
                0,
                0,
                0,
                timezone.utc,
            )
        except ValueError:
            # No year given so must figure out which year is the right one
            start_month, start_day = start.split("/")
            start = datetime(
                current.year, int(start_month), int(start_day), 0, 0, 0, 0, timezone.utc
            )
            if start > current:
                # This year's block out is still in the future but the past
                # year's blockout may still be active
                start = start.replace(year=start.year - 1)
        try:
            # If the block out specifies the year then no extra logic is needed
            stop_year, stop_month, stop_day = stop.split("/")
            stop = datetime(
                int(stop_year), int(stop_month), int(stop_day), 0, 0, 0, 0, timezone.utc
            )
        except ValueError:
            # No year given so must figure out which year is the right one
            stop_month, stop_day = stop.split("/")
            stop = datetime(
                start.year, int(stop_month), int(stop_day), 0, 0, 0, 0, timezone.utc
            )
            if stop < start:
                # The block out ends on a different year than it starts
                stop = stop.replace(year=start.year + 1)
        # advance the stop time by one day to make the definition inclusive
        stop += timedelta(days=1)
        if start < current and current < stop:
            # `t` is inside the block out.
            # Advance to the end of the block out.
            log.info(f"{current} is inside block out {block_out}, advancing to {stop}")
            t = stop.timestamp()
            blocked = True
    return t, blocked


def advance_time(t, time_step, offset=0):
    """Advance the time ensuring that the sampling falls
    over same discrete times (multiples of time_step)
    regardless of the current value of t.
    """
    return offset + ((t - offset) // time_step + 1) * time_step


@function_timer
def build_schedule(args, start_timestamp, stop_timestamp, patches, observer, sun, moon):
    log = Logger.get()

    sun_el_max = args.sun_el_max_deg * degree
    el_min = args.el_min_deg
    el_max = args.el_max_deg
    if args.elevations_deg is None:
        el_min = args.el_min_deg
        el_max = args.el_max_deg
    else:
        # Override the elevation limits
        el_min = 90
        el_max = 0
        for el in args.elevations_deg.split(","):
            el = float(el)
            el_min = min(el * 0.9, el_min)
            el_max = max(el * 1.1, el_max)
    el_min *= degree
    el_max *= degree
    fp_radius = args.fp_radius_deg * degree

    fname_out = args.out
    dir_out = os.path.dirname(fname_out)
    if dir_out:
        log.info(f"Creating '{dir_out}'")
        os.makedirs(dir_out, exist_ok=True)
    fout = open(fname_out, "w")

    fout.write(
        "#{:15} {:15} {:>15} {:>15} {:>15}\n".format(
            "Site", "Telescope", "Latitude [deg]", "Longitude [deg]", "Elevation [m]"
        )
    )
    fout.write(
        " {:15} {:15} {:15.3f} {:15.3f} {:15.1f}\n".format(
            args.site_name,
            args.telescope,
            np.degrees(observer.lat),
            np.degrees(observer.lon),
            observer.elevation,
        )
    )

    if args.verbose_schedule:
        fout_fmt0 = (
            "#{:>20} {:>20} {:>14} {:>14} {:>8} "
            "{:35} {:>8} {:>8} {:>8} {:>5} "
            "{:>8} {:>8} {:>8} {:>8} "
            "{:>8} {:>8} {:>8} {:>8} {:>5} "
            "{:>5} {:>3} {:>8}\n"
        )

        fout_fmt = (
            " {:20} {:20} {:14.6f} {:14.6f} {:8.2f} "
            "{:35} {:8.2f} {:8.2f} {:8.2f} {:5} "
            "{:8.2f} {:8.2f} {:8.2f} {:8.2f} "
            "{:8.2f} {:8.2f} {:8.2f} {:8.2f} {:5.2f} "
            "{:5} {:3} {:8.3f}\n"
        )
        fout.write(
            fout_fmt0.format(
                "Start time UTC",
                "Stop time UTC",
                "Start MJD",
                "Stop MJD",
                "Rotation",
                "Patch name",
                "Az min",
                "Az max",
                "El",
                "R/S",
                "Sun el1",
                "Sun az1",
                "Sun el2",
                "Sun az2",
                "Moon el1",
                "Moon az1",
                "Moon el2",
                "Moon az2",
                "Phase",
                "Pass",
                "Sub",
                "CTime",
            )
        )
    else:
        # Concise schedule format
        fout_fmt0 = "#{:>20} {:>20} {:>8} {:35} {:>8} {:>8} {:>8} {:>5} {:>3}\n"

        fout_fmt = " {:>20} {:>20} {:8.2f} {:35} {:8.2f} {:8.2f} {:8.2f} {:5} {:3}\n"
        fout.write(
            fout_fmt0.format(
                "Start time UTC",
                "Stop time UTC",
                "Rotation",
                "Patch name",
                "Az min",
                "Az max",
                "El",
                "Pass",
                "Sub",
            )
        )

    # Operational days
    ods = set()

    t = start_timestamp
    last_successful = t
    last_el = None
    while True:
        t, blocked = apply_blockouts(args, t)
        boresight_angle = get_boresight_angle(args, t)
        if t > stop_timestamp:
            break
        if t - last_successful > args.elevation_change_time_s:
            # It no longer matters what the last used elevation was
            last_el = None
        if t - last_successful > 86400 or blocked:
            # A long time has passed since the last successfully
            # scheduled scan.
            # Reset the individual patch az and el limits
            for patch in patches:
                patch.reset()
            if blocked:
                last_successful = t
            else:
                # Only try this once for every day.  Swapping
                # `t` <-> `last_successful` means that we will not trigger
                # this branch again without scheduling a succesful scan
                log.debug(
                    f"Resetting patches and returning to the last successful "
                    f"scan: {to_UTC(last_successful)}"
                )
                t, last_successful = last_successful, t

        # Determine which patches are observable at time t.

        log.debug(f"t = {to_UTC(t)}")
        # Determine which patches are visible
        observer.date = to_DJD(t)
        sun.compute(observer)
        if sun.alt > sun_el_max:
            log.debug(
                "Sun elevation is {:.2f} > {:.2f}. Moving on.".format(
                    sun.alt / degree, sun_el_max / degree
                )
            )
            t = advance_time(t, args.time_step_s)
            continue
        moon.compute(observer)

        visible, not_visible = get_visible(args, observer, sun, moon, patches, el_min)

        if len(visible) == 0:
            log.debug(f"No patches visible at {to_UTC(t)}: {not_visible}")
            t = advance_time(t, args.time_step_s)
            continue

        # Determine if a cooler cycle sets a limit for observing
        tstop_cooler = stop_timestamp
        for patch in patches:
            if isinstance(patch, CoolerCyclePatch):
                ttest = patch.last_cycle_end + patch.hold_time_max
                if ttest < tstop_cooler:
                    tstop_cooler = ttest

        # Order the targets by priority and attempt to observe with both
        # a rising and setting scans until we find one that can be
        # succesfully scanned.
        # If the criteria are not met, advance the time by a step
        # and try again

        prioritize(args, observer, visible, last_el)

        if args.pole_mode:
            success, t, el = attempt_scan_pole(
                args,
                observer,
                visible,
                not_visible,
                t,
                fp_radius,
                el_max,
                el_min,
                stop_timestamp,
                tstop_cooler,
                sun,
                moon,
                sun_el_max,
                fout,
                fout_fmt,
                ods,
                boresight_angle,
                # Pole scheduling does not (yet) implement
                # elevation change penalty
                # last_successful,
                # last_el,
            )
        else:
            success, t, el = attempt_scan(
                args,
                observer,
                visible,
                not_visible,
                t,
                fp_radius,
                stop_timestamp,
                tstop_cooler,
                sun,
                moon,
                sun_el_max,
                fout,
                fout_fmt,
                ods,
                boresight_angle,
                last_successful,
                last_el,
            )

        if args.operational_days and len(ods) > args.operational_days:
            break

        if not success:
            log.debug(f"No patches could be scanned at {to_UTC(t)}: {to_UTC(t)}")
            t = advance_time(t, args.time_step_s)
        else:
            last_successful = t
            last_el = el

    fout.close()
    return


def parse_args(opts=None):
    parser = argparse.ArgumentParser(
        description="Generate ground observation schedule.", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--site-name", required=False, default="LBL", help="Observing site name"
    )
    parser.add_argument(
        "--telescope",
        required=False,
        default="Telescope",
        help="Observing telescope name",
    )
    parser.add_argument(
        "--site-lon",
        required=False,
        default="-122.247",
        help="Observing site longitude [PyEphem string]",
    )
    parser.add_argument(
        "--site-lat",
        required=False,
        default="37.876",
        help="Observing site latitude [PyEphem string]",
    )
    parser.add_argument(
        "--site-alt",
        required=False,
        default=100,
        type=float,
        help="Observing site altitude [meters]",
    )
    parser.add_argument(
        "--scan-margin",
        required=False,
        default=0,
        type=float,
        help="Random fractional margin [0..1] added to the "
        "scans to smooth out edge effects",
    )
    parser.add_argument(
        "--ra-period",
        required=False,
        default=10,
        type=int,
        help="Period of patch position oscillations in RA [visits]",
    )
    parser.add_argument(
        "--ra-amplitude-deg",
        required=False,
        default=0,
        type=float,
        help="Amplitude of patch position oscillations in RA [deg]",
    )
    parser.add_argument(
        "--dec-period",
        required=False,
        default=10,
        type=int,
        help="Period of patch position oscillations in DEC [visits]",
    )
    parser.add_argument(
        "--dec-amplitude-deg",
        required=False,
        default=0,
        type=float,
        help="Amplitude of patch position oscillations in DEC [deg]",
    )
    parser.add_argument(
        "--elevation-penalty-limit",
        required=False,
        default=0,
        type=float,
        help="Assign a penalty to observing elevations below this limit [degrees]",
    )
    parser.add_argument(
        "--elevation-penalty-power",
        required=False,
        default=2,
        type=float,
        help="Power in the elevation penalty function [> 0]",
    )
    parser.add_argument(
        "--elevation-change-limit-deg",
        required=False,
        default=0,
        type=float,
        help="Assign a penalty to changes in elevation larger than this limit [degrees].  "
        "See --elevation-change-penalty and --elevation-change-time-s",
    )
    parser.add_argument(
        "--elevation-change-penalty",
        required=False,
        default=1,
        type=float,
        help="Multiplicative elevation change penalty triggered by "
        "--elevation-change-limit-deg",
    )
    parser.add_argument(
        "--elevation-change-time-s",
        required=False,
        default=0,
        type=float,
        help="Time it takes for the telescope to stabilize after a change in observing "
        "elevation [seconds].  Triggered by --elevation-change-limit-deg",
    )
    parser.add_argument(
        "--verbose-schedule",
        required=False,
        default=False,
        action="store_true",
        help="Write a 24-field verbose schedule "
        "instead of the concise 11-field schedule",
    )
    parser.add_argument(
        "--lock-az-range",
        required=False,
        default=False,
        action="store_true",
        help="Use the same azimuth range for all sub scans",
    )
    parser.add_argument(
        "--equalize-area",
        required=False,
        default=False,
        action="store_true",
        help="Adjust priorities to account for patch area",
    )
    parser.add_argument(
        "--equalize-time",
        required=False,
        action="store_true",
        dest="equalize_time",
        help="Modulate priority by integration time.",
    )
    parser.add_argument(
        "--equalize-scans",
        required=False,
        action="store_false",
        dest="equalize_time",
        help="Modulate priority by number of scans.",
    )
    parser.set_defaults(equalize_time=False)
    parser.add_argument(
        "--patch",
        required=True,
        action="append",
        help="Patch definition: "
        "name,weight,lon1,lat1,lon2,lat2 ... "
        "OR name,weight,lon,lat,width",
    )
    parser.add_argument(
        "--patch-coord",
        required=False,
        default="C",
        help="Sky patch coordinate system [C,E,G]",
    )
    parser.add_argument(
        "--el-min-deg",
        required=False,
        default=30,
        type=float,
        help="Minimum elevation for a CES",
    )
    parser.add_argument(
        "--el-max-deg",
        required=False,
        default=80,
        type=float,
        help="Maximum elevation for a CES",
    )
    parser.add_argument(
        "--el-step-deg",
        required=False,
        default=0,
        type=float,
        help="Optional step to apply to minimum elevation",
    )
    parser.add_argument(
        "--alternate",
        required=False,
        default=False,
        action="store_true",
        help="Alternate between rising and setting scans",
    )
    parser.add_argument(
        "--fp-radius-deg",
        required=False,
        default=0,
        type=float,
        help="Focal plane radius [deg]",
    )
    parser.add_argument(
        "--sun-avoidance-angle-deg",
        required=False,
        default=30,
        type=float,
        help="Minimum distance between the Sun and the bore sight [deg]",
    )
    parser.add_argument(
        "--sun-avoidance-altitude-deg",
        required=False,
        default=-18,
        type=float,
        help="Minimum altitude to apply Solar avoidance [deg]",
    )
    parser.add_argument(
        "--moon-avoidance-angle-deg",
        required=False,
        default=20,
        type=float,
        help="Minimum distance between the Moon and the bore sight [deg]",
    )
    parser.add_argument(
        "--moon-avoidance-altitude-deg",
        required=False,
        default=-18,
        type=float,
        help="Minimum altitude to apply Lunar avoidance [deg]",
    )
    parser.add_argument(
        "--sun-el-max-deg",
        required=False,
        default=90,
        type=float,
        help="Maximum allowed sun elevation [deg]",
    )
    parser.add_argument(
        "--boresight-angle-step-deg",
        required=False,
        default=0,
        type=float,
        help="Boresight rotation step size [deg]",
    )
    parser.add_argument(
        "--boresight-angle-min-deg",
        required=False,
        default=0,
        type=float,
        help="Boresight rotation angle minimum [deg]",
    )
    parser.add_argument(
        "--boresight-angle-max-deg",
        required=False,
        default=360,
        type=float,
        help="Boresight rotation angle maximum [deg]",
    )
    parser.add_argument(
        "--boresight-angle-time-min",
        required=False,
        default=0,
        type=float,
        help="Boresight rotation step interval [minutes]",
    )
    parser.add_argument(
        "--start",
        required=False,
        default="2000-01-01 00:00:00",
        help="UTC start time of the schedule",
    )
    parser.add_argument("--stop", required=False, help="UTC stop time of the schedule")
    parser.add_argument(
        "--block-out",
        required=False,
        action="append",
        help="Range of UTC calendar days to omit from scheduling in format "
        "START_MONTH/START_DAY-END_MONTH/END_DAY or "
        "START_YEAR/START_MONTH/START_DAY-END_YEAR/END_MONTH/END_DAY "
        "where YEAR, MONTH and DAY are integers. END days are inclusive",
    )
    parser.add_argument(
        "--operational-days",
        required=False,
        type=int,
        help="Number of operational days to schedule (empty days do not count)",
    )
    parser.add_argument(
        "--timezone",
        required=False,
        type=int,
        default=0,
        help="Offset to apply to MJD to separate operational days [hours]",
    )
    parser.add_argument(
        "--gap-s",
        required=False,
        default=100,
        type=float,
        help="Gap between CES:es [seconds]",
    )
    parser.add_argument(
        "--gap-small-s",
        required=False,
        default=10,
        type=float,
        help="Gap between split CES:es [seconds]",
    )
    parser.add_argument(
        "--time-step-s",
        required=False,
        default=600,
        type=float,
        help="Time step after failed target acquisition [seconds]",
    )
    parser.add_argument(
        "--one-scan-per-day",
        required=False,
        default=False,
        action="store_true",
        help="Pad each operational day to have only one CES",
    )
    parser.add_argument(
        "--ces-max-time-s",
        required=False,
        default=900,
        type=float,
        help="Maximum length of a CES [seconds]",
    )
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Write diagnostics, including patch plots.",
    )
    parser.add_argument(
        "--polmap",
        required=False,
        help="Include polarization from map in the plotted patches when --debug",
    )
    parser.add_argument(
        "--pol-min",
        required=False,
        type=float,
        help="Lower plotting range for polarization map",
    )
    parser.add_argument(
        "--pol-max",
        required=False,
        type=float,
        help="Upper plotting range for polarization map",
    )
    parser.add_argument(
        "--delay-sso-check",
        required=False,
        default=False,
        action="store_true",
        help="Only apply SSO check during simulated scan.",
    )
    parser.add_argument(
        "--pole-mode",
        required=False,
        default=False,
        action="store_true",
        help="Pole scheduling mode (no drift scan)",
    )
    parser.add_argument(
        "--pole-el-step-deg",
        required=False,
        default=0.25,
        type=float,
        help="Elevation step in pole scheduling mode [deg]",
    )
    parser.add_argument(
        "--pole-ces-time-s",
        required=False,
        default=3000,
        type=float,
        help="Time to scan at constant elevation in pole mode",
    )
    parser.add_argument(
        "--out", required=False, default="schedule.txt", help="Output filename"
    )
    parser.add_argument(
        "--boresight-offset-el-deg",
        required=False,
        default=0,
        type=float,
        help="Optional offset added to every observing elevation",
    )
    parser.add_argument(
        "--boresight-offset-az-deg",
        required=False,
        default=0,
        type=float,
        help="Optional offset added to every observing azimuth",
    )
    parser.add_argument(
        "--elevations-deg",
        required=False,
        help="Fixed observing elevations in a comma-separated list.",
    )
    parser.add_argument(
        "--partial-scans",
        required=False,
        action="store_true",
        dest="allow_partial_scans",
        help="Allow partials scans when full scans are not available.",
    )
    parser.add_argument(
        "--no-partial-scans",
        required=False,
        action="store_false",
        dest="allow_partial_scans",
        help="Allow partials scans when full scans are not available.",
    )
    parser.set_defaults(allow_partial_scans=False)
    # Pole raster scan arguments
    parser.add_argument(
        "--pole-raster-scan",
        required=False,
        default=False,
        action="store_true",
        help="Pole raster scan mode",
    )
    parser.add_argument(
        "--pole-raster-el-step-deg",
        required=False,
        default=1 / 60,
        type=float,
        help="Elevation step in pole raster scheduling mode [deg]",
    )
    parser.add_argument(
        "--az-rate-sky-deg",
        required=False,
        default=1.0,
        type=float,
        help="Azimuthal rate in pole raster scheduling mode [deg]",
    )
    parser.add_argument(
        "--az-accel-mount-deg",
        required=False,
        default=1.0,
        type=float,
        help="Azimuthal accleration in pole raster scheduling mode [deg]",
    )
    parser.add_argument(
        "--el-rate-deg",
        required=False,
        default=1.0,
        type=float,
        help="Elevation rate in pole raster scheduling mode [deg]",
    )
    parser.add_argument(
        "--el-accel-deg",
        required=False,
        default=1.0,
        type=float,
        help="Elevation accleration in pole raster scheduling mode [deg]",
    )

    args = None
    if opts is None:
        try:
            args = parser.parse_args()
        except SystemExit:
            sys.exit(0)
    else:
        try:
            args = parser.parse_args(opts)
        except SystemExit:
            sys.exit(0)

    if args.operational_days is None and args.stop is None:
        raise RuntimeError("You must provide --stop or --operational-days")

    stop_time = None
    if args.start.endswith("Z"):
        start_time = dateutil.parser.parse(args.start)
        if args.stop is not None:
            if not args.stop.endswith("Z"):
                raise RuntimeError("Either both or neither times must be given in UTC")
            stop_time = dateutil.parser.parse(args.stop)
    else:
        if args.timezone < 0:
            tz = "-{:02}00".format(-args.timezone)
        else:
            tz = "+{:02}00".format(args.timezone)
        start_time = dateutil.parser.parse(args.start + tz)
        if args.stop is not None:
            if args.stop.endswith("Z"):
                raise RuntimeError("Either both or neither times must be given in UTC")
            stop_time = dateutil.parser.parse(args.stop + tz)

    start_timestamp = start_time.timestamp()
    if stop_time is None:
        # Keep scheduling until the desired number of operational days is full.
        stop_timestamp = 2**60
    else:
        stop_timestamp = stop_time.timestamp()
    return args, start_timestamp, stop_timestamp


@function_timer
def parse_patch_sso(args, parts):
    log = Logger.get()
    log.info("SSO format")
    name = parts[0]
    weight = float(parts[2])
    radius = float(parts[3]) * degree
    patch = SSOPatch(
        name,
        weight,
        radius,
        el_min=args.el_min_deg * degree,
        el_max=args.el_max_deg * degree,
        elevations=args.elevations_deg,
    )
    return patch


@function_timer
def parse_patch_cooler(args, parts, last_cycle_end):
    log = Logger.get()
    log.info("Cooler cycle format")
    name = parts[0]
    weight = float(parts[2])
    power = float(parts[3])
    hold_time_min = float(parts[4])  # in hours
    hold_time_max = float(parts[5])  # in hours
    cycle_time = float(parts[6])  # in hours
    az = float(parts[7])
    el = float(parts[8])
    patch = CoolerCyclePatch(
        name,
        weight,
        power,
        hold_time_min,
        hold_time_max,
        cycle_time,
        az,
        el,
        last_cycle_end,
    )
    return patch


@function_timer
def parse_patch_horizontal(args, parts):
    """Parse an explicit patch definition line"""
    log = Logger.get()
    corners = []
    log.info("Horizontal format")
    name = parts[0]
    weight = float(parts[2])
    azmin = float(parts[3]) * degree
    azmax = float(parts[4]) * degree
    el = float(parts[5]) * degree
    scantime = float(parts[6])  # minutes
    patch = HorizontalPatch(name, weight, azmin, azmax, el, scantime)
    return patch


@function_timer
def parse_patch_explicit(args, parts):
    """Parse an explicit patch definition line"""
    log = Logger.get()
    corners = []
    log.info("Explicit-corners format: ")
    name = parts[0]
    i = 2
    definition = ""
    while i + 1 < len(parts):
        definition += " ({}, {})".format(parts[i], parts[i + 1])
        try:
            # Assume coordinates in degrees
            lon = float(parts[i]) * degree
            lat = float(parts[i + 1]) * degree
        except ValueError:
            # Failed simple interpreration, assume pyEphem strings
            lon = parts[i]
            lat = parts[i + 1]
        i += 2
        if args.patch_coord == "C":
            corner = ephem.Equatorial(lon, lat, epoch="2000")
        elif args.patch_coord == "E":
            corner = ephem.Ecliptic(lon, lat, epoch="2000")
        elif args.patch_coord == "G":
            corner = ephem.Galactic(lon, lat, epoch="2000")
        else:
            raise RuntimeError(f"Unknown coordinate system: {args.patch_coord}")
        corner = ephem.Equatorial(corner)
        if corner.dec > 80 * degree or corner.dec < -80 * degree:
            raise RuntimeError(
                f"{name} has at least one circumpolar corner. "
                "Circumpolar targeting not yet implemented"
            )
        patch_corner = ephem.FixedBody()
        patch_corner._ra = corner.ra
        patch_corner._dec = corner.dec
        corners.append(patch_corner)
    log.info(definition)
    return corners


@function_timer
def parse_patch_rectangular(args, parts):
    """Parse a rectangular patch definition line"""
    log = Logger.get()
    corners = []
    log.info("Rectangular format")
    name = parts[0]
    try:
        # Assume coordinates in degrees
        lon_min = float(parts[2]) * degree
        lat_max = float(parts[3]) * degree
        lon_max = float(parts[4]) * degree
        lat_min = float(parts[5]) * degree
    except ValueError:
        # Failed simple interpreration, assume pyEphem strings
        lon_min = parts[2]
        lat_max = parts[3]
        lon_max = parts[4]
        lat_min = parts[5]
    if args.patch_coord == "C":
        coordconv = ephem.Equatorial
    elif args.patch_coord == "E":
        coordconv = ephem.Ecliptic
    elif args.patch_coord == "G":
        coordconv = ephem.Galactic
    else:
        raise RuntimeError("Unknown coordinate system: {}".format(args.patch_coord))

    nw_corner = coordconv(lon_min, lat_max, epoch="2000")
    ne_corner = coordconv(lon_max, lat_max, epoch="2000")
    se_corner = coordconv(lon_max, lat_min, epoch="2000")
    sw_corner = coordconv(lon_min, lat_min, epoch="2000")

    lon_max = unwind_angle(lon_min, lon_max)
    if lon_min < lon_max:
        delta_lon = lon_max - lon_min
    else:
        delta_lon = lon_min - lon_max
    area = (np.cos(np.pi / 2 - lat_max) - np.cos(np.pi / 2 - lat_min)) * delta_lon

    corners_temp = []
    add_side(nw_corner, ne_corner, corners_temp, coordconv)
    add_side(ne_corner, se_corner, corners_temp, coordconv)
    add_side(se_corner, sw_corner, corners_temp, coordconv)
    add_side(sw_corner, nw_corner, corners_temp, coordconv)

    for corner in corners_temp:
        if corner.dec > 80 * degree or corner.dec < -80 * degree:
            raise RuntimeError(
                f"{name} has at least one circumpolar corner. "
                "Circumpolar targeting not yet implemented"
            )
        patch_corner = ephem.FixedBody()
        patch_corner._ra = corner.ra
        patch_corner._dec = corner.dec
        corners.append(patch_corner)
    return corners, area


@function_timer
def add_side(corner1, corner2, corners_temp, coordconv):
    """Add one side of a rectangle.

    Add one side of a rectangle with enough interpolation points.
    """
    step = np.radians(1)
    lon1 = corner1.ra
    lon2 = corner2.ra
    lat1 = corner1.dec
    lat2 = corner2.dec
    if lon1 == lon2:
        lon = lon1
        if lat1 < lat2:
            lat_step = step
        else:
            lat_step = -step
        for lat in np.arange(lat1, lat2, lat_step):
            corners_temp.append(ephem.Equatorial(coordconv(lon, lat, epoch="2000")))
    elif lat1 == lat2:
        lat = lat1
        if lon1 < lon2:
            lon_step = step / np.cos(lat)
        else:
            lon_step = -step / np.cos(lat)
        for lon in np.arange(lon1, lon2, lon_step):
            corners_temp.append(ephem.Equatorial(coordconv(lon, lat, epoch="2000")))
    else:
        raise RuntimeError("add_side: both latitude and longitude change")
    corners_temp.append(ephem.Equatorial(corner2))
    return


@function_timer
def parse_patch_center_and_width(args, parts):
    """Parse center-and-width patch definition"""
    log = Logger.get()
    corners = []
    log.info("Center-and-width format")
    try:
        # Assume coordinates in degrees
        lon = float(parts[2]) * degree
        lat = float(parts[3]) * degree
    except ValueError:
        # Failed simple interpreration, assume pyEphem strings
        lon = parts[2]
        lat = parts[3]
    width = float(parts[4]) * degree
    if args.patch_coord == "C":
        center = ephem.Equatorial(lon, lat, epoch="2000")
    elif args.patch_coord == "E":
        center = ephem.Ecliptic(lon, lat, epoch="2000")
    elif args.patch_coord == "G":
        center = ephem.Galactic(lon, lat, epoch="2000")
    else:
        raise RuntimeError("Unknown coordinate system: {}".format(args.patch_coord))
    center = ephem.Equatorial(center)
    # Synthesize 8 corners around the center
    phi = center.ra
    theta = center.dec
    r = width / 2
    ncorner = 8
    angstep = 2 * np.pi / ncorner
    for icorner in range(ncorner):
        ang = angstep * icorner
        delta_theta = np.cos(ang) * r
        delta_phi = np.sin(ang) * r / np.cos(theta + delta_theta)
        patch_corner = ephem.FixedBody()
        patch_corner._ra = phi + delta_phi
        patch_corner._dec = theta + delta_theta
        corners.append(patch_corner)
    return corners


@function_timer
def parse_patches(args, observer, sun, moon, start_timestamp, stop_timestamp):
    # Parse the patch definitions
    log = Logger.get()
    patches = []
    total_weight = 0
    for patch_def in args.patch:
        parts = patch_def.split(",")
        name = parts[0]
        log.info(f'Adding patch "{name}"')
        if parts[1].upper() == "HORIZONTAL":
            patch = parse_patch_horizontal(args, parts)
        elif parts[1].upper() == "SSO":
            patch = parse_patch_sso(args, parts)
        elif parts[1].upper() == "COOLER":
            patch = parse_patch_cooler(args, parts, start_timestamp)
        else:
            weight = float(parts[1])
            if np.isnan(weight):
                raise RuntimeError("Patch has NaN priority: {}".format(patch_def))
            if weight == 0:
                raise RuntimeError("Patch has zero priority: {}".format(patch_def))
            if len(parts[2:]) == 3:
                corners = parse_patch_center_and_width(args, parts)
                area = None
            elif len(parts[2:]) == 4:
                corners, area = parse_patch_rectangular(args, parts)
            else:
                corners = parse_patch_explicit(args, parts)
                area = None
            patch = Patch(
                name,
                weight,
                corners,
                el_min=args.el_min_deg * degree,
                el_max=args.el_max_deg * degree,
                el_step=args.el_step_deg * degree,
                alternate=args.alternate,
                site_lat=observer.lat,
                area=area,
                ra_period=args.ra_period,
                ra_amplitude=args.ra_amplitude_deg,
                dec_period=args.dec_period,
                dec_amplitude=args.dec_amplitude_deg,
                elevations=args.elevations_deg,
            )
        if args.equalize_area or args.debug:
            area = patch.get_area(observer, nside=32, equalize=args.equalize_area)
        total_weight += patch.weight
        patches.append(patch)

        if patches[-1].el_max0 is not None:
            el_max = patches[-1].el_max0 / degree
            log.debug(f"Highest possible observing elevation: {el_max:.2f} deg.")
        if patches[-1]._area is not None:
            log.debug(f"Sky fraction = {patch._area:.4f}")

    if args.debug:
        import matplotlib.pyplot as plt

        polmap = None
        if args.polmap:
            polmap = hp.read_map(args.polmap, [1, 2])
            bad = polmap[0] == hp.UNSEEN
            polmap = np.sqrt(polmap[0] ** 2 + polmap[1] ** 2) * 1e6
            polmap[bad] = hp.UNSEEN
        plt.style.use("default")
        cmap = cm.inferno
        cmap.set_under("w")
        plt.figure(figsize=[20, 4])
        plt.subplots_adjust(left=0.1, right=0.9)
        patch_color = "black"
        sun_color = "black"
        sun_lw = 8
        sun_avoidance_color = "gray"
        moon_color = "black"
        moon_lw = 2
        moon_avoidance_color = "gray"
        alpha = 0.5
        avoidance_alpha = 0.01
        sun_step = int(86400 * 1)
        moon_step = int(86400 * 0.1)
        for iplot, coord in enumerate("CEG"):
            scoord = {"C": "Equatorial", "E": "Ecliptic", "G": "Galactic"}[coord]
            title = scoord  # + ' patch locations'
            if polmap is None:
                nside = 256
                avoidance_map = np.zeros(12 * nside**2)
                # hp.mollview(np.zeros(12) + hp.UNSEEN, coord=coord, cbar=False,
                #            title='', sub=[1, 3, 1 + iplot], cmap=cmap)
            else:
                hp.mollview(
                    polmap,
                    coord="G" + coord,
                    cbar=True,
                    unit="$\mu$K",
                    min=args.polmin,
                    max=args.polmax,
                    norm="log",
                    cmap=cmap,
                    title=title,
                    sub=[1, 3, 1 + iplot],
                    notext=True,
                    format="%.1f",
                    xsize=1600,
                )
            # Plot sun and moon avoidance circle
            sunlon, sunlat = [], []
            moonlon, moonlat = [], []
            for lon, lat, sso, angle_min, alt_min, color, step, lw in [
                (
                    sunlon,
                    sunlat,
                    sun,
                    np.radians(args.sun_avoidance_angle_deg),
                    np.radians(args.sun_avoidance_altitude_deg),
                    sun_avoidance_color,
                    sun_step,
                    sun_lw,
                ),
                (
                    moonlon,
                    moonlat,
                    moon,
                    np.radians(args.moon_avoidance_angle_deg),
                    np.radians(args.moon_avoidance_altitude_deg),
                    moon_avoidance_color,
                    moon_step,
                    moon_lw,
                ),
            ]:
                for t in range(int(start_timestamp), int(stop_timestamp), step):
                    observer.date = to_DJD(t)
                    sso.compute(observer)
                    lon.append(np.degrees(sso.a_ra))
                    lat.append(np.degrees(sso.a_dec))
                    if angle_min <= 0 or sso.alt < alt_min:
                        continue
                    if polmap is None:
                        # accumulate avoidance map
                        vec = hp.dir2vec(lon[-1], lat[-1], lonlat=True)
                        pix = hp.query_disc(nside, vec, angle_min)
                        for p in pix:
                            avoidance_map[p] += 1
                    else:
                        # plot a circle around the location
                        clon, clat = [], []
                        phi = sso.a_ra
                        theta = sso.a_dec
                        r = angle_min
                        for ang in np.linspace(0, 2 * np.pi, 36):
                            dtheta = np.cos(ang) * r
                            dphi = np.sin(ang) * r / np.cos(theta + dtheta)
                            clon.append((phi + dphi) / degree)
                            clat.append((theta + dtheta) / degree)
                        hp.projplot(
                            clon,
                            clat,
                            "-",
                            color=color,
                            alpha=avoidance_alpha,
                            lw=lw,
                            threshold=1,
                            lonlat=True,
                            coord="C",
                        )
            if polmap is None:
                avoidance_map[avoidance_map == 0] = hp.UNSEEN
                hp.mollview(
                    avoidance_map,
                    coord="C" + coord,
                    cbar=False,
                    title="",
                    sub=[1, 3, 1 + iplot],
                    cmap=cmap,
                )
            hp.graticule(30, verbose=False)

            # Plot patches
            for patch in patches:
                lon = [corner._ra / degree for corner in patch.corners]
                lat = [corner._dec / degree for corner in patch.corners]
                if len(lon) == 0:
                    # Special patch without sky coordinates
                    continue
                lon.append(lon[0])
                lat.append(lat[0])
                log.info(f"{patch.name,} corners:\n lon = {lon}\n lat= {lat}")
                hp.projplot(
                    lon,
                    lat,
                    "-",
                    threshold=1,
                    lonlat=True,
                    coord="C",
                    color=patch_color,
                    lw=2,
                    alpha=alpha,
                )
                if len(patches) > 10:
                    continue
                # label the patch
                it = np.argmax(lat)
                area = patch.get_area(observer)
                title = "{} {:.2f}%".format(patch.name, 100 * area)
                hp.projtext(
                    lon[it],
                    lat[it],
                    title,
                    lonlat=True,
                    coord="C",
                    color=patch_color,
                    fontsize=14,
                    alpha=alpha,
                )
            if polmap is not None:
                # Plot Sun and Moon trajectory
                hp.projplot(
                    sunlon,
                    sunlat,
                    "-",
                    color=sun_color,
                    alpha=alpha,
                    threshold=1,
                    lonlat=True,
                    coord="C",
                    lw=sun_lw,
                )
                hp.projplot(
                    moonlon,
                    moonlat,
                    "-",
                    color=moon_color,
                    alpha=alpha,
                    threshold=1,
                    lonlat=True,
                    coord="C",
                    lw=moon_lw,
                )
                hp.projtext(
                    sunlon[0],
                    sunlat[0],
                    "Sun",
                    color=sun_color,
                    lonlat=True,
                    coord="C",
                    fontsize=14,
                    alpha=alpha,
                )
                hp.projtext(
                    moonlon[0],
                    moonlat[0],
                    "Moon",
                    color=moon_color,
                    lonlat=True,
                    coord="C",
                    fontsize=14,
                    alpha=alpha,
                )

        plt.savefig("patches.png")
        plt.close()

    # Normalize the weights
    for i in range(len(patches)):
        patches[i].weight /= total_weight
    return patches


def run_scheduler(opts=None):
    args, start_timestamp, stop_timestamp = parse_args(opts=opts)

    observer = ephem.Observer()
    observer.lon = args.site_lon
    observer.lat = args.site_lat
    observer.elevation = args.site_alt  # In meters
    observer.epoch = "2000"
    observer.temp = 0  # in Celcius
    observer.compute_pressure()

    sun = ephem.Sun()
    moon = ephem.Moon()

    patches = parse_patches(args, observer, sun, moon, start_timestamp, stop_timestamp)

    build_schedule(args, start_timestamp, stop_timestamp, patches, observer, sun, moon)
    return
