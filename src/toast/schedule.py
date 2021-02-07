# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import sys

import numpy as np

from .timing import function_timer, Timer

from .utils import Logger, Environment


class CES(object):
    def __init__(
        self,
        start_time,
        stop_time,
        boresight_angle,
        name,
        mjdstart,
        scan,
        subscan,
        azmin,
        azmax,
        el,
        season,
        start_date,
        rising,
        mindist_sun,
        mindist_moon,
        el_sun,
    ):
        self.start_time = start_time
        self.stop_time = stop_time
        self.boresight_angle = boresight_angle
        self.name = name
        self.mjdstart = mjdstart
        self.scan = scan
        self.subscan = subscan
        self.azmin = azmin
        self.azmax = azmax
        self.el = el
        self.season = season
        self.start_date = start_date
        self.rising = rising
        self.mindist_sun = mindist_sun
        self.mindist_moon = mindist_moon
        self.el_sun = el_sun


# datetime.datetime.utcfromtimestamp(self._time)


class Schedule(object):
    """Class representing an observing schedule.

    A schedule consists of a list of Constant Elevation Scans (CESs) for a telescope at
    a particular site.  The schedule can be constructed either by loading from a file
    or by passing the telescope and CES list directly.

    Args:
        file (str):  The path to the schedule file.
        file_split (tuple):  If not None and loading from a file, only use a subset of
            the schedule.  The arguments are (isplit, nsplit) and only observations
            that satisfy 'scan index modulo nsplit == isplit' are included.
        telescope (Telescope):  If not loading from a file, specify the Telescope
            instance.
        ceslist (list):  If not loading from a file, specify the list of CESs.
        sort (bool):  If True, sort the CES list by name.

    """

    def __init__(
        self, file=None, file_split=None, telescope=None, ceslist=None, sort=False
    ):
        if file is not None:
            if not os.path.isfile(file):
                msg = "No such schedule file '{}'".format(file)
                raise RuntimeError(msg)
            self.load(file, file_split)
        else:
            self.telescope = telescope
            self.ceslist = ceslist
        if sort:
            self.sort_ceslist()
        return

    def sort_ceslist(self):
        """Sort the list of CES by name."""
        if self.ceslist is None:
            return
        nces = len(self.ceslist)
        for i in range(nces - 1):
            for j in range(i + 1, nces):
                if self.ceslist[j].name < self.ceslist[j - 1].name:
                    temp = self.ceslist[j]
                    self.ceslist[j] = self.ceslist[j - 1]
                    self.ceslist[j - 1] = temp
        return

    @function_timer
    def min_sso_dist(el, azmin, azmax, sso_el1, sso_az1, sso_el2, sso_az2):
        """Return a rough minimum angular distance between the bore sight
        and a solar system object"""
        sso_vec1 = hp.dir2vec(sso_az1, sso_el1, lonlat=True)
        sso_vec2 = hp.dir2vec(sso_az2, sso_el2, lonlat=True)
        az1 = azmin
        az2 = azmax
        if az2 < az1:
            az2 += 360
        n = 100
        az = np.linspace(az1, az2, n)
        el = np.ones(n) * el
        vec = hp.dir2vec(az, el, lonlat=True)
        dist1 = np.degrees(np.arccos(np.dot(sso_vec1, vec)))
        dist2 = np.degrees(np.arccos(np.dot(sso_vec2, vec)))
        return min(np.amin(dist1), np.amin(dist2))

    @function_timer
    def load(self, file, file_split):
        """Load the observing schedule from a file.

        This method populates the internal telescope and CES list.  For parallel use,
        simply construct the schedule on one process and broadcast.

        Args:
            file (str):  The file to load.
            file_split (tuple):  If not None and loading from a file, only use a subset
                of the schedule.  The arguments are (isplit, nsplit) and only
                observations that satisfy 'scan index modulo nsplit == isplit' are
                included.

        Returns:
            None

        """
        isplit = None
        nsplit = None
        if file_split is not None:
            isplit, nsplit = file_split
        scan_counters = dict()
        all_ces = list()

        with open(file, "r") as f:
            while True:
                line = f.readline()
                if line.startswith("#"):
                    continue
                (
                    site_name,
                    telescope_name,
                    site_lat,
                    site_lon,
                    site_alt,
                ) = line.split()
                site = Site(site_name, site_lat, site_lon, float(site_alt))
                self.telescope = Telescope(telescope_name, site=site, coord="C")
                break
            last_name = None
            for line in f:
                if line.startswith("#"):
                    continue
                (
                    start_timestamp,
                    start_date,
                    stop_timestamp,
                    season,
                    mjdstart,
                    mjdstop,
                    boresight_angle,
                    name,
                    azmin,
                    azmax,
                    el,
                    scan,
                    subscan,
                    mindist_sun,
                    mindist_moon,
                    el_sun,
                    rising,
                ) = _parse_line(line)
                if nsplit is not None:
                    # Only accept 1 / `nsplit` of the rising and setting
                    # scans in patch `name`.  Selection is performed
                    # during the first subscan.
                    if name != last_name:
                        if name not in scan_counters:
                            scan_counters[name] = dict()
                        counter = scan_counters[name]
                        # Separate counters for rising and setting scans
                        if rising not in counter:
                            counter[rising] = 0
                        else:
                            counter[rising] += 1
                        iscan = counter[rising]
                    last_name = name
                    if iscan % nsplit != isplit:
                        continue
                all_ces.append(
                    CES(
                        start_time=start_timestamp,
                        stop_time=stop_timestamp,
                        boresight_angle=boresight_angle,
                        name=name,
                        mjdstart=mjdstart,
                        scan=scan,
                        subscan=subscan,
                        azmin=azmin,
                        azmax=azmax,
                        el=el,
                        season=season,
                        start_date=start_date,
                        rising=rising,
                        mindist_sun=mindist_sun,
                        mindist_moon=mindist_moon,
                        el_sun=el_sun,
                    )
                )
        self.ceslist = all_ces

    def _parse_line(line):
        """Parse one line of the schedule file"""
        if line.startswith("#"):
            return None

        fields = line.split()
        nfield = len(fields)
        if nfield == 22:
            # Deprecated prior to 2020-02 schedule format without boresight rotation field
            (
                start_date,
                start_time,
                stop_date,
                stop_time,
                mjdstart,
                mjdstop,
                name,
                azmin,
                azmax,
                el,
                rs,
                sun_el1,
                sun_az1,
                sun_el2,
                sun_az2,
                moon_el1,
                moon_az1,
                moon_el2,
                moon_az2,
                moon_phase,
                scan,
                subscan,
            ) = line.split()
            boresight_angle = 0
        else:
            # 2020-02 schedule format with boresight rotation field
            (
                start_date,
                start_time,
                stop_date,
                stop_time,
                mjdstart,
                mjdstop,
                boresight_angle,
                name,
                azmin,
                azmax,
                el,
                rs,
                sun_el1,
                sun_az1,
                sun_el2,
                sun_az2,
                moon_el1,
                moon_az1,
                moon_el2,
                moon_az2,
                moon_phase,
                scan,
                subscan,
            ) = line.split()
        start_time = start_date + " " + start_time
        stop_time = stop_date + " " + stop_time
        # Define season as a calendar year.  This can be
        # changed later and could even be in the schedule file.
        season = int(start_date.split("-")[0])
        try:
            start_time = dateutil.parser.parse(start_time + " +0000")
            stop_time = dateutil.parser.parse(stop_time + " +0000")
        except Exception:
            start_time = dateutil.parser.parse(start_time)
            stop_time = dateutil.parser.parse(stop_time)
        start_timestamp = start_time.timestamp()
        stop_timestamp = stop_time.timestamp()
        # useful metadata
        mindist_sun = min_sso_dist(
            *np.array([el, azmin, azmax, sun_el1, sun_az1, sun_el2, sun_az2]).astype(
                np.float
            )
        )
        mindist_moon = min_sso_dist(
            *np.array(
                [el, azmin, azmax, moon_el1, moon_az1, moon_el2, moon_az2]
            ).astype(np.float)
        )
        el_sun = max(float(sun_el1), float(sun_el2))
        return (
            start_timestamp,
            start_date,
            stop_timestamp,
            season,
            float(mjdstart),
            float(mjdstop),
            float(boresight_angle),
            name,
            float(azmin),
            float(azmax),
            float(el),
            int(scan),
            int(subscan),
            mindist_sun,
            mindist_moon,
            el_sun,
            rs.upper() == "R",
        )
