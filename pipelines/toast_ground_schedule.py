#!/usr/bin/env python

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

# This script creates CES schedule file that can be used as input
# to toast_ground_sim.py

import argparse
from datetime import datetime
import dateutil.parser

import numpy as np
import ephem
from scipy.constants import degree


def to_JD(t):
    # Unix time stamp to Julian date
    # (days since -4712-01-01 12:00:00 UTC)
    return t / 86400. + 2440587.5

def to_MJD(t):
    # Convert Unix time stamp to modified Julian date
    # (days since 1858-11-17 00:00:00 UTC)
    return to_JD(t) - 2400000.5

def to_DJD(t):
    # Convert Unix time stamp to Dublin Julian date
    # (days since 1899-12-31 12:00:00)
    # This is the time format used by PyEphem
    return to_JD(t) - 2415020


def main():

    parser = argparse.ArgumentParser(
        description='Generate ground observation schedule.',
        fromfile_prefix_chars='@')

    parser.add_argument('--site_name',
                        required=False, default='LBL',
                        help='Observing site name')
    parser.add_argument('--site_lon',
                        required=False, default='-122.247',
                        help='Observing site longitude [PyEphem string]')
    parser.add_argument('--site_lat',
                        required=False, default='37.876',
                        help='Observing site latitude [PyEphem string]')
    parser.add_argument('--site_alt',
                        required=False, default=100.0, type=np.float,
                        help='Observing site altitude [meters]')
    parser.add_argument('--patch',
                        required=True, action='append',
                        help='Patch definition: '
                        'name,weight,lon1,lat1,lon2,lat2 ...')
    parser.add_argument('--patch_coord',
                        required=False, default='C',
                        help='Sky patch coordinate system [C,E,G]')
    parser.add_argument('--el_min',
                        required=False, default=0.0, type=np.float,
                        help='Minimum elevation for a CES')
    parser.add_argument('--el_max',
                        required=False, default=90.0, type=np.float,
                        help='Maximum elevation for a CES')
    parser.add_argument('--fp_radius',
                        required=False, default=0.0, type=np.float,
                        help='Focal plane radius [deg]')
    parser.add_argument('--sun_avoidance_angle',
                        required=False, default=-15.0, type=np.float,
                        help='Solar elevation above which to apply '
                        'sun_angle_min [deg]')
    parser.add_argument('--sun_angle_min',
                        required=False, default=30.0, type=np.float,
                        help='Minimum distance between the Sun and '
                        'the bore sight [deg]')
    parser.add_argument('--moon_angle_min',
                        required=False, default=10.0, type=np.float,
                        help='Minimum distance between the Moon and '
                        'the bore sight [deg]')
    parser.add_argument('--sun_el_max',
                        required=False, default=90.0, type=np.float,
                        help='Maximum allowed sun elevation [deg]')
    parser.add_argument('--start',
                        required=False, default='2000-01-01 00:00:00',
                        help='UTC start time of the schedule')
    parser.add_argument('--stop',
                        required=False, default='2000-01-02 00:00:00',
                        help='UTC stop time of the schedule')
    parser.add_argument('--gap',
                        required=False, default=100, type=np.float,
                        help='Gap between CES:es [seconds]')
    parser.add_argument('--debug',
                        required=False, default=False, action='store_true',
                        help='Write diagnostics')

    args = parser.parse_args()

    try:
        start_time = dateutil.parser.parse(args.start + ' +0000')
        stop_time = dateutil.parser.parse(args.stop + ' +0000')
    except:
        start_time = dateutil.parser.parse(args.start)
        stop_time = dateutil.parser.parse(args.stop)

    start_timestamp = start_time.timestamp()
    stop_timestamp = stop_time.timestamp()

    if args.debug:
        import healpy as hp
        import matplotlib.pyplot as plt

    observer = ephem.Observer()
    observer.lon = args.site_lon
    observer.lat = args.site_lat
    observer.elevation = args.site_alt # In meters
    observer.epoch = '2000'
    observer.temp = 0 # in Celcius
    observer.compute_pressure()

    # Parse the patch definitions

    if args.debug:
        hp.graticule(30)

    patches = []
    hits = {}
    total_weight = 0
    for patch_def in args.patch:
        parts = patch_def.split(',')
        name = parts[0]
        hits[name] = 0
        weight = float(parts[1])
        total_weight += weight
        i = 2
        corners = []
        print('Adding patch "{}" {} '.format(name, weight), end='')
        while i < len(parts):
            print(' ({}, {})'.format(parts[i], parts[i+1]), end='')
            lon = float(parts[i]) * degree
            lat = float(parts[i+1]) * degree
            i += 2    
            if args.patch_coord == 'C':
                corner = ephem.Equatorial(lon, lat, epoch='2000')
            elif args.patch_coord == 'E':
                corner = ephem.Ecliptic(lon, lat, epoch='2000')
            elif args.patch_coord == 'G':
                corner = ephem.Galactic(lon, lat, epoch='2000')
            else:
                raise RuntimeError('Unknown coordinate system: {}'.format(
                    args.patch_coord))
            corner = ephem.Equatorial(corner)
            if corner.dec > 80*degree or corner.dec < -80*degree:
                raise RuntimeError(
                    '{} has at least one circumpolar corner. '
                    'Circumpolar targeting not yet implemented'.format(name))
            patch_corner = ephem.FixedBody()
            patch_corner._ra = corner.ra
            patch_corner._dec = corner.dec
            corners.append(patch_corner)
        print('')
        patches.append([name, weight, corners])

        if args.debug:
            lon = [corner._ra/degree for corner in corners]
            lat = [corner._dec/degree for corner in corners]
            lon.append(lon[0])
            lat.append(lat[0])
            hp.projplot(lon, lat, '-', threshold=1, lonlat=True)
            hp.projtext(np.amin(lon), np.amax(lat)+5, name, lonlat=True)

    # Normalize the weights
    for i in range(len(patches)):
        patches[i][1] /= total_weight

    if args.debug:
        plt.savefig('patches.png')
        plt.close()

    t = start_timestamp
    gap = args.gap
    sun = ephem.Sun()
    moon = ephem.Moon()
    tstep = 600
    fout = open('schedule.txt', 'w')

    fout.write('#{:15} {:15} {:15} {:15}\n'.format(
        'Site', 'Latitude [deg]', 'Longitude [deg]', 'Altitude [m]'))
    fout.write(' {:15} {:15} {:15} {:15.6f}\n'.format(
        args.site_name, args.site_lat, args.site_lon, args.site_alt))

    fout_fmt0 = '#{:20} {:20} {:14} {:14} ' \
                '{:15} {:8} {:8} {:8} {:5} ' \
                '{:8} {:8} {:8} {:8} ' \
                '{:8} {:8} {:8} {:8} {:5} ' \
                '{:5}\n'
                
    fout_fmt = ' {:20} {:20} {:14.6f} {:14.6f} ' \
               '{:15} {:8.2f} {:8.2f} {:8.2f} {:5} ' \
               '{:8.2f} {:8.2f} {:8.2f} {:8.2f} ' \
               '{:8.2f} {:8.2f} {:8.2f} {:8.2f} {:5.2f} ' \
               '{:5}\n'

    fout.write(
        fout_fmt0.format(
            'Start time UTC', 'Stop time UTC', 'Start MJD', 'Stop MJD',
            'Patch name', 'Az min', 'Az max', 'El', 'R/S',
            'Sun el1', 'Sun az1', 'Sun el2', 'Sun az2',
            'Moon el1', 'Moon az1', 'Moon el2', 'Moon az2', 'Phase',
            'Pass'))

    while t < stop_timestamp:
        # Determine which patches are visible
        observer.date = to_DJD(t)
        sun.compute(observer)
        sun_el = sun.alt / degree
        sun_az = sun.az / degree
        if sun_el > args.sun_el_max:
            t += tstep
            continue
        moon.compute(observer)
        moon_el = moon.alt / degree
        moon_az = moon.az / degree
        moon_phase = moon.moon_phase
        visible = []
        not_visible = []
        for (name, weight, corners) in patches:
            # Reject all patches that have even one corner too close
            # to the sun, all setting patches that are not completely
            # above el_min and all rising patches that do not have
            # at least one corner above al_min.
            in_view = True
            corners[0].compute(observer)
            el0 = corners[0].alt
            observer.date = to_DJD(t+100)
            corners[0].compute(observer)
            el1 = corners[0].alt
            rising = el1 > el0
            observer.date = to_DJD(t)
            els = np.zeros(len(corners))
            for i, corner in enumerate(corners):
                corner.compute(observer)
                if not rising and corner.alt / degree < args.el_min:
                    # At least one corner is too low
                    not_visible.append((
                        name, 'Too low {:.2f}'.format(corner.alt / degree)))
                    in_view = False
                    break
                if rising and corner.alt / degree > args.el_max:
                    # At least one corner is too high
                    not_visible.append((
                        name, 'Too high {:.2f}'.format(corner.alt / degree)))
                    in_view = False
                    break
                els[i] = corner.alt
                if sun.alt > args.sun_avoidance_angle * degree:
                    # Sun is high enough to apply sun_angle_min check
                    angle = ephem.separation(sun, corner) / degree
                    if angle < args.sun_angle_min:
                        # Patch is too close to the Sun
                        not_visible.append((
                            name, 'Too close to Sun {:.2f}'.format(angle)))
                        in_view = False
                        break
                angle = ephem.separation(moon, corner) / degree
                if angle < args.moon_angle_min:
                    # Patch is too close to the Moon
                    not_visible.append((
                        name, 'Too close to Moon {:.2f}'.format(angle)))
                    in_view = False
                    break
            if rising and in_view:
                elmax = np.amax(els) / degree
                if elmax < args.el_min:
                    # All corners are too low
                    not_visible.append((
                        name, 'Too low {:.2f}'.format(corner.alt / degree)))
                    in_view = False
            if not rising and in_view:
                elmin = np.amin(els) / degree
                if elmin > args.el_max:
                    # All corners are too high
                    not_visible.append((
                        name, 'Too high {:.2f}'.format(corner.alt / degree)))
                    in_view = False
            if in_view:
                visible.append((name, weight, corners))
        if len(visible) == 0:
            if args.debug:
                tstring = datetime.utcfromtimestamp(t).strftime(
                    '%Y-%m-%d %H:%M:%S %Z')
                print('No patches visible at {}: {}'.format(tstring, not_visible))
            t += tstep
            continue

        # Order the targets by priority and attempt to observe them
        # until we find one that meets all criteria:
        #   1) All corners cross the CES
        #   2) Sun does not move too close during the scan
        # If the criteria are not met, advance the time by a step
        # and try again
        for i in range(len(visible)-1):
            for j in range(i+1, len(visible)):
                iname, iweight, icorners = visible[i]
                ihit = hits[iname]
                jname, jweight, jcorners = visible[i]
                jhit = hits[jname]
                if ihit*jweight < jhit*iweight:
                    visible[i], visible[j] = visible[j], visible[i]

        success = False
        for (name, weight, corners) in visible:
            # Start by determining if the patch is rising or setting
            az0 = corners[0].az
            els0 = np.array([corner.alt for corner in corners])
            observer.date = to_DJD(t+100)
            for corner in corners:
                corner.compute(observer)
            els1 = np.array([corner.alt for corner in corners])
            rising = np.all(els0 < els1)
            ambiguous = np.any((els0 < els1) != rising)
            if ambiguous:
                # This patch is rotating.  It is unlikely all corners
                # would make across a CES line.
                not_visible.append((name, 'Rotating'))
                continue
            observer.date = to_DJD(t)
            for corner in corners:
                corner.compute(observer)
            # Then determine an elevation that all corners will cross
            ncorner = len(corners)
            azs = np.zeros(ncorner)
            els = np.zeros(ncorner)
            for i, corner in enumerate(corners):
                azs[i] = corner.az
                els[i] = corner.alt
            if rising:
                el = np.amax(els) + args.fp_radius * degree
            else:
                el = np.amin(els) - args.fp_radius * degree
            azmin = 1e10
            azmax = -1e10
            # and now track when all corners are past the elevation
            tstop = t
            to_cross = np.ones(len(corners), dtype=np.bool)
            old_az = azs.copy()
            old_el = els.copy()
            old_to_cross = to_cross.copy()
            while True:
                tstop += tstep / 10
                if tstop > stop_timestamp:
                    not_visible.append((name, 'Ran out of time'))
                    break
                observer.date = to_DJD(tstop)
                sun.compute(observer)
                if sun.alt / degree > args.sun_el_max:
                    not_visible.append((
                        name, 'Sun too high {:.2f}'.format(sun.alt / degree)))
                    break
                moon.compute(observer)
                sun_too_close = False
                moon_too_close = False
                for i, corner in enumerate(corners):
                    corner.compute(observer)
                    azs[i] = corner.az
                    els[i] = corner.alt
                    if sun.alt > args.sun_avoidance_angle * degree:
                        # Check if the sun has moved too close
                        angle = ephem.separation(sun, corner) / degree
                        if angle < args.sun_angle_min:
                            # Patch is too close to the Sun
                            not_visible.append((
                                name, 'Too close to Sun {:.2f}'.format(angle)))
                            sun_too_close = True
                            break
                    # Check if the sun has moved too close
                    angle = ephem.separation(moon, corner) / degree
                    if angle < args.moon_angle_min:
                        # Patch is too close to the Moon
                        not_visible.append((
                            name, 'Too close to Moon {:.2f}'.format(angle)))
                        moon_too_close = True
                        break
                if sun_too_close or moon_too_close:
                    break

                # The patch may change direction without all corners
                # crossing the CES line
                rising_now = np.all((old_el < els)[to_cross])
                ambiguous = np.any((old_el < els)[to_cross] != rising_now)
                if ambiguous or rising != rising_now:
                    # The patch changed direction without crossing the
                    # CES line.  Move on to the next target
                    not_visible.append((name, 'Changed direction'))
                    break
                if rising:
                    to_cross[els > el + args.fp_radius * degree] = False
                else:
                    to_cross[els < el - args.fp_radius * degree] = False
                if np.any(old_to_cross != to_cross):
                    # Record the azimuths for the corners at the time of
                    # the crossing
                    mask = old_to_cross != to_cross
                    azmin = min(
                        azmin, np.amin(old_az[mask]), np.amin(azs[mask]))
                    azmax = max(
                        azmax, np.amax(old_az[mask]), np.amax(azs[mask]))
                if np.all(np.logical_not(to_cross)):
                    # All corners made it across the CES line.
                    success = True
                    break
                old_az = azs.copy()
                old_el = els.copy()
                old_to_cross = to_cross.copy()
            if not success:
                # CES failed due to the Sun, Moon or patch changing direction.
                # Try the next patch instead.
                continue
            # Check if we are scanning across the zero meridian
            if azmax - azmin > np.pi:
                # we are, scan from the maximum to the minimum
                azmin, azmax = azmax, azmin
            # Add the focal plane radius to the scan width
            azmin = (azmin - args.fp_radius * degree) % (2*np.pi)
            azmax = (azmax + args.fp_radius * degree) % (2*np.pi)
            ces_start = datetime.utcfromtimestamp(t).strftime(
                '%Y-%m-%d %H:%M:%S %Z')
            ces_stop = datetime.utcfromtimestamp(tstop).strftime(
                '%Y-%m-%d %H:%M:%S %Z')
            # Create an entry in the schedule
            rising_string = 'R' if rising else 'S'
            hits[name] += 1
            fout.write(
                fout_fmt.format(
                    ces_start, ces_stop, to_MJD(t), to_MJD(tstop),
                    name,
                    azmin/degree, azmax/degree, el/degree,
                    rising_string,
                    sun_el, sun_az, sun.alt/degree, sun.az/degree,
                    moon_el, moon_az, moon.alt/degree, moon.az/degree,
                    moon_phase, hits[name]))
            # Advance the time
            t = tstop
            # Add the gap
            t += gap
            break

        if not success:
            if args.debug:
                tstring = datetime.utcfromtimestamp(t).strftime(
                    '%Y-%m-%d %H:%M:%S %Z')
                print('No patches visible at {}: {}'.format(tstring, not_visible))
            t += tstep

    fout.close()

if __name__ == '__main__':
    main()
