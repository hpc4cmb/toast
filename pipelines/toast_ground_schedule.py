#!/usr/bin/env python3

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

import toast.timing as timing

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


def prioritize(visible, hits):
    """ Order visible targets by priority and number of scans.
    """
    for i in range(len(visible)-1):
        for j in range(i+1, len(visible)):
            iname, iweight, icorners = visible[i]
            ihit = hits[iname]
            jname, jweight, jcorners = visible[j]
            jhit = hits[jname]
            if ihit*jweight < jhit*iweight:
                visible[i], visible[j] = visible[j], visible[i]

    return


def corner_coordinates(observer, corners):
    """ Return the corner coordinates in horizontal frame.
    """
    azs = []
    els = []
    for corner in corners:
        corner.compute(observer)
        azs.append(corner.az)
        els.append(corner.alt)

    return np.array(azs), np.array(els)


def attempt_scan(
        args, observer, visible, not_visible, t, fp_radius, el_max, el_min,
        tstep, stop_timestamp, sun, moon, sun_el_max, hits, fout, fout_fmt):
    """ Attempt scanning the visible patches in order until success.
    """
    success = False
    for (name, weight, corners) in visible:
        for rising in [True, False]:
            observer.date = to_DJD(t)
            if args.pole_mode:
                radius = 0
            else:
                radius = fp_radius
            el = get_constant_elevation(observer, corners, rising, radius,
                                        el_min, el_max, not_visible, name)
            if el is None:
                continue
            if args.pole_mode:
                pole_success = True
                subscan = -1
                while pole_success:
                    pole_success, azmins, azmaxs, aztimes, tstop \
                        = scan_patch_pole(
                            args, el, corners, t, fp_radius, observer, sun,
                            not_visible, name, tstep, stop_timestamp,
                            sun_el_max, rising)
                    if pole_success:
                        if success:
                            # Still the same scan
                            hits[name] -= 1
                            subscan += 1
                        t = add_scan(
                            args, t, tstop, aztimes, azmins, azmaxs, rising,
                            fp_radius, observer, sun, moon, fout, fout_fmt, hits,
                            name, el, subscan=subscan)
                        if rising:
                            el -= np.radians(args.pole_el_step)
                        else:
                            el += np.radians(args.pole_el_step)
                        success = True
                if success:
                    break
            else:
                success, azmins, azmaxs, aztimes, tstop = scan_patch(
                    el, corners, t, fp_radius, observer, sun, not_visible,
                    name, tstep, stop_timestamp, sun_el_max, rising)
                if success:
                    t = add_scan(args, t, tstop, aztimes, azmins, azmaxs, rising,
                                 fp_radius, observer, sun, moon, fout, fout_fmt, hits,
                                 name, el)
                    break
        if success:
            break

    return success, t


def get_constant_elevation(observer, corners, rising, fp_radius, el_min, el_max,
                           not_visible, name):
    """ Determine the elevation at which to scan.
    """
    azs, els = corner_coordinates(observer, corners)
    el = None
    if rising:
        ind = azs <= np.pi
        if np.sum(ind) == 0:
            not_visible.append((name, 'No rising corners'))
        else:
            el = np.amax(els[ind]) + fp_radius
    else:
        ind = azs >= np.pi
        if np.sum(ind) == 0:
            not_visible.append((name, 'No setting corners'))
        else:
            el = np.amin(els[ind]) - fp_radius

    if el is not None:
        if el < el_min:
            not_visible.append((
                name, 'el < el_min ({:.2f} < {:.2f}) rising = {}'.format(
                    el/degree, el_min/degree, rising)))
            el = None
        elif el > el_max:
            not_visible.append((
                name, 'el > el_max ({:.2f} > {:.2f}) rising = {}'.format(
                    el/degree, el_max/degree, rising)))
            el = None

    return el


def scan_patch(el, corners, t, fp_radius, observer, sun, not_visible,
               name, tstep, stop_timestamp, sun_el_max, rising):
    """ Attempt scanning the patch specified by corners at elevation el.
    """
    success = False
    # and now track when all corners are past the elevation
    tstop = t
    to_cross = np.ones(len(corners), dtype=np.bool)
    azmins, azmaxs, aztimes = [], [], []
    azs, els = corner_coordinates(observer, corners)
    while True:
        tstop += tstep / 10
        if tstop > stop_timestamp or tstop - t > 86400:
            not_visible.append((name, 'Ran out of time rising = {}'
                                ''.format(rising)))
            break
        observer.date = to_DJD(tstop)
        sun.compute(observer)
        if sun.alt > sun_el_max:
            not_visible.append((name, 'Sun too high {:.2f} rising = {}'
                                ''.format(sun.alt/degree, rising)))
            break
        azs, els = corner_coordinates(observer, corners)
        if rising:
            good = azs <= np.pi
            to_cross[np.logical_and(els > el+fp_radius, good)] = False
        else:
            good = azs >= np.pi
            to_cross[np.logical_and(els < el-fp_radius, good)] = False

        current_extent(azmins, azmaxs, aztimes, corners, fp_radius, el, azs, els,
                       rising, tstop)

        if not np.any(to_cross):
            # All corners made it across the CES line.
            success = True
            break

    return success, azmins, azmaxs, aztimes, tstop


def unwind_angle(alpha, beta):
    """ Minimize absolute difference between alpha and beta.

    Minimize the absolute difference by adding a multiple of
    2*pi to beta to match alpha.
    """
    while np.abs(alpha-beta-2*np.pi) < np.abs(alpha-beta):
        beta += 2*np.pi
    while np.abs(alpha-beta+2*np.pi) < np.abs(alpha-beta):
        beta -= 2*np.pi
    return beta


def scan_patch_pole(args, el, corners, t, fp_radius, observer, sun, not_visible,
               name, tstep, stop_timestamp, sun_el_max, rising):
    """ Attempt scanning the patch specified by corners at elevation el.

    The pole scheduling mode will not wait for the patch to drift across.
    """
    success = False
    # and now track when all corners are past the elevation
    tstop = t
    azmins, azmaxs, aztimes = [], [], []
    azs, els = corner_coordinates(observer, corners)
    while True:
        tstop += tstep / 10
        if tstop - t >= args.pole_ces_time:
            # Succesfully scanned the maximum time
            if len(azmins) > 0:
                success = True
            break
        if tstop > stop_timestamp or tstop - t > 86400:
            not_visible.append((name, 'Ran out of time rising = {}'
                                ''.format(rising)))
            break
        observer.date = to_DJD(tstop)
        sun.compute(observer)
        if sun.alt > sun_el_max:
            not_visible.append((name, 'Sun too high {:.2f} rising = {}'
                                ''.format(sun.alt/degree, rising)))
            break
        azs, els = corner_coordinates(observer, corners)
        if fp_radius == 0:
            radius = np.radians(1)
        else:
            radius = fp_radius
        current_extent_pole(
            azmins, azmaxs, aztimes, corners, radius, el, azs, els, tstop)

    return success, azmins, azmaxs, aztimes, tstop


def current_extent_pole(
        azmins, azmaxs, aztimes, corners, fp_radius, el, azs, els, tstop):
    """ Get the azimuthal extent of the patch along elevation el.

    Pole scheduling does not care if the patch is "rising" or "setting".
    """
    azs_cross = []
    for i in range(len(corners)):
        if np.abs(els[i]-el) < fp_radius:
            azs_cross.append(azs[i])
        j = (i + 1) % len(corners)
        if np.abs(els[j]-el) < fp_radius:
            azs_cross.append(azs[j])
        if np.abs(els[i]-el) < fp_radius \
           or np.abs(els[j]-el) < fp_radius:
            continue
        elif (els[i] - el)*(els[j] - el) < 0:
            # Record the location where a line between the corners
            # crosses el.
            az1 = azs[i]
            az2 = azs[j]
            el1 = els[i] - el
            el2 = els[j] - el
            if az2 - az1 > np.pi:
                az1 += 2*np.pi
            if az1 - az2 > np.pi:
                az2 += 2*np.pi
            az_cross = (az1 + el1*(az2 - az1)/(el1 - el2)) % (2*np.pi)
            azs_cross.append(az_cross)

    # Translate the azimuths at multiples of 2pi so they are in a
    # compact cluster

    for i in range(1, len(azs_cross)):
        azs_cross[i] = unwind_angle(azs_cross[0], azs_cross[i])

    if len(azs_cross) > 0:
        # DEBUG begin
        if np.ptp(np.array(azs_cross)) > np.pi or \
           np.ptp(np.array(azs_cross)) < np.radians(50):
            import pdb
            pdb.set_trace()
        # DEBUG end

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

    return


def current_extent(azmins, azmaxs, aztimes, corners, fp_radius, el, azs, els,
                   rising, tstop):
    """ Get the azimuthal extent of the patch along elevation el.

    Find the pairs of corners that are on opposite sides
    of the CES line.  Record the crossing azimuth of a
    line between the corners.

    """
    azs_cross = []
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        for el0 in [el-fp_radius, el, el+fp_radius]:
            if (els[i] - el0)*(els[j] - el0) < 0:
                az1 = azs[i]
                az2 = azs[j]
                el1 = els[i] - el0
                el2 = els[j] - el0
                if az2 - az1 > np.pi:
                    az1 += 2*np.pi
                if az1 - az2 > np.pi:
                    az2 += 2*np.pi
                az_cross = (az1 + el1*(az2 - az1)/(el1 - el2)) % (2*np.pi)
                azs_cross.append(az_cross)

    if len(azs_cross) > 1:
        azs_cross = np.sort(azs_cross)
        azmin = azs_cross[0]
        azmax = azs_cross[-1]
        if azmax - azmin > np.pi:
            # Patch crosses the zero meridian
            azmin, azmax = azmax, azmin
        azmins.append(azmin)
        azmaxs.append(azmax)
        aztimes.append(tstop)

    return


def add_scan(args, t, tstop, aztimes, azmins, azmaxs, rising, fp_radius,
             observer, sun, moon, fout, fout_fmt, hits, name, el, subscan=-1):
    """ Make an entry for a CES in the schedule file.
    """
    ces_time = tstop - t
    if ces_time > args.ces_max_time:
        nsub = np.int(np.ceil(ces_time / args.ces_max_time))
        ces_time /= nsub
    aztimes = np.array(aztimes)
    azmins = np.array(azmins)
    azmaxs = np.array(azmaxs)
    for i in range(1, azmins.size):
        azmins[i] = unwind_angle(azmins[0], azmins[i])
        azmaxs[i] = unwind_angle(azmaxs[0], azmaxs[i])
    #for i in range(azmins.size-1):
    #    if azmins[i+1] - azmins[i] > np.pi:
    #        azmins[i+1], azmaxs[i+1] = azmins[i+1]-2*np.pi, azmaxs[i+1]-2*np.pi
    #    if azmins[i+1] - azmins[i] < np.pi:
    #        azmins[i+1], azmaxs[i+1] = azmins[i+1]+2*np.pi, azmaxs[i+1]+2*np.pi
    rising_string = 'R' if rising else 'S'
    hits[name] += 1
    t1 = t
    while t1 < tstop:
        subscan += 1
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
        # Add the focal plane radius to the scan width
        fp_radius_eff = fp_radius / np.cos(el)
        azmin = (azmin - fp_radius_eff) % (2*np.pi)
        azmax = (azmax + fp_radius_eff) % (2*np.pi)
        # DEBUG begin
        #if np.abs(azmin-2*np.pi) < .1:
        #    import pdb
        #    pdb.set_trace()
        # DEBUG end
        ces_start = datetime.utcfromtimestamp(t1).strftime(
            '%Y-%m-%d %H:%M:%S %Z')
        ces_stop = datetime.utcfromtimestamp(t2).strftime(
            '%Y-%m-%d %H:%M:%S %Z')
        # Get the Sun and Moon locations at the beginning and end
        observer.date = to_DJD(t1)
        sun.compute(observer)
        moon.compute(observer)
        sun_az1, sun_el1 = sun.az/degree, sun.alt/degree
        moon_az1, moon_el1 = moon.az/degree, moon.alt/degree
        moon_phase1 = moon.phase
        observer.date = to_DJD(t2)
        sun.compute(observer)
        moon.compute(observer)
        sun_az2, sun_el2 = sun.az/degree, sun.alt/degree
        moon_az2, moon_el2 = moon.az/degree, moon.alt/degree
        moon_phase2 = moon.phase
        # Create an entry in the schedule
        fout.write(
            fout_fmt.format(
                ces_start, ces_stop, to_MJD(t1), to_MJD(t2),
                name,
                azmin/degree, azmax/degree, el/degree,
                rising_string,
                sun_el1, sun_az1, sun_el2, sun_az2,
                moon_el1, moon_az1, moon_el2, moon_az2,
                0.005*(moon_phase1 + moon_phase2), hits[name], subscan))
        t1 = t2 + args.gap_small
    # Advance the time
    t = tstop
    # Add the gap
    t += args.gap

    return t

def get_visible(observer, sun, moon, patches, el_min, sun_angle_min,
                sun_avoidance_angle, moon_angle_min):
    """ Determine which patches are visible.
    """
    visible = []
    not_visible = []
    for (name, weight, corners) in patches:
        # Reject all patches that have even one corner too close
        # to the Sun or the Moon and patches that are completely
        # below the horizon
        in_view = False
        for i, corner in enumerate(corners):
            corner.compute(observer)
            if corner.alt > el_min:
                # At least one corner is visible
                in_view = True
            if sun.alt > sun_avoidance_angle:
                # Sun is high enough to apply sun_angle_min check
                angle = ephem.separation(sun, corner)
                if angle < sun_angle_min:
                    # Patch is too close to the Sun
                    not_visible.append((
                        name,
                        'Too close to Sun {:.2f}'.format(angle / degree)))
                    in_view = False
                    break
            if moon.alt > 0:
                angle = ephem.separation(moon, corner)
                if angle < moon_angle_min:
                    # Patch is too close to the Moon
                    not_visible.append((
                        name,
                        'Too close to Moon {:.2f}'.format(angle / degree)))
                    in_view = False
                    break
            if i == len(corners)-1 and not in_view:
                not_visible.append((
                    name, 'Below the horizon.'))
        if in_view:
            visible.append((name, weight, corners))

    return visible, not_visible


def build_schedule(
        args, start_timestamp, stop_timestamp,
        sun_el_max, sun_avoidance_angle,
        sun_angle_min, moon_angle_min,
        el_min, el_max, fp_radius, patches):

    fout = open(args.out, 'w')

    fout.write('#{:15} {:15} {:15} {:15} {:15}\n'.format(
        'Site', 'Telescope',
        'Latitude [deg]', 'Longitude [deg]', 'Altitude [m]'))
    fout.write(' {:15} {:15} {:15} {:15} {:15.6f}\n'.format(
        args.site_name, args.telescope,
        args.site_lat, args.site_lon, args.site_alt))

    fout_fmt0 = '#{:20} {:20} {:14} {:14} ' \
                '{:15} {:8} {:8} {:8} {:5} ' \
                '{:8} {:8} {:8} {:8} ' \
                '{:8} {:8} {:8} {:8} {:5} ' \
                '{:5} {:3}\n'

    fout_fmt = ' {:20} {:20} {:14.6f} {:14.6f} ' \
               '{:15} {:8.2f} {:8.2f} {:8.2f} {:5} ' \
               '{:8.2f} {:8.2f} {:8.2f} {:8.2f} ' \
               '{:8.2f} {:8.2f} {:8.2f} {:8.2f} {:5.2f} ' \
               '{:5} {:3}\n'

    fout.write(
        fout_fmt0.format(
            'Start time UTC', 'Stop time UTC', 'Start MJD', 'Stop MJD',
            'Patch name', 'Az min', 'Az max', 'El', 'R/S',
            'Sun el1', 'Sun az1', 'Sun el2', 'Sun az2',
            'Moon el1', 'Moon az1', 'Moon el2', 'Moon az2', 'Phase',
            'Pass', 'Sub'))

    observer = ephem.Observer()
    observer.lon = args.site_lon
    observer.lat = args.site_lat
    observer.elevation = args.site_alt # In meters
    observer.epoch = '2000'
    observer.temp = 0 # in Celcius
    observer.compute_pressure()

    hits = {}
    for patch in patches:
        hits[patch[0]] = 0

    t = start_timestamp
    sun = ephem.Sun()
    moon = ephem.Moon()
    tstep = 600

    while t < stop_timestamp:
        # Determine which patches are observable at time t.

        if args.debug:
            tstring = datetime.utcfromtimestamp(t).strftime(
                '%Y-%m-%d %H:%M:%S %Z')
            print('t =  {}'.format(tstring), flush=True)
        # Determine which patches are visible
        observer.date = to_DJD(t)
        sun.compute(observer)
        if sun.alt > sun_el_max:
            if args.debug:
                print('Sun elevation is {:.2f} > {:.2f}. Moving on.'.format(
                    sun.alt/degree, sun_el_max/degree), flush=True)
            t += tstep
            continue
        moon.compute(observer)

        visible, not_visible = get_visible(
            observer, sun, moon, patches, el_min, sun_angle_min,
            sun_avoidance_angle, moon_angle_min)

        if len(visible) == 0:
            if args.debug:
                tstring = datetime.utcfromtimestamp(t).strftime(
                    '%Y-%m-%d %H:%M:%S %Z')
                print('No patches visible at {}: {}'.format(tstring, not_visible))
            t += tstep
            continue

        # Order the targets by priority and attempt to observe with both
        # a rising and setting scans until we find one that can be
        # succesfully scanned.
        # If the criteria are not met, advance the time by a step
        # and try again

        prioritize(visible, hits)

        success, t = attempt_scan(
            args, observer, visible, not_visible, t, fp_radius, el_max, el_min,
            tstep, stop_timestamp, sun, moon, sun_el_max, hits, fout, fout_fmt)

        if not success:
            if args.debug:
                tstring = datetime.utcfromtimestamp(t).strftime(
                    '%Y-%m-%d %H:%M:%S %Z')
                print('No patches could be scanned at {}: {}'.format(
                    tstring, not_visible), flush=True)
            t += tstep

    fout.close()

    return


def parse_args():

    parser = argparse.ArgumentParser(
        description='Generate ground observation schedule.',
        fromfile_prefix_chars='@')

    parser.add_argument('--site_name',
                        required=False, default='LBL',
                        help='Observing site name')
    parser.add_argument('--telescope',
                        required=False, default='Telescope',
                        help='Observing telescope name')
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
                        'name,weight,lon1,lat1,lon2,lat2 ... '
                        'OR name,weight,lon,lat,width')
    parser.add_argument('--patch_coord',
                        required=False, default='C',
                        help='Sky patch coordinate system [C,E,G]')
    parser.add_argument('--el_min',
                        required=False, default=30.0, type=np.float,
                        help='Minimum elevation for a CES')
    parser.add_argument('--el_max',
                        required=False, default=80.0, type=np.float,
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
                        required=False, default=20.0, type=np.float,
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
    parser.add_argument('--gap_small',
                        required=False, default=10, type=np.float,
                        help='Gap between split CES:es [seconds]')
    parser.add_argument('--ces_max_time',
                        required=False, default=900, type=np.float,
                        help='Maximum length of a CES [seconds]')
    parser.add_argument('--debug',
                        required=False, default=False, action='store_true',
                        help='Write diagnostics')
    parser.add_argument('--pole_mode',
                        required=False, default=False, action='store_true',
                        help='Pole scheduling mode (no drift scan)')
    parser.add_argument('--pole_el_step',
                        required=False, default=0.25, type=np.float,
                        help='Elevation step in pole scheduling mode [deg]')
    parser.add_argument('--pole_ces_time',
                        required=False, default=3000, type=np.float,
                        help='Time to scan at constant elevation in pole mode')
    parser.add_argument('--out',
                        required=False, default='schedule.txt',
                        help='Output filename')

    args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))

    try:
        start_time = dateutil.parser.parse(args.start + ' +0000')
        stop_time = dateutil.parser.parse(args.stop + ' +0000')
    except:
        start_time = dateutil.parser.parse(args.start)
        stop_time = dateutil.parser.parse(args.stop)

    start_timestamp = start_time.timestamp()
    stop_timestamp = stop_time.timestamp()

    return args, start_timestamp, stop_timestamp


def parse_patch_explicit(args, parts):
    """ Parse an explicit patch definition line
    """
    corners = []
    print('Explicit-corners format ', end='')
    while i < len(parts):
        print(' ({}, {})'.format(parts[2], parts[3]), end='')
        try:
            # Assume coordinates in degrees
            lon = float(parts[2]) * degree
            lat = float(parts[3]) * degree
        except:
            # Failed simple interpreration, assume pyEphem strings
            lon = parts[2]
            lat = parts[3]
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

    return corners


def parse_patch_rectangular(args, parts):
    """ Parse a rectangular patch definition line
    """
    corners = []
    print('Rectangular format ', end='')
    try:
        # Assume coordinates in degrees
        lon_min = float(parts[2]) * degree
        lat_max = float(parts[3]) * degree
        lon_max = float(parts[4]) * degree
        lat_min = float(parts[5]) * degree
    except:
        # Failed simple interpreration, assume pyEphem strings
        lon_min = parts[2]
        lat_max = parts[3]
        lon_max = parts[4]
        lat_min = parts[5]
    if args.patch_coord == 'C':
        coordconv = ephem.Equatorial
    elif args.patch_coord == 'E':
        coordconv = ephem.Ecliptic
    elif args.patch_coord == 'G':
        coordconv = ephem.Galactic
    else:
        raise RuntimeError('Unknown coordinate system: {}'.format(
            args.patch_coord))

    nw_corner = coordconv(lon_min, lat_max, epoch='2000')
    ne_corner = coordconv(lon_max, lat_max, epoch='2000')
    se_corner = coordconv(lon_max, lat_min, epoch='2000')
    sw_corner = coordconv(lon_min, lat_min, epoch='2000')

    corners_temp = []
    add_side(nw_corner, ne_corner, corners_temp, coordconv)
    add_side(ne_corner, se_corner, corners_temp, coordconv)
    add_side(se_corner, sw_corner, corners_temp, coordconv)
    add_side(sw_corner, nw_corner, corners_temp, coordconv)

    for corner in corners_temp:
        if corner.dec > 80*degree or corner.dec < -80*degree:
            raise RuntimeError(
                '{} has at least one circumpolar corner. '
                'Circumpolar targeting not yet implemented'.format(name))
        patch_corner = ephem.FixedBody()
        patch_corner._ra = corner.ra
        patch_corner._dec = corner.dec
        corners.append(patch_corner)

    return corners


def add_side(corner1, corner2, corners_temp, coordconv):
    """ Add one side of a rectangle.

    Add one side of a rectangle with enough interpolation points.
    """
    step = 5 * degree
    corners_temp.append(ephem.Equatorial(corner1))
    lon1 = corner1.ra
    lon2 = corner2.ra
    lat1 = corner1.dec
    lat2 = corner2.dec
    if lon1 == lon2:
        lon = lon1
        ninterp = int(np.abs(lat2 - lat1) // step)
        if ninterp > 0:
            interp_step = (lat2 - lat1) / (ninterp + 1)
            for iinterp in range(ninterp):
                lat = lat1 + iinterp*interp_step
                corners_temp.append(
                    ephem.Equatorial(coordconv(lon, lat, epoch='2000')))
    elif lat1 == lat2:
        lat = lat1
        ninterp = int(np.abs(lon2 - lon1) // step)
        if ninterp > 0:
            interp_step = (lon2 - lon1) / (ninterp + 1)
            for iinterp in range(ninterp):
                lon = lon1 + iinterp*interp_step
                corners_temp.append(
                    ephem.Equatorial(coordconv(lon, lat, epoch='2000')))
    else:
        raise RuntimeError('add_side: both latitude and longitude change')

    return


def parse_patch_center_and_width(args, parts):
    """ Parse center-and-width patch definition
    """
    corners = []
    print('Center-and-width format ', end='')
    try:
        # Assume coordinates in degrees
        lon = float(parts[2]) * degree
        lat = float(parts[3]) * degree
    except:
        # Failed simple interpreration, assume pyEphem strings
        lon = parts[2]
        lat = parts[3]
    width = float(parts[4]) * degree
    if args.patch_coord == 'C':
        center = ephem.Equatorial(lon, lat, epoch='2000')
    elif args.patch_coord == 'E':
        center = ephem.Ecliptic(lon, lat, epoch='2000')
    elif args.patch_coord == 'G':
        center = ephem.Galactic(lon, lat, epoch='2000')
    else:
        raise RuntimeError('Unknown coordinate system: {}'.format(
            args.patch_coord))
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


def parse_patches(args):
    # Parse the patch definitions

    patches = []
    total_weight = 0
    for patch_def in args.patch:
        parts = patch_def.split(',')
        name = parts[0]
        weight = float(parts[1])
        total_weight += weight
        print('Adding patch "{}" {} '.format(name, weight), end='')
        if len(parts[2:]) == 3:
            corners = parse_patch_center_and_width(args, parts)
        elif len(parts[2:]) == 4:
            corners = parse_patch_rectangular(args, parts)
        else:
            corners = parse_patch_explicit(args, parts)
        print('')
        patches.append([name, weight, corners])

    if args.debug:
        import matplotlib.pyplot as plt
        import healpy as hp
        plt.figure(figsize=[18, 12])
        for iplot, coord in enumerate('CEG'):
            hp.mollview(np.zeros(12), coord=coord, cbar=False,
                        title='Patch locations', sub=[2, 2, 1+iplot])
            hp.graticule(30)
            for name, weight, corners in patches:
                lon = [corner._ra/degree for corner in corners]
                lat = [corner._dec/degree for corner in corners]
                lon.append(lon[0])
                lat.append(lat[0])
                print('{} corners:\n lon = {}\n lat= {}'.format(name, lon, lat),
                      flush=True)
                hp.projplot(lon, lat, 'r-', threshold=1, lonlat=True, coord='C',
                            lw=2)
                hp.projtext(lon[0], lat[0], name, lonlat=True, coord='C',
                            fontsize=14)
        plt.savefig('patches.png')
        plt.close()

    # Normalize the weights
    for i in range(len(patches)):
        patches[i][1] /= total_weight

    return patches


def main():

    args, start_timestamp, stop_timestamp = parse_args()

    patches = parse_patches(args)
    autotimer = timing.auto_timer(timing.FILE())

    build_schedule(
        args, start_timestamp, stop_timestamp,
        args.sun_el_max*degree, args.sun_avoidance_angle*degree,
        args.sun_angle_min*degree, args.moon_angle_min*degree,
        args.el_min*degree, args.el_max*degree, args.fp_radius*degree, patches)


if __name__ == '__main__':
    main()
    tman = timing.timing_manager()
    tman.report()
