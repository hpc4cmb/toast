#!/usr/bin/env python

# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

# This script creates CES schedule file that can be used as input
# to toast_ground_sim.py

import argparse

import numpy as np

import ephem

from scipy.constants import degree

from datetime import datetime
import dateutil.parser


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

    parser.add_argument('--site_lon',
                        required=False, default=10.0,
                        help='Observing site longitude [pyEphem string]')
    parser.add_argument('--site_lat',
                        required=False, default=10.0,
                        help='Observing site latitude [pyEphem string]')
    parser.add_argument('--site_alt',
                        required=False, default=10.0, type=np.float,
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
    parser.add_argument('--sun_avoidance_angle',
                        required=False, default=-15.0, type=np.float,
                        help='Solar elevation above which to apply '
                        'sun_angle_min [deg]')
    parser.add_argument('--sun_angle_min',
                        required=False, default=90.0, type=np.float,
                        help='Minimum azimuthal distance between the Sun and '
                        'the sky patch [deg]')
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
            lat = (90 - float(parts[i+1])) * degree
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
            patch_corner = ephem.FixedBody()
            patch_corner._ra = corner.ra
            patch_corner._dec = corner.dec
            corners.append(patch_corner)
        print('')
        patches.append([name, weight, corners])

        if args.debug:
            phi = [corner._ra for corner in corners]
            theta = [corner._dec for corner in corners]
            phi.append(phi[0])
            theta.append(theta[0])
            hp.projplot(theta, phi, '-', threshold=1)
            hp.projtext(np.amin(theta)-5*degree,
                        np.amax(phi), name)

    # Normalize the weights
    for i in range(len(patches)):
        patches[i][1] /= total_weight

    if args.debug:
        plt.savefig('patches.png')
        plt.close()

    t = start_timestamp
    gap = args.gap
    sun = ephem.Sun()
    el_min = args.el_min
    tstep = 60
    fout = open('schedule.txt', 'w')
    fout_fmt0 = '#{:20} {:20} {:14} {:14} {:15} ' \
                '{:8} {:8} {:8} {:5} {:5} {:8} {:8} {:5}\n'
                
    fout_fmt = ' {:20} {:20} {:14.6f} {:14.6f} {:15} ' \
               '{:8.2f} {:8.2f} {:8.2f} {:5} {:5} {:8.2f} {:8.2f} {:5}\n'

    fout.write(
        fout_fmt0.format(
            'Start time UTC', 'Stop time UTC', 'Start MJD', 'Stop MJD',
            'Patch name', 'Az min', 'Az max', 'El',
            'R/S 1', 'R/S 2', 'Sun el 1', 'Sun el 2', 'Pass'))
    while t < stop_timestamp:
        # Determine which patches are visible
        observer.date = to_DJD(t)
        sun.compute(observer)
        sun_el = sun.alt / degree
        if sun_el > args.sun_el_max:
            t += tstep
            continue
        visible = []
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
                    in_view = False
                els[i] = corner.alt
                if sun.alt > args.sun_avoidance_angle:
                    # Sun is high enough to apply sun_angle_min check
                    angle = corner.az - sun.az
                    if angle < -2*np.pi:
                        angle += 2*np.pi
                    if angle > 2*np.pi:
                        angle -= 2*np.pi
                    angle = np.abs(angle) / degree
                    if angle < args.sun_angle_min:
                        # Patch is too close to the sun
                        in_view = False
                        break
            if rising and in_view:
                elmax = np.amax(els) / degree
                if elmax < args.el_min:
                    # All corners are too low
                    in_view = False
            if in_view:
                visible.append((name, weight, corners))
        if len(visible) == 0:
            t += tstep
            continue
        # Choose which patch to observe
        selected_name, selected_weight, selected_corners = visible[0]
        nhit = hits[selected_name]
        for name, weight, corners in visible[1:]:
            if hits[name]*selected_weight < nhit*weight:
                selected_name = name
                selected_weight = weight
                selected_corners = corners
                nhit = hits[name]
        hits[selected_name] += 1

        # Determine where to point and how long to observe

        # Start by determining if the patch is rising or setting
        az0 = selected_corners[0].az
        el0 = selected_corners[0].alt
        observer.date = to_DJD(t+100)
        selected_corners[0].compute(observer)
        az1 = selected_corners[0].az
        el1 = selected_corners[0].alt
        rising = el0 < el1
        observer.date = to_DJD(t)
        selected_corners[0].compute(observer)
        # Then determine an elevation that all corners will cross
        ncorner = len(selected_corners)
        azs = np.zeros(ncorner)
        els = np.zeros(ncorner)
        for i, corner in enumerate(selected_corners):
            azs[i] = corner.az
            els[i] = corner.alt
        if rising:
            el = np.amax(els)
        else:
            el = np.amin(els)
        azmin = np.amin(azs)
        azmax = np.amax(azs)
        # and now track when all corners are past the elevation
        tstop = t
        old_el = els[0]
        while True:
            tstop += tstep
            observer.date = to_DJD(tstop)
            sun.compute(observer)
            if sun.alt / degree > args.sun_el_max:
                break
            for i, corner in enumerate(selected_corners):
                corner.compute(observer)
                azs[i] = corner.az
                els[i] = corner.alt
            azmin = min(azmin, np.amin(azs))
            azmax = min(azmax, np.amax(azs))
            # Check if the sun has moved too close
            if sun.alt > args.sun_avoidance_angle:
                for az in [azmin, azmax]:
                    angle = az - sun.az
                    if angle < -2*np.pi:
                        angle += 2*np.pi
                    if angle > 2*np.pi:
                        angle -= 2*np.pi
                    angle = np.abs(angle) / degree
                    if angle < args.sun_angle_min:
                        print('Sun is too close at {}. Breaking CES'.format(t),
                              flush=True)
                        break

            # The patch may change direction without all corners
            # crossing the elevation
            rising_now = old_el < els[0]
            if rising_now:
                if np.amin(els) > el:
                    break
            else:
                if np.amax(els) < el:
                    break
            old_el = els[0]
        ces_start = datetime.utcfromtimestamp(t).strftime(
            '%Y-%m-%d %H:%M:%S %Z')
        ces_stop = datetime.utcfromtimestamp(tstop).strftime(
            '%Y-%m-%d %H:%M:%S %Z')
        # Create an entry in the schedule
        rising_string = 'R' if rising else 'S'
        rising_now_string = 'R' if rising_now else 'S'
        fout.write(
            fout_fmt.format(
                ces_start, ces_stop, to_MJD(t), to_MJD(tstop),
                selected_name,
                azmin/degree, azmax/degree, el/degree,
                rising_string, rising_now_string,
                sun_el, sun.alt/degree, hits[selected_name]))
        # Advance the time
        t = tstop
        # Add the gap
        t += gap

    fout.close()

if __name__ == '__main__':
    main()
