# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import datetime

import ephem
import numpy as np
from astropy import units as u

from ..intervals import IntervalList
from ..timing import Timer, function_timer


@function_timer
def scan_time(coord_in, coord_out, scanrate, scan_accel):
    """Compute time to scan between two coordinates.

    Given a coordinate range and scan parameters, determine the time taken to scan
    between them, assuming that the scan begins and ends at rest.

    """
    d = np.abs(coord_in - coord_out)
    t_accel = scanrate / scan_accel
    d_accel = 0.5 * scan_accel * t_accel**2
    if 2 * d_accel > d:
        # No time to reach scan speed
        d_accel = d / 2
        t_accel = np.sqrt(2 * d_accel / scan_accel)
        t_coast = 0
    else:
        d_coast = d - 2 * d_accel
        t_coast = d_coast / scanrate
    return 2 * t_accel + t_coast


@function_timer
def scan_profile(coord_in, coord_out, scanrate, scan_accel, times, nstep=10000):
    """scan between the coordinates assuming that the scan begins
    and ends at rest.  If there is more time than is needed, wait at the end.
    """
    if np.abs(coord_in - coord_out) < 1e-6:
        return np.zeros(len(times)) + coord_out

    d = np.abs(coord_in - coord_out)
    t_accel = scanrate / scan_accel
    d_accel = 0.5 * scan_accel * t_accel**2
    if 2 * d_accel > d:
        # No time to reach scan speed
        d_accel = d / 2
        t_accel = np.sqrt(2 * d_accel / scan_accel)
        t_coast = 0
        # Scan rate at the end of acceleration
        scanrate = t_accel * scan_accel
    else:
        d_coast = d - 2 * d_accel
        t_coast = d_coast / scanrate
    if coord_in > coord_out:
        scanrate *= -1
        scan_accel *= -1
    #
    t = []
    coord = []
    # Acceleration
    t.append(np.linspace(times[0], times[0] + t_accel, nstep))
    coord.append(coord_in + 0.5 * scan_accel * (t[-1] - t[-1][0]) ** 2)
    # Coasting
    if t_coast > 0:
        t.append(np.linspace(t[-1][-1], t[-1][-1] + t_coast, 3))
        coord.append(coord[-1][-1] + scanrate * (t[-1] - t[-1][0]))
    # Deceleration
    t.append(np.linspace(t[-1][-1], t[-1][-1] + t_accel, nstep))
    coord.append(
        coord[-1][-1]
        + scanrate * (t[-1] - t[-1][0])
        - 0.5 * scan_accel * (t[-1] - t[-1][0]) ** 2
    )
    # Wait
    if t[-1][-1] < times[-1]:
        t.append(np.linspace(t[-1][-1], times[-1], 3))
        coord.append(np.zeros(3) + coord_out)

    # Interpolate to the given time stamps
    t = np.hstack(t)
    coord = np.hstack(coord)

    return np.interp(times, t, coord)


@function_timer
def scan_between(
    time_start,
    az1,
    el1,
    az2,
    el2,
    az_rate,
    az_accel,
    el_rate,
    el_accel,
    nstep=10000,
):
    """Simulate motion between two coordinates.

    Using the specified Az / El rate and acceleration, simulate the motion in both
    coordinates from the one point to the other.

    Args:

    Returns:
        (tuple):  The (times, az, el) arrays.

    """
    az_time = scan_time(az1, az2, az_rate, az_accel)
    el_time = scan_time(el1, el2, el_rate, el_accel)
    time_tot = max(az_time, el_time)
    times = np.linspace(0, time_tot, nstep)
    az = scan_profile(az1, az2, az_rate, az_accel, times, nstep=nstep)
    el = scan_profile(el1, el2, el_rate, el_accel, times, nstep=nstep)
    return times + time_start, az, el


@function_timer
def simulate_elnod(
    t_start,
    rate,
    az_start,
    el_start,
    az_rate,
    az_accel,
    el_rate,
    el_accel,
    elnod_el,
    elnod_az,
    scan_min_az,
    scan_max_az,
    scan_min_el,
    scan_max_el,
):
    """Simulate an el-nod.

    Args:
        t_start (float):  The start time in seconds.
        az_start ()

    Returns:
        ()
    """

    time_last = t_start
    az_last = az_start
    el_last = el_start
    t = []
    az = []
    el = []
    for az_new, el_new in zip(elnod_az, elnod_el):
        if np.abs(az_last - az_new) > 1e-3 or np.abs(el_last - el_new) > 1e-3:
            tvec, azvec, elvec = scan_between(
                time_last,
                az_last,
                el_last,
                az_new,
                el_new,
                az_rate,
                az_accel,
                el_rate,
                el_accel,
            )
            t.append(tvec)
            az.append(azvec)
            el.append(elvec)
            time_last = tvec[-1]
        az_last = az_new
        el_last = el_new

    t = np.hstack(t)
    az = np.hstack(az)
    el = np.hstack(el)

    # Store the scan range.  We use the high resolution elevation
    # so actual sampling rate will not change the range.
    scan_min_az = min(scan_min_az, np.min(az))
    scan_max_az = min(scan_max_az, np.max(az))
    scan_min_el = min(scan_min_el, np.min(el))
    scan_max_el = min(scan_max_el, np.max(el))

    # Sample t/az/el down to the sampling rate
    nsample_elnod = int((t[-1] - t[0]) * rate)
    t_sample = np.arange(nsample_elnod) / rate + t_start
    az_sample = np.interp(t_sample, t, az)
    el_sample = np.interp(t_sample, t, el)

    return (
        t_sample,
        az_sample,
        el_sample,
        scan_min_az,
        scan_max_az,
        scan_min_el,
        scan_max_el,
    )


@function_timer
def oscillate_el(
    times,
    el,
    el_rate,
    el_accel,
    scan_min_el,
    scan_max_el,
    el_mod_amplitude,
    el_mod_rate,
    el_mod_sine=False,
):
    """Simulate oscillating elevation.

    The array of EL values is modified in place.  Updated min / max EL range is
    returned.

    Args:

    Returns:
        (tuple):  The new (min, max) range of the elevation.

    """
    tt = times - times[0]
    # Shift the starting time by a random phase
    np.random.seed(int(times[0] % 2**32))
    tt += np.random.rand() / el_mod_rate

    if el_mod_sine:
        # elevation is modulated along a sine wave
        angular_rate = 2 * np.pi * el_mod_rate
        el += el_mod_amplitude * np.sin(tt * angular_rate)

        # Check that we did not breach tolerances
        el_rate_max = np.amax(np.abs(np.diff(el)) / np.diff(tt))
        if np.any(el_rate_max > el_rate):
            raise RuntimeError(
                "Elevation oscillation requires {:.2f} deg/s but "
                "mount only allows {:.2f} deg/s".format(
                    np.degrees(el_rate_max), np.degrees(el_rate)
                )
            )
    else:
        # elevation is modulated using a constant rate.  We need to
        # calculate the appropriate scan rate to achieve the desired
        # amplitude and period
        t_mod = 1 / el_mod_rate
        # determine scan rate needed to reach given amplitude in given time
        a = el_accel
        b = -0.5 * el_accel * t_mod
        c = 2 * el_mod_amplitude
        if b**2 - 4 * a * c < 0:
            raise RuntimeError(
                "Cannot perform {:.2f} deg elevation oscillation in {:.2f} s "
                "with {:.2f} deg/s^2 acceleration".format(
                    np.degrees(el_mod_amplitude * 2),
                    np.degrees(el_accel),
                )
            )
        root1 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        root2 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        if root1 > 0:
            t_accel = root1
        else:
            t_accel = root2
        t_scan = 0.5 * t_mod - 2 * t_accel
        scanrate = t_accel * el_accel
        if scanrate > el_rate:
            raise RuntimeError(
                "Elevation oscillation requires {:.2f} > {:.2f} deg/s "
                "scan rate".format(
                    np.degrees(scanrate),
                    np.degrees(el_rate),
                )
            )

        # simulate a high resolution scan to interpolate

        t_interp = []
        el_interp = []
        n = 1000
        # accelerate
        t = np.linspace(0, t_accel, n)
        t_interp.append(t)
        el_interp.append(0.5 * el_accel * t**2)
        # scan
        t_last = t_interp[-1][-1]
        el_last = el_interp[-1][-1]
        t = np.linspace(0, t_scan, 2)
        t_interp.append(t_last + t)
        el_interp.append(el_last + t * scanrate)
        # decelerate
        t_last = t_interp[-1][-1]
        el_last = el_interp[-1][-1]
        t = np.linspace(0, 2 * t_accel, n)
        t_interp.append(t_last + t)
        el_interp.append(el_last + scanrate * t - 0.5 * el_accel * t**2)
        # scan
        t_last = t_interp[-1][-1]
        el_last = el_interp[-1][-1]
        t = np.linspace(0, t_scan, 2)
        t_interp.append(t_last + t)
        el_interp.append(el_last - t * scanrate)
        # decelerate
        t_last = t_interp[-1][-1]
        el_last = el_interp[-1][-1]
        t = np.linspace(0, t_accel, n)
        t_interp.append(t_last + t)
        el_interp.append(el_last - scanrate * t + 0.5 * el_accel * t**2)

        t_interp = np.hstack(t_interp)
        el_interp = np.hstack(el_interp)

        # finally, modulate the elevation around the input vector
        # start time is set to middle of the first stable scan
        # *** without a proper accelerating phase ***
        tt += t_accel + 0.5 * t_scan
        el += np.interp(tt % t_mod, t_interp, el_interp) - el_mod_amplitude

    new_min_el = min(scan_min_el, np.min(el))
    new_max_el = max(scan_max_el, np.max(el))

    return new_min_el, new_max_el


@function_timer
def step_el(
    times,
    az,
    el,
    el_rate,
    el_accel,
    scan_min_el,
    scan_max_el,
    el_mod_step,
    n=1000,
):
    """Simulate elevation steps after each scan pair"""

    sign = np.sign(el_mod_step)
    el_step = np.abs(el_mod_step)

    # simulate a single elevation step at high resolution

    t_accel = el_rate / el_accel
    el_step_accel = 0.5 * el_accel * t_accel**2
    if el_step > 2 * el_step_accel:
        # Step is large enough to reach elevation scan rate
        t_scan = (el_step - 2 * el_step_accel) / el_rate
    else:
        # Only partial acceleration and deceleration
        el_step_accel = np.abs(el_mod_step) / 2
        t_accel = np.sqrt(2 * el_step_accel / el_accel)
        t_scan = 0

    t_interp = []
    el_interp = []

    # accelerate
    t = np.linspace(0, t_accel, n)
    t_interp.append(t)
    el_interp.append(0.5 * el_accel * t**2)
    # scan
    if t_scan > 0:
        t_last = t_interp[-1][-1]
        el_last = el_interp[-1][-1]
        t = np.linspace(0, t_scan, n)
        t_interp.append(t_last + t)
        el_interp.append(el_last + t * el_rate)
    # decelerate
    t_last = t_interp[-1][-1]
    el_last = el_interp[-1][-1]
    el_rate_last = el_accel * t_accel
    t = np.linspace(0, t_accel, n)
    t_interp.append(t_last + t)
    el_interp.append(el_last + el_rate_last * t - 0.5 * el_accel * t**2)

    t_interp = np.hstack(t_interp)
    t_interp -= t_interp[t_interp.size // 2]
    el_interp = sign * np.hstack(el_interp)

    # isolate steps

    daz = np.diff(az)
    ind = np.where(daz[1:] * daz[:-1] < 0)[0] + 1
    ind = ind[1::2]

    # Modulate the elevation at each step

    for istep in ind:
        tstep = times[istep]
        el += np.interp(times - tstep, t_interp, el_interp)

    new_min_el = min(scan_min_el, np.min(el))
    new_max_el = max(scan_max_el, np.max(el))

    return new_min_el, new_max_el


@function_timer
def simulate_ces_scan(
    t_start,
    t_stop,
    rate,
    el,
    az_min,
    az_max,
    az_start,
    az_rate,
    fix_rate_on_sky,
    az_accel,
    scan_min_az,
    scan_max_az,
    cosecant_modulation=False,
    nstep=10000,
    randomize_phase=False,
):
    """Simulate a constant elevation scan."""

    # if samples <= 0:
    #     raise RuntimeError("CES requires a positive number of samples")
    #
    # if len(self._times) == 0:
    #     self._CES_start = self._firsttime
    # else:
    #     self._CES_start = self._times[-1] + 1 / self._rate

    # Begin by simulating one full scan with turnarounds at high sampling
    # It will be used to interpolate the full CES.

    ##azmin, azmax = [self._azmin_ces, self._azmax_ces]

    mirror_cosecant = False
    if cosecant_modulation:
        # We always simulate a rising cosecant scan and then
        # mirror it if necessary
        if az_min > np.pi:
            mirror_cosecant = True
        az_min %= np.pi
        az_max %= np.pi
        if az_min > az_max:
            raise RuntimeError(
                "Cannot scan across zero meridian with cosecant-modulated scan"
            )
    elif az_max < az_min:
        az_max += 2 * np.pi

    # The lists of data arrays, to be concatenated at the end.
    all_t = list()
    all_az = list()
    all_flags = list()

    if fix_rate_on_sky:
        # translate scan rate from sky to mount coordinates
        base_rate = az_rate / np.cos(el)
    else:
        # azimuthal rate is already in mount coordinates
        base_rate = az_rate
    # scan acceleration is already in the mount coordinates
    scan_accel = az_accel

    # left-to-right

    t = t_start
    tvec = None
    azvec = None
    t0 = t
    if cosecant_modulation:
        t1 = t0 + (np.cos(az_min) - np.cos(az_max)) / base_rate
        tvec = np.linspace(t0, t1, nstep, endpoint=True)
        azvec = np.arccos(np.cos(az_min) + base_rate * t0 - base_rate * tvec)
    else:
        # Constant scanning rate, only requires two data points
        t1 = t0 + (az_max - az_min) / base_rate
        tvec = np.array([t0, t1])
        azvec = np.array([az_min, az_max])
    all_t.append(np.array(tvec))
    all_az.append(np.array(azvec))
    range_scan_leftright = (t0, t1)

    # turnaround

    t = t1
    t0 = t
    if cosecant_modulation:
        dazdt = base_rate / np.abs(np.sin(az_max))
    else:
        dazdt = base_rate
    t1 = t0 + 2 * dazdt / scan_accel
    tvec = np.linspace(t0, t1, nstep, endpoint=True)[1:]
    azvec = az_max + (tvec - t0) * dazdt - 0.5 * scan_accel * (tvec - t0) ** 2
    all_t.append(np.array(tvec[:-1]))
    all_az.append(np.array(azvec[:-1]))
    range_turn_leftright = (t0, t1)

    # right-to-left

    t = t1
    tvec = []
    azvec = []
    t0 = t
    if cosecant_modulation:
        t1 = t0 + (np.cos(az_min) - np.cos(az_max)) / base_rate
        tvec = np.linspace(t0, t1, nstep, endpoint=True)
        azvec = np.arccos(np.cos(az_max) - base_rate * t0 + base_rate * tvec)
    else:
        # Constant scanning rate, only requires two data points
        t1 = t0 + (az_max - az_min) / base_rate
        tvec = np.array([t0, t1])
        azvec = np.array([az_max, az_min])
    all_t.append(np.array(tvec))
    all_az.append(np.array(azvec))
    range_scan_rightleft = (t0, t1)

    # turnaround

    t = t1
    t0 = t
    if cosecant_modulation:
        dazdt = base_rate / np.abs(np.sin(az_min))
    else:
        dazdt = base_rate
    t1 = t0 + 2 * dazdt / scan_accel
    tvec = np.linspace(t0, t1, nstep, endpoint=True)[1:]
    azvec = az_min - (tvec - t0) * dazdt + 0.5 * scan_accel * (tvec - t0) ** 2
    all_t.append(np.array(tvec))
    all_az.append(np.array(azvec))
    range_turn_rightleft = (t0, t1)

    # Concatenate

    tvec = np.hstack(all_t)
    azvec = np.hstack(all_az)

    # Limit azimuth to [-2pi, 2pi] but do not
    # introduce discontinuities with modulo.

    if np.amin(azvec) < -2 * np.pi:
        azvec += 2 * np.pi
    if np.amax(azvec) > 2 * np.pi:
        azvec -= 2 * np.pi

    if mirror_cosecant:
        # We always simulate a rising cosecant scan and then
        # mirror it if necessary
        azvec += np.pi

    # Update the scan range.  We use the high resolution azimuth so the
    # actual sampling rate will not change the range.

    new_min_az = min(scan_min_az, np.min(azvec))
    new_max_az = max(scan_max_az, np.max(azvec))

    # Now interpolate the simulated scan to timestamps.  The start time and
    # sample rate are enforced and the stop time is adjusted if needed to
    # produce a whole number of samples.

    samples = int((t_stop - t_start) * rate)
    times = t_start + np.arange(samples) / rate

    tmin, tmax = tvec[0], tvec[-1]
    tdelta = tmax - tmin

    if randomize_phase:
        np.random.seed(int(t_start % 2**32))
        t_off = -tdelta * np.random.rand()
    else:
        t_off = 0

    # For interpolation, shift the times to zero
    tvec -= tmin
    t_interp = (times - tmin - t_off) % tdelta

    az_sample = np.interp(t_interp, tvec, azvec)
    el_sample = np.zeros_like(az_sample) + el

    # The time intervals for various types of motion.  These are returned
    # and can be used to construct IntervalLists by the calling code.
    ival_scan_leftright = list()
    ival_scan_rightleft = list()
    ival_turn_leftright = list()
    ival_turn_rightleft = list()
    ival_throw_leftright = list()
    ival_throw_rightleft = list()
    ival_scan = list()

    # Repeat time intervals to cover the timestamps
    n_repeat = 1 + int((times[-1] - tmin) / tdelta)
    for rp in range(n_repeat):
        ival_scan_leftright.append(
            (range_scan_leftright[0] + t_off, range_scan_leftright[1] + t_off)
        )
        ival_turn_leftright.append(
            (range_turn_leftright[0] + t_off, range_turn_leftright[1] + t_off)
        )
        ival_scan_rightleft.append(
            (range_scan_rightleft[0] + t_off, range_scan_rightleft[1] + t_off)
        )
        ival_turn_rightleft.append(
            (range_turn_rightleft[0] + t_off, range_turn_rightleft[1] + t_off)
        )
        half_turn_leftright = 0.5 * (range_turn_leftright[1] - range_turn_leftright[0])
        half_turn_rightleft = 0.5 * (range_turn_rightleft[1] - range_turn_rightleft[0])
        ival_throw_leftright.append(
            (
                range_scan_leftright[0] + t_off - half_turn_rightleft,
                range_scan_leftright[1] + t_off + half_turn_leftright,
            )
        )
        ival_throw_rightleft.append(
            (
                range_scan_rightleft[0] + t_off - half_turn_leftright,
                range_scan_rightleft[1] + t_off + half_turn_rightleft,
            )
        )
        t_off += tdelta

    # Trim off the intervals if they extend past the timestamps
    for ival in [
        ival_scan_leftright,
        ival_scan_rightleft,
        ival_turn_leftright,
        ival_turn_rightleft,
        ival_throw_leftright,
        ival_throw_rightleft,
    ]:
        first = tuple(ival[-1])
        if first[1] < times[0]:
            # Whole interval before the start
            del ival[0]
        elif first[0] < times[0]:
            # interval is truncated
            ival[0] = (times[0], first[1])
        last = tuple(ival[-1])
        if last[0] > times[-1]:
            # Whole interval beyond the end
            del ival[-1]
        elif last[1] > times[-1]:
            # interval is truncated
            ival[-1] = (last[0], times[-1])

    return (
        times,
        az_sample,
        el_sample,
        new_min_az,
        new_max_az,
        ival_scan_leftright,
        ival_turn_leftright,
        ival_scan_rightleft,
        ival_turn_rightleft,
        ival_throw_leftright,
        ival_throw_rightleft,
    )


@function_timer
def add_solar_intervals(
    intervals,
    site,
    times,
    az_bore,
    el_bore,
    sun_up_interval,
    sun_close_interval,
    sun_close_distance,
):
    """Get the Sun's position in the horizontal coordinate system and
    translate it into 'Sun up' and 'Sun close' intervals.
    """

    observer = ephem.Observer()
    observer.lon = site.earthloc.lon.to_value(u.radian)
    observer.lat = site.earthloc.lat.to_value(u.radian)
    observer.elevation = site.earthloc.height.to_value(u.meter)
    observer.epoch = ephem.J2000
    observer.compute_pressure()
    observer.pressure = 0

    Sun = ephem.Sun()

    tstart = times[0]
    tstop = times[-1]
    # Motion of the Sun on sky is slow and interpolates well.
    # Get the horizontal position every minute and interpolate
    nstep = int((tstop - tstart) // 60 + 1)
    tvec = np.linspace(tstart, tstop, nstep)
    azvec = []
    elvec = []
    for tstep in tvec:
        observer.date = tstep / 86400.0 + 2440587.5 - 2415020
        Sun.compute(observer)
        azvec.append(Sun.az)
        elvec.append(Sun.alt)
    az_sun = np.interp(times, tvec, azvec)
    el_sun = np.interp(times, tvec, elvec)
    sun_up = el_sun > 0

    cos_lim = np.cos(sun_close_distance)
    cos_dist = np.sin(el_bore) * np.sin(el_sun) + np.cos(el_bore) * np.cos(
        el_sun
    ) * np.cos(az_bore - az_sun)
    sun_close = cos_dist > cos_lim

    # Translate the boolean vectors into intervals
    ivals_up, ivals_close = [], []
    for vec, ivals in zip([sun_up, sun_close], [ivals_up, ivals_close]):
        starts = np.where(np.diff(vec.astype(int)) == 1)[0] + 1
        stops = np.where(np.diff(vec.astype(int)) == -1)[0] + 1
        if vec[0]:
            starts = np.hstack([0, starts])
        if vec[-1]:
            stops = np.hstack([stops, vec.size - 1])
        for start, stop in zip(starts, stops):
            ivals.append((times[start], times[stop]))

    intervals[sun_up_interval] = IntervalList(times, timespans=ivals_up)
    intervals[sun_close_interval] = IntervalList(times, timespans=ivals_close)

    return
