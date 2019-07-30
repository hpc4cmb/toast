# Copyright (c) 2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

import numpy as np

from ..timing import function_timer, Timer
from ..utils import Logger, Environment

from .. import qarray as qa

from ..tod import (
    OpSimAtmosphere,
    atm_available_utils,
)

if atm_available_utils:
    from ..tod.atm import (
        atm_atmospheric_loading,
        atm_absorption_coefficient,
        atm_absorption_coefficient_vec,
    )


XAXIS, YAXIS, ZAXIS = np.eye(3)


def add_atmosphere_args(parser):
    """ Add the atmospheric simulation arguments
    """
    parser.add_argument(
        "--atmosphere",
        required=False,
        action="store_true",
        help="Add simulated atmoshere",
        dest="simulate_atmosphere",
    )
    parser.add_argument(
        "--no-atmosphere",
        required=False,
        action="store_false",
        help="Do not add simulated atmosphere",
        dest="simulate_atmosphere",
    )
    parser.set_defaults(simulate_atmosphere=False)

    parser.add_argument(
        "--focalplane-radius",
        required=False,
        type=np.float,
        help="Override focal plane radius [deg]",
    )
    
    parser.add_argument(
        "--atm-verbosity",
        required=False,
        default=0,
        type=np.int,
        help="Atmospheric sim verbosity level",
    )
    parser.add_argument(
        "--atm-lmin-center",
        required=False,
        default=0.01,
        type=np.float,
        help="Kolmogorov turbulence dissipation scale center",
    )
    parser.add_argument(
        "--atm-lmin-sigma",
        required=False,
        default=0.001,
        type=np.float,
        help="Kolmogorov turbulence dissipation scale sigma",
    )
    parser.add_argument(
        "--atm-lmax-center",
        required=False,
        default=10.0,
        type=np.float,
        help="Kolmogorov turbulence injection scale center",
    )
    parser.add_argument(
        "--atm-lmax-sigma",
        required=False,
        default=10.0,
        type=np.float,
        help="Kolmogorov turbulence injection scale sigma",
    )
    parser.add_argument(
        "--atm-gain",
        required=False,
        default=3e-5,
        type=np.float,
        help="Atmospheric gain factor.",
    )
    parser.add_argument(
        "--atm-zatm",
        required=False,
        default=40000.0,
        type=np.float,
        help="atmosphere extent for temperature profile",
    )
    parser.add_argument(
        "--atm-zmax",
        required=False,
        default=200.0,
        type=np.float,
        help="atmosphere extent for water vapor integration",
    )
    parser.add_argument(
        "--atm-xstep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in X direction",
    )
    parser.add_argument(
        "--atm-ystep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in Y direction",
    )
    parser.add_argument(
        "--atm-zstep",
        required=False,
        default=10.0,
        type=np.float,
        help="size of volume elements in Z direction",
    )
    parser.add_argument(
        "--atm-nelem-sim-max",
        required=False,
        default=1000,
        type=np.int,
        help="controls the size of the simulation slices",
    )
    parser.add_argument(
        "--atm-wind-dist",
        required=False,
        default=500.0,
        type=np.float,
        help="Maximum wind drift to simulate without discontinuity",
    )
    parser.add_argument(
        "--atm-z0-center",
        required=False,
        default=2000.0,
        type=np.float,
        help="central value of the water vapor distribution",
    )
    parser.add_argument(
        "--atm-z0-sigma",
        required=False,
        default=0.0,
        type=np.float,
        help="sigma of the water vapor distribution",
    )
    parser.add_argument(
        "--atm-T0-center",
        required=False,
        default=280.0,
        type=np.float,
        help="central value of the temperature distribution",
    )
    parser.add_argument(
        "--atm-T0-sigma",
        required=False,
        default=10.0,
        type=np.float,
        help="sigma of the temperature distribution",
    )
    parser.add_argument(
        "--atm-cache",
        required=False,
        default="atm_cache",
        help="Atmosphere cache directory",
    )
    # Common flag mask may already be added
    try:
        parser.add_argument(
            "--common-flag-mask",
            required=False,
            default=1,
            type=np.uint8,
            help="Common flag mask",
        )
    except argparse.ArgumentError:
        pass
    # `flush` may already be added
    try:
        parser.add_argument(
            "--flush",
            required=False,
            default=False,
            action="store_true",
            help="Flush every print statement.",
        )
    except argparse.ArgumentError:
        pass
    return


@function_timer
def simulate_atmosphere(args, comm, data, mc, cache_name=None, verbose=True):
    """ Simulate atmospheric signal and add it to `cache_name`.
    """
    if not args.simulate_atmosphere:
        return
    log = Logger.get()
    tmr = Timer()
    tmr.start()
    if comm.world_rank == 0 and verbose:
        log.info("Simulating atmosphere")
        if args.atm_cache and not os.path.isdir(args.atm_cache):
            try:
                os.makedirs(args.atm_cache)
            except FileExistsError:
                pass

    # Simulate the atmosphere signal
    atm = OpSimAtmosphere(
        out=cache_name,
        realization=mc,
        lmin_center=args.atm_lmin_center,
        lmin_sigma=args.atm_lmin_sigma,
        lmax_center=args.atm_lmax_center,
        gain=args.atm_gain,
        lmax_sigma=args.atm_lmax_sigma,
        zatm=args.atm_zatm,
        zmax=args.atm_zmax,
        xstep=args.atm_xstep,
        ystep=args.atm_ystep,
        zstep=args.atm_zstep,
        nelem_sim_max=args.atm_nelem_sim_max,
        verbosity=args.atm_verbosity,
        z0_center=args.atm_z0_center,
        z0_sigma=args.atm_z0_sigma,
        apply_flags=False,
        common_flag_mask=args.common_flag_mask,
        cachedir=args.atm_cache,
        flush=args.flush,
        wind_dist=args.atm_wind_dist,
    )
    atm.exec(data)

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    tmr.stop()
    if comm.world_rank == 0:
        tmr.report("Atmosphere simulation")
    return


@function_timer
def scale_atmosphere_by_frequency(args, comm, data, freq, mc, cache_name=None, verbose=True):
    """Scale atmospheric fluctuations by frequency.

    Assume that cached signal under totalname_freq is pure atmosphere
    and scale the absorption coefficient according to the frequency.

    If the focalplane is included in the observation and defines
    bandpasses for the detectors, the scaling is computed for each
    detector separately.

    """
    if not args.simulate_atmosphere:
        return

    log = Logger.get()
    if comm.world_rank == 0 and verbose:
        log.info("Scaling atmosphere by frequency")
    timer = Timer()
    timer.start()
    for obs in data.obs:
        tod = obs["tod"]
        todcomm = tod.mpicomm
        site_id = obs["site_id"]
        weather = obs["weather"]
        if "focalplane" in obs:
            focalplane = obs["focalplane"]
        else:
            focalplane = None
        start_time = obs["start_time"]
        weather.set(site_id, mc, start_time)
        altitude = obs["altitude"]
        air_temperature = weather.air_temperature
        surface_pressure = weather.surface_pressure
        pwv = weather.pwv
        # Use the entire processing group to sample the absorption
        # coefficient as a function of frequency
        freqmin = 0
        freqmax = 2 * freq
        nfreq = 1001
        freqstep = (freqmax - freqmin) / (nfreq - 1)
        nfreq_task = int(nfreq // todcomm.size) + 1
        my_ifreq_min = nfreq_task * todcomm.rank
        my_ifreq_max = min(nfreq, nfreq_task * (todcomm.rank + 1))
        my_nfreq = my_ifreq_max - my_ifreq_min
        if my_nfreq > 0:
            if atm_available_utils:
                my_freqs = freqmin + np.arange(my_ifreq_min, my_ifreq_max) * freqstep
                my_absorption = atm_absorption_coefficient_vec(
                    altitude,
                    air_temperature,
                    surface_pressure,
                    pwv,
                    my_freqs[0],
                    my_freqs[-1],
                    my_nfreq,
                )
            else:
                raise RuntimeError(
                    "Atmosphere utilities from libaatm are not available"
                )
        else:
            my_freqs = np.array([])
            my_absorption = np.array([])
        freqs = np.hstack(todcomm.allgather(my_freqs))
        absorption = np.hstack(todcomm.allgather(my_absorption))
        # loading = atm_atmospheric_loading(altitude, pwv, freq)
        for det in tod.local_dets:
            try:
                # Use detector bandpass from the focalplane
                center = focalplane[det]["bandcenter_ghz"]
                width = focalplane[det]["bandwidth_ghz"]
            except Exception:
                # Use default values for the entire focalplane
                center = freq
                width = 0.2 * freq
            nstep = 101
            # Interpolate the absorption coefficient to do a top hat
            # integral across the bandpass
            det_freqs = np.linspace(center - width / 2, center + width / 2, nstep)
            absorption_det = np.mean(np.interp(det_freqs, freqs, absorption))
            cachename = "{}_{}".format(cache_name, det)
            ref = tod.cache.reference(cachename)
            ref *= absorption_det
            del ref

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Atmosphere scaling")
    return


@function_timer
def update_atmospheric_noise_weights(args, comm, data, freq, mc, verbose=False):
    """Update atmospheric noise weights.

    Estimate the atmospheric noise level from weather parameters and
    encode it as a noise_scale in the observation.  Madam will apply
    the noise_scale to the detector weights.  This approach assumes
    that the atmospheric noise dominates over detector noise.  To be
    more precise, we would have to add the squared noise weights but
    we do not have their relative calibration.

    """
    if not args.simulate_atmosphere:
        for obs in data.obs:
            obs["noise_scale"] = 1.0
        return
    log = Logger.get()
    if comm.world_rank == 0 and verbose:
        log.info("Updating atmospheric noise weights")
    timer = Timer()
    timer.start()
    if atm_available_utils:
        for obs in data.obs:
            site_id = obs["site_id"]
            weather = obs["weather"]
            start_time = obs["start_time"]
            weather.set(site_id, mc, start_time)
            altitude = obs["altitude"]
            absorption = atm_absorption_coefficient(
                altitude,
                weather.air_temperature,
                weather.surface_pressure,
                weather.pwv,
                freq,
            )
            obs["noise_scale"] = absorption * weather.air_temperature
    else:
        raise RuntimeError("Atmosphere utilities from libaatm are not available")

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    timer.stop()
    if comm.world_rank == 0 and verbose:
        timer.report("Atmosphere weighting")
    return


@function_timer
def get_focalplane_radius(args, focalplane, rmin=1.0):
    """ Find the furthest angular distance from the boresight

    The number is needed for simulating the atmosphere.
    """
    if args.focalplane_radius:
        return args.focalplane_radius

    cosangs = []
    for det in focalplane:
        quat = focalplane[det]["quat"]
        vec = qa.rotate(quat, ZAXIS)
        cosangs.append(np.dot(ZAXIS, vec))
    mincos = np.amin(cosangs)
    maxdist = max(np.degrees(np.arccos(mincos)), rmin)
    return maxdist * 1.001


