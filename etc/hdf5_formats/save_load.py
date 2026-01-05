#!/usr/bin/env python3

# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Test saving and loading different HDF5 formats"""

import argparse

import numpy as np
from astropy import units as u

import toast
import toast.ops
from toast.utils import Logger

from toast.tests.helpers import create_ground_data


def main(opts=None):
    log = Logger.get()

    parser = argparse.ArgumentParser(description="Save or Load HDF5 Volumes.")

    parser.add_argument(
        "--out_volume",
        required=False,
        default=None,
        help="Output volume",
    )

    parser.add_argument(
        "--in_volume",
        required=False,
        default=None,
        help="Output volume",
    )

    parser.add_argument(
        "--no_meta",
        required=False,
        default=False,
        action="store_true",
        help="If True, disable metadata generation",
    )

    args = parser.parse_args(args=opts)

    if args.in_volume is None and args.out_volume is None:
        log.error("Nothing to do!")
        return

    world, procs, rank = toast.mpi.get_world()

    # Create data, either for writing or for comparison.

    ppp = 2
    freq_list = [(100 + 10 * x) * u.GHz for x in range(3)]
    data = create_ground_data(
        world,
        freqs=freq_list,
        pixel_per_process=ppp,
        split=True,
    )

    # Add extra metadata attribute
    if not args.no_meta:
        from toast.tests.io_hdf5 import ExtraMeta, create_other_meta

        for ob in data.obs:
            ob.extra = ExtraMeta()
        other = create_other_meta()
        ob.update(other)

    # Replace the simulated weather with the base class for testing
    for ob in data.obs:
        old_weather = ob.telescope.site.weather
        new_weather = toast.weather.Weather(
            time=old_weather.time,
            ice_water=old_weather.ice_water,
            liquid_water=old_weather.liquid_water,
            pwv=old_weather.pwv,
            humidity=old_weather.humidity,
            surface_pressure=old_weather.surface_pressure,
            surface_temperature=old_weather.surface_temperature,
            air_temperature=old_weather.air_temperature,
            west_wind=old_weather.west_wind,
            south_wind=old_weather.south_wind,
        )
        ob.telescope.site.weather = new_weather
        del old_weather

    # Simple detector pointing for el weighted noise
    detpointing_azel = toast.ops.PointingDetectorSimple(
        boresight="boresight_azel", quats="quats_azel"
    )

    # Create a noise model from focalplane detector properties
    default_model = toast.ops.DefaultNoiseModel()
    default_model.apply(data)

    # Make an elevation-dependent noise model
    el_model = toast.ops.ElevationNoise(
        noise_model="noise_model",
        out_model="el_weighted",
        detector_pointing=detpointing_azel,
    )
    el_model.apply(data)

    # Simulate noise and accumulate to signal
    sim_noise = toast.ops.SimNoise(noise_model=el_model.out_model)
    sim_noise.apply(data)

    config = toast.config.build_config(
        [
            detpointing_azel,
            default_model,
            el_model,
            sim_noise,
        ]
    )

    det_data_fields = [
        ("signal", None),
        ("flags", None),
    ]

    # det_data_fields = [
    #     ("signal", {"type": "flac", "quanta": 1.0e-14}),
    #     ("flags", None),
    # ]

    if args.in_volume is None:
        # We are saving the simulated data
        toast.ops.SaveHDF5(
            volume=args.out_volume,
            detdata=det_data_fields,
            config=config,
        ).apply(data)
    else:
        # We are loading data and comparing
        in_data = toast.Data(comm=data.comm)
        toast.ops.LoadHDF5(volume=args.in_volume).apply(in_data)
        obs_lookup = {x.name: x for x in data.obs}
        for ob in in_data.obs:
            orig = obs_lookup[ob.name]
            if not toast.ops.save_hdf5.obs_approx_equal(ob, orig):
                msg = f"-------- Proc {data.comm.world_rank} ---------\n{orig}\n"
                msg += f"NOT EQUAL TO {ob}"
                print(msg, flush=True)
            log.info_rank(
                f"Finished comparison of {ob.name}", comm=data.comm.comm_group
            )


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
