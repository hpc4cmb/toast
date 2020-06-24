# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from scipy.constants import degree

import healpy as hp

from .. import qarray as qa

from ..utils import Environment, name_UID, Logger, rate_from_times

from ..dist import distribute_uniform

from ..timing import function_timer, Timer

from ..tod import Interval, TOD, regular_intervals, AnalyticNoise

from ..operator import Operator

from ..observation import Observation

from ..config import ObjectConfig

from ..instrument import Telescope

from ..healpix import ang2vec

from .sim_hwp import simulate_hwp_angle


class SimGround(Operator):
    """Simulate a generic ground-based telescope.

    This uses an observing schedule to simulate observations of a ground based
    telescope.

    Args:
        config (dict): Configuration parameters.

    """

    def __init__(self, config):
        super().__init__(config)
        self._parse()

    @classmethod
    def defaults(cls):
        """(Class method) Return options supported by the operator and their defaults.

        This returns an ObjectConfig instance, and each entry should have a help
        string.

        Returns:
            (ObjectConfig): The options.

        """
        opts = ObjectConfig()

        opts.add("class", "toast.future_ops.SimGround", "The class name")

        opts.add("API", 0, "(Internal interface version for this operator)")

        opts.add("telescope", None, "This should be an instance of a Telescope")

        opts.add("start_time", 0.0, "The mission start time in seconds")

        opts.add("hwp_rpm", None, "The rate (in RPM) of the HWP rotation")

        opts.add(
            "hwp_step_deg", None, "For stepped HWP, the angle in degrees of each step"
        )

        opts.add(
            "hwp_step_time_m",
            None,
            "For stepped HWP, the time in minutes between steps",
        )

        boresight_angle = (0,)
        firsttime = (0.0,)
        rate = (100.0,)
        site_lon = (0,)
        site_lat = (0,)
        site_alt = (0,)
        el = (None,)
        azmin = (None,)
        azmax = (None,)
        el_nod = (None,)
        start_with_elnod = (True,)
        end_with_elnod = (False,)
        scanrate = (1,)
        scanrate_el = (None,)
        scan_accel = (0.1,)
        scan_accel_el = (None,)
        CES_start = (None,)
        CES_stop = (None,)
        el_min = (0,)
        sun_angle_min = (90,)
        sampsizes = (None,)
        sampbreaks = (None,)
        coord = ("C",)
        report_timing = (True,)
        hwprpm = (None,)
        hwpstep = (None,)
        hwpsteptime = (None,)
        cosecant_modulation = (False,)

        return opts

    def _parse(self):
        if "telescope" not in self.config:
            raise RuntimeError("Satellite simulations require a telescope")
        try:
            dets = self.config["telescope"].focalplane.detectors
        except:
            raise RuntimeError("'telescope' option should be an instance of Telescope")
        if "start_time" not in self.config:
            self.config["start_time"] = 0.0
        if "observation_time_h" not in self.config:
            raise RuntimeError("Time span of each observation must be specified")
        if "gap_time_h" not in self.config:
            self.config["gap_time_h"] = 0.0
        if "n_observation" not in self.config:
            raise RuntimeError("Number of observations must be specified")

    def exec(self, data, detectors=None):
        """Create observations containing simulated satellite pointing.

        Observations will be appended to the Data object.

        Args:
            data (toast.Data):  The distributed data.
            detectors (list):  A list of detector names or indices.  If None, this
                indicates a list of all detectors.

        Returns:
            None

        """
        log = Logger.get()
        focalplane = self.config["telescope"].focalplane
        comm = data.comm

        # List of detectors in this pipeline
        pipedets = list()
        for d in focalplane.detectors:
            if d not in detectors:
                continue
            pipedets.append(d)

        if comm.group_size > len(pipedets):
            if comm.world_rank == 0:
                log.error("process group is too large for the number of detectors")
                comm.comm_world.Abort()

        # Distribute the observations uniformly among groups

        groupdist = distribute_uniform(self.config["n_observation"], comm.ngroups)

        # Compute global time and sample ranges of all observations

        obsrange = regular_intervals(
            self.config["n_observation"],
            self.config["start_time"],
            0,
            focalplane.sample_rate,
            3600 * self.config["observation_time_h"],
            3600 * self.config["gap_time_h"],
        )

        # Every process group creates its observations

        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for ob in range(group_firstobs, group_firstobs + group_numobs):
            obname = "science_{:05d}".format(ob)
            obs = Observation(
                self.config["telescope"],
                name=obname,
                UID=name_UID(obname),
                samples=obsrange[ob].samples,
                detector_ranks=comm.group_size,
                mpicomm=comm.comm_group,
            )

            # Create standard shared objects.

            obs.create_times()
            obs.create_common_flags()

            # Rank zero of each grid column creates the data
            stamps = None
            if obs.grid_comm_col is None or obs.grid_comm_col.rank == 0:
                start_abs = obs.local_samples[0] + obsrange[ob].first
                start_time = (
                    obsrange[ob].start + float(start_abs) / focalplane.sample_rate
                )
                stop_time = (
                    start_time + float(obs.local_samples[1]) / focalplane.sample_rate
                )
                stamps = np.linspace(
                    start_time,
                    stop_time,
                    num=obs.local_samples[1],
                    endpoint=False,
                    dtype=np.float64,
                )
            obs.times().set(stamps, offset=(0,), fromrank=0)

            # Create boresight
            start_abs = obs.local_samples[0] + obsrange[ob].first
            degday = 360.0 / 365.25

            q_prec = None
            if obs.grid_comm_col is None or obs.grid_comm_col.rank == 0:
                q_prec = slew_precession_axis(
                    first_samp=start_abs,
                    n_samp=obs.local_samples[1],
                    sample_rate=focalplane.sample_rate,
                    deg_day=degday,
                )

            satellite_scanning(
                obs,
                sample_offset=start_abs,
                q_prec=q_prec,
                spin_period_m=self.config["spin_period_m"],
                spin_angle_deg=self.config["spin_angle_deg"],
                prec_period_m=self.config["prec_period_m"],
                prec_angle_deg=self.config["prec_angle_deg"],
            )

            # Set HWP angle

            simulate_hwp_angle(
                obs,
                obsrange[ob].start,
                self.config["hwp_rpm"],
                self.config["hwp_step_deg"],
                self.config["hwp_step_time_m"],
            )

            data.obs.append(obs)

        return

    def finalize(self, data):
        """Perform any final operations / communication.

        This calls the finalize() method on all operators in sequence.

        Args:
            data (toast.Data):  The distributed data.

        Returns:
            None

        """
        return

    def requires(self):
        """List of Observation keys directly used by this Operator.
        """
        return list()

    def provides(self):
        """List of Observation keys generated by this Operator.
        """
        prov = [
            "TIMESTAMPS",
            "BORESIGHT_RADEC",
            "BORESIGHT_RESPONSE",
            "COMMON_FLAGS",
            "HWP_ANGLE",
            "POSITION",
            "VELOCITY",
        ]
        return prov

    def accelerators(self):
        """List of accelerators supported by this Operator.
        """
        return list()
