# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import QTable
from scipy.constants import degree

from .. import qarray as qa
from ..coordinates import azel_to_radec
from ..dist import distribute_discrete, distribute_uniform
from ..healpix import ang2vec
from ..instrument import Focalplane, Session, Telescope
from ..intervals import IntervalList, regular_intervals
from ..noise_sim import AnalyticNoise
from ..observation import Observation
from ..observation import default_values as defaults
from ..schedule import GroundSchedule
from ..timing import GlobalTimers, Timer, function_timer
from ..traits import (
    Bool,
    Float,
    Instance,
    Int,
    List,
    Quantity,
    Unicode,
    Unit,
    trait_docs,
)
from ..utils import (
    Environment,
    Logger,
    astropy_control,
    memreport,
    name_UID,
    rate_from_times,
)
from ..weather import SimWeather
from .flag_intervals import FlagIntervals
from .operator import Operator
from .sim_ground_utils import (
    add_solar_intervals,
    oscillate_el,
    simulate_ces_scan,
    simulate_elnod,
    step_el,
)
from .sim_hwp import simulate_hwp_response


@trait_docs
class SimGround(Operator):
    """Simulate a generic ground-based telescope scanning.

    This simulates ground-based pointing in constant elevation scans for a telescope
    located at a particular site and using an pre-created schedule.

    The created observations define several interval lists to describe regions where
    the telescope is scanning left, right or in a turnaround or El-nod.  A shared
    flag array is also created with bits sets for these same properties.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    telescope = Instance(
        klass=Telescope, allow_none=True, help="This must be an instance of a Telescope"
    )

    session_split_key = Unicode(
        None, allow_none=True, help="Focalplane key for splitting into observations"
    )

    weather = Unicode(
        None,
        allow_none=True,
        help="Name of built-in weather site (e.g. 'atacama', 'south_pole') or path to HDF5 file",
    )

    realization = Int(0, help="The realization index")

    schedule = Instance(
        klass=GroundSchedule, allow_none=True, help="Instance of a GroundSchedule"
    )

    timezone = Int(
        0, help="The (integer) timezone offset in hours from UTC to apply to schedule"
    )

    randomize_phase = Bool(
        False,
        help="If True, the Constant Elevation Scan will begin at a randomized phase.",
    )

    scan_rate_az = Quantity(
        1.0 * u.degree / u.second,
        help="The sky or mount azimuth scanning rate.  See `fix_rate_on_sky`",
    )

    fix_rate_on_sky = Bool(
        True,
        help="If True, `scan_rate_az` is given in sky coordinates and azimuthal rate "
        "on mount will be adjusted to meet it.  If False, `scan_rate_az` is used as "
        "the mount azimuthal rate.",
    )

    scan_rate_el = Quantity(
        1.0 * u.degree / u.second,
        allow_none=True,
        help="The sky elevation scanning rate",
    )

    scan_accel_az = Quantity(
        1.0 * u.degree / u.second**2,
        help="Mount scanning rate acceleration for turnarounds",
    )

    scan_accel_el = Quantity(
        1.0 * u.degree / u.second**2,
        allow_none=True,
        help="Mount elevation rate acceleration.",
    )

    scan_cosecant_modulation = Bool(
        False, help="Modulate the scan rate according to 1/sin(az) for uniform depth"
    )

    sun_angle_min = Quantity(
        90.0 * u.degree, help="Minimum angular distance for the scan and the Sun"
    )

    el_mod_step = Quantity(
        0.0 * u.degree, help="Amount to step elevation after each left-right scan pair"
    )

    el_mod_rate = Quantity(
        0.0 * u.Hz, help="Modulate elevation continuously at this rate"
    )

    el_mod_amplitude = Quantity(1.0 * u.degree, help="Range of elevation modulation")

    el_mod_sine = Bool(
        False, help="Modulate elevation with a sine wave instead of a triangle wave"
    )

    distribute_time = Bool(
        False,
        help="Distribute observation data along the time axis rather than detector axis",
    )

    detset_key = Unicode(
        None,
        allow_none=True,
        help="If specified, use this column of the focalplane detector_data to group detectors",
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key to initialize",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to initialize",
    )

    hwp_angle = Unicode(
        None, allow_none=True, help="Observation shared key for HWP angle"
    )

    azimuth = Unicode(defaults.azimuth, help="Observation shared key for Azimuth")

    elevation = Unicode(defaults.elevation, help="Observation shared key for Elevation")

    boresight_azel = Unicode(
        defaults.boresight_azel, help="Observation shared key for boresight AZ/EL"
    )

    boresight_radec = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight RA/DEC"
    )

    position = Unicode(defaults.position, help="Observation shared key for position")

    velocity = Unicode(defaults.velocity, help="Observation shared key for velocity")

    hwp_rpm = Float(None, allow_none=True, help="The rate (in RPM) of the HWP rotation")

    hwp_step = Quantity(
        None, allow_none=True, help="For stepped HWP, the angle of each step"
    )

    hwp_step_time = Quantity(
        None, allow_none=True, help="For stepped HWP, the time between steps"
    )

    elnod_start = Bool(False, help="Perform an el-nod before the scan")

    elnod_end = Bool(False, help="Perform an el-nod after the scan")

    elnods = List([], help="List of relative el_nods")

    elnod_every_scan = Bool(False, help="Perform el nods every scan")

    scanning_interval = Unicode(
        defaults.scanning_interval, help="Interval name for scanning"
    )

    turnaround_interval = Unicode(
        defaults.turnaround_interval, help="Interval name for turnarounds"
    )

    throw_leftright_interval = Unicode(
        defaults.throw_leftright_interval,
        help="Interval name for left to right scans + turnarounds",
    )

    throw_rightleft_interval = Unicode(
        defaults.throw_rightleft_interval,
        help="Interval name for right to left scans + turnarounds",
    )

    throw_interval = Unicode(
        defaults.throw_interval, help="Interval name for scan + turnaround intervals"
    )

    scan_leftright_interval = Unicode(
        defaults.scan_leftright_interval, help="Interval name for left to right scans"
    )

    turn_leftright_interval = Unicode(
        defaults.turn_leftright_interval,
        help="Interval name for turnarounds after left to right scans",
    )

    scan_rightleft_interval = Unicode(
        defaults.scan_rightleft_interval, help="Interval name for right to left scans"
    )

    turn_rightleft_interval = Unicode(
        defaults.turn_rightleft_interval,
        help="Interval name for turnarounds after right to left scans",
    )

    elnod_interval = Unicode(defaults.elnod_interval, help="Interval name for elnods")

    sun_up_interval = Unicode(
        defaults.sun_up_interval, help="Interval name for times when the sun is up"
    )

    sun_close_interval = Unicode(
        defaults.sun_close_interval,
        help="Interval name for times when the sun is close",
    )

    sun_close_distance = Quantity(45.0 * u.degree, help="'Sun close' flagging distance")

    max_pwv = Quantity(
        None, allow_none=True, help="Maximum PWV for the simulated weather."
    )

    median_weather = Bool(
        False,
        help="Use median weather parameters instead of sampling from the distributions",
    )

    invalid_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask to raise invalid flags with"
    )

    turnaround_mask = Int(
        defaults.turnaround, help="Bit mask to raise turnaround flags with"
    )

    leftright_mask = Int(
        defaults.scan_leftright, help="Bit mask to raise left-to-right flags with"
    )

    rightleft_mask = Int(
        defaults.scan_rightleft, help="Bit mask to raise right-to-left flags with"
    )

    sun_up_mask = Int(defaults.sun_up, help="Bit mask to raise Sun up flags with")

    sun_close_mask = Int(
        defaults.sun_close, help="Bit mask to raise Sun close flags with"
    )

    elnod_mask = Int(defaults.elnod, help="Bit mask to raise elevation nod flags with")

    @traitlets.validate("telescope")
    def _check_telescope(self, proposal):
        tele = proposal["value"]
        if tele is not None:
            try:
                dets = tele.focalplane.detectors
            except Exception:
                raise traitlets.TraitError(
                    "telescope must be a Telescope instance with a focalplane"
                )
        return tele

    @traitlets.validate("schedule")
    def _check_schedule(self, proposal):
        sch = proposal["value"]
        if sch is not None:
            if not isinstance(sch, GroundSchedule):
                raise traitlets.TraitError(
                    "schedule must be an instance of a GroundSchedule"
                )
        return sch

    # Cross-check HWP parameters

    @traitlets.validate("hwp_angle")
    def _check_hwp_angle(self, proposal):
        hwp_angle = proposal["value"]
        if hwp_angle is None:
            if self.hwp_rpm is not None or self.hwp_step is not None:
                raise traitlets.TraitError(
                    "Cannot simulate HWP without a shared data key"
                )
        else:
            if self.hwp_rpm is None and self.hwp_step is None:
                raise traitlets.TraitError("Cannot simulate HWP without parameters")
        return hwp_angle

    @traitlets.validate("hwp_rpm")
    def _check_hwp_rpm(self, proposal):
        hwp_rpm = proposal["value"]
        if hwp_rpm is not None:
            if self.hwp_angle is None:
                raise traitlets.TraitError(
                    "Cannot simulate rotating HWP without a shared data key"
                )
            if self.hwp_step is not None:
                raise traitlets.TraitError("HWP cannot rotate *and* step.")
        else:
            if self.hwp_angle is not None and self.hwp_step is None:
                raise traitlets.TraitError("Cannot simulate HWP without parameters")
        return hwp_rpm

    @traitlets.validate("hwp_step")
    def _check_hwp_step(self, proposal):
        hwp_step = proposal["value"]
        if hwp_step is not None:
            if self.hwp_angle is None:
                raise traitlets.TraitError(
                    "Cannot simulate stepped HWP without a shared data key"
                )
            if self.hwp_rpm is not None:
                raise traitlets.TraitError("HWP cannot rotate *and* step.")
        else:
            if self.hwp_angle is not None and self.hwp_rpm is None:
                raise traitlets.TraitError("Cannot simulate HWP without parameters")
        return hwp_step

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        if self.schedule is None:
            raise RuntimeError(
                "The schedule attribute must be set before calling exec()"
            )

        # Check valid combinations of options

        if (self.elnod_start or self.elnod_end) and len(self.elnods) == 0:
            raise RuntimeError(
                "If simulating elnods, you must specify the list of offsets"
            )

        if len(self.schedule.scans) == 0:
            raise RuntimeError("Schedule has no scans!")

        # Data distribution in the detector and sample directions
        comm = data.comm
        det_ranks = comm.group_size
        samp_ranks = 1
        if self.distribute_time:
            det_ranks = 1
            samp_ranks = comm.group_size

        # Get per-observation telescopes
        obs_tele = self._obs_telescopes(data, det_ranks, detectors)

        # The global start is the beginning of the first scan
        mission_start = self.schedule.scans[0].start

        # Although there is no requirement that the sampling is contiguous from one
        # session to the next, for simulations there is no need to restart the
        # sampling clock for each one.  In order to help with load balancing, we
        # distribute all observations across all sessions among process groups.
        # We distribute these in sequence to minimize the number of boresight
        # scanning calculations need to be done by each group.

        obs_info = list()

        rate = self.telescope.focalplane.sample_rate.to_value(u.Hz)
        incr = 1.0 / rate
        off = 0

        for scan in self.schedule.scans:
            ffirst = rate * (scan.start - mission_start).total_seconds()
            first = int(ffirst)
            if ffirst - first > 1.0e-3 * incr:
                first += 1
            start = first * incr + mission_start.timestamp()
            ns = 1 + int(rate * (scan.stop.timestamp() - start))
            stop = (ns - 1) * incr + mission_start.timestamp()

            # The session name is the same as the historical observation name,
            # which allows re-use of previously cached atmosphere sims.
            sname = f"{scan.name}-{scan.scan_indx}-{scan.subscan_indx}"

            for obkey, (obtele, detsets) in obs_tele.items():
                if obkey == "ALL":
                    obs_name = sname
                else:
                    obs_name = f"{sname}_{obkey}"
                obs_info.append(
                    {
                        "name": obs_name,
                        "sname": sname,
                        "obkey": obkey,
                        "scan": scan,
                        "start": start,
                        "stop": stop,
                        "samples": ns,
                        "offset": off,
                    }
                )
            off += ns

        # FIXME:  Re-enable this when using astropy for coordinate transforms.
        # # Ensure that astropy IERS is downloaded
        # astropy_control(max_future=self.schedule.scans[-1].stop)

        # Distribute the sessions uniformly among groups.  We take each scan and
        # weight it by the duration in samples.

        obs_samples = [x["samples"] for x in obs_info]
        groupdist = distribute_discrete(obs_samples, comm.ngroups)

        # Every process group creates its observations

        group_first_obs = groupdist[comm.group][0]
        group_num_obs = groupdist[comm.group][1]

        last_session = None
        for obindx in range(group_first_obs, group_first_obs + group_num_obs):
            scan = obs_info[obindx]["scan"]
            sname = obs_info[obindx]["sname"]
            obs_name = obs_info[obindx]["name"]

            sys_mem_str = memreport(
                msg="(whole node)", comm=data.comm.comm_group, silent=True
            )
            msg = f"Group {data.comm.group} begin observation {obs_name} "
            msg += f"with {sys_mem_str}"
            log.debug_rank(msg, comm=data.comm.comm_group)

            # Simulate the boresight pattern.  If this observation is in the same
            # session as the previous observation, just re-use the pointing.

            if sname != last_session:
                (
                    times,
                    az,
                    el,
                    sample_sets,
                    scan_min_az,
                    scan_max_az,
                    scan_min_el,
                    scan_max_el,
                    ival_elnod,
                    ival_scan_leftright,
                    ival_scan_rightleft,
                    ival_throw_leftright,
                    ival_throw_rightleft,
                    ival_turn_leftright,
                    ival_turn_rightleft,
                ) = self._simulate_scanning(
                    scan, obs_info[obindx]["samples"], rate, comm, samp_ranks
                )

                # Create weather realization common to all observations in the session
                weather = None
                site = self.telescope.site
                if self.weather is not None:
                    # Every session has a unique site with unique weather
                    # realization.
                    site = copy.deepcopy(site)
                    mid_time = scan.start + (scan.stop - scan.start) / 2
                    try:
                        weather = SimWeather(
                            time=mid_time,
                            name=self.weather,
                            site_uid=site.uid,
                            realization=self.realization,
                            max_pwv=self.max_pwv,
                            median_weather=self.median_weather,
                        )
                    except RuntimeError:
                        # must be a file
                        weather = SimWeather(
                            time=mid_time,
                            file=self.weather,
                            site_uid=site.uid,
                            realization=self.realization,
                            max_pwv=self.max_pwv,
                            median_weather=self.median_weather,
                        )
                    site.weather = weather

                session = Session(
                    sname,
                    start=datetime.fromtimestamp(times[0]).astimezone(timezone.utc),
                    end=datetime.fromtimestamp(times[-1]).astimezone(timezone.utc),
                )

            # Create the observation

            obtele, detsets = obs_tele[obs_info[obindx]["obkey"]]

            # Instantiate new telescope with site that may be unique to this session
            telescope = Telescope(
                obtele.name,
                uid=obtele.uid,
                focalplane=obtele.focalplane,
                site=site,
            )

            ob = Observation(
                comm,
                telescope,
                len(times),
                name=obs_name,
                uid=name_UID(obs_name),
                session=session,
                detector_sets=detsets,
                process_rows=det_ranks,
                sample_sets=sample_sets,
            )

            # Scan limits
            ob["scan_el"] = scan.el  # Nominal elevation
            ob["scan_min_az"] = scan_min_az * u.radian
            ob["scan_max_az"] = scan_max_az * u.radian
            ob["scan_min_el"] = scan_min_el * u.radian
            ob["scan_max_el"] = scan_max_el * u.radian

            # Create and set shared objects for timestamps, position, velocity, and
            # boresight.

            ob.shared.create_column(
                self.times,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.position,
                shape=(ob.n_local_samples, 3),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.velocity,
                shape=(ob.n_local_samples, 3),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.azimuth,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.elevation,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.boresight_azel,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
            )
            ob.shared.create_column(
                self.boresight_radec,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
            )

            # Optionally initialize detector data.  Note that the
            # detectors in each observation have already been pruned
            # during the splitting.

            if self.det_data is not None:
                exists_data = ob.detdata.ensure(
                    self.det_data,
                    dtype=np.float64,
                    create_units=self.det_data_units,
                )

            if self.det_flags is not None:
                exists_flags = ob.detdata.ensure(
                    self.det_flags,
                    dtype=np.uint8,
                )

            # Only the first rank of the process grid columns sets / computes these.

            if sname != last_session:
                stamps = None
                position = None
                velocity = None
                az_data = None
                el_data = None
                bore_azel = None
                bore_radec = None

                if ob.comm_col_rank == 0:
                    stamps = times[
                        ob.local_index_offset : ob.local_index_offset
                        + ob.n_local_samples
                    ]
                    az_data = az[
                        ob.local_index_offset : ob.local_index_offset
                        + ob.n_local_samples
                    ]
                    el_data = el[
                        ob.local_index_offset : ob.local_index_offset
                        + ob.n_local_samples
                    ]
                    # Get the motion of the site for these times.
                    position, velocity = site.position_velocity(stamps)
                    # Convert Az / El to quaternions.  Remember that the azimuth is
                    # measured clockwise and the longitude counter-clockwise.  We define
                    # the focalplane coordinate X-axis to be pointed in the direction
                    # of decreasing elevation.
                    bore_azel = qa.from_lonlat_angles(
                        -(az_data), el_data, np.zeros_like(el_data)
                    )

                    if scan.boresight_angle.to_value(u.radian) != 0:
                        zaxis = np.array([0, 0, 1.0])
                        rot = qa.rotation(
                            zaxis, scan.boresight_angle.to_value(u.radian)
                        )
                        bore_azel = qa.mult(bore_azel, rot)
                    # Convert to RA / DEC.  Use pyephem for now.
                    bore_radec = azel_to_radec(site, stamps, bore_azel, use_ephem=True)

            ob.shared[self.times].set(stamps, offset=(0,), fromrank=0)
            ob.shared[self.azimuth].set(az_data, offset=(0,), fromrank=0)
            ob.shared[self.elevation].set(el_data, offset=(0,), fromrank=0)
            ob.shared[self.position].set(position, offset=(0, 0), fromrank=0)
            ob.shared[self.velocity].set(velocity, offset=(0, 0), fromrank=0)
            ob.shared[self.boresight_azel].set(bore_azel, offset=(0, 0), fromrank=0)
            ob.shared[self.boresight_radec].set(bore_radec, offset=(0, 0), fromrank=0)

            # Simulate HWP angle

            simulate_hwp_response(
                ob,
                ob_time_key=self.times,
                ob_angle_key=self.hwp_angle,
                ob_mueller_key=None,
                hwp_start=obs_info[obindx]["start"] * u.second,
                hwp_rpm=self.hwp_rpm,
                hwp_step=self.hwp_step,
                hwp_step_time=self.hwp_step_time,
            )

            # Create interval lists for our motion.  Since we simulated the scan on
            # every process, we don't need to communicate the global timespans of the
            # intervals (using create or create_col).  We can just create them directly.

            ob.intervals[self.throw_leftright_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_throw_leftright
            )
            ob.intervals[self.throw_rightleft_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_throw_rightleft
            )
            ob.intervals[self.throw_interval] = (
                ob.intervals[self.throw_leftright_interval]
                | ob.intervals[self.throw_rightleft_interval]
            )
            ob.intervals[self.scan_leftright_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_scan_leftright
            )
            ob.intervals[self.turn_leftright_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_turn_leftright
            )
            ob.intervals[self.scan_rightleft_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_scan_rightleft
            )
            ob.intervals[self.turn_rightleft_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_turn_rightleft
            )
            ob.intervals[self.elnod_interval] = IntervalList(
                ob.shared[self.times], timespans=ival_elnod
            )
            ob.intervals[self.scanning_interval] = (
                ob.intervals[self.scan_leftright_interval]
                | ob.intervals[self.scan_rightleft_interval]
            )
            ob.intervals[self.turnaround_interval] = (
                ob.intervals[self.turn_leftright_interval]
                | ob.intervals[self.turn_rightleft_interval]
            )

            # Get the Sun's position in horizontal coordinates and define
            # "Sun up" and "Sun close" intervals according to it

            add_solar_intervals(
                ob.intervals,
                site,
                ob.shared[self.times],
                ob.shared[self.azimuth].data,
                ob.shared[self.elevation].data,
                self.sun_up_interval,
                self.sun_close_interval,
                self.sun_close_distance,
            )

            msg = f"Group {data.comm.group} finished observation {obs_name}:\n"
            msg += f"{ob}"
            log.verbose_rank(msg, comm=data.comm.comm_group)

            obmem = ob.memory_use()
            obmem_gb = obmem / 1024**3
            msg = f"Observation {ob.name} using {obmem_gb:0.2f} GB of total memory"
            log.debug_rank(msg, comm=ob.comm.comm_group)

            data.obs.append(ob)
            last_session = sname

        # For convenience, we additionally create a shared flag field with bits set
        # according to the different intervals.  This basically just saves workflows
        # from calling the FlagIntervals operator themselves.  Here we set the bits
        # according to what was done in toast2, so the scanning interval has no bits
        # set.

        flag_intervals = FlagIntervals(
            shared_flags=self.shared_flags,
            shared_flag_bytes=1,
            view_mask=[
                (self.turnaround_interval, self.turnaround_mask),
                (self.throw_leftright_interval, self.leftright_mask),
                (self.throw_rightleft_interval, self.rightleft_mask),
                (self.sun_up_interval, self.sun_up_mask),
                (self.sun_close_interval, self.sun_close_mask),
                (self.elnod_interval, self.elnod_mask),
            ],
        )
        flag_intervals.apply(data, detectors=None)

    def _simulate_scanning(self, scan, n_samples, rate, comm, samp_ranks):
        """Simulate the boresight Az/El pointing for one session."""
        log = Logger.get()

        # Currently, El nods happen before or after the formal scan start / end.
        # This means that we don't know ahead of time the total number of samples
        # in the observation.  That in turn means we cannot create the observation
        # until after we simulate the motion, and therefore we do not yet have the
        # the process grid established.  Normally only rank zero of each grid
        # column would compute and store this data in shared memory.  However, since
        # we do not have that grid yet, every process simulates the scan.  This
        # should be relatively cheap.

        incr = 1.0 / rate

        # Track the az / el range of all motion during this scan, including
        # el nods and any el modulation / steps.  These will be stored as
        # observation metadata after the simulation.
        scan_min_el = scan.el.to_value(u.radian)
        scan_max_el = scan_min_el
        scan_min_az = scan.az_min.to_value(u.radian)
        scan_max_az = scan.az_max.to_value(u.radian)

        # Time range of the science scans
        start_time = scan.start
        stop_time = start_time + timedelta(seconds=(float(n_samples - 1) / rate))

        # The total simulated scan data (including el nods)
        times = list()
        az = list()
        el = list()

        # The time ranges we will build up to construct intervals later
        ival_elnod = list()
        ival_scan_leftright = None
        ival_turn_leftright = None
        ival_scan_rightleft = None
        ival_turn_rightleft = None

        # Compute relative El Nod steps
        elnod_el = None
        elnod_az = None
        if len(self.elnods) > 0:
            elnod_el = np.array([(scan.el + x).to_value(u.radian) for x in self.elnods])
            elnod_az = np.zeros_like(elnod_el) + scan.az_min.to_value(u.radian)

        # Sample sets.  Although Observations support data distribution in any
        # shape process grid, this operator only supports 2 cases:  distributing
        # by detector and distributing by time.  We want to ensure that

        sample_sets = list()

        # Do starting El nod.  We do this before the start of the scheduled scan.
        if self.elnod_start:
            (
                elnod_times,
                elnod_az_data,
                elnod_el_data,
                scan_min_az,
                scan_max_az,
                scan_min_el,
                scan_max_el,
            ) = simulate_elnod(
                scan.start.timestamp(),
                rate,
                scan.az_min.to_value(u.radian),
                scan.el.to_value(u.radian),
                self.scan_rate_az.to_value(u.radian / u.second),
                self.scan_accel_az.to_value(u.radian / u.second**2),
                self.scan_rate_el.to_value(u.radian / u.second),
                self.scan_accel_el.to_value(u.radian / u.second**2),
                elnod_el,
                elnod_az,
                scan_min_az,
                scan_max_az,
                scan_min_el,
                scan_max_el,
            )
            if len(elnod_times) > 0:
                # Shift these elnod times so that they end one sample before the
                # start of the scan.
                sample_sets.append([len(elnod_times)])
                t_elnod = elnod_times[-1] - elnod_times[0]
                elnod_times -= t_elnod + incr
                times.append(elnod_times)
                az.append(elnod_az_data)
                el.append(elnod_el_data)
                ival_elnod.append((elnod_times[0], elnod_times[-1]))

        # Now do the main scan
        (
            scan_times,
            scan_az_data,
            scan_el_data,
            scan_min_az,
            scan_max_az,
            ival_scan_leftright,
            ival_turn_leftright,
            ival_scan_rightleft,
            ival_turn_rightleft,
            ival_throw_leftright,
            ival_throw_rightleft,
        ) = simulate_ces_scan(
            start_time.timestamp(),
            stop_time.timestamp(),
            rate,
            scan.el.to_value(u.radian),
            scan.az_min.to_value(u.radian),
            scan.az_max.to_value(u.radian),
            scan.az_min.to_value(u.radian),
            self.scan_rate_az.to_value(u.radian / u.second),
            self.fix_rate_on_sky,
            self.scan_accel_az.to_value(u.radian / u.second**2),
            scan_min_az,
            scan_max_az,
            cosecant_modulation=self.scan_cosecant_modulation,
            randomize_phase=self.randomize_phase,
        )

        # Do any adjustments to the El motion
        if self.el_mod_rate.to_value(u.Hz) > 0:
            scan_min_el, scan_max_el = oscillate_el(
                scan_times,
                scan_el_data,
                self.scan_rate_el.to_value(u.radian / u.second),
                self.scan_accel_el.to_value(u.radian / u.second**2),
                scan_min_el,
                scan_max_el,
                self.el_mod_amplitude.to_value(u.radian),
                self.el_mod_rate.to_value(u.Hz),
                el_mod_sine=self.el_mod_sine,
            )
        if self.el_mod_step.to_value(u.radian) > 0:
            scan_min_el, scan_max_el = step_el(
                scan_times,
                scan_az_data,
                scan_el_data,
                self.scan_rate_el.to_value(u.radian / u.second),
                self.scan_accel_el.to_value(u.radian / u.second**2),
                scan_min_el,
                scan_max_el,
                self.el_mod_step.to_value(u.radian),
            )

        # When distributing data, ensure that each process has a whole number of
        # complete scans.
        scan_indices = np.searchsorted(
            scan_times, [x[0] for x in ival_scan_leftright], side="left"
        )
        sample_sets.extend([[x] for x in scan_indices[1:] - scan_indices[:-1]])
        remainder = len(scan_times) - scan_indices[-1]
        if remainder > 0:
            sample_sets.append([remainder])

        times.append(scan_times)
        az.append(scan_az_data)
        el.append(scan_el_data)

        # FIXME:  The CES scan simulation above ends abruptly.  We should implement
        # a deceleration to zero in Az here before doing the final el nod.

        # Do ending El nod.  Start this one sample after the science scan.
        if self.elnod_end:
            (
                elnod_times,
                elnod_az_data,
                elnod_el_data,
                scan_min_az,
                scan_max_az,
                scan_min_el,
                scan_max_el,
            ) = simulate_elnod(
                scan_times[-1] + incr,
                rate,
                scan_az_data[-1],
                scan_el_data[-1],
                self.scan_rate_az.to_value(u.radian / u.second),
                self.scan_accel_az.to_value(u.radian / u.second**2),
                self.scan_rate_el.to_value(u.radian / u.second),
                self.scan_accel_el.to_value(u.radian / u.second**2),
                elnod_el,
                elnod_az,
                scan_min_az,
                scan_max_az,
                scan_min_el,
                scan_max_el,
            )
            if len(elnod_times) > 0:
                sample_sets.append([len(elnod_times)])
                times.append(elnod_times)
                az.append(elnod_az_data)
                el.append(elnod_el_data)
                ival_elnod.append((elnod_times[0], elnod_times[-1]))

        times = np.hstack(times)
        az = np.hstack(az)
        el = np.hstack(el)

        # If we are distributing by time, ensure we have enough sample sets for the
        # number of processes.
        if self.distribute_time:
            if samp_ranks > len(sample_sets):
                if comm.group_rank == 0:
                    msg = f"Group {comm.group} with {comm.group_size} processes cannot distribute {len(sample_sets)} sample sets."
                    log.error(msg)
                    raise RuntimeError(msg)

        return (
            times,
            az,
            el,
            sample_sets,
            scan_min_az,
            scan_max_az,
            scan_min_el,
            scan_max_el,
            ival_elnod,
            ival_scan_leftright,
            ival_scan_rightleft,
            ival_throw_leftright,
            ival_throw_rightleft,
            ival_turn_leftright,
            ival_turn_rightleft,
        )

    def _obs_telescopes(self, data, det_ranks, detectors):
        """Split our session telescope by focalplane key."""

        log = Logger.get()

        session_tele_name = self.telescope.name
        session_tele_uid = self.telescope.uid
        site = self.telescope.site
        session_fp = self.telescope.focalplane

        if self.session_split_key is None and detectors is None:
            # All one observation and all detectors.
            if self.detset_key is None:
                detsets = None
            else:
                detsets = session_fp.detector_groups(self.detset_key)
            return {"ALL": (self.telescope, detsets)}

        # First cut the table down to only the detectors we are considering
        fp_input = session_fp.detector_data
        if detectors is None:
            keep_rows = [True for x in range(len(fp_input))]
        else:
            dcheck = set(detectors)
            keep_rows = [True if x in dcheck else False for x in fp_input["name"]]

        session_keys = np.unique(fp_input[self.session_split_key])

        obs_tele = dict()
        for key in session_keys:
            fp_rows = np.logical_and(
                fp_input[self.session_split_key] == key,
                keep_rows,
            )
            fp_detdata = QTable(fp_input[fp_rows])

            fp = Focalplane(
                detector_data=fp_detdata,
                sample_rate=session_fp.sample_rate,
                field_of_view=session_fp.field_of_view,
                thinfp=session_fp.thinfp,
            )

            # List of detectors in this pipeline
            pipedets = None
            if detectors is None:
                pipedets = fp.detectors
            else:
                pipedets = list()
                check_det = set(detectors)
                for det in fp.detectors:
                    if det in check_det:
                        pipedets.append(det)

            # Group by detector sets
            if self.detset_key is None:
                detsets = None
            else:
                dsets = fp.detector_groups(self.detset_key)
                if detectors is None:
                    detsets = dsets
                else:
                    # Prune to include only the detectors we are using.
                    detsets = dict()
                    pipe_check = set(pipedets)
                    for k, v in dsets.items():
                        detsets[k] = list()
                        for d in v:
                            if d in pipe_check:
                                detsets[k].append(d)

            # Verify that we have enough detector data for all of our processes.  If we
            # are distributing by time, we check the sample sets on a per-observation
            # basis later.  If we are distributing by detector, we must have at least
            # one detector set for each process.

            if not self.distribute_time:
                # distributing by detector...
                n_detset = None
                if detsets is None:
                    # Every detector is independently distributed
                    n_detset = len(pipedets)
                else:
                    n_detset = len(detsets)
                if det_ranks > n_detset:
                    if data.comm.group_rank == 0:
                        msg = f"Group {data.comm.group} with {data.comm.group_size}"
                        msg += f" processes cannot distribute {n_detset} detector sets."
                        log.error(msg)
                        raise RuntimeError(msg)

            safe_key = str(key).replace(" ", "-")
            tele_name = f"{session_tele_name}_{safe_key}"
            obs_tele[safe_key] = (
                Telescope(
                    tele_name,
                    uid=session_tele_uid,
                    focalplane=fp,
                    site=site,
                ),
                detsets,
            )
        return obs_tele

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {
            "shared": [
                self.times,
                self.shared_flags,
                self.azimuth,
                self.elevation,
                self.boresight_azel,
                self.boresight_radec,
                self.hwp_angle,
                self.position,
                self.velocity,
            ],
            "detdata": list(),
            "intervals": [
                self.scanning_interval,
                self.turnaround_interval,
                self.scan_leftright_interval,
                self.scan_rightleft_interval,
                self.turn_leftright_interval,
                self.turn_rightleft_interval,
                self.throw_interval,
                self.throw_leftright_interval,
                self.throw_rightleft_interval,
            ],
        }
        if self.det_data is not None:
            prov["detdata"].append(self.det_data)
        if self.det_flags is not None:
            prov["detdata"].append(self.det_flags)
        return prov
