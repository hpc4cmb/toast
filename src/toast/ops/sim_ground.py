# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import datetime

import copy

import traitlets

import numpy as np

from scipy.constants import degree

import healpy as hp

from astropy import units as u
from toast.weather import SimWeather

from .. import qarray as qa

from ..utils import Environment, name_UID, Logger, rate_from_times, astropy_control

from ..dist import distribute_uniform, distribute_discrete

from ..timing import function_timer, Timer

from ..intervals import Interval, regular_intervals, IntervalList

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity, List

from ..observation import Observation

from ..instrument import Telescope

from ..schedule import GroundSchedule

from ..coordinates import azel_to_radec

from ..healpix import ang2vec

from .operator import Operator

from .sim_hwp import simulate_hwp_response

from .flag_intervals import FlagIntervals

from .sim_ground_utils import simulate_elnod, simulate_ces_scan, add_solar_intervals, oscillate_el, step_el


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

    weather = Unicode(
        None,
        allow_none=True,
        help="Name of built-in weather site (e.g. 'atacama', 'south_pole') or path to HDF5 file",
    )

    schedule = Instance(
        klass=GroundSchedule, allow_none=True, help="Instance of a GroundSchedule"
    )

    timezone = Int(
        0, help="The (integer) timezone offset in hours from UTC to apply to schedule"
    )

    scan_rate_az = Quantity(
        1.0 * u.degree / u.second, help="The sky azimuth scanning rate"
    )

    scan_rate_el = Quantity(
        None, allow_none=True, help="The sky elevation scanning rate"
    )

    scan_accel_az = Quantity(
        1.0 * u.degree / u.second ** 2,
        help="Mount scanning rate acceleration for turnarounds",
    )

    scan_accel_el = Quantity(
        None, allow_none=True, help="Mount elevation rate acceleration."
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

    times = Unicode("times", help="Observation shared key for timestamps")

    shared_flags = Unicode("flags", help="Observation shared key for common flags")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key to initialize"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to initialize"
    )

    hwp_angle = Unicode("hwp_angle", help="Observation shared key for HWP angle")

    azimuth = Unicode("azimuth", help="Observation shared key for Azimuth")

    elevation = Unicode("elevation", help="Observation shared key for Elevation")

    boresight_azel = Unicode(
        "boresight_azel", help="Observation shared key for boresight AZ/EL"
    )

    boresight_radec = Unicode(
        "boresight_radec", help="Observation shared key for boresight RA/DEC"
    )

    position = Unicode("position", help="Observation shared key for position")

    velocity = Unicode("velocity", help="Observation shared key for velocity")

    hwp_rpm = Float(None, allow_none=True, help="The rate (in RPM) of the HWP rotation")

    hwp_step = Quantity(
        None, allow_none=True, help="For stepped HWP, the angle of each step"
    )

    hwp_step_time = Quantity(
        None, allow_none=True, help="For stepped HWP, the time between steps"
    )

    elnod_start = Bool(False, help="Perform an el-nod before the scan")

    elnod_end = Bool(False, help="Perform an el-nod after the scan")

    elnods = List(None, allow_none=True, help="List of relative el_nods")

    elnod_every_scan = Bool(False, help="Perform el nods every scan")

    scanning_interval = Unicode("scanning", help="Interval name for scanning")

    turnaround_interval = Unicode("turnaround", help="Interval name for turnarounds")

    scan_leftright_interval = Unicode(
        "scan_leftright", help="Interval name for left to right scans"
    )

    turn_leftright_interval = Unicode(
        "turn_leftright", help="Interval name for turnarounds after left to right scans"
    )

    scan_rightleft_interval = Unicode(
        "scan_rightleft", help="Interval name for right to left scans"
    )

    turn_rightleft_interval = Unicode(
        "turn_rightleft", help="Interval name for turnarounds after right to left scans"
    )

    elnod_interval = Unicode("elnod", help="Interval name for elnods")

    sun_up_interval = Unicode(
        "sun_up", help="Interval name for times when the sun is up"
    )

    sun_close_interval = Unicode(
        "sun_close", help="Interval name for times when the sun is close"
    )

    sun_close_distance = Quantity(45.0 * u.degree, help="'Sun close' flagging distance")

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        if self.schedule is None:
            raise RuntimeError(
                "The schedule attribute must be set before calling exec()"
            )

        focalplane = self.telescope.focalplane
        rate = focalplane.sample_rate.to_value(u.Hz)
        comm = data.comm

        # List of detectors in this pipeline
        pipedets = None
        if detectors is None:
            pipedets = focalplane.detectors
        else:
            pipedets = list()
            for det in focalplane.detectors:
                if det in detectors:
                    pipedets.append(det)

        # Check valid combinations of options

        if (self.elnod_start or self.elnod_end) and len(self.elnods) == 0:
            raise RuntimeError(
                "If simulating elnods, you must specify the list of offsets"
            )

        # The global start is the beginning of the first scan

        mission_start = self.schedule.scans[0].start

        # Although there is no requirement that the sampling is contiguous from one
        # observation to the next, for simulations there is no need to restart the
        # sampling clock each observation.

        scan_starts = list()
        scan_stops = list()
        scan_offsets = list()
        scan_samples = list()

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
            scan_starts.append(start)
            scan_stops.append(stop)
            scan_samples.append(ns)
            scan_offsets.append(off)
            off += ns

        # FIXME:  Re-enable this when using astropy for coordinate transforms.
        # # Ensure that astropy IERS is downloaded
        # astropy_control(max_future=self.schedule.scans[-1].stop)

        # Distribute the observations uniformly among groups.  We take each scan and
        # weight it by the duration in samples.

        groupdist = distribute_discrete(scan_samples, comm.ngroups)

        det_ranks = comm.group_size
        if self.distribute_time:
            det_ranks = 1

        # Every process group creates its observations

        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            scan = self.schedule.scans[obindx]

            # Currently, El nods happen before or after the formal scan start / end.
            # This means that we don't know ahead of time the total number of samples
            # in the observation.  That in turn means we cannot create the observation
            # until after we simulate the motion, and therefore we do not yet have the
            # the process grid established.  Normally only rank zero of each grid
            # column would compute and store this data in shared memory.  However, since
            # we do not have that grid yet, every process simulates the scan.  This
            # should be relatively cheap.

            # Track the az / el range of all motion during this scan, including
            # el nods and any el modulation / steps.  These will be stored as
            # observation metadata after the simulation.
            scan_min_el = scan.el.to_value(u.radian)
            scan_max_el = scan_min_el
            scan_min_az = scan.az_min.to_value(u.radian)
            scan_max_az = scan.az_max.to_value(u.radian)

            # Time range of the science scans
            start_time = scan.start
            stop_time = start_time + datetime.timedelta(
                seconds=(float(scan_samples[obindx] - 1) / rate)
            )

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
                elnod_el = np.array([x.to_value(u.radian) for x in self.elnods])
                elnod_az = np.zeros_like(elnod_el) + scan.az_min.to_value(u.radian)

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
                    scan.start,
                    rate,
                    scan.az_min.to_value(u.radian),
                    scan.el.to_value(u.radian),
                    self.scan_rate_az.to_value(u.radian / u.second),
                    self.scan_accel_az.to_value(u.radian / u.second ** 2),
                    self.scan_rate_el.to_value(u.radian / u.second),
                    self.scan_accel_el.to_value(u.radian / u.second ** 2),
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
            ) = simulate_ces_scan(
                start_time.timestamp(),
                stop_time.timestamp(),
                rate,
                scan.el.to_value(u.radian),
                scan.az_min.to_value(u.radian),
                scan.az_max.to_value(u.radian),
                scan.az_min.to_value(u.radian),
                self.scan_rate_az.to_value(u.radian / u.second),
                self.scan_accel_az.to_value(u.radian / u.second ** 2),
                scan_min_az,
                scan_max_az,
                cosecant_modulation=self.scan_cosecant_modulation,
            )

            # Do any adjustments to the El motion
            if self.el_mod_rate.to_value(u.Hz) > 0:
                scan_min_el, scan_max_el = oscillate_el(
                    scan_times,
                    scan_el_data,
                    self.scan_rate_el.to_value(u.radian / u.second),
                    self.scan_accel_el.to_value(u.radian / u.second ** 2),
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
                    self.scan_accel_el.to_value(u.radian / u.second ** 2),
                    scan_min_el,
                    scan_max_el,
                    self.el_mod_step.to_value(u.radian),
                )

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
                    self.scan_accel_az.to_value(u.radian / u.second ** 2),
                    self.scan_rate_el.to_value(u.radian / u.second),
                    self.scan_accel_el.to_value(u.radian / u.second ** 2),
                    elnod_el,
                    elnod_az,
                    scan_min_az,
                    scan_max_az,
                    scan_min_el,
                    scan_max_el,
                )
                if len(elnod_times) > 0:
                    times.append(elnod_times)
                    az.append(elnod_az_data)
                    el.append(elnod_el_data)
                    ival_elnod.append((elnod_times[0], elnod_times[-1]))

            times = np.hstack(times)
            az = np.hstack(az)
            el = np.hstack(el)

            # Create the observation, now that we know the total number of samples.
            # We copy the original site information and add weather information for
            # this observation if needed.

            weather = None
            site = self.telescope.site

            if self.weather is not None:
                # Every observation has a unique site with unique weather
                # realization.
                mid_time = scan.start + (scan.stop - scan.start) / 2
                try:
                    weather = SimWeather(time=mid_time, name=self.weather)
                except RuntimeError:
                    # must be a file
                    weather = SimWeather(time=mid_time, file=self.weather)
                site = copy.deepcopy(self.telescope.site)
                site.weather = weather

            # Since we have a constant focalplane for all observations, we just use
            # a reference to the input rather than copying.
            telescope = Telescope(
                self.telescope.name,
                uid=self.telescope.uid,
                focalplane=focalplane,
                site=site,
            )

            name = f"{scan.name}_{int(scan.start.timestamp())}"
            ob = Observation(
                telescope,
                len(times),
                name=name,
                uid=name_UID(name),
                comm=comm.comm_group,
                process_rows=det_ranks,
            )

            # Scan limits
            ob["scan_min_az"] = scan_min_az * u.radian
            ob["scan_max_az"] = scan_max_az * u.radian
            ob["scan_min_el"] = scan_min_el * u.radian
            ob["scan_max_el"] = scan_max_el * u.radian

            # Create and set shared objects for timestamps, position, velocity, and
            # boresight.

            ob.shared.create(
                self.times,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.position,
                shape=(ob.n_local_samples, 3),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.velocity,
                shape=(ob.n_local_samples, 3),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.azimuth,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.elevation,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.boresight_azel,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.boresight_radec,
                shape=(ob.n_local_samples, 4),
                dtype=np.float64,
                comm=ob.comm_col,
            )

            # Optionally initialize detector data

            dets = ob.select_local_detectors(detectors)

            if self.det_data is not None:
                ob.detdata.ensure(self.det_data, dtype=np.float64, detectors=dets)

            if self.det_flags is not None:
                ob.detdata.ensure(self.det_flags, dtype=np.uint8, detectors=dets)

            # Only the first rank of the process grid columns sets / computes these.

            stamps = None
            position = None
            velocity = None
            az_data = None
            el_data = None
            bore_azel = None
            bore_radec = None

            if ob.comm_col_rank == 0:
                stamps = times[
                    ob.local_index_offset : ob.local_index_offset + ob.n_local_samples
                ]
                az_data = az[
                    ob.local_index_offset : ob.local_index_offset + ob.n_local_samples
                ]
                el_data = el[
                    ob.local_index_offset : ob.local_index_offset + ob.n_local_samples
                ]
                # Get the motion of the site for these times.
                position, velocity = site.position_velocity(stamps)
                # Convert Az / El to quaternions.
                # Remember that the azimuth is measured clockwise and the
                # longitude counter-clockwise.
                bore_azel = qa.from_angles(
                    np.pi / 2 - el_data,
                    -(az_data),
                    np.zeros_like(el_data),
                    IAU=False,
                )
                if scan.boresight_angle.to_value(u.radian) != 0:
                    zaxis = np.array([0, 0, 1.0])
                    rot = qa.rotation(zaxis, scan.boresight_angle.to_value(u.radian))
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
                hwp_start=scan_starts[obindx] * u.second,
                hwp_rpm=self.hwp_rpm,
                hwp_step=self.hwp_step,
                hwp_step_time=self.hwp_step_time,
            )

            # Create interval lists for our motion.  Since we simulated the scan on
            # every process, we don't need to communicate the global timespans of the
            # intervals (using create or create_col).  We can just create them directly.

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

            data.obs.append(ob)

        # For convenience, we additionally create a shared flag field with bits set
        # according to the different intervals.  This basically just saves workflows
        # from calling the FlagIntervals operator themselves.  Here we set the bits
        # according to what was done in toast2, so the scanning interval has no bits
        # set.

        flag_intervals = FlagIntervals(
            shared_flags=self.shared_flags,
            shared_flag_bytes=1,
            view_mask=[
                (self.turnaround_interval, 1),
                (self.scan_leftright_interval, 2),
                (self.scan_rightleft_interval, 4),
                (self.turn_leftright_interval, 3),
                (self.turn_rightleft_interval, 5),
                (self.sun_up_interval, 8),
                (self.sun_close_interval, 16),
                (self.elnod_interval, 32),
            ],
        )
        flag_intervals.apply(data, detectors=None)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        return {
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
            ]
        }

    def _accelerators(self):
        return list()
