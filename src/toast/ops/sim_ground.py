# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from scipy.constants import degree

import healpy as hp

from astropy import units as u

import ephem

from .. import qarray as qa

from ..utils import Environment, name_UID, Logger, rate_from_times

from ..dist import distribute_uniform

from ..timing import function_timer, Timer

from ..intervals import Interval, regular_intervals

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from ..observation import Observation

from ..instrument import Telescope

from ..schedule import Schedule

from ..healpix import ang2vec

from .operator import Operator

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

    schedule = Instance(klass=Schedule, allow_none=True, help="The observing schedule")

    scan_rate = Quantity(1.0 * u.degree / u.second, help="The sky scanning rate")

    scan_rate_el = Quantity(
        None, allow_none=True, help="The sky elevation scanning rate"
    )

    scan_accel = Quantity(
        0.1 * u.degree / u.second ** 2,
        help="Mount scanning rate acceleration for turnarounds",
    )

    scan_accel_el = Quantity(
        None, allow_none=True, help="Mount elevation rate acceleration."
    )

    cosecant_modulation = Bool(
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

    hwp_angle = Unicode("hwp_angle", help="Observation shared key for HWP angle")

    boresight_azel = Unicode(
        "boresight_radec", help="Observation shared key for boresight AZ/EL"
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

    start_with_elnod = Bool(True, help="Perform an el-nod before the scan")

    end_with_elnod = Bool(False, help="Perform an el-nod after the scan")

    el_nod = List(None, allow_none=True, help="List of relative el_nods")

    scanning_interval = Unicode("scanning", help="Interval name for scanning")

    turnaround_interval = Unicode("turnaround", help="Interval name for turnarounds")

    scan_left_interval = Unicode("scan_left", help="Interval name for left-going scans")

    scan_right_interval = Unicode(
        "scan_right", help="Interval name for right-going scans"
    )

    el_nod_interval = Unicode("elnod", help="Interval name for elnods")

    sun_up_interval = Unicode(
        "sun_up", help="Interval name for times when the sun is up"
    )

    sun_close_interval = Unicode(
        "sun_close", help="Interval name for times when the sun is close"
    )

    @traitlets.validate("schedule")
    def _check_schedule(self, proposal):
        sch = proposal["value"]
        if sch is not None:
            if not isinstance(sch, Schedule):
                raise traitlets.TraitError("schedule must be an instance of a Schedule")
            if sch.telescope is None:
                raise traitlets.TraitError("schedule must have a telescope")
            if sch.ceslist is None:
                raise traitlets.TraitError("schedule must have a list of CESs")
            tele = sch.telescope
            try:
                dets = tele.focalplane.detectors
            except Exception:
                raise traitlets.TraitError(
                    "schedule telescope must be a Telescope instance with a focalplane"
                )
        return sch

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._AU = 149597870.7
        self._radperday = 0.01720209895
        self._radpersec = self._radperday / 86400.0
        self._earthspeed = self._radpersec * self._AU

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        if self.schedule is None:
            raise RuntimeError(
                "The schedule attribute must be set before calling exec()"
            )
        focalplane = self.schedule.telescope.focalplane
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

        # Distribute the observations among groups in a load balanced way based on
        # the duration of each CES.

        num_obs = len(self.schedule.ceslist)
        obs_sizes = np.array(
            [int(x.stop_time - x.start_time) + 1 for x in self.schedule.ceslist]
        )

        groupdist = distribute_discrete(obs_sizes, comm.ngroups)

        # Set the size of the process grid for each group

        det_ranks = comm.group_size
        if self.distribute_time:
            det_ranks = 1

        # Every process group creates its observations

        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            # The CES for this observation
            ces = self.schedule.ceslist[obindx]

            # Set the boresight pointing based on the given scan parameters

            timer = Timer()
            if self._report_timing:
                if mpicomm is not None:
                    mpicomm.Barrier()
                timer.start()

            self._times = np.array([])
            self._commonflags = np.array([], dtype=np.uint8)
            self._az = np.array([])
            self._el = np.array([])

            nsample_elnod = 0
            if start_with_elnod:
                # Begin with an el-nod
                nsample_elnod = self.simulate_elnod(
                    self._firsttime, azmin * degree, el * degree
                )
                if nsample_elnod > 0:
                    t_elnod = self._times[-1] - self._times[0]
                    # Shift the time stamps so that the CES starts at the prescribed time
                    self._times -= t_elnod
                    self._firsttime -= t_elnod

            nsample_ces = self.simulate_scan(samples)

            if end_with_elnod and self._elnod_az is not None:
                # Append en el-nod after the CES
                self._elnod_az[:] = self._az[-1]
                nsample_elnod = self.simulate_elnod(
                    self._times[-1], self._az[-1], self._el[-1]
                )
            self._lasttime = self._times[-1]
            samples = self._times.size

            if self._report_timing:
                if mpicomm is not None:
                    mpicomm.Barrier()
                if mpicomm is None or mpicomm.rank == 0:
                    timer.report_clear("TODGround: simulate scan")

            # Create a list of subscans that excludes the turnarounds.
            # All processes in the group still have all samples.

            self.subscans = []
            self._subscan_min_length = 10  # in samples
            for istart, istop in zip(self._stable_starts, self._stable_stops):
                if istop - istart < self._subscan_min_length or istart < nsample_elnod:
                    self._commonflags[istart:istop] |= self.TURNAROUND
                    continue
                start = self._firsttime + istart / self._rate
                stop = self._firsttime + istop / self._rate
                self.subscans.append(
                    Interval(start=start, stop=stop, first=istart, last=istop - 1)
                )

            if len(self._stable_stops) > 0:
                self._commonflags[self._stable_stops[-1] :] |= self.TURNAROUND

            if np.sum((self._commonflags & self.TURNAROUND) == 0) == 0 and do_ces:
                raise RuntimeError(
                    "The entire TOD is flagged as turnaround. Samplerate too low "
                    "({} Hz) or scanrate too high ({} deg/s)?"
                    "".format(rate, scanrate)
                )

            if self._report_timing:
                if mpicomm is not None:
                    mpicomm.Barrier()
                if mpicomm is None or mpicomm.rank == 0:
                    timer.report_clear("TODGround: list valid intervals")

            self._fp = detectors
            self._detlist = sorted(list(self._fp.keys()))

            # call base class constructor to distribute data

            props = {
                "site_lon": site_lon,
                "site_lat": site_lat,
                "site_alt": site_alt,
                "azmin": azmin,
                "azmax": azmax,
                "el": el,
                "scanrate": scanrate,
                "scan_accel": scan_accel,
                "el_min": el_min,
                "sun_angle_min": sun_angle_min,
            }
            super().__init__(
                mpicomm,
                self._detlist,
                samples,
                sampsizes=[samples],
                sampbreaks=None,
                meta=props,
                **kwargs
            )

            if self._report_timing:
                if mpicomm is not None:
                    mpicomm.Barrier()
                if mpicomm is None or mpicomm.rank == 0:
                    timer.report_clear("TODGround: call base class constructor")

            self.translate_pointing()

            self.crop_vectors()

            if self._report_timing:
                if mpicomm is not None:
                    mpicomm.Barrier()
                if mpicomm is None or mpicomm.rank == 0:
                    timer.report_clear("TODGround: translate scan pointing")

            # If HWP parameters are specified, simulate and cache HWP angle

            simulate_hwp(self, hwprpm, hwpstep, hwpsteptime)

            # Check that we do not have too many processes for our data distribution.

            if self.distribute_time:
                # We are distributing data by scan sets
                if comm.group_size > len(self.schedule.ceslist):
                    msg = "process group is too large for the number of CESs"
                    if comm.world_rank == 0:
                        log.error(msg)
                    raise RuntimeError(msg)
            else:
                # We are distributing data by detector sets.
                if comm.group_size > len(pipedets):
                    msg = "process group is too large for the number of detectors"
                    if comm.world_rank == 0:
                        log.error(msg)
                    raise RuntimeError(msg)

            ob = Observation(
                self.schedule.telescope,
                obsrange[obindx].samples,
                name=ces.name,
                UID=name_UID(ces.name),
                comm=comm.comm_group,
                process_rows=det_ranks,
            )

            # Create shared objects for timestamps, common flags, position,
            # and velocity.
            ob.shared.create(
                self.times,
                shape=(ob.n_local_samples,),
                dtype=np.float64,
                comm=ob.comm_col,
            )
            ob.shared.create(
                self.shared_flags,
                shape=(ob.n_local_samples,),
                dtype=np.uint8,
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

            # Rank zero of each grid column creates the data
            stamps = None
            position = None
            velocity = None
            if ob.comm_col_rank == 0:
                start_abs = ob.local_index_offset + obsrange[obindx].first
                start_time = (
                    obsrange[obindx].start + float(start_abs) / focalplane.sample_rate
                )
                stop_time = (
                    start_time + float(ob.n_local_samples) / focalplane.sample_rate
                )
                stamps = np.linspace(
                    start_time,
                    stop_time,
                    num=ob.n_local_samples,
                    endpoint=False,
                    dtype=np.float64,
                )
                # For this simple class, assume that the Earth is located
                # along the X axis at time == 0.0s.  We also just use the
                # mean values for distance and angular speed.  Classes for
                # real experiments should obviously use ephemeris data.
                rad = np.fmod(
                    (start_time - self.start_time.to_value(u.second)) * self._radpersec,
                    2.0 * np.pi,
                )
                ang = radinc * np.arange(ob.n_local_samples, dtype=np.float64) + rad
                x = self._AU * np.cos(ang)
                y = self._AU * np.sin(ang)
                z = np.zeros_like(x)
                position = np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

                ang = (
                    radinc * np.arange(ob.n_local_samples, dtype=np.float64)
                    + rad
                    + (0.5 * np.pi)
                )
                x = self._earthspeed * np.cos(ang)
                y = self._earthspeed * np.sin(ang)
                z = np.zeros_like(x)
                velocity = np.ravel(np.column_stack((x, y, z))).reshape((-1, 3))

            ob.shared[self.times].set(stamps, offset=(0,), fromrank=0)
            ob.shared[self.position].set(position, offset=(0, 0), fromrank=0)
            ob.shared[self.velocity].set(velocity, offset=(0, 0), fromrank=0)

            # Create boresight pointing
            start_abs = ob.local_index_offset + obsrange[obindx].first
            degday = 360.0 / 365.25

            q_prec = None
            if ob.comm_col_rank == 0:
                q_prec = slew_precession_axis(
                    first_samp=start_abs,
                    n_samp=ob.n_local_samples,
                    sample_rate=focalplane.sample_rate,
                    deg_day=degday,
                )

            satellite_scanning(
                ob,
                self.boresight,
                sample_offset=start_abs,
                q_prec=q_prec,
                spin_period_m=self.spin_period.to_value(u.minute),
                spin_angle_deg=self.spin_angle.to_value(u.degree),
                prec_period_m=self.prec_period.to_value(u.minute),
                prec_angle_deg=self.prec_angle.to_value(u.degree),
            )

            # Set HWP angle

            simulate_hwp_response(
                ob,
                ob_time_key=self.times,
                ob_angle_key=self.hwp_angle,
                ob_mueller_key=None,
                hwp_start=obsrange[obindx].start * u.second,
                hwp_rpm=self.hwp_rpm,
                hwp_step=self.hwp_step,
                hwp_step_time=self.hwp_step_time,
            )

            data.obs.append(ob)

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
                self.boresight,
                self.hwp_angle,
                self.position,
                self.velocity,
            ]
        }

    def _accelerators(self):
        return list()
