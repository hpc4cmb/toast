# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from scipy.constants import degree

import healpy as hp

from astropy import units as u

from .. import qarray as qa

from ..utils import Environment, name_UID, Logger, rate_from_times

from ..dist import distribute_discrete

from ..timing import function_timer, Timer

from ..intervals import Interval

from ..schedule import SatelliteSchedule

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from ..observation import Observation

from ..instrument import Telescope

from ..healpix import ang2vec

from .operator import Operator

from .sim_hwp import simulate_hwp_response


@function_timer
def satellite_scanning(
    ob,
    ob_key,
    sample_offset=0,
    q_prec=None,
    spin_period_m=1.0,
    spin_angle_deg=85.0,
    prec_period_m=0.0,
    prec_angle_deg=0.0,
):
    """Generate boresight quaternions for a generic satellite.

    Given scan strategy parameters and the relevant angles
    and rates, generate an array of quaternions representing
    the rotation of the ecliptic coordinate axes to the
    boresight.

    Args:
        ob (Observation): The observation to populate.
        ob_key (str): The observation shared key to create.
        sample_offset (int): The global offset in samples from the start
            of the mission.
        q_prec (ndarray): If None (the default), then the
            precession axis will be fixed along the
            X axis.  If a 1D array of size 4 is given,
            This will be the fixed quaternion used
            to rotate the Z coordinate axis to the
            precession axis.  If a 2D array of shape
            (nsim, 4) is given, this is the time-varying
            rotation of the Z axis to the precession axis.
        spin_period_m (float): The period (in minutes) of the
            rotation about the spin axis.
        spin_angle_deg (float): The opening angle (in degrees)
            of the boresight from the spin axis.
        prec_period_m (float): The period (in minutes) of the
            rotation about the precession axis.
        prec_angle_deg (float): The opening angle (in degrees)
            of the spin axis from the precession axis.

    """
    env = Environment.get()
    tod_buffer_length = env.tod_buffer_length()

    first_samp = ob.local_index_offset
    n_samp = ob.n_local_samples
    ob.shared.create(ob_key, shape=(n_samp, 4), dtype=np.float64, comm=ob.comm_col)

    # Temporary buffer
    boresight = None

    # Only the first process in each grid column simulates the shared boresight data

    if ob.comm_col_rank == 0:
        boresight = np.zeros((n_samp, 4), dtype=np.float64)

        # Compute effective sample rate
        (sample_rate, dt, _, _, _) = rate_from_times(ob.shared["times"])

        spin_rate = None
        if spin_period_m > 0.0:
            spin_rate = 1.0 / (60.0 * spin_period_m)
        else:
            spin_rate = 0.0
        spin_angle = spin_angle_deg * np.pi / 180.0

        prec_rate = None
        if prec_period_m > 0.0:
            prec_rate = 1.0 / (60.0 * prec_period_m)
        else:
            prec_rate = 0.0
        prec_angle = prec_angle_deg * np.pi / 180.0

        xaxis = np.array([1, 0, 0], dtype=np.float64)
        yaxis = np.array([0, 1, 0], dtype=np.float64)
        zaxis = np.array([0, 0, 1], dtype=np.float64)

        if q_prec is not None:
            if (q_prec.shape[0] != n_samp) or (q_prec.shape[1] != 4):
                raise RuntimeError("q_prec array has wrong dimensions")

        buf_off = 0
        buf_n = tod_buffer_length
        while buf_off < n_samp:
            if buf_off + buf_n > n_samp:
                buf_n = n_samp - buf_off
            bslice = slice(buf_off, buf_off + buf_n)

            satrot = np.empty((buf_n, 4), np.float64)
            if q_prec is None:
                # in this case, we just have a fixed precession axis, pointing
                # along the ecliptic X axis.
                satrot[:, :] = np.tile(
                    qa.rotation(np.array([0.0, 1.0, 0.0]), np.pi / 2), buf_n
                ).reshape(-1, 4)
            elif q_prec.flatten().shape[0] == 4:
                # we have a fixed precession axis.
                satrot[:, :] = np.tile(q_prec.flatten(), buf_n).reshape(-1, 4)
            else:
                # we have full vector of quaternions
                satrot[:, :] = q_prec[bslice, :]

            # Time-varying rotation about precession axis.
            # Increment per sample is
            # (2pi radians) X (precrate) / (samplerate)
            # Construct quaternion from axis / angle form.

            # print("satrot = ", satrot[-1])

            precang = np.arange(buf_n, dtype=np.float64)
            precang += float(buf_off + first_samp + sample_offset)
            precang *= prec_rate / sample_rate
            precang = 2.0 * np.pi * (precang - np.floor(precang))

            cang = np.cos(0.5 * precang)
            sang = np.sin(0.5 * precang)

            precaxis = np.multiply(
                sang.reshape(-1, 1), np.tile(zaxis, buf_n).reshape(-1, 3)
            )

            precrot = np.concatenate((precaxis, cang.reshape(-1, 1)), axis=1)

            # Rotation which performs the precession opening angle
            precopen = qa.rotation(np.array([1.0, 0.0, 0.0]), prec_angle)

            # Time-varying rotation about spin axis.  Increment
            # per sample is
            # (2pi radians) X (spinrate) / (samplerate)
            # Construct quaternion from axis / angle form.

            spinang = np.arange(buf_n, dtype=np.float64)
            spinang += float(buf_off + first_samp + sample_offset)
            spinang *= spin_rate / sample_rate
            spinang = 2.0 * np.pi * (spinang - np.floor(spinang))

            cang = np.cos(0.5 * spinang)
            sang = np.sin(0.5 * spinang)

            spinaxis = np.multiply(
                sang.reshape(-1, 1), np.tile(zaxis, buf_n).reshape(-1, 3)
            )

            spinrot = np.concatenate((spinaxis, cang.reshape(-1, 1)), axis=1)

            # Rotation which performs the spin axis opening angle

            spinopen = qa.rotation(np.array([1.0, 0.0, 0.0]), spin_angle)

            # compose final rotation

            boresight[bslice, :] = qa.mult(
                satrot, qa.mult(precrot, qa.mult(precopen, qa.mult(spinrot, spinopen)))
            )
            buf_off += buf_n

    ob.shared[ob_key].set(boresight, offset=(0, 0), fromrank=0)

    return


@trait_docs
class SimSatellite(Operator):
    """Simulate a generic satellite motion.

    This simulates satellite pointing in regular intervals ("science scans") that
    may have some gaps in between for cooler cycles or other events.  The precession
    axis (anti-sun direction) is continuously slewed.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    telescope = Instance(
        klass=Telescope, allow_none=True, help="This must be an instance of a Telescope"
    )

    schedule = Instance(
        klass=SatelliteSchedule, allow_none=True, help="Instance of a SatelliteSchedule"
    )

    spin_angle = Quantity(
        30.0 * u.degree, help="The opening angle of the boresight from the spin axis"
    )

    prec_angle = Quantity(
        65.0 * u.degree,
        help="The opening angle of the spin axis from the precession axis",
    )

    hwp_rpm = Float(None, allow_none=True, help="The rate (in RPM) of the HWP rotation")

    hwp_step = Quantity(
        None, allow_none=True, help="For stepped HWP, the angle of each step"
    )

    hwp_step_time = Quantity(
        None, allow_none=True, help="For stepped HWP, the time between steps"
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

    times = Unicode("times", help="Observation shared key for timestamps")

    shared_flags = Unicode("flags", help="Observation shared key for common flags")

    hwp_angle = Unicode("hwp_angle", help="Observation shared key for HWP angle")

    boresight = Unicode("boresight_radec", help="Observation shared key for boresight")

    position = Unicode("position", help="Observation shared key for position")

    velocity = Unicode("velocity", help="Observation shared key for velocity")

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
            if not isinstance(sch, SatelliteSchedule):
                raise traitlets.TraitError(
                    "schedule must be an instance of a SatelliteSchedule"
                )
        return sch

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        zaxis = np.array([0, 0, 1], dtype=np.float64)
        log = Logger.get()
        if self.telescope is None:
            raise RuntimeError(
                "The telescope attribute must be set before calling exec()"
            )
        if self.schedule is None:
            raise RuntimeError(
                "The schedule attribute must be set before calling exec()"
            )
        focalplane = self.telescope.focalplane
        rate = focalplane.sample_rate.to_value(u.Hz)
        site = self.telescope.site
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

        # Group by detector sets and prune to include only the detectors we
        # are using.
        detsets = None
        if self.detset_key is not None:
            detsets = dict()
            dsets = focalplane.detector_groups(self.detset_key)
            for k, v in dsets.items():
                detsets[k] = list()
                for d in v:
                    if d in pipedets:
                        detsets[k].append(d)

        # Data distribution in the detector direction
        det_ranks = comm.group_size
        if self.distribute_time:
            det_ranks = 1

        # Verify that we have enough data for all of our processes.  If we are
        # distributing by time, we have no sample sets, so can accomodate any
        # number of processes.  If we are distributing by detector, we must have
        # at least one detector set for each process.

        if not self.distribute_time:
            # distributing by detector...
            n_detset = None
            if detsets is None:
                # Every detector is independently distributed
                n_detset = len(pipedets)
            else:
                n_detset = len(detsets)
            if det_ranks > n_detset:
                if comm.group_rank == 0:
                    msg = f"Group {comm.group} has {comm.group_size} processes but {n_detset} detector sets."
                    log.error(msg)
                    raise RuntimeError(msg)

        # The global start is the beginning of the first scan

        mission_start = self.schedule.scans[0].start

        # Satellite motion is continuous across multiple observations, so we simulate
        # continuous sampling and find the actual start / stop times for the samples
        # that fall in each scan time range.

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

        # Distribute the observations uniformly among groups.  We take each scan and
        # weight it by the duration.

        groupdist = distribute_discrete(scan_samples, comm.ngroups)

        # Every process group creates its observations

        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            scan = self.schedule.scans[obindx]

            ob = Observation(
                self.telescope,
                scan_samples[obindx],
                name=f"{scan.name}_{int(scan.start.timestamp())}",
                uid=name_UID(scan.name),
                comm=comm.comm_group,
                detector_sets=detsets,
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
            q_prec = None

            if ob.comm_col_rank == 0:
                start_time = scan_starts[obindx] + float(ob.local_index_offset) / rate
                stop_time = start_time + float(ob.n_local_samples - 1) / rate
                stamps = np.linspace(
                    start_time,
                    stop_time,
                    num=ob.n_local_samples,
                    endpoint=True,
                    dtype=np.float64,
                )

                # Get the motion of the site for these times.
                position, velocity = site.position_velocity(stamps)

                # Get the quaternions for the precession axis.  For now, assume that
                # it simply points away from the solar system barycenter

                pos_norm = np.sqrt((position * position).sum(axis=1)).reshape(-1, 1)
                pos_norm = 1.0 / pos_norm
                prec_axis = pos_norm * position
                q_prec = qa.from_vectors(
                    np.tile(zaxis, ob.n_local_samples).reshape(-1, 3), prec_axis
                )

            ob.shared[self.times].set(stamps, offset=(0,), fromrank=0)
            ob.shared[self.position].set(position, offset=(0, 0), fromrank=0)
            ob.shared[self.velocity].set(velocity, offset=(0, 0), fromrank=0)

            # Create boresight pointing

            satellite_scanning(
                ob,
                self.boresight,
                sample_offset=scan_offsets[obindx],
                q_prec=q_prec,
                spin_period_m=scan.spin_period.to_value(u.minute),
                spin_angle_deg=self.spin_angle.to_value(u.degree),
                prec_period_m=scan.prec_period.to_value(u.minute),
                prec_angle_deg=self.prec_angle.to_value(u.degree),
            )

            # Set HWP angle

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
