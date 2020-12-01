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

from ..dist import distribute_uniform

from ..timing import function_timer, Timer

from ..intervals import Interval, regular_intervals

from ..noise_sim import AnalyticNoise

from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from ..observation import Observation

from ..instrument import Telescope

from ..healpix import ang2vec

from .operator import Operator

from .sim_hwp import simulate_hwp_response


@function_timer
def slew_precession_axis(first_samp=0, n_samp=None, sample_rate=None, deg_day=None):
    """Generate quaternions for constantly slewing precession axis.

    This constructs quaternions which rotates the Z coordinate axis
    to the X/Y plane, and then slowly rotates this.  This can be used
    to generate quaternions for the precession axis used in satellite
    scanning simulations.

    Args:
        first_samp (int): The offset in samples from the start
            of rotation.
        n_samp (int): The number of samples to simulate.
        sample_rate (float): The sampling rate in Hz.
        deg_day (float): The rotation rate in degrees per day.

    """
    env = Environment.get()
    tod_buffer_length = env.tod_buffer_length()

    zaxis = np.array([0.0, 0.0, 1.0])

    # this is the increment in radians per sample
    angincr = deg_day * (np.pi / 180.0) / (24.0 * 3600.0 * sample_rate)

    result = np.zeros((n_samp, 4), dtype=np.float64)

    # Compute the time-varying quaternions representing the rotation
    # from the coordinate frame to the precession axis frame.  The
    # angle of rotation is fixed (PI/2), but the axis starts at the Y
    # coordinate axis and sweeps.

    buf_off = 0
    buf_n = tod_buffer_length
    while buf_off < n_samp:
        if buf_off + buf_n > n_samp:
            buf_n = n_samp - buf_off
        bslice = slice(buf_off, buf_off + buf_n)

        satang = np.arange(buf_n, dtype=np.float64)
        satang *= angincr
        satang += angincr * (buf_off + first_samp)
        # satang += angincr * firstsamp + (np.pi / 2)

        cang = np.cos(satang)
        sang = np.sin(satang)

        # this is the time-varying rotation axis
        sataxis = np.concatenate(
            (cang.reshape(-1, 1), sang.reshape(-1, 1), np.zeros((buf_n, 1))), axis=1
        )

        result[bslice, :] = qa.from_vectors(
            np.tile(zaxis, buf_n).reshape(-1, 3), sataxis
        )
        buf_off += buf_n

    return result


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

    start_time = Quantity(0.0 * u.second, help="The mission start time")

    observation_time = Quantity(0.1 * u.hour, help="The time span for each observation")

    gap_time = Quantity(0.0 * u.hour, help="The gap between each observation")

    num_observations = Int(1, help="The number of observations to simulate")

    spin_period = Quantity(
        10.0 * u.minute, help="The period of the rotation about the spin axis"
    )

    spin_angle = Quantity(
        30.0 * u.degree, help="The opening angle of the boresight from the spin axis"
    )

    prec_period = Quantity(
        50.0 * u.minute, help="The period of the rotation about the precession axis"
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

    times = Unicode("times", help="Observation shared key for timestamps")

    shared_flags = Unicode(
        "shared_flags", help="Observation shared key for common flags"
    )

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._AU = 149597870.7
        self._radperday = 0.01720209895
        self._radpersec = self._radperday / 86400.0
        self._earthspeed = self._radpersec * self._AU

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        if self.telescope is None:
            raise RuntimeError(
                "The telescope attribute must be set before calling exec()"
            )
        focalplane = self.telescope.focalplane
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

        if comm.group_size > len(pipedets):
            if comm.world_rank == 0:
                log.error("process group is too large for the number of detectors")
                comm.comm_world.Abort()

        # Distribute the observations uniformly among groups

        groupdist = distribute_uniform(self.num_observations, comm.ngroups)

        # Compute global time and sample ranges of all observations

        obsrange = regular_intervals(
            self.num_observations,
            self.start_time.to_value(u.second),
            0,
            focalplane.sample_rate,
            self.observation_time.to_value(u.second),
            self.gap_time.to_value(u.second),
        )

        det_ranks = comm.group_size
        if self.distribute_time:
            det_ranks = 1

        # Every process group creates its observations

        radinc = self._radpersec / focalplane.sample_rate

        group_firstobs = groupdist[comm.group][0]
        group_numobs = groupdist[comm.group][1]

        for obindx in range(group_firstobs, group_firstobs + group_numobs):
            obname = "science_{:05d}".format(obindx)
            ob = Observation(
                self.telescope,
                obsrange[obindx].samples,
                name=obname,
                UID=name_UID(obname),
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
