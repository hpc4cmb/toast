# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..atm import AtmSim, available_atm, available_utils
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, Timer
from .operator import Operator
from .pipeline import Pipeline
from .sim_tod_atm_utils import ObserveAtmosphere

if available_atm:
    from ..atm import AtmSim

if available_utils:
    from ..atm import atm_absorption_coefficient_vec, atm_atmospheric_loading_vec


@trait_docs
class SimAtmosphere(Operator):
    """Operator which generates atmosphere timestreams for detectors.

    All processes collectively generate the atmospheric realization. Then each process
    passes through its local data and observes the atmosphere.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating atmosphere timestreams",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector frame",
    )

    detector_weights = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector weights",
    )

    polarization_fraction = Float(
        0,
        help="Polarization fraction (only Q polarization).",
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    turnaround_interval = Unicode(
        "turnaround", allow_none=True, help="Interval name for turnarounds"
    )

    realization = Int(
        0, help="If simulating multiple realizations, the realization index"
    )

    component = Int(
        123456, help="The component index to use for this atmosphere simulation"
    )

    lmin_center = Quantity(
        0.01 * u.meter, help="Kolmogorov turbulence dissipation scale center"
    )

    lmin_sigma = Quantity(
        0.001 * u.meter, help="Kolmogorov turbulence dissipation scale sigma"
    )

    lmax_center = Quantity(
        10.0 * u.meter, help="Kolmogorov turbulence injection scale center"
    )

    lmax_sigma = Quantity(
        10.0 * u.meter, help="Kolmogorov turbulence injection scale sigma"
    )

    gain = Float(1e-5, help="Scaling applied to the simulated TOD")

    zatm = Quantity(40000.0 * u.meter, help="Atmosphere extent for temperature profile")

    zmax = Quantity(
        2000.0 * u.meter, help="Atmosphere extent for water vapor integration"
    )

    xstep = Quantity(100.0 * u.meter, help="Size of volume elements in X direction")

    ystep = Quantity(100.0 * u.meter, help="Size of volume elements in Y direction")

    zstep = Quantity(100.0 * u.meter, help="Size of volume elements in Z direction")

    z0_center = Quantity(
        2000.0 * u.meter, help="Central value of the water vapor distribution"
    )

    z0_sigma = Quantity(0.0 * u.meter, help="Sigma of the water vapor distribution")

    wind_dist = Quantity(
        3000.0 * u.meter,
        help="Maximum wind drift before discarding the volume and creating a new one",
    )

    fade_time = Quantity(
        60.0 * u.s,
        help="Fade in/out time to avoid a step at wind break.",
    )

    sample_rate = Quantity(
        None,
        allow_none=True,
        help="Rate at which to sample atmospheric TOD before interpolation.  "
        "Default is no interpolation.",
    )

    nelem_sim_max = Int(10000, help="Controls the size of the simulation slices")

    n_bandpass_freqs = Int(
        100,
        help="The number of sampling frequencies used when convolving the bandpass with atmosphere absorption and loading",
    )

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="Directory to use for loading / saving atmosphere realizations",
    )

    overwrite_cache = Bool(
        False, help="If True, redo and overwrite any cached atmospheric realizations."
    )

    cache_only = Bool(
        False, help="If True, only cache the atmosphere, do not observe it."
    )

    debug_spectrum = Bool(False, help="If True, dump out Kolmogorov debug files")

    debug_tod = Bool(False, help="If True, dump TOD to pickle files")

    debug_snapshots = Bool(
        False, help="If True, dump snapshots of the atmosphere slabs to pickle files"
    )

    debug_plots = Bool(False, help="If True, make plots of the debug snapshots")

    add_loading = Bool(True, help="Add elevation-dependent loading.")

    field_of_view = Quantity(
        None,
        allow_none=True,
        help="Override the focalplane field of view",
    )

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("detector_weights")
    def _check_detector_weights(self, proposal):
        detweights = proposal["value"]
        if detweights is not None:
            if not isinstance(detweights, Operator):
                raise traitlets.TraitError(
                    "detector_weights should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "quats",
                "weights",
                "mode",
            ]:
                if not detweights.has_trait(trt):
                    msg = f"detector_weights operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detweights

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not available_utils:
            log = Logger.get()
            msg = "TOAST was compiled without the libaatm library, which is "
            msg += "required for observations of simulated atmosphere."
            log.error(msg)
            raise RuntimeError(msg)
        if not available_atm:
            log = Logger.get()
            msg = "TOAST was compiled without the SuiteSparse package, which is "
            msg += "required for observations of simulated atmosphere."
            log.error(msg)
            raise RuntimeError(msg)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        # Since each process group has the same set of observations, we use the group
        # communicator for collectively simulating the atmosphere slab for each
        # observation.
        comm = data.comm.comm_group
        group = data.comm.group
        rank = data.comm.group_rank
        comm_node = data.comm.comm_group_node
        comm_node_rank = data.comm.comm_group_node_rank

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Name of the intervals for ranges valid for a given wind chunk
        wind_intervals = "wind"

        # A view that combines user input and wind breaks
        if self.view is None and self.detector_pointing.view is None:
            temporary_view = wind_intervals
        else:
            temporary_view = "temporary_view"

        # Observation key for storing the atmosphere sims
        atm_sim_key = f"{self.name}_atm_sim"

        # Observation key for storing absorption and loading
        absorption_key = f"{self.name}_absorption"
        if self.add_loading:
            loading_key = f"{self.name}_loading"
        else:
            loading_key = None

        # Set up the observing operator
        if self.shared_flags is None:
            # Cannot observe samples that have no pointing
            shared_flags = self.detector_pointing.shared_flags
            shared_flag_mask = self.detector_pointing.shared_flag_mask
        else:
            # Trust that the user has provided a flag that excludes samples
            # without pointing
            shared_flags = self.shared_flags
            shared_flag_mask = self.shared_flag_mask

        observe_atm = ObserveAtmosphere(
            times=self.times,
            det_data=self.det_data,
            quats_azel=self.detector_pointing.quats,
            view=temporary_view,
            shared_flags=shared_flags,
            shared_flag_mask=shared_flag_mask,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            wind_view=wind_intervals,
            fade_time=self.fade_time,
            sim=atm_sim_key,
            absorption=absorption_key,
            loading=loading_key,
            n_bandpass_freqs=self.n_bandpass_freqs,
            gain=self.gain,
            polarization_fraction=self.polarization_fraction,
            sample_rate=self.sample_rate,
            debug_tod=self.debug_tod,
        )
        if self.detector_weights is not None:
            observe_atm.weights_mode = self.detector_weights.mode
            observe_atm.weights = self.detector_weights.weights

        for iobs, ob in enumerate(data.obs):
            if ob.name is None:
                msg = "Atmosphere simulation requires each observation to have a name"
                raise RuntimeError(msg)

            if ob.comm_row_size > 1:
                # FIXME: we can remove this check after making the _get_time_range()
                # method more general below.
                msg = "Atmosphere simulation requires data distributed by detector, not time"
                raise RuntimeError(msg)

            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Prefix for logging
            log_prefix = f"{group} : {ob.name} : "

            # The site and weather for this observation
            site = ob.telescope.site
            if not hasattr(site, "weather") or site.weather is None:
                raise RuntimeError(
                    "Cannot simulate atmosphere for sites without weather"
                )
            weather = site.weather

            # Make sure detector data output exists
            exists = ob.detdata.ensure(self.det_data, detectors=dets)

            # Check that our view is fully covered by detector pointing.  If the
            # detector_pointing view is None, then it has all samples.  If our own
            # view was None, then it would have been set to the detector_pointing
            # view above.
            if (view is not None) and (self.detector_pointing.view is not None):
                if ob.intervals[view] != ob.intervals[self.detector_pointing.view]:
                    # We need to check intersection
                    intervals = ob.intervals[self.view]
                    detector_intervals = ob.intervals[self.detector_pointing.view]
                    intersection = detector_intervals & intervals
                    if intersection != intervals:
                        msg = "view {} is not fully covered by valid detector pointing".format(
                            self.view
                        )
                        raise RuntimeError(msg)

            # The full time range of this observation.  The processes are arranged
            # in a grid, and processes along the same column have the same timestamps.
            # So the first process of the group has the earliest times and the last
            # process of the group has the latest times.

            times = ob.shared[self.times]

            tmin_tot = times[0]
            tmax_tot = times[-1]
            if comm is not None:
                tmin_tot = comm.bcast(tmin_tot, root=0)
                tmax_tot = comm.bcast(tmax_tot, root=(data.comm.group_size - 1))

            key1, key2, counter1, counter2 = self._get_rng_keys(ob)

            cachedir = self._get_cache_dir(ob, comm)

            ob[atm_sim_key] = list()

            if comm is not None:
                comm.Barrier()
            if rank == 0:
                msg = f"{log_prefix}Setting up atmosphere simulation"
                log.debug(msg)

            scan_range = self._get_scan_range(ob, comm, log_prefix)

            # Compute the absorption and loading for this observation
            self._compute_absorption_and_loading(ob, absorption_key, loading_key, comm)

            # Loop over the time span in "wind_time"-sized chunks.
            # wind_time is intended to reflect the correlation length
            # in the atmospheric noise.

            wind_times = list()

            tmr = Timer()
            if comm is not None:
                comm.Barrier()
            tmr.start()

            tmin = tmin_tot
            istart = 0
            counter1start = counter1

            while tmin < tmax_tot:
                if comm is not None:
                    comm.Barrier()
                istart, istop, tmax = self._get_time_range(
                    tmin, istart, times, tmax_tot, ob, weather
                )
                wind_times.append((tmin, tmax))

                if rank == 0:
                    log.debug(
                        f"{log_prefix}Instantiating atmosphere for t = "
                        f"{tmin - tmin_tot:10.1f} s - {tmax - tmin_tot:10.1f} s "
                        f"out of {tmax_tot - tmin_tot:10.1f} s"
                    )

                ind = slice(istart, istop)
                nind = istop - istart

                rmin = 0
                rmax = 100
                scale = 10
                counter2start = counter2
                counter1 = counter1start
                xstep_current = u.Quantity(self.xstep)
                ystep_current = u.Quantity(self.ystep)
                zstep_current = u.Quantity(self.zstep)

                sim_list = list()

                while rmax < 100000:
                    sim, counter2 = self._simulate_atmosphere(
                        weather,
                        scan_range,
                        tmin,
                        tmax,
                        comm,
                        comm_node,
                        comm_node_rank,
                        key1,
                        key2,
                        counter1,
                        counter2start,
                        cachedir,
                        log_prefix,
                        tmin_tot,
                        tmax_tot,
                        xstep_current,
                        ystep_current,
                        zstep_current,
                        rmin,
                        rmax,
                    )
                    sim_list.append(sim)

                    if self.debug_plots or self.debug_snapshots:
                        self._plot_snapshots(
                            sim,
                            log_prefix,
                            ob.name,
                            scan_range,
                            tmin,
                            tmax,
                            comm,
                            rmin,
                            rmax,
                        )

                    rmin = rmax
                    rmax *= scale
                    xstep_current *= np.sqrt(scale)
                    ystep_current *= np.sqrt(scale)
                    zstep_current *= np.sqrt(scale)
                    counter1 += 1

                ob[atm_sim_key].append(sim_list)
                tmin = tmax

            # Create the wind intervals
            ob.intervals.create_col(wind_intervals, wind_times, ob.shared[self.times])

            # Create temporary intervals by combining views
            if temporary_view != wind_intervals:
                ob.intervals[temporary_view] = (
                    ob.intervals[view] & ob.intervals[wind_intervals]
                )

            if not self.cache_only:
                # Observation pipeline.  We do not want to store persistent detector
                # pointing, so we build a small pipeline that runs one detector at a
                # time on only the current observation.
                pipe_data = data.select(obs_index=iobs)

                operators = [self.detector_pointing]
                if self.detector_weights is not None:
                    operators.append(self.detector_weights)
                operators.append(observe_atm)

                observe_pipe = Pipeline(operators=operators, detector_sets=["SINGLE"])
                observe_pipe.apply(pipe_data)

            # Delete the atmosphere slabs for this observation
            for wind_slabs in ob[atm_sim_key]:
                for slab in wind_slabs:
                    slab.close()
            del ob[atm_sim_key]

            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                log.debug(
                    f"{log_prefix}Simulate and observe atmosphere:  "
                    f"{tmr.seconds()} seconds"
                )

            if temporary_view != wind_intervals:
                del ob.intervals[temporary_view]
            del ob.intervals[wind_intervals]

    def _get_rng_keys(self, obs):
        """Get random number keys and counters for an observation.

        The observation and telescope UID values are 32bit integers.  The realization
        index is typically smaller, and should fit within 2^16.  The component index
        is a small integer.

        The random number generator accepts a key and a counter, each made of two 64bit
        integers.  For a given observation we set these as:
            key 1 = site UID * 2^32 + telescope UID
            key 2 = observation UID * 2^32 + realization * 2^16 + component
            counter 1 = hierarchical cone counter
            counter 2 = 0 (this is incremented per RNG stream sample)

        Args:
            obs (Observation):  The current observation.

        Returns:
            (tuple): The key1, key2, counter1, counter2 to use.

        """
        telescope = obs.telescope.uid
        site = obs.telescope.site.uid
        obsid = obs.uid

        # site UID in higher bits, telescope UID in lower bits
        key1 = site * 2**32 + telescope

        # Observation UID in higher bits, realization and component in lower bits
        key2 = obsid * 2**32 + self.realization * 2**16 + self.component

        # This tracks the number of cones simulated due to the wind speed.
        counter1 = 0

        # Starting point for the observation, incremented for each slice.
        counter2 = 0

        return key1, key2, counter1, counter2

    @function_timer
    def _compute_absorption_and_loading(self, obs, absorption_key, loading_key, comm):
        """Compute the (common) absorption and loading prior to bandpass convolution."""

        if obs.telescope.focalplane.bandpass is None:
            raise RuntimeError("Focalplane does not define bandpass")
        altitude = obs.telescope.site.earthloc.height
        weather = obs.telescope.site.weather
        bandpass = obs.telescope.focalplane.bandpass

        freq_min, freq_max = bandpass.get_range()
        n_freq = self.n_bandpass_freqs
        freqs = np.linspace(freq_min, freq_max, n_freq)
        if comm is None:
            ntask = 1
            my_rank = 0
        else:
            ntask = comm.size
            my_rank = comm.rank
        n_freq_task = int(np.ceil(n_freq / ntask))
        my_start = min(my_rank * n_freq_task, n_freq)
        my_stop = min(my_start + n_freq_task, n_freq)
        my_n_freq = my_stop - my_start

        if my_n_freq > 0:
            absorption = atm_absorption_coefficient_vec(
                altitude.to_value(u.meter),
                weather.air_temperature.to_value(u.Kelvin),
                weather.surface_pressure.to_value(u.Pa),
                weather.pwv.to_value(u.mm),
                freqs[my_start].to_value(u.GHz),
                freqs[my_stop - 1].to_value(u.GHz),
                my_n_freq,
            )
            loading = atm_atmospheric_loading_vec(
                altitude.to_value(u.meter),
                weather.air_temperature.to_value(u.Kelvin),
                weather.surface_pressure.to_value(u.Pa),
                weather.pwv.to_value(u.mm),
                freqs[my_start].to_value(u.GHz),
                freqs[my_stop - 1].to_value(u.GHz),
                my_n_freq,
            )
        else:
            absorption, loading = [], []

        if comm is not None:
            absorption = np.hstack(comm.allgather(absorption))
            loading = np.hstack(comm.allgather(loading))
        obs[absorption_key] = absorption
        obs[loading_key] = loading

    def _get_cache_dir(self, obs, comm):
        obsid = obs.uid
        if self.cache_dir is None:
            cachedir = None
        else:
            # The number of atmospheric realizations can be large.  Use
            # sub-directories under cachedir.
            subdir = str(int((obsid % 1000) // 100))
            subsubdir = str(int((obsid % 100) // 10))
            subsubsubdir = str(obsid % 10)
            cachedir = os.path.join(self.cache_dir, subdir, subsubdir, subsubsubdir)
            if (comm is None) or (comm.rank == 0):
                # Handle a rare race condition when two process groups
                # are creating the cache directories at the same time
                while True:
                    try:
                        os.makedirs(cachedir, exist_ok=True)
                    except OSError:
                        continue
                    except FileNotFoundError:
                        continue
                    else:
                        break
        return cachedir

    @function_timer
    def _get_scan_range(self, obs, comm, prefix):
        if self.field_of_view is not None:
            fov = self.field_of_view
        else:
            fov = obs.telescope.focalplane.field_of_view
        fp_radius = 0.5 * fov.to_value(u.radian)

        # work in parallel

        if comm is None:
            rank = 0
            ntask = 1
        else:
            rank = comm.rank
            ntask = comm.size

        # Create a fake focalplane of detectors in a circle around the boresight

        xaxis, yaxis, zaxis = np.eye(3)
        ndet = 64
        phidet = np.linspace(0, 2 * np.pi, ndet, endpoint=False)
        detquats = []
        thetarot = qa.rotation(yaxis, fp_radius)
        for phi in phidet:
            phirot = qa.rotation(zaxis, phi)
            detquat = qa.mult(phirot, thetarot)
            detquats.append(detquat)

        # Get fake detector pointing

        az = []
        el = []
        quats = obs.shared[self.detector_pointing.boresight][rank::ntask].copy()
        for detquat in detquats:
            vecs = qa.rotate(qa.mult(quats, detquat), zaxis)
            theta, phi = hp.vec2ang(vecs)
            az.append(2 * np.pi - phi)
            el.append(np.pi / 2 - theta)
        az = np.unwrap(np.hstack(az))
        el = np.hstack(el)

        # find the extremes

        azmin = np.amin(az)
        azmax = np.amax(az)
        elmin = np.amin(el)
        elmax = np.amax(el)

        if azmin < -2 * np.pi:
            azmin += 2 * np.pi
            azmax += 2 * np.pi
        elif azmax > 2 * np.pi:
            azmin -= 2 * np.pi
            azmax -= 2 * np.pi

        # Combine results

        if comm is not None:
            azmin = comm.allreduce(azmin, op=MPI.MIN)
            azmax = comm.allreduce(azmax, op=MPI.MAX)
            elmin = comm.allreduce(elmin, op=MPI.MIN)
            elmax = comm.allreduce(elmax, op=MPI.MAX)

        return azmin * u.radian, azmax * u.radian, elmin * u.radian, elmax * u.radian

    @function_timer
    def _get_time_range(self, tmin, istart, times, tmax_tot, obs, weather):
        # FIXME:  This whole function assumes that the data is distributed in the
        # detector direction, and that each process therefore has the full time
        # range.  We should instead collect the full timestamps and turnaround
        # intervals onto a single process and do this calculation there before
        # broadcasting the result.

        while times[istart] < tmin:
            istart += 1

        # Translate the wind speed to time span of a correlated interval
        wx = weather.west_wind.to_value(u.meter / u.second)
        wy = weather.south_wind.to_value(u.meter / u.second)
        w = np.sqrt(wx**2 + wy**2)
        wind_time = self.wind_dist.to_value(u.meter) / w

        tmax = tmin + wind_time
        if tmax < tmax_tot:
            istop = istart
            while istop < len(times) and times[istop] < tmax:
                istop += 1
            # Extend the scan to the next turnaround, if we have them
            if self.turnaround_interval is not None:
                iturn = 0
                while iturn < len(obs.intervals[self.turnaround_interval]) - 1 and (
                    times[istop] > obs.intervals[self.turnaround_interval][iturn].stop
                ):
                    iturn += 1
                if times[istop] > obs.intervals[self.turnaround_interval][iturn].stop:
                    # We are past the last turnaround.
                    # Extend to the end of the observation.
                    istop = len(times)
                    tmax = tmax_tot
                else:
                    # Stop time is either before or in the middle of the turnaround.
                    # Extend to the start of the turnaround.
                    while istop < len(times) and (
                        times[istop]
                        < obs.intervals[self.turnaround_interval][iturn].start
                    ):
                        istop += 1
                    if istop < len(times):
                        tmax = times[istop]
                    else:
                        tmax = tmax_tot
            else:
                tmax = times[istop]
        else:
            tmax = tmax_tot
            istop = len(times)
        return istart, istop, np.ceil(tmax)

    @function_timer
    def _simulate_atmosphere(
        self,
        weather,
        scan_range,
        tmin,
        tmax,
        comm,
        comm_node,
        comm_node_rank,
        key1,
        key2,
        counter1,
        counter2,
        cachedir,
        prefix,
        tmin_tot,
        tmax_tot,
        xstep,
        ystep,
        zstep,
        rmin,
        rmax,
    ):
        log = Logger.get()
        rank = 0
        tmr = Timer()
        if comm is not None:
            rank = comm.rank
            comm.Barrier()
        tmr.start()

        T0_center = weather.air_temperature
        wx = weather.west_wind
        wy = weather.south_wind
        w_center = np.sqrt(wx**2 + wy**2)
        wdir_center = np.arctan2(wy, wx)

        azmin, azmax, elmin, elmax = scan_range

        sim = AtmSim(
            azmin,
            azmax,
            elmin,
            elmax,
            tmin,
            tmax,
            self.lmin_center,
            self.lmin_sigma,
            self.lmax_center,
            self.lmax_sigma,
            w_center,
            0 * u.meter / u.second,
            wdir_center,
            0 * u.radian,
            self.z0_center,
            self.z0_sigma,
            T0_center,
            0 * u.Kelvin,
            self.zatm,
            self.zmax,
            xstep,
            ystep,
            zstep,
            self.nelem_sim_max,
            comm,
            key1,
            key2,
            counter1,
            counter2,
            cachedir,
            rmin * u.meter,
            rmax * u.meter,
            write_debug=self.debug_spectrum,
            node_comm=comm_node,
            node_rank_comm=comm_node_rank,
        )

        if comm is not None:
            comm.Barrier()
        if rank == 0:
            tmr.stop()
            log.debug(
                f"{prefix}SimAtmosphere: Initialize atmosphere: {tmr.seconds()} seconds"
            )
            tmr.clear()
            tmr.start()

        # Check if the cache already exists.

        use_cache = False
        have_cache = False
        if rank == 0:
            if cachedir is not None:
                # We are saving to cache
                use_cache = True
            fname = None
            if cachedir is not None:
                fname = os.path.join(
                    cachedir, "{}_{}_{}_{}.h5".format(key1, key2, counter1, counter2)
                )
                if os.path.isfile(fname):
                    if self.overwrite_cache:
                        os.remove(fname)
                    else:
                        have_cache = True
            if have_cache:
                log.debug(
                    f"{prefix}Loading the atmosphere for t = {tmin - tmin_tot} from {fname}"
                )
            else:
                log.debug(
                    f"{prefix}Simulating the atmosphere for t = {tmin - tmin_tot}"
                )
        if comm is not None:
            use_cache = comm.bcast(use_cache, root=0)

        err = sim.simulate(use_cache=use_cache)
        if err != 0:
            raise RuntimeError(prefix + "Simulation failed.")

        # Advance the sample counter in case wind_time broke the
        # observation in parts

        counter2 += 100000000

        if comm is not None:
            comm.Barrier()
        if rank == 0:
            op = None
            if have_cache:
                op = "Loaded"
            else:
                op = "Simulated"
            tmr.stop()
            log.debug(
                f"{prefix}SimAtmosphere: {op} atmosphere: {tmr.seconds()} seconds"
            )
            tmr.clear()
            tmr.start()

        return sim, counter2

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [
                self.boresight,
            ],
            "detdata": list(),
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [
                self.det_data,
            ],
        }
        return prov

    @function_timer
    def _plot_snapshots(
        self, sim, prefix, obsname, scan_range, tmin, tmax, comm, rmin, rmax
    ):
        """Create snapshots of the atmosphere"""
        log = Logger.get()
        from ..vis import set_matplotlib_backend

        set_matplotlib_backend()
        import pickle

        import matplotlib.pyplot as plt

        azmin, azmax, elmin, elmax = scan_range
        azmin = azmin.to_value(u.radian)
        azmax = azmax.to_value(u.radian)
        elmin = elmin.to_value(u.radian)
        elmax = elmax.to_value(u.radian)

        # elstep = np.radians(0.01)
        elstep = (elmax - elmin) / 320
        azstep = elstep * np.cos(0.5 * (elmin + elmax))
        azgrid = np.linspace(azmin, azmax, int((azmax - azmin) / azstep) + 1)
        elgrid = np.linspace(elmin, elmax, int((elmax - elmin) / elstep) + 1)
        AZ, EL = np.meshgrid(azgrid, elgrid)
        nn = AZ.size
        az = AZ.ravel()
        el = EL.ravel()
        atmdata = np.zeros(nn, dtype=np.float64)
        atmtimes = np.zeros(nn, dtype=np.float64)

        rank = 0
        ntask = 1
        if comm is not None:
            rank = comm.rank
            ntask = comm.size

        r = 0
        t = 0
        my_snapshots = []
        vmin = 1e30
        vmax = -1e30
        tstep = 1
        for i, t in enumerate(np.arange(tmin, tmax, tstep)):
            if i % ntask != rank:
                continue
            err = sim.observe(atmtimes + t, az, el, atmdata, r)
            if err != 0:
                raise RuntimeError(prefix + "Observation failed")
            atmdata *= self.gain
            vmin = min(vmin, np.amin(atmdata))
            vmax = max(vmax, np.amax(atmdata))
            atmdata2d = atmdata.reshape(AZ.shape)
            my_snapshots.append((t, r, atmdata2d.copy()))

        if self.debug_snapshots:
            outdir = "snapshots"
            if rank == 0:
                try:
                    os.makedirs(outdir)
                except FileExistsError:
                    pass
            fn = os.path.join(
                outdir,
                "atm_{}_{}_t_{}_{}_r_{}_{}.pck".format(
                    obsname, rank, int(tmin), int(tmax), int(rmin), int(rmax)
                ),
            )
            with open(fn, "wb") as fout:
                pickle.dump([azgrid, elgrid, my_snapshots], fout)

        log.debug("Snapshots saved in {}".format(fn))

        if self.debug_plots:
            if comm is not None:
                vmin = comm.allreduce(vmin, op=MPI.MIN)
                vmax = comm.allreduce(vmax, op=MPI.MAX)

            for t, r, atmdata2d in my_snapshots:
                plt.figure(figsize=[12, 4])
                plt.imshow(
                    atmdata2d,
                    interpolation="nearest",
                    origin="lower",
                    extent=np.degrees(
                        [
                            0,
                            (azmax - azmin) * np.cos(0.5 * (elmin + elmax)),
                            elmin,
                            elmax,
                        ]
                    ),
                    cmap=plt.get_cmap("Blues"),
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar()
                ax = plt.gca()
                ax.set_title("t = {:15.1f} s, r = {:15.1f} m".format(t, r))
                ax.set_xlabel("az [deg]")
                ax.set_ylabel("el [deg]")
                ax.set_yticks(np.degrees([elmin, elmax]))
                plt.savefig(
                    "atm_{}_t_{:04}_r_{:04}.png".format(obsname, int(t), int(r))
                )
                plt.close()

        del my_snapshots

        return
