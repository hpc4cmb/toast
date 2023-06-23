# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import scipy.interpolate
import traitlets
from astropy import units as u
from numpy.core.fromnumeric import size

from .. import qarray as qa
from ..atm import available_atm
from ..mpi import MPI
from ..observation import default_values as defaults
from ..observation_dist import global_interval_times
from ..pointing_utils import scan_range_lonlat
from ..timing import GlobalTimers, function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, Timer
from .operator import Operator

if available_atm:
    from ..atm import AtmSim


@trait_docs
class GenerateAtmosphere(Operator):
    """Operator which simulates or loads atmosphere realizations

    For each observing session, this operator simulates (or loads from disk) the
    atmosphere realization.  The simulated data is stored in the Data container.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    boresight = Unicode(
        defaults.boresight_azel, help="Observation shared key for Az/El boresight"
    )

    wind_intervals = Unicode("wind", help="Intervals to create for wind breaks")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    output = Unicode(
        "atm_sim", help="Data key to store the dictionary of sims per session"
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

    cache_dir = Unicode(
        None,
        allow_none=True,
        help="Directory to use for loading / saving atmosphere realizations",
    )

    overwrite_cache = Bool(
        False, help="If True, redo and overwrite any cached atmospheric realizations."
    )

    cache_only = Bool(False, help="If True, only cache the atmosphere on disk.")

    debug_spectrum = Bool(False, help="If True, dump out Kolmogorov debug files")

    debug_snapshots = Bool(
        False, help="If True, dump snapshots of the atmosphere slabs to pickle files"
    )

    debug_plots = Bool(False, help="If True, make plots of the debug snapshots")

    field_of_view = Quantity(
        None,
        allow_none=True,
        help="Override the focalplane field of view",
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Since each process group has the same set of observations, we use the group
        # communicator for collectively simulating the atmosphere slab for each
        # observation.
        group = data.comm.group

        # The atmosphere sims are created and stored for each observing session.
        # The output data key contains a dictionary of sims, keyed on session name.
        sim_output = dict()

        # Split the observations according to their observing session.  This split
        # also checks that every session has a name.
        data_sessions = data.split(obs_session_name=True, require_full=True)

        # Multiple process groups might have observations from the same session.
        # If we are caching the data, we only want one group to do that per session.
        # First we find the unique sessions for every process group and simulate
        # those.  Then every group will simulate or load the shared sessions.

        group_sessions = [x for x in data_sessions.keys()]
        session_gen = None
        if data.comm.ngroups > 1:
            # We have multiple groups
            if data.comm.group_rank == 0:
                # The first processes in all groups do this
                p_sessions = data.comm.comm_group_rank.gather(group_sessions, root=0)
                if group == 0:
                    # One global process builds the lookup
                    session_gen = dict()
                    for iproc, ps in enumerate(p_sessions):
                        for s in ps:
                            if s not in session_gen:
                                # The first process to encounter the session will
                                # generate it.
                                session_gen[s] = iproc
            session_gen = data.comm.comm_world.bcast(session_gen, root=0)
        else:
            # Only one group
            session_gen = {x: 0 for x in group_sessions}

        # Find which sessions we have that we are NOT generating
        shared_sessions = set([x for x in group_sessions if session_gen[x] != group])

        # First generate all sessions we are responsible for
        for sname, sdata in data_sessions.items():
            if session_gen[sname] == group:
                msg = f"Group {group} Simulate or load session {sname} in first round"
                log.debug_rank(msg, comm=sdata.comm.comm_group)
                sim_output[sname] = self._sim_or_load(sname, sdata)

        # Wait for all groups to finish
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Now simulate or load any shared sessions
        for sname, sdata in data_sessions.items():
            if sname in shared_sessions:
                msg = f"Group {group} Simulate or load shared session {sname} in second round"
                log.debug_rank(msg, comm=sdata.comm.comm_group)
                sim_output[sname] = self._sim_or_load(sname, sdata)

        if not self.cache_only:
            # Add output to data
            data[self.output] = sim_output

    def _sim_or_load(self, sname, sdata):
        """Simulate or load all atmosphere slabs for a single session."""
        log = Logger.get()

        comm = sdata.comm.comm_group
        group = sdata.comm.group
        rank = sdata.comm.group_rank
        comm_node = sdata.comm.comm_group_node
        comm_node_rank = sdata.comm.comm_group_node_rank

        # Prefix for logging
        log_prefix = f"{group} : {sname} : "

        # List of simulated slabs for each wind interval
        output = None
        if not self.cache_only:
            output = list()

        # For each session, check that the observations have the same site.
        site = None
        for ob in sdata.obs:
            if site is None:
                site = ob.telescope.site
            elif ob.telescope.site != site:
                msg = f"Different sites found for observations within the same "
                msg += f"session: {site} != {ob.telescope.site}"
                log.error(msg)
                raise RuntimeError(msg)

        if not hasattr(site, "weather") or site.weather is None:
            raise RuntimeError("Cannot simulate atmosphere for sites without weather")
        weather = site.weather

        # The session is by definition the same for all observations in this split.
        # Use the first observation to compute some quantities common to all.

        first_obs = sdata.obs[0]

        # The full time range of this session.  The processes are arranged
        # in a grid, and processes along the same column have the same timestamps.
        # So the first process of the group has the earliest times and the last
        # process of the group has the latest times.

        times = first_obs.shared[self.times]

        tmin_tot = times[0]
        tmax_tot = times[-1]
        if comm is not None:
            tmin_tot = comm.bcast(tmin_tot, root=0)
            tmax_tot = comm.bcast(tmax_tot, root=(sdata.comm.group_size - 1))

        # RNG values
        key1, key2, counter1, counter2 = self._get_rng_keys(first_obs)

        # Path to the cache location
        cachedir = self._get_cache_dir(first_obs, comm)

        log.debug_rank(f"{log_prefix}Setting up atmosphere simulation", comm=comm)

        # Although each observation in the session has the same boresight pointing,
        # they have independent fields of view / extent.  We find the maximal
        # Az / El range across all observations in the session.

        azmin = None
        azmax = None
        elmin = None
        elmax = None
        for ob in sdata.obs:
            ob_azmin, ob_azmax, ob_elmin, ob_elmax = scan_range_lonlat(
                ob,
                self.boresight,
                self.shared_flags,
                self.shared_flag_mask,
                field_of_view=self.field_of_view,
                is_azimuth=True,
            )
            ob_azmin = ob_azmin.to_value(u.rad)
            ob_azmax = ob_azmax.to_value(u.rad)
            ob_elmin = ob_elmin.to_value(u.rad)
            ob_elmax = ob_elmax.to_value(u.rad)

            if azmin is None:
                azmin = ob_azmin
                azmax = ob_azmax
                elmin = ob_elmin
                elmax = ob_elmax
            else:
                azmin = min(azmin, ob_azmin)
                azmax = max(azmax, ob_azmax)
                elmin = min(elmin, ob_elmin)
                elmax = max(elmax, ob_elmax)
        scan_range = (azmin * u.rad, azmax * u.rad, elmin * u.rad, elmax * u.rad)

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
                tmin, istart, times, tmax_tot, first_obs, weather
            )
            wind_times.append((tmin, tmax))

            if rank == 0:
                log.debug(
                    f"{log_prefix}Instantiating atmosphere for t = "
                    f"{tmin - tmin_tot:10.1f} s - {tmax - tmin_tot:10.1f} s "
                    f"out of {tmax_tot - tmin_tot:10.1f} s"
                )

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
                if not self.cache_only:
                    sim_list.append(sim)

                if self.debug_plots or self.debug_snapshots:
                    self._plot_snapshots(
                        sim,
                        log_prefix,
                        sname,
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

            if not self.cache_only:
                output.append(sim_list)
            tmin = tmax

        if not self.cache_only:
            # Create the wind intervals
            for ob in sdata.obs:
                ob.intervals.create_col(
                    self.wind_intervals, wind_times, ob.shared[self.times]
                )
        return output

    def _get_rng_keys(self, obs):
        """Get random number keys and counters for an observing session.

        The session and telescope UID values are 32bit integers.  The realization
        index is typically smaller, and should fit within 2^16.  The component index
        is a small integer.

        The random number generator accepts a key and a counter, each made of two 64bit
        integers.  For a given observation we set these as:
            key 1 = site UID * 2^32 + telescope UID
            key 2 = session UID * 2^32 + realization * 2^16 + component
            counter 1 = hierarchical cone counter
            counter 2 = 0 (this is incremented per RNG stream sample)

        Args:
            obs (Observation):  One observation in the session.

        Returns:
            (tuple): The key1, key2, counter1, counter2 to use.

        """
        telescope = obs.telescope.uid
        site = obs.telescope.site.uid
        session = obs.session.uid

        # site UID in higher bits, telescope UID in lower bits
        key1 = site * 2**32 + telescope

        # Observation UID in higher bits, realization and component in lower bits
        key2 = session * 2**32 + self.realization * 2**16 + self.component

        # This tracks the number of cones simulated due to the wind speed.
        counter1 = 0

        # Starting point for the observation, incremented for each slice.
        counter2 = 0

        return key1, key2, counter1, counter2

    def _get_cache_dir(self, obs, comm):
        session_id = obs.session.uid
        if self.cache_dir is None:
            cachedir = None
        else:
            # The number of atmospheric realizations can be large.  Use
            # sub-directories under cachedir.
            subdir = str(int((session_id % 1000) // 100))
            subsubdir = str(int((session_id % 100) // 10))
            subsubsubdir = str(session_id % 10)
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
    def _get_time_range(self, tmin, istart, times, tmax_tot, obs, weather):
        # Do this calculation on one process.  Get the times and intervals here.

        turn_ilist = global_interval_times(
            obs.dist, obs.intervals, self.turnaround_interval, join=False
        )

        all_times = times
        if obs.comm_row_size > 1 and obs.comm_col_rank == 0:
            all_times = obs.comm_row.gather(times, root=0)
            if obs.comm_row_rank == 0:
                all_times = np.concatenate(all_times)

        # FIXME:  The code below is explicitly looping over numpy arrays.  If this is
        # too slow, we should move to using numpy functions like searchsorted, etc.

        istop = None
        tmax = None
        if obs.comm.group_rank == 0:
            while all_times[istart] < tmin:
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
                        times[istop]
                        > obs.intervals[self.turnaround_interval][iturn].stop
                    ):
                        iturn += 1
                    if (
                        times[istop]
                        > obs.intervals[self.turnaround_interval][iturn].stop
                    ):
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
                istop = len(all_times)
            tmax = np.ceil(tmax)

        if obs.comm.comm_group is not None:
            istart = obs.comm.comm_group.bcast(istart, root=0)
            istop = obs.comm.comm_group.bcast(istop, root=0)
            tmax = obs.comm.comm_group.bcast(tmax, root=0)

        return istart, istop, tmax

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

        msg = f"{prefix}SimulateAtmosphere:  Initialize atmosphere"
        log.debug_rank(msg, comm=comm, timer=tmr)

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

        op = None
        if have_cache:
            op = "Loaded"
        else:
            op = "Simulated"
        msg = f"{prefix}SimAtmosphere: {op} atmosphere"
        log.debug_rank(msg, comm=comm, timer=tmr)
        return sim, counter2

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
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.turnaround_interval is not None:
            req["intervals"].append(self.turnaround_interval)
        return req

    def _provides(self):
        prov = {
            "global": [self.output],
            "meta": [self.wind_intervals],
            "shared": list(),
            "detdata": list(),
        }
        return prov
