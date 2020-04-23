# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from ..utils import Logger, memreport

from ..timing import function_timer, Timer

from ..op import Operator

from .atm import available, available_utils, available_mpi

if available_utils:
    from .atm import (
        atm_absorption_coefficient,
        atm_absorption_coefficient_vec,
        atm_atmospheric_loading,
        atm_atmospheric_loading_vec,
    )

if available:
    from .atm import AtmSim

if available_mpi:
    from .atm import AtmSimMPI

from toast.mpi import MPI

import toast.qarray as qa


class OpSimAtmosphere(Operator):
    """Operator which generates atmosphere timestreams.

    All processes collectively generate the atmospheric realization.
    Then each process passes through its local data and observes the
    atmosphere.

    This operator is only compatible with TOD objects that can return
    AZ/EL pointing.

    Args:
        out (str): accumulate data to the cache with name
            <out>_<detector>.  If the named cache objects do not exist,
            then they are created.
        realization (int): if simulating multiple realizations, the
            realization index.
        component (int): the component index to use for this noise
            simulation.
        lmin_center (float): Kolmogorov turbulence dissipation scale
            center.
        lmin_sigma (float): Kolmogorov turbulence dissipation scale
            sigma.
        lmax_center (float): Kolmogorov turbulence injection scale
             center.
        lmax_sigma (float): Kolmogorov turbulence injection scale sigma.
        gain (float): Scaling applied to the simulated TOD.
        zatm (float): atmosphere extent for temperature profile.
        zmax (float): atmosphere extent for water vapor integration.
        xstep (float): size of volume elements in X direction.
        ystep (float): size of volume elements in Y direction.
        zstep (float): size of volume elements in Z direction.
        nelem_sim_max (int): controls the size of the simulation slices.
        verbosity (int): more information is printed for values > 0.
        z0_center (float):  central value of the water vapor
             distribution.
        z0_sigma (float):  sigma of the water vapor distribution.
        common_flag_name (str):  Cache name of the output common flags.
            If it already exists, it is used.  Otherwise flags
            are read from the tod object and stored in the cache under
            common_flag_name.
        common_flag_mask (byte):  Bitmask to use when flagging data
           based on the common flags.
        flag_name (str):  Cache name of the output detector flags will
            be <flag_name>_<detector>.  If the object exists, it is
            used.  Otherwise flags are read from the tod object.
        flag_mask (byte):  Bitmask to use when flagging data
           based on the detector flags.
        apply_flags (bool):  When True, flagged samples are not
             simulated.
        report_timing (bool):  Print out time taken to initialize,
             simulate and observe
        wind_dist (float):  Maximum wind drift before discarding the
            volume and creating a new one [meters].
        cachedir (str):  Directory to use for loading and saving
            atmosphere realizations.  Set to None to disable caching.
        flush (bool):  Flush all print statements
        freq (float):  Observing frequency in GHz.
    """

    def __init__(
        self,
        out="atm",
        realization=0,
        component=123456,
        lmin_center=0.01,
        lmin_sigma=0.001,
        lmax_center=10,
        lmax_sigma=10,
        zatm=40000.0,
        zmax=2000.0,
        xstep=100.0,
        ystep=100.0,
        zstep=100.0,
        nelem_sim_max=10000,
        verbosity=0,
        gain=1,
        z0_center=2000,
        z0_sigma=0,
        apply_flags=False,
        common_flag_name=None,
        common_flag_mask=255,
        flag_name=None,
        flag_mask=255,
        report_timing=True,
        wind_dist=10000,
        cachedir=".",
        flush=False,
        freq=None,
    ):
        if not available:
            msg = (
                "TOAST not compiled with atmosphere simulation support (requires "
                "SuiteSparse)"
            )
            raise RuntimeError(msg)
        # Call the parent class constructor
        super().__init__()

        self._out = out
        self._realization = realization
        self._component = component
        self._lmin_center = lmin_center
        self._lmin_sigma = lmin_sigma
        self._lmax_center = lmax_center
        self._lmax_sigma = lmax_sigma
        self._gain = gain
        self._zatm = zatm
        self._zmax = zmax
        self._xstep = xstep
        self._ystep = ystep
        self._zstep = zstep
        self._nelem_sim_max = nelem_sim_max
        self._verbosity = verbosity
        self._cachedir = cachedir
        self._flush = flush
        self._freq = freq

        self._z0_center = z0_center
        self._z0_sigma = z0_sigma

        self._apply_flags = apply_flags
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._report_timing = report_timing
        self._wind_dist = wind_dist
        self._wind_time = None

    @function_timer
    def exec(self, data):
        """Generate atmosphere timestreams.

        This iterates over all observations and detectors and generates
        the atmosphere timestreams.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            None

        """
        if data.comm.comm_world is not None:
            if not available_mpi:
                msg = (
                    "MPI is used by the data distribution, but TOAST was not built "
                    "with MPI-enabled atmosphere simulation support."
                )
                raise RuntimeError(msg)

        log = Logger.get()
        group = data.comm.group
        for obs in data.obs:
            try:
                obsname = obs["name"]
            except Exception:
                obsname = "observation"
            prefix = "{} : {} : ".format(group, obsname)
            tod = self._get_from_obs("tod", obs)
            comm = tod.mpicomm
            rank = 0
            if comm is not None:
                rank = comm.rank
            site = self._get_from_obs("site_id", obs)
            weather = self._get_from_obs("weather", obs)

            # Get the observation time span and initialize the weather
            # object if one is provided.
            times = tod.local_times()
            tmin = times[0]
            tmax = times[-1]
            tmin_tot = tmin
            tmax_tot = tmax
            if comm is not None:
                tmin_tot = comm.allreduce(tmin, op=MPI.MIN)
                tmax_tot = comm.allreduce(tmax, op=MPI.MAX)
            tmin_tot = np.floor(tmin_tot)
            tmax_tot = np.ceil(tmax_tot)
            weather.set(site, self._realization, tmin_tot)

            key1, key2, counter1, counter2 = self._get_rng_keys(obs)

            absorption = self._get_absorption_and_loading(obs)

            cachedir = self._get_cache_dir(obs, comm)

            if comm is not None:
                comm.Barrier()
            if rank == 0:
                log.info("{}Setting up atmosphere simulation".format(prefix))

            # Cache the output common flags
            common_ref = tod.local_common_flags(self._common_flag_name)

            scan_range = self._get_scan_range(obs, comm, prefix)

            # Loop over the time span in "wind_time"-sized chunks.
            # wind_time is intended to reflect the correlation length
            # in the atmospheric noise.

            tmr = Timer()
            if self._report_timing:
                if comm is not None:
                    comm.Barrier()
                tmr.start()

            tmin = tmin_tot
            istart = 0
            counter1start = counter1
            while tmin < tmax_tot:
                if comm is not None:
                    comm.Barrier()
                if rank == 0:
                    log.info(
                        "{}Instantiating atmosphere for t = {}".format(
                            prefix, tmin - tmin_tot
                        )
                    )

                istart, istop, tmax = self._get_time_range(
                    tmin, istart, times, tmax_tot, common_ref, tod, weather
                )

                ind = slice(istart, istop)
                nind = istop - istart

                # Optimization indicates that we can skip the first r-range:
                # r = [0, 100]
                # We start from r = [100, 1000] and scale the volume element size accordingly
                rmin = 0
                rmax = 100
                scale = 10
                counter2start = counter2
                counter1 = counter1start
                xstart, ystart, zstart = self._xstep, self._ystep, self._zstep
                # self._xstep *= scale ** 0.5
                # self._ystep *= scale ** 0.5
                # self._zstep *= scale ** 0.5

                while rmax < 100000:
                    sim, counter2 = self._simulate_atmosphere(
                        weather,
                        scan_range,
                        tmin,
                        tmax,
                        comm,
                        key1,
                        key2,
                        counter1,
                        counter2start,
                        cachedir,
                        prefix,
                        tmin_tot,
                        tmax_tot,
                        rmin,
                        rmax,
                    )

                    if self._verbosity > 15:
                        self._plot_snapshots(
                            sim,
                            prefix,
                            obsname,
                            scan_range,
                            tmin,
                            tmax,
                            comm,
                            rmin,
                            rmax,
                        )

                    self._observe_atmosphere(
                        sim,
                        tod,
                        comm,
                        prefix,
                        common_ref,
                        istart,
                        nind,
                        ind,
                        scan_range,
                        times,
                        absorption,
                    )

                    del sim

                    rmin = rmax
                    rmax *= scale
                    self._xstep *= np.sqrt(scale)
                    self._ystep *= np.sqrt(scale)
                    self._zstep *= np.sqrt(scale)
                    counter1 += 1

                if self._verbosity > 5:
                    self._save_tod(
                        obsname, tod, times, istart, nind, ind, comm, common_ref
                    )

                self._xstep, self._ystep, self._zstep = xstart, ystart, zstart
                tmin = tmax

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.stop()
                tmr.report("{}Simulated and observed atmosphere".format(prefix))
        return

    @function_timer
    def _save_tod(self, obsname, tod, times, istart, nind, ind, comm, common_ref):
        import pickle

        rank = 0
        if comm is not None:
            rank = comm.rank

        t = times[ind]
        tmin, tmax = t[0], t[-1]
        outdir = "snapshots"
        if rank == 0:
            try:
                os.makedirs(outdir)
            except FileExistsError:
                pass

        try:
            good = common_ref[ind] & tod.UNSTABLE == 0
        except:
            good = slice(0, nind)

        for det in tod.local_dets:
            # Cache the output signal
            cachename = "{}_{}".format(self._out, det)
            ref = tod.cache.reference(cachename)[ind]
            try:
                # Some TOD classes provide a shortcut to Az/El
                az, el = tod.read_azel(detector=det, local_start=istart, n=nind)
            except Exception as e:
                azelquat = tod.read_pntg(
                    detector=det, local_start=istart, n=nind, azel=True
                )
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, phi = qa.to_position(azelquat)
                # Azimuth is measured in the opposite direction
                # than longitude
                az = 2 * np.pi - phi
                el = np.pi / 2 - theta

            fn = os.path.join(
                outdir,
                "atm_tod_{}_{}_t_{}_{}.pck".format(obsname, det, int(tmin), int(tmax)),
            )
            with open(fn, "wb") as fout:
                pickle.dump([det, t[good], az[good], el[good], ref[good]], fout)

        return

    @function_timer
    def _plot_snapshots(
        self, sim, prefix, obsname, scan_range, tmin, tmax, comm, rmin, rmax
    ):
        """ Create snapshots of the atmosphere

        """
        from ..vis import set_backend

        set_backend()
        import matplotlib.pyplot as plt
        import pickle

        azmin, azmax, elmin, elmax = scan_range

        # elstep = np.radians(0.01)
        elstep = (elmax - elmin) / 320
        azstep = elstep * np.cos(0.5 * (elmin + elmax))
        azgrid = np.linspace(azmin, azmax, (azmax - azmin) // azstep + 1)
        elgrid = np.linspace(elmin, elmax, (elmax - elmin) // elstep + 1)
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
            if self._gain:
                atmdata *= self._gain
            vmin = min(vmin, np.amin(atmdata))
            vmax = max(vmax, np.amax(atmdata))
            atmdata2d = atmdata.reshape(AZ.shape)
            my_snapshots.append((t, r, atmdata2d.copy()))

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

        print("Snapshots saved in {}".format(fn), flush=True)

        """
        vmin = comm.allreduce(vmin, op=MPI.MIN)
        vmax = comm.allreduce(vmax, op=MPI.MAX)

        for t, r, atmdata2d in my_snapshots:
            plt.figure(figsize=[12, 4])
            plt.imshow(
                atmdata2d,
                interpolation="nearest",
                origin="lower",
                extent=np.degrees(
                    [0, (azmax - azmin) * np.cos(0.5 * (elmin + elmax)), elmin, elmax]
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
            plt.savefig("atm_{}_t_{:04}_r_{:04}.png".format(obsname, int(t), int(r)))
            plt.close()
        """

        del my_snapshots

        return

    def _get_from_obs(self, name, obs):
        """ Extract value for name from observation.

        If name is not defined in observation, raise an exception.

        """
        if name in obs:
            return obs[name]
        else:
            raise RuntimeError(
                "Error simulating atmosphere: observation "
                'does not define "{}"'.format(name)
            )

    def _get_rng_keys(self, obs):
        """
        The random number generator accepts a key and a counter,
        each made of two 64bit integers.
        Following tod_math.py we set
        key1 = realization * 2^32 + telescope * 2^16 + component
        key2 = obsindx * 2^32
        counter1 = hierarchical cone counter
        counter2 = sample in stream (incremented internally in the atm code)
        """
        telescope = self._get_from_obs("telescope_id", obs)
        site = self._get_from_obs("site_id", obs)
        obsindx = self._get_from_obs("id", obs)
        key1 = self._realization * 2 ** 32 + telescope * 2 ** 16 + self._component
        key2 = site * 2 ** 16 + obsindx
        counter1 = 0
        counter2 = 0
        return key1, key2, counter1, counter2

    @function_timer
    def _get_absorption_and_loading(self, obs):
        altitude = self._get_from_obs("altitude", obs)
        weather = self._get_from_obs("weather", obs)
        tod = self._get_from_obs("tod", obs)
        if self._freq is not None:
            if not available_utils:
                msg = (
                    "TOAST not compiled with libaatm support- absorption and "
                    "loading unavailable"
                )
                raise RuntimeError(msg)
            absorption = atm_absorption_coefficient(
                altitude,
                weather.air_temperature,
                weather.surface_pressure,
                weather.pwv,
                self._freq,
            )
            loading = atm_atmospheric_loading(
                altitude,
                weather.air_temperature,
                weather.surface_pressure,
                weather.pwv,
                self._freq,
            )
            tod.meta["loading"] = loading
        else:
            absorption = None

        return absorption

    def _get_cache_dir(self, obs, comm):
        obsindx = self._get_from_obs("id", obs)
        if self._cachedir is None:
            cachedir = None
        else:
            # The number of atmospheric realizations can be large.  Use
            # sub-directories under cachedir.
            subdir = str(int((obsindx % 1000) // 100))
            subsubdir = str(int((obsindx % 100) // 10))
            subsubsubdir = str(obsindx % 10)
            cachedir = os.path.join(self._cachedir, subdir, subsubdir, subsubsubdir)
            if (comm is None) or (comm.rank == 0):
                # Handle a rare race condition when two process groups
                # are creating the cache directories at the same time
                while True:
                    print("Creating {}".format(cachedir), flush=True)
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
        tod = self._get_from_obs("tod", obs)
        fp_radius = np.radians(self._get_from_obs("fpradius", obs))
        # Read the extent of the AZ/EL boresight pointing, and use that
        # to compute the range of angles needed for simulating the slab.

        (min_az_bore, max_az_bore, min_el_bore, max_el_bore) = tod.scan_range
        # print("boresight scan range = {}, {}, {}, {}".format(
        # min_az_bore, max_az_bore, min_el_bore, max_el_bore))

        # Use a fixed focal plane radius so that changing the actual
        # set of detectors will not affect the simulated atmosphere.

        elfac = 1 / np.cos(max_el_bore + fp_radius)
        azmin = min_az_bore - fp_radius * elfac
        azmax = max_az_bore + fp_radius * elfac
        if azmin < -2 * np.pi:
            azmin += 2 * np.pi
            azmax += 2 * np.pi
        elif azmax > 2 * np.pi:
            azmin -= 2 * np.pi
            azmax -= 2 * np.pi
        elmin = min_el_bore - fp_radius
        elmax = max_el_bore + fp_radius

        if comm is not None:
            azmin = comm.allreduce(azmin, op=MPI.MIN)
            azmax = comm.allreduce(azmax, op=MPI.MAX)
            elmin = comm.allreduce(elmin, op=MPI.MIN)
            elmax = comm.allreduce(elmax, op=MPI.MAX)

        if elmin < 0 or elmax > np.pi / 2:
            raise RuntimeError(
                "{}Error in CES elevation: elmin = {:.3f} deg, elmax = {:.3f} deg, "
                "elmin_bore = {:.3f} deg, elmax_bore = {:.3f} deg, "
                "fp_radius = {:.3f} deg".format(
                    prefix,
                    np.degrees(elmin),
                    np.degrees(elmax),
                    np.degrees(min_el_bore),
                    np.degrees(max_el_bore),
                    np.degrees(fp_radius),
                )
            )

        return azmin, azmax, elmin, elmax

    @function_timer
    def _get_time_range(self, tmin, istart, times, tmax_tot, common_ref, tod, weather):
        while times[istart] < tmin:
            istart += 1

        # Translate the wind speed to time span of a correlated interval
        wx = weather.west_wind
        wy = weather.south_wind
        w = np.sqrt(wx ** 2 + wy ** 2)
        self._wind_time = self._wind_dist / w

        tmax = tmin + self._wind_time
        if tmax < tmax_tot:
            # Extend the scan to the next turnaround
            istop = istart
            while istop < times.size and times[istop] < tmax:
                istop += 1
            while istop < times.size and (common_ref[istop] | tod.TURNAROUND == 0):
                istop += 1
            if istop < times.size:
                tmax = times[istop]
            else:
                tmax = tmax_tot
        else:
            tmax = tmax_tot
            istop = times.size

        return istart, istop, np.ceil(tmax)

    @function_timer
    def _simulate_atmosphere(
        self,
        weather,
        scan_range,
        tmin,
        tmax,
        comm,
        key1,
        key2,
        counter1,
        counter2,
        cachedir,
        prefix,
        tmin_tot,
        tmax_tot,
        rmin,
        rmax,
    ):
        log = Logger.get()
        rank = 0
        if comm is not None:
            rank = comm.rank
        tmr = Timer()
        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            tmr.start()

        T0_center = weather.air_temperature
        wx = weather.west_wind
        wy = weather.south_wind
        w_center = np.sqrt(wx ** 2 + wy ** 2)
        wdir_center = np.arctan2(wy, wx)

        azmin, azmax, elmin, elmax = scan_range

        if cachedir is None:
            # The wrapper requires a string argument
            use_cache = False
            cachedir = ""
        else:
            use_cache = True

        sim = None
        if comm is None:
            sim = AtmSim(
                azmin,
                azmax,
                elmin,
                elmax,
                tmin,
                tmax,
                self._lmin_center,
                self._lmin_sigma,
                self._lmax_center,
                self._lmax_sigma,
                w_center,
                0,
                wdir_center,
                0,
                self._z0_center,
                self._z0_sigma,
                T0_center,
                0,
                self._zatm,
                self._zmax,
                self._xstep,
                self._ystep,
                self._zstep,
                self._nelem_sim_max,
                self._verbosity,
                key1,
                key2,
                counter1,
                counter2,
                cachedir,
                rmin,
                rmax,
            )
        else:
            sim = AtmSimMPI(
                azmin,
                azmax,
                elmin,
                elmax,
                tmin,
                tmax,
                self._lmin_center,
                self._lmin_sigma,
                self._lmax_center,
                self._lmax_sigma,
                w_center,
                0,
                wdir_center,
                0,
                self._z0_center,
                self._z0_sigma,
                T0_center,
                0,
                self._zatm,
                self._zmax,
                self._xstep,
                self._ystep,
                self._zstep,
                self._nelem_sim_max,
                self._verbosity,
                comm,
                key1,
                key2,
                counter1,
                counter2,
                cachedir,
                rmin,
                rmax,
            )

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                tmr.report_clear(
                    "{}OpSimAtmosphere: Initialize atmosphere".format(prefix)
                )

        if rank == 0:
            fname = os.path.join(
                cachedir,
                "{}_{}_{}_{}_metadata.txt".format(key1, key2, counter1, counter2),
            )
            if use_cache and os.path.isfile(fname):
                log.info(
                    "{}Loading the atmosphere for t = {} from {}".format(
                        prefix, tmin - tmin_tot, fname
                    )
                )
                cached = True
            else:
                log.info(
                    "{}Simulating the atmosphere for t = {}".format(
                        prefix, tmin - tmin_tot, fname
                    )
                )
                cached = False

        err = sim.simulate(use_cache)
        if err != 0:
            raise RuntimeError(prefix + "Simulation failed.")

        # Advance the sample counter in case wind_time broke the
        # observation in parts

        counter2 += 100000000

        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            if rank == 0:
                op = None
                if cached:
                    op = "Loaded"
                else:
                    op = "Simulated"
                tmr.report_clear("{}OpSimAtmosphere: {} atmosphere".format(prefix, op))

        return sim, counter2

    @function_timer
    def _observe_atmosphere(
        self,
        sim,
        tod,
        comm,
        prefix,
        common_ref,
        istart,
        nind,
        ind,
        scan_range,
        times,
        absorption,
    ):
        log = Logger.get()
        rank = 0
        if comm is not None:
            rank = comm.rank
        tmr = Timer()
        if self._report_timing:
            if comm is not None:
                comm.Barrier()
            tmr.start()

        azmin, azmax, elmin, elmax = scan_range

        nsamp = tod.local_samples[1]

        if rank == 0:
            log.info("{}Observing the atmosphere".format(prefix))

        ngood_tot = 0
        nbad_tot = 0

        for det in tod.local_dets:
            # Cache the output signal
            cachename = "{}_{}".format(self._out, det)
            if tod.cache.exists(cachename):
                ref = tod.cache.reference(cachename)
            else:
                ref = tod.cache.create(cachename, np.float64, (nsamp,))

            # Cache the output flags
            flag_ref = tod.local_flags(det, self._flag_name)

            if self._apply_flags:
                good = np.logical_and(
                    common_ref[ind] & self._common_flag_mask == 0,
                    flag_ref[ind] & self._flag_mask == 0,
                )
                ngood = np.sum(good)
            else:
                try:
                    good = common_ref[ind] & tod.UNSTABLE == 0
                    ngood = np.sum(good)
                except:
                    good = slice(0, nind)
                    ngood = nind
            if ngood == 0:
                continue

            try:
                # Some TOD classes provide a shortcut to Az/El
                az, el = tod.read_azel(detector=det, local_start=istart, n=nind)
                az = az[good]
                el = el[good]
            except Exception as e:
                azelquat = tod.read_pntg(
                    detector=det, local_start=istart, n=nind, azel=True
                )[good]
                # Convert Az/El quaternion of the detector back into
                # angles for the simulation.
                theta, phi = qa.to_position(azelquat)
                # Azimuth is measured in the opposite direction
                # than longitude
                az = 2 * np.pi - phi
                el = np.pi / 2 - theta

            if np.ptp(az) < np.pi:
                azmin_det = np.amin(az)
                azmax_det = np.amax(az)
            else:
                # Scanning across the zero azimuth.
                azmin_det = np.amin(az[az > np.pi]) - 2 * np.pi
                azmax_det = np.amax(az[az < np.pi])
            elmin_det = np.amin(el)
            elmax_det = np.amax(el)
            if (
                not (azmin <= azmin_det and azmax_det <= azmax)
                and not (
                    azmin <= azmin_det - 2 * np.pi and azmax_det - 2 * np.pi <= azmax
                )
            ) or not (elmin <= elmin_det and elmin_det <= elmax):
                # DEBUG begin
                import pickle

                with open("bad_quats_{}_{}.pck".format(rank, det), "wb") as fout:
                    pickle.dump(
                        [scan_range, az, el, azelquat, tod._boresight_azel], fout
                    )
                # DEBUG end
                raise RuntimeError(
                    prefix + "Detector Az/El: [{:.5f}, {:.5f}], "
                    "[{:.5f}, {:.5f}] is not contained in "
                    "[{:.5f}, {:.5f}], [{:.5f} {:.5f}]"
                    "".format(
                        azmin_det,
                        azmax_det,
                        elmin_det,
                        elmax_det,
                        azmin,
                        azmax,
                        elmin,
                        elmax,
                    )
                )

            # Integrate detector signal

            atmdata = np.zeros(ngood, dtype=np.float64)

            err = sim.observe(times[ind][good], az, el, atmdata, -1.0)
            if err != 0:
                # Observing failed
                bad = np.abs(atmdata) < 1e-30
                nbad = np.sum(bad)
                log.error(
                    "{}OpSimAtmosphere: Observing FAILED for {} ({:.2f} %) samples. "
                    "det = {}, rank = {}".format(
                        prefix, nbad, nbad * 100 / ngood, det, rank
                    )
                )
                atmdata[bad] = 0
                flag_ref[ind][good][bad] = 255
                nbad_tot += nbad
            ngood_tot += ngood

            if self._gain:
                atmdata *= self._gain

            if absorption is not None:
                # Apply the frequency-dependent absorption-coefficient
                atmdata *= absorption

            ref[ind][good] += atmdata

            del ref

        if comm is not None:
            comm.Barrier()
            ngood_tot = comm.reduce(ngood_tot)
            nbad_tot = comm.reduce(nbad_tot)
        if rank == 0 and nbad_tot > 0:
            print(
                "{}WARNING: Observe atmosphere FAILED on {:.2f}% of samples".format(
                    prefix, nbad_tot * 100 / ngood_tot
                )
            )
        if self._report_timing:
            if rank == 0:
                tmr.report_clear("{}OpSimAtmosphere: Observe atmosphere".format(prefix))
        return
