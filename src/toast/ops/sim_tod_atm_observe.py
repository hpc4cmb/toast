# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import scipy.interpolate
import traitlets
from astropy import units as u
from numpy.core.fromnumeric import size

from .. import qarray as qa
from ..atm import available_utils
from ..mpi import MPI
from ..observation import default_values as defaults
from ..observation_dist import global_interval_times
from ..timing import GlobalTimers, function_timer
from ..traits import Bool, Float, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger, Timer, unit_conversion
from .operator import Operator

if available_utils:
    from ..atm import atm_absorption_coefficient_vec, atm_atmospheric_loading_vec


@trait_docs
class ObserveAtmosphere(Operator):
    """Operator which uses detector pointing to observe a simulated atmosphere slab."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for accumulating dipole timestreams",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    quats_azel = Unicode(
        defaults.quats_azel,
        allow_none=True,
        help="Observation detdata key for detector quaternions",
    )

    weights = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for detector Stokes weights",
    )

    weights_mode = Unicode("IQU", help="Stokes weights mode (eg. 'I', 'IQU', 'QU')")

    polarization_fraction = Float(
        0,
        help="Polarization fraction (only Q polarization).",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of valid data in all observations"
    )

    wind_view = Unicode(
        "wind", help="The view of times matching individual simulated atmosphere slabs"
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

    sim = Unicode("atm_sim", help="Data key with the dictionary of sims per session")

    absorption = Unicode(
        None, allow_none=True, help="The observation key for the absorption"
    )

    loading = Unicode(None, allow_none=True, help="The observation key for the loading")

    n_bandpass_freqs = Int(
        100,
        help="The number of sampling frequencies used when convolving the bandpass "
        "with atmosphere absorption and loading",
    )

    sample_rate = Quantity(
        None,
        allow_none=True,
        help="Rate at which to sample atmospheric TOD before interpolation.  "
        "Default is no interpolation.",
    )

    fade_time = Quantity(
        None,
        allow_none=True,
        help="Fade in/out time to avoid a step at wind break.",
    )

    gain = Float(1.0, help="Scaling applied to the simulated TOD")

    debug_tod = Bool(False, help="If True, dump TOD to pickle files")

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        gt = GlobalTimers.get()

        for trait in ("absorption",):
            if getattr(self, trait) is None:
                msg = f"You must set the '{trait}' trait before calling exec()"
                raise RuntimeError(msg)

        gt.start("ObserveAtmosphere:  total")

        comm = data.comm.comm_group
        group = data.comm.group
        rank = data.comm.group_rank

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Get the session name for this observation
            session_name = ob.session.name

            gt.start("ObserveAtmosphere:  per-observation setup")

            # Bandpass-specific unit conversion, relative to 150GHz
            absorption, loading = self._detector_absorption_and_loading(
                ob, self.absorption, self.loading, dets
            )

            # Make sure detector data output exists
            exists = ob.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )

            # Unit conversion from ATM timestream (K) to det data units
            scale = unit_conversion(u.K, ob.detdata[self.det_data].units)

            # Prefix for logging
            log_prefix = f"{group} : {ob.name}"

            # The current wind-driven timespan
            cur_wind = 0

            # Loop over views
            views = ob.view[self.view]

            ngood_tot = 0
            nbad_tot = 0

            gt.stop("ObserveAtmosphere:  per-observation setup")
            for vw in range(len(views)):
                times = views.shared[self.times][vw]

                # Determine the wind interval we are in, and hence which atmosphere
                # simulation to use.  The wind intervals are already guaranteed
                # by the calling code to break on the data view boundaries.
                if len(views) > 1:
                    while (
                        cur_wind < (len(ob.view[self.wind_view]) - 1)
                        and times[0]
                        > ob.view[self.wind_view].shared[self.times][cur_wind][-1]
                    ):
                        cur_wind += 1

                # Get the flags if needed
                sh_flags = None
                if self.shared_flags is not None:
                    sh_flags = (
                        np.array(views.shared[self.shared_flags][vw])
                        & self.shared_flag_mask
                    )

                sim_list = data[self.sim][session_name][cur_wind]

                for det in dets:
                    gt.start("ObserveAtmosphere:  detector setup")
                    flags = None
                    if self.det_flags is not None:
                        flags = (
                            np.array(views.detdata[self.det_flags][vw][det])
                            & self.det_flag_mask
                        )
                        if sh_flags is not None:
                            flags |= sh_flags
                    elif sh_flags is not None:
                        flags = sh_flags

                    good = slice(None, None, None)
                    ngood = len(views.detdata[self.det_data][vw][det])
                    if flags is not None:
                        good = flags == 0
                        ngood = np.sum(good)

                    if ngood == 0:
                        continue
                    ngood_tot += ngood

                    # Detector Az / El quaternions for good samples
                    azel_quat = views.detdata[self.quats_azel][vw][det][good]

                    # Convert Az/El quaternion of the detector back into
                    # angles from the simulation.
                    theta, phi, _ = qa.to_iso_angles(azel_quat)

                    # Stokes weights for observing polarized atmosphere
                    if self.weights is None:
                        weights_I = 1
                        weights_Q = 0
                    else:
                        weights = views.detdata[self.weights][vw][det][good]
                        if "I" in self.weights_mode:
                            ind = self.weights_mode.index("I")
                            weights_I = weights[:, ind].copy()
                        else:
                            weights_I = 0
                        if "Q" in self.weights_mode:
                            ind = self.weights_mode.index("Q")
                            weights_Q = weights[:, ind].copy()
                        else:
                            weights_Q = 0

                    # Azimuth is measured in the opposite direction
                    # than longitude
                    az = 2 * np.pi - phi
                    el = np.pi / 2 - theta

                    az = np.unwrap(az)
                    while np.amin(az) < 0:
                        az += 2 * np.pi
                    while np.amax(az) > 2 * np.pi:
                        az -= 2 * np.pi
                    azmin_det = np.amin(az)
                    azmax_det = np.amax(az)
                    elmin_det = np.amin(el)
                    elmax_det = np.amax(el)

                    tmin_det = times[good][0]
                    tmax_det = times[good][-1]

                    # We may be interpolating some of the time samples

                    if self.sample_rate is None:
                        t_interp = times[good]
                        az_interp = az
                        el_interp = el
                    else:
                        n_interp = int(
                            (tmax_det - tmin_det) * self.sample_rate.to_value(u.Hz)
                        )
                        t_interp = np.linspace(tmin_det, tmax_det, n_interp)
                        # Az is discontinuous if we scan across az=0.  To interpolate,
                        # we must unwrap it first ...
                        az_interp = np.interp(t_interp, times[good], np.unwrap(az))
                        # ... however, the checks later assume 0 < az < 2pi
                        az_interp[az_interp < 0] += 2 * np.pi
                        az_interp[az_interp > 2 * np.pi] -= 2 * np.pi
                        el_interp = np.interp(t_interp, times[good], el)

                    # Integrate detector signal across all slabs at different altitudes

                    atmdata = np.zeros(t_interp.size, dtype=np.float64)

                    gt.stop("ObserveAtmosphere:  detector setup")
                    gt.start("ObserveAtmosphere:  detector AtmSim.observe")
                    for icur, cur_sim in enumerate(sim_list):
                        if cur_sim.tmin > tmin_det or cur_sim.tmax < tmax_det:
                            msg = (
                                f"{log_prefix} : {det} "
                                f"Detector time: [{tmin_det:.1f}, {tmax_det:.1f}], "
                                f"is not contained in [{cur_sim.tmin:.1f}, "
                                f"{cur_sim.tmax:.1f}]"
                            )
                            raise RuntimeError(msg)
                        if (
                            not (
                                cur_sim.azmin <= azmin_det
                                and azmax_det <= cur_sim.azmax
                            )
                            and not (
                                cur_sim.azmin <= azmin_det - 2 * np.pi
                                and azmax_det - 2 * np.pi <= cur_sim.azmax
                            )
                        ) or not (
                            cur_sim.elmin <= elmin_det and elmin_det <= cur_sim.elmax
                        ):
                            msg = (
                                f"{log_prefix} : {det} "
                                f"Detector Az/El: [{azmin_det:.5f}, {azmax_det:.5f}], "
                                f"[{elmin_det:.5f}, {elmax_det:.5f}] is not contained "
                                f"in [{cur_sim.azmin:.5f}, {cur_sim.azmax:.5f}], "
                                f"[{cur_sim.elmin:.5f} {cur_sim.elmax:.5f}]"
                            )
                            raise RuntimeError(msg)

                        err = cur_sim.observe(
                            t_interp, az_interp, el_interp, atmdata, -1.0
                        )

                        # Dump timestream snapshot
                        if self.debug_tod:
                            first = ob.intervals[self.view][vw].first
                            last = ob.intervals[self.view][vw].last
                            self._save_tod(
                                f"post{icur}",
                                ob,
                                self.times,
                                first,
                                last,
                                det,
                                raw=atmdata,
                            )

                        if err != 0:
                            # import pdb
                            # import matplotlib.pyplot as plt
                            # pdb.set_trace()
                            # Observing failed
                            if self.sample_rate is None:
                                full_data = atmdata
                            else:
                                # Interpolate to full sample rate, make sure to flag
                                # samples around a failed time sample
                                test_data = atmdata.copy()
                                for i in [-2, -1, 1, 2]:
                                    test_data *= np.roll(atmdata, i)
                                interp = scipy.interpolate.interp1d(
                                    t_interp,
                                    test_data,
                                    kind="previous",
                                    copy=False,
                                )
                                full_data = interp(times[good])
                            bad = np.abs(full_data) < 1e-30
                            nbad = np.sum(bad)
                            log.error(
                                f"{log_prefix} : {det} "
                                f"ObserveAtmosphere failed for {nbad} "
                                f"({nbad * 100 / ngood:.2f} %) samples.  "
                                f"det = {det}, rank = {rank}"
                            )
                            # If any samples failed the simulation, flag them as bad
                            if nbad > 0:
                                if self.det_flags is None:
                                    log.warning(
                                        "Some samples failed atmosphere simulation, "
                                        "but no det flag field was specified.  "
                                        "Cannot flag samples"
                                    )
                                else:
                                    views.detdata[self.det_flags][vw][det][good][
                                        bad
                                    ] |= self.det_flag_mask
                                    nbad_tot += nbad
                    gt.stop("ObserveAtmosphere:  detector AtmSim.observe")

                    # Optionally, interpolate the atmosphere to full sample rate
                    if self.sample_rate is not None:
                        gt.start("ObserveAtmosphere:  detector interpolate")
                        interp = scipy.interpolate.interp1d(
                            t_interp,
                            atmdata,
                            kind="quadratic",
                            copy=False,
                        )
                        atmdata = interp(times[good])
                        gt.stop("ObserveAtmosphere:  detector interpolate")

                    gt.start("ObserveAtmosphere:  detector accumulate")

                    # Dump timestream snapshot
                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "precal", ob, self.times, first, last, det, raw=atmdata
                        )

                    # Calibrate the atmospheric fluctuations to appropriate bandpass
                    if absorption is not None:
                        atmdata *= self.gain * absorption[det]

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postcal", ob, self.times, first, last, det, raw=atmdata
                        )

                    # If we are simulating disjoint wind views, we need to suppress
                    # a jump between them

                    if len(views) > 1 and self.fade_time is not None:
                        atmdata -= np.mean(atmdata)  # Add thermal loading after this
                        fsample = (times.size - 1) / (times[-1] - times[0])
                        nfade = min(
                            int(self.fade_time.to_value(u.s) * fsample),
                            atmdata.size // 2,
                        )
                        if vw < len(views) - 1:
                            # Fade out the end
                            atmdata[-nfade:] *= np.arange(nfade - 1, -1, -1) / nfade
                        if vw > 0:
                            # Fade out the beginning
                            atmdata[:nfade] *= np.arange(nfade) / nfade

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postfade", ob, self.times, first, last, det, raw=atmdata
                        )

                    # Add polarization.  In our simple model, there is only Q-polarization
                    # and the polarization fraction is constant.
                    pfrac = self.polarization_fraction
                    atmdata *= weights_I + weights_Q * pfrac

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postpol", ob, self.times, first, last, det, raw=atmdata
                        )

                    if loading is not None:
                        # Add the elevation-dependent atmospheric loading
                        atmdata += loading[det] / np.sin(el)

                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "postload", ob, self.times, first, last, det, raw=atmdata
                        )

                    # Add contribution to output
                    views.detdata[self.det_data][vw][det, good] += scale * atmdata
                    gt.stop("ObserveAtmosphere:  detector accumulate")

                    # Dump timestream snapshot
                    if self.debug_tod:
                        first = ob.intervals[self.view][vw].first
                        last = ob.intervals[self.view][vw].last
                        self._save_tod(
                            "final",
                            ob,
                            self.times,
                            first,
                            last,
                            det,
                            detdata=self.det_data,
                        )

            if nbad_tot > 0:
                frac = nbad_tot / (ngood_tot + nbad_tot) * 100
                log.error(
                    f"{log_prefix}: Observe atmosphere FAILED on {frac:.2f}% of samples"
                )
        gt.stop("ObserveAtmosphere:  total")

    @function_timer
    def _detector_absorption_and_loading(self, obs, absorption_key, loading_key, dets):
        """Bandpass-specific unit conversion and loading"""

        if obs.telescope.focalplane.bandpass is None:
            raise RuntimeError("Focalplane does not define bandpass")
        bandpass = obs.telescope.focalplane.bandpass

        n_freq = self.n_bandpass_freqs

        absorption_det = None
        loading_det = None
        if loading_key is not None:
            loading_det = dict()
        if absorption_key is not None:
            absorption_det = dict()
        for det in dets:
            freq_min, freq_max = bandpass.get_range(det=det)
            fkey = f"{freq_min} {freq_max}"
            freqs = np.linspace(freq_min, freq_max, n_freq)
            if absorption_key is not None:
                absorption_det[det] = bandpass.convolve(
                    det, freqs, obs[absorption_key][fkey], rj=True
                )
            if loading_key is not None:
                loading_det[det] = bandpass.convolve(
                    det, freqs, obs[loading_key][fkey], rj=True
                )

        return absorption_det, loading_det

    @function_timer
    def _save_tod(
        self,
        prefix,
        ob,
        times,
        first,
        last,
        det,
        raw=None,
        detdata=None,
    ):
        import pickle

        outdir = "snapshots"
        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass

        timestamps = ob.shared[times].data
        tmin = int(timestamps[first])
        tmax = int(timestamps[last])
        slc = slice(first, last + 1, 1)

        ddata = None
        if raw is not None:
            ddata = raw
        else:
            ddata = ob.detdata[detdata][det, slc]

        fn = os.path.join(
            outdir,
            f"atm_tod_{prefix}_{ob.name}_{det}_t_{tmin}_{tmax}.pck",
        )
        with open(fn, "wb") as fout:
            pickle.dump([det, timestamps[slc], ddata], fout)
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "global": [self.sim],
            "shared": [
                self.times,
            ],
            "detdata": [
                self.det_data,
                self.quats_azel,
            ],
            "intervals": [
                self.wind_view,
            ],
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.weights is not None:
            req["detdata"].append(self.weights)
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
            "intervals": list(),
        }
        return prov
