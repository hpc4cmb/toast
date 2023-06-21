# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..atm import available_atm, available_utils
from ..data import Data
from ..mpi import MPI
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger, Timer
from .operator import Operator
from .pipeline import Pipeline
from .sim_tod_atm_generate import GenerateAtmosphere
from .sim_tod_atm_observe import ObserveAtmosphere

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

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
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

        # The atmosphere sims are created and stored for each observing session.
        # This data key contains a dictionary of sims, keyed on session name.
        atm_sim_key = "atm_sim"

        # Generate (or load) the atmosphere realizations for all sessions
        gen_atm = GenerateAtmosphere(
            times=self.times,
            boresight=self.detector_pointing.boresight,
            wind_intervals=wind_intervals,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            output=atm_sim_key,
            turnaround_interval=self.turnaround_interval,
            realization=self.realization,
            component=self.component,
            lmin_center=self.lmin_center,
            lmin_sigma=self.lmin_sigma,
            lmax_center=self.lmax_center,
            lmax_sigma=self.lmax_sigma,
            gain=self.gain,
            zatm=self.zatm,
            zmax=self.zmax,
            xstep=self.xstep,
            ystep=self.ystep,
            zstep=self.zstep,
            z0_center=self.z0_center,
            z0_sigma=self.z0_sigma,
            wind_dist=self.wind_dist,
            fade_time=self.fade_time,
            sample_rate=self.sample_rate,
            nelem_sim_max=self.nelem_sim_max,
            cache_dir=self.cache_dir,
            overwrite_cache=self.overwrite_cache,
            cache_only=self.cache_only,
            debug_spectrum=self.debug_spectrum,
            debug_snapshots=self.debug_snapshots,
            debug_plots=self.debug_plots,
            field_of_view=self.field_of_view,
        )
        gen_atm.apply(data)

        if self.cache_only:
            # In this case, the simulated slabs were written to disk but never stored
            # in the output data key.
            return

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
            det_data_units=self.det_data_units,
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

            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            tmr = Timer()
            tmr.start()

            # Prefix for logging
            log_prefix = f"{group} : {ob.name} : "

            # Make sure detector data output exists
            exists = ob.detdata.ensure(
                self.det_data,
                detectors=dets,
                create_units=self.det_data_units,
            )

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

            # Compute the absorption and loading for this observation
            self._common_absorption_and_loading(
                ob, dets, absorption_key, loading_key, comm
            )

            # Create temporary intervals by combining views
            if temporary_view != wind_intervals:
                ob.intervals[temporary_view] = (
                    ob.intervals[view] & ob.intervals[wind_intervals]
                )

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

            # Delete the absorption and loading for this observation
            if absorption_key is not None:
                del ob[absorption_key]
            if loading_key is not None:
                del ob[loading_key]

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

        # Delete the atmosphere slabs for all sessions
        for sname in list(data[atm_sim_key].keys()):
            for wind_slabs in data[atm_sim_key][sname]:
                for slab in wind_slabs:
                    slab.close()
            del data[atm_sim_key][sname]
        del data[atm_sim_key]

    @function_timer
    def _common_absorption_and_loading(
        self, obs, dets, absorption_key, loading_key, comm
    ):
        """Compute the (common) absorption and loading prior to bandpass convolution."""

        if absorption_key is None and loading_key is None:
            return

        if obs.telescope.focalplane.bandpass is None:
            raise RuntimeError("Focalplane does not define bandpass")
        altitude = obs.telescope.site.earthloc.height
        weather = obs.telescope.site.weather
        bandpass = obs.telescope.focalplane.bandpass

        if absorption_key is None and loading_key is None:
            # Nothing to do
            return

        generate_absorption = False
        if absorption_key is not None:
            if absorption_key in obs:
                for det in dets:
                    if det not in obs[absorption_key]:
                        generate_absorption = True
                        break
            else:
                generate_absorption = True

        generate_loading = False
        if loading_key is not None:
            if loading_key in obs:
                for det in dets:
                    if det not in obs[loading_key]:
                        generate_loading = True
                        break
            else:
                generate_loading = True

        if (not generate_loading) and (not generate_absorption):
            # Nothing to do for these detectors
            return

        if generate_loading:
            if loading_key in obs:
                # Delete stale data
                del obs[loading_key]
            obs[loading_key] = dict()

        if generate_absorption:
            if absorption_key in obs:
                # Delete stale data
                del obs[absorption_key]
            obs[absorption_key] = dict()

        # The focalplane likely has groups of detectors whose bandpass spans
        # the same frequency range.  First we build this grouping.

        freq_groups = dict()
        for det in dets:
            dfmin, dfmax = bandpass.get_range(det=det)
            fkey = f"{dfmin} {dfmax}"
            if fkey not in freq_groups:
                freq_groups[fkey] = list()
            freq_groups[fkey].append(det)

        # Work on each frequency group of detectors.  Collectively use the
        # processes in the group to do the calculation.

        for fkey, fdets in freq_groups.items():
            freq_min, freq_max = bandpass.get_range(det=fdets[0])
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

            absorption = list()
            loading = list()

            if my_n_freq > 0:
                if generate_absorption:
                    absorption = atm_absorption_coefficient_vec(
                        altitude.to_value(u.meter),
                        weather.air_temperature.to_value(u.Kelvin),
                        weather.surface_pressure.to_value(u.Pa),
                        weather.pwv.to_value(u.mm),
                        freqs[my_start].to_value(u.GHz),
                        freqs[my_stop - 1].to_value(u.GHz),
                        my_n_freq,
                    )
                if generate_loading:
                    loading = atm_atmospheric_loading_vec(
                        altitude.to_value(u.meter),
                        weather.air_temperature.to_value(u.Kelvin),
                        weather.surface_pressure.to_value(u.Pa),
                        weather.pwv.to_value(u.mm),
                        freqs[my_start].to_value(u.GHz),
                        freqs[my_stop - 1].to_value(u.GHz),
                        my_n_freq,
                    )

            if comm is not None:
                if generate_absorption:
                    absorption = np.hstack(comm.allgather(absorption))
                if generate_loading:
                    loading = np.hstack(comm.allgather(loading))
            if generate_absorption:
                obs[absorption_key][fkey] = absorption
            if generate_loading:
                obs[loading_key][fkey] = loading

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [
                self.boresight,
            ],
            "detdata": [self.detdata],
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
