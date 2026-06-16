# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
import warnings
from time import time

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import QTable
from scipy.signal import fftconvolve, firwin

from .. import qarray as qa
from ..data import Data
from ..instrument import Focalplane, Telescope
from ..intervals import IntervalList
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..noise import Noise
from ..observation import Observation
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, List, Quantity, Unicode, trait_docs
from ..utils import Logger, dtype_to_aligned, name_UID
from .operator import Operator


@trait_docs
class PairDifference(Operator):
    """Form pair sums and differences of the TOD"""

    allowed_modes = ("", "I", "QU", "IQU")

    API = Int(0, help="Internal interface version for this operator")

    fpkeys = List(
        ["pixel"],
        allow_none=False,
        help="List of focalplane keys that must match for detectors to "
        "be differenced.  If more than two detectors match, an error "
        "is raised",
    )

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to.  Use ';' if multiple "
        "signal flavors should be demodulated.",
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for detector sample flagging"
    )

    pairdiff_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for pairdiff flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    noise_model = Unicode(
        "noise_model",
        allow_none=True,
        help="Observation key containing the noise model",
    )

    keep_dets_frac = Float(
        0.1,
        help="If less than this fraction of detectors are good, cut the observation",
    )

    purge = Bool(False, help="Remove inputs after pairdifferencing")

    in_place = Bool(False, help="Modify the data object in-place.  Implies purge=True.")

    mode = Unicode("IQU", help="Return I, QU or IQU timestreams.")

    # Intervals?

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("shared_flag_mask")
    def _check_shared_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        mode = proposal["value"]
        if mode not in self.allowed_modes:
            msg = f"mode must be one of {self.allowed_modes}"
            raise traitlets.TraitError(msg)
        return mode

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _get_detector_pairs(self, obs):
        log = Logger.get()
        
        fp = obs.telescope.focalplane
        for fpkey in self.fpkeys:
            if fpkey not in fp.properties:
                msg = f"{fpkey} is not a property of the focalplane in {obs.name}"
                raise RuntimeError(msg)

        # Find detectors that have matching keys

        detector_sets = {}
        for det in fp.detectors:
            full_key = None
            for fpkey in self.fpkeys:
                value = str(fp[det][fpkey])
                if full_key is None:
                    full_key = f"{fpkey}={value}"
                else:
                    full_key += f";{fpkey}={value}"
            if full_key not in detector_sets:
                detector_sets[full_key] = [det]
            elif len(detector_sets[full_key]) == 1:
                detector_sets[full_key].append(det)
            else:
                msg = f"Too many detectors matching {full_key}"
                raise RuntimeError(msg)
                
        # Discard sets that have only one detector.  Raise error if
        # there are more than two in a set

        detector_pairs = {}
        det2pair = {}
        nkeep = 0
        ndrop = 0
        for full_key, detector_set in detector_sets.items():
            ndet = len(detector_set)
            if ndet == 1:
                ndrop += 1
                continue
            elif ndet == 2:
                det1, det2 = sorted(detector_set)
                det_pair = "{det1}_{det2}"
                detector_pairs[det_pair] = [det1, det2]
                det2pair[det1] = det_pair
                det2pair[det2] = det_pair
                nkeep += 1
            else:
                msg = f"{full_key} matched {ndet} detectors"
                raise RuntimeError(msg)
        log.debug_rank(
            f"Obs = {obs.name} : kept {nkeep} detector pairs, dropped {ndrop} "
            f"pairless detectors",
            comm=obs.comm.comm_group,
        )

        return detector_pairs, det2pair

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.in_place:
            self.pairdiff_data = None
        else:
            self.pairdiff_data = Data()

        # Each detector pair one or two pseudo detectors

        self.prefixes = []
        if "I" in self.mode:
            self.prefixes.append("pairsum")
        if "QU" in self.mode:
            self.prefixes.append("pairdiff")
        if len(self.prefixes) == 0:
            raise RuntimeError("There are no pseudo detectors to create")

        timer = Timer()
        timer.start()

        # The list of pair-differenced observations.  This will either be placed in a new
        # data object or the list will be swapped into the existing data object.
        pairdiff_obs = []

        for obs in data.obs:
            # Find the detector pairs
            detector_pairs, det2pair = self._get_detector_pairs(obs)

            # Redistribute so pairs are always on a single task
            temp_ob = obs.duplicate(times=self.times)
            temp_ob.redistribute(
                obs.comm.group_size,
                times=self.times,
                override_detector_sets=list(detector_pairs.values()),
            )

            # Find the subset of pairs that are operational

            local_dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            my_dets = []
            my_pairs = []  # list of detector pairs
            local_pairs = {}  # dictionary of detector pairs
            for det_pair, (det1, det2) in detector_pairs.items():
                if det1 in local_dets and det2 in local_dets:
                    my_pairs += det_pair
                    my_dets += [det1, det2]
                    local_pairs[det_pair] = [det1, det2]

            if obs.comm.comm_group is None:
                all_pairs = my_pairs
                all_dets = my_dets
            else:
                proc_pairs = obs.comm.comm_group.gather(my_pairs, root=0)
                proc_dets = obs.comm.comm_group.gather(my_dets, root=0)
                all_dets = []
                all_pairs = []
                if obs.comm.comm_group.rank == 0:
                    for ppairs in proc_pairs:
                        all_pairs += ppairs
                    for pdets in proc_dets:
                        all_dets += pdets
                all_pairs = obs.comm.comm_group.bcast(all_pairs, root=0)
                all_dets = obs.comm.comm_group.bcast(all_dets, root=0)

            all_pairs = set(all_pairs)
            all_dets = set(all_dets)
            dropped_pairs = []
            for det_pair in detector_pairs:
                if det_pair not in all_pairs:
                    dropped_pairs.append(det_pair)
            for dropped_pair in dropped_pairs:
                del detector_pairs[dropped_pair]
            
            # Create a new observation to hold the demodulated and downsampled data

            pairdiff_telescope = self._pairdiff_telescope(temp_ob, detector_pairs)

            pairdiff_name = f"pairdiff_{obs.name}"
            pairdiff_ob = Observation(
                temp_ob.comm,
                pairdiff_telescope,
                temp_ob.dist.samples,
                name=pairdiff_name,
                uid=name_UID(pairdiff_name),
                session=temp_ob.session,
                detector_sets=temp_ob.dist.detector_sets,
                process_rows=temp_ob.dist.process_rows,
                sample_sets=temp_ob.dist.sample_sets,
            )

            # Allocate storage

            pairdiff_dets = []
            for det_pair, (det1, det2) in detector_pairs.items():
                for prefix in self.prefixes:
                    pairdiff_dets.append(f"{prefix}_{det_pair}")

            self._pairdiff_shared_data(temp_ob, pairdiff_ob)

            for det_data in self.det_data:
                exists_data = pairdiff_ob.detdata.ensure(
                    det_data,
                    detectors=demod_dets,
                    dtype=np.float64,
                    create_units=temp_ob.detdata[det_data].units,
                )
            exists_flags = pairdiff_ob.detdata.ensure(
                self.det_flags, detectors=pairdiff_dets, dtype=np.uint8
            )

            self._pairdiff_flags(temp_ob, pairdiff_ob, local_pairs)
            self._pairdiff_signal(data, temp_ob, pairdiff_ob, local_pairs)
            self._pairdiff_noise(temp_ob, pairdiff_ob, local_pairs)

            pairdiff_ob.intervals = temp_ob.intervals
            self._pairdiff_metadata(temp_ob, pairdiff_ob)

            pairdiff_obs.append(pairdiff_ob)

            if self.in_place or self.purge:
                # Input observations are not saved
                obs.clear()
            del temp_ob

            log.debug_rank(
                f"Pairdifferenced observation {obs.name} in",
                comm=data.comm.comm_group,
                timer=timer,
            )
        if self.in_place:
            data.obs.clear()
            data.obs = pairdiff_obs
        else:
            self.pairdiff_data.obs = pairdiff_obs

    @function_timer
    def _pairdiff_telescope(self, obs, detector_pairs):
        fp = obs.telescope.focalplane
        field_names = fp.properties
        # Initialize fields to empty lists
        fields = {name: list() for name in field_names}
        for det_pair, (det1, det2) in detector_pairs.items():
            for field_name in field_names:
                # Each detector pair translates into one or two entries
                for prefix in self.prefixes:
                    if field_name == "name":
                        fields[field_name].append(f"{prefix}_{det_pair}")
                    else:
                        fields[field_name].append(fp[det1][field_name])
        pairdiff_det_data = QTable(
            [fields[field_name] for field_name in field_names], names=field_names
        )
        my_all = list()
        for name in pairdiff_det_data["name"]:
            my_all.append(name)

        pairdiff_focalplane = Focalplane(
            detector_data=pairdiff_det_data,
            field_of_view=fp.field_of_view,
            sample_rate=fp.sample_rate,
        )
        pairdiff_name = f"pairdiff_{obs.telescope.name}"
        pairdiff_telescope = Telescope(
            name=pairdiff_name,
            uid=name_UID(pairdiff_name),
            focalplane=pairdiff_focalplane,
            site=obs.telescope.site,
        )
        return pairdiff_telescope

    @function_timer
    def _pairdiff_shared_data(self, obs, pairdiff_obs):
        """Copy shared data"""
        for field in obs.shared.keys():
            shobj = obs.shared[field]
            commtype = obs.shared.comm_type(field)
            if commtype == "group":
                # Using full group communicator, just copy to new obs.
                pairdiff_obs.shared.assign_mpishared(field, shobj, commtype)
            elif commtype == "row":
                # Shared in the sample direction (per-detector object like a beam,
                # bandpass, etc).  This means that downsampling does not effect the
                # shared object.  Just copy to the new obs.
                pairdiff_obs.shared.assign_mpishared(field, shobj, commtype)
            elif commtype == "column":
                # Shared in the detector direction.
                # Set the data on one process
                if obs.comm_col_rank == 0:
                    values = np.ascontiguousarray(obs.shared[field].data)
                    n_samp = len(values)
                else:
                    n_samp = None
                    values = None

                # Data type
                dtype = shobj.dtype
                if obs.comm_col is not None:
                    n_samp = obs.comm_col.bcast(n_samp, root=0)
                shp = [n_samp]
                shp.extend(shobj.shape[1:])
                shp = tuple(shp)

                # Create the object and set
                pairdiff_obs.shared.create_column(
                    field,
                    shape=shp,
                    dtype=dtype,
                )
                pairdiff_obs.shared[field].set(values, fromrank=0)
            else:
                msg = "Only shared objects using the group, row, and column "
                msg += "communicators can be pairdifferenced"
                raise RuntimeError(msg)
        return

    @function_timer
    def _pairdiff_metadata(self, obs, pairdiff_obs):
        """Copy over metadata"""

        # Metadata dictionary

        for key, value in obs.items():
            if key in pairdiff_obs:
                # Already pairdifferenced
                continue
            demod_obs[key] = value

        # Other observation attributes

        for key, value in vars(obs).items():
            if key.startswith("_"):
                continue
            if hasattr(pairdiff_obs, key):
                # Already pairdifferenced
                continue
            setattr(demod_obs, key, value)

        return

    @function_timer
    def _pairdiff_signal(self, data, obs, pairdiff_obs, detector_pairs):
        """Pairdifference signal TOD"""

        for det_pair, (det1, det2) in detector_pairs.items():
            for flavor in self.det_data:
                signal1 = obs[flavor][det1]
                signal2 = obs[flavor][det2]
                for prefix in self.prefixes:
                    pairdiff_det = f"{prefix}_{det_pair}"
                    if prefix == "pairsum":
                        signal = 0.5 * (signal1 + signal2)
                    elif prefix == "pairdiff":
                        signal = 0.5 * (signal1 - signal2)
                    else:
                        msg = "Unknown pairdiff prefix: {prefix}"
                        raise RuntimeError(msg)
                    pairdiff_obs[flavor][pairdiff_det] = signal

        return

    @function_timer
    def _pairdiff_flags(self, obs, pairdiff_obs, detector_pairs):
        """Pairdiff flags"""

        shared_flags = obs.shared[self.shared_flags].data
        pairdiff_obs.shared[self.shared_flags].set(
            shared_flags, offset=(0,), fromrank=0
        )

        input_det_flags = obs.local_detector_flags
        output_det_flags = dict()

        for det_pair, (det1, det2) in detector_pairs.items():
            # per-sample flags
            flags1 = obs.detdata[self.det_flags][det1]
            flags2 = obs.detdata[self.det_flags][det2]
            flags = flags1 | flags2
            # per detector flags
            det_flag = input_det_flags[det1] | input_det_flags[det2]
            for prefix in self.prefixes:
                pairdiff_det = f"{prefix}_{det_pair}"
                pairdiff_obs.detdata[self.det_flags][pairdiff_det] = flags
                output_det_flags[pairdiff_det] = det_flag
        pairdiff_obs.update_local_detector_flags(output_det_flags)

        return

    @function_timer
    def _pairdiff_noise(self, obs, pairdiff_obs, detector_pairs):
        """Add Noise objects for the new detectors"""
        if self.noise_model is None:
            return

        noise = obs[self.noise_model]

        pairdiff_detectors = []
        pairdiff_freqs = {}
        pairdiff_psds = {}
        pairdiff_indices = {}
        pairdiff_weights = {}

        for det_pair, (det1, det2) in detector_pairs.items():
            weight1 = noise.weight(det1)
            weight2 = noise.weight(det2)
            weight = (weight1 + weight2) / 2
            # index  - ignored
            rate = noise.rate(det1)
            freq = noise.freq(det1)
            psd1 = noise.psd(det1)
            psd2 = noise.psd(det2)
            psd = (psd1 + psd2) / 2
            n_mode = len(self.prefixes)
            for indexoff, prefix in enumerate(self.prefixes):
                pairdiff_det = f"{prefix}_{det_pair}"
                pairdiff_detectors.append(pairdiff_det)
                pairdiff_freqs[pairdiff_det] = freq
                pairdiff_psds[pairdiff_det] = psd
                pairdiff_indices[pairdiff_det] = noise.index(det1) * n_mode + indexoff
                pairdiff_weights[pairdiff_det] = weight
        pairdiff_obs[self.noise_model] = Noise(
            detectors=pairdiff_detectors,
            freqs=pairdiff_freqs,
            psds=pairdiff_psds,
            indices=pairdiff_indices,
            detweights=pairdiff_weights,
        )
        return

    def _finalize(self, data, **kwargs):
        return self.demod_data

    def _requires(self):
        req = {
            "shared": [self.times],
            "detdata": [self.det_data],
        }
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        return dict()


@trait_docs
class StokesWeightsPairDiff(Operator):
    """Compute the Stokes pointing weights for pairdifferenced data"""

    allowed_modes = ("I", "QU", "IQU")

    API = Int(0, help="Internal interface version for this operator")

    mode = Unicode("IQU", help="The Stokes weights to generate")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    weights = Unicode(
        defaults.weights, help="Observation detdata key for output weights"
    )

    single_precision = Bool(False, help="If True, use 32bit float in output")

    detector_pointing_in = Instance(
        klass=Operator,
        allow_none=True,
        help="Pointing operator in the native Q/U frame, typically az/el.  "
        "Must be set if `detector_pointing_out` is set.  Has no effect if "
        " `detector_pointing_out` is not set.",
    )

    detector_pointing_out = Instance(
        klass=Operator,
        allow_none=True,
        help="Pointing operator for the desired frame, typically RA/Dec.  "
        "Requires `detector_pointing_in` to be set.",
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        mode = proposal["value"]
        if mode not in self.allowed_modes:
            msg = f"Invalid mode (must be one of {self.allowed_modes})"
            raise traitlets.TraitError(msg)
        return mode

    @traitlets.validate("detector_pointing_in")
    def _check_detector_pointing_in(self, proposal):
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
                "det_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("detector_pointing_out")
    def _check_detector_pointing_out(self, proposal):
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
                "det_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @function_timer
    def _get_delta(self, data, ob, det):
        """Get the polarization angle in the input and output
        frames to rotate Q and U accordingly

        """
        if self.detector_pointing_out is None:
            return None

        if det.startswith("demod4r") or det.startswith("demod4i"):
            # Get input and output detector pointing
            ob_data = data.select(obs_name=ob.name)
            # detector pointing will short-circuit if detdata already has the required key
            reset = self.detector_pointing_in.quats == self.detector_pointing_out.quats
            if reset and self.detector_pointing_in.quats in ob_data.obs[0].detdata:
                del ob_data.obs[0].detdata[self.detector_pointing_in.quats]
            # Get input pointing
            self.detector_pointing_in.apply(ob_data, detectors=[det])
            quats_in = ob_data.obs[0].detdata[self.detector_pointing_in.quats][det]
            psi_in = qa.to_iso_angles(quats_in)[2]
            if reset and self.detector_pointing_out.quats in ob_data.obs[0].detdata:
                del ob_data.obs[0].detdata[self.detector_pointing_out.quats]
            # Get output pointing
            self.detector_pointing_out.apply(ob_data, detectors=[det])
            quats_out = ob_data.obs[0].detdata[self.detector_pointing_out.quats][det]
            psi_out = qa.to_iso_angles(quats_out)[2]
            if reset:
                # Purge the quaternions to avoid confusion later
                del ob_data.obs[0].detdata[self.detector_pointing_out.quats]
            # Get the difference in position angle
            delta = psi_out - psi_in
            delta = delta[:, np.newaxis]
        else:
            delta = None

        return delta

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        nnz = len(self.mode)

        if self.detector_pointing_in is None and self.detector_pointing_out is not None:
            raise RuntimeError(
                "You must set the input detector pointing with output pointing"
            )

        if self.single_precision:
            dtype = np.float32
        else:
            dtype = np.float64

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)

            exists_weights = obs.detdata.ensure(
                self.weights,
                sample_shape=(nnz,),
                dtype=dtype,
                detectors=dets,
            )

            if len(dets) == 0:
                continue

            nsample = obs.n_local_samples
            ones = np.ones(nsample, dtype=dtype)
            zeros = np.zeros(nsample, dtype=dtype)
            weights = obs.detdata[self.weights]
            if self.mode == "I":
                i_weights = ones
                q_weights = zeros
                u_weights = zeros
                no_weights = zeros
            elif self.mode == "QU":
                i_weights = np.column_stack([zeros, zeros])
                q_weights = np.column_stack([ones, zeros])
                u_weights = np.column_stack([zeros, ones])
                no_weights = np.column_stack([zeros, zeros])
            elif self.mode == "IQU":
                i_weights = np.column_stack([ones, zeros, zeros])
                q_weights = np.column_stack([zeros, ones, zeros])
                u_weights = np.column_stack([zeros, zeros, ones])
                no_weights = np.column_stack([zeros, zeros, zeros])

            for det in dets:
                props = obs.telescope.focalplane[det]
                if "pol_efficiency" in props.colnames:
                    eta = props["pol_efficiency"]
                else:
                    eta = 1.0

                # Check if we need to rotate Q/U weights between reference frames
                delta = self._get_delta(data, obs, det)

                if det.startswith("demod0"):
                    # Stokes I only
                    weights[det] = i_weights
                elif det.startswith("demod4r"):
                    # Stokes Q only
                    if delta is None:
                        weights[det] = q_weights * eta
                    else:
                        # Q' = Qcos(2psi) + Usin(2psi)
                        weights[det] = (
                            np.cos(2 * delta) * q_weights
                            + np.sin(2 * delta) * u_weights
                        ) * eta
                elif det.startswith("demod4i"):
                    # Stokes U only
                    if delta is None:
                        weights[det] = u_weights * eta
                    else:
                        # U' = Ucos(2psi) - Qsin(2psi)
                        weights[det] = (
                            np.cos(2 * delta) * u_weights
                            - np.sin(2 * delta) * q_weights
                        ) * eta
                else:
                    # Not an I/Q/U pseudo detector
                    weights[det] = no_weights

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": list(),
            "detdata": list(),
        }
        return req

    def _provides(self):
        return {"detdata": self.weights}
