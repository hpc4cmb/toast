# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, use_mpi

import os

import traitlets

import numpy as np

from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned

from ..traits import trait_docs, Int, Unicode, Bool, Dict

from ..timing import function_timer

from .operator import Operator

from .memory_counter import MemoryCounter

from .madam_utils import (
    log_time_memory,
    stage_local,
    stage_in_turns,
    restore_local,
    restore_in_turns,
)


madam = None
if use_mpi:
    try:
        import libmadam_wrapper as madam
    except ImportError:
        madam = None


def available():
    """(bool): True if libmadam is found in the library search path."""
    global madam
    return (madam is not None) and madam.available


@trait_docs
class Madam(Operator):
    """Operator which passes data to libmadam for map-making."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    params = Dict(dict(), help="Parameters to pass to madam")

    paramfile = Unicode(
        None, allow_none=True, help="Read madam parameters from this file"
    )

    times = Unicode("times", help="Observation shared key for timestamps")

    det_data = Unicode("signal", help="Observation detdata key for the timestream data")

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    pixels = Unicode("pixels", help="Observation detdata key for output pixel indices")

    weights = Unicode("weights", help="Observation detdata key for output weights")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    pixels_nested = Bool(True, help="True if pixel indices are in NESTED ordering")

    det_out = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output destriped timestreams",
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    purge_det_data = Bool(
        False,
        help="If True, clear all observation detector data after copying to madam buffers",
    )

    purge_pointing = Bool(
        False,
        help="If True, clear all observation detector pointing data after copying to madam buffers",
    )

    restore_det_data = Bool(
        False, help="If True, restore detector data to observations on completion",
    )

    restore_pointing = Bool(
        False, help="If True, restore detector pointing to observations on completion",
    )

    mcmode = Bool(
        False,
        help="If true, Madam will store auxiliary information such as pixel matrices and noise filter.",
    )

    copy_groups = Int(
        1,
        help="The processes on each node are split into this number of groups to copy data in turns",
    )

    translate_timestamps = Bool(
        False, help="Translate timestamps to enforce monotonity."
    )

    noise_scale = Unicode(
        "noise_scale",
        help="Observation key with optional scaling factor for noise PSDs",
    )

    mem_report = Bool(
        False, help="Print system memory use while staging / unstaging data."
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    @traitlets.validate("restore_det_data")
    def _check_restore_det_data(self, proposal):
        check = proposal["value"]
        if check and not self.purge_det_data:
            raise traitlets.TraitError(
                "Cannot set restore_det_data since purge_det_data is False"
            )
        if check and self.det_out is not None:
            raise traitlets.TraitError(
                "Cannot set restore_det_data since det_out is not None"
            )
        return check

    @traitlets.validate("restore_pointing")
    def _check_restore_pointing(self, proposal):
        check = proposal["value"]
        if check and not self.purge_pointing:
            raise traitlets.TraitError(
                "Cannot set restore_pointing since purge_pointing is False"
            )
        return check

    @traitlets.validate("det_out")
    def _check_det_out(self, proposal):
        check = proposal["value"]
        if check is not None and self.restore_det_data:
            raise traitlets.TraitError(
                "If det_out is not None, restore_det_data should be False"
            )
        return check

    @traitlets.validate("params")
    def _check_params(self, proposal):
        check = proposal["value"]
        if "info" not in check:
            # The user did not specify the info level- set it from the toast loglevel
            env = Environment.get()
            level = env.log_level()
            if level == "DEBUG":
                check["info"] = 2
            elif level == "VERBOSE":
                check["info"] = 3
            else:
                check["info"] = 1
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not available():
            raise RuntimeError("Madam is either not installed or MPI is disabled")
        self._cached = False
        self._logprefix = "Madam:"

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to the buffers.

        """
        if self._cached:
            madam.clear_caches()
            self._cached = False
        for atr in ["timestamps", "signal", "pixels", "pixweights"]:
            atrname = "_madam_{}".format(atr)
            rawname = "{}_raw".format(atrname)
            if hasattr(self, atrname):
                delattr(self, atrname)
                raw = getattr(self, rawname)
                if raw is not None:
                    raw.clear()
                setattr(self, rawname, None)
                setattr(self, atrname, None)

    def __del__(self):
        self.clear()

    @function_timer
    def _exec(self, data, detectors=None):
        log = Logger.get()

        if not available:
            raise RuntimeError("libmadam is not available")

        if len(data.obs) == 0:
            raise RuntimeError(
                "Madam requires every supplied data object to "
                "contain at least one observation"
            )

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check that the pointing is set
        if self.pixels is None or self.weights is None:
            raise RuntimeError(
                "You must set the pixels and weights before calling exec()"
            )

        # Combine parameters from an external file and other parameters passed in

        params = None
        repeat_keys = ["detset", "detset_nopol", "survey"]

        if self.paramfile is not None:
            if data.comm.world_rank == 0:
                params = dict()
                line_pat = re.compile(r"(\S+)\s+=\s+(\S+)")
                comment_pat = re.compile(r"^\s*\#.*")
                with open(self.paramfile, "r") as f:
                    for line in f:
                        if comment_pat.match(line) is None:
                            line_mat = line_pat.match(line)
                            if line_mat is not None:
                                k = line_mat.group(1)
                                v = line_mat.group(2)
                                if k in repeat_keys:
                                    if k not in params:
                                        params[k] = [v]
                                    else:
                                        params[k].append(v)
                                else:
                                    params[k] = v
            if data.comm.world_comm is not None:
                params = data.comm.world_comm.bcast(params, root=0)
            for k, v in self.params.items():
                if k in repeat_keys:
                    if k not in params:
                        params[k] = [v]
                    else:
                        params[k].append(v)
                else:
                    params[k] = v
        else:
            params = dict(self.params)

        # Set madam parameters that depend on our traits
        if self.mcmode:
            params["mcmode"] = True
        else:
            params["mcmode"] = False

        if self.det_out is not None:
            params["write_tod"] = True
        else:
            params["write_tod"] = False

        # Check input parameters and compute the sizes of Madam data objects
        if data.comm.world_rank == 0:
            msg = "{} Computing data sizes".format(self._logprefix)
            log.info(msg)
        (
            all_dets,
            nsamp,
            nnz,
            nnz_full,
            nnz_stride,
            interval_starts,
            psd_freqs,
            nside,
        ) = self._prepare(params, data, detectors)

        if data.comm.world_rank == 0:
            msg = "{} Copying toast data to buffers".format(self._logprefix)
            log.info(msg)
        psdinfo, signal_dtype, pixels_dtype, weight_dtype = self._stage_data(
            params,
            data,
            all_dets,
            nsamp,
            nnz,
            nnz_full,
            nnz_stride,
            interval_starts,
            psd_freqs,
            nside,
        )

        if data.comm.world_rank == 0:
            msg = "{} Destriping data".format(self._logprefix)
            log.info(msg)
        self._destripe(params, data, all_dets, interval_starts, psdinfo)

        if data.comm.world_rank == 0:
            msg = "{} Copying buffers back to toast data".format(self._logprefix)
            log.info(msg)
        self._unstage_data(
            params,
            data,
            all_dets,
            nsamp,
            nnz,
            nnz_full,
            interval_starts,
            signal_dtype,
            pixels_dtype,
            nside,
            weight_dtype,
        )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.noise_model],
            "shared": [self.times,],
            "detdata": [self.det_data, self.pixels, self.weights],
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = {"detdata": list()}
        if self.det_out is not None:
            prov["detdata"].append(self.det_out)
        return prov

    def _accelerators(self):
        return list()

    @function_timer
    def _prepare(self, params, data, detectors):
        """Examine the data and determine quantities needed to set up Madam buffers"""
        log = Logger.get()
        timer = Timer()
        timer.start()

        if "nside_map" not in params:
            raise RuntimeError(
                "Madam 'nside_map' must be set in the parameter dictionary"
            )
        nside = int(params["nside_map"])

        # Madam requires a fixed set of detectors and pointing matrix non-zeros.
        # Here we find the superset of local detectors used, and also the number
        # of pointing matrix elements.

        nsamp = 0

        # Madam uses monolithic data buffers and specifies contiguous data intervals
        # in that buffer.  The starting sample index is used to mark the transition
        # between data intervals.
        interval_starts = list()

        # This quantity is only used for printing the fraction of samples in valid
        # ranges specified by the View.  Only samples actually in the view are copied
        # to Madam buffers.
        nsamp_valid = 0

        all_dets = set()
        nnz_full = None
        psd_freqs = None

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            all_dets.update(dets)

            # Check that the timestamps exist.
            if self.times not in ob.shared:
                msg = "Shared timestamps '{}' does not exist in observation '{}'".format(
                    self.times, ob.name
                )
                raise RuntimeError(msg)

            # Check that the detector data and pointing exists in the observation
            if self.det_data not in ob.detdata:
                msg = "Detector data '{}' does not exist in observation '{}'".format(
                    self.det_data, ob.name
                )
                raise RuntimeError(msg)
            if self.pixels not in ob.detdata:
                msg = "Detector pixels '{}' does not exist in observation '{}'".format(
                    self.pixels, ob.name
                )
                raise RuntimeError(msg)
            if self.weights not in ob.detdata:
                msg = "Detector data '{}' does not exist in observation '{}'".format(
                    self.weights, ob.name
                )
                raise RuntimeError(msg)

            # Check that the noise model exists, and that the PSD frequencies are the
            # same across all observations (required by Madam).
            if self.noise_model not in ob:
                msg = "Noise model '{}' not in observation '{}'".format(
                    self.noise_model, ob.name
                )
                raise RuntimeError(msg)
            if psd_freqs is None:
                psd_freqs = np.array(
                    ob[self.noise_model].freq(ob.local_detectors[0]), dtype=np.float64
                )
            else:
                check_freqs = ob[self.noise_model].freq(ob.local_detectors[0])
                if not np.allclose(psd_freqs, check_freqs):
                    raise RuntimeError(
                        "All PSDs passed to Madam must have the same frequency binning."
                    )

            # Get the number of pointing weights and verify that it is constant
            # across observations.
            ob_nnz = None
            if len(ob.detdata[self.weights].detector_shape) == 1:
                # The pointing weights just have one dimension (samples)
                ob_nnz = 1
            else:
                ob_nnz = ob.detdata[self.weights].detector_shape[-1]

            if nnz_full is None:
                nnz_full = ob_nnz
            elif ob_nnz != nnz_full:
                msg = "observation '{}' has {} pointing weights per sample, not {}".format(
                    ob.name, ob_nnz, nnz_full
                )
                raise RuntimeError(msg)

            # Are we using a view of the data?  If so, we will only be copying data in
            # those valid intervals.
            if self.view is not None:
                if self.view not in ob.intervals:
                    msg = "View '{}' does not exist in observation {}".format(
                        self.view, ob.name
                    )
                    raise RuntimeError(msg)
                # Go through all the intervals that will be used for our data view
                # and accumulate the number of samples.
                for intvw in ob.intervals[self.view]:
                    interval_starts.append(nsamp_valid)
                    nsamp_valid += intvw.last - intvw.first + 1
            else:
                interval_starts.append(nsamp_valid)
                nsamp_valid += ob.n_local_samples
            nsamp += ob.n_local_samples

        if data.comm.world_rank == 0:
            log.info(
                "{}{:.2f} % of samples are included in valid intervals.".format(
                    self._logprefix, nsamp_valid * 100.0 / nsamp
                )
            )

        nsamp = nsamp_valid

        interval_starts = np.array(interval_starts, dtype=np.int64)
        all_dets = list(all_dets)
        ndet = len(all_dets)

        nnz = None
        nnz_stride = None
        if "temperature_only" in params and params["temperature_only"] in [
            "T",
            "True",
            "TRUE",
            "true",
            True,
        ]:
            # User has requested a temperature-only map.
            if nnz_full not in [1, 3]:
                raise RuntimeError(
                    "Madam cannot make a temperature map with nnz == {}".format(
                        nnz_full
                    )
                )
            nnz = 1
            nnz_stride = nnz_full
        else:
            nnz = nnz_full
            nnz_stride = 1

        if data.comm.world_rank == 0 and "path_output" in params:
            os.makedirs(params["path_output"], exist_ok=True)

        # Inspect the valid intervals across all observations to
        # determine the number of samples per detector

        data.comm.comm_world.barrier()
        timer.stop()
        if data.comm.world_rank == 0:
            msg = "{}  Compute data dimensions: {:0.1f} s".format(
                self._logprefix, timer.seconds()
            )
            log.debug(msg)

        return (
            all_dets,
            nsamp,
            nnz,
            nnz_full,
            nnz_stride,
            interval_starts,
            psd_freqs,
            nside,
        )

    @function_timer
    def _stage_data(
        self,
        params,
        data,
        all_dets,
        nsamp,
        nnz,
        nnz_full,
        nnz_stride,
        interval_starts,
        psd_freqs,
        nside,
    ):
        """Create madam-compatible buffers.

        Collect the data into Madam buffers.  If we are purging TOAST data to save
        memory, then optionally limit the number of processes that are copying at once.

        """
        log = Logger.get()
        timer = Timer()

        nodecomm = data.comm.comm_world.Split_type(
            MPI.COMM_TYPE_SHARED, data.comm.world_rank
        )

        # Determine how many processes per node should copy at once.
        n_copy_groups = 1
        if self.purge_det_data or self.purge_pointing:
            # We will be purging some data- see if we should reduce the number of
            # processes copying in parallel (if we are not purging data, there
            # is no benefit to staggering the copy).
            if self.copy_groups > 0:
                n_copy_groups = min(self.copy_groups, nodecomm.size)

        if not self._cached:
            # Only do this if we have not cached the data yet.
            log_time_memory(
                data,
                prefix=self._logprefix,
                mem_msg="Before staging",
                full_mem=self.mem_report,
            )

        # Copy timestamps and PSDs all at once, since they are never purged.

        psds = dict()

        timer.start()

        if not self._cached:
            timestamp_storage, _ = dtype_to_aligned(madam.TIMESTAMP_TYPE)
            self._madam_timestamps_raw = timestamp_storage.zeros(nsamp)
            self._madam_timestamps = self._madam_timestamps_raw.array()

            interval = 0
            time_offset = 0.0

            for ob in data.obs:
                for vw in ob.view[self.view].shared[self.times]:
                    offset = interval_starts[interval]
                    slc = slice(offset, offset + len(vw), 1)
                    self._madam_timestamps[slc] = vw
                    if self.translate_timestamps:
                        off = self._madam_timestamps[offset] - time_offset
                        self._madam_timestamps[slc] -= off
                        time_offset = self._madam_timestamps[slc][-1] + 1.0
                    interval += 1

                # Get the noise object for this observation and create new
                # entries in the dictionary when the PSD actually changes.  The detector
                # weights are obtained from the noise model.

                nse = ob[self.noise_model]
                nse_scale = 1.0
                if self.noise_scale is not None:
                    if self.noise_scale in ob:
                        nse_scale = float(ob[self.noise_scale])

                for det in all_dets:
                    if det not in ob.local_detectors:
                        continue
                    psd = nse.psd(det) * nse_scale ** 2
                    detw = nse.detector_weight(det)
                    if det not in psds:
                        psds[det] = [(0.0, psd, detw)]
                    else:
                        if not np.allclose(psds[det][-1][1], psd):
                            psds[det] += [(ob.shared[self.times][0], psd, detw)]

            log_time_memory(
                data,
                timer=timer,
                timer_msg="Copy timestamps and PSDs",
                prefix=self._logprefix,
                mem_msg="After timestamp staging",
                full_mem=self.mem_report,
            )

        # Copy the signal.  We always need to do this, even if we are running MCs.

        signal_dtype = data.obs[0].detdata[self.det_data].dtype

        if self._cached:
            # We have previously created the madam buffers.  We just need to fill
            # them from the toast data.  Since both already exist we just copy the
            # contents.
            stage_local(
                data,
                nsamp,
                self.view,
                all_dets,
                self.det_data,
                self._madam_signal,
                interval_starts,
                1,
                1,
                None,
                None,
                None,
                None,
                do_purge=False,
            )
        else:
            # Signal buffers do not yet exist
            if self.purge_det_data:
                # Allocate in a staggered way.
                self._madam_signal_raw, self._madam_signal = stage_in_turns(
                    data,
                    nodecomm,
                    n_copy_groups,
                    nsamp,
                    self.view,
                    all_dets,
                    self.det_data,
                    madam.SIGNAL_TYPE,
                    interval_starts,
                    1,
                    1,
                    None,
                    None,
                    None,
                    None,
                )
            else:
                # Allocate and copy all at once.
                storage, _ = dtype_to_aligned(madam.SIGNAL_TYPE)
                self._madam_signal_raw = storage.zeros(nsamp * len(all_dets))
                self._madam_signal = self._madam_signal_raw.array()

                stage_local(
                    data,
                    nsamp,
                    self.view,
                    all_dets,
                    self.det_data,
                    self._madam_signal,
                    interval_starts,
                    1,
                    1,
                    None,
                    None,
                    None,
                    None,
                    do_purge=False,
                )

        log_time_memory(
            data,
            timer=timer,
            timer_msg="Copy signal",
            prefix=self._logprefix,
            mem_msg="After signal staging",
            full_mem=self.mem_report,
        )

        # Copy the pointing

        pixels_dtype = data.obs[0].detdata[self.pixels].dtype
        weight_dtype = data.obs[0].detdata[self.weights].dtype

        if not self._cached:
            # We do not have the pointing yet.
            if self.purge_pointing:
                # Allocate in a staggered way.
                self._madam_pixels_raw, self._madam_pixels = stage_in_turns(
                    data,
                    nodecomm,
                    n_copy_groups,
                    nsamp,
                    self.view,
                    all_dets,
                    self.pixels,
                    madam.PIXEL_TYPE,
                    interval_starts,
                    1,
                    1,
                    self.shared_flags,
                    self.shared_flag_mask,
                    self.det_flags,
                    self.det_flag_mask,
                )

                self._madam_pixweights_raw, self._madam_pixweights = stage_in_turns(
                    data,
                    nodecomm,
                    n_copy_groups,
                    nsamp,
                    self.view,
                    all_dets,
                    self.weights,
                    madam.WEIGHT_TYPE,
                    interval_starts,
                    nnz,
                    nnz_stride,
                    None,
                    None,
                    None,
                    None,
                )
            else:
                # Allocate and copy all at once.
                storage, _ = dtype_to_aligned(madam.PIXEL_TYPE)
                self._madam_pixels_raw = storage.zeros(nsamp * len(all_dets))
                self._madam_pixels = self._madam_pixels_raw.array()

                stage_local(
                    data,
                    nsamp,
                    self.view,
                    all_dets,
                    self.pixels,
                    self._madam_pixels,
                    interval_starts,
                    1,
                    1,
                    self.shared_flags,
                    self.shared_flag_mask,
                    self.det_flags,
                    self.det_flag_mask,
                    do_purge=False,
                )

                storage, _ = dtype_to_aligned(madam.WEIGHT_TYPE)
                self._madam_pixweights_raw = storage.zeros(nsamp * len(all_dets) * nnz)
                self._madam_pixweights = self._madam_pixweights_raw.array()

                stage_local(
                    data,
                    nsamp,
                    self.view,
                    all_dets,
                    self.weights,
                    self._madam_pixweights,
                    interval_starts,
                    nnz,
                    nnz_stride,
                    None,
                    None,
                    None,
                    None,
                    do_purge=False,
                )

            log_time_memory(
                data,
                timer=timer,
                timer_msg="Copy pointing",
                prefix=self._logprefix,
                mem_msg="After pointing staging",
                full_mem=self.mem_report,
            )

        psdinfo = None

        if not self._cached:
            # Detectors weights.  Madam assumes a single noise weight for each detector
            # that is constant.  We set this based on the first observation or else use
            # uniform weighting.

            ndet = len(all_dets)
            detweights = np.ones(ndet, dtype=np.float64)

            if len(psds) > 0:
                npsdbin = len(psd_freqs)
                npsd = np.zeros(ndet, dtype=np.int64)
                psdstarts = []
                psdvals = []
                for idet, det in enumerate(all_dets):
                    if det not in psds:
                        raise RuntimeError("Every detector must have at least one PSD")
                    psdlist = psds[det]
                    npsd[idet] = len(psdlist)
                    for psdstart, psd, detw in psdlist:
                        psdstarts.append(psdstart)
                        psdvals.append(psd)
                    detweights[idet] = psdlist[0][2]
                npsdtot = np.sum(npsd)
                psdstarts = np.array(psdstarts, dtype=np.float64)
                psdvals = np.hstack(psdvals).astype(madam.PSD_TYPE)
                npsdval = psdvals.size
            else:
                # Uniform weighting
                npsd = np.ones(ndet, dtype=np.int64)
                npsdtot = np.sum(npsd)
                psdstarts = np.zeros(npsdtot)
                npsdbin = 10
                fsample = 10.0
                psd_freqs = np.arange(npsdbin) * fsample / npsdbin
                npsdval = npsdbin * npsdtot
                psdvals = np.ones(npsdval)

            psdinfo = (detweights, npsd, psdstarts, psd_freqs, psdvals)

            log_time_memory(
                data, timer=timer, timer_msg="Collect PSD info", prefix=self._logprefix,
            )
        timer.stop()
        del nodecomm

        return psdinfo, signal_dtype, pixels_dtype, weight_dtype

    @function_timer
    def _unstage_data(
        self,
        params,
        data,
        all_dets,
        nsamp,
        nnz,
        nnz_full,
        interval_starts,
        signal_dtype,
        pixels_dtype,
        nside,
        weight_dtype,
    ):
        """
        Restore data to TOAST observations.

        Optionally copy the signal and pointing back to TOAST if we previously
        purged it to save memory.  Also copy the destriped timestreams if desired.

        """
        log = Logger.get()
        timer = Timer()

        nodecomm = data.comm.comm_world.Split_type(
            MPI.COMM_TYPE_SHARED, data.comm.world_rank
        )

        # Determine how many processes per node should copy at once.
        n_copy_groups = 1
        if self.purge_det_data or self.purge_pointing:
            # We MAY be restoring some data- see if we should reduce the number of
            # processes copying in parallel (if we are not purging data, there
            # is no benefit to staggering the copy).
            if self.copy_groups > 0:
                n_copy_groups = min(self.copy_groups, nodecomm.size)

        log_time_memory(
            data,
            prefix=self._logprefix,
            mem_msg="Before un-staging",
            full_mem=self.mem_report,
        )

        # Copy the signal

        timer.start()

        out_name = self.det_data
        if self.det_out is not None:
            out_name = self.det_out

        if self.det_out is not None or (self.purge_det_data and self.restore_det_data):
            # We are copying some kind of signal back
            if not self.mcmode:
                # We are not running multiple realizations, so delete as we copy.
                restore_in_turns(
                    data,
                    nodecomm,
                    n_copy_groups,
                    nsamp,
                    self.view,
                    all_dets,
                    out_name,
                    signal_dtype,
                    self._madam_signal,
                    self._madam_signal_raw,
                    interval_starts,
                    1,
                    0,
                    True,
                )
                del self._madam_signal
                del self._madam_signal_raw
            else:
                # We want to re-use the signal buffer, just copy.
                restore_local(
                    data,
                    nsamp,
                    self.view,
                    all_dets,
                    out_name,
                    signal_dtype,
                    self._madam_signal,
                    interval_starts,
                    1,
                    0,
                    True,
                )

            log_time_memory(
                data,
                timer=timer,
                timer_msg="Copy signal",
                prefix=self._logprefix,
                mem_msg="After restoring signal",
                full_mem=self.mem_report,
            )

        # Copy the pointing

        if self.purge_pointing and self.restore_pointing:
            # We previously purged it AND we want it back.
            if not self.mcmode:
                # We are not running multiple realizations, so delete as we copy.
                restore_in_turns(
                    data,
                    nodecomm,
                    n_copy_groups,
                    nsamp,
                    self.view,
                    all_dets,
                    self.pixels,
                    pixels_dtype,
                    self._madam_pixels,
                    self._madam_pixels_raw,
                    interval_starts,
                    1,
                    nside,
                    self.pixels_nested,
                )
                del self._madam_pixels
                del self._madam_pixels_raw
                restore_in_turns(
                    data,
                    nodecomm,
                    n_copy_groups,
                    nsamp,
                    self.view,
                    all_dets,
                    self.weights,
                    weigths_dtype,
                    self._madam_pixweights,
                    self._madam_pixweights_raw,
                    interval_starts,
                    nnz,
                    0,
                    True,
                )
                del self._madam_pixweights
                del self._madam_pixweights_raw
            else:
                # We want to re-use the pointing, just copy.
                restore_local(
                    data,
                    nsamp,
                    self.view,
                    all_dets,
                    self.pixels,
                    pixels_dtype,
                    self._madam_pixels,
                    interval_starts,
                    1,
                    nside,
                    self.pixels_nested,
                )
                restore_local(
                    data,
                    nsamp,
                    self.view,
                    all_dets,
                    self.weights,
                    weight_dtype,
                    self._madam_pixweights,
                    interval_starts,
                    nnz,
                    0,
                    True,
                )

            log_time_memory(
                data,
                timer=timer,
                timer_msg="Copy pointing",
                prefix=self._logprefix,
                mem_msg="After restoring pointing",
                full_mem=self.mem_report,
            )

        del nodecomm
        return

    @function_timer
    def _destripe(self, params, data, dets, interval_starts, psdinfo):
        """Destripe the buffered data"""
        log_time_memory(
            data,
            prefix=self._logprefix,
            mem_msg="Just before libmadam.destripe",
            full_mem=self.mem_report,
        )

        if self._cached:
            # destripe
            outpath = ""
            if "path_output" in params:
                outpath = params["path_output"]
            outpath = outpath.encode("ascii")
            madam.destripe_with_cache(
                data.comm.comm_world,
                self._madam_timestamps,
                self._madam_pixels,
                self._madam_pixweights,
                self._madam_signal,
                outpath,
            )
        else:
            (detweights, npsd, psdstarts, psd_freqs, psdvals) = psdinfo

            # destripe
            madam.destripe(
                data.comm.comm_world,
                params,
                dets,
                detweights,
                self._madam_timestamps,
                self._madam_pixels,
                self._madam_pixweights,
                self._madam_signal,
                interval_starts,
                npsd,
                psdstarts,
                psd_freqs,
                psdvals,
            )
            if self.mcmode:
                self._cached = True
        return
