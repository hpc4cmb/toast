# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, use_mpi

import os

import traitlets

import numpy as np

from ..utils import Logger, Timer, GlobalTimers, dtype_to_aligned

from ..traits import trait_docs, Int, Unicode, Bool, Dict

from ..timing import function_timer

from .operator import Operator

from .memory_counter import MemoryCounter


madam = None
if use_mpi:
    try:
        import libmadam_wrapper as madam
    except ImportError:
        madam = None


@trait_docs
class Madam(Operator):
    """Operator which passes data to libmadam for map-making."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    params = Dict(dict(), help="Parameters to pass to madam")

    times = Unicode("times", help="Observation shared key for timestamps")

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached = False

    @classmethod
    def available(cls):
        """(bool): True if libmadam is found in the library search path."""
        return madam is not None and madam.available

    def clear(self):
        """Delete the underlying memory.

        This will forcibly delete the C-allocated memory and invalidate all python
        references to the buffers.  DO NOT CALL THIS unless you are sure all references
        are no longer being used.

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

        if not self.available:
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

        # Check input parameters and compute the sizes of Madam data objects
        (
            all_dets,
            nsamp,
            nnz,
            nnz_full,
            nnz_stride,
            interval_starts,
            psd_freqs,
            nside,
        ) = self._prepare(data, detectors)

        psdinfo, signal_type, pixels_dtype, weight_dtype = self._stage_data(
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

        # self._destripe(pars, dets, periods, psdinfo)
        #
        # self._unstage_data(
        #     nsamp,
        #     nnz,
        #     nnz_full,
        #     obs_period_ranges,
        #     dets,
        #     signal_type,
        #     pixels_dtype,
        #     nside,
        #     weight_dtype,
        # )

        return

    def __del__(self):
        if self._cached:
            madam.clear_caches()
            self._cached = False

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.noise_model],
            "shared": [
                self.times,
            ],
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
    def _prepare(self, data, detectors):
        """Examine the data and determine quantities needed to set up Madam buffers"""
        log = Logger.get()
        # timer = Timer()
        # timer.start()

        if "nside_map" not in self.params:
            raise RuntimeError(
                "Madam 'nside_map' must be set in the parameter dictionary"
            )
        nside = int(self.params["nside_map"])

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
                msg = (
                    "Shared timestamps '{}' does not exist in observation '{}'".format(
                        self.times, ob.name
                    )
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
                "Madam: {:.2f} % of samples are included in valid "
                "intervals.".format(nsamp_valid * 100.0 / nsamp)
            )

        nsamp = nsamp_valid

        all_dets = list(all_dets)
        ndet = len(all_dets)

        nnz = None
        nnz_stride = None
        if "temperature_only" in self.params and self.params["temperature_only"] in [
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

        if data.comm.world_rank == 0 and "path_output" in self.params:
            os.makedirs(self.params["path_output"], exist_ok=True)

        # Inspect the valid intervals across all observations to
        # determine the number of samples per detector

        data.comm.comm_world.Barrier()
        # if self._rank == 0:
        #     log.debug()
        #     timer.report_clear("Collect dataset dimensions")

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

        # Memory counting operator
        mem_count = MemoryCounter(silent=True)

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

        # self._comm.Barrier()
        # timer_tot = Timer()
        # timer_tot.start()

        # Copy timestamps and PSDs all at once, since they are never purged.

        timestamp_storage, _ = dtype_to_aligned(madam.TIMESTAMP_TYPE)
        self._madam_timestamps_raw = timestamp_storage.zeros(nsamp)
        self._madam_timestamps = self._madam_timestamps_raw.array()
        psds = dict()

        interval = 0
        time_offset = 0.0

        for ob in data.obs:
            if self.view is not None:
                for vw in ob.view[self.view].shared[self.times]:
                    offset = interval_starts[interval]
                    slc = slice(offset, offset + len(vw), 1)
                    self._madam_timestamps[slc] = vw
                    if self.translate_timestamps:
                        off = self._madam_timestamps[offset] - time_offset
                        self._madam_timestamps[slc] -= off
                        time_offset = self._madam_timestamps[slc][-1] + 1.0
                    interval += 1
            else:
                offset = interval_starts[interval]
                slc = slice(offset, offset + ob.n_local_samples, 1)
                self._madam_timestamps[slc] = ob.shared[self.times]
                if self.translate_timestamps:
                    off = self._madam_timestamps[offset] - time_offset
                    self._madam_timestamps[slc] -= off
                    time_offset = self._madam_timestamps[slc][-1] + 1.0
                interval += 1

            # Get the noise object for this observation and create new
            # entries in the dictionary when the PSD actually changes
            nse = ob[self.noise_model]
            nse_scale = 1.0
            if self.noise_scale is not None:
                if self.noise_scale in ob:
                    nse_scale = float(ob[self.noise_scale])

            for det in all_dets:
                if det not in ob.local_detectors:
                    continue
                psd = nse.psd(det) * nse_scale ** 2
                if det not in psds:
                    psds[det] = [(0.0, psd)]
                else:
                    if not np.allclose(psds[det][-1][1], psd):
                        psds[det] += [(ob.shared[self.times][0], psd)]

        def copy_local(detdata_name, madam_dtype, dnnz, do_flags=False, do_purge=False):
            """Helper function to create a madam buffer from a local detdata key."""
            storage, _ = dtype_to_aligned(madam_dtype)
            n_all_det = len(all_dets)
            raw = storage.zeros(nsamp * n_all_det)
            wrapped = raw.array()
            interval = 0
            for ob in data.obs:
                if self.view is not None:
                    for vw in ob.view[self.view].detdata[detdeta_name]:
                        offset = interval_starts[interval]
                        flags = None
                        if do_flags:
                            if (
                                self.shared_flags is not None
                                or self.det_flags is not None
                            ):
                                # Using flags
                                flags = np.zeros(len(vw), dtype=np.uint8)
                            if self.shared_flags is not None:
                                flags |= (
                                    ob.view[self.view].shared[self.shared_flags]
                                    & self.shared_flag_mask
                                )

                        for idet, det in enumerate(all_dets):
                            if det not in ob.local_detectors:
                                continue
                            slc = slice(
                                (idet * nsamp + offset) * dnnz,
                                (idet * nsamp + offset + len(vw)) * dnnz,
                                1,
                            )
                            if dnnz > 1:
                                wrapped[slc] = vw[idet].flatten()[::nnz_stride]
                            else:
                                wrapped[slc] = vw[idet].flatten()
                            detflags = None
                            if do_flags:
                                if self.det_flags is None:
                                    detflags = flags
                                else:
                                    detflags = np.copy(flags)
                                    detflags |= (
                                        ob.view[self.view].detdata[self.det_flags][idet]
                                        & self.det_flag_mask
                                    )
                                # The do_flags option should only be true if we are
                                # processing the pixel indices (which is how madam
                                # effectively implements flagging).  So we will set
                                # all flagged samples to "-1"
                                if detflags is not None:
                                    # sanity check
                                    if nnz != 1:
                                        raise RuntimeError(
                                            "Internal error on madam copy.  Only pixel indices should be flagged."
                                        )
                                    wrapped[slc][detflags != 0] = -1
                        interval += 1
                else:
                    offset = interval_starts[interval]
                    flags = None
                    if do_flags:
                        if self.shared_flags is not None or self.det_flags is not None:
                            # Using flags
                            flags = np.zeros(ob.n_local_samples, dtype=np.uint8)
                        if self.shared_flags is not None:
                            flags |= (
                                ob.shared[self.shared_flags] & self.shared_flag_mask
                            )
                    for idet, det in enumerate(all_dets):
                        if det not in ob.local_detectors:
                            continue
                        slc = slice(
                            (idet * nsamp + offset) * dnnz,
                            (idet * nsamp + offset + ob.n_local_samples) * dnnz,
                            1,
                        )
                        if dnnz > 1:
                            wrapped[slc] = ob.detdata[detdata_name][idet].flatten()[
                                ::nnz_stride
                            ]
                        else:
                            wrapped[slc] = ob.detdata[detdata_name][idet].flatten()
                        detflags = None
                        if do_flags:
                            if self.det_flags is None:
                                detflags = flags
                            else:
                                detflags = np.copy(flags)
                                detflags |= (
                                    ob.detdata[self.det_flags][idet]
                                    & self.det_flag_mask
                                )
                            # The do_flags option should only be true if we are
                            # processing the pixel indices (which is how madam
                            # effectively implements flagging).  So we will set
                            # all flagged samples to "-1"
                            if detflags is not None:
                                # sanity check
                                if dnnz != 1:
                                    raise RuntimeError(
                                        "Internal error on madam copy.  Only pixel indices should be flagged."
                                    )
                                wrapped[slc][detflags != 0] = -1
                    interval += 1
                if do_purge:
                    del ob.detdata[detdata_name]
            return raw, wrapped

        def copy_in_turns(detdata_name, madam_dtype, dnnz, do_flags):
            """When purging data, take turns copying it."""
            raw = None
            wrapped = None
            for copying in range(n_copy_groups):
                if nodecomm.rank % n_copy_groups == copying:
                    # Our turn to copy data
                    raw, wrapped = copy_local(
                        detdata_name,
                        madam_dtype,
                        dnnz,
                        do_flags=do_flags,
                        do_purge=True,
                    )
                nodecomm.barrier()
            return raw, wrapped

        # Copy the signal

        if self.purge_det_data:
            self._madam_signal_raw, self._madam_signal = copy_in_turns(
                self.det_data, madam.SIGNAL_TYPE, 1, do_flags=False
            )
        else:
            self._madam_signal_raw, self._madam_signal = copy_local(
                self.det_data, madam.SIGNAL_TYPE, 1, do_flags=False, do_purge=False
            )

        # Copy the pointing

        if self.purge_pointing:
            self._madam_pixels_raw, self._madam_pixels = copy_in_turns(
                self.pixels, madam.PIXEL_TYPE, 1, do_flags=True
            )
            self._madam_weights_raw, self._madam_weights = copy_in_turns(
                self.weights, madam.WEIGHT_TYPE, nnz, do_flags=False
            )
        else:
            self._madam_pixels_raw, self._madam_pixels = copy_local(
                self.pixels, madam.PIXEL_TYPE, 1, do_flags=True, do_purge=False
            )
            self._madam_weights_raw, self._madam_weights = copy_local(
                self.weights, madam.WEIGHT_TYPE, nnz, do_flags=False, do_purge=False
            )

        # Madam uses constant detector weights?

        # # detweights is either a dictionary of weights specified at
        # # construction time, or else we use uniform weighting.
        # detw = {}
        # if self._detw is None:
        #     for idet, det in enumerate(detectors):
        #         detw[det] = 1.0
        # else:
        #     detw = self._detw
        #
        # detweights = np.zeros(ndet, dtype=np.float64)
        # for idet, det in enumerate(detectors):
        #     detweights[idet] = detw[det]
        #
        # if len(psds) > 0:
        #     npsdbin = len(psdfreqs)
        #
        #     npsd = np.zeros(ndet, dtype=np.int64)
        #     psdstarts = []
        #     psdvals = []
        #     for idet, det in enumerate(detectors):
        #         if det not in psds:
        #             raise RuntimeError("Every detector must have at least " "one PSD")
        #         psdlist = psds[det]
        #         npsd[idet] = len(psdlist)
        #         for psdstart, psd in psdlist:
        #             psdstarts.append(psdstart)
        #             psdvals.append(psd)
        #     npsdtot = np.sum(npsd)
        #     psdstarts = np.array(psdstarts, dtype=np.float64)
        #     psdvals = np.hstack(psdvals).astype(madam.PSD_TYPE)
        #     npsdval = psdvals.size
        # else:
        #     npsd = np.ones(ndet, dtype=np.int64)
        #     npsdtot = np.sum(npsd)
        #     psdstarts = np.zeros(npsdtot)
        #     npsdbin = 10
        #     fsample = 10.0
        #     psdfreqs = np.arange(npsdbin) * fsample / npsdbin
        #     npsdval = npsdbin * npsdtot
        #     psdvals = np.ones(npsdval)
        # psdinfo = (detweights, npsd, psdstarts, psdfreqs, psdvals)
        # if self._rank == 0 and self._verbose:
        #     timer_tot.report_clear("Collect PSD info")
        # return psdinfo, signal_dtype, pixels_dtype, weight_dtype

    # def _unstage_data(self):
    #     pass
    #
    # @function_timer
    # def _destripe(self, pars, dets, periods, psdinfo):
    #     """Destripe the buffered data"""
    #     if self._verbose:
    #         memreport("just before calling libmadam.destripe", self._comm)
    #     if self._cached:
    #         # destripe
    #         outpath = ""
    #         if "path_output" in self.params:
    #             outpath = self.params["path_output"]
    #         outpath = outpath.encode("ascii")
    #         madam.destripe_with_cache(
    #             self._comm,
    #             self._madam_timestamps,
    #             self._madam_pixels,
    #             self._madam_pixweights,
    #             self._madam_signal,
    #             outpath,
    #         )
    #     else:
    #         (detweights, npsd, psdstarts, psdfreqs, psdvals) = psdinfo
    #
    #         # destripe
    #         madam.destripe(
    #             self._comm,
    #             pars,
    #             dets,
    #             detweights,
    #             self._madam_timestamps,
    #             self._madam_pixels,
    #             self._madam_pixweights,
    #             self._madam_signal,
    #             periods,
    #             npsd,
    #             psdstarts,
    #             psdfreqs,
    #             psdvals,
    #         )
    #
    #         if self._mcmode:
    #             self._cached = True
    #     return
