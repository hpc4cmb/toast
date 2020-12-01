# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, use_mpi

import traitlets

import numpy as np

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Dict

from ..timing import function_timer

from .operator import Operator

from .clear import Clear

from .copy import Copy


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

    view = Unicode(None, allow_none=True, help="Use this view of the data in all observations")

    pixels_nested = Bool(True, help="True if pixel indices are in NESTED ordering")

    det_out = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output destriped timestreams",
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    purge = Bool(
        False, help="If True, clear all observation data after copying to madam buffers"
    )

    purge_det_data = Bool(
        False,
        help="If True, clear all observation detector data after copying to madam buffers",
    )

    purge_pointing = Bool(
        False,
        help="If True, clear all observation detector pointing data after copying to madam buffers",
    )

    purge_flags = Bool(
        False,
        help="If True, clear all observation detector flags after copying to madam buffers",
    )

    mcmode = Bool(
        False,
        help="If true, Madam will store auxiliary information such as pixel matrices and noise filter.",
    )

    conserve_memory = Int(0, help="Stagger the Madam buffer staging on each node.")

    translate_timestamps = Bool(
        False, help="Translate timestamps to enforce monotonity."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached = False

    @classmethod
    def available(cls):
        """(bool): True if libmadam is found in the library search path."""
        return (madam is not None and madam.available)

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

        # Check purging
        if self.purge:
            # Purging everything
            self.purge_det_data = True
            self.purge_pointing = True
            self.purge_flags = True

        # Madam-compatible data buffers
        self._madam_timestamps = None
        self._madam_pixels = None
        self._madam_pixweights = None
        self._madam_signal = None



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
            "detdata": [
                self.det_data,
                self.pixels,
                self.weights
            ],
            "intervals": list()
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
        """Examine the data and determine quantities needed to set up Madam data"""
        log = Logger.get()
        timer = Timer()
        timer.start()

        # Madam requires a fixed set of detectors and pointing matrix non-zeros.
        # Here we find the superset of local detectors used, and also the number
        # of pointing matrix elements.

        nsamp = 0
        all_dets = set()
        nnz_full = None
        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            all_dets.add(dets)

            # Are we using a view of the data?  If so, we will only consider data in
            # those valid intervals.
            if self.view is not None:
                if self.view not in ob.intervals:
                    msg = "View '{}' does not exist in observation {}".format(self.view, ob.name)
                    raise RuntimeError(msg)

            nsamp += ob.n_local_samples

            # Check that the detector data and pointing exists in the observation
            if self.det_data not in ob.detdata:
                msg = "Detector data '{}' does not exist in observation '{}'".format(self.det_data, ob.name)
                raise RuntimeError(msg)
            if self.pixels not in ob.detdata:
                msg = "Detector pixels '{}' does not exist in observation '{}'".format(self.pixels, ob.name)
                raise RuntimeError(msg)
            if self.weights not in ob.detdata:
                msg = "Detector data '{}' does not exist in observation '{}'".format(self.weights, ob.name)
                raise RuntimeError(msg)

            # Get the number of pointing weights and verify that it is constant
            # across observations.
            ob_nnz = None
                if len(ob.detdata[self.weights].detector_shape) == 1:
                    # The pointing weights just have one dimension (samples)
                    ob_full = 1
                else:
                    ob_full = ob.detdata[self.weights].detector_shape[-1]

            if nnz_full is None:
                nnz_full = ob_nnz
            elif ob_nnz != nnz_full:
                msg = "observation '{}' has {} pointing weights per sample, not {}".format(ob.name, ob_nnz, nnz_full)
                raise RuntimeError(msg)

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
                    "Madam cannot make a temperature map with nnz == {}".format(nnz_full)
                )
            nnz = 1
            nnz_stride = nnz_full
        else:
            nnz = nnz_full
            nnz_stride = 1

        if "nside_map" not in self.params:
            raise RuntimeError(
                "Madam 'nside_map' must be set in the parameter dictionary"
            )
        nside = int(self.params["nside_map"])

        if data.comm.world_rank == 0 and "path_output" in self.params:
            os.makedirs(self.params["path_output"], exist_ok=True)

        # Inspect the valid intervals across all observations to
        # determine the number of samples per detector

        obs_period_ranges, psdfreqs, periods, nsamp = self._get_period_ranges(
            dets, nsamp
        )

        self._comm.Barrier()
        if self._rank == 0:
            log.debug()
            timer.report_clear("Collect dataset dimensions")

        return (
            all_dets,
            nsamp,
            ndet,
            nnz,
            nnz_full,
            nnz_stride,
            periods,
            obs_period_ranges,
            psdfreqs,
            nside,
        )
