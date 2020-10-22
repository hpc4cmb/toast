# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

import numpy as np

from ..utils import Environment, Logger

from ..traits import trait_docs, Int, Unicode, Bool

from ..healpix import HealpixPixels

from ..operator import Operator

from ..timing import function_timer

from .. import qarray as qa

from ..pixels import PixelDistribution

from .._libtoast import pointing_matrix_healpix


@trait_docs
class PointingHealpix(Operator):
    """Operator which generates I/Q/U healpix pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.  An optional dictionary of pointing weight calibration factors
    may be specified for each observation.

    For each observation, the cross-polar response for every detector is obtained from
    the Focalplane, and if a HWP angle timestream exists, then a perfect HWP Mueller
    matrix is included in the response.

    The timestream model is then (see Jones, et al, 2006):

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a} + U \\sin{2a}\\right]\\right]

    Or, if a HWP is included in the response with time varying angle "w", then
    the total response is:

    .. math::
        d = cal \\left[\\frac{(1+eps)}{2} I + \\frac{(1-eps)}{2} \\left[Q \\cos{2a+4w} + U \\sin{2a+4w}\\right]\\right]

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    nside = Int(64, help="The NSIDE resolution")

    nside_submap = Int(16, help="The NSIDE of the submap resolution")

    nest = Bool(False, help="If True, used NESTED ordering instead of RING")

    mode = Unicode("I", help="The Stokes weights to generate (I or IQU)")

    boresight = Unicode("boresight_radec", help="Observation shared key for boresight")

    hwp_angle = Unicode("hwp_angle", help="Observation shared key for HWP angle")

    flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    flag_mask = Int(0, help="Bit mask value for optional flagging")

    pixels = Unicode("pixels", help="Observation detdata key for output pixel indices")

    weights = Unicode("weights", help="Observation detdata key for output weights")

    quats = Unicode(
        "quats",
        allow_none=True,
        help="Observation detdata key for output quaternions (for debugging)",
    )

    create_dist = Unicode(
        None,
        allow_none=True,
        help="Create the submap distribution for all detectors and store in the Data key specified",
    )

    single_precision = Bool(False, help="If True, use 32bit int / float in output")

    cal = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of pointing weight calibration for each det",
    )

    @traitlets.validate("nside")
    def _check_nside(self, proposal):
        check = proposal["value"]
        if ~check & (check - 1) != check - 1:
            raise traitlets.TraitError("Invalid NSIDE value")
        return check

    @traitlets.validate("nside_submap")
    def _check_nside_submap(self, proposal):
        check = proposal["value"]
        if ~check & (check - 1) != check - 1:
            raise traitlets.TraitError("Invalid NSIDE submap value")
        if check > self.nside:
            newval = 16
            if newval > self.nside:
                newval = self.nside
            log = Logger.get()
            log.warning(
                "NSIDE submap greater than NSIDE.  Setting to {} instead".format(newval)
            )
            check = newval
        return check

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        check = proposal["value"]
        if check not in ["I", "IQU"]:
            raise traitlets.TraitError("Invalid mode (must be 'I' or 'IQU')")
        return check

    @traitlets.validate("flag_mask")
    def _check_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize the healpix pixels object
        self.hpix = HealpixPixels(self.nside)

        self._nnz = 1
        if self.mode == "IQU":
            self._nnz = 3

        self._n_pix = 12 * self.nside ** 2
        self._n_pix_submap = 12 * self.nside_submap ** 2
        self._n_submap = (self.nside // self.nside_submap) ** 2

        self._local_submaps = None
        if self.create_dist is not None:
            self._local_submaps = np.zeros(self._n_submap, dtype=np.bool)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        # We do the calculation over buffers of timestream samples to reduce memory
        # overhead from temporary arrays.
        tod_buffer_length = env.tod_buffer_length()

        for obs in data.obs:
            # Get the detectors we are using for this observation
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Get the flags if needed
            flags = None
            if self.flags is not None:
                flags = obs.shared[self.flags]
                flags &= self.flag_mask

            # Boresight pointing quaternions
            boresight = obs.shared[self.boresight]

            # Focalplane for this observation
            focalplane = obs.telescope.focalplane

            # Optional calibration
            cal = None
            if self.cal is not None:
                cal = obs[self.cal]

            # Create output data for the pixels, weights and optionally the
            # detector quaternions.

            if self.single_precision:
                obs.detdata.create(self.pixels, shape=(1,), dtype=np.int32)
                obs.create_detector_data(
                    self.config["pixels"],
                    shape=(n_samp,),
                    dtype=np.int32,
                    detectors=dets,
                )
                obs.create_detector_data(
                    self.config["weights"],
                    shape=(n_samp, self._nnz),
                    dtype=np.float32,
                    detectors=dets,
                )
            else:
                obs.create_detector_data(
                    self.config["pixels"],
                    shape=(n_samp,),
                    dtype=np.int64,
                    detectors=dets,
                )
                obs.create_detector_data(
                    self.config["weights"],
                    shape=(n_samp, self._nnz),
                    dtype=np.float64,
                    detectors=dets,
                )

            if self.config["quats"] is not None:
                obs.create_detector_data(
                    self.config["quats"],
                    shape=(n_samp, 4),
                    dtype=np.float64,
                    detectors=dets,
                )

            for det in dets:
                props = focalplane[det]

                # Get the cross polar response from the focalplane
                epsilon = 0.0
                if "pol_leakage" in props:
                    epsilon = props["pol_leakage"]

                # Detector quaternion offset from the boresight
                detquat = props["quat"]

                # Timestream of detector quaternions
                quats = qa.mult(boresight, detquat)
                if self.config["quats"] is not None:
                    obs[self.config["quats"]][det][:] = quats

                # Cal for this detector
                dcal = 1.0
                if cal is not None:
                    dcal = cal[det]

                # Buffered pointing calculation
                buf_off = 0
                buf_n = tod_buffer_length
                while buf_off < n_samp:
                    if buf_off + buf_n > n_samp:
                        buf_n = n_samp - buf_off
                    bslice = slice(buf_off, buf_off + buf_n)

                    # This buffer of detector quaternions
                    detp = quats[bslice, :].reshape(-1)

                    # Buffer of HWP angle
                    hslice = None
                    if hwpang is not None:
                        hslice = hwpang[bslice].reshape(-1)

                    # Buffer of flags
                    fslice = None
                    if flags is not None:
                        fslice = flags[bslice].reshape(-1)

                    # Pixel and weight buffers
                    pxslice = obs[self.config["pixels"]][det][bslice].reshape(-1)
                    wtslice = obs[self.config["weights"]][det][bslice].reshape(-1)

                    pbuf = pxslice
                    wbuf = wtslice
                    if self.config["single_precision"]:
                        pbuf = np.zeros(len(pxslice), dtype=np.int64)
                        wbuf = np.zeros(len(wtslice), dtype=np.float64)

                    pointing_matrix_healpix(
                        self.hpix,
                        self.config["nest"],
                        epsilon,
                        dcal,
                        self.config["mode"],
                        detp,
                        hslice,
                        fslice,
                        pxslice,
                        wtslice,
                    )

                    if self.config["single_precision"]:
                        pxslice[:] = pbuf.astype(np.int32)
                        wtslice[:] = wbuf.astype(np.float32)

                    buf_off += buf_n

                if self.config["create_dist"] is not None:
                    self._local_submaps[
                        obs[self.config["pixels"]][det] // self._n_pix_submap
                    ] = True
        return

    def finalize(self, data):
        """Perform any final operations / communication.

        Args:
            data (toast.Data):  The distributed data.

        Returns:
            (PixelDistribution):  Return the final submap distribution or None.

        """
        # Optionally return the submap distribution
        if self.config["create_dist"] is not None:
            submaps = None
            if self.config["single_precision"]:
                submaps = np.arange(self._n_submap, dtype=np.int32)[self._local_submaps]
            else:
                submaps = np.arange(self._n_submap, dtype=np.int64)[self._local_submaps]
            data[self.config["create_dist"]] = PixelDistribution(
                n_pix=self._n_pix,
                n_submap=self._n_submap,
                local_submaps=submaps,
                comm=data.comm.comm_world,
            )
        return

    def requires(self):
        """List of Observation keys directly used by this Operator."""
        req = ["BORESIGHT_RADEC", "HWP_ANGLE"]
        if self.config["flags"] is not None:
            req.append(self.config["flags"])
        if self.config["cal"] is not None:
            req.append(self.config["cal"])
        return req

    def provides(self):
        """List of Observation keys generated by this Operator."""
        prov = [self.config["pixels"], self.config["weights"]]
        if self.config["quats"] is not None:
            prov.append(self.config["quats"])
        return prov

    def accelerators(self):
        """List of accelerators supported by this Operator."""
        return list()

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return {
            "meta": [
                self.noise_model,
            ],
            "shared": [
                self.times,
            ],
        }

    def _provides(self):
        return {
            "detdata": [
                self.out,
            ]
        }

    def _accelerators(self):
        return list()
