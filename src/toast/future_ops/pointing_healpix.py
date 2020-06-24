# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..utils import Environment, Logger

from ..healpix import HealpixPixels

from ..operator import Operator

from ..config import ObjectConfig

from ..timing import function_timer

from .. import qarray as qa

from ..pixels import PixelDistribution

from .._libtoast import pointing_matrix_healpix


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

    Args:
        config (dict): Configuration parameters.

    """

    def __init__(self, config):
        super().__init__(config)
        self._parse()

        # Initialize the healpix pixels object
        self.hpix = HealpixPixels(self.config["nside"])

        self._nnz = 1
        if self.config["mode"] == "IQU":
            self._nnz = 3

        self._n_pix = 12 * self.config["nside"] ** 2
        self._n_pix_submap = 12 * self.config["nside_submap"] ** 2
        self._n_submap = (self.config["nside"] // self.config["nside_submap"]) ** 2

        self._local_submaps = None
        if self.config["create_dist"] is not None:
            self._local_submaps = np.zeros(self._n_submap, dtype=np.bool)

    @classmethod
    def defaults(cls):
        """(Class method) Return options supported by the operator and their defaults.

        This returns an ObjectConfig instance, and each entry should have a help
        string.

        Returns:
            (ObjectConfig): The options.

        """
        opts = ObjectConfig()

        opts.add("class", "toast.future_ops.PointingHealpix", "The class name")

        opts.add("API", 0, "(Internal interface version for this operator)")

        opts.add("pixels", "pixels", "The observation name of the output pixels")

        opts.add("weights", "weights", "The observation name of the output weights")

        opts.add(
            "quats",
            None,
            "If not None, save detector quaternions to this name (for debugging)",
        )

        opts.add("nside", 64, "The NSIDE resolution")

        opts.add("nside_submap", 16, "The submap resolution")

        opts.add("nest", False, "If True, use NESTED ordering instead of RING")

        opts.add("mode", "I", "The Stokes weights to generate (I or IQU)")

        opts.add("flags", None, "Optional common timestream flags to apply")

        opts.add("flag_mask", 0, "Bit mask value for optional flagging")

        opts.add(
            "create_dist",
            None,
            "Create the submap distribution for all detectors and store in the Data key specified",
        )

        opts.add("single_precision", False, "If True, use 32bit int / float in output")

        opts.add(
            "cal",
            None,
            "The observation key with a dictionary of pointing weight calibration for each det",
        )

        return opts

    def _parse(self):
        log = Logger.get()
        if self.config["nside_submap"] >= self.config["nside"]:
            newsub = self.config["nside"] // 4
            if newsub == 0:
                newsub = 1
            log.warning("nside_submap >= nside, setting to {}".format(newsub))
            self.config["nside_submap"] = newsub
        if self.config["mode"] not in ["I", "IQU"]:
            msg = "Invalide mode '{}', allowed values are 'I' and 'IQU'".format(
                self.config["mode"]
            )
            log.error(msg)
            raise RuntimeError(msg)

    @function_timer
    def exec(self, data, detectors=None):
        """Create pixels and weights.

        This iterates over all observations and specified detectors, and creates
        the pixel and weight arrays representing the pointing matrix.  Data is stored
        in newly created DetectorData members of each observation.

        The locally hit submaps are optionally computed.  This is typically only done
        when initially computing the pointing for all detectors.

        Args:
            data (toast.Data):  The distributed data.
            detectors (list):  A list of detector names or indices.  If None, this
                indicates a list of all detectors.

        Returns:
            None

        """
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

            # The number of samples on this process
            n_samp = obs.local_samples[1]

            # See if we have a HWP angle
            hwpang = None
            try:
                hwpang = obs.hwp_angle
            except KeyError:
                if obs.mpicomm is None or obs.mpicomm.rank == 0:
                    msg = "Observation {} has no HWP angle- not including in response".format(
                        obs.name
                    )
                    log.verbose(msg)

            # Get the flags if needed
            flags = None
            if self.config["flags"] is not None:
                flags = obs.get_common_flags(keyname=self.config["flags"])
                flags &= self.config["flag_mask"]

            # Boresight pointing quaternions
            boresight = obs.boresight_radec

            # Focalplane for this observation
            focalplane = obs.telescope.focalplane

            # Optional calibration
            cal = None
            if self.config["cal"] is not None:
                cal = obs[self.config["cal"]]

            # Create output data for the pixels, weights and optionally the
            # detector quaternions.

            if self.config["single_precision"]:
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
        """List of Observation keys directly used by this Operator.
        """
        req = ["BORESIGHT_RADEC", "HWP_ANGLE"]
        if self.config["flags"] is not None:
            req.append(self.config["flags"])
        if self.config["cal"] is not None:
            req.append(self.config["cal"])
        return req

    def provides(self):
        """List of Observation keys generated by this Operator.
        """
        prov = [self.config["pixels"], self.config["weights"]]
        if self.config["quats"] is not None:
            prov.append(self.config["quats"])
        return prov

    def accelerators(self):
        """List of accelerators supported by this Operator.
        """
        return list()
