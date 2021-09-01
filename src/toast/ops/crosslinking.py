# Copyright (c) 2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import traitlets

import numpy as np

from ..utils import Logger

from ..mpi import MPI

from .. import qarray as qa

from ..data import Data

from ..traits import trait_docs, Int, Unicode, Bool, Float, Instance

from ..timing import function_timer, Timer

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ..observation import default_names as obs_names

from .operator import Operator

from .pipeline import Pipeline

from .delete import Delete

from .copy import Copy

from .arithmetic import Subtract

from .pointing_healpix import PointingHealpix

from .mapmaker_utils import BuildNoiseWeighted


@trait_docs
class CrossLinking(Operator):
    """ Evaluate an ACT-style crosslinking map
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="This must be an instance of a pointing operator.  "
        "Used exclusively for pixel numbers, not pointing weights.",
    )

    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for output quaternions",
    )

    pixel_dist = Unicode(
        "pixel_dist",
        help="The Data key where the PixelDist object should be stored",
    )

    det_flags = Unicode(
        obs_names.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(255, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        obs_names.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional telescope flagging")

    output_dir = Unicode(
        ".",
        help="Write output data products to this directory",
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    sync_type = Unicode(
        "alltoallv", help="Communication algorithm: 'allreduce' or 'alltoallv'"
    )

    signal = "dummy_signal"
    weights = "crosslinking_weights"
    crosslinking_map = "crosslinking_map"

    @traitlets.validate("pointing")
    def _check_pointing(self, proposal):
        pntg = proposal["value"]
        if pntg is not None:
            if not isinstance(pntg, Operator):
                raise traitlets.TraitError("pointing should be an Operator instance")
            # Check that this operator has the traits we expect
            for trt in ["pixels", "weights", "create_dist", "view"]:
                if not pntg.has_trait(trt):
                    msg = "pointing operator should have a '{}' trait".format(trt)
                    raise traitlets.TraitError(msg)
        return pntg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_weights(self, data, obs, detectors):
        """ Evaluate the special pointing matrix
        """

        nsample = obs.n_local_samples
        focalplane = obs.telescope.focalplane
        dets = obs.select_local_detectors(detectors)

        obs.detdata.ensure(self.signal, detectors=dets)
        obs.detdata.ensure(self.weights, sample_shape=(3,), detectors=dets)
        
        for det in dets:
            signal = obs.detdata[self.signal][det]
            signal[:] = 1
            weights = obs.detdata[self.weights][det]
            try:
                # Use cached detector quaternions
                quat = obs.detdata[self.pointing.detector_pointing.quats][det]
            except KeyError:
                # Compute the detector quaternions
                obs_data = Data(comm=data.comm)
                obs_data._internal = data._internal
                obs_data.obs = [obs]
                self.pointing.detector_pointing.apply(obs_data, detectors=[det])
                obs_data.obs.clear()
                del obs_data
                quat = obs.detdata[self.pointing.detector_pointing.quats][det]
            # measure the scan direction wrt the local meridian
            # for each sample
            theta, phi = qa.to_position(quat)
            theta = np.pi / 2 - theta
            # scan direction across the reference sample
            dphi = (np.roll(phi, -1) - np.roll(phi, 1))
            dtheta = np.roll(theta, -1) - np.roll(theta, 1)
            # except first and last sample
            for dx, x in (dphi, phi), (dtheta, theta):
                dx[0] = x[1] - x[0]
                dx[-1] = x[-1] - x[-2]
            # scale dphi to on-sky
            dphi *= np.cos(theta)
            # Avoid overflows
            tiny = np.abs(dphi) < 1e-30
            if np.any(tiny):
                ang = np.zeros(nsample)
                ang[tiny] = np.sign(dtheta) * np.sign(dphi) * np.pi / 2
                not_tiny = np.logical_not(tiny)
                ang[not_tiny] = np.arctan(dtheta[not_tiny] / dphi[not_tiny])
            else:
                ang = np.arctan(dtheta / dphi)

            weights[:] = np.vstack(
                [np.ones(nsample), np.cos(2 * ang), np.sin(2 * ang)]
            ).T
        return

    def _purge_weights(self, obs):
        """ Discard special pointing matrix and dummy signal
        """
        del obs.detdata[self.signal]
        del obs.detdata[self.weights]
        return

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if data.comm.world_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        # Get detector weights

        for obs in data.obs:
            self._get_weights(data, obs, detectors)
        
        # To accumulate, need the pixel numbers and distribution.

        create_dist_save = self.pointing.create_dist
        if self.pixel_dist in data:
            # Pixel distribution exists, we can expand pointing
            # one detector at a time
            self.pointing.create_dist = None
        else:
            self.pointing.create_dist = self.pixel_dist

        if self.pointing.create_dist:
            # Need the pixel distribution so might as well cache the pixel
            # numbers in the process.  Otherwise they get expanded twice.
            self.pointing.apply(data, detectors=detectors)
            operators = []
        else:
            # Expand pixel numbers on the fly.
            # If they already exist, the operator does nothing.
            operators = [self.pointing]

        build_zmap = BuildNoiseWeighted(
            pixel_dist=self.pixel_dist,
            zmap=self.crosslinking_map,
            view=self.pointing.view,
            pixels=self.pointing.pixels,
            weights=self.weights,
            noise_model=self.noise_model,
            det_data=self.signal,
            det_flags=self.det_flags,
            det_flag_mask=self.det_flag_mask,
            shared_flags=self.shared_flags,
            shared_flag_mask=self.shared_flag_mask,
            sync_type=self.sync_type,
        )

        operators.append(build_zmap)
        Pipeline(
            operators=operators, detector_sets=["SINGLE"],
        ).apply(data, detectors=detectors)

        self.pointing.create_dist = create_dist_save

        # Write out the results

        fname = os.path.join(self.output_dir, "crosslinking.fits")
        write_healpix_fits(
            data[self.crosslinking_map], fname, nest=self.pointing.nest
        )
        data[self.crosslinking_map].clear()
        del data[self.crosslinking_map]
                    
        for obs in data.obs:
            self._purge_weights(obs)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.pointing.detector_pointing.requires()
        return req

    def _provides(self):
        return {
        }

    def _accelerators(self):
        return list()
