# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from ..accelerator import ImplementationType
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Instance, Int, Unicode, trait_docs
from ..utils import Environment, Logger
from ..qarray import mult, to_iso_angles
from .operator import Operator

@trait_docs
class DerivativesWeights(Operator):
    """Operator which generates pointing weights for I and derivatives of I with
    respect to theta and phi, to order 1 (mode dI) or 2 (mode d2I).

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.  By definition, the detector coordinate frame has the X-axis
    aligned with the polarization sensitive direction.  An optional dictionary of
    beam error factors may be specified for each observation.
    
    These factors are an overall calibration factor cal, beam centroid error dx/dy,
    differential beam fwhm dsigma, and ellipticity dp/dc. Since we are focused
    on total intensity, there is no HWP term or detector polarisation efficiency.

    If the hwp_angle field is specified, then an ideal HWP Mueller matrix is inserted
    in the optics chain before the linear polarizer.  In this case, the fp_gamma key
    name must be specified and each detector must have a value in the focalplane
    table.

    The timestream model without a HWP in COSMO convention is:

    .. math::
        d = cal \\left[I + \\frac{1 - \\epsilon}{1 + \\epsilon} \\left[Q \\cos\\left(2\\alpha\\right) + U \\sin\\left(2\\alpha\\right) \\right] \\right]

    When a HWP is present, we have:

    .. math::
        d = cal \\left[I + \\frac{1 - \\epsilon}{1 + \\epsilon} \\left[Q \\cos\\left(2(\\alpha - 2\\omega) \\right) - U \\sin\\left(2(\\alpha - 2\\omega) \\right) \\right] \\right]

    The detector orientation angle "alpha" in COSMO convention is measured in a
    right-handed sense from the local meridian and the HWP angle "omega" is also
    measured from the local meridian.  The omega value can be described in terms of
    alpha, a fixed per-detector offset gamma, and the time varying HWP angle measured
    from the focalplane coordinate frame X-axis:

    .. math::
        \\omega = \\alpha + {\\gamma}_{HWP}(t) - {\\gamma}_{DET}

    See documentation for a full treatment of this math.

    By default, this operator uses the "COSMO" convention for Q/U.  If the "IAU" trait
    is set to True, then resulting weights will differ by the sign of the U Stokes
    weight.

    If the view trait is not specified, then this operator will use the same data
    view as the detector pointing operator when computing the pointing matrix pixels
    and weights.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    mode = Unicode("dI", help="The Stokes weights to generate (dI or d2I)")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    weights = Unicode(
        defaults.weights, help="Observation detdata key for output weights"
    )

    single_precision = Bool(False, help="If True, use 32bit float in output")

    cal = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of pointing weight "
        "calibration for each det",
    )
    
    dx = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of detector offset "
        "for each det in the x direction",
    )
    
    dy = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of detector offset "
        "for each det in the y direction",
    )

    dsigma = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of detector fwhm error for each det",
    )
        
    dp = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of detector ellipticity "
        "for each det in one direction",
    )
    
    dc = Unicode(
        None,
        allow_none=True,
        help="The observation key with a dictionary of detector ellipticity "
        "for each det in the other direction",
    )

    IAU = Bool(False, help="If True, use the IAU convention rather than COSMO")

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
                "det_mask",
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @traitlets.validate("mode")
    def _check_mode(self, proposal):
        check = proposal["value"]
        if check not in ["dI", "d2I"]:
            raise traitlets.TraitError("Invalid mode (must be 'dI' or 'd2I')")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=None, **kwargs):
        env = Environment.get()
        log = Logger.get()

        self._nnz = 6

        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)

        if self.detector_pointing is None:
            raise RuntimeError("The detector_pointing trait must be set")

        # Expand detector pointing
        quats_name = self.detector_pointing.quats

        view = self.view
        if view is None:
            # Use the same data view as detector pointing
            view = self.detector_pointing.view

        # Expand detector pointing
        self.detector_pointing.apply(data, detectors=detectors, use_accel=use_accel)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(
                detectors, flagmask=self.detector_pointing.det_mask
            )
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

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
                        msg = (
                            f"view {self.view} is not fully covered by valid "
                            "detector pointing"
                        )
                        raise RuntimeError(msg)

            # Create (or re-use) output data for the weights
            if self.single_precision:
                exists = ob.detdata.ensure(
                    self.weights,
                    sample_shape=(self._nnz,),
                    dtype=np.float32,
                    detectors=dets,
                    accel=use_accel,
                )
            else:
                exists = ob.detdata.ensure(
                    self.weights,
                    sample_shape=(self._nnz,),
                    dtype=np.float64,
                    detectors=dets,
                    accel=use_accel,
                )

            quat_indx = ob.detdata[quats_name].indices(dets)
            weight_indx = ob.detdata[self.weights].indices(dets)
            # Do we already have pointing for all requested detectors?
            if exists:
                # Yes
                if data.comm.group_rank == 0:
                    msg = (
                        f"Group {data.comm.group}, ob {ob.name}, Stokes weights "
                        f"already computed for {dets}"
                    )
                    log.verbose(msg)
                continue

            # FIXME:  temporary hack until instrument classes are also pre-staged
            # to GPU
            focalplane = ob.telescope.focalplane
            det_qoff = np.zeros((len(dets), 4), dtype=np.float64)
            # Get the boresight offsets from the focal plane
            for idet, d in enumerate(dets):
                det_qoff[idet] = focalplane[d]["quat"]
            qbore = ob.shared["boresight_radec"]
            
            nsamp = len(qbore)
            ndets = len(det_qoff)
            det_qoff = np.stack([det_qoff for _ in range(nsamp)], axis=1)
            qbore = np.stack([qbore for _ in range(ndets)], axis=0)
            theta, _, psi = to_iso_angles(mult(qbore, det_qoff))

            # Get the per-detector calibration
            if self.cal is None:
                cal = np.array([1.0 for x in dets], np.float64)
            else:
                cal = np.array([ob[self.cal][x] for x in dets], np.float64)
            cal = np.stack([cal for _ in range(nsamp)], axis=1)
            # Per-detector pointing error
            if self.dx is None:
                dx = np.array([0.0 for x in dets], np.float64)
            else:
                dx = np.array([ob[self.dx][x] for x in dets], np.float64)
            if self.dy is None:
                dy = np.array([0.0 for x in dets], np.float64)
            else:
                dy = np.array([ob[self.dy][x] for x in dets], np.float64)
            dx = np.stack([dx for _ in range(nsamp)], axis=1)
            dy = np.stack([dy for _ in range(nsamp)], axis=1)
            #Per-detector fwhm/sigma error    
            if self.dsigma is None:
                dsigma = np.array([0.0 for x in dets], np.float64)
            else:
                dsigma = np.array([ob[self.dsigma][x] for x in dets], np.float64)
            dsigma = np.stack([dsigma for _ in range(nsamp)], axis=1)
            #Per-detector ellipticity
            if self.dp is None:
                dp = np.array([0.0 for x in dets], np.float64)
            else:
                dp = np.array([ob[self.dp][x] for x in dets], np.float64)
            if self.dc is None:
                dc = np.array([0.0 for x in dets], np.float64)
            else:
                dc = np.array([ob[self.dc][x] for x in dets], np.float64)
            dp = np.stack([dp for _ in range(nsamp)], axis=1)
            dc = np.stack([dc for _ in range(nsamp)], axis=1)
            
            
            wc = np.cos(psi)
            wc2 = np.cos(2*psi)
            ws = np.sin(psi)
            ws2 = np.sin(2*psi)
            inv_tan_theta = np.cos(theta)/np.sin(theta)
            
            if self.mode == "d2I":
                weights = np.empty((ndets,nsamp,6))
                weights[:,:,3] = dsigma + dp * wc2 - dc * ws2 #d2theta
                weights[:,:,4] = -2.0 * dp * ws2 + 2.0 * dc * wc2 #dphi dtheta
                weights[:,:,5] = dsigma + dp * wc2 + dc * ws2 #dphi2
            else:
                weights = np.empty((ndets, nsamp,3))
            
            weights[:,:,0] = cal # gain error
            weights[:,:,1] = dx * ws - dy * wc #dtheta
            weights[:,:,2] = -dx * wc - dy * ws + (dp * ws2 - dc * wc2) * inv_tan_theta #dphi
            ob.detdata[self.weights][dets,:] = weights
            
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        if "detdata" not in req:
            req["detdata"] = list()
        req["detdata"].append(self.weights)
        if self.cal is not None:
            req["meta"].append(self.cal)
        if self.hwp_angle is not None:
            req["shared"].append(self.hwp_angle)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.weights)
        return prov

    def _implementations(self):
        return [
            ImplementationType.DEFAULT,
            ImplementationType.COMPILED,
            ImplementationType.NUMPY,
            ImplementationType.JAX,
        ]

    def _supports_accel(self):
        if (self.detector_pointing is not None) and (
            self.detector_pointing.supports_accel()
        ):
            return True
        else:
            return False
