# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u
from scipy.optimize import Bounds, least_squares

from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@trait_docs
class DefaultNoiseModel(Operator):
    """Create a default noise model from focalplane parameters.

    A noise model is used by other operations such as simulating noise timestreams
    and also map making.  This operator uses the detector properties from the
    focalplane in each observation to create a simple AnalyticNoise model.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="The observation key for storing the noise model"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            if ob.telescope.focalplane.noise is None:
                raise RuntimeError("Focalplane does not have a noise model")
            ob[self.noise_model] = ob.telescope.focalplane.noise

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov


@trait_docs
class FitNoiseModel(Operator):
    """Perform a least squares fit to an existing noise model.

    This takes an existing estimated noise model and attempts to fit each
    spectrum to 1/f parameters.

    If the output model is not specified, then the input is modified in place.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="The observation key containing the input noise model"
    )

    out_model = Unicode(
        None, allow_none=True, help="Create a new noise model with this name"
    )

    f_min = Quantity(1.0e-5 * u.Hz, help="Low-frequency rolloff of model in the fit")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            msg = "FitNoiseModel will fit all detectors- ignoring input detector list"
            log.warning(msg)

        for ob in data.obs:
            in_model = ob[self.noise_model]
            # We will use the best fit parameters from each detector as
            # the starting guess for the next detector.
            params = None
            nse_freqs = dict()
            nse_psds = dict()
            for det in ob.local_detectors:
                freqs = in_model.freq(det)
                in_psd = in_model.psd(det)
                fitted, result = self._fit_psd(freqs, in_psd, params)
                if result.success:
                    params = result.x
                nse_freqs[det] = freqs
                nse_psds[det] = fitted
            if ob.comm.comm_group is not None:
                all_nse_freqs = ob.comm.comm_group.gather(nse_freqs, root=0)
                all_nse_psds = ob.comm.comm_group.gather(nse_psds, root=0)
                nse_indx = None
                if ob.comm.group_rank == 0:
                    nse_freqs = dict()
                    nse_psds = dict()
                    for pval in all_nse_freqs:
                        nse_freqs.update(pval)
                    for pval in all_nse_psds:
                        nse_psds.update(pval)
                    nse_indx = dict()
                    for det in in_model.detectors:
                        nse_indx[det] = in_model.index(det)
                nse_indx = ob.comm.comm_group.bcast(nse_indx, root=0)
                nse_freqs = ob.comm.comm_group.bcast(nse_freqs, root=0)
                nse_psds = ob.comm.comm_group.bcast(nse_psds, root=0)
            else:
                nse_indx = dict()
                for det in in_model.detectors:
                    nse_indx[det] = in_model.index(det)

            new_model = Noise(
                detectors=in_model.detectors,
                freqs=nse_freqs,
                psds=nse_psds,
                indices=nse_indx,
            )

            if self.out_model is None or self.noise_model == self.out_model:
                # We are replacing the input
                del ob[self.noise_model]
                ob[self.noise_model] = new_model
            else:
                # We are storing this in a new key
                ob[self.out_model] = new_model
        return

    def _evaluate_model(self, freqs, fmin, net, fknee, alpha):
        ktemp = np.power(fknee, alpha)
        mtemp = np.power(fmin, alpha)
        temp = np.power(freqs, alpha)
        psd = (temp + ktemp) / (temp + mtemp)
        psd *= net**2
        return psd

    def _fit_fun(self, x, *args, **kwargs):
        net = x[0]
        fknee = x[1]
        alpha = x[2]
        freqs = kwargs["freqs"]
        data = kwargs["data"]
        fmin = kwargs["fmin"]
        current = self._evaluate_model(freqs, fmin, net, fknee, alpha)
        # We weight the residual so that the high-frequency values specifying
        # the white noise plateau / NET are more important.  Also the lowest
        # estimated bin is usually garbage.  We arbitrarily choose a weighting
        # of:
        #
        #  W = f_nyquist - (f_nyquist / (1 + f^2))
        #
        weights = np.ones_like(current) * freqs[-1]
        weights -= freqs[-1] / (1.0 + freqs**2)
        resid = np.multiply(weights, current - data)
        return resid

    def _fit_psd(self, freqs, data, guess=None):
        psd_unit = data.unit
        raw_freqs = freqs.to_value(u.Hz)
        raw_data = data.value
        raw_fmin = self.f_min.to_value(u.Hz)
        x_0 = guess
        if x_0 is None:
            x_0 = np.array([1.0, 0.1, 1.0])
        result = least_squares(
            self._fit_fun,
            x_0,
            kwargs={"freqs": raw_freqs, "data": raw_data, "fmin": raw_fmin},
        )
        fit_data = data
        if result.success:
            fit_data = (
                self._evaluate_model(
                    raw_freqs, raw_fmin, result.x[0], result.x[1], result.x[2]
                )
                * psd_unit
            )
        return fit_data, result

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov
