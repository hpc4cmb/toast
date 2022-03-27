# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u
from scipy.optimize import curve_fit, least_squares

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
                fitted, result = self._fit_log_psd(freqs, in_psd, guess=params)
                if result.success:
                    # This was a good fit
                    params = result.x
                else:
                    msg = f"FitNoiseModel observation {ob.name}, det {det} failed."
                    log.warning(msg)
                    msg = f"  Best Result = {result}"
                    log.verbose(msg)
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

    def _estimate_net(self, freqs, data):
        """Estimate the NET from the high frequency PSD.

        This assumes that at high frequency the PSD has a white noise "plateau".  A simple
        parabola is fit to the last bit of the spectrum and this is used to compute the
        NET.

        Args:
            freqs (array):  The frequency values in Hz
            data (array):  The PSD in arbitrary units

        Returns:
            (float):  The estimated NET.

        """
        log = Logger.get()

        def quad_func(x, a, b, c):
            # Parabola
            return a * (x - b) ** 2 + c

        def lin_func(x, a, b, c):
            # Line
            return a * (x - b) + c

        n_psd = len(data)
        offset = int(0.8 * n_psd)
        try_quad = True
        if n_psd - offset < 10:
            # Too few points
            try_quad = False
            if n_psd < 10:
                # Crazy...
                offset = 0
            else:
                offset = n_psd - 10

        ffreq = np.log(freqs[offset:])
        fdata = np.log(data[offset:])
        if try_quad:
            try:
                params, params_cov = curve_fit(
                    quad_func, ffreq, fdata, p0=[1.0, ffreq[-1], fdata[-1]]
                )
                # It worked!
                fdata = quad_func(ffreq, params[0], params[1], params[2])
                fdata = np.exp(fdata)
                return np.sqrt(fdata[-1])
            except RuntimeError:
                pass

        params, params_cov = curve_fit(
            lin_func, ffreq, fdata, p0=[0.0, ffreq[-1], fdata[-1]]
        )
        fdata = lin_func(ffreq, params[0], params[1], params[2])
        fdata = np.exp(fdata)
        net = np.sqrt(fdata[-1])
        return net

    def _evaluate_model(self, freqs, fmin, net, fknee, alpha):
        """Evaluate the noise model

        Given the input frequencies, NET, slope alpha, f_min and f_knee,
        evaluate the PSD as:

        PSD = NET^2 * [ (f^alpha + f_knee^alpha) / (f^alpha + f_min^alpha) ]

        Args:
            freqs (array):  The input frequencies in Hz
            fmin (float):  The extreme low-frequency rolloff
            fknee (float):  The knee frequency
            alpha (float):  The slope parameter

        Returns:
            (array):  The model PSD

        """
        ktemp = np.power(fknee, alpha)
        mtemp = np.power(fmin, alpha)
        temp = np.power(freqs, alpha)
        psd = (temp + ktemp) / (temp + mtemp)
        psd *= net**2
        return psd

    def _evaluate_log_model(self, freqs, fmin, net, fknee, alpha):
        """Evaluate the natural log of the noise model

        Given the input frequencies, NET, slope alpha, f_min and f_knee,
        evaluate the ln(PSD) as:

        ln(PSD) = 2 * ln(NET) + ln(f^alpha + f_knee^alpha) - ln(f^alpha + f_min^alpha)

        Args:
            freqs (array):  The input frequencies in Hz
            fmin (float):  The extreme low-frequency rolloff
            fknee (float):  The knee frequency
            alpha (float):  The slope parameter

        Returns:
            (array):  The log of the model PSD

        """
        f_alpha = np.power(freqs, alpha)
        fknee_alpha = np.power(fknee, alpha)
        fmin_alpha = np.power(fmin, alpha)
        psd = (
            2.0 * np.log(net)
            + np.log(f_alpha + fknee_alpha)
            - np.log(f_alpha + fmin_alpha)
        )
        return psd

    def _fit_log_fun(self, x, *args, **kwargs):
        """Evaluate the weighted residual in log space.

        For the given set of parameters, this evaluates the model log PSD and computes the
        residual from the real data.  This residual is further weighted so that the better
        constrained high-frequency values have more significance.  We arbitrarily choose a
        weighting of:

            W = f_nyquist - (f_nyquist / (1 + f^2))

        Args:
            x (array):  The current model parameters
            kwargs:  The fixed information is passed in through the least squares solver.

        Returns:
            (array):  The array of residuals

        """
        freqs = kwargs["freqs"]
        logdata = kwargs["logdata"]
        fmin = kwargs["fmin"]
        net = kwargs["net"]
        fknee = x[0]
        alpha = x[1]
        current = self._evaluate_log_model(freqs, fmin, net, fknee, alpha)

        # Weight the difference so that low frequencies do not impact the fit.  This is
        # basically a high-pass butterworth.
        n_freq = len(freqs)
        hp = np.arange(n_freq, dtype=np.float64)
        hp *= 2.0 / n_freq
        weights = 0.1 + 2.0 / np.sqrt(1.0 + np.power(hp, -4))
        resid = np.multiply(weights, current - logdata)
        # print(
        #     f"      current-data = {current - logdata}, weights = {weights}, resid = {resid}"
        # )
        return resid

    def _fit_log_jac(self, x, *args, **kwargs):
        """Evaluate the partial derivatives of model.

        This returns the Jacobian containing the partial derivatives of the log-space
        model with respect to the fit parameters.

        Args:
            x (array):  The current model parameters
            kwargs:  The fixed information is passed in through the least squares solver.

        Returns:
            (array):  The Jacobian

        """
        freqs = kwargs["freqs"]
        fmin = kwargs["fmin"]
        fknee = x[0]
        alpha = x[1]
        n_freq = len(freqs)

        log_freqs = np.log(freqs)
        f_alpha = np.power(freqs, alpha)
        fknee_alpha = np.power(fknee, alpha)
        fmin_alpha = np.power(fmin, alpha)

        fkalpha = f_alpha + fknee_alpha
        fmalpha = f_alpha + fmin_alpha

        J = np.empty((n_freq, x.size), dtype=np.float64)

        # Partial derivative wrt f_knee
        J[:, 0] = alpha * np.power(fknee, alpha - 1.0) / fkalpha

        # Partial derivative wrt alpha
        J[:, 1] = (f_alpha * log_freqs + fknee_alpha * np.log(fknee)) / fkalpha - (
            f_alpha * log_freqs + fmin_alpha * np.log(fmin)
        ) / fmalpha
        return J

    def _fit_log_psd(self, freqs, data, guess=None):
        """Perform a log-space fit to model PSD parameters.

        Args:
            freqs (Quantity):  The frequency values
            data (Quantity):  The estimated input PSD
            guess (array):  Optional starting point guess

        Returns:
            (array):  The best fit PSD model or the input if the fit fails

        """
        log = Logger.get()
        psd_unit = data.unit

        # We cut the lowest frequency bin value, and any leading negative values,
        # since these are usually due to poor estimation.
        raw_freqs = freqs.to_value(u.Hz)
        raw_data = data.value
        n_skip = 1
        while raw_data[n_skip] <= 0:
            n_skip += 1

        input_freqs = raw_freqs[n_skip:]
        input_data = raw_data[n_skip:]
        # Force all points to be positive
        bad = input_data <= 0
        n_bad = np.count_nonzero(bad)
        if n_bad > 0:
            log.warning(
                "Some PSDs have negative values.  Change noise estimation parameters."
            )
        input_data[bad] = 1.0e-6
        input_log_data = np.log(input_data)

        raw_fmin = self.f_min.to_value(u.Hz)

        net = self._estimate_net(input_freqs, input_data)

        midfreq = 0.5 * input_freqs[-1]
        bounds = (
            [input_freqs[0] / 2.0, 0.01],
            [input_freqs[-1], 10.0],
        )
        x_0 = guess
        if x_0 is None:
            x_0 = np.array([midfreq, 1.0])

        result = least_squares(
            self._fit_log_fun,
            x_0,
            jac=self._fit_log_jac,
            bounds=bounds,
            xtol=1.0e-10,
            gtol=1.0e-10,
            ftol=1.0e-10,
            max_nfev=500,
            kwargs={
                "freqs": input_freqs,
                "logdata": input_log_data,
                "fmin": raw_fmin,
                "net": net,
            },
        )
        log.verbose(f"PSD fit NET={net}, bounds={bounds}, guess={x_0}, result={result}")
        if result.success:
            best_fit = self._evaluate_model(
                raw_freqs, raw_fmin, net, result.x[0], result.x[1]
            )
            return best_fit * psd_unit, result
        else:
            return data, result

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov
