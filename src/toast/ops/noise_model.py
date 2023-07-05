# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import types

import numpy as np
import traitlets
from astropy import units as u
from scipy.optimize import Bounds, curve_fit, least_squares

from ..noise import Noise
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Float, Int, Quantity, Unicode, trait_docs
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

        noise_keys = set(["psd_fmin", "psd_fknee", "psd_alpha", "psd_net"])

        for ob in data.obs:
            fp_data = ob.telescope.focalplane.detector_data
            has_parameters = False
            for key in noise_keys:
                if key not in fp_data.colnames:
                    break
            else:
                has_parameters = True
            if not has_parameters:
                msg = f"Observation {ob.name} does not have a focalplane with "
                msg += "noise parameters.  Skipping."
                log.warning(msg)
                ob[self.noise_model] = None
                continue

            local_dets = set(ob.local_detectors)

            dets = []
            fmin = {}
            fknee = {}
            alpha = {}
            NET = {}
            rates = {}
            indices = {}

            for row in fp_data:
                name = row["name"]
                if name not in local_dets:
                    continue
                dets.append(name)
                rates[name] = ob.telescope.focalplane.sample_rate
                fmin[name] = row["psd_fmin"]
                fknee[name] = row["psd_fknee"]
                alpha[name] = row["psd_alpha"]
                NET[name] = row["psd_net"]
                indices[name] = row["uid"]

            ob[self.noise_model] = AnalyticNoise(
                rate=rates,
                fmin=fmin,
                detectors=dets,
                fknee=fknee,
                alpha=alpha,
                NET=NET,
                indices=indices,
            )

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

    If the data has been filtered with a low-pass, then the high frequency spectral
    points are not representative of the actual white noise plateau.  In this case,
    The min / max frequencies to consider can be specified.

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

    white_noise_min = Quantity(
        None,
        allow_none=True,
        help="The minimum frequency to consider for the white noise plateau",
    )

    white_noise_max = Quantity(
        None,
        allow_none=True,
        help="The maximum frequency to consider for the white noise plateau",
    )

    least_squares_xtol = Float(
        None,
        allow_none=True,
        help="The xtol value passed to the least_squares solver",
    )

    least_squares_ftol = Float(
        1.0e-10,
        allow_none=True,
        help="The ftol value passed to the least_squares solver",
    )

    least_squares_gtol = Float(
        None,
        allow_none=True,
        help="The gtol value passed to the least_squares solver",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if detectors is not None:
            msg = "FitNoiseModel will fit all detectors- ignoring input detector list"
            log.warning(msg)

        if self.white_noise_max is not None:
            # Ensure that the min is also set
            if self.white_noise_min is None:
                msg = "You must set both of the min / max values or none of them"
                raise RuntimeError(msg)

        for ob in data.obs:
            in_model = ob[self.noise_model]
            # We will use the best fit parameters from each detector as
            # the starting guess for the next detector.
            params = None
            nse_rate = dict()
            nse_fmin = dict()
            nse_fknee = dict()
            nse_alpha = dict()
            nse_NET = dict()
            nse_indx = dict()
            for det in ob.local_detectors:
                freqs = in_model.freq(det)
                in_psd = in_model.psd(det)
                props = self._fit_log_psd(freqs, in_psd, guess=params)
                if props["fit_result"].success:
                    # This was a good fit
                    params = props["fit_result"].x
                else:
                    msg = f"FitNoiseModel observation {ob.name}, det {det} failed, "
                    msg += f"using white noise with NET = {props['NET']}"
                    log.warning(msg)
                    msg = f"  Best Result = {props['fit_result']}"
                    log.verbose(msg)
                nse_indx[det] = in_model.index(det)
                nse_rate[det] = 2.0 * freqs[-1]
                nse_fmin[det] = props["fmin"]
                nse_fknee[det] = props["fknee"]
                nse_alpha[det] = props["alpha"]
                nse_NET[det] = props["NET"]

            new_model = AnalyticNoise(
                detectors=ob.local_detectors,
                rate=nse_rate,
                fmin=nse_fmin,
                fknee=nse_fknee,
                alpha=nse_alpha,
                NET=nse_NET,
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
        resid = current - logdata
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
            (dict):  Dictionary of fit parameters

        """
        log = Logger.get()
        psd_unit = data.unit
        ret = dict()

        def _get_err_ret():
            eret = dict()
            eret["fit_result"] = types.SimpleNamespace()
            eret["fit_result"].success = False
            eret["NET"] = 0.0 * np.sqrt(1.0 * psd_unit)
            eret["fmin"] = 0.0 * u.Hz
            eret["fknee"] = 0.0 * u.Hz
            eret["alpha"] = 0.0
            return eret

        # We cut the lowest frequency bin value, and any leading negative values,
        # since these are usually due to poor estimation.  If the user has specified
        # a maximum frequency for the white noise plateau, then we also stop our
        # fit at that point.
        raw_freqs = freqs.to_value(u.Hz)
        raw_data = data.value
        n_raw = len(raw_data)
        n_skip = 1
        while n_skip < n_raw and raw_data[n_skip] <= 0:
            n_skip += 1
        if n_skip == n_raw:
            msg = f"All {n_raw} PSD values were negative.  Giving up."
            log.warning(msg)
            ret = _get_err_ret()
            return ret

        n_trim = 0
        if self.white_noise_max is not None:
            max_hz = self.white_noise_max.to_value(u.Hz)
            for f in raw_freqs:
                if f > max_hz:
                    n_trim += 1

        if n_skip + n_trim >= n_raw:
            msg = f"All {n_raw} PSD values either negative or above plateau."
            log.warning(msg)
            ret = _get_err_ret()
            return ret

        input_freqs = raw_freqs[n_skip : n_raw - n_trim]
        input_data = raw_data[n_skip : n_raw - n_trim]
        # Force all points to be positive
        bad = input_data <= 0
        n_bad = np.count_nonzero(bad)
        if n_bad > 0:
            msg = "Some PSDs have negative values.  Consider changing "
            msg += "noise estimation parameters."
            log.warning(msg)
        input_data[bad] = 1.0e-6
        input_log_data = np.log(input_data)

        # print(f"FIT: input {input_freqs} {input_data} {input_log_data}")

        raw_fmin = self.f_min.to_value(u.Hz)

        if self.white_noise_max is None:
            net = self._estimate_net(input_freqs, input_data)
            # print(f"FIT:  estimated NET = {net}")
        else:
            plateau_samples = np.logical_and(
                (input_freqs > self.white_noise_min.to_value(u.Hz)),
                (input_freqs < self.white_noise_max.to_value(u.Hz)),
            )
            net = np.sqrt(np.mean(input_data[plateau_samples]))
            # print(f"FIT:  NET from plateau = {net}")

        midfreq = 0.5 * input_freqs[-1]

        bounds = (
            np.array([input_freqs[0], 0.1]),
            np.array([input_freqs[-1], 10.0]),
        )
        x_0 = guess
        if x_0 is None:
            x_0 = np.array([midfreq, 1.0])

        # print(f"FIT:  starting guess = {x_0}")

        result = least_squares(
            self._fit_log_fun,
            x_0,
            jac=self._fit_log_jac,
            bounds=bounds,
            xtol=self.least_squares_xtol,
            gtol=self.least_squares_gtol,
            ftol=self.least_squares_ftol,
            max_nfev=500,
            verbose=0,
            kwargs={
                "freqs": input_freqs,
                "logdata": input_log_data,
                "fmin": raw_fmin,
                "net": net,
            },
        )

        # print(f"FIT:  [{n_skip}:{n_raw}-{n_trim}] {result}")

        ret["fit_result"] = result
        ret["NET"] = net * np.sqrt(1.0 * psd_unit)
        ret["fmin"] = self.f_min
        if result.success:
            ret["fknee"] = result.x[0] * u.Hz
            ret["alpha"] = result.x[1]
        else:
            ret["fknee"] = 0.0 * u.Hz
            ret["alpha"] = 1.0

        # print(f"FIT ret = {ret}")

        log.verbose(f"PSD fit NET={net}, bounds={bounds}, guess={x_0}, result={result}")
        return ret

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return dict()

    def _provides(self):
        prov = {"meta": [self.noise_model]}
        return prov


@trait_docs
class FlagNoiseFit(Operator):
    """Operator which flags detectors that have outlier noise properties."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    noise_model = Unicode(
        "noise_model", help="The observation key containing the noise model"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_processing, help="Bit mask to raise flags with"
    )

    sigma_NET = Float(5.0, help="Flag detectors with NET values outside this range")

    sigma_fknee = Float(
        None,
        allow_none=True,
        help="Flag detectors with knee frequency values outside this range",
    )

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.det_flags is None:
            raise RuntimeError("You must set det_flags before calling exec()")

        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            if self.noise_model not in obs:
                msg = f"Observation {obs.name} does not contain noise model {self.noise_model}"
                raise RuntimeError(msg)

            exists = obs.detdata.ensure(self.det_flags, dtype=np.uint8, detectors=dets)

            model = obs[self.noise_model]

            local_net = list()
            local_fknee = list()

            # If we have an analytic noise model from a simulation or fit, then we can access
            # the properties directly.  If not, we will use the detector weight as a proxy
            # for the NET and make a crude estimate of the knee frequency.

            for det in dets:
                try:
                    NET = model.NET(det)
                except AttributeError:
                    wt = model.detector_weight(det)
                    NET = np.sqrt(1.0 / (wt * model.rate(det)))
                local_net.append(NET.to_value(u.K * np.sqrt(1.0 * u.second)))
                if self.sigma_fknee is not None:
                    try:
                        fknee = model.fknee(det)
                        local_fknee.append(fknee.to_value(u.Hz))
                    except AttributeError:
                        msg = f"Observation {obs.name}, noise model {self.noise_model} "
                        msg += "has no f_knee estimate.  Use FitNoiseModel before flagging."

            # Gather data across the process column

            net_mean = None
            net_std = None
            fknee_mean = None
            fknee_std = None
            if obs.comm_col is None:
                all_net = np.array(local_net)
                net_mean = np.mean(all_net)
                net_std = np.std(all_net)
                if self.sigma_fknee is not None:
                    all_fknee = np.array(local_fknee)
                    fknee_mean = np.mean(all_fknee)
                    fknee_std = np.std(all_fknee)
            else:
                all_net = obs.comm_col.gather(local_net, root=0)
                if obs.comm_col_rank == 0:
                    all_net = np.array([val for plist in all_net for val in plist])
                    net_mean = np.mean(all_net)
                    net_std = np.std(all_net)
                net_mean = obs.comm_col.bcast(net_mean, root=0)
                net_std = obs.comm_col.bcast(net_std, root=0)
                if self.sigma_fknee is not None:
                    all_fknee = obs.comm_col.gather(local_fknee, root=0)
                    if obs.comm_col_rank == 0:
                        all_fknee = np.array(
                            [val for plist in all_fknee for val in plist]
                        )
                        fknee_mean = np.mean(all_fknee)
                        fknee_std = np.std(all_fknee)
                    fknee_mean = obs.comm_col.bcast(fknee_mean, root=0)
                    fknee_std = obs.comm_col.bcast(fknee_std, root=0)

            # Flag outlier detectors

            local_cut_NET = list()
            local_cut_fknee = list()
            new_flags = dict()
            for idet, det in enumerate(dets):
                if np.absolute(local_net[idet] - net_mean) > net_std * self.sigma_NET:
                    msg = f"obs {obs.name}, det {det} has NET {local_net[idet]} "
                    msg += f" that is > {self.sigma_NET} x {net_std} from {net_mean}"
                    log.info(msg)
                    obs.detdata[self.det_flags][det, :] |= self.det_flag_mask
                    new_flags[det] = self.det_flag_mask
                if self.sigma_fknee is not None:
                    if (
                        np.absolute(local_fknee[idet] - fknee_mean)
                        > fknee_std * self.sigma_fknee
                    ):
                        msg = f"obs {obs.name}, det {det} has fknee "
                        msg += f"{local_fknee[idet]} that is > {self.sigma_fknee} "
                        msg += f"x {fknee_std} from {fknee_mean}"
                        log.info(msg)
                        obs.detdata[self.det_flags][det, :] |= self.det_flag_mask
                        new_flags[det] = self.det_flag_mask
            obs.update_local_detector_flags(new_flags)

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [
                self.noise_model,
            ],
            "shared": [],
            "detdata": [],
            "intervals": [],
        }
        return req

    def _provides(self):
        prov = {
            "meta": [],
            "shared": [],
            "detdata": [self.det_flags],
        }
        return prov
