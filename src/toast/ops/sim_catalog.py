# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import h5py
import healpy as hp
import numpy as np
import toml
import traitlets
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import units as u
from scipy.constants import c, h, k
from scipy.interpolate import RectBivariateSpline
from scipy.signal import fftconvolve

from .. import qarray as qa
from ..coordinates import azel_to_radec, to_DJD, to_JD, to_MJD
from ..data import Data
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import Environment, Logger, unit_conversion
from .operator import Operator


# Only the following keys will be parsed in the source catalog.
# Not all are required

SUPPORTED_KEYS = [
    "ra_deg",
    "dec_deg",
    "freqs_ghz",
    "flux_density_Jy",
    "flux_density_mJy",
    "times_mjd",
    "pol_frac",
    "pol_angle_deg",
]


@trait_docs
class SimCatalog(Operator):
    """Operator that generates variable and static point source signal.

    Signal is generated by sampling the provided beam map at appropriate
    locations and scaling the resulting signal to match the perceived
    intensity of the source.

    Source SED is convolved with the detector bandpass recorded in the
    focalplane table.

    Example catalog entries:

    .. highlight:: toml
    .. code-block:: toml

        [example_static_source]
        # Celestial coordinate are always given in degrees
        ra_deg = 30
        dec_deg = -30
        # the SED can be specified using an arbitrary number of
        # frequency bins.  The SED is interpolated in log-log space to
        # convolve with the detector bandpass
        # Use either `flux_density_mJy` or `flux_density_Jy` and adjust
        # the values accordingly
        freqs_ghz = [ 1.0, 1000.0,]
        flux_density_mJy = [ 10.0, 1.0,]
        # Omitting polarization fraction results in an
        # unpolarized source
        pol_frac = 0.1
        pol_angle_deg = 0

        [example_variable_source]
        ra_deg = 30
        dec_deg = -25
        freqs_ghz = [ 1.0, 1000.0,]
        # An arbitrary number of SED vectors can be provided but the
        # location of the frequency bins is fixed.  Effective SED is
        # interpolated between the specified epochs.
        flux_density_Jy = [ [ 10.0, 1.0,], [ 30.0, 10.0,], [ 10.0, 1.0,],]
        # Omitting the times_mjd entry resuls in a static source
        times_mjd = [ 59000.0, 60000.0, 61000.0,]
        # The polarization properties can also vary
        pol_frac = [ 0.05, 0.15, 0.05,]
        pol_angle_deg = [ 45, 45, 45,]

        [example_transient_source]
        ra_deg = 30
        dec_deg = -20
        freqs_ghz = [ 1.0, 1000.0,]
        flux_density_Jy = [ [ 10.0, 1.0,], [ 30.0, 10.0,],]
        # Difference between a variable and transient source is
        # simply that the specified epochs do not cover the entire
        # simulation time span.  The operator will not extrapolate
        # outside the epochs.
        times_mjd = [ 59410.0, 59411.0,]
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    hwp_angle = Unicode(
        defaults.hwp_angle,
        help="Observation shared key for HWP angle",
    )

    catalog_file = Unicode(
        None,
        allow_none=True,
        help="Name of the TOML catalog file",
    )

    beam_file = Unicode(
        None,
        allow_none=True,
        help="HDF5 file that stores the simulated beam. "
        "If None, a symmetric Gaussian based on the instrument model will be used.",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    det_mask = Int(
        defaults.det_mask_nonscience,
        help="Bit mask value for per-detector flagging",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector frame",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    @traitlets.validate("catalog_file")
    def _check_catalog_file(self, proposal):
        filename = proposal["value"]
        if filename is not None and not os.path.isfile(filename):
            raise traitlets.TraitError(f"Catalog file does not exist: {filename}")
        return filename

    @traitlets.validate("beam_file")
    def _check_beam_file(self, proposal):
        beam_file = proposal["value"]
        if beam_file is not None and not os.path.isfile(beam_file):
            raise traitlets.TraitError(f"{beam_file} is not a valid beam file")
        return beam_file

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
                "quats",
                "coord_in",
                "coord_out",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    @function_timer
    def _load_catalog(self):
        log = Logger.get()
        # Load the TOML into a dictionary
        with open(self.catalog_file, "r") as f:
            self.catalog = toml.loads(f.read())
        # Check that the necessary keys are defined for every source
        for source_name, source_dict in self.catalog.items():
            for key in ["ra_deg", "dec_deg", "freqs_ghz"]:
                if key not in source_dict:
                    msg = (
                        f"Catalog parsing error: '{source_name}' "
                        f"in '{self.catalog_file}' does not define '{key}'"
                    )
                    raise RuntimeError(msg)
            key1 = "flux_density_Jy"
            key2 = "flux_density_mJy"
            if key1 in source_dict and key2 in source_dict:
                msg = (
                    f"Catalog parsing error: '{source_name}' "
                    f"in '{self.catalog_file}' defines both "
                    f"'{key1}' and '{key2}'"
                )
                raise RuntimeError(msg)
            if key1 not in source_dict and key2 not in source_dict:
                msg = (
                    f"Catalog parsing error: '{source_name}' "
                    f"in '{self.catalog_file}' does not define "
                    f"'{key1}' or '{key2}'"
                )
                raise RuntimeError(msg)
        # Extra keys are allowed but produce warnings
        for source_name, source_dict in self.catalog.items():
            if key not in SUPPORTED_KEYS:
                msg = (
                    f"WARNING: '{source_name}' entry to '{self.catalog_file}'"
                    f"contains an unsupported key: '{key}'"
                )
                log.warning(msg)
        # Translate each source position into a vector for rapid
        # distance calculations
        for source_name, source_dict in self.catalog.items():
            lon = source_dict["ra_deg"]
            lat = source_dict["dec_deg"]
            source_dict["vec"] = hp.dir2vec(lon, lat, lonlat=True)
        return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store of per-detector beam properties.  Eventually we could modify the
        # operator traits to list files per detector, per wafer, per tube, etc.
        # For now, we use the same beam for all detectors, so this will have only
        # one entry.
        self.beam_props = dict()

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

        for trait in "catalog_file", "detector_pointing":
            value = getattr(self, trait)
            if value is None:
                msg = f"You must set `{trait}` before running SimCatalog"
                raise RuntimeError(msg)

        self._load_catalog()

        for obs in data.obs:
            prefix = f"{comm.group} : {obs.name}"

            # Make sure detector data output exists.  If not, create it
            # with units of Kelvin.

            dets = obs.select_local_detectors(detectors, flagmask=self.det_mask)
            exists = obs.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )
            det_units = obs.detdata[self.det_data].units
            scale = unit_conversion(u.K, det_units)

            self._observe_catalog(
                data,
                obs,
                prefix,
                dets,
                scale,
            )

        return

    @function_timer
    def _get_beam_map(self, det, focalplane):
        """
        Construct a 2-dimensional interpolator for the beam
        """
        # Read in the simulated beam.  We could add operator traits to
        # specify whether to load different beams based on detector,
        # wafer, tube, etc and check that key here.
        log = Logger.get()
        if "ALL" in self.beam_props:
            # We have already read the single beam file.
            beam_dict = self.beam_props["ALL"]
        else:
            if self.beam_file is None:
                # Use the FWHM to generate a beam dictionary
                fwhm = focalplane[det]["fwhm"]
                sigma = fwhm * gaussian_fwhm_to_sigma
                w = 2 * fwhm
                n = 101  # Should be odd to include origin
                x = np.linspace(-w, w, n)
                y = np.linspace(-w, w, n)
                X, Y = np.meshgrid(x, y)
                model = np.exp(-(X**2 + Y**2) / (2 * sigma**2)).to_value()
                beam_dict = {
                    "data": model,
                    "size": 2 * w,
                    "npix": n,
                    "res": 2 * w / (n - 1),
                }
            else:
                with h5py.File(self.beam_file, "r") as f:
                    beam_dict = {}
                    beam_dict["data"] = f["beam"][:]
                    beam_dict["size"] = f["beam"].attrs["size"] * u.degree
                    beam_dict["res"] = f["beam"].attrs["res"] * u.degree
                    beam_dict["npix"] = f["beam"].attrs["npix"]
                    self.beam_props["ALL"] = beam_dict

        model = beam_dict["data"].copy()
        model /= np.amax(model)

        # DEBUG begin
        # These commands add a tail to the beam that points towards the horizon
        # nx, ny = np.shape(model)
        # nhalf = nx // 2
        # w = 10
        # model[nhalf - w : nhalf + w + 1, 0 : nhalf] = 1
        # DEBUG end

        w = beam_dict["size"].to_value(u.rad) / 2
        n = beam_dict["npix"]
        x = np.linspace(-w, w, n)
        beam = RectBivariateSpline(x, x, model)
        # Farthest distance (corner) where beam data is available
        r = np.sqrt(w**2 + w**2)

        # Measure the solid angle using the interpolator

        x = np.linspace(-w, w, 10 * n + 1)
        dx = (x[1] - x[0]) * u.rad
        beam_solid_angle = np.sum(beam(x, x)) * dx**2

        return beam, r, beam_solid_angle

    @function_timer
    def _observe_catalog(
        self,
        data,
        obs,
        prefix,
        dets,
        scale,
    ):
        """
        Observe the catalog with each detector in tod
        """
        log = Logger.get()

        # Get a view of the data which contains just this single
        # observation
        obs_data = data.select(obs_name=obs.name)
        focalplane = obs.telescope.focalplane

        times_mjd = to_MJD(obs.shared[self.times].data)
        if self.hwp_angle in obs.shared:
            hwp_angle = obs.shared[self.hwp_angle].data
        else:
            hwp_angle = None
        beam = None

        for idet, det in enumerate(dets):
            bandpass = obs.telescope.focalplane.bandpass
            signal = obs.detdata[self.det_data][det]

            self.detector_pointing.apply(obs_data, detectors=[det])
            try:
                det_quat = obs_data.obs[0].detdata[self.detector_pointing.quats][det]
            except:
                import pdb

                pdb.set_trace()

            # Convert Az/El quaternion of the detector into angles
            # `psi` includes the rotation to the detector polarization
            # sensitive direction

            det_theta, det_phi, det_psi = qa.to_iso_angles(det_quat)
            det_vec = hp.dir2vec(det_theta, det_phi).T.copy()
            try:
                det_psi_pol = focalplane[det]["pol_angle"]
            except KeyError:
                det_psi_pol = focalplane[det]["pol_ang"]
            # gamma angle is required when dealing with a HWP
            if hwp_angle is not None:
                det_gamma = focalplane[det]["gamma"]
            else:
                det_gamma = None

            # For now, we use the first detector's beam for all detectors.
            # Will be revisited when more refined beam products become available
            if beam is None or not "ALL" in self.beam_props:
                beam, beam_radius, beam_solid_angle = self._get_beam_map(
                    det, focalplane
                )
            dp_radius = np.cos(beam_radius)

            for source_name, source_dict in self.catalog.items():
                # Is this source close enough to register?
                dp = np.dot(det_vec, source_dict["vec"])
                hit = dp > dp_radius
                nhit = np.sum(hit)
                if nhit == 0:
                    continue

                # Get the appropriate source SED and convolve with the
                # detector bandpass
                if "times_mjd" in source_dict:
                    source_times = np.array(source_dict["times_mjd"])
                    ind = np.array(np.searchsorted(source_times, times_mjd))
                    # When time stamps fall outside the period covered by
                    # source time, we assume the source went quiet
                    good = np.logical_and(ind > 0, ind < len(source_times))
                    hit *= good
                    nhit = np.sum(hit)
                    if nhit == 0:
                        # This source is not active during our observation
                        continue
                    ind = ind[hit]
                    lengths = source_times[ind] - source_times[ind - 1]
                    right_weights = (source_times[ind] - times_mjd[hit]) / lengths
                    left_weights = 1 - right_weights
                    # useful shorthands
                    freq = np.array(source_dict["freqs_ghz"]) * u.GHz
                    if "flux_density_Jy" in source_dict:
                        seds = np.array(source_dict["flux_density_Jy"]) * u.Jy
                    elif "flux_density_mJy" in source_dict:
                        seds = np.array(source_dict["flux_density_mJy"]) * u.mJy
                    else:
                        msg = f"No flux density for {source_name}"
                        raise RuntimeError(msg)
                    # Mean SED used for bandpass convolution
                    wright = np.mean(right_weights)
                    wleft = 1 - wright
                    cindex = int(np.median(ind))
                    sed_mean = wleft * seds[cindex - 1] + wright * seds[cindex]
                    # Time-dependent amplitude to scale the mean SED
                    cfreq = bandpass.center_frequency(det, alpha=-1)
                    amplitudes = []
                    for sed in seds:
                        # Interpolate the SED to the detector central frequency
                        # in log-log domain where power-law spectra are
                        # linear
                        amp = np.exp(
                            np.interp(
                                np.log(cfreq.to_value(u.GHz)),
                                np.log(freq.to_value(u.GHz)),
                                np.log(sed.to_value(u.Jy)),
                            )
                        )
                        amplitudes.append(amp)
                    amplitudes = np.array(amplitudes)
                    # This is the time-dependent amplitude relative to
                    # sed_mean
                    amplitude = (
                        left_weights * amplitudes[ind - 1]
                        + right_weights * amplitudes[ind]
                    )
                    amplitude /= (
                        wleft * amplitudes[cindex - 1] + wright * amplitudes[cindex]
                    )
                    if "pol_frac" in source_dict:
                        pol_fracs = np.array(source_dict["pol_frac"])
                        pol_frac = (
                            left_weights * pol_fracs[ind - 1]
                            + right_weights * pol_fracs[ind]
                        )
                        pol_angles = np.unwrap(np.radians(source_dict["pol_angle_deg"]))
                        pol_angle = np.array(
                            left_weights * pol_angles[ind - 1]
                            + right_weights * pol_angles[ind]
                        )
                    else:
                        pol_frac = None
                        pol_angle = None
                else:
                    freq = np.array(source_dict["freqs_ghz"]) * u.GHz
                    if "flux_density_Jy" in source_dict:
                        sed_mean = np.array(source_dict["flux_density_Jy"]) * u.Jy
                    elif "flux_density_mJy" in source_dict:
                        sed_mean = np.array(source_dict["flux_density_mJy"]) * u.mJy
                    else:
                        import pdb

                        pdb.set_trace()
                        msg = f"No flux density for {source_name}"
                        raise RuntimeError(msg)
                    if "pol_frac" in source_dict:
                        pol_frac = np.array(source_dict["pol_frac"])
                        pol_angle = np.radians(source_dict["pol_angle_deg"])
                    else:
                        pol_frac = None
                        pol_angle = None
                    amplitude = 1

                # Convolve the SED with the detector bandpass
                flux_density = bandpass.convolve(
                    det,
                    freq,
                    sed_mean.to_value(u.Jy),
                )

                # Convert the flux density to peak temperature
                temperature = (
                    flux_density
                    / beam_solid_angle.to_value(u.rad**2)
                    / bandpass.kcmb2jysr(det)
                )

                # Modulate the temperature in time
                temperature = temperature * amplitude

                # modulate temperature by polarization
                if pol_frac is not None:
                    Q = temperature * pol_frac * np.cos(2 * pol_angle)
                    U = temperature * pol_frac * np.sin(2 * pol_angle)
                    psi = det_psi[hit]
                    if hwp_angle is not None:
                        psi = 2 * (det_gamma.to_value(u.rad) - hwp_angle[hit]) - psi
                        # COSMO convention, note the sign for U
                        temperature += Q * np.cos(2 * psi) - U * np.sin(2 * psi)
                    else:
                        # COSMO convention, note the sign for U
                        temperature += Q * np.cos(2 * psi) + U * np.sin(2 * psi)

                # Interpolate the beam map at appropriate locations
                source_theta = np.radians(90 - source_dict["dec_deg"])
                source_phi = np.radians(source_dict["ra_deg"])
                phi_diff = (det_phi[hit] - source_phi + np.pi) % (2 * np.pi) - np.pi
                x = phi_diff * np.cos(np.pi / 2 - det_theta[hit])
                y = det_theta[hit] - source_theta
                # Rotate into the beam frame
                psi = det_psi[hit] - det_psi_pol.to_value(u.rad)
                x_beam = np.cos(psi) * x - np.sin(psi) * y
                y_beam = np.sin(psi) * x + np.cos(psi) * y
                sig = beam(x_beam, y_beam, grid=False) * temperature
                signal[hit] += scale * sig

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [
                self.times,
            ],
        }
        return req

    def _provides(self):
        return {
            "detdata": [
                self.det_data,
            ]
        }

    def _accelerators(self):
        return list()
