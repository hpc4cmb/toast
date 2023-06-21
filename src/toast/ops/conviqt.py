# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import tempfile
import warnings

import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Dict, Instance, Int, Quantity, Unicode, Unit, trait_docs
from ..utils import (
    Environment,
    GlobalTimers,
    Logger,
    Timer,
    dtype_to_aligned,
    unit_conversion,
)
from .operator import Operator

conviqt = None

if use_mpi:
    try:
        import libconviqt_wrapper as conviqt
    except ImportError:
        conviqt = None


def available():
    """(bool): True if libconviqt is found in the library search path."""
    global conviqt
    return (conviqt is not None) and conviqt.available


@trait_docs
class SimConviqt(Operator):
    """Operator which uses libconviqt to generate beam-convolved timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    comm = Instance(
        klass=MPI_Comm,
        allow_none=True,
        help="MPI communicator to use for the convolution. libConviqt does "
        "not work without MPI.",
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid, help="Bit mask value for optional detector flagging"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=False,
        help="Observation detdata key for accumulating convolved timestreams",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    calibrate = Bool(
        True,
        allow_none=False,
        help="Calibrate intensity to 1.0, rather than (1 + epsilon) / 2. "
        "Calibrate has no effect if the beam is found to be normalized rather than "
        "scaled with the leakage factor.",
    )

    dxx = Bool(
        True,
        allow_none=False,
        help="The beam frame is either Dxx or Pxx. Pxx includes the rotation to "
        "polarization sensitive basis, Dxx does not. When Dxx=True, detector "
        "orientation from attitude quaternions is corrected for the polarization "
        "angle.",
    )

    pol = Bool(
        True,
        allow_none=False,
        help="Toggle simulated signal polarization",
    )

    mc = Int(
        None,
        allow_none=True,
        help="Monte Carlo index used in synthesizing the input file names.",
    )

    @traitlets.validate("mc")
    def _check_mc(self, proposal):
        check = proposal["value"]
        if check is not None and check < 0:
            raise traitlets.TraitError("MC index cannot be negative")
        return check

    beammmax = Int(
        -1,
        allow_none=False,
        help="Beam maximum m.  Actual resolution in the Healpix FITS file may differ. "
        "If not set, will use the maximum expansion order from file.",
    )

    lmax = Int(
        -1,
        allow_none=False,
        help="Maximum ell (and m).  Actual resolution in the Healpix FITS file may "
        "differ.  If not set, will use the maximum expansion order from file.",
    )

    order = Int(
        13,
        allow_none=False,
        help="Conviqt order parameter (expert mode)",
    )

    verbosity = Int(
        0,
        allow_none=False,
        help="",
    )

    normalize_beam = Bool(
        False,
        allow_none=False,
        help="Normalize beam to have unit response to temperature monopole.",
    )

    remove_dipole = Bool(
        False,
        allow_none=False,
        help="Suppress the temperature dipole in sky_file.",
    )

    remove_monopole = Bool(
        False,
        allow_none=False,
        help="Suppress the temperature monopole in sky_file.",
    )

    apply_flags = Bool(
        False,
        allow_none=False,
        help="Only synthesize signal for unflagged samples.",
    )

    fwhm = Quantity(
        4.0 * u.arcmin,
        allow_none=False,
        help="Width of a symmetric gaussian beam already present in the skyfile "
        "(will be deconvolved away).",
    )

    sky_file_dict = Dict(
        {},
        help="Dictionary of files containing the sky a_lm expansions. An entry for "
        "each detector name must be present. If provided, supersedes `sky_file`.",
    )

    sky_file = Unicode(
        None,
        allow_none=True,
        help="File containing the sky a_lm expansion.  Tag {detector} will be "
        "replaced with the detector name",
    )

    beam_file_dict = Dict(
        {},
        help="Dictionary of files containing the beam a_lm expansions. An entry for "
        "each detector name must be present. If provided, supersedes `beam_file`.",
    )

    beam_file = Unicode(
        None,
        allow_none=True,
        help="File containing the beam a_lm expansion.  Tag {detector} will be "
        "replaced with the detector name.",
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Shared flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @property
    def available(self):
        """Return True if libconviqt is found in the library search path."""
        return conviqt is not None and conviqt.available

    hwp_angle = Unicode(
        None, allow_none=True, help="Observation shared key for HWP angle"
    )

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        if self.comm is None:
            raise RuntimeError("libconviqt requires MPI")

        if self.detector_pointing is None:
            raise RuntimeError("detector_pointing cannot be None.")

        if self.hwp_angle is not None:
            raise RuntimeError("Standard conviqt operator cannot handle HWP angle")

        log = Logger.get()

        timer = Timer()
        timer.start()

        self.units = data.detector_units(self.det_data)
        if self.units is None:
            # This means that the data does not yet exist
            self.units = self.det_data_units

        all_detectors = self._get_all_detectors(data, detectors)

        for det in all_detectors:
            verbose = self.comm.rank == 0 and self.verbosity > 0

            # Expand detector pointing
            self.detector_pointing.apply(data, detectors=[det])

            if det in self.sky_file_dict:
                sky_file = self.sky_file_dict[det]
            else:
                sky_file = self.sky_file.format(detector=det, mc=self.mc)
            sky = self.get_sky(sky_file, det, verbose)

            if det in self.beam_file_dict:
                beam_file = self.beam_file_dict[det]
            else:
                beam_file = self.beam_file.format(detector=det, mc=self.mc)

            beam = self.get_beam(beam_file, det, verbose)

            detector = self.get_detector(det)

            theta, phi, psi_det, psi_pol, psi_beam, hwp_angle = self.get_pointing(
                data, det, verbose
            )

            pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)
            del theta, phi, psi_det, psi_pol, psi_beam, hwp_angle

            convolved_data = self.convolve(sky, beam, detector, pnt, det, verbose)

            self.calibrate_signal(data, det, beam, convolved_data, verbose)
            self.save(data, det, convolved_data, verbose)

            del pnt, detector, beam, sky

            if verbose:
                timer.report_clear(f"conviqt process detector {det}")

        return

    def _get_all_detectors(self, data, detectors):
        """Assemble a list of detectors across all processes and
        observations in `self._comm`.
        """
        my_dets = set()
        for obs in data.obs:
            # Get the detectors we are using for this observation
            obs_dets = obs.select_local_detectors(detectors)
            for det in obs_dets:
                my_dets.add(det)
            # Make sure detector data output exists
            exists = obs.detdata.ensure(
                self.det_data, detectors=detectors, create_units=self.units
            )
        all_dets = self.comm.gather(my_dets, root=0)
        if self.comm.rank == 0:
            for some_dets in all_dets:
                my_dets.update(some_dets)
            my_dets = sorted(my_dets)
        all_dets = self.comm.bcast(my_dets, root=0)
        return all_dets

    def _get_psi_pol(self, focalplane, det):
        """Parse polarization angle in radians from the focalplane
        dictionary.  The angle is relative to the Pxx basis.
        """
        if det not in focalplane:
            raise RuntimeError(f"focalplane does not include {det}")
        props = focalplane[det]
        if "psi_pol" in props.colnames:
            psi_pol = props["psi_pol"].to_value(u.radian)
        elif "pol_angle" in props.colnames:
            warnings.warn(
                "Use psi_pol and psi_uv rather than pol_angle", DeprecationWarning
            )
            psi_pol = props["pol_angle"].to_value(u.radian)
        else:
            raise RuntimeError(f"focalplane[{det}] does not include psi_pol")
        return psi_pol

    def _get_psi_uv(self, focalplane, det):
        """Parse Pxx basis angle in radians from the focalplane
        dictionary.  The angle is measured from Dxx to Pxx basis.
        """
        if det not in focalplane:
            raise RuntimeError(f"focalplane does not include {det}")
        props = focalplane[det]
        if "psi_uv_deg" in props.colnames:
            psi_uv = props["psi_uv"].to_value(u.radian)
        else:
            msg = f"focalplane[{det}] does not include psi_uv. "
            msg += "Valid column names are {props.colnames}"
            warnings.warn(msg)
            psi_uv = 0
        return psi_uv

    def _get_epsilon(self, focalplane, det):
        """Parse polarization leakage (epsilon) from the focalplane
        object or dictionary.
        """
        if det not in focalplane:
            raise RuntimeError(f"focalplane does not include {det}")
        props = focalplane[det]
        if "pol_leakage" in props.colnames:
            epsilon = focalplane[det]["pol_leakage"]
        else:
            # Assume zero polarization leakage
            epsilon = 0
        return epsilon

    def get_sky(self, skyfile, det, verbose, pol=None):
        timer = Timer()
        timer.start()
        if pol is None:
            pol = self.pol
        sky = conviqt.Sky(
            self.lmax,
            pol,
            skyfile,
            self.fwhm.to_value(u.arcmin),
            self.comm,
        )
        if self.remove_monopole:
            sky.remove_monopole()
        if self.remove_dipole:
            sky.remove_dipole()
        if verbose:
            timer.report_clear(f"initialize sky for detector {det}")
        return sky

    def get_beam(self, beamfile, det, verbose, pol=None):
        timer = Timer()
        timer.start()
        if pol is None:
            pol = self.pol
        beam = conviqt.Beam(self.lmax, self.beammmax, pol, beamfile, self.comm)
        if self.normalize_beam:
            beam.normalize()
        if verbose:
            timer.report_clear(f"initialize beam for detector {det}")
        return beam

    def get_detector(self, det):
        """We always create the detector with zero leakage and scale
        the returned TOD ourselves
        """
        detector = conviqt.Detector(name=det, epsilon=0)
        return detector

    def get_pointing(self, data, det, verbose):
        """Return the detector pointing as ZYZ Euler angles without the
        polarization sensitive angle.  These angles are to be compatible
        with Pxx or Dxx frame beam products
        """
        # We need the three pointing angles to describe the
        # pointing.  local_pointing() returns the attitude quaternions.
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
        timer = Timer()
        timer.start()
        all_theta = []
        all_phi = []
        all_psi_det = []
        all_psi_pol = []
        all_psi_beam = []
        all_hwp_angle = []
        for obs in data.obs:
            if det not in obs.local_detectors:
                continue
            focalplane = obs.telescope.focalplane
            # Loop over views
            views = obs.view[self.view]
            for view in range(len(views)):
                # Get the flags if needed
                flags = None
                if self.apply_flags:
                    if self.shared_flags is not None:
                        flags = np.array(views.shared[self.shared_flags][view])
                        flags &= self.shared_flag_mask
                    if self.det_flags is not None:
                        detflags = np.array(views.detdata[self.det_flags][view][det])
                        detflags &= self.det_flag_mask
                        if flags is not None:
                            flags |= detflags
                        else:
                            flags = detflags

                # Timestream of detector quaternions
                quats = views.detdata[self.detector_pointing.quats][view][det]
                if verbose:
                    timer.report_clear(f"get detector pointing for {det}")

                if flags is not None:
                    quats = quats.copy()
                    quats[flags != 0] = nullquat
                    if verbose:
                        timer.report_clear(f"initialize flags for detector {det}")

                # Note on angles:
                # - psi_det is the angle of right-hand rotation about the line of sight
                #   from the local Southward meridian to the detector polarization
                #   orientation.
                # - psi_pol is the angle from the beam frame to detector polarization
                #   orientation also about the line of sight.
                # - psi_beam is the angle of right-hand rotation about the line of sight
                #   from the local Southward meridian to the beam frame.

                theta, phi, psi_det = qa.to_iso_angles(quats)

                psi_pol = self._get_psi_pol(focalplane, det)
                if self.dxx:
                    # Add angle between Dxx (focalplane) and Pxx
                    psi_pol += self._get_psi_uv(focalplane, det)

                # Beam orientation
                psi_beam = psi_det - psi_pol

                # Separately we store the psi_pol angle, so that we can recover the
                # angle relative to the local meridian when computing the weights.
                psi_pol = np.ones(psi_det.size) * psi_pol

                if self.hwp_angle is None:
                    det_hwp_angle = np.zeros_like(psi_det)
                else:
                    hwp_angle = views.shared[self.hwp_angle][view]
                    # The HWP angle in the detector frame is the angle in the
                    # focalplane frame minus the angle between the focalplane
                    # and detector frames
                    props = focalplane[det]
                    if "gamma" not in props.colnames:
                        msg = (
                            "When using a HWP, the focalplane 'gamma' column must exist"
                        )
                        raise RuntimeError(msg)
                    det_hwp_angle = hwp_angle - props["gamma"].to_value(u.radian)
                    psi_pol += 2 * det_hwp_angle
                all_hwp_angle.append(det_hwp_angle)
                all_theta.append(theta)
                all_phi.append(phi)
                all_psi_det.append(psi_det)
                all_psi_pol.append(psi_pol)
                all_psi_beam.append(psi_beam)

        if len(all_theta) > 0:
            all_theta = np.hstack(all_theta)
            all_phi = np.hstack(all_phi)
            all_psi_det = np.hstack(all_psi_det)
            all_psi_pol = np.hstack(all_psi_pol)
            all_psi_beam = np.hstack(all_psi_beam)
            all_hwp_angle = np.hstack(all_hwp_angle)
        else:
            # This process has no data for this detector.  Ensure that
            # we return an empty array, not a list
            all_theta = np.array(all_theta)
            all_phi = np.array(all_phi)
            all_psi_det = np.array(all_psi_det)
            all_psi_pol = np.array(all_psi_pol)
            all_psi_beam = np.array(all_psi_beam)
            all_hwp_angle = np.array(all_hwp_angle)

        if verbose:
            timer.report_clear(f"compute pointing angles for detector {det}")
        return all_theta, all_phi, all_psi_det, all_psi_pol, all_psi_beam, all_hwp_angle

    def get_buffer(self, theta, phi, psi, det, verbose):
        """Pack the pointing into the conviqt pointing array"""
        timer = Timer()
        timer.start()
        pnt = conviqt.Pointing(len(theta))
        if pnt._nrow > 0:
            arr = pnt.data()
            arr[:, 0] = phi
            arr[:, 1] = theta
            arr[:, 2] = psi
        if verbose:
            timer.report_clear(f"pack input array for detector {det}")
        return pnt

    def convolve(self, sky, beam, detector, pnt, det, verbose, pol=None):
        timer = Timer()
        timer.start()
        if pol is None:
            pol = self.pol
        convolver = conviqt.Convolver(
            sky,
            beam,
            detector,
            pol,
            self.lmax,
            self.beammmax,
            self.order,
            self.verbosity,
            self.comm,
        )
        convolver.convolve(pnt)
        if verbose:
            timer.report_clear(f"convolve detector {det}")

        # The pointer to the data will have changed during
        # the convolution call ...

        if pnt._nrow > 0:
            arr = pnt.data()
            convolved_data = arr[:, 3].astype(np.float64)
        else:
            convolved_data = None
        if verbose:
            timer.report_clear(f"extract convolved data for {det}")

        del convolver

        return convolved_data

    def calibrate_signal(self, data, det, beam, convolved_data, verbose):
        """By default, libConviqt results returns a signal that conforms to
        TOD = (1 + epsilon) / 2 * intensity + (1 - epsilon) / 2 * polarization.

        When calibrate = True, we rescale the TOD to
        TOD = intensity + (1 - epsilon) / (1 + epsilon) * polarization
        """
        if not self.calibrate or beam.normalized():
            return
        timer = Timer()
        timer.start()
        offset = 0
        for obs in data.obs:
            if det not in obs.local_detectors:
                continue
            focalplane = obs.telescope.focalplane
            epsilon = self._get_epsilon(focalplane, det)
            # Make sure detector data output exists
            exists = obs.detdata.ensure(
                self.det_data, detectors=[det], create_units=self.units
            )
            # Loop over views
            views = obs.view[self.view]
            for view in views.detdata[self.det_data]:
                nsample = len(view[det])
                convolved_data[offset : offset + nsample] *= 2 / (1 + epsilon)
                offset += nsample
        if verbose:
            timer.report_clear(f"calibrate detector {det}")
        return

    def save(self, data, det, convolved_data, verbose):
        """Store the convolved data."""
        timer = Timer()
        timer.start()
        offset = 0
        scale = unit_conversion(u.K, self.units)
        for obs in data.obs:
            if det not in obs.local_detectors:
                continue
            # Loop over views
            views = obs.view[self.view]
            for view in views.detdata[self.det_data]:
                nsample = len(view[det])
                view[det] += scale * convolved_data[offset : offset + nsample]
                offset += nsample
        if verbose:
            timer.report_clear(f"save detector {det}")
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req["global"].extend([self.pixel_dist, self.covariance])
        req["meta"].extend([self.noise_model])
        req["shared"] = [self.boresight]
        if "detdata" not in req:
            req["detdata"] = list()
        req["detdata"].append(self.det_data)
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prov["detdata"].append(self.det_data)
        return prov


class SimWeightedConviqt(SimConviqt):
    """Operator which uses libconviqt to generate beam-convolved timestreams.
    This operator should be used in presence of a spinning  HWP which  makes
    the beam time-dependent, constantly mapping the co- and cross polar
    responses on to each other.  In OpSimConviqt we assume the beam to be static.
    """

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        if self.comm is None:
            raise RuntimeError("libconviqt requires MPI")

        if self.detector_pointing is None:
            raise RuntimeError("detector_pointing cannot be None.")

        log = Logger.get()

        timer = Timer()
        timer.start()

        self.units = data.detector_units(self.det_data)
        if self.units is None:
            # This means that the data does not yet exist
            self.units = self.det_data_units

        # Expand detector pointing
        self.detector_pointing.apply(data, detectors=detectors)

        all_detectors = self._get_all_detectors(data, detectors)

        for det in all_detectors:
            verbose = self.comm.rank == 0 and self.verbosity > 0

            # Expand detector pointing
            self.detector_pointing.apply(data, detectors=[det])

            if det in self.sky_file_dict:
                sky_file = self.sky_file_dict[det]
            else:
                sky_file = self.sky_file.format(detector=det, mc=self.mc)
            sky = self.get_sky(sky_file, det, verbose)

            if det in self.beam_file_dict:
                beam_file = self.beam_file_dict[det]
            else:
                beam_file = self.beam_file.format(detector=det, mc=self.mc)

            beamI00, beam0I0, beam00I = self.get_beam(beam_file, det, verbose)

            detector = self.get_detector(det)

            theta, phi, psi_det, psi_pol, psi_beam, hwp_angle = self.get_pointing(
                data, det, verbose
            )

            # I-beam convolution
            pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)
            convolved_data = self.convolve(sky, beamI00, detector, pnt, det, verbose)
            del pnt

            angle_arg = 2 * psi_pol

            # Q-beam convolution
            pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)
            convolved_data += np.cos(angle_arg) * self.convolve(
                sky, beam0I0, detector, pnt, det, verbose
            )
            del pnt

            # U-beam convolution
            pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)
            convolved_data += np.sin(angle_arg) * self.convolve(
                sky, beam00I, detector, pnt, det, verbose
            )
            del theta, phi, psi_det, psi_pol, psi_beam, hwp_angle

            self.calibrate_signal(
                data,
                det,
                beamI00,
                convolved_data,
                verbose,
            )
            self.save(data, det, convolved_data, verbose)

            del pnt, detector, beamI00, beam0I0, beam00I, sky

            if verbose:
                timer.report_clear(f"conviqt process detector {det}")

        return

    def get_beam(self, beamfile, det, verbose):
        timer = Timer()
        timer.start()
        beam_file_i00 = beamfile.replace(".fits", "_I000.fits")
        beam_file_0i0 = beamfile.replace(".fits", "_0I00.fits")
        beam_file_00i = beamfile.replace(".fits", "_00I0.fits")
        beami00 = conviqt.Beam(
            self.lmax, self.beammmax, self.pol, beam_file_i00, self.comm
        )
        beam0i0 = conviqt.Beam(
            self.lmax, self.beammmax, self.pol, beam_file_0i0, self.comm
        )
        beam00i = conviqt.Beam(
            self.lmax, self.beammmax, self.pol, beam_file_00i, self.comm
        )

        if verbose:
            timer.report_clear(f"initialize beam for detector {det}")
        return beami00, beam0i0, beam00i


class SimTEBConviqt(SimConviqt):
    """
    Operator that uses libconviqt to generate beam-convolved timestreams.
    This operator should be used in presence of a spinning HWP which makes the beam time-dependent,
    constantly mapping the co- and cross-polar responses on to each other.
    In the parent class OpSimConviqt we assume the beam to be static.


    The convolution  is performed by  coupling each IQU component of the signal propertly as:
    :math:`skyT_lm * beamT_lm, skyE_lm * Re{P}, skyB_lm * Im{P}`.
    FIXME : check above math

    For extra details please refer to [this note ](https://giuspugl.github.io/reports/Notes_TEB_convolution.html)
    """

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        if self.comm is None:
            raise RuntimeError("libconviqt requires MPI")

        if self.detector_pointing is None:
            raise RuntimeError("detector_pointing cannot be None.")

        log = Logger.get()

        timer = Timer()
        timer.start()

        self.units = data.detector_units(self.det_data)
        if self.units is None:
            # This means that the data does not yet exist
            self.units = self.det_data_units

        # Expand detector pointing
        self.detector_pointing.apply(data, detectors=detectors)

        all_detectors = self._get_all_detectors(data, detectors)

        for det in all_detectors:
            verbose = self.comm.rank == 0 and self.verbosity > 0

            # Find one process that has focalplane data for this detector
            # and broadcast the focalplane properties

            for obs in data.obs:
                focalplane = obs.telescope.focalplane
                have_det = det in focalplane
                if have_det:
                    break
            have_det_comm = self.comm.allgather(have_det)
            source = np.argwhere(have_det_comm).ravel()[0]
            if self.comm.rank == source:
                det_dict = {}
                for key in focalplane[det].colnames:
                    det_dict[key] = focalplane[det][key]
                det_dict["detector"] = det
                det_dict["mc"] = self.mc
            else:
                det_dict = None
            det_dict = self.comm.bcast(det_dict, root=source)

            # Expand detector pointing
            self.detector_pointing.apply(data, detectors=[det])

            if det in self.sky_file_dict:
                sky_file = self.sky_file_dict[det]
            else:
                sky_file = self.sky_file.format(**det_dict)

            skyT, skyEB, skyBE = self.get_TEB_sky(sky_file, det, verbose)

            if det in self.beam_file_dict:
                beam_file = self.beam_file_dict[det]
            else:
                beam_file = self.beam_file.format(**det_dict)

            beam_T, beam_P = self.get_TP_beam(beam_file, det, verbose)

            detector = self.get_detector(det)

            theta, phi, psi_det, psi_pol, psi_beam, hwp_angle = self.get_pointing(
                data, det, verbose
            )

            # T-convolution
            pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)

            convolved_data = self.convolve(
                skyT, beam_T, detector, pnt, det, verbose, pol=False
            )

            if self.pol:
                del (pnt,)
                angle_arg = 4.0 * hwp_angle
                # EB-convolution
                pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)
                convolved_data += np.cos(angle_arg) * self.convolve(
                    skyEB, beam_P, detector, pnt, det, verbose, pol=True
                )
                del (pnt,)
                # BE-convolution
                pnt = self.get_buffer(theta, phi, psi_beam, det, verbose)
                convolved_data += np.sin(angle_arg) * self.convolve(
                    skyBE, beam_P, detector, pnt, det, verbose, pol=True
                )

            del skyEB, skyBE

            del theta, phi, psi_det, psi_pol, psi_beam, hwp_angle

            self.calibrate_signal(
                data,
                det,
                beam_T,
                convolved_data,
                verbose,
            )
            self.save(data, det, convolved_data, verbose)

            del pnt, detector, beam_T, beam_P, skyT

            if verbose:
                timer.report_clear(f"conviqt process detector {det}")

        return

    def get_TEB_sky(self, skyfile, det, verbose):
        if os.path.isfile(skyfile):
            skyT = self.get_sky(skyfile, det, verbose, pol=False)
            # generate temporary files to use libconviqt facilities
            slmE = hp.read_alm(skyfile, hdu=2)
            slmB = hp.read_alm(skyfile, hdu=3)
            with tempfile.TemporaryDirectory() as tempdir:
                fname_temp = os.path.join(tempdir, "slm.fits")
                hp.write_alm(
                    fname_temp,
                    np.vstack([slmE * 0, slmE, slmB]),
                    lmax=self.lmax,
                    overwrite=True,
                )
                skyEB = self.get_sky(fname_temp, det, verbose, pol=True)
                hp.write_alm(
                    fname_temp,
                    np.vstack([slmE * 0, slmB, -slmE]),
                    lmax=self.lmax,
                    overwrite=True,
                )
                skyBE = self.get_sky(fname_temp, det, verbose, pol=True)
            del slmE, slmB
        else:
            # Assume the component files are on disk
            skyfile_T = skyfile.replace(".fits", "_T.fits")
            skyfile_EB = skyfile.replace(".fits", "_EB.fits")
            skyfile_BE = skyfile.replace(".fits", "_BE.fits")
            for fname in skyfile_T, skyfile_EB, skyfile_BE:
                if not os.path.isfile(fname):
                    msg = f"No TEB sky at {skyfile} and no component sky at {fname}"
                    raise RuntimeError(msg)
            skyT = self.get_sky(skyfile_T, det, verbose, pol=False)
            skyEB = self.get_sky(skyfile_EB, det, verbose, pol=True)
            skyBE = self.get_sky(skyfile_BE, det, verbose, pol=True)

        return skyT, skyEB, skyBE

    def get_TP_beam(self, beamfile, det, verbose):
        timer = Timer()
        timer.start()
        if os.path.isfile(beamfile):
            beamT = conviqt.Beam(
                lmax=self.lmax,
                mmax=self.beammmax,
                pol=False,
                beamfile=beamfile,
                comm=self.comm,
            )
            # generate temporary files to use libconviqt facilities
            blmE, mmaxE = hp.read_alm(beamfile, hdu=2, return_mmax=True)
            blmB, mmaxB = hp.read_alm(beamfile, hdu=3, return_mmax=True)
            if mmaxE != mmaxB:
                msg = f"Mismatch: mmatE={mmaxE}, mmaxB={mmaxB}"
                raise RuntimeError(msg)
            with tempfile.TemporaryDirectory() as tempdir:
                fname_temp = os.path.join(tempdir, "blm.fits")
                hp.write_alm(
                    fname_temp,
                    np.vstack([blmE * 0, blmE, blmB]),
                    lmax=self.lmax,
                    mmax_in=mmaxE,
                    overwrite=True,
                )
                beamP = conviqt.Beam(
                    lmax=self.lmax,
                    mmax=self.beammmax,
                    pol=True,
                    beamfile=fname_temp,
                    comm=self.comm,
                )
            del blmE, blmB
        else:
            beam_file_T = beamfile.replace(".fits", "_T.fits")
            beamT = conviqt.Beam(
                lmax=self.lmax,
                mmax=self.beammmax,
                pol=False,
                beamfile=beam_file_T,
                comm=self.comm,
            )
            beam_file_P = beamfile.replace(".fits", "_P.fits")
            beamP = conviqt.Beam(
                lmax=self.lmax,
                mmax=self.beammmax,
                pol=True,
                beamfile=beam_file_P,
                comm=self.comm,
            )

        if verbose:
            timer.report_clear(f"initialize beam for detector {det}")
        return beamT, beamP
