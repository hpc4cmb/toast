# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import warnings

from astropy import units as u
import numpy as np
import traitlets

from ..mpi import MPI, use_mpi, Comm
from .operator import Operator
from .. import qarray as qa
from ..timing import function_timer
from ..traits import trait_docs, Int, Unicode, Bool, Dict, Quantity, Instance
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned


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
        klass=MPI.Comm,
        allow_none=False,
        help="MPI communicator to use for the convolution. libConviqt does "
        "not work without MPI.",
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=False,
        help="Operator that translates boresight pointing into detector frame",
    )

    times = Unicode(
        "times",
        allow_none=False,
        help="Observation shared key for timestamps",
    )
    quats = Unicode(
        None,
        allow_none=True,
        help="Observation detdata key for quaternions to use",
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional detector flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        "signal",
        allow_none=False,
        help="Observation detdata key for accumulating convolved timestreams",
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
        if check < 0:
            raise traitlets.TraitError("MC index must be nonnegative")
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
        None,
        allow_none=True,
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
        None,
        allow_none=True,
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

    @traitlets.validate("hwp_angle")
    def _check_for_hwp(self, proposal):
        # Check that there is no rotating HWP
        check = proposal["value"]
        if check is not None:
            raise RuntimeError(
                "SimConviqt cannot handle a HWP. Please use SimWeightedConviqt"
            )
        return check

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        log = Logger.get()

        timer = Timer()
        timer.start()

        # Expand detector pointing
        if self.quats:
            quats_name = self.quats
            keep_quats = True
        else:
            quats_name = "quats"
            keep_quats = False
        self.detector_pointing.quats = quats_name
        self.detector_pointing.apply(data, detectors=detectors)

        all_detectors = self._get_all_detectors(data, detectors)

        for det in all_detectors:
            verbose = self.comm.rank == 0 and self.verbosity > 0

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

            theta, phi, psi, psi_pol = self.get_pointing(
                data, det, verbose, detectors, quats_name
            )
            pnt = self.get_buffer(theta, phi, psi, det, verbose)
            del theta, phi, psi

            convolved_data = self.convolve(sky, beam, detector, pnt, det, verbose)

            self.calibrate_signal(data, det, beam, convolved_data, verbose, detectors)
            self.cache(data, det, convolved_data, verbose, detectors)

            del pnt, detector, beam, sky

            if verbose:
                timer.report_clear("conviqt process detector {}".format(det))

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
            raise RuntimeError("focalplane does not include {}".format(det))
        props = focalplane[det]
        if "psi_pol" in props.colnames:
            psi_pol = props["pol_angle"].to_value(u.radian)
        elif "pol_angle" in props.colnames:
            warnings.warn(
                "Use psi_pol and psi_uv rather than pol_angle", DeprecationWarning
            )
            psi_pol = props["pol_angle"].to_value(u.radian)
        else:
            raise RuntimeError("focalplane[{}] does not include psi_pol".format(det))
        return psi_pol

    def _get_psi_uv(self, focalplane, det):
        """Parse Pxx basis angle in radians from the focalplane
        dictionary.  The angle is measured from Dxx to Pxx basis.
        """
        if det not in focalplane:
            raise RuntimeError("focalplane does not include {}".format(det))
        props = focalplane[det]
        if "psi_uv_deg" in props.colnames:
            psi_uv = props["psi_uv"].to_value(u.radian)
        else:
            raise RuntimeError("focalplane[{}] does not include psi_uv".format(det))
        return psipol

    def _get_epsilon(self, focalplane, det):
        """Parse polarization leakage (epsilon) from the focalplane
        object or dictionary.
        """
        if det not in focalplane:
            raise RuntimeError("focalplane does not include {}".format(det))
        props = focalplane[det]
        if "pol_leakage" in props.colnames:
            epsilon = focalplane[det]["pol_leakage"]
        else:
            # Assume zero polarization leakage
            epsilon = 0
        return epsilon

    def get_sky(self, skyfile, det, verbose):
        timer = Timer()
        timer.start()
        sky = conviqt.Sky(
            self.lmax,
            self.pol,
            skyfile,
            self.fwhm.to_value(u.arcmin),
            self.comm,
        )
        if self.remove_monopole:
            sky.remove_monopole()
        if self.remove_dipole:
            sky.remove_dipole()
        if verbose:
            timer.report_clear("initialize sky for detector {}".format(det))
        return sky

    def get_beam(self, beamfile, det, verbose):
        timer = Timer()
        timer.start()
        beam = conviqt.Beam(self.lmax, self.beammmax, self.pol, beamfile, self.comm)
        if self.normalize_beam:
            beam.normalize()
        if verbose:
            timer.report_clear("initialize beam for detector {}".format(det))
        return beam

    def get_detector(self, det):
        """We always create the detector with zero leakage and scale
        the returned TOD ourselves
        """
        detector = conviqt.Detector(name=det, epsilon=0)
        return detector

    def get_pointing(self, data, det, verbose, detectors, quats_name):
        """Return the detector pointing as ZYZ Euler angles without the
        polarization sensitive angle.  These angles are to be compatible
        with Pxx or Dxx frame beam products
        """
        # We need the three pointing angles to describe the
        # pointing.  local_pointing() returns the attitude quaternions.
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
        timer = Timer()
        timer.start()
        all_theta, all_phi, all_psi, all_psi_pol = [], [], [], []
        for obs in data.obs:
            if det not in obs.select_local_detectors(detectors):
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
                quats = views.detdata[quats_name][view][det]
                if verbose:
                    timer.report_clear(f"get detector pointing for {det}")

                if flags is not None:
                    quats = quats.copy()
                    quats[flags != 0] = nullquat
                    if verbose:
                        timer.report_clear(f"initialize flags for detector {det}")

                theta, phi, psi = qa.to_angles(quats)
                # Polarization angle in the Pxx basis
                psi_pol = self._get_psi_pol(focalplane, det)
                if self.dxx:
                    # Add angle between Dxx and Pxx
                    psi_pol += self._get_psi_uv(focalplane, det)
                psi -= psi_pol
                psi_pol = np.ones(psi.size) * psi_pol
                all_theta.append(theta)
                all_phi.append(phi)
                all_psi.append(psi)
                all_psi_pol.append(psi_pol)

        if len(all_theta) > 0:
            all_theta = np.hstack(all_theta)
            all_phi = np.hstack(all_phi)
            all_psi = np.hstack(all_psi)
            all_psi_pol = np.hstack(all_psi_pol)

        if verbose:
            timer.report_clear("compute pointing angles for detector {}".format(det))
        return all_theta, all_phi, all_psi, all_psi_pol

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
            timer.report_clear("pack input array for detector {}".format(det))
        return pnt

    def convolve(self, sky, beam, detector, pnt, det, verbose):
        timer = Timer()
        timer.start()
        convolver = conviqt.Convolver(
            sky,
            beam,
            detector,
            self.pol,
            self.lmax,
            self.beammmax,
            self.order,
            self.verbosity,
            self.comm,
        )
        convolver.convolve(pnt)
        if verbose:
            timer.report_clear("convolve detector {}".format(det))

        # The pointer to the data will have changed during
        # the convolution call ...

        if pnt._nrow > 0:
            arr = pnt.data()
            convolved_data = arr[:, 3].astype(np.float64)
        else:
            convolved_data = None
        if verbose:
            timer.report_clear("extract convolved data for {}".format(det))

        del convolver

        return convolved_data

    def calibrate_signal(self, data, det, beam, convolved_data, verbose, detectors):
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
            if det not in obs.select_local_detectors(detectors):
                continue
            focalplane = obs.telescope.focalplane
            epsilon = self._get_epsilon(focalplane, det)
            # Make sure detector data output exists
            obs.detdata.ensure(self.det_data, detectors=[det])
            # Loop over views
            views = obs.view[self.view]
            for view in views.detdata[self.det_data]:
                nsample = len(view[det])
                convolved_data[offset : offset + nsample] *= 2 / (1 + epsilon)
                offset += nsample
        if verbose:
            timer.report_clear(f"calibrate detector {det}")
        return

    def cache(self, data, det, convolved_data, verbose, detectors):
        """Inject the convolved data into the TOD cache."""
        timer = Timer()
        timer.start()
        offset = 0
        for obs in data.obs:
            dets = obs.select_local_detectors(detectors)
            if det not in dets:
                continue
            # Make sure detector data output exists
            obs.detdata.ensure(self.det_data, detectors=[det])
            # Loop over views
            views = obs.view[self.view]
            for view in views.detdata[self.det_data]:
                ref = view[det]
                nsample = len(ref)
                ref[:] += convolved_data[offset : offset + nsample]
                offset += nsample
        if verbose:
            timer.report_clear(f"cache detector {det}")
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req["meta"].extend([self.noise_model, self.pixel_dist, self.covariance])
        req["shared"] = [self.times, self.boresight]
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {"meta": list(), "shared": list(), "detdata": [self.det_data]}
        return prov

    def _accelerators(self):
        return list()


class SimWeightedConviqt(SimConviqt):
    """Operator which uses libconviqt to generate beam-convolved timestreams.
    This operator should be used in presence of a spinning  HWP which  makes the beam time-dependent,
    constantly mapping the co- and cross polar responses on to each other.  In OpSimConviqt we assume the  beam to be static.

    This passes through each observation and loops over each detector.
    For each detector, it produces the beam-convolved timestream.

    Args:
        comm (MPI.Comm) : MPI communicator to use for the convolution.
            libConviqt does not work without MPI.
        sky_file (dict or str) : File containing the sky a_lm expansion.
            Tag {detector} will be replaced with the detector name
            If sky_file is a dict, then each detector must have an entry.
        beam_file (dict or str) : File containing the beam a_lm expansion.
            Tag {detector} will be replaced with the detector name.
            If beam_file is a dict, then each detector must have an entry.
        lmax (int) : Maximum ell (and m).  Actual resolution in the
            Healpix FITS file may differ.  If not set, will use the
            maximum expansion order from file.
        beammmax (int) : beam maximum m.  Actual resolution in the
            Healpix FITS file may differ.  If not set, will use the
            maximum expansion order from file.
        pol (bool) : boolean to determine if polarized simulation is needed
        fwhm (float) : width of a symmetric gaussian beam [in arcmin] already
            present in the skyfile (will be deconvolved away).
        order (int) : conviqt order parameter (expert mode)
        calibrate (bool) : Calibrate intensity to 1.0, rather than
            (1 + epsilon) / 2.  Calibrate has no effect if the beam is found
            to be normalized rather than scaled with the leakage factor.
        dxx (bool) : The beam frame is either Dxx or Pxx.  Pxx includes the
            rotation to polarization sensitive basis, Dxx does not.  When
            Dxx=True, detector orientation from attitude quaternions is
            corrected for the polarization angle.
        out (str): the name of the cache object (<name>_<detector>) to
            use for output of the detector timestream.
        mc (int): Monte Carlo index used in synthesizing the input file names.

    """

    """
    def __init__(
        self,
        comm,
        sky_file,
        beam_file,
        lmax=0,
        beammmax=0,
        pol=True,
        fwhm=4.0,
        order=13,
        calibrate=True,
        dxx=True,
        out="conviqt",
        quat_name=None,
        flag_name=None,
        flag_mask=255,
        common_flag_name=None,
        common_flag_mask=255,
        apply_flags=False,
        remove_monopole=False,
        remove_dipole=False,
        normalize_beam=False,
        verbosity=0,
        mc=None,
    ):
        # Call the parent class constructor
        super(OpSimConviqt, self).__init__()

        self._comm = comm
        self._sky_file = sky_file
        self._beam_file = beam_file
        self._lmax = lmax
        self._beammmax = beammmax
        self._pol = pol
        self._fwhm = fwhm
        self._order = order
        self._calibrate = calibrate
        self._dxx = dxx
        self._quat_name = quat_name
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._apply_flags = apply_flags
        self._remove_monopole = remove_monopole
        self._remove_dipole = remove_dipole
        self._normalize_beam = normalize_beam
        self._verbosity = verbosity
        self._mc = mc

        self._out = out

    @function_timer
    def exec(self, data):
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        timer = Timer()
        timer.start()

        detectors = self._get_detectors(data)

        for det in detectors:
            verbose = self._comm.rank == 0 and self._verbosity > 0

            try:
                sky_file = self._sky_file[det]
            except TypeError:
                sky_file = self._sky_file.format(detector=det, mc=self._mc)
            sky = self.get_sky(sky_file, det, verbose)

            try:
                beam_file = self._beam_file[det]
            except TypeError:
                beam_file = self._beam_file.format(detector=det, mc=self._mc)
                beam_file_i00 = beam_file.replace(".fits", "_I000.fits")
                beam_file_0i0 = beam_file.replace(".fits", "_0I00.fits")
                beam_file_00i = beam_file.replace(".fits", "_00I0.fits")

            beamI00 = self.get_beam(beam_file_i00, det, verbose)
            beam0I0 = self.get_beam(beam_file_0i0, det, verbose)
            beam00I = self.get_beam(beam_file_00i, det, verbose)
            detector = self.get_detector(det)

            theta, phi, psi, psi_pol = self.get_pointing(data, det, verbose)

            # I-beam convolution
            pnt = self.get_buffer(theta, phi, psi, det, verbose)
            convolved_data = self.convolve(sky, beamI00, detector, pnt, det, verbose)

            # Q-beam convolution
            del pnt
            pnt = self.get_buffer(theta, phi, psi, det, verbose)
            convolved_data += np.cos(2 * psi_pol) * self.convolve(
                sky, beam0I0, detector, pnt, det, verbose
            )

            # U-beam convolution
            del pnt
            pnt = self.get_buffer(theta, phi, psi, det, verbose)
            convolved_data += np.sin(2 * psi_pol) * self.convolve(
                sky, beam00I, detector, pnt, det, verbose
            )
            del theta, phi, psi

            self.calibrate(data, det, beamI00, convolved_data, verbose)

            self.cache(data, det, convolved_data, verbose)

            del pnt, detector, beamI00, beam0I0, beam00I, sky

            if verbose:
                timer.report_clear("conviqt process detector {}".format(det))

        return
"""
