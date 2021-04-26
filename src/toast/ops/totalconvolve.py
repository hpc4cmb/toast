# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import warnings

from astropy import units as u
import numpy as np
import traitlets
import healpy as hp

from ..mpi import MPI, use_mpi, Comm
from .operator import Operator
from .. import qarray as qa
from ..timing import function_timer
from ..traits import trait_docs, Int, Float, Unicode, Bool, Dict, Quantity, Instance
from ..utils import Logger, Environment, Timer, GlobalTimers, dtype_to_aligned


totalconvolve = None

try:
    import ducc0.totalconvolve as totalconvolve
except ImportError:
    totalconvolve = None


def available():
    """(bool): True if ducc0.totalconvolve is found in the library search path."""
    global totalconvolve
    return totalconvolve is not None


@trait_docs
class SimTotalconvolve(Operator):
    """Operator which uses ducc0.totalconvolve to generate beam-convolved timestreams."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    comm = Instance(
        klass=MPI.Comm,
        allow_none=True,
        help="MPI communicator to use for the convolution.",
    )

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight pointing into detector frame",
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
        if check is not None and check < 0:
            raise traitlets.TraitError("MC index cannot be negative")
        return check

    beammmax = Int(
        -1,
        allow_none=False,
        help="Beam maximum m.  Actual resolution in the Healpix FITS file may differ. "
        "If not set, will use the maximum expansion order from file.",
    )

    oversampling_factor = Float(
        1.8,
        allow_none=False,
        help="Oversampling factor for total convolution (useful range is 1.5-2.0)",
    )

    epsilon = Float(
        1e-5,
        allow_none=False,
        help="Relative accuracy of the interpolation step",
    )

    lmax = Int(
        -1,
        allow_none=False,
        help="Maximum ell (and m).  Actual resolution in the Healpix FITS file may "
        "differ.  If not set, will use the maximum expansion order from file.",
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
        """Return True if ducc0.totalconvolve is found in the library search path."""
        return totalconvolve is not None

    hwp_angle = Unicode(
        None, allow_none=True, help="Observation shared key for HWP angle"
    )

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if not self.available:
            raise RuntimeError("ducc0.totalconvolve is not available")

        if self.detector_pointing is None:
            raise RuntimeError("detector_pointing cannot be None.")

        log = Logger.get()

        timer = Timer()
        timer.start()

        env = Environment.get()
        nthreads = env.max_threads()

        all_detectors = self._get_all_detectors(data, detectors)

        for det in all_detectors:
            verbose = self.verbosity > 0
            if use_mpi:
                verbose = verbose and self.comm.rank == 0
                if self.comm.size > 1:
                    log.warning(
                        "communicator size>1: totalconvolve will work, "
                        "but will waste CPU and memory. To be fixed in "
                        "future releases."
                    )

            # Expand detector pointing
            self.detector_pointing.apply(data, detectors=[det])

            if det in self.sky_file_dict:
                sky_file = self.sky_file_dict[det]
            else:
                sky_file = self.sky_file.format(detector=det, mc=self.mc)

            if det in self.beam_file_dict:
                beam_file = self.beam_file_dict[det]
            else:
                beam_file = self.beam_file.format(detector=det, mc=self.mc)

            lmax, mmax = self.get_lmmax(sky_file, beam_file)
            sky = self.get_sky(sky_file, lmax, det, verbose)
            beam = self.get_beam(beam_file, lmax, mmax, det, verbose)

            theta, phi, psi, psi_pol = self.get_pointing(data, det, verbose)
            pnt = self.get_buffer(theta, phi, psi, det, verbose)
            del theta, phi, psi
            if self.hwp_angle is None:
                psi_pol = None

            convolved_data = self.convolve(
                sky, beam, lmax, mmax, pnt, psi_pol, det, nthreads, verbose
            )
            del psi_pol

            self.calibrate_signal(data, det, beam, convolved_data, verbose)
            self.save(data, det, convolved_data, verbose)

            del pnt, beam, sky

            if verbose:
                timer.report_clear(f"totalconvolve process detector {det}")

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
            obs.detdata.ensure(self.det_data, detectors=detectors)
        if use_mpi:
            all_dets = self.comm.gather(my_dets, root=0)
            if self.comm.rank == 0:
                for some_dets in all_dets:
                    my_dets.update(some_dets)
                my_dets = sorted(my_dets)
            all_dets = self.comm.bcast(my_dets, root=0)
        else:
            all_dets = my_dets
        return all_dets

    def _get_psi_pol(self, focalplane, det):
        """Parse polarization angle in radians from the focalplane
        dictionary.  The angle is relative to the Pxx basis.
        """
        if det not in focalplane:
            raise RuntimeError(f"focalplane does not include {det}")
        props = focalplane[det]
        if "psi_pol" in props.colnames:
            psi_pol = props["pol_angle"].to_value(u.radian)
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
            raise RuntimeError(f"focalplane[{det}] does not include psi_uv")
        return psipol

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

    def get_lmmax(self, skyfile, beamfile):
        """Determine the actual lmax and beammmax to use for the convolution
        from class parameters and values in the files.
        """
        ncomp = 3 if self.pol else 1
        slmax, blmax, bmmax = -1, -1, -1
        for i in range(ncomp):
            # for sky and beam respectively, lmax is the max of all components
            alm_tmp, mmax_tmp = hp.fitsfunc.read_alm(
                skyfile, hdu=i + 1, return_mmax=True
            )
            lmax_tmp = hp.sphtfunc.Alm.getlmax(alm_tmp.shape[0], mmax_tmp)
            slmax = max(slmax, lmax_tmp)
            alm_tmp, mmax_tmp = hp.fitsfunc.read_alm(
                beamfile, hdu=i + 1, return_mmax=True
            )
            lmax_tmp = hp.sphtfunc.Alm.getlmax(alm_tmp.shape[0], mmax_tmp)
            blmax = max(blmax, lmax_tmp)
            # for the beam, determine also the largest mmax present
            bmmax = max(bmmax, mmax_tmp)
        # no need to go higher than the lower of the lmax from sky and beam
        lmax_out = min(slmax, blmax)
        mmax_out = bmmax
        # if parameters are lower than the detected values, reduce even further
        if self.lmax != -1:
            lmax_out = min(lmax_out, self.lmax)
        if self.beammmax != -1:
            mmax_out = min(mmax_out, self.beammmax)
        return lmax_out, mmax_out

    def load_alm(self, file, lmax, mmax):
        def read_comp(file, comp, out):
            almX, mmaxX = hp.fitsfunc.read_alm(file, hdu=comp + 1, return_mmax=True)
            lmaxX = hp.sphtfunc.Alm.getlmax(almX.shape[0], mmaxX)

            ofs1, ofs2 = 0, 0
            mylmax = min(lmax, lmaxX)
            for m in range(0, min(mmax, mmaxX) + 1):
                out[comp, ofs1 : ofs1 + mylmax - m + 1] = almX[
                    ofs2 : ofs2 + mylmax - m + 1
                ]
                ofs1 += lmax - m + 1
                ofs2 += lmaxX - m + 1

        ncomp = 3 if self.pol else 1
        res = np.zeros(
            (ncomp, hp.sphtfunc.Alm.getsize(lmax, mmax)), dtype=np.complex128
        )
        for i in range(ncomp):
            read_comp(file, i, res)
        return res

    def get_sky(self, skyfile, lmax, det, verbose):
        timer = Timer()
        timer.start()
        sky = self.load_alm(skyfile, lmax, lmax)
        fwhm = self.fwhm.to_value(u.radian)
        if fwhm != 0:
            gauss = hp.sphtfunc.gauss_beam(fwhm, lmax, pol=True)
            for i in range(sky.shape[0]):
                sky[i] = hp.sphtfunc.almxfl(
                    sky[i], 1.0 / gauss[:, i], mmax=lmax, inplace=True
                )
        if self.remove_monopole:
            sky[0, 0] = 0
        if self.remove_dipole:
            sky[0, 1] = 0
            sky[0, lmax + 1] = 0
        if verbose:
            timer.report_clear(f"initialize sky for detector {det}")
        return sky

    def get_beam(self, beamfile, lmax, mmax, det, verbose):
        timer = Timer()
        timer.start()
        beam = self.load_alm(beamfile, lmax, mmax)
        if self.normalize_beam:
            beam *= 1.0 / (2 * np.sqrt(np.pi) * beam[0, 0])
        if verbose:
            timer.report_clear(f"initialize beam for detector {det}")
        return beam

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
        all_theta, all_phi, all_psi, all_psi_pol = [], [], [], []
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

                theta, phi, psi = qa.to_angles(quats)
                # Polarization angle in the Pxx basis
                psi_pol = self._get_psi_pol(focalplane, det)
                if self.dxx:
                    # Add angle between Dxx and Pxx
                    psi_pol += self._get_psi_uv(focalplane, det)
                psi -= psi_pol
                psi_pol = np.ones(psi.size) * psi_pol
                if self.hwp_angle is not None:
                    hwp_angle = views.shared[self.hwp_angle][view]
                    psi_pol += 2 * hwp_angle
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
            timer.report_clear(f"compute pointing angles for detector {det}")
        return all_theta, all_phi, all_psi, all_psi_pol

    def get_buffer(self, theta, phi, psi, det, verbose):
        """Pack the pointing into the pointing array"""
        timer = Timer()
        timer.start()
        pnt = np.empty((len(theta), 3))
        pnt[:, 0] = theta
        pnt[:, 1] = phi
        pnt[:, 2] = psi + np.pi  # FIXME: not clear yet why this is necessary
        if verbose:
            timer.report_clear(f"pack input array for detector {det}")
        return pnt

    def convolve(self, sky, beam, lmax, mmax, pnt, psi_pol, det, nthreads, verbose):
        timer = Timer()
        timer.start()

        if self.hwp_angle is None:
            # simply compute TT+EE+BB
            convolver = totalconvolve.Interpolator(
                np.array(sky),
                np.array(beam),
                False,
                int(lmax),
                int(mmax),
                epsilon=float(self.epsilon),
                ofactor=float(self.oversampling_factor),
                nthreads=nthreads,
            )
            convolved_data = convolver.interpol(pnt).reshape((-1,))
        else:
            # TT
            convolver = totalconvolve.Interpolator(
                np.array([sky[0]]),
                np.array([beam[0]]),
                False,
                int(lmax),
                int(mmax),
                epsilon=float(self.epsilon),
                ofactor=float(self.oversampling_factor),
                nthreads=nthreads,
            )
            convolved_data = convolver.interpol(pnt).reshape((-1,))
            if self.pol:
                # EE+BB
                slm = np.array([sky[1], sky[2]])
                blm = np.array([beam[1], beam[2]])
                convolver = totalconvolve.Interpolator(
                    slm,
                    blm,
                    False,
                    int(lmax),
                    int(mmax),
                    epsilon=float(self.epsilon),
                    ofactor=float(self.oversampling_factor),
                    nthreads=nthreads,
                )
                convolved_data += np.cos(4 * psi_pol) * convolver.interpol(pnt).reshape(
                    (-1,)
                )
                # -EB+BE
                blm = np.array([-beam[2], beam[1]])
                convolver = totalconvolve.Interpolator(
                    slm,
                    blm,
                    False,
                    int(lmax),
                    int(mmax),
                    epsilon=float(self.epsilon),
                    ofactor=float(self.oversampling_factor),
                    nthreads=nthreads,
                )
                convolved_data += np.sin(4 * psi_pol) * convolver.interpol(pnt).reshape(
                    (-1,)
                )

        if verbose:
            timer.report_clear(f"convolve detector {det}")
            timer.report_clear(f"extract convolved data for {det}")

        del convolver
        convolved_data *= 0.5  # FIXME: not sure where this factor comes from
        return convolved_data

    def calibrate_signal(self, data, det, beam, convolved_data, verbose):
        """By default, libConviqt results returns a signal that conforms to
        TOD = (1 + epsilon) / 2 * intensity + (1 - epsilon) / 2 * polarization.

        When calibrate = True, we rescale the TOD to
        TOD = intensity + (1 - epsilon) / (1 + epsilon) * polarization
        """
        if not self.calibrate:  # or beam.normalized():
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

    def save(self, data, det, convolved_data, verbose):
        """Store the convolved data."""
        timer = Timer()
        timer.start()
        offset = 0
        for obs in data.obs:
            if det not in obs.local_detectors:
                continue
            # Loop over views
            views = obs.view[self.view]
            for view in views.detdata[self.det_data]:
                nsample = len(view[det])
                view[det] += convolved_data[offset : offset + nsample]
                offset += nsample
        if verbose:
            timer.report_clear(f"save detector {det}")
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = self.detector_pointing.requires()
        req["meta"].extend([self.noise_model, self.pixel_dist, self.covariance])
        req["shared"] = [self.boresight]
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = self.detector_pointing.provides()
        prob["detdata"].append(self.det_data)
        return prov

    def _accelerators(self):
        return list()
