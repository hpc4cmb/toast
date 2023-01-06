# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import warnings

import healpy as hp
import numpy as np
import traitlets
from astropy import units as u

from .. import qarray as qa
from ..mpi import MPI, Comm, MPI_Comm, use_mpi
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import (
    Bool,
    Dict,
    Float,
    Instance,
    Int,
    Quantity,
    Unicode,
    Unit,
    trait_docs,
)
from ..utils import Environment, GlobalTimers, Logger, Timer, dtype_to_aligned
from .operator import Operator

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
        klass=MPI_Comm,
        allow_none=True,
        help="MPI communicator to use for the convolution.",
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

        verbose = self.verbosity > 0
        if use_mpi:
            self.comm.barrier()
            verbose = verbose and self.comm.rank == 0
            if self.comm.size > 1 and self.comm.rank == 0:
                log.warning(
                    "communicator size>1: totalconvolve will work, "
                    "but will waste memory. To be fixed in future releases."
                )

        all_detectors = self._get_all_detectors(data, detectors)

        for det in all_detectors:
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
            exists = obs.detdata.ensure(
                self.det_data, detectors=detectors, create_units=self.det_data_units
            )
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

                theta, phi, psi = qa.to_iso_angles(quats)
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
        pnt[:, 2] = psi
        pnt[:, 2] += np.pi  # FIXME: not clear yet why this is necessary
        if verbose:
            timer.report_clear(f"pack input array for detector {det}")
        return pnt

    # simple approach when there is only one task
    def conv_and_interpol_serial(
        self, skycomp, beamcomp, lmax, mmax, pnt, nthreads, t_conv, t_inter
    ):
        t_conv.start()
        plan = totalconvolve.ConvolverPlan(
            lmax=lmax,
            kmax=mmax,
            sigma=self.oversampling_factor,
            epsilon=self.epsilon,
            nthreads=nthreads,
        )
        cube = np.empty((plan.Npsi(), plan.Ntheta(), plan.Nphi()), dtype=np.float64)
        cube[()] = 0

        # convolution part
        for icomp in range(skycomp.shape[0]):
            plan.getPlane(skycomp[icomp, :], beamcomp[icomp, :], 0, cube[0:1])
            for mbeam in range(1, mmax + 1):
                plan.getPlane(
                    skycomp[icomp, :],
                    beamcomp[icomp, :],
                    mbeam,
                    cube[2 * mbeam - 1 : 2 * mbeam + 1],
                )

        plan.prepPsi(cube)
        t_conv.stop()
        t_inter.start()
        res = np.empty(pnt.shape[0], dtype=np.float64)
        plan.interpol(cube, 0, 0, pnt[:, 0], pnt[:, 1], pnt[:, 2], res)
        t_inter.stop()
        return res

    # MPI version storing the full data cube at every MPI task (wasteful)
    def conv_and_interpol_mpi(
        self, skycomp, beamcomp, lmax, mmax, pnt, nthreads, t_conv, t_inter
    ):
        t_conv.start()
        plan = totalconvolve.ConvolverPlan(
            lmax=lmax,
            kmax=mmax,
            sigma=self.oversampling_factor,
            epsilon=self.epsilon,
            nthreads=nthreads,
        )
        myrank, nranks = self.comm.rank, self.comm.size
        cube = np.empty((plan.Npsi(), plan.Ntheta(), plan.Nphi()), dtype=np.float64)
        cube[()] = 0

        # convolution part
        # the work in this nested loop can be distributed over skycomp.shape[0]*(mmax+1) tasks
        for icomp in range(skycomp.shape[0]):
            if (icomp * (mmax + 1)) % nranks == myrank:
                plan.getPlane(skycomp[icomp, :], beamcomp[icomp, :], 0, cube[0:1])
            for mbeam in range(1, mmax + 1):
                if (icomp * (mmax + 1) + mbeam) % nranks == myrank:
                    plan.getPlane(
                        skycomp[icomp, :],
                        beamcomp[icomp, :],
                        mbeam,
                        cube[2 * mbeam - 1 : 2 * mbeam + 1],
                    )
        if nranks > 1:  # broadcast the results
            for icomp in range(skycomp.shape[0]):
                self.comm.Bcast(
                    [cube[0:1], MPI.DOUBLE], root=(icomp * (mmax + 1)) % nranks
                )
            for mbeam in range(1, mmax + 1):
                self.comm.Bcast(
                    [cube[2 * mbeam - 1 : 2 * mbeam + 1], MPI.DOUBLE],
                    root=(icomp * (mmax + 1) + mbeam) % nranks,
                )

        plan.prepPsi(cube)
        t_conv.stop()
        t_inter.start()
        res = np.empty(pnt.shape[0], dtype=np.float64)
        plan.interpol(cube, 0, 0, pnt[:, 0], pnt[:, 1], pnt[:, 2], res)
        t_inter.stop()
        return res

    # MPI version with shared memory tricks, storing the full data cube only
    # once per node.
    def conv_and_interpol_mpi_shmem(
        self, skycomp, beamcomp, lmax, mmax, pnt, nthreads, t_conv, t_inter
    ):
        from pshmem import MPIShared

        t_conv.start()
        plan = totalconvolve.ConvolverPlan(
            lmax=lmax,
            kmax=mmax,
            sigma=self.oversampling_factor,
            epsilon=self.epsilon,
            nthreads=nthreads,
        )

        with MPIShared(
            (plan.Npsi(), plan.Ntheta(), plan.Nphi()), np.float64, self.comm
        ) as shm:
            cube = shm.data
            # Create a separate communicator on every node.
            intracomm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
            # Create a communicator with all master tasks of the intracomms;
            # on every other task, intercomm will be MPI.COMM_NULL.
            color = 0 if intracomm.rank == 0 else MPI.UNDEFINED
            intercomm = self.comm.Split(color)
            if intracomm.rank == 0:
                # We are on the master task of intracomm and all other tasks on
                # the node will be idle during the next computation step,
                # so we can hijack all their threads.
                nodeplan = totalconvolve.ConvolverPlan(
                    lmax=lmax,
                    kmax=mmax,
                    sigma=self.oversampling_factor,
                    epsilon=self.epsilon,
                    nthreads=intracomm.size * nthreads,
                )
                mynode, nnodes = intercomm.rank, intercomm.size
                cube[()] = 0.0

                # Convolution part
                # The skycomp.shape[0]*(mmax+1) work items in this nested loop
                # are distributed among the nodes in a round-robin fashion.
                for icomp in range(skycomp.shape[0]):
                    if (icomp * (mmax + 1)) % nnodes == mynode:
                        nodeplan.getPlane(
                            skycomp[icomp, :], beamcomp[icomp, :], 0, cube[0:1]
                        )
                    for mbeam in range(1, mmax + 1):
                        if (icomp * (mmax + 1) + mbeam) % nnodes == mynode:
                            nodeplan.getPlane(
                                skycomp[icomp, :],
                                beamcomp[icomp, :],
                                mbeam,
                                cube[2 * mbeam - 1 : 2 * mbeam + 1],
                            )
                if nnodes > 1:  # results must be broadcast to all nodes
                    for icomp in range(skycomp.shape[0]):
                        intercomm.Bcast(
                            [cube[0:1], MPI.DOUBLE], root=(icomp * (mmax + 1)) % nnodes
                        )
                    for mbeam in range(1, mmax + 1):
                        intercomm.Bcast(
                            [cube[2 * mbeam - 1 : 2 * mbeam + 1], MPI.DOUBLE],
                            root=(icomp * (mmax + 1) + mbeam) % nnodes,
                        )
                nodeplan.prepPsi(cube)
                del nodeplan

            t_conv.stop()
            t_inter.start()
            # Interpolation part
            # No fancy communication is necessary here, since every task has
            # access to the full data cube.
            res = np.empty(pnt.shape[0], dtype=np.float64)
            plan.interpol(cube, 0, 0, pnt[:, 0], pnt[:, 1], pnt[:, 2], res)
            t_inter.stop()
            del plan
            del cube
        return res

    def conv_and_interpol(
        self, skycomp, beamcomp, lmax, mmax, pnt, nthreads, t_conv, t_inter
    ):
        if (not use_mpi) or (self.comm.size == 1):
            return self.conv_and_interpol_serial(
                skycomp, beamcomp, lmax, mmax, pnt, nthreads, t_conv, t_inter
            )
        else:
            return self.conv_and_interpol_mpi_shmem(
                skycomp, beamcomp, lmax, mmax, pnt, nthreads, t_conv, t_inter
            )

    def convolve(self, sky, beam, lmax, mmax, pnt, psi_pol, det, nthreads, verbose):
        t_conv = Timer()
        t_inter = Timer()

        if self.hwp_angle is None:
            # simply compute TT+EE+BB
            convolved_data = self.conv_and_interpol(
                np.array(sky),
                np.array(beam),
                lmax,
                mmax,
                pnt,
                nthreads,
                t_conv,
                t_inter,
            )
        else:
            # TT
            convolved_data = self.conv_and_interpol(
                np.array([sky[0]]),
                np.array([beam[0]]),
                lmax,
                mmax,
                pnt,
                nthreads,
                t_conv,
                t_inter,
            )
            if self.pol:
                # EE+BB
                slm = np.array([sky[1], sky[2]])
                blm = np.array([beam[1], beam[2]])
                convolved_data += np.cos(4 * psi_pol) * self.conv_and_interpol(
                    slm, blm, lmax, mmax, pnt, nthreads, t_conv, t_inter
                ).reshape((-1,))
                # -EB+BE
                blm = np.array([-beam[2], beam[1]])
                convolved_data += np.sin(4 * psi_pol) * self.conv_and_interpol(
                    slm, blm, lmax, mmax, pnt, nthreads, t_conv, t_inter
                ).reshape((-1,))

        if verbose:
            t_conv.report_clear(f"convolve detector {det}")
            t_inter.report_clear(f"extract convolved data for {det}")

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
            exists = obs.detdata.ensure(
                self.det_data, detectors=[det], create_units=self.det_data_units
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
        req["global"].extend([self.pixel_dist, self.covariance])
        req["meta"].extend([self.noise_model])
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
        prov["detdata"].append(self.det_data)
        return prov
