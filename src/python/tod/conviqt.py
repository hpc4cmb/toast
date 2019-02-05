# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import MPI, MPI_Comm

import os

import ctypes as ct
from ctypes.util import find_library

import healpy as hp
import numpy as np
import numpy.ctypeslib as npc

from .. import qarray as qa
from ..dist import Comm, Data
from ..op import Operator
from ..tod import TOD
from ..tod import Interval

from .. import timing as timing

try:
    import libconviqt_wrapper as conviqt
except:
    conviqt = None


class OpSimConviqt(Operator):
    """
    Operator which uses libconviqt to generate beam-convolved timestreams.

    This passes through each observation and loops over each detector.
    For each detector, it produces the beam-convolved timestream.

    Args:
        lmax (int): Maximum ell (and m). Actual resolution in the Healpix FITS
            file may differ.
        beammmax (int): beam maximum m. Actual resolution in the Healpix FITS file
            may differ.
        detectordata (list): list of (detector_name, detector_sky_file,
            detector_beam_file, epsilon, psipol[radian]) tuples
        pol (bool) : boolean to determine if polarized simulation is needed
        fwhm (float) : width of a symmetric gaussian beam [in arcmin] already
            present in the skyfile (will be deconvolved away).
        order (int) : conviqt order parameter (expert mode)
        calibrate (bool) : Calibrate intensity to 1.0, rather than (1+epsilon)/2
        dxx (bool) : The beam frame is either Dxx or Pxx. Pxx includes the
            rotation to polarization sensitive basis, Dxx does not.  When
            Dxx=True, detector orientation from attitude quaternions is
            corrected for the polarization angle.
        out (str): the name of the cache object (<name>_<detector>) to
            use for output of the detector timestream.
    """

    def __init__(
        self,
        lmax,
        beammmax,
        detectordata,
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
    ):
        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._lmax = lmax
        self._beammmax = beammmax
        self._detectordata = {}
        for entry in detectordata:
            self._detectordata[entry[0]] = entry[1:]
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

        self._out = out

    @property
    def available(self):
        """
        (bool): True if libconviqt is found in the library search path.
        """
        return conviqt is not None and conviqt.available

    def exec(self, data):
        """
        Loop over all observations and perform the convolution.

        This is done one detector at a time.  For each detector, all data
        products are read from disk.

        Args:
            data (toast.Data): The distributed data.
        """
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        autotimer = timing.auto_timer(type(self).__name__)

        xaxis, yaxis, zaxis = np.eye(3)
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)

        for obs in data.obs:
            tstart_obs = MPI.Wtime()
            tod = obs["tod"]
            comm = tod.mpicomm
            intrvl = obs["intervals"]
            offset, nsamp = tod.local_samples

            for det in tod.local_dets:
                tstart_det = MPI.Wtime()
                try:
                    skyfile, beamfile, epsilon, psipol = self._detectordata[det]
                except:
                    raise Exception(
                        "ERROR: conviqt object not initialized to convolve "
                        "detector {}. Available detectors are {}".format(
                            det, self._detectordata.keys()
                        )
                    )

                sky = self.get_sky(skyfile, comm, det, tod)

                beam = self.get_beam(beamfile, comm, det, tod)

                detector = self.get_detector(det, epsilon)

                theta, phi, psi = self.get_pointing(tod, det, psipol)

                pnt = self.get_buffer(theta, phi, psi, tod, nsamp, det)

                convolved_data = self.convolve(
                    sky, beam, detector, comm, pnt, tod, nsamp, det
                )

                cachename = "{}_{}".format(self._out, det)
                if not tod.cache.exists(cachename):
                    tod.cache.create(cachename, np.float64, (nsamp,))
                ref = tod.cache.reference(cachename)
                if ref.size != convolved_data.size:
                    raise RuntimeError(
                        "{} already exists in tod.cache but has wrong size: {} "
                        "!= {}".format(cachename, ref.size, convolved_data.size)
                    )
                ref[:] += convolved_data

                del pnt
                del detector
                del beam
                del sky

                tstop = MPI.Wtime()
                if self._verbosity > 0 and tod.mpicomm.rank == 0:
                    print(
                        "{} processed in {:.2f}s".format(det, tstop - tstart_det),
                        flush=True,
                    )

            tstop = MPI.Wtime()
            if self._verbosity > 0 and tod.mpicomm.rank == 0:
                print(
                    "{} convolved in {:.2f}s".format("observation", tstop - tstart_obs),
                    flush=True,
                )

        return

    def get_sky(self, skyfile, comm, det, tod):
        tstart = MPI.Wtime()
        sky = conviqt.Sky(self._lmax, self._pol, skyfile, self._fwhm, comm)
        if self._remove_monopole:
            sky.remove_monopole()
        if self._remove_dipole:
            sky.remove_dipole()
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print(
                "{} sky initialized in {:.2f}s".format(det, tstop - tstart), flush=True
            )
        return sky

    def get_beam(self, beamfile, comm, det, tod):
        tstart = MPI.Wtime()
        beam = conviqt.Beam(self._lmax, self._beammmax, self._pol, beamfile, comm)
        if self._normalize_beam:
            beam.normalize()
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print(
                "{} beam initialized in {:.2f}s".format(det, tstop - tstart), flush=True
            )
        return beam

    def get_detector(self, det, epsilon):
        detector = conviqt.Detector(name=det, epsilon=epsilon)
        return detector

    def get_pointing(self, tod, det, psipol):
        # We need the three pointing angles to describe the
        # pointing. local_pointing returns the attitude quaternions.
        tstart = MPI.Wtime()
        pdata = tod.local_pointing(det, self._quat_name)
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print("{} pointing read in {:.2f}s".format(det, tstop - tstart), flush=True)

        if self._apply_flags:
            tstart = MPI.Wtime()
            common = tod.local_common_flags(self._common_flag_name)
            flags = tod.local_flags(det, self._flag_name)
            common = common & self._common_flag_mask
            flags = flags & self._flag_mask
            totflags = np.copy(flags)
            totflags |= common
            pdata = pdata.copy()
            pdata[totflags != 0] = nullquat
            tstop = MPI.Wtime()
            if self._verbosity > 0 and tod.mpicomm.rank == 0:
                print(
                    "{} flags initialized in {:.2f}s".format(det, tstop - tstart),
                    flush=True,
                )

        tstart = MPI.Wtime()
        theta, phi, psi = qa.to_angles(pdata)
        # Is the psi angle in Pxx or Dxx? Pxx will include the
        # detector polarization angle, Dxx will not.
        if self._dxx:
            psi -= psipol
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print(
                "{} pointing angles computed in {:.2f}s".format(det, tstop - tstart),
                flush=True,
            )
        return theta, phi, psi

    def get_buffer(self, theta, phi, psi, tod, nsamp, det):
        """
        Pack the pointing into the conviqt pointing array
        """
        tstart = MPI.Wtime()
        pnt = conviqt.Pointing(nsamp)
        arr = pnt.data()
        arr[:, 0] = phi
        arr[:, 1] = theta
        arr[:, 2] = psi
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print(
                "{} input array packed in {:.2f}s".format(det, tstop - tstart),
                flush=True,
            )
        return pnt

    def convolve(self, sky, beam, detector, comm, pnt, tod, nsamp, det):
        tstart = MPI.Wtime()
        convolver = conviqt.Convolver(
            sky,
            beam,
            detector,
            self._pol,
            self._lmax,
            self._beammmax,
            self._order,
            self._verbosity,
            comm,
        )
        convolver.convolve(pnt, self._calibrate)
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print("{} convolved in {:.2f}s".format(det, tstop - tstart), flush=True)

        # The pointer to the data will have changed during
        # the convolution call ...

        tstart = MPI.Wtime()
        arr = pnt.data()
        convolved_data = arr[:, 3].astype(np.float64)
        tstop = MPI.Wtime()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            print(
                "{} convolved data extracted in {:.2f}s".format(det, tstop - tstart),
                flush=True,
            )

        del convolver

        return convolved_data
