# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..mpi import use_mpi

import numpy as np

from .. import qarray as qa

from ..op import Operator

from ..timing import function_timer, Timer

conviqt = None

if use_mpi:
    try:
        import libconviqt_wrapper as conviqt
    except ImportError:
        pass


class OpSimConviqt(Operator):
    """Operator which uses libconviqt to generate beam-convolved timestreams.

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
        # Call the parent class constructor
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
        """Return True if libconviqt is found in the library search path.
        """
        return conviqt is not None and conviqt.available

    @function_timer
    def exec(self, data):
        """Loop over all observations and perform the convolution.

        This is done one detector at a time.  For each detector, all data
        products are read from disk.

        Args:
            data (toast.Data): The distributed data.

        """
        if not self.available:
            raise RuntimeError("libconviqt is not available")

        xaxis, yaxis, zaxis = np.eye(3)

        tmobs = Timer()
        tmdet = Timer()

        for obs in data.obs:
            tmobs.clear()
            tmobs.start()
            tod = obs["tod"]
            comm = tod.mpicomm
            rank = 0
            if comm is not None:
                rank = comm.rank
            offset, nsamp = tod.local_samples

            for det in tod.local_dets:
                tmdet.clear()
                tmdet.start()
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

                tmdet.stop()
                if self._verbosity > 0 and rank == 0:
                    msg = "conviqt process detector {}".format(det)
                    tmdet.report(msg)

            tmobs.stop()
            if self._verbosity > 0 and rank == 0:
                msg = "conviqt process observation {}".format(obs["name"])
                tmobs.report(msg)

        return

    def get_sky(self, skyfile, comm, det, tod):
        rank = 0
        if comm is not None:
            rank = comm.rank
        tm = Timer()
        tm.start()
        sky = conviqt.Sky(self._lmax, self._pol, skyfile, self._fwhm, comm)
        if self._remove_monopole:
            sky.remove_monopole()
        if self._remove_dipole:
            sky.remove_dipole()
        tm.stop()
        if self._verbosity > 0 and rank == 0:
            msg = "initialize sky for detector {}".format(det)
            tm.report(msg)
        return sky

    def get_beam(self, beamfile, comm, det, tod):
        rank = 0
        if comm is not None:
            rank = comm.rank
        tm = Timer()
        tm.start()
        beam = conviqt.Beam(self._lmax, self._beammmax, self._pol, beamfile, comm)
        if self._normalize_beam:
            beam.normalize()
        tm.stop()
        if self._verbosity > 0 and rank == 0:
            msg = "initialize beam for detector {}".format(det)
            tm.report(msg)
        return beam

    def get_detector(self, det, epsilon):
        detector = conviqt.Detector(name=det, epsilon=epsilon)
        return detector

    def get_pointing(self, tod, det, psipol):
        # We need the three pointing angles to describe the
        # pointing. local_pointing returns the attitude quaternions.
        nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
        rank = 0
        if tod.comm is not None:
            rank = tod.comm.rank
        tm = Timer()
        tm.start()
        pdata = tod.local_pointing(det, self._quat_name)
        tm.stop()
        if self._verbosity > 0 and rank == 0:
            msg = "get detector pointing for {}".format(det)
            tm.report(msg)

        if self._apply_flags:
            tm.clear()
            tm.start()
            common = tod.local_common_flags(self._common_flag_name)
            flags = tod.local_flags(det, self._flag_name)
            common = common & self._common_flag_mask
            flags = flags & self._flag_mask
            totflags = np.copy(flags)
            totflags |= common
            pdata = pdata.copy()
            pdata[totflags != 0] = nullquat
            tm.stop()
            if self._verbosity > 0 and rank == 0:
                msg = "initialize flags for detector {}".format(det)
                tm.report(msg)

        tm.clear()
        tm.start()
        theta, phi, psi = qa.to_angles(pdata)
        # Is the psi angle in Pxx or Dxx? Pxx will include the
        # detector polarization angle, Dxx will not.
        if self._dxx:
            psi -= psipol
        tm.stop()
        if self._verbosity > 0 and rank == 0:
            msg = "compute pointing angles for detector {}".format(det)
            tm.report(msg)
        return theta, phi, psi

    def get_buffer(self, theta, phi, psi, tod, nsamp, det):
        """Pack the pointing into the conviqt pointing array
        """
        rank = 0
        if tod.comm is not None:
            rank = tod.comm.rank
        tm = Timer()
        tm.start()
        pnt = conviqt.Pointing(nsamp)
        arr = pnt.data()
        arr[:, 0] = phi
        arr[:, 1] = theta
        arr[:, 2] = psi
        tm.stop()
        if self._verbosity > 0 and rank == 0:
            msg = "pack input array for detector {}".format(det)
            tm.report(msg)
        return pnt

    def convolve(self, sky, beam, detector, comm, pnt, tod, nsamp, det):
        rank = 0
        if comm is not None:
            rank = comm.rank
        tm = Timer()
        tm.start()
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
        tm.stop()
        if self._verbosity > 0 and rank == 0:
            msg = "convolve detector {}".format(det)
            tm.report(msg)

        # The pointer to the data will have changed during
        # the convolution call ...

        tm.clear()
        tm.start()
        arr = pnt.data()
        convolved_data = arr[:, 3].astype(np.float64)
        tm.stop()
        if self._verbosity > 0 and tod.mpicomm.rank == 0:
            msg = "extract convolved data for {}".format(det)
            tm.report(msg)

        del convolver

        return convolved_data
