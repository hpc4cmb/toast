# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import healpy as hp
import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from .. import qarray as qa
from ..intervals import IntervalList
from ..observation import default_values as defaults
from ..ops.real_space_convolve.kernels_numpy import _beam_sum, _spin2_transport
from .helpers import close_data, create_outdir, create_satellite_data
from .mpi import MPITestCase

ZAXIS = np.array([0.0, 0.0, 1.0])


def beam_from_directions(directions, weights_I, weights_P=None):
    """Build a beam from node directions in the frame of a beam map.

    The beam map has the beam center along X, with offsets in declination along Z
    and in right ascension along Y.  A detector looks along its Z axis and is
    sensitive to polarization along its X axis, so the map direction (x, y, z)
    is the detector-frame direction (-z, y, x).

    """
    b = np.atleast_2d(directions)
    b = b / np.linalg.norm(b, axis=1)[:, None]
    v = np.stack([-b[:, 2], b[:, 1], b[:, 0]], axis=1)
    quats = qa.from_vectors(np.tile(ZAXIS, (len(v), 1)), v)
    return ops.RealSpaceBeam(quats, weights_I, weights_P)


def bearing(z1, s1, p1, z2, s2, p2):
    """Bearing east of north at point 1 towards point 2."""
    return np.arctan2(np.sin(p2 - p1) * s2, s1 * z2 - z1 * s2 * np.cos(p2 - p1))


def interp_iqu(maps, theta, phi):
    """Bilinear I, Q, U at each direction, with Q and U of each pixel transported to
    that direction."""
    nside = hp.get_nside(maps[0])
    pix, wts = hp.get_interp_weights(nside, theta, phi)
    theta_p, phi_p = hp.pix2ang(nside, pix)
    pix_angles = (np.cos(theta_p), np.sin(theta_p), phi_p)
    los_angles = (np.cos(theta), np.sin(theta), phi)
    delta = bearing(*pix_angles, *los_angles) - bearing(*los_angles, *pix_angles)
    pol = np.sum(wts * (maps[1][pix] + 1j * maps[2][pix]) * np.exp(2j * delta), 0)
    return np.sum(wts * maps[0][pix], 0), pol.real, pol.imag


class RealSpaceConvolveTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        self.rank = 0 if self.comm is None else self.comm.rank

    def barrier(self):
        if self.comm is not None:
            self.comm.barrier()

    def write_map(self, name, maps):
        path = os.path.join(self.outdir, name)
        if self.rank == 0:
            hp.write_map(path, maps, overwrite=True, dtype=np.float64)
        self.barrier()
        return path

    def write_beam(self, name, beam):
        path = os.path.join(self.outdir, name)
        if self.rank == 0:
            beam.save(path)
        self.barrier()
        return path

    def random_map(self, nside, fwhm_deg):
        lmax = 3 * nside - 1
        cl = np.zeros((4, lmax + 1))
        cl[:3, 2:] = 1.0 / np.arange(2, lmax + 1)[None, :] ** 2
        np.random.seed(1234)
        return hp.synfast(cl, nside, fwhm=np.radians(fwhm_deg), new=True)

    def pointing(self, mode):
        detpointing = ops.PointingDetectorSimple()
        weights = ops.StokesWeights(
            mode=mode,
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing,
        )
        return detpointing, weights

    def test_beam_file(self):
        quats = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
        w = np.array([0.2, 0.3, 0.5])
        for weights_P in (None, 2 * w):
            path = os.path.join(self.outdir, f"beam_{self.rank}.npz")
            ops.RealSpaceBeam(quats, w, weights_P).save(path)
            beam = ops.RealSpaceBeam.load(path)
            nt.assert_array_equal(beam.quats, quats)
            nt.assert_array_equal(beam.weights_I, w)
            if weights_P is None:
                self.assertIsNone(beam.weights_P)
            else:
                nt.assert_array_equal(beam.weights_P, weights_P)
        with self.assertRaises(ValueError):
            ops.RealSpaceBeam(quats[:0], w[:0])
        with self.assertRaises(ValueError):
            ops.RealSpaceBeam(quats[None], w)
        with self.assertRaises(ValueError):
            ops.RealSpaceBeam(quats, np.stack([w, w]))
        with self.assertRaises(ValueError):
            ops.RealSpaceBeam(quats, w, w[:2])

    def test_transport(self):
        np.random.seed(1)
        n = 10000
        z1 = np.random.uniform(-0.95, 0.95, n)
        p1 = np.random.uniform(0, 2 * np.pi, n)
        sep = np.radians(np.random.uniform(0.5, 60, n) / 60)
        az = np.random.uniform(0, 2 * np.pi, n)
        s1 = np.sqrt(1 - z1**2)
        z2 = z1 * np.cos(sep) + s1 * np.sin(sep) * np.cos(az)
        s2 = np.sqrt(1 - z2**2)
        p2 = p1 + np.arctan2(np.sin(az) * np.sin(sep) * s1, np.cos(sep) - z1 * z2)
        c2d, s2d = _spin2_transport(z1, s1, p1, z2, s2, p2)
        delta = bearing(z1, s1, p1, z2, s2, p2) - bearing(z2, s2, p2, z1, s1, p1)
        nt.assert_allclose(c2d, np.cos(2 * delta), rtol=0, atol=1e-10)
        nt.assert_allclose(s2d, np.sin(2 * delta), rtol=0, atol=1e-10)
        # Coincident points
        c2c, s2c = _spin2_transport(z1, s1, p1, z1, s1, p1)
        nt.assert_array_equal(c2c, 1.0)
        nt.assert_array_equal(s2c, 0.0)

    def test_orientation(self):
        # A node 1 degree from the center, along +declination of the beam map,
        # lies opposite the polarization direction of the detector.
        data = create_satellite_data(self.comm, obs_time=2 * u.minute)
        nside = 64
        maps = self.random_map(nside, 3.0)
        delta = np.radians(1.0)
        beam = beam_from_directions([np.cos(delta), 0.0, np.sin(delta)], [1.0])
        detpointing, weights = self.pointing("I")
        ops.SimRealSpaceConvolve(
            maps=[maps[0]],
            beam_file=self.write_beam("north.npz", beam),
            det_data="conv",
            stokes_weights=weights,
        ).apply(data)
        weights.apply(data)
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                theta, phi, psi = qa.to_iso_angles(ob.detdata[detpointing.quats][det])
                los = hp.ang2vec(theta, phi)
                e_theta = np.stack(
                    [
                        np.cos(theta) * np.cos(phi),
                        np.cos(theta) * np.sin(phi),
                        -np.sin(theta),
                    ],
                    axis=-1,
                )
                e_phi = np.stack(
                    [-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1
                )
                pol = np.cos(psi)[:, None] * e_theta + np.sin(psi)[:, None] * e_phi
                node = np.cos(delta) * los - np.sin(delta) * pol
                expected = hp.get_interp_val(maps[0], *hp.vec2ang(node))
                nt.assert_allclose(
                    ob.detdata["conv"][det],
                    expected,
                    rtol=0,
                    atol=1e-12 * np.std(maps[0]),
                )
        close_data(data)

    def test_pencil_intensity(self):
        # A one-node beam at the center matches InterpolateHealpixMap, which reads
        # the map in single precision.
        data = create_satellite_data(self.comm, obs_time=2 * u.minute)
        maps = self.random_map(64, 3.0)
        path = self.write_map("intensity.fits", maps[0])
        beam = beam_from_directions([1.0, 0.0, 0.0], [1.0])
        detpointing, weights = self.pointing("I")
        ops.SimRealSpaceConvolve(
            file=path,
            beam_file=self.write_beam("pencil.npz", beam),
            det_data="conv",
            stokes_weights=weights,
        ).apply(data)
        ops.InterpolateHealpixMap(
            file=path,
            det_data="interp",
            detector_pointing=detpointing,
            stokes_weights=weights,
        ).apply(data)
        weights.apply(data)
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                nt.assert_allclose(
                    ob.detdata["conv"][det],
                    ob.detdata["interp"][det],
                    rtol=0,
                    atol=1e-6 * np.std(maps[0]),
                )
        close_data(data)

    def test_pencil_polarization(self):
        # A one-node beam at the center on an IQU map, against bilinear weights
        # and the two-bearing transport of each pixel to the line of sight.
        data = create_satellite_data(self.comm, obs_time=2 * u.minute)
        nside = 64
        maps = self.random_map(nside, 3.0)
        beam = beam_from_directions([1.0, 0.0, 0.0], [1.0], [1.0])
        detpointing, weights = self.pointing("IQU")
        ops.SimRealSpaceConvolve(
            maps=list(maps),
            beam_file=self.write_beam("pencil_pol.npz", beam),
            det_data="conv",
            stokes_weights=weights,
        ).apply(data)
        weights.apply(data)
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                theta, phi, _ = qa.to_iso_angles(ob.detdata[detpointing.quats][det])
                iqu = interp_iqu(maps, theta, phi)
                w = ob.detdata[weights.weights][det]
                expected = sum(w[:, i] * iqu[i] for i in range(3))
                nt.assert_allclose(
                    ob.detdata["conv"][det], expected, rtol=0, atol=1e-10 * np.std(maps)
                )
        close_data(data)

    def test_gaussian_beam(self):
        # A symmetric Gaussian beam against the harmonic convolution of
        # healpy.smoothing, read at the line of sight.  This pins the sign of the
        # parallel transport.  The transport angle grows with the beam size and
        # the latitude, and the test scan stays below 22 degrees, so the beam is
        # wide: the bilinear interpolation leaves about 0.6% of the Q and U rms,
        # and a reversed transport about 13%.
        data = create_satellite_data(self.comm, obs_time=2 * u.minute)
        nside = 64
        fwhm = np.radians(10.0)
        sigma = fwhm / np.sqrt(8 * np.log(2))
        maps = self.random_map(nside, 5.0)

        # Grid the beam on equal steps in right ascension and declination, with the
        # cell solid angle cos(dec) in the weights.
        step = sigma / 4
        offsets = np.arange(-4 * sigma, 4 * sigma + step / 2, step)
        ra, dec = np.meshgrid(offsets, offsets)
        ra, dec = ra.ravel(), dec.ravel()
        keep = ra**2 + dec**2 <= (4 * sigma) ** 2
        ra, dec = ra[keep], dec[keep]
        directions = np.stack(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1
        )
        r2 = 2 * (1 - directions[:, 0])
        node_weights = np.exp(-r2 / (2 * sigma**2)) * np.cos(dec)
        node_weights /= np.sum(node_weights)
        beam = beam_from_directions(directions, node_weights, node_weights)

        detpointing, weights = self.pointing("IQU")
        ops.SimRealSpaceConvolve(
            maps=list(maps),
            beam_file=self.write_beam("gauss.npz", beam),
            det_data="conv",
            stokes_weights=weights,
        ).apply(data)
        weights.apply(data)

        smoothed = hp.smoothing(maps, fwhm=fwhm, pol=True)
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                theta, phi, _ = qa.to_iso_angles(ob.detdata[detpointing.quats][det])
                iqu = interp_iqu(smoothed, theta, phi)
                w = ob.detdata[weights.weights][det]
                expected = sum(w[:, i] * iqu[i] for i in range(3))
                nt.assert_allclose(
                    ob.detdata["conv"][det],
                    expected,
                    rtol=0,
                    atol=0.03 * np.std(smoothed[1:]),
                )
        close_data(data)

    def test_errors(self):
        data = create_satellite_data(self.comm, obs_time=1 * u.minute)
        nside = 16
        iqu = list(self.random_map(nside, 10.0))
        beam_I = self.write_beam("err_I.npz", beam_from_directions([1.0, 0, 0], [1.0]))
        beam_IP = self.write_beam(
            "err_IP.npz", beam_from_directions([1.0, 0, 0], [1.0], [1.0])
        )
        _, weights_I = self.pointing("I")
        _, weights_IQU = self.pointing("IQU")
        _, weights_iau = self.pointing("IQU")
        weights_iau.IAU = True
        valid = dict(maps=iqu, beam_file=beam_IP, stokes_weights=weights_IQU)
        bad = [
            (dict(beam_file=None), "'beam_file'"),
            (dict(stokes_weights=None), "'stokes_weights'"),
            (dict(maps=None), "exactly one"),
            (dict(file="unused.fits"), "exactly one"),
            (dict(stokes_weights=weights_iau), "IAU"),
            (dict(maps=iqu[:1]), "components"),
            (dict(stokes_weights=weights_I), "components"),
            (dict(maps=iqu[:2], stokes_weights=weights_I), "components"),
            (dict(beam_file=beam_I), "Q and U weights"),
        ]
        for change, message in bad:
            traits = dict(valid, **change)
            with self.assertRaisesRegex(RuntimeError, message):
                ops.SimRealSpaceConvolve(det_data="conv", **traits).apply(data)
        close_data(data)

    def test_accumulate_view_mask(self):
        # The signal is added to existing data, only inside the view of the
        # detector pointing, and only for unflagged detectors.
        data = create_satellite_data(self.comm, obs_time=1 * u.minute)
        maps = self.random_map(16, 10.0)
        beam = beam_from_directions([1.0, 0.0, 0.0], [1.0])
        detpointing, weights = self.pointing("I")
        detpointing.view = "half"
        for ob in data.obs:
            half = ob.n_local_samples // 2
            ob.intervals["half"] = IntervalList(
                ob.shared[defaults.times], samplespans=[(0, half)]
            )
            ob.detdata.ensure("conv", create_units=u.K)
            ob.detdata["conv"][:] = 1.0
        convolve = ops.SimRealSpaceConvolve(
            maps=[maps[0]],
            beam_file=self.write_beam("accum.npz", beam),
            det_data="conv",
            stokes_weights=weights,
        )
        convolve.apply(data)
        convolve.apply(data)
        weights.apply(data)
        for ob in data.obs:
            half = ob.n_local_samples // 2
            good = ob.select_local_detectors(flagmask=defaults.det_mask_invalid)
            for det in ob.local_detectors:
                conv = ob.detdata["conv"][det]
                if det not in good:
                    nt.assert_array_equal(conv, 1.0)
                    continue
                theta, phi, _ = qa.to_iso_angles(
                    ob.detdata[detpointing.quats][det][:half]
                )
                expected = 1.0 + 2 * hp.get_interp_val(maps[0], theta, phi)
                nt.assert_allclose(
                    conv[:half], expected, rtol=0, atol=1e-12 * np.std(maps[0])
                )
                nt.assert_array_equal(conv[half:], 1.0)
        close_data(data)

    def test_polar_pointing(self):
        # The test observations stay at low latitudes, so the transport near the
        # poles is checked on the beam sum directly, against the two-bearing
        # transport, within 1 degree of both poles.
        nside = 64
        maps = self.random_map(nside, 3.0)
        np.random.seed(2)
        n = 1000
        theta = np.radians(np.random.uniform(0.001, 1.0, n))
        theta[n // 2 :] = np.pi - theta[n // 2 :]
        phi = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, 2 * np.pi, n)
        quats = qa.from_iso_angles(theta, phi, psi)
        center = np.array([[0.0, 0.0, 1.0]])
        result = _beam_sum(quats, nside, maps, center, np.ones(1), np.ones(1))
        expected = interp_iqu(maps, theta, phi)
        for i in range(3):
            nt.assert_allclose(
                result[i], expected[i], rtol=0, atol=1e-10 * np.std(maps)
            )

    def test_pipeline_single(self):
        # A pipeline that runs one detector at a time calls exec once per detector,
        # with the map kept between calls, and gives the same result.
        data = create_satellite_data(self.comm, obs_time=1 * u.minute)
        maps = self.random_map(16, 10.0)
        path = self.write_map("pipeline.fits", maps)
        beam_file = self.write_beam(
            "pipeline.npz", beam_from_directions([1.0, 0.01, 0.0], [1.0], [1.0])
        )
        _, weights = self.pointing("IQU")
        for key, sets in (("direct", ["ALL"]), ("single", ["SINGLE"])):
            convolve = ops.SimRealSpaceConvolve(
                file=path, beam_file=beam_file, det_data=key, stokes_weights=weights
            )
            ops.Pipeline(operators=[convolve], detector_sets=sets).apply(data)
        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                nt.assert_array_equal(
                    ob.detdata["single"][det], ob.detdata["direct"][det]
                )
        close_data(data)
