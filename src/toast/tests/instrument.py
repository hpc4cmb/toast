# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u
from astropy.table import Column, QTable

from ..instrument import Focalplane
from ..instrument_coords import iso_to_xieta, quat_to_xieta, xieta_to_iso, xieta_to_quat
from ..instrument_sim import (
    fake_hexagon_focalplane,
    fake_rhombihex_focalplane,
    hex_gamma_angles_qu,
    hex_gamma_angles_radial,
    hex_layout,
)
from ..io import H5File
from ._helpers import create_outdir
from .mpi import MPITestCase


class InstrumentTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.xieta_space = np.pi / 6

    def plot_positions(self, outfile, theta, phi, psi):
        import matplotlib.pyplot as plt

        space_deg = self.xieta_space * 180.0 / np.pi
        fwhm = 0.3 * space_deg
        detradius = 0.5 * fwhm
        size_deg = 3 * space_deg
        xfigsize = int(size_deg) + 1
        yfigsize = int(size_deg) + 1
        figdpi = 75

        # Compute the font size to use for detector labels
        fontpix = 0.5 * figdpi
        fontpt = int(0.75 * fontpix)

        fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
        ax = fig.add_subplot(1, 1, 1)

        half_width = 0.6 * size_deg
        half_height = 0.6 * size_deg
        ax.set_xlabel("Boresight Xi Degrees", fontsize="medium")
        ax.set_ylabel("Boresight Eta Degrees", fontsize="medium")
        ax.set_xlim([-half_width, half_width])
        ax.set_ylim([-half_height, half_height])

        for th, ph, ps in zip(theta, phi, psi):
            xi, eta, gamma = iso_to_xieta(th, ph, ps)
            th_deg = th * 180.0 / np.pi
            ph_deg = ph * 180.0 / np.pi
            ps_deg = ps * 180.0 / np.pi

            xpos = xi * 180.0 / np.pi
            ypos = eta * 180.0 / np.pi
            # Polang is plotted relative to visualization x/y coords
            polang = 1.5 * np.pi - gamma

            detface = "none"
            circ = plt.Circle((xpos, ypos), radius=detradius, fc=detface, ec="k")
            ax.add_artist(circ)

            ascale = 2.0

            xtail = xpos - ascale * detradius * np.cos(polang)
            ytail = ypos - ascale * detradius * np.sin(polang)
            dx = ascale * 2.0 * detradius * np.cos(polang)
            dy = ascale * 2.0 * detradius * np.sin(polang)

            detcolor = "black"
            ax.arrow(
                xtail,
                ytail,
                dx,
                dy,
                width=0.1 * detradius,
                head_width=0.3 * detradius,
                head_length=0.3 * detradius,
                fc=detcolor,
                ec=detcolor,
                length_includes_head=True,
            )
        fig.suptitle("Focalplane on Sky Seen by Observer")
        plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
        plt.close()

    def check_xieta(self, actual, desired):
        """Check that input / output angles are the same."""
        check_xi = np.array(actual[0])
        check_eta = np.array(actual[1])
        check_gamma = np.array(actual[2])
        xi = np.array(desired[0])
        eta = np.array(desired[1])
        gamma = np.array(desired[2])
        # Convert the gamma angles to +/- PI
        try:
            lt = len(gamma)
            for ang in gamma, check_gamma:
                high = ang > np.pi
                low = ang < -np.pi
                ang[high] -= 2 * np.pi
                ang[low] += 2 * np.pi
        except TypeError:
            # scalar values
            for ang in gamma, check_gamma:
                if ang > np.pi:
                    ang -= 2 * np.pi
                if ang < -np.pi:
                    ang += 2 * np.pi
        if not np.allclose(check_xi, xi, rtol=1.0e-7, atol=1.0e-6):
            print(f"XiEta xi check failed:")
            print(f"{np.transpose((check_xi, xi))}")
            print(f" = {np.transpose((check_xi*180/np.pi, xi*180/np.pi))}")
            raise ValueError("Xi values not equal")
        if not np.allclose(check_eta, eta, rtol=1.0e-7, atol=1.0e-6):
            print(f"XiEta Eta check failed:")
            print(f"{np.transpose((check_eta, eta))}")
            print(f" = {np.transpose((check_eta*180/np.pi, eta*180/np.pi))}")
            raise ValueError("Eta values not equal")
        if not np.allclose(check_gamma, gamma, rtol=1.0e-7, atol=1.0e-6):
            print(f"XiEta gamma check failed:")
            print(f"{np.transpose((check_gamma, gamma))}")
            print(f" = {np.transpose((check_gamma*180/np.pi, gamma*180/np.pi))}")
            raise ValueError("Gamma values not equal")

    def test_coords(self):
        xi = np.tile(np.array([-self.xieta_space, 0.0, self.xieta_space]), 3)
        eta = np.repeat(np.array([-self.xieta_space, 0.0, self.xieta_space]), 3)
        gamma = np.array(
            [
                1 * np.pi / 4,
                0 * np.pi / 4,
                7 * np.pi / 4,
                2 * np.pi / 4,
                0 * np.pi / 4,
                6 * np.pi / 4,
                3 * np.pi / 4,
                4 * np.pi / 4,
                5 * np.pi / 4,
            ]
        )
        theta, phi, psi = xieta_to_iso(xi, eta, gamma)
        if self.comm is None or self.comm.rank == 0:
            self.plot_positions(
                os.path.join(self.outdir, "test_coords.pdf"), theta, phi, psi
            )
        check_xi, check_eta, check_gamma = iso_to_xieta(theta, phi, psi)
        self.check_xieta((check_xi, check_eta, check_gamma), (xi, eta, gamma))

        # Test that the gamma angle is recovered correctly at the origin
        xi[:] = 0.0
        eta[:] = 0.0
        qt = xieta_to_quat(xi, eta, gamma)
        check_xi, check_eta, check_gamma = quat_to_xieta(qt)
        self.check_xieta((check_xi, check_eta, check_gamma), (xi, eta, gamma))

    def test_hex_radial_layout(self):
        npix = 7
        pol = hex_gamma_angles_radial(npix)
        space_deg = self.xieta_space * 180.0 / np.pi
        fwhm = 0.3 * space_deg
        fp = hex_layout(npix, space_deg * u.degree, "", "", pol)
        quat = np.array([fp[f"{p}"]["quat"] for p in range(npix)])
        xi, eta, gamma = quat_to_xieta(quat)
        theta, phi, psi = xieta_to_iso(xi, eta, gamma)
        if self.comm is None or self.comm.rank == 0:
            self.plot_positions(
                os.path.join(self.outdir, "hex_radial_layout.pdf"), theta, phi, psi
            )

    def test_hex_qu_layout(self):
        npix = 7
        pol = hex_gamma_angles_qu(npix)
        space_deg = self.xieta_space * 180.0 / np.pi
        fwhm = 0.3 * space_deg
        fp = hex_layout(npix, space_deg * u.degree, "", "", pol)
        quat = np.array([fp[f"{p}"]["quat"] for p in range(npix)])
        xi, eta, gamma = quat_to_xieta(quat)
        theta, phi, psi = xieta_to_iso(xi, eta, gamma)
        if self.comm is None or self.comm.rank == 0:
            self.plot_positions(
                os.path.join(self.outdir, "hex_qu_layout.pdf"), theta, phi, psi
            )

    def test_focalplane(self):
        names = ["det_01a", "det_01b", "det_02a", "det_02b"]
        quats = [np.array([0, 0, 0, 1], dtype=np.float64) for x in range(len(names))]
        detdata = QTable([names, quats], names=["name", "quat"])
        fp = Focalplane(detector_data=detdata, sample_rate=10.0 * u.Hz)

        fp_file = os.path.join(self.outdir, "focalplane.h5")
        check_file = os.path.join(self.outdir, "check.h5")

        with H5File(fp_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()

        newfp = Focalplane()

        with H5File(fp_file, "r", comm=self.comm) as f:
            newfp.load_hdf5(f.handle, comm=self.comm)

        self.assertTrue(newfp == fp)

    def test_focalplane_full(self):
        names = ["det_01a", "det_01b", "det_02a", "det_02b"]
        quats = [np.array([0, 0, 0, 1], dtype=np.float64) for x in range(len(names))]
        ndet = len(names)
        # Noise parameters (optional)
        psd_fmin = np.ones(ndet) * 1e-5 * u.Hz
        psd_fknee = np.ones(ndet) * 1e-2 * u.Hz
        psd_alpha = np.ones(ndet) * 1.0
        psd_NET = np.ones(ndet) * 1e-3 * u.K * u.s**0.5
        # Bandpass parameters (optional)
        bandcenter = np.ones(ndet) * 1e2 * u.GHz
        bandwidth = bandcenter * 0.1

        detdata = QTable(
            [
                names,
                quats,
                psd_fmin,
                psd_fknee,
                psd_alpha,
                psd_NET,
                bandcenter,
                bandwidth,
            ],
            names=[
                "name",
                "quat",
                "psd_fmin",
                "psd_fknee",
                "psd_alpha",
                "psd_net",
                "bandcenter",
                "bandwidth",
            ],
        )
        fp = Focalplane(detector_data=detdata, sample_rate=10.0 * u.Hz)

        fp_file = os.path.join(self.outdir, "focalplane_full.h5")
        check_file = os.path.join(self.outdir, "check_full.h5")

        with H5File(fp_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()

        newfp = Focalplane()

        with H5File(fp_file, "r", comm=self.comm) as f:
            newfp.load_hdf5(f.handle, comm=self.comm)

        # Test convolving with bandpass
        freqs = np.linspace(50, 150, 100) * u.GHz
        values = np.linspace(0, 1, 100)
        result1 = newfp.bandpass.convolve(names[-1], freqs, values, rj=False)
        result2 = newfp.bandpass.convolve(names[-1], freqs, values, rj=True)

        with H5File(check_file, "w", comm=self.comm) as f:
            newfp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()

    def test_sim_focalplane_hex(self):
        # For visual checks, set the FWHM to the size of the
        # close-packed diameter.

        # Number of hex positions
        n_pix = 7

        # Number of positions across the long axis
        n_pos_width = 3

        # Overall width of the projection
        width = 2.0 * u.degree

        # FWHM
        fwhm = width / n_pos_width

        fp = fake_hexagon_focalplane(
            n_pix=n_pix,
            width=width,
            sample_rate=100.0 * u.Hz,
            epsilon=0.05,
            fwhm=fwhm,
            bandcenter=150 * u.Hz,
            bandwidth=20 * u.Hz,
            psd_net=0.05 * u.K * np.sqrt(1 * u.second),
            psd_fmin=1.0e-5 * u.Hz,
            psd_alpha=1.2,
            psd_fknee=0.05 * u.Hz,
        )

        fake_file = os.path.join(self.outdir, "fake_hex.h5")
        with H5File(fake_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is None or self.comm.rank == 0:
            from ..instrument_sim import plot_focalplane

            pltfile = os.path.join(self.outdir, "fake_hex_xyz.pdf")
            fig = plot_focalplane(
                fp,
                width=3.0 * u.degree,
                height=3.0 * u.degree,
                outfile=pltfile,
                show_labels=True,
                show_centers=True,
                show_gamma=True,
            )
            pltfile = os.path.join(self.outdir, "fake_hex_xieta.pdf")
            fig = plot_focalplane(
                fp,
                width=3.0 * u.degree,
                height=3.0 * u.degree,
                outfile=pltfile,
                show_labels=True,
                show_centers=True,
                xieta=True,
                show_gamma=True,
            )
            del fig

        if self.comm is not None:
            self.comm.barrier()

    def test_sim_focalplane_rhombihex(self):
        # Number of per-rhombus positions
        n_pix_rhombus = 16

        # Number of positions across the short axis of one rhombus
        n_pos_short = 4

        # Overall width of the total focalplane
        width = 2.0 * u.degree

        # FWHM.  Just set to support cleaner plot.
        fwhm = 0.5 * width / (n_pos_short + 1)

        fp = fake_rhombihex_focalplane(
            n_pix_rhombus=n_pix_rhombus,
            width=width,
            sample_rate=100.0 * u.Hz,
            epsilon=0.05,
            fwhm=fwhm,
            bandcenter=150 * u.Hz,
            bandwidth=20 * u.Hz,
            psd_net=0.05 * u.K * np.sqrt(1 * u.second),
            psd_fmin=1.0e-5 * u.Hz,
            psd_alpha=1.2,
            psd_fknee=0.05 * u.Hz,
        )

        fake_file = os.path.join(self.outdir, "fake_rhombihex.h5")
        with H5File(fake_file, "w", comm=self.comm) as f:
            fp.save_hdf5(f.handle, comm=self.comm)

        if self.comm is None or self.comm.rank == 0:
            from ..instrument_sim import plot_focalplane

            pltfile = os.path.join(self.outdir, "fake_rhombihex_xyz.pdf")
            fig = plot_focalplane(
                fp,
                width=3.0 * u.degree,
                height=3.0 * u.degree,
                outfile=pltfile,
                show_labels=True,
                show_centers=True,
                show_gamma=True,
            )
            pltfile = os.path.join(self.outdir, "fake_rhombihex_xieta.pdf")
            fig = plot_focalplane(
                fp,
                width=3.0 * u.degree,
                height=3.0 * u.degree,
                outfile=pltfile,
                show_labels=True,
                show_centers=True,
                xieta=True,
                show_gamma=True,
            )
            del fig

        if self.comm is not None:
            self.comm.barrier()
