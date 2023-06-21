# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .. import qarray as qa
from .mpi import MPITestCase


class QarrayTest(MPITestCase):
    def setUp(self):
        # data
        self.q1 = np.array([0.50487417, 0.61426059, 0.60118994, 0.07972857])
        self.q1inv = np.array([-0.50487417, -0.61426059, -0.60118994, 0.07972857])
        self.q2 = np.array([0.43561544, 0.33647027, 0.40417115, 0.73052901])
        self.qtonormalize = np.array([[1.0, 2, 3, 4], [2, 3, 4, 5]])
        self.qnormalized = np.array(
            [
                [0.18257419, 0.36514837, 0.54772256, 0.73029674],
                [0.27216553, 0.40824829, 0.54433105, 0.68041382],
            ]
        )
        self.vec = np.array([0.57734543, 0.30271255, 0.75831218])
        self.vec2 = np.array(
            [[0.57734543, 8.30271255, 5.75831218], [1.57734543, 3.30271255, 0.75831218]]
        )
        self.qeasy = np.array([[0.3, 0.3, 0.1, 0.9], [0.3, 0.3, 0.1, 0.9]])
        # results from Quaternion
        # CHANGED SIGN TO COMPLY WITH THE ONLINE QUATERNION CALCULATOR
        self.mult_result = -1 * np.array(
            [-0.44954009, -0.53339352, -0.37370443, 0.61135101]
        )
        self.rot_by_q1 = np.array([0.4176698, 0.84203849, 0.34135482])
        self.rot_by_q2 = np.array([0.8077876, 0.3227185, 0.49328689])

    def test_inv(self):
        np.testing.assert_array_almost_equal(qa.inv(self.q1), self.q1inv)
        return

    def test_norm(self):
        np.testing.assert_array_almost_equal(
            qa.norm(self.q1), self.q1 / np.linalg.norm(self.q1)
        )
        np.testing.assert_array_almost_equal(
            qa.norm(self.qtonormalize), self.qnormalized
        )
        return

    def test_mult_onequaternion(self):
        my_mult_result = qa.mult(self.q1, self.q2)
        np.testing.assert_array_almost_equal(my_mult_result, self.mult_result)
        return

    def test_mult_qarray(self):
        dim = (3, 1)
        qarray1 = np.tile(self.q1, dim)
        qarray2 = np.tile(self.q2, dim)
        my_mult_result = qa.mult(qarray1, qarray2)
        np.testing.assert_array_almost_equal(
            my_mult_result, np.tile(self.mult_result, dim)
        )

        check = qa.mult(self.q1, self.q2)
        res = qa.mult(np.tile(self.q1, 10).reshape((-1, 4)), self.q2)
        np.testing.assert_array_almost_equal(res, np.tile(check, 10).reshape((-1, 4)))

        nulquat = np.array([0.0, 0.0, 0.0, 1.0])
        check = qa.mult(self.q1, nulquat)
        res = qa.mult(np.tile(self.q1, 10).reshape((-1, 4)), nulquat)
        np.testing.assert_array_almost_equal(res, np.tile(check, 10).reshape((-1, 4)))
        return

    def test_rotate_onequaternion(self):
        my_rot_result = qa.rotate(self.q1, self.vec)
        np.testing.assert_array_almost_equal(my_rot_result, self.rot_by_q1)
        return

    def test_rotate_qarray(self):
        my_rot_result = qa.rotate(np.vstack([self.q1, self.q2]), self.vec)
        np.testing.assert_array_almost_equal(
            my_rot_result, np.vstack([self.rot_by_q1, self.rot_by_q2]).reshape((2, 3))
        )

        zaxis = np.array([0.0, 0.0, 1.0])

        nsamp = 1000
        theta = (1.0 / np.pi) * np.arange(nsamp, dtype=np.float64)
        phi = (10.0 / (2.0 * np.pi)) * np.arange(nsamp, dtype=np.float64)
        pa = np.zeros(nsamp, dtype=np.float64)

        quats = qa.from_iso_angles(theta, phi, pa)

        check = np.zeros((nsamp, 3), dtype=np.float64)
        for i in range(nsamp):
            check[i, :] = qa.rotate(quats[i], zaxis)

        dir = qa.rotate(quats, zaxis)

        np.testing.assert_array_almost_equal(dir, check)
        return

    def test_slerp(self):
        q = qa.norm(np.array([[2.0, 3, 4, 5], [6.0, 7, 8, 9]]))
        time = np.array([0.0, 9])
        targettime = np.array([0, 3, 4.5, 9])
        q_interp = qa.slerp(targettime, time, q)
        self.assertEqual(len(q_interp), 4)
        np.testing.assert_array_almost_equal(q_interp[0], q[0])
        np.testing.assert_array_almost_equal(q_interp[-1], q[-1])
        np.testing.assert_array_almost_equal(
            q_interp[1], qa.norm(q[0] * 2 / 3 + q[1] / 3), decimal=4
        )
        np.testing.assert_array_almost_equal(
            q_interp[2], qa.norm((q[0] + q[1]) / 2), decimal=4
        )
        return

    def test_rotation(self):
        np.testing.assert_array_almost_equal(
            qa.rotation(np.array([0.0, 0.0, 1.0]), np.radians(30)),
            np.array([0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))]),
        )
        return

    def test_toaxisangle(self):
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.radians(30.0)
        q = np.array([0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))])
        qaxis, qangle = qa.to_axisangle(q)
        np.testing.assert_array_almost_equal(axis, qaxis)
        self.assertAlmostEqual(angle, qangle)
        return

    def test_exp(self):
        # Exponential test from:
        #     http://world.std.com/~sweetser/java/qcalc/qcalc.html
        np.testing.assert_array_almost_equal(
            qa.exp(self.qeasy),
            np.array(
                [
                    [0.71473568, 0.71473568, 0.23824523, 2.22961712],
                    [0.71473568, 0.71473568, 0.23824523, 2.22961712],
                ]
            ),
        )
        return

    def test_ln(self):
        # Log test from: http://world.std.com/~sweetser/java/qcalc/qcalc.html
        np.testing.assert_array_almost_equal(
            qa.ln(self.qeasy),
            np.array(
                [
                    [0.31041794, 0.31041794, 0.10347265, 0.0],
                    [0.31041794, 0.31041794, 0.10347265, 0.0],
                ]
            ),
        )
        return

    def test_pow(self):
        # Pow test from: http://world.std.com/~sweetser/java/qcalc/qcalc.html
        np.testing.assert_array_almost_equal(
            qa.pow(self.qeasy, 3.0),
            np.array([[0.672, 0.672, 0.224, 0.216], [0.672, 0.672, 0.224, 0.216]]),
        )
        np.testing.assert_array_almost_equal(
            qa.pow(self.qeasy, 0.1),
            np.array(
                [
                    [0.03103127, 0.03103127, 0.01034376, 0.99898305],
                    [0.03103127, 0.03103127, 0.01034376, 0.99898305],
                ]
            ),
        )
        return

    def test_torotmat(self):
        # Rotmat test from Quaternion
        np.testing.assert_array_almost_equal(
            qa.to_rotmat(self.qeasy[0]),
            np.array(
                [
                    [8.00000000e-01, -2.77555756e-17, 6.00000000e-01],
                    [3.60000000e-01, 8.00000000e-01, -4.80000000e-01],
                    [-4.80000000e-01, 6.00000000e-01, 6.40000000e-01],
                ]
            ),
        )
        return

    def test_fromrotmat(self):
        np.testing.assert_array_almost_equal(
            self.qeasy[0], qa.from_rotmat(qa.to_rotmat(self.qeasy[0]))
        )
        return

    def test_fromvectors(self):
        angle = np.radians(30)

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([np.cos(angle), np.sin(angle), 0])
        np.testing.assert_array_almost_equal(
            qa.from_vectors(v1, v2),
            np.array([0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))]),
        )

        nulquat = np.array([0.0, 0.0, 0.0, 1.0])
        zaxis = np.array([0.0, 0.0, 1.0])
        check = qa.from_vectors(zaxis, zaxis)
        np.testing.assert_array_almost_equal(check, nulquat)

        q = qa.from_vectors(
            np.tile(v1, 3).reshape(-1, 3), np.tile(v2, 3).reshape(-1, 3)
        )

        comp = np.tile(
            np.array([0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))]), 3
        ).reshape(-1, 4)

        np.testing.assert_array_almost_equal(q, comp)
        return

    def test_fp(self):
        xaxis = np.array([1, 0, 0], dtype=np.float64)
        zaxis = np.array([0, 0, 1], dtype=np.float64)

        radius = np.deg2rad(10.0)

        for pos in range(6):
            posang = np.deg2rad(pos * 60.0)
            posrot = qa.rotation(zaxis, posang + np.pi / 2.0)
            radrot = qa.rotation(xaxis, radius)
            detrot = qa.mult(posrot, radrot)

            detdir = qa.rotate(detrot, zaxis)

            check = np.dot(detdir, zaxis)

            np.testing.assert_almost_equal(check, np.cos(radius))
        return

    def check_iso(self, actual, desired):
        """Check that input / output angles are the same."""
        check_theta = np.array(actual[0])
        check_phi = np.array(actual[1])
        check_psi = np.array(actual[2])
        theta = np.array(desired[0])
        phi = np.array(desired[1])
        psi = np.array(desired[2])
        # Convert the phi / psi angles to +/- PI
        try:
            lt = len(theta)
            extreme_theta = np.logical_or(
                np.isclose(theta, 0.0), np.isclose(theta, np.pi)
            )
            psi[extreme_theta] += phi[extreme_theta]
            phi[extreme_theta] = 0.0
            check_psi[extreme_theta] += check_phi[extreme_theta]
            check_phi[extreme_theta] = 0.0
            for ang in phi, psi, check_phi, check_psi:
                high = ang > np.pi
                low = ang <= -np.pi
                ang[high] -= 2 * np.pi
                ang[low] += 2 * np.pi
        except TypeError:
            # scalar values
            if np.isclose(theta, 0.0) or np.isclose(theta, np.pi):
                psi += phi
                phi = 0.0
                check_psi += check_phi
                check_phi = 0.0
            for ang in phi, psi, check_phi, check_psi:
                if ang > np.pi:
                    ang -= 2 * np.pi
                if ang <= -np.pi:
                    ang += 2 * np.pi
        if not np.allclose(check_theta, theta, rtol=1.0e-7, atol=1.0e-6):
            print(f"ISO theta check failed:")
            print(f"{np.transpose((check_theta, theta))}", flush=True)
            raise ValueError("ISO theta values not equal")
        if not np.allclose(check_phi, phi, rtol=1.0e-7, atol=1.0e-6):
            print(f"ISO phi check failed:")
            print(f"{np.transpose((check_phi, phi))}", flush=True)
            raise ValueError("ISO phi values not equal")
        if not np.allclose(check_psi, psi, rtol=1.0e-7, atol=1.0e-6):
            print(f"ISO psi check failed:")
            print(f"{np.transpose((check_psi, psi))}", flush=True)
            raise ValueError("ISO psi values not equal")

    def check_zx(self, actual, desired):
        """Check that input / output vectors are the same."""
        check_z = np.array(actual[0])
        check_x = np.array(actual[1])
        z = np.array(desired[0])
        x = np.array(desired[1])
        if not np.allclose(check_z, z, rtol=1.0e-7, atol=1.0e-6):
            print(f"Z check failed:")
            print(f"{np.transpose((check_z, z))}", flush=True)
            raise ValueError("Z values not equal")
        if not np.allclose(check_x, x, rtol=1.0e-7, atol=1.0e-6):
            print(f"X check failed:")
            print(f"{np.transpose((check_x, x))}", flush=True)
            raise ValueError("X values not equal")

    def test_angles(self):
        xaxis = np.array([1.0, 0.0, 0.0])
        zaxis = np.array([0.0, 0.0, 1.0])

        # Test a few specific angle cases
        all_theta = [
            0.0,
            np.pi / 2,
            np.pi / 4,
            np.pi / 4,
            np.pi,
            np.pi / 2,
        ]
        all_phi = [
            0.0,
            0.0,
            np.pi / 4,
            np.pi / 4,
            np.pi,
            np.pi / 2,
        ]
        all_psi = [
            0.0,
            0.0,
            0.0,
            np.pi / 2,
            np.pi,
            np.pi / 2,
        ]
        all_z = [
            np.array([0.0, 0.0, 1.0]),  # no change
            np.array([1.0, 0.0, 0.0]),  # rotated to the x-axis
            np.array(
                [
                    np.cos(np.pi / 4) * np.cos(np.pi / 4),
                    np.cos(np.pi / 4) * np.sin(np.pi / 4),
                    np.sin(np.pi / 4),
                ]
            ),
            np.array(
                [
                    np.cos(np.pi / 4) * np.cos(np.pi / 4),
                    np.cos(np.pi / 4) * np.sin(np.pi / 4),
                    np.sin(np.pi / 4),
                ]
            ),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        all_x = [
            np.array([1.0, 0.0, 0.0]),  # no change
            np.array([0.0, 0.0, -1.0]),  # rotated to -z axis
            np.array(
                [
                    np.cos(np.pi / 4) * np.cos(np.pi / 4),
                    np.cos(np.pi / 4) * np.sin(np.pi / 4),
                    -np.sin(np.pi / 4),
                ]
            ),
            np.array(
                [
                    -np.cos(np.pi / 4),
                    np.sin(np.pi / 4),
                    0.0,
                ]
            ),
            np.array([-1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
        ]
        for theta, phi, psi, vz, vx in zip(all_theta, all_phi, all_psi, all_z, all_x):
            quat = qa.from_iso_angles(theta, phi, psi)
            zrot = qa.rotate(quat, zaxis)
            xrot = qa.rotate(quat, xaxis)
            # print(f"{(zrot, xrot)} {(vz, vx)}")
            self.check_zx((zrot, xrot), (vz, vx))

            check_theta, check_phi, check_psi = qa.to_iso_angles(quat)
            # print(f"  {(check_theta, check_phi, check_psi)} {(theta, phi, psi)}")
            self.check_iso((check_theta, check_phi, check_psi), (theta, phi, psi))

        ntheta = 3
        nphi = 3
        npsi = 3
        n = ntheta * nphi * npsi

        theta = np.zeros(n, dtype=np.float64)
        phi = np.zeros(n, dtype=np.float64)
        psi = np.zeros(n, dtype=np.float64)

        for i in range(ntheta):
            toff = i * nphi * npsi
            for j in range(nphi):
                poff = j * npsi
                for k in range(npsi):
                    theta[toff + poff + k] = i * np.pi / float(ntheta)
                    phi[toff + poff + k] = j * 2.0 * np.pi / float(nphi)
                    psi[toff + poff + k] = k * 2.0 * np.pi / float(npsi)
        # print(f"Input {np.transpose((theta, phi, psi))}")
        quat = qa.from_iso_angles(theta, phi, psi)
        dir = qa.rotate(quat, np.tile(zaxis, n).reshape((n, 3)))
        orient = qa.rotate(quat, np.tile(xaxis, n).reshape((n, 3)))
        # print(f"Vec {np.transpose((dir, orient), axes=(1, 0, 2))}")

        check_theta, check_phi, check_psi = qa.to_iso_angles(quat)
        # print(f"Check {np.transpose((check_theta, check_phi, check_psi))}")
        self.check_iso((check_theta, check_phi, check_psi), (theta, phi, psi))

    def test_angles_zero(self):
        # Test roundtrip ISO angle conversions at the origin
        psi = np.array(
            [
                0.0,
                45.0,
                90.0,
                135.0,
                180.0,
                225.0,
                270.0,
                315.0,
                360.0,
            ]
        )
        psi *= np.pi / 180.0
        theta = np.zeros_like(psi)
        phi = np.zeros_like(psi)
        quat = qa.from_iso_angles(theta, phi, psi)
        check_theta, check_phi, check_psi = qa.to_iso_angles(quat)
        self.check_iso((check_theta, check_phi, check_psi), (theta, phi, psi))

    def test_depths(self):
        # Verify that qarray methods preserve the depths of their inputs
        np.testing.assert_equal(qa.mult(self.q1, self.q2).shape, (4,))
        np.testing.assert_equal(qa.mult(np.atleast_2d(self.q1), self.q2).shape, (1, 4))

        np.testing.assert_equal(qa.inv(self.q1).shape, (4,))
        np.testing.assert_equal(qa.inv(np.atleast_2d(self.q1)).shape, (1, 4))

        np.testing.assert_equal(np.shape(qa.amplitude(self.q1)), ())
        np.testing.assert_equal(np.shape(qa.amplitude(np.atleast_2d(self.q1))), (1,))

        np.testing.assert_equal(qa.norm(self.q1).shape, (4,))
        np.testing.assert_equal(qa.norm(np.atleast_2d(self.q1)).shape, (1, 4))

        np.testing.assert_equal(qa.rotate(self.q1, self.vec).shape, (3,))
        np.testing.assert_equal(
            qa.rotate(np.atleast_2d(self.q1), self.vec).shape, (1, 3)
        )
        np.testing.assert_equal(
            qa.rotate(self.q1, np.atleast_2d(self.vec)).shape, (1, 3)
        )

        q = qa.norm(np.array([[2.0, 3, 4, 5], [6.0, 7, 8, 9]]))
        time = np.array([0.0, 9])
        np.testing.assert_equal(np.shape(qa.slerp(0, time, q)), (4,))
        np.testing.assert_equal(np.shape(qa.slerp([0], time, q)), (1, 4))

        np.testing.assert_equal(qa.exp(self.q1).shape, (4,))
        np.testing.assert_equal(qa.exp(np.atleast_2d(self.q1)).shape, (1, 4))

        np.testing.assert_equal(qa.ln(self.q1).shape, (4,))
        np.testing.assert_equal(qa.ln(np.atleast_2d(self.q1)).shape, (1, 4))

        np.testing.assert_equal(qa.pow(self.q1, 2).shape, (4,))
        np.testing.assert_equal(qa.pow(np.atleast_2d(self.q1), 2).shape, (1, 4))
        np.testing.assert_equal(qa.pow(self.q1, [2]).shape, (1, 4))
        np.testing.assert_equal(qa.pow(np.atleast_2d(self.q1), [2]).shape, (1, 4))

        np.testing.assert_equal(qa.rotation([0, 0, 1], np.pi).shape, (4,))
        np.testing.assert_equal(qa.rotation([[0, 0, 1]], np.pi).shape, (1, 4))
        np.testing.assert_equal(qa.rotation([0, 0, 1], [np.pi]).shape, (1, 4))
        np.testing.assert_equal(qa.rotation([[0, 0, 1]], [np.pi]).shape, (1, 4))

        ret1 = qa.to_axisangle(self.q1)
        ret2 = qa.to_axisangle(np.atleast_2d(self.q1))
        np.testing.assert_equal(ret1[0].shape, (3,))
        np.testing.assert_equal(np.shape(ret1[1]), ())
        np.testing.assert_equal(ret2[0].shape, (1, 3))
        np.testing.assert_equal(np.shape(ret2[1]), (1,))

        np.testing.assert_equal(np.shape(qa.from_iso_angles(0, 0, 0)), (4,))
        np.testing.assert_equal(np.shape(qa.from_iso_angles([0], 0, 0)), (1, 4))
        np.testing.assert_equal(np.shape(qa.from_iso_angles([0], [0], [0])), (1, 4))

        ret1 = qa.to_iso_angles(self.q1)
        ret2 = qa.to_iso_angles(np.atleast_2d(self.q1))
        np.testing.assert_equal(np.shape(ret1[0]), ())
        np.testing.assert_equal(np.shape(ret1[1]), ())
        np.testing.assert_equal(np.shape(ret1[2]), ())
        np.testing.assert_equal(np.shape(ret2[0]), (1,))
        np.testing.assert_equal(np.shape(ret2[1]), (1,))
        np.testing.assert_equal(np.shape(ret2[2]), (1,))
        return
