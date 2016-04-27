# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import sys
import os

import numpy as np

from toast.mpirunner import MPITestCase

#import quaternionarray as qarray
import toast.tod.qarray as qarray


class QarrayTest(MPITestCase):

    def setUp(self):
        #data
        self.q1 = np.array([[ 0.50487417,  0.61426059,  0.60118994,  0.07972857]])
        self.q1inv = np.array([[ -0.50487417,  -0.61426059,  -0.60118994,  0.07972857]])
        self.q2 = np.array([[ 0.43561544,  0.33647027,  0.40417115,  0.73052901]])
        self.q1_2 = np.array([[ 0.50487417,  0.61426059,  0.60118994,  0.07972857],[ 0.43561544,  0.33647027,  0.40417115,  0.73052901]])
        self.qtonormalize = np.array([[1.,2,3,4],[2,3,4,5]])
        self.qnormalized = np.array([[0.18257419,  0.36514837,  0.54772256,  0.73029674],[ 0.27216553,  0.40824829,  0.54433105,  0.68041382]])
        self.vec = np.array([ 0.57734543,  0.30271255,  0.75831218])
        self.vec_2d = np.array([[ 0.57734543,  0.30271255,  0.75831218]]) # 2d array required
        self.vec2 = np.array([[ 0.57734543,  8.30271255,  5.75831218, 0.87937215],[ 1.57734543,  3.30271255,  0.75831218, 0.67128248]])
        self.qeasy = np.array([[.3,.3,.1,.9],[.3,.3,.1,.9]])
        #results from Quaternion CHANGED SIGN TO COMPLY WITH THE ONLINE QUATERNION CALCULATOR
        self.mult_result = -1*np.array([[-0.44954009, -0.53339352, -0.37370443,  0.61135101]])
        self.rot_by_q1 = np.array([[0.4176698, 0.84203849, 0.34135482]])
        self.rot_by_q2 = np.array([[0.8077876, 0.3227185, 0.49328689]])

        
    def test_arraylist_dot_onedimarrays(self):
        q_result = qarray.arraylist_dot(self.vec_2d, (self.vec_2d +1))
        np.testing.assert_array_almost_equal(q_result, np.dot(self.vec, self.vec +1)) 

    def test_arraylist_dot_multidim(self):
        q_result = qarray.arraylist_dot(self.vec2, (self.vec2 +1))
        result = np.hstack(np.dot(v1,v2) for v1,v2 in zip(self.vec2, self.vec2+1))
        np.testing.assert_array_almost_equal(q_result.flatten(), result.flatten()) 

    def test_inv(self):
        q_result = qarray.inv(self.q1)
        np.testing.assert_array_almost_equal(q_result, self.q1inv)

    def test_norm(self):
        q_result1 = qarray.norm(self.q1)
        q_result2 = qarray.norm(self.qtonormalize)
        np.testing.assert_array_almost_equal(q_result1, self.q1/np.linalg.norm(self.q1))
        np.testing.assert_array_almost_equal(q_result2, self.qnormalized)

    def test_norm_inplace(self):
        q1 = self.q1
        qarray.norm_inplace(q1)
        qarray.norm_inplace(self.qtonormalize)
        np.testing.assert_array_almost_equal(q1, q1/np.linalg.norm(q1))
        np.testing.assert_array_almost_equal(self.qtonormalize , self.qnormalized)

    def test_mult_onequaternion(self):
        q_result = qarray.mult(self.q1, self.q2)
        self.assertEquals( q_result.shape[0], 1)
        self.assertEquals( q_result.shape[1], 4)
        np.testing.assert_array_almost_equal(q_result , self.mult_result)

    def test_mult_qarray(self):
        dim = (3, 1)
        qarray1 = np.tile(self.q1, dim)
        qarray2 = np.tile(self.q2, dim)
        q_result = qarray.mult(qarray1, qarray2)
        np.testing.assert_array_almost_equal(q_result , np.tile(self.mult_result,dim))

    def test_rotate_onequaternion(self):
        vec_result = qarray.rotate(self.q1, self.vec_2d)
        np.testing.assert_array_almost_equal(vec_result , self.rot_by_q1)
        
    def test_rotate_qarray(self):
        vec_result = qarray.rotate(np.vstack([self.q1,self.q2]), np.vstack([self.vec_2d, self.vec_2d]))
        np.testing.assert_array_almost_equal(vec_result , np.vstack([self.rot_by_q1, self.rot_by_q2]))

    def test_slerp(self):
        q = qarray.norm(np.array([[2., 3, 4, 5],
                      [6, 7, 8, 9]]))
        time = np.array([0., 9])
        targettime = np.array([0, 3, 4.5, 9])
        q_interp = qarray.slerp(targettime, time, q)
        self.assertEquals(q_interp.shape[1], 4)

        np.testing.assert_array_almost_equal(q_interp[0], q[0])
        np.testing.assert_array_almost_equal(q_interp[-1], q[-1])
        np.testing.assert_array_almost_equal(q_interp[1], qarray.norm( ( (q[0]*2 + q[1])/3 ).reshape(1,4) )[0], decimal=4 )
        np.testing.assert_array_almost_equal(q_interp[2], qarray.norm( ( (q[0] + q[1])/2 ).reshape(1,4) )[0], decimal=4 )

    def test_rotation(self):
        q_result = qarray.rotation(np.array([[0.,0.,1.]]), np.array([np.radians(30)]))
        np.testing.assert_array_almost_equal(
            q_result,  np.array([[0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))]])
            )

    def test_exp(self):
        """Exponential test from: http://world.std.com/~sweetser/java/qcalc/qcalc.html"""
        q_result = qarray.exp(self.qeasy)
        np.testing.assert_array_almost_equal(
            q_result,  np.array([[ 0.71473568,  0.71473568,  0.23824523,  2.22961712],[ 0.71473568,  0.71473568,  0.23824523,  2.22961712]])
            )

    def test_ln(self):
        """Log test from: http://world.std.com/~sweetser/java/qcalc/qcalc.html"""
        q_result = qarray.ln(self.qeasy)
        np.testing.assert_array_almost_equal(
            q_result,  np.array([[ 0.31041794,  0.31041794,  0.10347265,  0.        ],[ 0.31041794,  0.31041794,  0.10347265,  0.        ]])
            )

    def test_pow(self):
        """Pow test from: http://world.std.com/~sweetser/java/qcalc/qcalc.html"""
        pow_index = np.array([3.,3.])
        q_result = qarray.pow(self.qeasy, pow_index)
        np.testing.assert_array_almost_equal(
            q_result, np.array([[ 0.672,  0.672,  0.224,  0.216],[ 0.672,  0.672,  0.224,  0.216]])
            )
        pow_index = np.array([.1,.1])
        q_result = qarray.pow(self.qeasy, pow_index)
        np.testing.assert_array_almost_equal(
            q_result, np.array([[ 0.03103127,  0.03103127,  0.01034376,  0.99898305],[ 0.03103127,  0.03103127,  0.01034376,  0.99898305]])
            )

    def test_toaxisangle(self):
        axis = np.array([0.,0.,1.])
        angle = np.radians(30)
        q = np.array([0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))])
        qaxis, qangle = qarray.to_axisangle(q)
        np.testing.assert_array_almost_equal(axis, qaxis)
        self.assertAlmostEqual(angle, qangle)

    def test_torotmat(self):
        """Rotmat test from Quaternion"""
        rotmat = qarray.to_rotmat(self.qeasy[0])
        np.testing.assert_array_almost_equal(rotmat,
                                            np.array([[  8.00000000e-01,  -2.77555756e-17,   6.00000000e-01],
                                                   [  3.60000000e-01,   8.00000000e-01,  -4.80000000e-01],
                                                   [ -4.80000000e-01,   6.00000000e-01,   6.40000000e-01]])
        )

    def test_fromrotmat(self):
        rotmat = qarray.to_rotmat(self.qeasy[0])
        q_result = qarray.from_rotmat(rotmat)
        np.testing.assert_array_almost_equal(self.qeasy[0], q_result)

    def test_fromvectors(self):
        axis = np.array([0,0,1])
        angle = np.radians(30)
        v1 = np.array([1., 0, 0])
        v2 = np.array([np.cos(angle), np.sin(angle), 0])
        q_result = qarray.from_vectors(v1, v2)
        np.testing.assert_array_almost_equal(q_result, np.array([0, 0, np.sin(np.radians(15)), np.cos(np.radians(15))]))

