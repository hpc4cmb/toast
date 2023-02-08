# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .. import ops as ops
from .. import rng
from ..covariance import covariance_apply
from ..pixels import PixelData, PixelDistribution
from ..pixels_io_healpix import write_healpix_fits
from ._helpers import (
    close_data,
    create_fake_sky,
    create_outdir,
    create_satellite_data,
    create_satellite_data_big,
)
from .mpi import MPITestCase


class SimCrossTalkTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)

    def test_xtalk_matrices(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)

        xtalk = ops.CrossTalk(det_data=key)
        xtalk.apply(data)

        invxtalk = ops.MitigateCrossTalk(det_data=key)
        invxtalk.apply(data)
        dets = list(xtalk.xtalk_mat.keys())
        ndet = len(dets)
        M = np.zeros((ndet, ndet))
        invM = np.zeros((ndet, ndet))
        for ii, det in enumerate(dets):
            M[ii, :] = np.array(list(xtalk.xtalk_mat[det].values()))
            M[ii, ii] = 1
            invM[ii, :] = np.array(list(invxtalk.inv_xtalk_mat[det].values()))

        np.testing.assert_almost_equal(invM.dot(M), np.eye(ndet), decimal=4)

        close_data(data)

    def test_xtalk(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)
        detdata_old = data.obs[0].detdata[key].data.copy()

        xtalk = ops.CrossTalk(det_data=key)
        xtalk.apply(data)
        invxtalk = ops.MitigateCrossTalk(det_data=key)
        invxtalk.apply(data)

        np.testing.assert_almost_equal(
            detdata_old, data.obs[0].detdata[key].data, decimal=8
        )

        close_data(data)

    def test_xtalk_big(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data_big(self.comm, pixel_per_process=7)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)

        xtalk = ops.CrossTalk(det_data=key)
        xtalk.apply(data)
        detdata_old = data.obs[0].detdata[key].data.copy()

        xtalk = ops.CrossTalk(det_data=key)
        xtalk.apply(data)

        invxtalk = ops.MitigateCrossTalk(det_data=key)
        invxtalk.apply(data)
        np.testing.assert_almost_equal(
            detdata_old, data.obs[0].detdata[key].data, decimal=8
        )

        close_data(data)

    def test_xtalk_errors(self):
        # Create a fake satellite data set for testing
        data = create_satellite_data(self.comm)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)

        xtalk = ops.CrossTalk(det_data=key)
        xtalk.apply(data)
        detdata_old = data.obs[0].detdata[key].data.copy()

        xtalk = ops.CrossTalk(det_data=key)
        xtalk.apply(data)
        epsilon = 1e-3
        invxtalk = ops.MitigateCrossTalk(det_data=key, error_coefficients=epsilon)
        invxtalk.apply(data)
        np.testing.assert_almost_equal(
            detdata_old, data.obs[0].detdata[key].data, decimal=2
        )

        close_data(data)

    """
    def test_xtalk_file(self):

        # Create a fake satellite data set for testing
        data = create_satellite_data  (
            self.comm )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)
        detdata_old = data.obs[0].detdata[key].data.copy()

        xtalk  = ops.CrossTalk (det_data=key,
            xtalk_mat_file='/Users/peppe/work/satellite_sims/crosstalk/lb_sim_191212.npz' )
        xtalk .apply(data)

        invxtalk  = ops.MitigateCrossTalk (det_data=key,
            xtalk_mat_file='/Users/peppe/work/satellite_sims/crosstalk/lb_sim_191212.npz' )
        invxtalk .apply(data)
        dets= list(xtalk.xtalk_mat.keys() )
        ndet = len (dets )
        M = np.zeros((ndet,ndet ))
        invM = np.zeros((ndet,ndet ))
        for ii,det in enumerate( dets ) :
            M[ii,:]= np.array(list (xtalk.xtalk_mat[det].values() ))
            M[ii,ii]=1
            invM[ii,:]= np.array(list (invxtalk.inv_xtalk_mat[det].values() ))
        np.testing.assert_almost_equal(
            invM.dot(M) , np.eye(ndet ), decimal=4)

        np.testing.assert_almost_equal(
             detdata_old , data.obs[0].detdata[key].data , decimal=8)

        return
    """
