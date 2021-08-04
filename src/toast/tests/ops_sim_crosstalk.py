# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import healpy as hp
from .mpi import MPITestCase

from ..vis import set_matplotlib_backend

from .. import ops as ops
from .. import rng

from ..pixels import PixelDistribution, PixelData

from ..pixels_io import write_healpix_fits

from ..covariance import covariance_apply
from ._helpers import (
    create_outdir,
    create_satellite_data,
    create_satellite_data_big,
    create_fake_sky,
)
class SimCrossTalkTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
    def plot_xtalk_mat(self, matdic ):
        for ob, dets  in matdic.items () :
            ndet = len(dets )
            M = np.zeros((ndet,ndet ))

            for  ii,  values in enumerate(dets.values() ):
                M[ii,:]= values
                M[ii,ii] = 0
        set_matplotlib_backend()
        import matplotlib.pyplot as plt
        outfile = os.path.join(self.outdir, "xtalk_matrix.png")
        plt.imshow(M,cmap='Greys'); plt.show ()
        plt.savefig(outfile)
        plt.close()


    def test_xtalk(self):

        # Create a fake satellite data set for testing
        data = create_satellite_data(
            self.comm)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)

        xtalk  = ops.CrossTalk (det_data=key )
        xtalk .apply(data)
        #import pdb; pdb.set_trace()
        self. plot_xtalk_mat(xtalk.xtalk_mat )
        return

    def test_xtalk_big(self):

        # Create a fake satellite data set for testing
        data = create_satellite_data_big (
            self.comm, pixel_per_process=7)

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)
        # Simulate noise using this model
        key = "my_signal"
        sim_noise = ops.SimNoise(det_data=key)
        sim_noise.apply(data)

        xtalk  = ops.CrossTalk (det_data=key )
        xtalk .apply(data)

        return
