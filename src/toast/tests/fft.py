# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from ..fft import r1d_backward, r1d_forward
from ..rng import random
from ._helpers import create_outdir
from .mpi import MPITestCase


class FFTTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        self.length = 65536
        self.input_one = random(self.length, counter=[0, 0], key=[0, 0])
        self.compare_one = np.copy(self.input_one)
        self.nbatch = 5
        self.input_batch = np.zeros((self.nbatch, self.length), dtype=np.float64)
        for b in range(self.nbatch):
            self.input_batch[b, :] = random(self.length, counter=[0, 0], key=[0, b])
        self.compare_batch = np.array(self.input_batch)

    def test_roundtrip(self):
        output = r1d_forward(self.input_one)
        check = r1d_backward(output)

        if (self.comm is None) or (self.comm.rank == 0):
            # One process dumps debugging info
            import matplotlib.pyplot as plt

            savefile = os.path.join(self.outdir, "out_one_fft.txt")
            np.savetxt(savefile, np.transpose([self.compare_one, check]), delimiter=" ")
            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(np.arange(self.length), check, c="red", label="Output")
            ax.plot(np.arange(self.length), self.compare_one, c="black", label="Input")
            ax.legend(loc=1)
            plt.title("FFT One Input and Output")
            savefile = os.path.join(self.outdir, "out_one_fft.png")
            plt.savefig(savefile)
            plt.close()

        np.testing.assert_array_almost_equal(check, self.compare_one)

        output = r1d_forward(self.input_batch)
        check = r1d_backward(output)

        if (self.comm is None) or (self.comm.rank == 0):
            # One process dumps debugging info
            import matplotlib.pyplot as plt

            for b in range(self.nbatch):
                savefile = os.path.join(self.outdir, "out_batch_{}_fft.txt".format(b))
                np.savetxt(
                    savefile,
                    np.transpose([self.compare_batch[b], check[b]]),
                    delimiter=" ",
                )
                fig = plt.figure(figsize=(12, 8), dpi=72)
                ax = fig.add_subplot(1, 1, 1, aspect="auto")
                ax.plot(np.arange(self.length), check[b], c="red", label="Output")
                ax.plot(
                    np.arange(self.length),
                    self.compare_batch[b],
                    c="black",
                    label="Input",
                )
                ax.legend(loc=1)
                plt.title("FFT Batch {} Input and Output".format(b))
                savefile = os.path.join(self.outdir, "out_batch_{}_fft.png".format(b))
                plt.savefig(savefile)
                plt.close()

        np.testing.assert_array_almost_equal(check, self.compare_batch)
        return
