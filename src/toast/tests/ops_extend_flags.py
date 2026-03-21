# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..utils import extend_flags
from .helpers import (
    close_data,
    create_satellite_data,
    create_outdir,
    fake_flags,
)
from .mpi import MPITestCase


def check_buffering(input, output, buffer):
    n_samp = len(input)
    # Check that all output samples within range are flagged
    for isamp in range(n_samp):
        if input[isamp] == 0:
            continue
        istart = isamp - buffer
        if istart < 0:
            istart = 0
        istop = isamp + buffer
        if istop >= n_samp:
            istop = n_samp - 1
        for it in range(istart, istop, 1):
            if output[it] == 0:
                msg = f"Flagged input {isamp}:  output {it} is not flagged"
                print(msg, flush=True)
                return False
    # Check that all flagged output samples are in range of an input flag
    for isamp in range(n_samp):
        if output[isamp] == 0:
            continue
        istart = isamp - buffer
        if istart < 0:
            istart = 0
        istop = isamp + buffer + 1
        if istop >= n_samp:
            istop = n_samp - 1
        total = 0
        for it in range(istart, istop, 1):
            if input[it] != 0:
                total += 1
        if total == 0:
            flgstr = ",".join([f"{input[x]}" for x in range(istart, istop)])
            msg = f"Flagged output {isamp}:  no input samples "
            msg += f"flagged in {istart}:{istop}\n{flgstr}"
            print(msg, flush=True)
            return False
    return True


class FlagGapsTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)
        if (
            ("CONDA_BUILD" in os.environ)
            or ("CIBUILDWHEEL" in os.environ)
            or ("CI" in os.environ)
        ):
            self.make_plots = False
        else:
            self.make_plots = True

    def create_test_data(self):
        # Create fake satellite observations for testing.
        data = create_satellite_data(self.comm)

        # Create an uncorrelated noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        # Simulate fake instrumental noise
        sim_noise = ops.SimNoise(noise_model="noise_model")
        sim_noise.apply(data)

        # Create flagged samples
        fake_flags(data, do_half=False, do_random=True)

        return data

    def test_gap_flag_func(self):
        # Create some test data.
        data = self.create_test_data()

        # Make a copy for later comparison
        ops.Copy(
            shared=[(defaults.shared_flags, "input")],
            detdata=[(defaults.det_flags, "input")],
        ).apply(data)

        # Manually process the data
        shared_buffer = 5
        det_buffer = 10
        for ob in data.obs:
            # Shared flags
            if ob.comm_col_rank == 0:
                new_flags = np.copy(ob.shared[defaults.shared_flags].data)
                extend_flags(new_flags, defaults.shared_mask_invalid, shared_buffer)
            else:
                new_flags = None
            ob.shared[defaults.shared_flags].set(new_flags)
            # Detector flags
            for det in ob.local_detectors:
                extend_flags(
                    ob.detdata[defaults.det_flags][det],
                    defaults.det_mask_invalid,
                    det_buffer,
                )

        # Diagnostic plots of one detector on each process.
        if self.make_plots:
            import matplotlib.pyplot as plt

            for ob in data.obs:
                det = ob.select_local_detectors(flagmask=defaults.det_mask_nonscience)[
                    0
                ]
                n_all_samp = ob.n_all_samples
                n_plot = 2
                fig_height = 6 * n_plot
                pltsamp = 200

                for first, last in [
                    (0, 150),
                    (0, n_all_samp),
                    (n_all_samp // 2 - pltsamp, n_all_samp // 2 + pltsamp),
                ]:
                    plot_slc = slice(first, last, 1)
                    outfile = os.path.join(
                        self.outdir,
                        f"gap-flagged-func_{ob.name}_{det}_{first}-{last}.pdf",
                    )

                    samp_indx = np.arange(n_all_samp)
                    input_shflags = ob.shared["input"].data
                    input_detflags = ob.detdata["input"][det]
                    detflags = ob.detdata[defaults.det_flags][det]
                    shflags = ob.shared[defaults.shared_flags].data

                    fig = plt.figure(figsize=(12, fig_height), dpi=72)
                    ax = fig.add_subplot(n_plot, 1, 1, aspect="auto")
                    # Det flags
                    ax.plot(
                        samp_indx[plot_slc],
                        input_detflags[plot_slc],
                        color="black",
                        label=f"{det} Input",
                    )
                    ax.plot(
                        samp_indx[plot_slc],
                        detflags[plot_slc],
                        color="red",
                        label=f"{det} Gap-Flagged",
                    )
                    ax.legend(loc="best")

                    # shared flags
                    ax = fig.add_subplot(n_plot, 1, 2, aspect="auto")
                    ax.plot(
                        samp_indx[plot_slc],
                        input_shflags[plot_slc],
                        color="black",
                        label="Shared Input",
                    )
                    ax.plot(
                        samp_indx[plot_slc],
                        shflags[plot_slc],
                        color="red",
                        label="Shared Gap-Flagged",
                    )
                    ax.legend(loc="best")
                    fig.suptitle(f"Obs {ob.name}: {first} - {last}")
                    fig.savefig(outfile)
                    plt.close(fig)

        # Check results
        for ob in data.obs:
            self.assertTrue(
                check_buffering(
                    ob.shared["input"].data,
                    ob.shared[defaults.shared_flags].data,
                    shared_buffer,
                )
            )
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_nonscience):
                self.assertTrue(
                    check_buffering(
                        ob.detdata["input"][det],
                        ob.detdata[defaults.det_flags][det],
                        det_buffer,
                    )
                )

        close_data(data)

    def test_gap_flag_operator(self):
        # Create some test data.
        data = self.create_test_data()

        # Make a copy for later comparison
        ops.Copy(
            shared=[(defaults.shared_flags, "input")],
            detdata=[(defaults.det_flags, "input")],
        ).apply(data)

        # Run flagging
        shared_buffer = 5
        det_buffer = 10
        ops.ExtendFlags(
            shared_flag_mask=defaults.shared_mask_invalid,
            det_flag_mask=defaults.det_mask_invalid,
            shared_buffer_samples=shared_buffer,
            det_buffer_samples=det_buffer,
        ).apply(data)

        # Diagnostic plots of one detector on each process.
        if self.make_plots:
            import matplotlib.pyplot as plt

            for ob in data.obs:
                det = ob.select_local_detectors(flagmask=defaults.det_mask_nonscience)[
                    0
                ]
                n_all_samp = ob.n_all_samples
                n_plot = 2
                fig_height = 6 * n_plot
                pltsamp = 200

                for first, last in [
                    (0, n_all_samp),
                    (n_all_samp // 2 - pltsamp, n_all_samp // 2 + pltsamp),
                ]:
                    plot_slc = slice(first, last, 1)
                    outfile = os.path.join(
                        self.outdir,
                        f"gap-flagged-ops_{ob.name}_{det}_{first}-{last}.pdf",
                    )

                    samp_indx = np.arange(n_all_samp)
                    input_shflags = ob.shared["input"].data
                    input_detflags = ob.detdata["input"][det]
                    detflags = ob.detdata[defaults.det_flags][det]
                    shflags = ob.shared[defaults.shared_flags].data

                    fig = plt.figure(figsize=(12, fig_height), dpi=72)
                    ax = fig.add_subplot(n_plot, 1, 1, aspect="auto")
                    # Det flags
                    ax.plot(
                        samp_indx[plot_slc],
                        input_detflags[plot_slc],
                        color="black",
                        label=f"{det} Input",
                    )
                    ax.plot(
                        samp_indx[plot_slc],
                        detflags[plot_slc],
                        color="red",
                        label=f"{det} Gap-Flagged",
                    )
                    ax.legend(loc="best")

                    # shared flags
                    ax = fig.add_subplot(n_plot, 1, 2, aspect="auto")
                    ax.plot(
                        samp_indx[plot_slc],
                        input_shflags[plot_slc],
                        color="black",
                        label="Shared Input",
                    )
                    ax.plot(
                        samp_indx[plot_slc],
                        shflags[plot_slc],
                        color="red",
                        label="Shared Gap-Flagged",
                    )
                    ax.legend(loc="best")
                    fig.suptitle(f"Obs {ob.name}: {first} - {last}")
                    fig.savefig(outfile)
                    plt.close(fig)

        # Check results
        for ob in data.obs:
            self.assertTrue(
                check_buffering(
                    ob.shared["input"].data,
                    ob.shared[defaults.shared_flags].data,
                    shared_buffer,
                )
            )
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_nonscience):
                self.assertTrue(
                    check_buffering(
                        ob.detdata["input"][det],
                        ob.detdata[defaults.det_flags][det],
                        det_buffer,
                    )
                )

        close_data(data)
