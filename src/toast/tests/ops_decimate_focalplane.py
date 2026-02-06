# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .. import ops as ops
from ..mpi import MPI
from ..observation import default_values as defaults
from .helpers import close_data, create_ground_data, create_outdir
from .mpi import MPITestCase


class DecimateFocalplaneTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)

    def test_no_decimation(self):
        """Test that nskip=1 results in no flagging."""
        data = create_ground_data(self.comm, pixel_per_process=10)

        # Count detectors before decimation
        pre_total_dets = 0
        for obs in data.obs:
            pre_total_dets += len(
                obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            )

        # Apply decimation with nskip=1 (should not flag anything)
        decimate = ops.DecimateFocalplane(
            nskip=1,
            detectors_per_pixel=1,
            det_mask=defaults.det_mask_invalid,
        )
        decimate.apply(data)

        # Count detectors after decimation
        post_total_dets = 0
        for obs in data.obs:
            post_total_dets += len(
                obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            )

        if data.comm.comm_world is not None:
            pre_total_dets = data.comm.comm_world.allreduce(
                pre_total_dets, op=MPI.SUM
            )
            post_total_dets = data.comm.comm_world.allreduce(
                post_total_dets, op=MPI.SUM
            )

        # Should have the same number of detectors
        self.assertEqual(pre_total_dets, post_total_dets)

        close_data(data)

    def test_basic_decimation(self):
        """Test basic decimation with detectors_per_pixel=1."""
        data = create_ground_data(self.comm, pixel_per_process=20)

        # Count detectors before decimation
        pre_good_dets = 0
        pre_total_dets = 0
        for obs in data.obs:
            local_dets = obs.local_detectors
            pre_total_dets += len(local_dets)
            pre_good_dets += len(
                obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            )

        # Apply decimation - keep every 5th detector
        nskip = 5
        decimate = ops.DecimateFocalplane(
            nskip=nskip,
            detectors_per_pixel=1,
            det_mask=defaults.det_mask_invalid,
        )
        decimate.apply(data)

        # Count detectors after decimation
        post_good_dets = 0
        post_total_dets = 0
        for obs in data.obs:
            local_dets = obs.local_detectors
            post_total_dets += len(local_dets)
            post_good_dets += len(
                obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            )

        if data.comm.comm_world is not None:
            pre_good_dets = data.comm.comm_world.allreduce(
                pre_good_dets, op=MPI.SUM
            )
            pre_total_dets = data.comm.comm_world.allreduce(
                pre_total_dets, op=MPI.SUM
            )
            post_good_dets = data.comm.comm_world.allreduce(
                post_good_dets, op=MPI.SUM
            )
            post_total_dets = data.comm.comm_world.allreduce(
                post_total_dets, op=MPI.SUM
            )

        # Total detectors should remain the same
        self.assertEqual(pre_total_dets, post_total_dets)

        # Should have approximately 1/nskip of the original good detectors left
        expected_good = pre_good_dets // nskip
        # Allow some tolerance due to rounding and detector distribution
        tolerance = max(10, pre_good_dets // 20)  # 5% tolerance or at least 10
        self.assertLess(
            abs(post_good_dets - expected_good),
            tolerance,
            f"Expected ~{expected_good} good detectors, got {post_good_dets}",
        )

        close_data(data)

    def test_pixel_grouping(self):
        """Test decimation with detectors_per_pixel=2."""
        data = create_ground_data(
            self.comm, pixel_per_process=30, flagged_pixels=False
        )

        # Apply decimation with pixel grouping
        nskip = 3
        detectors_per_pixel = 2
        decimate = ops.DecimateFocalplane(
            nskip=nskip,
            detectors_per_pixel=detectors_per_pixel,
            det_mask=defaults.det_mask_invalid,
        )
        decimate.apply(data)

        # Verify that detectors are flagged in groups
        # Since flagged_pixels=False, all detectors start unflagged
        for obs in data.obs:
            local_dets = sorted(obs.local_detectors)
            ndet = len(local_dets)
            npix = ndet // detectors_per_pixel

            for ipix in range(npix):
                # Check if this pixel should be flagged
                should_be_flagged = (ipix % nskip != 0)

                offset = ipix * detectors_per_pixel
                for idet in range(offset, offset + detectors_per_pixel):
                    if idet >= ndet:
                        break
                    det = local_dets[idet]
                    # Get detector flags
                    det_flag = obs.local_detector_flags[det]
                    is_flagged = (det_flag & defaults.det_mask_invalid) != 0

                    if should_be_flagged:
                        self.assertTrue(
                            is_flagged,
                            f"Detector {det} in pixel {ipix} (ipix % nskip != 0) should be flagged",
                        )
                    else:
                        # This pixel should be kept - detectors should not be flagged
                        self.assertFalse(
                            is_flagged,
                            f"Detector {det} in pixel {ipix} (ipix % nskip == 0) should not be flagged",
                        )

        close_data(data)

    def test_interaction_with_existing_flags(self):
        """Test that decimation works correctly with pre-existing detector flags."""
        data = create_ground_data(self.comm, pixel_per_process=20)

        # Pre-flag some detectors
        for obs in data.obs:
            det_flags = {}
            local_dets = sorted(obs.local_detectors)
            # Flag every 7th detector
            for idx, det in enumerate(local_dets):
                if idx % 7 == 0:
                    det_flags[det] = defaults.det_mask_invalid
            obs.update_local_detector_flags(det_flags)

        # Count pre-existing flags
        pre_flagged = 0
        pre_total = 0
        for obs in data.obs:
            local_dets = obs.local_detectors
            pre_total += len(local_dets)
            pre_flagged += len(local_dets) - len(
                obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            )

        # Apply decimation
        nskip = 5
        decimate = ops.DecimateFocalplane(
            nskip=nskip,
            detectors_per_pixel=1,
            det_mask=defaults.det_mask_invalid,
        )
        decimate.apply(data)

        # Count flags after decimation
        post_flagged = 0
        post_total = 0
        for obs in data.obs:
            local_dets = obs.local_detectors
            post_total += len(local_dets)
            post_flagged += len(local_dets) - len(
                obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            )

        if data.comm.comm_world is not None:
            pre_flagged = data.comm.comm_world.allreduce(pre_flagged, op=MPI.SUM)
            pre_total = data.comm.comm_world.allreduce(pre_total, op=MPI.SUM)
            post_flagged = data.comm.comm_world.allreduce(post_flagged, op=MPI.SUM)
            post_total = data.comm.comm_world.allreduce(post_total, op=MPI.SUM)

        # Should have more flags after decimation
        self.assertGreater(post_flagged, pre_flagged)

        # Total detectors should remain the same
        self.assertEqual(pre_total, post_total)

        close_data(data)

    def test_decimation_pattern(self):
        """Test that the decimation pattern is correct - keep every nskip-th detector."""
        data = create_ground_data(
            self.comm, pixel_per_process=50, single_group=True, flagged_pixels=False
        )

        nskip = 4
        decimate = ops.DecimateFocalplane(
            nskip=nskip,
            detectors_per_pixel=1,
            det_mask=defaults.det_mask_invalid,
        )
        decimate.apply(data)

        # Check the pattern for each observation
        # Since we used flagged_pixels=False, all detectors start unflagged
        for obs in data.obs:
            local_dets = sorted(obs.local_detectors)
            ndet = len(local_dets)

            kept_count = 0
            flagged_count = 0

            for idx in range(ndet):
                det = local_dets[idx]
                det_flag = obs.local_detector_flags[det]
                is_flagged = (det_flag & defaults.det_mask_invalid) != 0

                # Every nskip-th detector (0, nskip, 2*nskip, ...) should NOT be flagged by decimation
                # All others should be flagged
                if idx % nskip == 0:
                    # This detector should be kept (not flagged by decimation)
                    self.assertFalse(
                        is_flagged,
                        f"Detector {det} at index {idx} should not be flagged (idx % nskip == 0)",
                    )
                    kept_count += 1
                else:
                    # This detector should be flagged by decimation
                    self.assertTrue(
                        is_flagged,
                        f"Detector {det} at index {idx} should be flagged (idx % nskip != 0)",
                    )
                    flagged_count += 1

            # Verify we kept approximately 1/nskip of detectors
            expected_kept = ndet // nskip
            self.assertGreaterEqual(kept_count, expected_kept)
            self.assertLessEqual(kept_count, expected_kept + 1)

        close_data(data)

    def test_mpi_reduction_counts(self):
        """Test that MPI reductions produce correct counts."""
        data = create_ground_data(self.comm, pixel_per_process=25)

        nskip = 6
        decimate = ops.DecimateFocalplane(
            nskip=nskip,
            detectors_per_pixel=1,
            det_mask=defaults.det_mask_invalid,
        )

        # Before decimation
        pre_total = 0
        pre_good = 0
        for obs in data.obs:
            local_dets = obs.local_detectors
            good_dets = obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            pre_total += len(local_dets)
            pre_good += len(good_dets)

        decimate.apply(data)

        # After decimation
        post_total = 0
        post_good = 0
        for obs in data.obs:
            local_dets = obs.local_detectors
            good_dets = obs.select_local_detectors(flagmask=defaults.det_mask_invalid)
            post_total += len(local_dets)
            post_good += len(good_dets)

        if data.comm.comm_world is not None:
            pre_total_global = data.comm.comm_world.allreduce(pre_total, op=MPI.SUM)
            pre_good_global = data.comm.comm_world.allreduce(pre_good, op=MPI.SUM)
            post_total_global = data.comm.comm_world.allreduce(post_total, op=MPI.SUM)
            post_good_global = data.comm.comm_world.allreduce(post_good, op=MPI.SUM)
        else:
            pre_total_global = pre_total
            pre_good_global = pre_good
            post_total_global = post_total
            post_good_global = post_good

        # Total should not change
        self.assertEqual(pre_total_global, post_total_global)

        # Good detectors should decrease
        self.assertLess(post_good_global, pre_good_global)

        # The fraction of good detectors should be approximately 1/nskip
        fraction = post_good_global / pre_good_global
        expected_fraction = 1.0 / nskip
        # Allow 20% relative tolerance due to rounding
        self.assertLess(
            abs(fraction - expected_fraction),
            expected_fraction * 0.3,
            f"Expected fraction ~{expected_fraction:.3f}, got {fraction:.3f}",
        )

        close_data(data)

    def test_different_pixel_groupings(self):
        """Test various detectors_per_pixel settings."""
        for detectors_per_pixel in [1, 2, 3]:
            data = create_ground_data(self.comm, pixel_per_process=30)

            nskip = 4
            decimate = ops.DecimateFocalplane(
                nskip=nskip,
                detectors_per_pixel=detectors_per_pixel,
                det_mask=defaults.det_mask_invalid,
            )
            decimate.apply(data)

            # Verify some detectors were flagged
            post_good = 0
            post_total = 0
            for obs in data.obs:
                local_dets = obs.local_detectors
                good_dets = obs.select_local_detectors(
                    flagmask=defaults.det_mask_invalid
                )
                post_total += len(local_dets)
                post_good += len(good_dets)

            if data.comm.comm_world is not None:
                post_total = data.comm.comm_world.allreduce(post_total, op=MPI.SUM)
                post_good = data.comm.comm_world.allreduce(post_good, op=MPI.SUM)

            # Should have fewer good detectors after decimation
            self.assertLess(post_good, post_total)

            close_data(data)
