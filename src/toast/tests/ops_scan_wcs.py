# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops as ops
from ..observation import default_values as defaults
from ..pixels import PixelData
from ..pixels_io_wcs import read_wcs
from .helpers import (
    close_data,
    create_fake_mask,
    create_fake_wcs_map,
    create_ground_data,
    create_outdir,
)
from .mpi import MPITestCase


class ScanWCSTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, subdir=fixture_name)
        np.random.seed(123456)

    def _test_wcs(self, suffix):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
        )

        # Pixelization
        pixels = ops.PixelsWCS(
            projection="CAR",
            resolution=(0.05 * u.degree, 0.05 * u.degree),
            dimensions=(),
            auto_bounds=True,
            detector_pointing=detpointing_radec,
            create_dist="pixel_dist",
            use_astropy=True,
        )
        pixels.apply(data)
        pixels.create_dist = None

        # Stokes weights
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
        )
        weights.apply(data)

        # Create fake polarized sky signal
        input_file = os.path.join(self.outdir, f"fake.{suffix}")
        map_key = "fake_map"
        data[map_key] = create_fake_wcs_map(
            input_file,
            data["pixel_dist"],
            pixels.wcs,
            pixels.wcs_shape,
            fwhm=10.0 * u.arcmin,
            I_scale=0.001,
            Q_scale=0.0001,
            U_scale=0.0001,
        )
        map_data = data[map_key]

        # Scan map into timestreams
        scanner = ops.ScanMap(
            det_data=defaults.det_data,
            pixels=pixels.pixels,
            weights=weights.weights,
            map_key=map_key,
        )
        scanner.apply(data)

        # Run the scanning from the file
        scan_wcs = ops.ScanWCSMap(
            file=input_file,
            det_data="test",
            pixel_pointing=pixels,
            stokes_weights=weights,
        )
        scan_wcs.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                np.testing.assert_almost_equal(
                    ob.detdata["test"][det], ob.detdata[defaults.det_data][det]
                )

        # Bin a map

        default_model = ops.DefaultNoiseModel(noise_model="noise_model")
        default_model.apply(data)

        binner = ops.BinMap(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=default_model.noise_model,
        )

        mapmaker = ops.MapMaker(
            det_data="test",
            binning=binner,
            write_hits=True,
            write_binmap=True,
            output_dir=self.outdir,
        )
        mapmaker.apply(data)

        # Check that the output map is consistent with the input map in all hit pixels

        if data.comm.world_rank == 0:
            output_file = os.path.join(self.outdir, f"{mapmaker.name}_binmap.fits")
            image_in = read_wcs(input_file)
            image_out = read_wcs(output_file)
            good = image_out != 0
            np.testing.assert_almost_equal(image_in[good], image_out[good])

        close_data(data)

    def test_wcs_fits(self):
        self._test_wcs("fits")

    def test_wcs_hdf5(self):
        self._test_wcs("hdf5")

    def test_wcs_mask(self):
        # Create fake observing of a small patch
        data = create_ground_data(self.comm)

        # Simple detector pointing
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec,
        )

        # Pixelization
        pixels = ops.PixelsWCS(
            projection="CAR",
            resolution=(0.05 * u.degree, 0.05 * u.degree),
            dimensions=(),
            auto_bounds=True,
            detector_pointing=detpointing_radec,
            create_dist="pixel_dist",
            use_astropy=True,
        )
        pixels.apply(data)
        pixels.create_dist = None

        # Stokes weights
        weights = ops.StokesWeights(
            mode="IQU",
            hwp_angle=defaults.hwp_angle,
            detector_pointing=detpointing_radec,
        )
        weights.apply(data)

        # Create fake mask pixel values locally
        create_fake_mask(data, "pixel_dist", "fake_mask")

        # Write this to a file
        input_file = os.path.join(self.outdir, "fake_mask.fits")
        data["fake_mask"].write(input_file)

        # Scan mask into timestreams
        scanner = ops.ScanMask(
            det_flags=defaults.det_flags,
            det_flags_value=defaults.det_mask_invalid,
            pixels=pixels.pixels,
            mask_key="fake_mask",
        )
        scanner.apply(data)

        # Run the scanning from the file
        scan_wcs = ops.ScanWCSMask(
            file=input_file,
            det_flags="test_flags",
            det_flags_value=defaults.det_mask_invalid,
            pixel_pointing=pixels,
        )
        scan_wcs.apply(data)

        # Check that the sets of timestreams match.

        for ob in data.obs:
            for det in ob.select_local_detectors(flagmask=defaults.det_mask_invalid):
                np.testing.assert_equal(
                    ob.detdata["test_flags"][det], ob.detdata[defaults.det_flags][det]
                )

        close_data(data)
