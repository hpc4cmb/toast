# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for generating fake sky maps."""

import os
import tempfile
import urllib

import astropy.io.fits as af
import healpy as hp
import numpy as np
from astropy import units as u
from scipy.ndimage import gaussian_filter

from ... import ops, rng
from ...observation import default_values as defaults
from ...pixels import PixelData
from ...pixels_io_wcs import write_wcs


def create_fake_healpix_file(
    out_file,
    nside,
    fwhm=10.0 * u.arcmin,
    lmax=256,
    I_scale=1.0,
    Q_scale=1.0,
    U_scale=1.0,
    units=u.K,
):
    """Generate a fake sky map on disk with one process.

    This should only be called on a single process!

    Args:
        out_file (str):  The output FITS map.
        nside (int):  The NSIDE value.
        fwhm (Quantity):  The beam smoothing FWHM.
        lmax (int):  The ell_max of the expansion for smoothing.
        I_scale (float):  The overall scaling factor of the intensity map.
        Q_scale (float):  The overall scaling factor of the Stokes Q map.
        U_scale (float):  The overall scaling factor of the Stokes U map.
        units (Unit):  The map units to write to the file.

    Returns:
        None

    """
    npix = 12 * nside**2
    off = 0
    maps = list()
    for scale in [I_scale, Q_scale, U_scale]:
        vals = np.array(
            rng.random(
                npix,
                key=(12345, 6789),
                counter=(0, off),
                sampler="gaussian",
            ),
            dtype=np.float64,
        )
        vals = hp.smoothing(vals, fwhm=fwhm.to_value(u.radian), lmax=lmax)
        maps.append(scale * vals)
        off += npix
    del vals
    unit_str = f"{units}"
    maps = hp.reorder(maps, r2n=True)
    hp.write_map(
        out_file,
        maps,
        nest=True,
        fits_IDL=False,
        coord="C",
        column_units=unit_str,
        dtype=np.float32,
    )
    del maps


def create_fake_wcs_file(
    out_file,
    wcs,
    wcs_shape,
    fwhm=10.0 * u.arcmin,
    I_scale=1.0,
    Q_scale=1.0,
    U_scale=1.0,
    units=u.K,
):
    """Generate a fake sky map on disk with one process.

    This should only be called on a single process!

    Args:
        out_file (str):  The output FITS map.
        wcs (astropy.wcs.WCS):  The WCS structure.
        wcs_shape (tuple):  The image dimensions in longitude, latitude.
        fwhm (Quantity):  The beam smoothing FWHM.
        I_scale (float):  The overall scaling factor of the intensity map.
        Q_scale (float):  The overall scaling factor of the Stokes Q map.
        U_scale (float):  The overall scaling factor of the Stokes U map.
        units (Unit):  The map units to write to the file.

    Returns:
        None

    """
    # Image dimensions
    n_row, n_col = wcs_shape
    # Get the smoothing kernel FWHM in terms of pixels
    lat_res_deg = np.absolute(wcs.wcs.cdelt[0])
    lon_res_deg = np.absolute(wcs.wcs.cdelt[1])
    lat_fwhm = fwhm.to_value(u.degree) / lat_res_deg
    lon_fwhm = fwhm.to_value(u.degree) / lon_res_deg

    image_shape = (3, n_row, n_col)
    image = np.zeros(image_shape, dtype=np.float64)

    np.random.seed(987654321)
    for imap, scale in enumerate([I_scale, Q_scale, U_scale]):
        temp = np.random.normal(loc=0.0, scale=scale, size=(n_row, n_col))
        image[imap, :, :] = gaussian_filter(temp, sigma=(lat_fwhm, lon_fwhm))

    write_wcs(out_file, image, wcs, units)
    del image


def create_fake_healpix_map(
    out_file,
    pixel_dist,
    fwhm=10.0 * u.arcmin,
    lmax=256,
    I_scale=1.0,
    Q_scale=1.0,
    U_scale=1.0,
    units=u.K,
):
    """Create and load a healpix map into a PixelData object.

    This starts from a pre-made PixelDistribution and uses that to determine
    the Healpix NSIDE.  It then creates a map on disk and loads it into a
    distributed PixelData object.

    Args:
        out_file (str):  The generated FITS map.
        pixel_dist (PixelDistribution):  The pixel distribution to use.
        fwhm (Quantity):  The beam smoothing FWHM.
        lmax (int):  The ell_max of the expansion for smoothing.
        I_scale (float):  The overall scaling factor of the intensity map.
        Q_scale (float):  The overall scaling factor of the Stokes Q map.
        U_scale (float):  The overall scaling factor of the Stokes U map.
        units (Unit):  The map units to write to the file.

    Returns:
        (PixelData):  The distributed map.

    """
    comm = pixel_dist.comm
    if comm is None:
        rank = 0
    else:
        rank = comm.rank
    npix = pixel_dist.n_pix
    nside = hp.npix2nside(npix)
    if rank == 0:
        create_fake_healpix_file(
            out_file,
            nside,
            fwhm=fwhm,
            lmax=lmax,
            I_scale=I_scale,
            Q_scale=Q_scale,
            U_scale=U_scale,
            units=units,
        )
    if comm is not None:
        comm.barrier()
    pix = PixelData(pixel_dist, np.float64, n_value=3, units=units)
    pix.read(out_file)
    return pix


def create_fake_wcs_map(
    out_file,
    pixel_dist,
    wcs,
    wcs_shape,
    fwhm=10.0 * u.arcmin,
    I_scale=1.0,
    Q_scale=1.0,
    U_scale=1.0,
    units=u.K,
):
    """Create and load a WCS map into a PixelData object.

    This starts from a pre-made PixelDistribution and WCS information (from,
    for example, a PixelsWCS instance).  It then creates a map on disk and loads
    it into a distributed PixelData object.

    Args:
        out_file (str):  The generated FITS map.
        pixel_dist (PixelDistribution):  The pixel distribution to use.
        wcs (astropy.wcs.WCS):  The WCS structure.
        wcs_shape (tuple):  The image dimensions in longitude, latitude.
        fwhm (Quantity):  The beam smoothing FWHM.
        I_scale (float):  The overall scaling factor of the intensity map.
        Q_scale (float):  The overall scaling factor of the Stokes Q map.
        U_scale (float):  The overall scaling factor of the Stokes U map.
        units (Unit):  The map units to write to the file.

    Returns:
        (PixelData):  The distributed map.

    """
    comm = pixel_dist.comm
    if comm is None:
        rank = 0
    else:
        rank = comm.rank
    if rank == 0:
        create_fake_wcs_file(
            out_file,
            wcs,
            wcs_shape,
            fwhm=fwhm,
            I_scale=I_scale,
            Q_scale=Q_scale,
            U_scale=U_scale,
            units=units,
        )
    if comm is not None:
        comm.barrier()
    pix = PixelData(pixel_dist, np.float64, n_value=3, units=units)
    pix.read(out_file)
    return pix


def create_fake_healpix_scanned_tod(
    data,
    pixel_pointing,
    stokes_weights,
    out_file,
    pixel_dist,
    map_key="fake_sky",
    fwhm=10.0 * u.arcmin,
    lmax=256,
    I_scale=1.0,
    Q_scale=1.0,
    U_scale=1.0,
    det_data=defaults.det_data,
):
    """Create a fake healpix map and scan this into detector timestreams.

    The pixel distribution is created if it does not exist.  The map is created
    on disk and then loaded into a PixelData object.  The pointing expansion and
    map scanning are pipelined so that detector pointing does not need to be stored
    persistently.

    Args:
        data (Data):  The data container.
        pixel_pointing (Operator):  The healpix pixelization operator.
        stokes_weights (Operator):  The detector weights operator.
        out_file (str):  The generated FITS map.
        pixel_dist (PixelDistribution):  The pixel distribution to use.
        map_key (str):  The data key to hold the generated map in memory.
        fwhm (Quantity):  The beam smoothing FWHM.
        lmax (int):  The ell_max of the expansion for smoothing.
        I_scale (float):  The overall scaling factor of the intensity map.
        Q_scale (float):  The overall scaling factor of the Stokes Q map.
        U_scale (float):  The overall scaling factor of the Stokes U map.
        det_data (str):  The detdata name of the output scanned signal.

    Returns:
        None

    """
    if pixel_dist not in data:
        # Build the pixel distribution
        build_dist = ops.BuildPixelDistribution(
            pixel_dist=pixel_dist,
            pixel_pointing=pixel_pointing,
        )
        build_dist.apply(data)

    if map_key in data:
        msg = f"Generated map '{map_key}' already exists in data"
        raise RuntimeError(msg)

    # Use detector data units for the map, if it exists.
    first_obs = data.obs[0]
    if det_data in first_obs.detdata:
        units = first_obs.detdata[det_data].units
    else:
        units = u.K

    # Create detector data if needed
    for ob in data.obs:
        exists = ob.detdata.ensure(
            det_data,
            create_units=units,
        )

    # Create and load the map
    data[map_key] = create_fake_healpix_map(
        out_file,
        data[pixel_dist],
        fwhm=fwhm,
        lmax=lmax,
        I_scale=I_scale,
        Q_scale=Q_scale,
        U_scale=U_scale,
        units=units,
    )

    # Scan map into timestreams
    scanner = ops.ScanMap(
        det_data=det_data,
        pixels=pixel_pointing.pixels,
        weights=stokes_weights.weights,
        map_key=map_key,
    )
    scan_pipe = ops.Pipeline(
        detector_sets=["SINGLE"],
        operators=[
            pixel_pointing,
            stokes_weights,
            scanner,
        ],
    )
    scan_pipe.apply(data)

    # Cleanup, to avoid any conflicts with the calling code.
    for ob in data.obs:
        for buf in [
            pixel_pointing.pixels,
            stokes_weights.weights,
            pixel_pointing.detector_pointing.quats,
        ]:
            if buf in ob:
                del ob[buf]


def create_fake_wcs_scanned_tod(
    data,
    pixel_pointing,
    stokes_weights,
    out_file,
    pixel_dist,
    map_key="fake_sky",
    fwhm=10.0 * u.arcmin,
    I_scale=1.0,
    Q_scale=1.0,
    U_scale=1.0,
    det_data=defaults.det_data,
):
    """Create a fake WCS map and scan this into detector timestreams.

    The pixel distribution is created if it does not exist.  The map is created
    on disk and then loaded into a PixelData object.  The pointing expansion and
    map scanning are pipelined so that detector pointing does not need to be stored
    persistently.

    Args:
        data (Data):  The data container.
        pixel_pointing (Operator):  The healpix pixelization operator.
        stokes_weights (Operator):  The detector weights operator.
        out_file (str):  The generated FITS map.
        pixel_dist (PixelDistribution):  The pixel distribution to use.
        map_key (str):  The data key to hold the generated map in memory.
        fwhm (Quantity):  The beam smoothing FWHM.
        I_scale (float):  The overall scaling factor of the intensity map.
        Q_scale (float):  The overall scaling factor of the Stokes Q map.
        U_scale (float):  The overall scaling factor of the Stokes U map.
        det_data (str):  The detdata name of the output scanned signal.

    Returns:
        None

    """
    if pixel_dist not in data:
        # Build the pixel distribution
        build_dist = ops.BuildPixelDistribution(
            pixel_dist=pixel_dist,
            pixel_pointing=pixel_pointing,
        )
        build_dist.apply(data)

    if map_key in data:
        msg = f"Generated map '{map_key}' already exists in data"
        raise RuntimeError(msg)

    # Use detector data units for the map, if it exists.
    first_obs = data.obs[0]
    if det_data in first_obs.detdata:
        units = first_obs.detdata[det_data].units
    else:
        units = u.K

    # Create detector data if needed
    for ob in data.obs:
        exists = ob.detdata.ensure(
            det_data,
            create_units=units,
        )

    # Get the WCS info from the pixel operator
    wcs = pixel_pointing.wcs
    wcs_shape = pixel_pointing.wcs_shape

    # Create and load the map
    data[map_key] = create_fake_wcs_map(
        out_file,
        data[pixel_dist],
        wcs,
        wcs_shape,
        fwhm=fwhm,
        I_scale=I_scale,
        Q_scale=Q_scale,
        U_scale=U_scale,
        units=units,
    )

    # Scan map into timestreams
    scanner = ops.ScanMap(
        det_data=det_data,
        pixels=pixel_pointing.pixels,
        weights=stokes_weights.weights,
        map_key=map_key,
    )
    scan_pipe = ops.Pipeline(
        detector_sets=["SINGLE"],
        operators=[
            pixel_pointing,
            stokes_weights,
            scanner,
        ],
    )
    scan_pipe.apply(data)

    # Cleanup, to avoid any conflicts with the calling code.
    for ob in data.obs:
        for buf in [
            pixel_pointing.pixels,
            stokes_weights.weights,
            pixel_pointing.detector_pointing.quats,
        ]:
            if buf in ob:
                del ob[buf]


def create_fake_mask(data, dist_key, mask_key):
    np.random.seed(987654321)
    dist = data[dist_key]
    pix_data = PixelData(dist, np.uint8, n_value=1)
    # Just replicate the fake data across all local submaps
    off = 0
    for submap in range(dist.n_submap):
        mask_data = np.random.normal(size=dist.n_pix_submap) > 0.5
        if submap in dist.local_submaps:
            pix_data.data[off, :, 0] = mask_data
            off += 1
    data[mask_key] = pix_data


def create_fake_sky_alm(lmax=128, fwhm=10 * u.degree, pol=True, pointsources=False):
    if pointsources:
        nside = 512
        while nside < lmax:
            nside *= 2
        npix = 12 * nside**2
        m = np.zeros(npix)
        for lon in np.linspace(-180, 180, 6):
            for lat in np.linspace(-80, 80, 6):
                m[hp.ang2pix(nside, lon, lat, lonlat=True)] = 1
        if pol:
            m = np.vstack([m, m, m])
        m = hp.smoothing(m, fwhm=fwhm.to_value(u.radian))
        a_lm = hp.map2alm(m, lmax=lmax)
    else:
        # Power spectrum
        if pol:
            cl = np.ones(4 * (lmax + 1)).reshape([4, -1])
        else:
            cl = np.ones(lmax + 1)
        # Draw a_lm
        nside = 2
        while 4 * nside < lmax:
            nside *= 2
        _, a_lm = hp.synfast(
            cl,
            nside,
            alm=True,
            lmax=lmax,
            fwhm=fwhm.to_value(u.radian),
            verbose=False,
        )

    return a_lm


def create_fake_beam_alm(
    lmax=128,
    mmax=10,
    fwhm_x=10 * u.degree,
    fwhm_y=10 * u.degree,
    pol=True,
    separate_IQU=False,
    separate_TP=False,
    detB_beam=False,
    normalize_beam=False,
):
    # pick an nside >= lmax to be sure that the a_lm will be fairly accurate
    nside = 2
    while nside < lmax:
        nside *= 2
    npix = 12 * nside**2
    pix = np.arange(npix)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    sigma_z = fwhm_x.to_value(u.radian) / np.sqrt(8 * np.log(2))
    sigma_y = fwhm_y.to_value(u.radian) / np.sqrt(8 * np.log(2))
    beam = np.exp(-(z**2 / 2 / sigma_z**2 + y**2 / 2 / sigma_y**2))
    beam[x < 0] = 0
    beam_map = np.zeros([3, npix])
    beam_map[0] = beam
    if detB_beam:
        # we make sure that the two detectors within the same pair encode
        # two beams with the  flipped sign in Q   U beams
        beam_map[1] = -beam
    else:
        beam_map[1] = beam
    blm = hp.map2alm(beam_map, lmax=lmax, mmax=mmax)
    hp.rotate_alm(blm, psi=0, theta=-np.pi / 2, phi=0, lmax=lmax, mmax=mmax)

    if normalize_beam:
        # We make sure that the simulated beams are normalized in the test
        # for the normalization we follow the convention adopted in Conviqt,
        # i.e. the monopole term in the map is left unchanged
        idx = hp.Alm.getidx(lmax=lmax, l=0, m=0)
        norm = 2 * np.pi * blm[0, idx].real

    else:
        norm = 1.0

    blm /= norm
    if separate_IQU:
        empty = np.zeros_like(beam_map[0])
        beam_map_I = np.vstack([beam_map[0], empty, empty])
        beam_map_Q = np.vstack([empty, beam_map[1], empty])
        beam_map_U = np.vstack([empty, empty, beam_map[1]])
        try:
            blmi00 = (
                hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, verbose=False, pol=True)
                / norm
            )
            blm0i0 = (
                hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, verbose=False, pol=True)
                / norm
            )
            blm00i = (
                hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, verbose=False, pol=True)
                / norm
            )
        except TypeError:
            # older healpy which does not have verbose keyword
            blmi00 = hp.map2alm(beam_map_I, lmax=lmax, mmax=mmax, pol=True) / norm
            blm0i0 = hp.map2alm(beam_map_Q, lmax=lmax, mmax=mmax, pol=True) / norm
            blm00i = hp.map2alm(beam_map_U, lmax=lmax, mmax=mmax, pol=True) / norm
        for b_lm in blmi00, blm0i0, blm00i:
            hp.rotate_alm(b_lm, psi=0, theta=-np.pi / 2, phi=0, lmax=lmax, mmax=mmax)
        return [blmi00, blm0i0, blm00i]

    elif separate_TP:
        blmT = blm[0].copy()
        blmP = blm.copy()
        blmP[0] = 0

        return [blmT, blmP]
    else:
        return blm


def fetch_nominal_cmb_cls(out_file=None):
    """Retrieve some nominal angular power spectra for testing.

    Do not use this function if you care about the details.  The goal of this function is just
    to make it easy to synthesize realistic-looking CMB skies for testing.

    Args:
        out_file (str):  If not None save the file persistently at this location and load as
            needed.  If None, download to a tempfile.

    Returns:
        (list):  The list of cls, suitable for use with healpy.synfast.

    """
    url_root = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"
    url_file = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
    url = f"{url_root}{url_file}"

    def _parse_file(path):
        raw = np.transpose(np.loadtxt(path, usecols=(0, 1, 2, 3, 4)))
        # Get the first ell value
        raw_start = raw[0][0]
        pad_lines = int(raw_start)
        ncl = pad_lines + len(raw[0])
        # Build list of spectra
        cl = list()
        for spec in range(4):
            dat = np.zeros(ncl, dtype=np.float32)
            dat[pad_lines:] = raw[1 + spec]
            cl.append(dat)
        return cl

    if out_file is None:
        # Retrieve to a temp location
        with tempfile.TemporaryDirectory as dir:
            out_file = os.path.join(dir, url_file)
            urllib.request.urlretrieve(url, out_file)
            c_ell = _parse_file(out_file)
    else:
        # We are loading the file from a persistent location
        if not os.path.isfile(out_file):
            # Download it
            urllib.request.urlretrieve(url, out_file)
        c_ell = _parse_file(out_file)

    return c_ell
