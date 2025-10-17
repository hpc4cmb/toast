# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import warnings

import astropy.io.fits as af
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from . import qarray as qa
from .pixels_io_healpix import read_healpix

_matplotlib_backend = None


def set_matplotlib_backend(backend="pdf"):
    """Set the matplotlib backend."""
    global _matplotlib_backend
    if _matplotlib_backend is not None:
        return
    try:
        _matplotlib_backend = backend
        import matplotlib

        matplotlib.use(_matplotlib_backend, force=False)
    except:
        msg = "Could not set the matplotlib backend to '{}'".format(_matplotlib_backend)
        warnings.warn(msg)


def plot_noise_estim(
    fname,
    est_freq,
    est_psd,
    fit_freq=None,
    fit_psd=None,
    true_net=None,
    true_freq=None,
    true_psd=None,
    semilog=False,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(1, 1, 1)
    if (true_freq is not None) and (true_psd is not None):
        # Plot the truth
        if semilog:
            ax.semilogx(
                true_freq.to_value(u.Hz),
                true_psd.to_value(u.K**2 * u.s),
                color="black",
                label="Input Truth",
            )
        else:
            ax.loglog(
                true_freq.to_value(u.Hz),
                true_psd.to_value(u.K**2 * u.s),
                color="black",
                label="Input Truth",
            )
        if true_net is not None:
            net = true_net.to_value(u.K / u.Hz**0.5)
            ax.axhline(
                net**2,
                label=f"NET = {net:0.2e} K" + r" / $\sqrt{\mathrm{Hz}}$",
                linestyle="--",
                color="black",
            )
    if semilog:
        ax.semilogx(
            est_freq.to_value(u.Hz),
            est_psd.to_value(u.K**2 * u.s),
            color="red",
            label="Estimated",
        )
    else:
        ax.loglog(
            est_freq.to_value(u.Hz),
            est_psd.to_value(u.K**2 * u.s),
            color="red",
            label="Estimated",
        )
    if (fit_freq is not None) and (fit_psd is not None):
        if semilog:
            ax.semilogx(
                fit_freq.to_value(u.Hz),
                fit_psd.to_value(u.K**2 * u.s),
                color="blue",
                label="Fit to 1/f Model",
            )
        else:
            ax.loglog(
                fit_freq.to_value(u.Hz),
                fit_psd.to_value(u.K**2 * u.s),
                color="blue",
                label="Fit to 1/f Model",
            )
    ax.set_xlim(est_freq[0].to_value(u.Hz), est_freq[-1].to_value(u.Hz))
    ax.set_ylim(
        np.amin(est_psd.to_value(u.K**2 * u.s)),
        1.1 * np.amax(est_psd.to_value(u.K**2 * u.s)),
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [K$^2$ / Hz]")
    ax.legend(loc="best")
    if fname is None:
        plt.show()
    else:
        fig.savefig(fname)
    plt.close()


def plot_map_path(in_file, format, suffix=None, out_dir=None):
    in_dir = os.path.dirname(in_file)
    in_base = os.path.basename(in_file)
    in_root, in_ext = os.path.splitext(in_base)
    if suffix is None:
        out_file = f"{in_root}.{format}"
    else:
        out_file = f"{in_root}{suffix}.{format}"
    if out_dir is None:
        return os.path.join(in_dir, out_file)
    else:
        return os.path.join(out_dir, out_file)


def plot_wcs_maps(
    hitfile=None,
    mapfile=None,
    range_I=None,
    range_Q=None,
    range_U=None,
    max_hits=None,
    truth=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    is_azimuth=False,
    cmap="viridis",
    format="pdf",
    out_dir=None,
):
    """Plot WCS projected output maps.

    This is a helper function to plot typical outputs of the mapmaker.

    Args:
        hitfile (str):  Path to the hits file.
        mapfile (str):  Path to the map file.
        range_I (tuple):  The min / max values of the Intensity map to plot.
        range_Q (tuple):  The min / max values of the Q map to plot.
        range_U (tuple):  The min / max values of the U map to plot.
        max_hits (int):  The max hits to plot.
        truth (str):  Path to the input truth map in the case of simulations.
        xmin (float):  Fraction (0.0-1.0) of the minimum X view.
        xmax (float):  Fraction (0.0-1.0) of the maximum X view.
        ymin (float):  Fraction (0.0-1.0) of the minimum Y view.
        ymin (float):  Fraction (0.0-1.0) of the maximum Y view.
        is_azimuth (bool):  If True, swap direction of longitude axis.
        cmap (str): The color map name to use.
        format (str): The output image format.

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figdpi = 100

    current_cmap = mpl.colormaps[cmap]
    current_cmap.set_bad(color="gray")

    def plot_single(wcs, hdata, hindx, vmin, vmax, out):
        xwcs = wcs.pixel_shape[0]
        ywcs = wcs.pixel_shape[1]
        fig_x = xwcs / figdpi
        fig_y = ywcs / figdpi
        figsize = (fig_x, fig_y)
        fig = plt.figure(figsize=figsize, dpi=figdpi)
        ax = fig.add_subplot(projection=wcs, slices=("x", "y", hindx))
        im = ax.imshow(
            hdata[hindx, :, :],
            cmap=current_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        if is_azimuth:
            ax.invert_xaxis()
        ax.grid(color="white", ls="solid")
        ax.set_xlabel(f"{wcs.wcs.ctype[0]}")
        ax.set_ylabel(f"{wcs.wcs.ctype[1]}")
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        plt.colorbar(im, orientation="vertical")
        plt.savefig(out, format=format)
        plt.close()

    def map_range(hdata):
        minval = np.amin(hdata)
        maxval = np.amax(hdata)
        margin = 0.05 * (maxval - minval)
        if margin == 0:
            margin = -1
        minval -= margin
        maxval += margin
        return minval, maxval

    def sym_range(hdata):
        minval, maxval = map_range(hdata)
        ext = max(np.absolute(minval), np.absolute(maxval))
        return -ext, ext

    def flag_unhit(hitmask, mdata):
        if hitmask is None:
            return
        for mindx in range(mdata.shape[0]):
            mdata[mindx, hitmask] = np.nan

    def sub_mono(mdata):
        goodpix = np.logical_and(
            np.logical_not(np.isnan(mdata)),
            mdata != 0,
        )
        mono = np.mean(mdata[goodpix])
        print(f"Monopole = {mono}")
        mdata[goodpix] -= mono

    hitmask = None
    if hitfile is not None:
        hdulist = af.open(hitfile)
        hdu = hdulist[0]
        hitmask = np.array(hdu.data[0, :, :] == 0)
        wcs = WCS(hdu.header)
        maxhits = 0.5 * np.amax(hdu.data[0, :, :])
        if max_hits is not None:
            maxhits = max_hits
        plot_single(
            wcs,
            hdu.data,
            0,
            0,
            maxhits,
            plot_map_path(hitfile, format, out_dir=out_dir),
        )
        del hdu
        hdulist.close()

    if mapfile is not None:
        hdulist = af.open(mapfile)
        hdu = hdulist[0]
        wcs = WCS(hdu.header)
        mapdata = np.array(hdu.data)
        del hdu

        if truth is not None:
            thdulist = af.open(truth)
            thdu = thdulist[0]

        flag_unhit(hitmask, mapdata)

        sub_mono(mapdata[0])
        mmin, mmax = sym_range(mapdata[0, :, :])
        if range_I is not None:
            mmin, mmax = range_I
        plot_single(
            wcs,
            mapdata,
            0,
            mmin,
            mmax,
            plot_map_path(mapfile, format, suffix="_I", out_dir=out_dir),
        )
        if truth is not None:
            tmin, tmax = sym_range(thdu.data[0, :, :])
            mapdata[0, :, :] -= thdu.data[0, :, :]
            plot_single(
                wcs,
                mapdata,
                0,
                tmin,
                tmax,
                plot_map_path(mapfile, format, suffix="_resid_I", out_dir=out_dir),
            )

        if mapdata.shape[0] > 1:
            sub_mono(mapdata[1])
            mmin, mmax = sym_range(mapdata[1, :, :])
            if range_Q is not None:
                mmin, mmax = range_Q
            plot_single(
                wcs,
                mapdata,
                1,
                mmin,
                mmax,
                plot_map_path(mapfile, format, suffix="_Q", out_dir=out_dir),
            )
            if truth is not None:
                tmin, tmax = sym_range(thdu.data[1, :, :])
                mapdata[1, :, :] -= thdu.data[1, :, :]
                plot_single(
                    wcs,
                    mapdata,
                    1,
                    tmin,
                    tmax,
                    plot_map_path(mapfile, format, suffix="_resid_Q", out_dir=out_dir),
                )

            sub_mono(mapdata[2])
            mmin, mmax = sym_range(mapdata[2, :, :])
            if range_U is not None:
                mmin, mmax = range_U
            plot_single(
                wcs,
                mapdata,
                2,
                mmin,
                mmax,
                plot_map_path(mapfile, format, suffix="_U", out_dir=out_dir),
            )
            if truth is not None:
                tmin, tmax = sym_range(thdu.data[2, :, :])
                mapdata[2, :, :] -= thdu.data[2, :, :]
                plot_single(
                    wcs,
                    mapdata,
                    2,
                    tmin,
                    tmax,
                    plot_map_path(mapfile, format, suffix="_resid_U", out_dir=out_dir),
                )

        if truth is not None:
            del thdu
            thdulist.close()

        hdulist.close()


def plot_projected_quats(
    outfile,
    qbore=None,
    qdet=None,
    valid=slice(None),
    scale=1.0,
    equal_aspect=False,
):
    """Plot a list of quaternion arrays in longitude / latitude."""

    set_matplotlib_backend()
    import matplotlib.pyplot as plt

    # Convert boresight and detector quaternions to angles

    qbang = None
    if qbore is not None:
        qbang = np.zeros((3, qbore.shape[0]), dtype=np.float64)
        qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(qbore)
        qbang[0] *= 180.0 / np.pi
        qbang[1] *= 180.0 / np.pi
        lon_min = np.amin(qbang[0])
        lon_max = np.amax(qbang[0])
        lat_min = np.amin(qbang[1])
        lat_max = np.amax(qbang[1])

    qdang = None
    if qdet is not None:
        n_qdet = len(qdet)
        n_samp = len(qdet[0])
        qdang = np.zeros((n_qdet, 3, n_samp), dtype=np.float64)
        for det in range(n_qdet):
            qdang[det, 0], qdang[det, 1], qdang[det, 2] = qa.to_lonlat_angles(qdet[det])
            qdang[det, 0] *= 180.0 / np.pi
            qdang[det, 1] *= 180.0 / np.pi
        lon_min = np.amin(qdang[:, 0])
        lon_max = np.amax(qdang[:, 0])
        lat_min = np.amin(qdang[:, 1])
        lat_max = np.amax(qdang[:, 1])

    # Set the sizes of shapes based on the plot range

    span_lon = lon_max - lon_min
    lon_max += 0.1 * span_lon
    lon_min -= 0.1 * span_lon
    span_lat = lat_max - lat_min
    lat_max += 0.1 * span_lat
    lat_min -= 0.1 * span_lat
    if equal_aspect:
        if span_lon > span_lat:
            diff = span_lon - span_lat
            span_lat = span_lon
            lat_max += diff / 2
            lat_min -= diff / 2
        else:
            diff = span_lat - span_lon
            span_lon = span_lat
            lon_max += diff / 2
            lon_min -= diff / 2
        span = span_lon
    else:
        span = min(span_lon, span_lat)

    bmag = 0.03 * span * scale
    dmag = 0.02 * span * scale

    if span_lat > span_lon:
        fig_y = 10
        fig_x = fig_y * (span_lon / span_lat)
        if fig_x < 4:
            fig_x = 4
    else:
        fig_x = 10
        fig_y = fig_x * (span_lat / span_lon)
        if fig_y < 4:
            fig_y = 4

    figdpi = 100

    fig = plt.figure(figsize=(fig_x, fig_y), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    # Compute the font size to use for detector labels
    fontpix = 0.1 * figdpi
    fontpt = int(0.75 * fontpix)

    # Recall that so far we are working with angles looking "inward"
    # from the celestial sphere.  Instead, we want to plot the locations
    # and orientation angles looking "outward" as projected on the sky.
    # This means that the longitude axis is inverted.  Also, in our formalism,
    # the HWP and detector orientations are at zero when aligned with the
    # boresight frame X-axis.  For ground-based experiments, the boresight
    # frame X-axis is aligned with the direction of decreasing elevation.

    # Plot boresight if we have it

    if qbang is not None:
        ax.scatter(qbang[0][valid], qbang[1][valid], color="black", marker="x")
        for ln, lt, ps in np.transpose(qbang)[valid]:
            wd = 0.05 * bmag
            dx = bmag * np.sin(ps)
            dy = -bmag * np.cos(ps)
            ax.arrow(
                ln,
                lt,
                dx,
                dy,
                width=wd,
                head_width=4.0 * wd,
                head_length=0.2 * bmag,
                length_includes_head=True,
                ec="red",
                fc="red",
            )

    # Plot detectors if we have them

    if qdang is not None:
        for idet, dang in enumerate(qdang):
            ax.scatter(dang[0][valid], dang[1][valid], color="blue", marker=".")
            for ln, lt, ps in np.transpose(dang)[valid]:
                wd = 0.05 * dmag
                dx = dmag * np.sin(ps)
                dy = -dmag * np.cos(ps)
                ax.arrow(
                    ln,
                    lt,
                    dx,
                    dy,
                    width=wd,
                    head_width=4.0 * wd,
                    head_length=0.2 * dmag,
                    length_includes_head=True,
                    ec="blue",
                    fc="blue",
                )
            ax.text(
                dang[0][valid][0] + (idet % 2) * 1.5 * dmag,
                dang[1][valid][0] + 1.0 * dmag,
                f"{idet:02d}",
                color="k",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

    ax.set_xlim((lon_min, lon_max))
    ax.set_ylim((lat_min, lat_max))
    ax.set_xlabel("Longitude Degrees", fontsize="medium")
    ax.set_ylabel("Latitude Degrees", fontsize="medium")

    fig.suptitle("Projected Pointing and Polarization on Sky")

    # Invert x axis so that longitude reflects what we would see from
    # inside the celestial sphere
    plt.gca().invert_xaxis()

    plt.savefig(outfile)
    plt.close()


def plot_healpix_maps(
    hitfile=None,
    mapfile=None,
    range_I=None,
    range_Q=None,
    range_U=None,
    max_hits=None,
    truth=None,
    gnomview=False,
    gnomres=None,
    cmap="viridis",
    format="pdf",
    out_dir=None,
):
    """Plot Healpix projected output maps.

    This is a helper function to plot typical outputs of the mapmaker.

    Args:
        hitfile (str):  Path to the hits file.
        mapfile (str):  Path to the map file.
        range_I (tuple):  The min / max values of the Intensity map to plot.
        range_Q (tuple):  The min / max values of the Q map to plot.
        range_U (tuple):  The min / max values of the U map to plot.
        max_hits (int):  The max hits value to plot.
        truth (str):  Path to the input truth map in the case of simulations.
        gnomview (bool):  If True, use a gnomview projection centered on the
            mean of hit pixel locations.
        gnomres (float):  The resolution in arcminutes to pass to gnomview.
            If None, it will be estimated from the data.
        cmap (str): The color map name to use.
        format (str): The output image format.

    """
    set_matplotlib_backend()

    import matplotlib.pyplot as plt

    figsize = (12, 6)
    figdpi = 100

    def plot_single(data, vmin, vmax, out, gnomrot=None, reso=4.0, xsize=1000):
        file_base = os.path.splitext(os.path.basename(out))[0]
        if gnomrot is not None:
            hp.gnomview(
                map=data,
                rot=gnomrot,
                xsize=xsize,
                reso=reso,
                nest=True,
                cmap=cmap,
                min=vmin,
                max=vmax,
                title=file_base,
            )
        else:
            hp.mollview(
                data,
                xsize=xsize,
                nest=True,
                cmap=cmap,
                min=vmin,
                max=vmax,
                title=file_base,
            )
        plt.savefig(out, format=format)
        plt.close()

    def map_range(data):
        minval = np.amin(data)
        maxval = np.amax(data)
        margin = 0.05 * (maxval - minval)
        if margin == 0:
            margin = -1
        minval -= margin
        maxval += margin
        return minval, maxval

    def sym_range(data):
        minval, maxval = map_range(data)
        ext = max(np.absolute(minval), np.absolute(maxval))
        return -ext, ext

    hitdata = None
    gnomrot = None
    xsize = 1600
    goodhits = slice(None)
    if hitfile is not None:
        hitdata = read_healpix(hitfile, field=None, nest=True)
        maxhits = np.amax(hitdata)
        if max_hits is not None:
            maxhits = max_hits
        npix = len(hitdata)
        goodhits = hitdata > 0
        goodindx = np.arange(npix, dtype=np.int32)[goodhits]
        lon, lat = hp.pix2ang(
            hp.npix2nside(len(hitdata)),
            goodindx,
            nest=True,
            lonlat=True,
        )
        mlon = np.mean(lon)
        mlat = np.mean(lat)
        if gnomres is None:
            gnomres = 1.1 * (np.amax(lat) - np.amin(lat)) / xsize
            gnomres *= 60
        if gnomview:
            gnomrot = (mlon, mlat, 0.0)
            print(f"Using gnomview reso={gnomres}, gnomrot={gnomrot}")
        plot_single(
            hitdata,
            0,
            maxhits,
            plot_map_path(hitfile, format, out_dir=out_dir),
            gnomrot=gnomrot,
            reso=gnomres,
            xsize=xsize,
        )
    badhits = np.logical_not(goodhits)

    mapdata = None
    truthdata = None
    if mapfile is not None:
        mapdata = read_healpix(mapfile, field=None, nest=True)
        if truth is not None:
            truthdata = hp.read_map(truth, field=None, nest=True)

        if len(mapdata.shape) > 1:
            # We have Q/U data too
            imapdata = mapdata[0]
            qmapdata = mapdata[1]
            umapdata = mapdata[2]
        else:
            # Only I
            imapdata = mapdata
        # Stokes I
        imapdata[badhits] = hp.UNSEEN
        mono = np.mean(imapdata[goodhits])
        print(f"Monopole = {mono}")
        imapdata[goodhits] -= mono

        mmin, mmax = sym_range(imapdata[goodhits])
        if range_I is not None:
            mmin, mmax = range_I
        plot_single(
            imapdata,
            mmin,
            mmax,
            plot_map_path(mapfile, format, suffix="_I", out_dir=out_dir),
            gnomrot=gnomrot,
            reso=gnomres,
            xsize=xsize,
        )
        if truth is not None:
            truthdata[0][badhits] = hp.UNSEEN
            tmin, tmax = sym_range(truthdata[0][goodhits])
            plot_single(
                truthdata[0],
                tmin,
                tmax,
                plot_map_path(mapfile, format, suffix="_input_I", out_dir=out_dir),
                gnomrot=gnomrot,
                reso=gnomres,
                xsize=xsize,
            )
            imapdata[goodhits] -= truthdata[0][goodhits]
            plot_single(
                imapdata,
                tmin,
                tmax,
                plot_map_path(mapfile, format, suffix="_resid_I", out_dir=out_dir),
                gnomrot=gnomrot,
                reso=gnomres,
                xsize=xsize,
            )

        if len(mapdata.shape) > 1:
            qmapdata[badhits] = hp.UNSEEN
            umapdata[badhits] = hp.UNSEEN

            # Stokes Q
            mmin, mmax = sym_range(qmapdata[goodhits])
            if range_Q is not None:
                mmin, mmax = range_Q
            plot_single(
                qmapdata,
                mmin,
                mmax,
                plot_map_path(mapfile, format, suffix="_Q", out_dir=out_dir),
                gnomrot=gnomrot,
                reso=gnomres,
                xsize=xsize,
            )
            if truth is not None:
                truthdata[1][badhits] = hp.UNSEEN
                tmin, tmax = sym_range(truthdata[1][goodhits])
                plot_single(
                    truthdata[1],
                    tmin,
                    tmax,
                    plot_map_path(mapfile, format, suffix="_input_Q", out_dir=out_dir),
                    gnomrot=gnomrot,
                    reso=gnomres,
                    xsize=xsize,
                )
                qmapdata[goodhits] -= truthdata[1][goodhits]
                plot_single(
                    qmapdata,
                    tmin,
                    tmax,
                    plot_map_path(mapfile, format, suffix="_resid_Q", out_dir=out_dir),
                    gnomrot=gnomrot,
                    reso=gnomres,
                    xsize=xsize,
                )

            # Stokes U
            mmin, mmax = sym_range(umapdata[goodhits])
            if range_U is not None:
                mmin, mmax = range_U
            plot_single(
                umapdata,
                mmin,
                mmax,
                plot_map_path(mapfile, format, suffix="_U", out_dir=out_dir),
                gnomrot=gnomrot,
                reso=gnomres,
                xsize=xsize,
            )
            if truth is not None:
                truthdata[2][badhits] = hp.UNSEEN
                tmin, tmax = sym_range(truthdata[2][goodhits])
                plot_single(
                    truthdata[2],
                    tmin,
                    tmax,
                    plot_map_path(mapfile, format, suffix="_input_U", out_dir=out_dir),
                    gnomrot=gnomrot,
                    reso=gnomres,
                    xsize=xsize,
                )
                umapdata[goodhits] -= truthdata[2][goodhits]
                plot_single(
                    umapdata,
                    tmin,
                    tmax,
                    plot_map_path(mapfile, format, suffix="_resid_U", out_dir=out_dir),
                    gnomrot=gnomrot,
                    reso=gnomres,
                    xsize=xsize,
                )

    del truthdata
    del mapdata
    del hitdata
