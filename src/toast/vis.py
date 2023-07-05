# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import warnings

import astropy.io.fits as af
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

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
                label=f"NET = {net:0.2e} K" + " / $\sqrt{\mathrm{Hz}}$",
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


def plot_wcs_maps(
    hitfile=None, mapfile=None, range_I=None, range_Q=None, range_U=None, truth=None
):
    """Plot WCS projected output maps.

    This is a helper function to plot typical outputs of the mapmaker.

    Args:
        hitfile (str):  Path to the hits file.
        mapfile (str):  Path to the map file.
        range_I (tuple):  The min / max values of the Intensity map to plot.
        range_Q (tuple):  The min / max values of the Q map to plot.
        range_U (tuple):  The min / max values of the U map to plot.
        truth (str):  Path to the input truth map in the case of simulations.

    """
    import matplotlib.pyplot as plt

    figsize = (12, 12)
    figdpi = 100

    def plot_single(wcs, hdata, hindx, vmin, vmax, out):
        fig = plt.figure(figsize=figsize, dpi=figdpi)
        ax = fig.add_subplot(projection=wcs, slices=("x", "y", hindx))
        im = ax.imshow(
            np.transpose(hdu.data[hindx, :, :]), cmap="jet", vmin=vmin, vmax=vmax
        )
        ax.grid(color="white", ls="solid")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, orientation="vertical")
        plt.savefig(out, format="pdf")
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

    def sub_mono(hitdata, mdata):
        if hitdata is None:
            return
        goodpix = np.logical_and((hitdata > 0), (mdata != 0))
        mono = np.mean(mdata[goodpix])
        print(f"Monopole = {mono}")
        mdata[goodpix] -= mono
        mdata[np.logical_not(goodpix)] = 0

    hitdata = None
    if hitfile is not None:
        hdulist = af.open(hitfile)
        hdu = hdulist[0]
        hitdata = np.array(hdu.data[0, :, :])
        wcs = WCS(hdu.header)
        maxhits = np.amax(hdu.data[0, :, :])
        plot_single(wcs, hdu, 0, 0, maxhits, f"{hitfile}.pdf")
        del hdu
        hdulist.close()

    if mapfile is not None:
        hdulist = af.open(mapfile)
        hdu = hdulist[0]
        wcs = WCS(hdu.header)

        if truth is not None:
            thdulist = af.open(truth)
            thdu = thdulist[0]

        sub_mono(hitdata, hdu.data[0, :, :])
        mmin, mmax = sym_range(hdu.data[0, :, :])
        if range_I is not None:
            mmin, mmax = range_I
        plot_single(wcs, hdu, 0, mmin, mmax, f"{mapfile}_I.pdf")
        if truth is not None:
            tmin, tmax = sym_range(thdu.data[0, :, :])
            hdu.data[0, :, :] -= thdu.data[0, :, :]
            plot_single(wcs, hdu, 0, tmin, tmax, f"{mapfile}_resid_I.pdf")

        if hdu.data.shape[0] > 1:
            mmin, mmax = sym_range(hdu.data[1, :, :])
            if range_Q is not None:
                mmin, mmax = range_Q
            plot_single(wcs, hdu, 1, mmin, mmax, f"{mapfile}_Q.pdf")
            if truth is not None:
                tmin, tmax = sym_range(thdu.data[1, :, :])
                hdu.data[1, :, :] -= thdu.data[1, :, :]
                plot_single(wcs, hdu, 1, tmin, tmax, f"{mapfile}_resid_Q.pdf")

            mmin, mmax = sym_range(hdu.data[2, :, :])
            if range_U is not None:
                mmin, mmax = range_U
            plot_single(wcs, hdu, 2, mmin, mmax, f"{mapfile}_U.pdf")
            if truth is not None:
                tmin, tmax = sym_range(thdu.data[2, :, :])
                hdu.data[2, :, :] -= thdu.data[2, :, :]
                plot_single(wcs, hdu, 2, tmin, tmax, f"{mapfile}_resid_U.pdf")

        if truth is not None:
            del thdu
            thdulist.close()
        del hdu
        hdulist.close()
