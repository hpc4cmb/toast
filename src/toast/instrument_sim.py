# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from . import qarray as qa

from .instrument import Focalplane


def cartesian_to_quat(offsets):
    """Convert cartesian angle offsets and rotation into quaternions.

    Focalplane geometries are often described in terms of wafer locations or
    separations given in simple X/Y angle offsets from a center point.
    this helper function converts such parameters into a quaternion describing
    the rotation.

    Args:
        offsets (list of arrays):  each item of the list has 3 elements for
            the X / Y angle offsets in degrees and the rotation in degrees
            about the Z axis.

    Returns:
        (list): list of quaternions for each item in the input list.

    """
    centers = list()
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    for off in offsets:
        angrot = qa.rotation(zaxis, off[2] * np.pi / 180.0)
        wx = off[0] * np.pi / 180.0
        wy = off[1] * np.pi / 180.0
        wz = np.sqrt(1.0 - (wx * wx + wy * wy))
        wdir = np.array([wx, wy, wz])
        posrot = qa.from_vectors(zaxis, wdir)
        centers.append(qa.mult(posrot, angrot))
    return centers


def hex_nring(npix):
    """
    For a hexagonal layout with a given number of pixels, return the
    number of rings.
    """
    test = npix - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    if test != 0:
        raise RuntimeError(
            "{} is not a valid number of pixels for a hexagonal layout".format(npix)
        )
    return nrings


def hex_row_col(npix, pix):
    """
    For a hexagonal layout, indexed in a "spiral" scheme (see hex_layout),
    this function returnes the "row" and "column" of a pixel.
    The row is zero along the main vertex-vertex axis, and is positive
    or negative above / below this line of pixels.
    """
    if pix >= npix:
        raise ValueError("pixel value out of range")
    test = npix - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    if pix == 0:
        row = 0
        col = nrings - 1
    else:
        test = pix - 1
        ring = 1
        while (test - 6 * ring) >= 0:
            test -= 6 * ring
            ring += 1
        sector = int(test / ring)
        steps = np.mod(test, ring)
        coloff = nrings - ring - 1
        if sector == 0:
            row = steps
            col = coloff + 2 * ring - steps
        elif sector == 1:
            row = ring
            col = coloff + ring - steps
        elif sector == 2:
            row = ring - steps
            col = coloff
        elif sector == 3:
            row = -steps
            col = coloff
        elif sector == 4:
            row = -ring
            col = coloff + steps
        elif sector == 5:
            row = -ring + steps
            col = coloff + ring + steps
    return (row, col)


def hex_pol_angles_qu(npix, offset=0.0):
    """Generates a vector of detector polarization angles.

    The returned angles can be used to construct a hexagonal detector layout.
    This scheme alternates pixels between 0/90 and +/- 45 degrees.

    Args:
        npix (int): the number of pixels locations in the hexagon.
        offset (float): the constant angle offset in degrees to apply.

    Returns:
        (array):  The detector polarization angles.

    """
    pol = np.zeros(npix, dtype=np.float64)
    for pix in range(npix):
        # get the row / col of the pixel
        row, col = hex_row_col(npix, pix)
        if np.mod(col, 2) == 0:
            pol[pix] = 0.0 + offset
        else:
            pol[pix] = 45.0 + offset
    return pol


def hex_pol_angles_radial(npix, offset=0.0):
    """Generates a vector of detector polarization angles.

    The returned angles can be used to construct a hexagonal detector layout.
    This scheme orients the bolometer along the radial direction of the
    hexagon.

    Args:
        npix (int): the number of pixels locations in the hexagon.
        offset (float): the constant angle offset in degrees to apply.

    Returns:
        (array):  The detector polarization angles.

    """
    sixty = np.pi / 3.0
    thirty = np.pi / 6.0
    pol = np.zeros(npix, dtype=np.float64)
    pol[0] = 0.0
    for pix in range(1, npix):
        # find ring for this pix
        test = pix - 1
        ring = 1
        while (test - 6 * ring) >= 0:
            test -= 6 * ring
            ring += 1
        sectors = int(test / ring)
        sectorsteps = np.mod(test, ring)
        midline = 0.5 * np.sqrt(3) * float(ring)
        edgedist = float(sectorsteps) - 0.5 * float(ring)
        relang = np.arctan2(edgedist, midline)
        pol[pix] = (sectors * sixty + thirty + relang) * 180.0 / np.pi + offset
    return pol


def hex_layout(
    npix, angwidth, prefix, suffix, pol, center=np.array([0, 0, 0, 1], dtype=np.float64)
):
    """Return detectors in a hexagon layout.

    This maps the physical positions of pixels into angular positions
    from the hexagon center.  The X axis in the hexagon frame is along
    the vertex-to-opposite-vertex direction.  The Y axis is along
    flat-to-opposite-flat direction.  The origin is at the center of
    the wafer.  For example::

        Y ^             O O O
        |              O O O O
        |             O O + O O
        +--> X         O O O O
                        O O O

    Each pixel is numbered 1..npix and each detector is named by the
    prefix, the pixel number, and the suffix.  The first pixel is at
    the center, and then the pixels are numbered moving outward in rings.

    The extent of the hexagon is directly specified by the angwidth
    parameter.  These, along with the npix parameter, constrain the packing
    locations of the pixel centers.

    Args:
        npix (int): number of pixels packed onto wafer.
        angwidth (float): the angle (in degrees) subtended by the width.
        prefix (str): the detector name prefix.
        suffix (str): the detector name suffix.
        pol (ndarray): 1D array of detector polarization angles.  The
            rotation is applied to the hexagon center prior to rotation
            to the pixel location.
        center (ndarray): quaternion offset of the center of the layout.

    Returns:
        (dict) A dictionary keyed on detector name, with each value itself a
            dictionary of detector properties.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
    sixty = np.pi / 3.0
    thirty = np.pi / 6.0
    rtthree = np.sqrt(3.0)
    rtthreebytwo = 0.5 * rtthree

    angwidth = angwidth * np.pi / 180.0

    # compute the diameter (vertex to vertex width)
    angdiameter = angwidth / np.cos(thirty)

    # find the angular packing size of one detector
    test = npix - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    pixdiam = angdiameter / (2 * nrings - 1)

    # convert pol vector to radians
    pol *= np.pi / 180.0

    # number of digits for pixel indexing

    ndigit = 0
    test = npix
    while test > 0:
        test = test // 10
        ndigit += 1

    nameformat = "{{}}{{:0{}d}}{{}}".format(ndigit)

    # compute positions of all detectors

    dets = {}

    for pix in range(npix):
        dname = nameformat.format(prefix, pix, suffix)

        polrot = qa.rotation(zaxis, pol[pix])

        # center pixel has no offset
        pixrot = nullquat

        if pix != 0:
            # Not at the center, find ring for this pix
            test = pix - 1
            ring = 1
            while (test - 6 * ring) >= 0:
                test -= 6 * ring
                ring += 1
            sectors = int(test / ring)
            sectorsteps = np.mod(test, ring)

            # Convert angular steps around the ring into the angle and distance
            # in polar coordinates.  Each "sector" of 60 degrees is essentially
            # an equilateral triangle, and each step is equally spaced along the
            # the edge opposite the vertex:
            #
            #          O
            #         O O (step 2)
            #        O   O (step 1)
            #       X O O O (step 0)
            #
            # For a given ring, "R" (center is R=0), there are R steps along
            # the sector edge.  The line from the origin to the opposite edge
            # that bisects this triangle has length R*sqrt(3)/2.  For each
            # equally-spaced step, we use the right triangle formed with this
            # bisection line to compute the angle and radius within this sector.

            # the distance from the origin to the midpoint of the opposite side.
            midline = rtthreebytwo * float(ring)

            # the distance along the opposite edge from the midpoint (positive
            # or negative)
            edgedist = float(sectorsteps) - 0.5 * float(ring)

            # the angle relative to the midpoint line (positive or negative)
            relang = np.arctan2(edgedist, midline)

            # total angle is based on number of sectors we have and the angle
            # within the final sector.
            pixang = sectors * sixty + thirty + relang

            pixdist = rtthreebytwo * pixdiam * float(ring) / np.cos(relang)

            pixx = np.sin(pixdist) * np.cos(pixang)
            pixy = np.sin(pixdist) * np.sin(pixang)
            pixz = np.cos(pixdist)
            pixdir = np.array([pixx, pixy, pixz], dtype=np.float64)
            norm = np.sqrt(np.dot(pixdir, pixdir))
            pixdir /= norm

            pixrot = qa.from_vectors(zaxis, pixdir)

        dprops = {}
        dprops["quat"] = qa.mult(center, qa.mult(pixrot, polrot))
        dprops["polangle_deg"] = pol[pix]

        dets[dname] = dprops

    return dets


def rhomb_dim(npix):
    """
    For a rhombus layout, return the dimension of one side.
    """
    dim = int(np.sqrt(float(npix)))
    if dim ** 2 != npix:
        raise ValueError("number of pixels for a rhombus wafer must be square")
    return dim


def rhomb_row_col(npix, pix):
    """
    For a rhombus layout, indexed from top to bottom (see rhombus_layout),
    this function returnes the "row" and "column" of a pixel.  The column
    starts at zero on the left hand side of a row.
    """
    if pix >= npix:
        raise ValueError("pixel value out of range")
    dim = rhomb_dim(npix)
    col = pix
    rowcnt = 1
    row = 0
    while (col - rowcnt) >= 0:
        col -= rowcnt
        row += 1
        if row >= dim:
            rowcnt -= 1
        else:
            rowcnt += 1
    return (row, col)


def rhomb_pol_angles_qu(npix, offset=0.0):
    """Generates a vector of detector polarization angles.

    The returned angles can be used to construct a rhombus detector layout.
    This scheme alternates pixels between 0/90 and +/- 45 degrees.

    Args:
        npix (int): the number of pixels locations in the rhombus.
        offset (float): the constant angle offset in degrees to apply.

    Returns:
        (array): The detector polarization angles.

    """
    pol = np.zeros(npix, dtype=np.float64)
    for pix in range(npix):
        # get the row / col of the pixel
        row, col = rhomb_row_col(npix, pix)
        if np.mod(col, 2) == 0:
            pol[pix] = 45.0 + offset
        else:
            pol[pix] = 0.0 + offset
    return pol


def rhombus_layout(
    npix,
    angwidth,
    prefix,
    suffix,
    polang,
    center=np.array([0, 0, 0, 1], dtype=np.float64),
):
    """Return detectors in a rhombus layout.

    This particular rhombus geometry is essentially a third of a
    hexagon.  In other words the aspect ratio of the rhombus is
    constrained to have the long dimension be sqrt(3) times the short
    dimension.

    This function maps the physical positions of pixels into angular
    positions from the rhombus center.  The X axis is along the short
    direction.  The Y axis is along longer direction.  The origin is
    at the center of the rhombus.  For example::

                          O
        Y ^              O O
        |               O O O
        |              O O O O
        +--> X          O O O
                         O O
                          O

    Each pixel is numbered 1..npix and each detector is named by the
    prefix, the pixel number, and the suffix.  The first pixel is at the
    "top", and then the pixels are numbered moving downward and left to
    right.

    The extent of the rhombus is directly specified by the angwidth parameter.
    This, along with the npix parameter, constrain the packing locations of
    the pixel centers.

    Args:
        npix (int): number of pixels packed onto wafer.
        angwidth (float): the angle (in degrees) subtended by the short
            dimension.
        prefix (str): the detector name prefix.
        suffix (str): the detector name suffix.
        polang (ndarray): 1D array of detector polarization angles.  The
            rotation is applied to the hexagon center prior to rotation
            to the pixel location.
        center (ndarray): quaternion offset of the center of the layout.

    Returns:
        (dict):  A dictionary keyed on detector name, with each value itself a
            dictionary of detector properties.

    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    nullquat = np.array([0, 0, 0, 1], dtype=np.float64)
    rtthree = np.sqrt(3.0)

    angwidth = angwidth * np.pi / 180.0
    dim = rhomb_dim(npix)

    # compute the height
    angheight = rtthree * angwidth

    # find the angular packing size of one detector
    pixdiam = angwidth / dim

    # convert pol vector to radians
    pol = polang * np.pi / 180.0

    # number of digits for pixel indexing

    ndigit = 0
    test = npix
    while test > 0:
        test = test // 10
        ndigit += 1

    nameformat = "{{}}{{:0{}d}}{{}}".format(ndigit)

    # compute positions of all detectors

    dets = {}

    for pix in range(npix):
        dname = nameformat.format(prefix, pix, suffix)

        polrot = qa.rotation(zaxis, pol[pix])

        pixrow, pixcol = rhomb_row_col(npix, pix)

        rowang = 0.5 * rtthree * ((dim - 1) - pixrow) * pixdiam
        relrow = pixrow
        if pixrow >= dim:
            relrow = (2 * dim - 2) - pixrow
        colang = (float(pixcol) - float(relrow) / 2.0) * pixdiam
        distang = np.sqrt(rowang ** 2 + colang ** 2)
        zang = np.cos(distang)
        pixdir = np.array([colang, rowang, zang], dtype=np.float64)
        norm = np.sqrt(np.dot(pixdir, pixdir))
        pixdir /= norm

        pixrot = qa.from_vectors(zaxis, pixdir)

        dprops = {}
        dprops["quat"] = qa.mult(center, qa.mult(pixrot, polrot))

        dets[dname] = dprops

    return dets


def fake_hexagon_focalplane(
    n_pix=7,
    width_deg=5.0,
    samplerate=1.0,
    epsilon=0.0,
    net=1.0,
    fmin=0.0,
    alpha=1.0,
    fknee=0.05,
):
    pol_A = hex_pol_angles_qu(n_pix, offset=0.0)
    pol_B = hex_pol_angles_qu(n_pix, offset=90.0)
    quat_A = hex_layout(n_pix, width_deg, "D", "A", pol_A)
    quat_B = hex_layout(n_pix, width_deg, "D", "B", pol_B)

    det_data = dict(quat_A)
    det_data.update(quat_B)

    nrings = hex_nring(n_pix)
    detfwhm = 0.5 * 60.0 * width_deg / (2 * nrings - 1)

    for det in det_data.keys():
        det_data[det]["pol_leakage"] = epsilon
        det_data[det]["fmin"] = fmin
        det_data[det]["fknee"] = fknee
        det_data[det]["alpha"] = alpha
        det_data[det]["NET"] = net
        det_data[det]["fwhm_arcmin"] = detfwhm
        det_data[det]["fsample"] = samplerate

    return Focalplane(detector_data=det_data, sample_rate=samplerate)


def plot_focalplane(
    dets, width, height, outfile, fwhm=None, facecolor=None, polcolor=None, labels=None
):
    """Visualize a dictionary of detectors.

    This makes a simple plot of the detector positions on the projected
    focalplane.

    To avoid python overhead in large MPI jobs, we place the matplotlib
    import inside this function, so that it is only imported when the
    function is actually called.

    If the detector dictionary contains a key "fwhm", that will be assumed
    to be in arcminutes.  Otherwise a nominal value is used.

    If the detector dictionary contains a key "viscolor", then that color
    will be used.

    Args:
        dets (dict): dictionary of detector quaternions.
        width (float): width of plot in degrees.
        height (float): height of plot in degrees.
        outfile (str): output PNG path.  If None, then matplotlib will be
            used for inline plotting.
        fwhm (dict): dictionary of detector beam FWHM in arcminutes, used
            to draw the circles to scale.
        facecolor (dict): dictionary of color values for the face of each
            detector circle.
        polcolor (dict): dictionary of color values for the polarization
            arrows.
        labels (dict): plot this text in the center of each pixel.

    Returns:
        None

    """
    if outfile is not None:
        import matplotlib
        import warnings

        # Try to force matplotlib to not use any Xwindows backend.
        warnings.filterwarnings("ignore")
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xfigsize = int(width)
    yfigsize = int(height)
    figdpi = 100

    # Compute the font size to use for detector labels
    fontpix = 0.2 * figdpi
    fontpt = int(0.75 * fontpix)

    fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    half_width = 0.5 * width
    half_height = 0.5 * height
    ax.set_xlabel("Degrees", fontsize="large")
    ax.set_ylabel("Degrees", fontsize="large")
    ax.set_xlim([-half_width, half_width])
    ax.set_ylim([-half_height, half_height])

    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    yaxis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for d, quat in dets.items():

        # radius in degrees
        detradius = 0.5 * 5.0 / 60.0
        if fwhm is not None:
            detradius = 0.5 * fwhm[d] / 60.0

        # rotation from boresight
        rdir = qa.rotate(quat, zaxis).flatten()
        ang = np.arctan2(rdir[1], rdir[0])

        orient = qa.rotate(quat, xaxis).flatten()
        polang = np.arctan2(orient[1], orient[0])

        mag = np.arccos(rdir[2]) * 180.0 / np.pi
        xpos = mag * np.cos(ang)
        ypos = mag * np.sin(ang)

        detface = "none"
        if facecolor is not None:
            detface = facecolor[d]

        circ = plt.Circle((xpos, ypos), radius=detradius, fc=detface, ec="k")
        ax.add_artist(circ)

        ascale = 2.0

        xtail = xpos - ascale * detradius * np.cos(polang)
        ytail = ypos - ascale * detradius * np.sin(polang)
        dx = ascale * 2.0 * detradius * np.cos(polang)
        dy = ascale * 2.0 * detradius * np.sin(polang)

        detcolor = "black"
        if polcolor is not None:
            detcolor = polcolor[d]

        ax.arrow(
            xtail,
            ytail,
            dx,
            dy,
            width=0.1 * detradius,
            head_width=0.3 * detradius,
            head_length=0.3 * detradius,
            fc=detcolor,
            ec=detcolor,
            length_includes_head=True,
        )

        if labels is not None:
            xsgn = 1.0
            if dx < 0.0:
                xsgn = -1.0
            labeloff = 0.05 * xsgn * fontpix * len(labels[d]) / figdpi
            ax.text(
                (xtail + 1.1 * dx + labeloff),
                (ytail + 1.1 * dy),
                labels[d],
                color="k",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=1.0),
            )

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
        plt.close()
    return fig
