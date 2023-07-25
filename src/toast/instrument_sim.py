# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from astropy import units as u
from astropy.table import Column, QTable

from . import qarray as qa
from .instrument import Focalplane
from .instrument_coords import quat_to_xieta, xieta_to_quat
from .vis import set_matplotlib_backend


def hex_nring(npix):
    """Return the number of rings in a hexagonal layout.

    For a hexagonal layout with a given number of positions, return the
    number of rings.

    Args:
        npos (int): The number of positions.

    Returns:
        (int): The number of rings.

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


def hex_xieta_row_col(npos, pos):
    """Return the location of a given hexagon position.

    This function is used when laying out polarization angles that need to alternate
    in a particular way.  The "row" is zero along the main vertex-vertex direction
    and can be positive or negative.  The "column" is always >= 0 and increments from
    left to right in Xi / Eta plane.  For example, the (row, col) values for npos=19
    would be:

                                    ( 2, 0)   ( 2, 1)   ( 2, 2)
        Eta ^
            |                   ( 1, 0)   ( 1, 1)   ( 1, 2)   ( 1, 3)
            |
            +--> Xi         ( 0, 0)  ( 0, 1)  ( 0, 2)  ( 0, 3)  ( 0, 4)

                                (-1, 0)   (-1, 1)   (-1, 2)   (-1, 3)

                                     (-2, 0)   (-2, 1)   (-2, 2)

    Args:
        npos (int): The number of positions.
        pos (int): The position.

    Returns:
        (tuple): The (row, column) location of the position.

    """
    if pos >= npos:
        raise ValueError("position value out of range")
    test = npos - 1
    nrings = 1
    while (test - 6 * nrings) >= 0:
        test -= 6 * nrings
        nrings += 1
    if pos == 0:
        row = 0
        col = nrings - 1
    else:
        test = pos - 1
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


def hex_gamma_angles_qu(npix, offset=u.Quantity(0.0, u.degree)):
    """Generates a vector of detector polarization angles.

    The returned angles can be used to construct a hexagonal detector layout.
    This scheme alternates pixels between 0/90 and +/- 45 degrees.  The angles
    specify the "gamma" rotation angle in the xi, eta, gamma coordinate system.

    Args:
        npix (int): the number of pixels locations in the hexagon.
        offset (float): the constant angle offset in degrees to apply.

    Returns:
        (array):  The detector polarization angles.

    """
    pol = np.zeros(npix, dtype=np.float64)
    for pix in range(npix):
        # get the row / col of the pixel
        row, col = hex_xieta_row_col(npix, pix)
        if np.mod(col, 2) == 0:
            pol[pix] = 0.0 + offset.to_value(u.degree)
        else:
            pol[pix] = 45.0 + offset.to_value(u.degree)
    return u.Quantity(pol, u.degree)


def hex_gamma_angles_radial(npix, offset=u.Quantity(0.0, u.degree)):
    """Generates a vector of detector polarization angles.

    The returned angles can be used to construct a hexagonal detector layout.
    This scheme orients the bolometer along the radial direction of the
    hexagon.  The angles specify the "gamma" rotation angle in the xi, eta, gamma
    coordinate system.

    Args:
        npix (int): the number of pixels locations in the hexagon.
        offset (Quantity): the constant angle offset to apply.

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

        # Note:  this is the counter-clockwise (right-handed) angle in the xi/eta
        # plane starting at the positive xi axis.
        pixang = sectors * sixty + thirty + relang

        # Convert to the gamma angle, which is zero at the X (-eta) axis and
        # increasing in a right-handed sense from that axis in the X/Y plane.
        pol[pix] = (1.5 * np.pi - pixang) * 180.0 / np.pi + offset.to_value(u.degree)
    return u.Quantity(pol, u.degree)


def hex_layout(npos, angwidth, prefix, suffix, pol, center=None, pos_offset=0):
    """Construct a hexagonal layout of positions.

    This function first creates a hexagon of positions using the Xi / Eta / Gamma
    projected coordinate system.  The array of "pol" angles specify the gamma
    rotation angle clockwise from the Eta axis.  It then converts each of these
    positions into a quaternion that describes the rotation from the hexagon
    X / Y / Z coordinate frame into the detector frame with the Z axis along the line
    of sight and the X axis along the polarization sensitive direction.

    For example with 19 positions:

                           11    10    09           Eta
                                                    ^
                        12    03    02    08        |
                                                    |
        Y <---+     13     04    00    01     07    +---> Xi
              |
              |         14    05    06    18
              V
                           15    16    17
              X

    Each pixel is numbered 0...npos-1 and each detector is named by the
    prefix, the pixel number, and the suffix.  The first pixel is at
    the center, and then the pixels are numbered moving outward in rings.

    The extent of the hexagon is directly specified by the angwidth
    parameter.  These, along with the npos parameter, constrain the packing
    locations of the pixel centers.

    NOTE:  The "angwidth" parameter specifies the width across the long axis
    (the vertex-vertex axis).  This is the distance between the pixel centers,
    not including any beam FWHM or field of view.

    If the "center" argument is specified, then each quaternion is additionally
    multiplied by this in order to shift all positions to be relative to a new
    center.

    Args:
        npos (int): number of pixels packed onto wafer.
        angwidth (Quantity): the angle subtended by the width.
        prefix (str): the detector name prefix.
        suffix (str): the detector name suffix.
        pol (Quantity): 1D array of detector polarization angles.  These are the
            "gamma" angle in the xi / eta / gamma coordinate system.
        center (ndarray): quaternion offset of the center of the layout (or None).
        pos_offset (int): starting index of position numbers.

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

    # The diameter (vertex to vertex width)
    angdiameter = angwidth.to_value(u.radian)

    # Find the angular packing size of one pixel.  A pixel extends halfway
    # to its neighbors, and so the diameter is equal to the pixel center
    # distance.
    nrings = hex_nring(npos)
    pixdiam = angdiameter / (2 * nrings - 2)

    # number of digits for pixel indexing
    ndigit = int(np.log10(npos)) + 1
    nameformat = "{{}}{{:0{}d}}{{}}".format(ndigit)

    # compute positions of all detectors

    dets = {}

    for pix in range(npos):
        dname = nameformat.format(prefix, pix + pos_offset, suffix)

        xi = 0
        eta = 0
        gamma = pol[pix].to_value(u.radian)

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

            xi = np.sin(pixdist) * np.cos(pixang)
            eta = np.sin(pixdist) * np.sin(pixang)

        dprops = {}
        if center is None:
            dprops["quat"] = xieta_to_quat(xi, eta, gamma)
            dprops["gamma"] = gamma
        else:
            dprops["quat"] = qa.mult(center, xieta_to_quat(xi, eta, gamma))
            _, _, temp_gamma = quat_to_xieta(dprops["quat"])
            dprops["gamma"] = temp_gamma

        dets[dname] = dprops

    return dets


def rhomb_dim(npos):
    """Compute the dimensions of a rhombus.

    For a rhombus with the specified number of positions, return the dimension
    of one side.  This function is just a check around a sqrt.

    Args:
        npos (int): The number of positions.

    Returns:
        (int): The dimension of one side.

    """
    dim = int(np.sqrt(float(npos)))
    if dim**2 != npos:
        raise ValueError("number of positions for a rhombus layout must be square")
    return dim


def rhomb_xieta_row_col(npos, pos):
    """Return the location of a given position.

    For a rhombus layout, this function returnes the "row" and "column" of a
    position.  The column starts at zero on the left hand side of a row.
    For example, the (row, col) values for npos=16 would be:

                                          (0, 0)
        Eta ^
            |                         (1, 0)  (1, 1)
            |
            +--> Xi               (2, 0)  (2, 1)  (2, 2)

                              (3, 0)  (3, 1)  (3, 2)  (3, 3)

                                  (4, 0)  (4, 1)  (4, 2)

                                      (5, 0)  (5, 1)

                                          (6, 0)


    Args:
        npos (int): The number of positions.
        pos (int): The position.

    Returns:
        (tuple): The (row, column) location of the position.

    """
    if pos >= npos:
        raise ValueError("pixel value out of range")
    dim = rhomb_dim(npos)
    col = pos
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


def rhomb_gamma_angles_qu(npix, offset=u.Quantity(0.0, u.degree)):
    """Generates a vector of detector polarization angles.

    The returned angles can be used to construct a rhombus detector layout.
    This scheme alternates pixels between 0/90 and +/- 45 degrees.

    Args:
        npix (int): the number of pixels locations in the rhombus.
        offset (Quantity): the constant angle offset to apply.

    Returns:
        (array): The detector polarization angles.

    """
    pol = np.zeros(npix, dtype=np.float64)
    for pix in range(npix):
        # get the row / col of the pixel
        row, col = rhomb_xieta_row_col(npix, pix)
        if np.mod(col, 2) == 0:
            pol[pix] = 45.0 + offset.to_value(u.degree)
        else:
            pol[pix] = 0.0 + offset.to_value(u.degree)
    return u.Quantity(pol, u.degree)


def rhombus_layout(npos, angwidth, prefix, suffix, pol, center=None, pos_offset=0):
    """Return positions in a rhombus layout.

    This particular rhombus geometry is essentially a third of a hexagon.  In other
    words the aspect ratio of the rhombus is constrained to have the long dimension
    be sqrt(3) times the short dimension.

    This function first creates a rhombus of positions using the Xi / Eta / Gamma
    projected coordinate system.  The array of "pol" angles specify the gamma
    rotation angle clockwise from the Eta axis.  It then converts each of these
    positions into a quaternion that describes the rotation from the rhombus
    X / Y / Z coordinate frame into the detector frame with the Z axis along the line
    of sight and the X axis along the polarization sensitive direction.

    For example with 16 positions:

                              00

                           01    02         Eta
                                             ^
                        03    04    05       |
                                             |
        Y <---+     06     07    08    09    +---> Xi
              |
              |         10    11    12
              V
                           13    14
              X
                              15

    Each pixel is numbered 0...npos-1 and each detector is named by the prefix, the
    pixel number, and the suffix.  The first pixel is at the "top", and then the
    pixels are numbered moving downward and left to right.

    The extent of the rhombus is directly specified by the angwidth parameter.
    This, along with the npos parameter, constrain the packing locations of
    the pixel centers.

    NOTE:  The "angwidth" parameter is the angular distance from the extreme
    pixel centers along the short axis.

    Args:
        npos (int): number of pixels packed onto wafer.
        angwidth (Quantity): the angle subtended by the short dimension.
        prefix (str): the detector name prefix.
        suffix (str): the detector name suffix.
        pol (Quantity): 1D array of detector polarization angles.  These are the
            "gamma" angle in the xi / eta / gamma coordinate system.
        center (ndarray): quaternion offset of the center of the layout.
        pos_offset (int): starting index of position numbers.

    Returns:
        (dict):  A dictionary keyed on detector name, with each value itself a
            dictionary of detector properties.

    """
    rtthree = np.sqrt(3.0)

    # Dimensions of the rhombus
    dim = rhomb_dim(npos)

    # Find the angular packing size of one pixel
    pixdiam = angwidth.to_value(u.radian) / (dim - 1)

    # Number of digits for pixel indexing
    ndigit = int(np.log10(npos)) + 1
    nameformat = "{{}}{{:0{}d}}{{}}".format(ndigit)

    # Compute positions of all detectors

    dets = {}

    for pix in range(npos):
        dname = nameformat.format(prefix, pix + pos_offset, suffix)

        xi = 0
        eta = 0
        gamma = pol[pix].to_value(u.radian)

        pixrow, pixcol = rhomb_xieta_row_col(npos, pix)

        rowang = 0.5 * rtthree * ((dim - 1) - pixrow) * pixdiam
        relrow = pixrow
        if pixrow >= dim:
            relrow = (2 * dim - 2) - pixrow
        colang = (float(pixcol) - float(relrow) / 2.0) * pixdiam

        xi = colang
        eta = rowang

        dprops = {}
        if center is None:
            dprops["quat"] = xieta_to_quat(xi, eta, gamma)
            dprops["gamma"] = gamma
        else:
            dprops["quat"] = qa.mult(center, xieta_to_quat(xi, eta, gamma))
            _, _, temp_gamma = quat_to_xieta(dprops["quat"])
            dprops["gamma"] = temp_gamma

        dets[dname] = dprops

    return dets


def rhombus_hex_layout(
    rhombus_npos, rhombus_width, prefix, suffix, gap=0 * u.radian, pol=None
):
    """Construct a hexagon from 3 rhombi.

    Args:
        rhombus_npos (int):  The number of pixel locations in one rhombus.
        rhombus_width (Quantity):  The angle subtended by pixel center-to-center
            distance along the short dimension of one rhombus.
        prefix (str): the detector name prefix.
        suffix (str): the detector name suffix.
        gap (Quantity):  The *additional* gap between the edges of the rhombi.  By
            default, the rhombi are aligned such that the center spacing across
            the gap is the same as the center spacing within a rhombus.
        pol (array, Quantity): The polarization angle of each position on each
            rhombus before the rhombus is rotated into place.  This can either
            describe the same angles to be applied to each rhombus (if it has
            length rhombus_npos), or a unique set of angles (if it has length
            3 * rhombus_npos).

    Returns:
        (dict):  A dictionary keyed on detector name, with each value itself a
            dictionary of detector properties.  This will include the quaternion
            and gamma angle for each detector.

    """
    # Width of one rhombus in radians
    width_rad = rhombus_width.to_value(u.radian)

    # Dimension of one rhombus
    dim = rhomb_dim(rhombus_npos)

    # Total gap between rhombi in radians.  This is the normal pixel spacing
    # plus the additional gap.
    gap_rad = gap.to_value(u.radian) + (width_rad / (dim - 1))

    # Quaternion offsets of the 3 rhombi
    centers = [
        xieta_to_quat(
            0.25 * np.sqrt(3.0) * width_rad + 0.5 * gap_rad,
            -0.25 * width_rad - 0.5 * gap_rad / np.sqrt(3.0),
            np.pi / 6,
        ),
        xieta_to_quat(
            0.0,
            0.5 * width_rad + gap_rad / np.sqrt(3.0),
            -0.5 * np.pi,
        ),
        xieta_to_quat(
            -0.25 * np.sqrt(3.0) * width_rad - 0.5 * gap_rad,
            -0.25 * width_rad - 0.5 * gap_rad / np.sqrt(3.0),
            5 * np.pi / 6,
        ),
    ]
    rhomb_gamma = [
        np.pi / 6,
        -0.5 * np.pi,
        5 * np.pi / 6,
    ]

    # The polarization rotation for each rhombus
    rhombus_pol = list()
    if pol is None:
        # No polarization rotation
        for irhomb in range(3):
            rhombus_pol.append(u.Quantity(np.zeros(rhombus_npos), u.radian))
    elif len(pol) == rhombus_npos:
        # Replicating polarization for all rhombi
        for irhomb in range(3):
            rhombus_pol.append(pol)
    elif len(pol) == 3 * rhombus_npos:
        for irhomb in range(3):
            rhombus_pol.append(pol[irhomb * rhombus_npos : (irhomb + 1) * rhombus_npos])
    else:
        msg = "Invalid length of pos_rotate argument"
        raise RuntimeError(msg)

    all_pix = dict()
    for irhomb, cent in enumerate(centers):
        props = rhombus_layout(
            rhombus_npos,
            rhombus_width,
            "",
            "",
            rhombus_pol[irhomb],
            center=cent,
            pos_offset=irhomb * rhombus_npos,
        )
        all_pix.update(props)

    # Number of digits for pixel indexing
    ndigit = int(np.log10(3 * rhombus_npos)) + 1
    nameformat = "{{}}{{:0{}d}}{{}}".format(ndigit)

    # Update the detector gamma angles to include the rotation of each rhombus.
    # Also ensure that the pixel naming format is sufficient
    result = dict()
    for pstr, props in all_pix.items():
        ipix = int(pstr)
        irhomb = ipix // rhombus_npos
        pname = nameformat.format(prefix, ipix, suffix)
        _, _, temp_gamma = quat_to_xieta(props["quat"])
        props["gamma"] = temp_gamma + rhomb_gamma[irhomb]
        # Range reduction
        while props["gamma"] >= 2 * np.pi:
            props["gamma"] -= 2 * np.pi
        while props["gamma"] < -2 * np.pi:
            props["gamma"] += 2 * np.pi
        result[pname] = props
    return result


def boresight_layout(npix, prefix, suffix, pol, center=None, pos_offset=0):
    """Construct a focalplane with all pixels at the boresight.

    This is designed to aid in testing.  The array of "pol" angles specify the gamma
    rotation angle clockwise from the Eta axis.  Each output quaternion describes the
    rotation from the focalplane X / Y / Z coordinate frame into the detector frame
    with the Z axis along the line of sight and the X axis along the polarization
    sensitive direction.

    Each pixel is numbered 0...npix-1 and each detector is named by the
    prefix, the pixel number, and the suffix.

    If the "center" argument is specified, then each quaternion is additionally
    multiplied by this in order to shift all positions to be relative to a new
    center.

    Args:
        npix (int): number of pixels at the boresight.
        prefix (str): the detector name prefix.
        suffix (str): the detector name suffix.
        pol (Quantity): 1D array of detector polarization angles.  These are the
            "gamma" angle in the xi / eta / gamma coordinate system.
        center (ndarray): quaternion offset of the center of the layout (or None).
        pos_offset (int): starting index of position numbers.

    Returns:
        (dict) A dictionary keyed on detector name, with each value itself a
            dictionary of detector properties.

    """
    # number of digits for pixel indexing
    ndigit = int(np.log10(npix)) + 1
    nameformat = "{{}}{{:0{}d}}{{}}".format(ndigit)

    # compute positions of all detectors
    dets = {}
    for pix in range(npix):
        dname = nameformat.format(prefix, pix + pos_offset, suffix)
        xi = 0
        eta = 0
        gamma = pol[pix].to_value(u.radian)
        dprops = {}
        if center is None:
            dprops["quat"] = xieta_to_quat(xi, eta, gamma)
            dprops["gamma"] = gamma
        else:
            dprops["quat"] = qa.mult(center, xieta_to_quat(xi, eta, gamma))
            _, _, temp_gamma = quat_to_xieta(dprops["quat"])
            dprops["gamma"] = temp_gamma
        dets[dname] = dprops
    return dets


def fake_hexagon_focalplane(
    n_pix=7,
    width=5.0 * u.degree,
    sample_rate=1.0 * u.Hz,
    epsilon=0.0,
    fwhm=10.0 * u.arcmin,
    bandcenter=150 * u.GHz,
    bandwidth=20 * u.GHz,
    psd_net=0.1 * u.K * np.sqrt(1 * u.second),
    psd_fmin=0.0 * u.Hz,
    psd_alpha=1.0,
    psd_fknee=0.05 * u.Hz,
    fwhm_sigma=0.0 * u.arcmin,
    bandcenter_sigma=0 * u.GHz,
    bandwidth_sigma=0 * u.GHz,
    random_seed=123456,
):
    """Create a simple focalplane model for testing.

    This function creates a basic focalplane with hexagon-packed pixels, each with
    two orthogonal detectors.  It is intended for unit tests, benchmarking, etc where
    a Focalplane is needed but the details are less important.  In addition to nominal
    detector properties, this function adds other simulation-specific parameters to
    the metadata.

    Args:
        n_pix (int):  The number of pixels with hexagonal packing
            (e.g. 1, 7, 19, 37, 61, etc).
        width (Quantity):  The angular width of the focalplane field of view on the sky.
        sample_rate (Quantity):  The sample rate for all detectors.
        epsilon (float):  The cross-polar response for all detectors.
        fwhm (Quantity):  The beam FWHM
        bandcenter (Quantity):  The detector band center.
        bandwidth (Quantity):  The detector band width.
        psd_net (Quantity):  The Noise Equivalent Temperature of each detector.
        psd_fmin (Quantity):  The frequency below which to roll off the 1/f spectrum.
        psd_alpha (float):  The spectral slope.
        psd_fknee (Quantity):  The 1/f knee frequency.
        fwhm_sigma (Quantity):  Draw random detector FWHM values from a normal
            distribution with this width.
        bandcenter_sigma (Quantity):  Draw random bandcenter values from a normal
            distribution with this width.
        bandwidth_sigma (Quantity):  Draw random bandwidth values from a normal
            distribution with this width.
        random_seed (int):  The seed to use for numpy random.

    Returns:
        (Focalplane):  The fake focalplane.

    """
    zaxis = np.array([0.0, 0.0, 1.0])
    center = None

    pol_A = hex_gamma_angles_qu(n_pix, offset=0.0 * u.degree)
    pol_B = hex_gamma_angles_qu(n_pix, offset=90.0 * u.degree)
    props_A = hex_layout(n_pix, width, "D", "A", pol_A, center=center)
    props_B = hex_layout(n_pix, width, "D", "B", pol_B, center=center)

    temp_data = dict(props_A)
    temp_data.update(props_B)

    # Sort by detector name so that detector pairs are together
    det_data = {x: temp_data[x] for x in sorted(temp_data.keys())}

    nrings = hex_nring(n_pix)

    n_det = len(det_data)

    nominal_freq = str(int(bandcenter.to_value(u.GHz)))
    det_names = [f"{x}-{nominal_freq}" for x in det_data.keys()]
    det_gamma = u.Quantity([det_data[x]["gamma"] for x in det_data.keys()], u.radian)

    det_table = QTable(
        [
            Column(name="name", data=det_names),
            Column(name="quat", data=[det_data[x]["quat"] for x in det_data.keys()]),
            Column(name="pol_leakage", length=n_det, unit=None),
            Column(name="psi_pol", length=n_det, unit=u.rad),
            Column(name="gamma", length=n_det, unit=u.rad),
            Column(name="fwhm", length=n_det, unit=u.arcmin),
            Column(name="psd_fmin", length=n_det, unit=u.Hz),
            Column(name="psd_fknee", length=n_det, unit=u.Hz),
            Column(name="psd_alpha", length=n_det, unit=None),
            Column(name="psd_net", length=n_det, unit=(u.K * np.sqrt(1.0 * u.second))),
            Column(name="bandcenter", length=n_det, unit=u.GHz),
            Column(name="bandwidth", length=n_det, unit=u.GHz),
            Column(
                name="pixel", data=[x.rstrip("A").rstrip("B") for x in det_data.keys()]
            ),
        ]
    )

    np.random.seed(random_seed)

    for idet, det in enumerate(det_data.keys()):
        det_table[idet]["pol_leakage"] = epsilon
        # psi_pol is the rotation from the PXX beam frame to the polarization
        # sensitive direction.
        if det.endswith("A"):
            det_table[idet]["psi_pol"] = 0 * u.rad
        else:
            det_table[idet]["psi_pol"] = np.pi / 2 * u.rad
        det_table[idet]["gamma"] = det_gamma[idet]
        det_table[idet]["fwhm"] = fwhm * (
            1 + np.random.randn() * fwhm_sigma.to_value(fwhm.unit)
        )
        det_table[idet]["bandcenter"] = bandcenter * (
            1 + np.random.randn() * bandcenter_sigma.to_value(bandcenter.unit)
        )
        det_table[idet]["bandwidth"] = bandwidth * (
            1 + np.random.randn() * bandwidth_sigma.to_value(bandcenter.unit)
        )
        det_table[idet]["psd_fmin"] = psd_fmin
        det_table[idet]["psd_fknee"] = psd_fknee
        det_table[idet]["psd_alpha"] = psd_alpha
        det_table[idet]["psd_net"] = psd_net

    return Focalplane(
        detector_data=det_table,
        sample_rate=sample_rate,
        field_of_view=1.1 * (width + 2 * fwhm),
    )


def fake_rhombihex_focalplane(
    n_pix_rhombus=4,
    width=5.0 * u.degree,
    gap=0 * u.radian,
    sample_rate=1.0 * u.Hz,
    epsilon=0.0,
    fwhm=10.0 * u.arcmin,
    bandcenter=150 * u.GHz,
    bandwidth=20 * u.GHz,
    psd_net=0.1 * u.K * np.sqrt(1 * u.second),
    psd_fmin=0.0 * u.Hz,
    psd_alpha=1.0,
    psd_fknee=0.05 * u.Hz,
    fwhm_sigma=0.0 * u.arcmin,
    bandcenter_sigma=0 * u.GHz,
    bandwidth_sigma=0 * u.GHz,
    random_seed=123456,
):
    """Create a simple focalplane model for testing.

    This function constructs a hexagonal layout using 3 rhombi.  Each pixel has two
    orthogonal detectors.  It is intended for unit tests, benchmarking, etc where
    a Focalplane is needed but the details are less important.  In addition to nominal
    detector properties, this function adds other simulation-specific parameters to
    the metadata.

    Args:
        n_pix_rhombus (int):  The (square) number of pixels in each of the 3 rhombi.
        width (Quantity):  The angular width of the focalplane field of view on the sky.
        gap (Quantity):  The *additional* gap between the edges of the rhombi.
        sample_rate (Quantity):  The sample rate for all detectors.
        epsilon (float):  The cross-polar response for all detectors.
        fwhm (Quantity):  The beam FWHM
        bandcenter (Quantity):  The detector band center.
        bandwidth (Quantity):  The detector band width.
        psd_net (Quantity):  The Noise Equivalent Temperature of each detector.
        psd_fmin (Quantity):  The frequency below which to roll off the 1/f spectrum.
        psd_alpha (float):  The spectral slope.
        psd_fknee (Quantity):  The 1/f knee frequency.
        fwhm_sigma (Quantity):  Draw random detector FWHM values from a normal
            distribution with this width.
        bandcenter_sigma (Quantity):  Draw random bandcenter values from a normal
            distribution with this width.
        bandwidth_sigma (Quantity):  Draw random bandwidth values from a normal
            distribution with this width.
        random_seed (int):  The seed to use for numpy random.

    Returns:
        (Focalplane):  The fake focalplane.

    """
    xaxis, yaxis, zaxis = np.eye(3)

    # The dimension of one rhombus
    dim = rhomb_dim(n_pix_rhombus)

    # The width of one rhombus, assuming the nominal gap of one pixel width
    rhomb_width = 0.5 * width

    # Polarization orientations within one rhombus
    pol_A = rhomb_gamma_angles_qu(n_pix_rhombus, offset=0.0 * u.degree)
    pol_B = rhomb_gamma_angles_qu(n_pix_rhombus, offset=90.0 * u.degree)

    det_A = rhombus_hex_layout(n_pix_rhombus, rhomb_width, "D", "A", gap=gap, pol=pol_A)
    det_B = rhombus_hex_layout(n_pix_rhombus, rhomb_width, "D", "B", gap=gap, pol=pol_B)
    full_fp = dict(det_A)
    full_fp.update(det_B)

    # Sort by detector name so that detector pairs are together
    det_data = {x: full_fp[x] for x in sorted(full_fp.keys())}

    n_det = len(det_data)

    nominal_freq = str(int(bandcenter.to_value(u.GHz)))
    det_names = [f"{x}-{nominal_freq}" for x in det_data.keys()]
    det_gamma = u.Quantity([det_data[x]["gamma"] for x in det_data.keys()], u.radian)

    det_table = QTable(
        [
            Column(name="name", data=det_names),
            Column(name="quat", data=[det_data[x]["quat"] for x in det_data.keys()]),
            Column(name="pol_leakage", length=n_det, unit=None),
            Column(name="psi_pol", length=n_det, unit=u.rad),
            Column(name="gamma", length=n_det, unit=u.rad),
            Column(name="fwhm", length=n_det, unit=u.arcmin),
            Column(name="psd_fmin", length=n_det, unit=u.Hz),
            Column(name="psd_fknee", length=n_det, unit=u.Hz),
            Column(name="psd_alpha", length=n_det, unit=None),
            Column(name="psd_net", length=n_det, unit=(u.K * np.sqrt(1.0 * u.second))),
            Column(name="bandcenter", length=n_det, unit=u.GHz),
            Column(name="bandwidth", length=n_det, unit=u.GHz),
            Column(
                name="pixel", data=[x.rstrip("A").rstrip("B") for x in det_data.keys()]
            ),
        ]
    )

    np.random.seed(random_seed)

    for idet, det in enumerate(det_data.keys()):
        det_table[idet]["pol_leakage"] = epsilon
        # psi_pol is the rotation from the PXX beam frame to the polarization
        # sensitive direction.
        if det.endswith("A"):
            det_table[idet]["psi_pol"] = 0 * u.rad
        else:
            det_table[idet]["psi_pol"] = np.pi / 2 * u.rad
        det_table[idet]["gamma"] = det_gamma[idet]
        det_table[idet]["fwhm"] = fwhm * (
            1 + np.random.randn() * fwhm_sigma.to_value(fwhm.unit)
        )
        det_table[idet]["bandcenter"] = bandcenter * (
            1 + np.random.randn() * bandcenter_sigma.to_value(bandcenter.unit)
        )
        det_table[idet]["bandwidth"] = bandwidth * (
            1 + np.random.randn() * bandwidth_sigma.to_value(bandcenter.unit)
        )
        det_table[idet]["psd_fmin"] = psd_fmin
        det_table[idet]["psd_fknee"] = psd_fknee
        det_table[idet]["psd_alpha"] = psd_alpha
        det_table[idet]["psd_net"] = psd_net

    return Focalplane(
        detector_data=det_table,
        sample_rate=sample_rate,
        field_of_view=1.1 * (width + 2 * fwhm),
    )


def fake_boresight_focalplane(
    n_pix=1,
    sample_rate=1.0 * u.Hz,
    epsilon=0.0,
    fwhm=10.0 * u.arcmin,
    bandcenter=150 * u.GHz,
    bandwidth=20 * u.GHz,
    psd_net=0.1 * u.K * np.sqrt(1 * u.second),
    psd_fmin=0.0 * u.Hz,
    psd_alpha=1.0,
    psd_fknee=0.05 * u.Hz,
    fwhm_sigma=0.0 * u.arcmin,
    bandcenter_sigma=0 * u.GHz,
    bandwidth_sigma=0 * u.GHz,
    random_seed=123456,
):
    """Create a focalplane with all detectors at the boresight for testing.

    This function creates a basic focalplane with the specified number of pixels, each
    with two orthogonal detectors.  All detectors are placed at the boresight.  It is
    intended for unit tests.  In addition to nominal detector properties, this function
    adds other simulation-specific parameters to the metadata.

    Args:
        n_pix (int):  The number of pixels to place at the boresight.
        sample_rate (Quantity):  The sample rate for all detectors.
        epsilon (float):  The cross-polar response for all detectors.
        fwhm (Quantity):  The beam FWHM
        bandcenter (Quantity):  The detector band center.
        bandwidth (Quantity):  The detector band width.
        psd_net (Quantity):  The Noise Equivalent Temperature of each detector.
        psd_fmin (Quantity):  The frequency below which to roll off the 1/f spectrum.
        psd_alpha (float):  The spectral slope.
        psd_fknee (Quantity):  The 1/f knee frequency.
        fwhm_sigma (Quantity):  Draw random detector FWHM values from a normal
            distribution with this width.
        bandcenter_sigma (Quantity):  Draw random bandcenter values from a normal
            distribution with this width.
        bandwidth_sigma (Quantity):  Draw random bandwidth values from a normal
            distribution with this width.
        random_seed (int):  The seed to use for numpy random.

    Returns:
        (Focalplane):  The fake focalplane.

    """
    center = None
    pol_A = hex_gamma_angles_qu(n_pix, offset=0.0 * u.degree)
    pol_B = hex_gamma_angles_qu(n_pix, offset=90.0 * u.degree)
    det_A = boresight_layout(n_pix, "D", "A", pol_A, center=center)
    det_B = boresight_layout(n_pix, "D", "B", pol_B, center=center)

    temp_data = dict(det_A)
    temp_data.update(det_B)

    # Sort by detector name so that detector pairs are together
    det_data = {x: temp_data[x] for x in sorted(temp_data.keys())}

    n_det = len(det_data)

    nominal_freq = str(int(bandcenter.to_value(u.GHz)))
    det_names = [f"{x}-{nominal_freq}" for x in det_data.keys()]
    det_gamma = u.Quantity([det_data[x]["gamma"] for x in det_data.keys()], u.radian)

    det_table = QTable(
        [
            Column(name="name", data=det_names),
            Column(name="quat", data=[det_data[x]["quat"] for x in det_data.keys()]),
            Column(name="pol_leakage", length=n_det, unit=None),
            Column(name="psi_pol", length=n_det, unit=u.rad),
            Column(name="gamma", length=n_det, unit=u.rad),
            Column(name="fwhm", length=n_det, unit=u.arcmin),
            Column(name="psd_fmin", length=n_det, unit=u.Hz),
            Column(name="psd_fknee", length=n_det, unit=u.Hz),
            Column(name="psd_alpha", length=n_det, unit=None),
            Column(name="psd_net", length=n_det, unit=(u.K * np.sqrt(1.0 * u.second))),
            Column(name="bandcenter", length=n_det, unit=u.GHz),
            Column(name="bandwidth", length=n_det, unit=u.GHz),
            Column(
                name="pixel", data=[x.rstrip("A").rstrip("B") for x in det_data.keys()]
            ),
        ]
    )

    np.random.seed(random_seed)

    for idet, det in enumerate(det_data.keys()):
        det_table[idet]["pol_leakage"] = epsilon
        # psi_pol is the rotation from the PXX beam frame to the polarization
        # sensitive direction.
        if det.endswith("A"):
            det_table[idet]["psi_pol"] = 0 * u.rad
        else:
            det_table[idet]["psi_pol"] = np.pi / 2 * u.rad
        det_table[idet]["gamma"] = det_gamma[idet]
        det_table[idet]["fwhm"] = fwhm * (
            1 + np.random.randn() * fwhm_sigma.to_value(fwhm.unit)
        )
        det_table[idet]["bandcenter"] = bandcenter * (
            1 + np.random.randn() * bandcenter_sigma.to_value(bandcenter.unit)
        )
        det_table[idet]["bandwidth"] = bandwidth * (
            1 + np.random.randn() * bandwidth_sigma.to_value(bandcenter.unit)
        )
        det_table[idet]["psd_fmin"] = psd_fmin
        det_table[idet]["psd_fknee"] = psd_fknee
        det_table[idet]["psd_alpha"] = psd_alpha
        det_table[idet]["psd_net"] = psd_net

    return Focalplane(
        detector_data=det_table,
        sample_rate=sample_rate,
        field_of_view=3 * fwhm,
    )


def plot_focalplane(
    focalplane=None,
    width=None,
    height=None,
    outfile=None,
    show_labels=False,
    face_color=None,
    pol_color=None,
    xieta=False,
    show_centers=False,
    show_gamma=False,
):
    """Visualize a projected Focalplane.

    This makes a simple plot of the detector positions on the projected focalplane.
    By default, this plots the focalplane in the boresight X / Y / Z frame, as seen
    by incoming photons.  If `xieta` is set to True, the focalplane is plotted in
    Xi / Eta / Gamma coordinates as seen from the observer looking out at the sky.

    To avoid python overhead in large MPI jobs, we place the matplotlib import inside
    this function, so that it is only imported when the function is actually called.

    Args:
        focalplane (Focalplane):  The focalplane to plot
        width (Quantity):  Width of plot.
        height (Quantity):  Height of plot.
        outfile (str):  Output PDF path.  If None, then matplotlib will be
            used for inline plotting.
        show_labels (bool):  If True, plot detector names.
        face_color (dict): dictionary of color values for the face of each
            detector circle.
        pol_color (dict): dictionary of color values for the polarization
            arrows.
        xieta (bool):  Plot in observer xi/eta/gamma coordinates rather than
            boresight X/Y/Z.
        show_centers (bool):  If True, label the pixel centers.
        show_gamma (bool):  If True, show gamma angle (for debugging).

    Returns:
        (Figure):  The figure.

    """
    if focalplane is None:
        raise RuntimeError("You must specify a Focalplane instance")

    if outfile is not None:
        set_matplotlib_backend(backend="pdf")

    import matplotlib.pyplot as plt

    if width is None:
        width = 10.0 * u.degree

    if height is None:
        height = 10.0 * u.degree

    width_deg = width.to_value(u.degree)
    height_deg = height.to_value(u.degree)

    xfigsize = int(width_deg) + 1
    yfigsize = int(height_deg) + 1
    figdpi = 100

    # Compute the font size to use for detector labels
    fontpix = 0.05 * figdpi
    fontpt = int(0.75 * fontpix)

    fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    half_width = 0.6 * width_deg
    half_height = 0.6 * height_deg
    if xieta:
        ax.set_xlabel(r"Boresight $\xi$ Degrees", fontsize="medium")
        ax.set_ylabel(r"Boresight $\eta$ Degrees", fontsize="medium")
    else:
        ax.set_xlabel("Boresight X Degrees", fontsize="medium")
        ax.set_ylabel("Boresight Y Degrees", fontsize="medium")
    ax.set_xlim([-half_width, half_width])
    ax.set_ylim([-half_height, half_height])

    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    yaxis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for d in focalplane.detectors:
        quat = focalplane[d]["quat"]
        fwhm = focalplane[d]["fwhm"].to_value(u.arcmin)

        # radius in degrees
        detradius = 0.5 * 5.0 / 60.0
        if fwhm is not None:
            detradius = 0.5 * fwhm / 60.0

        if xieta:
            xi, eta, gamma = quat_to_xieta(quat)
            xpos = xi * 180.0 / np.pi
            ypos = eta * 180.0 / np.pi
            # Polang is plotted relative to visualization x/y coords
            polang = 1.5 * np.pi - gamma
            plot_gamma = polang
        else:
            # rotation from boresight
            rdir = qa.rotate(quat, zaxis).flatten()
            mag = np.arccos(rdir[2]) * 180.0 / np.pi
            ang = np.arctan2(rdir[1], rdir[0])
            orient = qa.rotate(quat, xaxis).flatten()
            polang = np.arctan2(orient[1], orient[0])
            xpos = mag * np.cos(ang)
            ypos = mag * np.sin(ang)
            xi, eta, gamma = quat_to_xieta(quat)
            plot_gamma = gamma

        detface = "none"
        if face_color is not None:
            detface = face_color[d]

        circ = plt.Circle((xpos, ypos), radius=detradius, fc=detface, ec="k")
        ax.add_artist(circ)

        ascale = 1.5

        xtail = xpos - ascale * detradius * np.cos(polang)
        ytail = ypos - ascale * detradius * np.sin(polang)
        dx = ascale * 2.0 * detradius * np.cos(polang)
        dy = ascale * 2.0 * detradius * np.sin(polang)

        detcolor = "black"
        if pol_color is not None:
            detcolor = pol_color[d]

        if show_centers:
            ysgn = -1.0
            if dx < 0.0:
                ysgn = 1.0
            ax.text(
                (xpos + 0.1 * dx),
                (ypos + 0.1 * ysgn * dy),
                f"({xpos:0.4f}, {ypos:0.4f})",
                color="green",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

        if show_labels:
            xsgn = 1.0
            if dx < 0.0:
                xsgn = -1.0
            labeloff = 0.05 * xsgn * fontpix * len(d) / figdpi
            ax.text(
                (xtail + 1.3 * dx + labeloff),
                (ytail + 1.2 * dy),
                d,
                color="k",
                fontsize=fontpt,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(fc="w", ec="none", pad=1, alpha=0.0),
            )

        if show_gamma:
            ax.arrow(
                xtail,
                ytail,
                1.3 * dx,
                1.3 * dy,
                width=0.1 * detradius,
                head_width=0.2 * detradius,
                head_length=0.2 * detradius,
                fc="gray",
                ec="gray",
                length_includes_head=True,
            )
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

    # Draw a "mini" coordinate axes for reference
    xmini = -0.8 * half_width
    ymini = -0.8 * half_height
    xlen = 0.1 * half_width
    ylen = 0.1 * half_height
    mini_width = 0.005 * half_width
    mini_head_width = 3 * mini_width
    mini_head_len = 3 * mini_width
    if xieta:
        aprops = [
            (xlen, 0, "-", r"$\xi$"),
            (0, ylen, "-", r"$\eta$"),
            (-xlen, 0, "--", "Y"),
            (0, -ylen, "--", "X"),
        ]
    else:
        aprops = [
            (xlen, 0, "-", "X"),
            (0, ylen, "-", "Y"),
            (-xlen, 0, "--", r"$\eta$"),
            (0, -ylen, "--", r"$\xi$"),
        ]
    for ap in aprops:
        lx = xmini + 1.5 * ap[0]
        ly = ymini + 1.5 * ap[1]
        lw = figdpi / 200.0
        ax.arrow(
            xmini,
            ymini,
            ap[0],
            ap[1],
            width=mini_width,
            head_width=mini_head_width,
            head_length=mini_head_len,
            fc="k",
            ec="k",
            linestyle=ap[2],
            linewidth=lw,
            length_includes_head=True,
        )
        ax.text(
            lx,
            ly,
            ap[3],
            color="k",
            fontsize=int(figdpi / 10),
            horizontalalignment="center",
            verticalalignment="center",
        )

    st = "Focalplane Looking Towards Observer"
    if xieta:
        st = "Focalplane on Sky From Observer"
    fig.suptitle(st)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=figdpi, bbox_inches="tight", format="pdf")
        plt.close()
    return fig
