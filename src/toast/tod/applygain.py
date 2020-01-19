# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from astropy.io import fits

from ..op import Operator

from ..timing import function_timer

from .tod_math import calibrate

from ..utils import Logger


def write_calibration_file(filename, gain):
    """Write gains to a FITS file in the standard TOAST format

    Args:
        filename (string): output filename, overwritten by default
        gain (dict):  Dictionary, key "TIME" has the common timestamps,
            other keys are channel names their values are the gains

    """
    log = Logger.get()
    detectors = list(gain.keys())
    detectors.remove("TIME")

    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="DETECTORS",
                    array=detectors,
                    unit="",
                    format="{0}A".format(max([len(x) for x in detectors])),
                )
            ]
        ),
    ]
    hdus[1].header["EXTNAME"] = "DETECTORS"

    cur_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="TIME", array=gain["TIME"], unit="s", format="1D")]
    )
    cur_hdu.header["EXTNAME"] = "TIME"
    hdus.append(cur_hdu)

    gain_table = np.zeros(
        (len(detectors), len(gain["TIME"])), dtype=gain[detectors[0]].dtype
    )
    for i_det, det in enumerate(detectors):
        gain_table[i_det, :] = gain[det]

    gainhdu = fits.ImageHDU(gain_table)
    gainhdu.header["EXTNAME"] = "GAINS"
    hdus.append(gainhdu)

    fits.HDUList(hdus).writeto(filename, overwrite=True)

    log.info("Gains written to file {}".format(filename))
    return


class OpApplyGain(Operator):
    """Operator which applies gains to timelines.

    Args:
        gain (dict):  Dictionary, key "TIME" has the common timestamps,
            other keys are channel names their values are the gains
        name (str):  Name of the output signal cache object will be
            <name_in>_<detector>.  If the object exists, it is used as
            input.  Otherwise signal is read using the tod read method.

    """

    def __init__(self, gain, name=None):
        self._gain = gain
        self._name = name
        # Call the parent class constructor
        super().__init__()

    @function_timer
    def exec(self, data):
        """Apply the gains.

        Args:
            data (toast.Data): The distributed data.

        """
        for obs in data.obs:

            tod = obs["tod"]

            for det in tod.local_dets:

                # Cache the output signal
                ref = tod.local_signal(det, self._name)
                obs_times = tod.read_times()

                calibrate(
                    obs_times,
                    ref,
                    self._gain["TIME"],
                    self._gain[det],
                    order=0,
                    inplace=True,
                )

                assert np.isnan(ref).sum() == 0, "The signal timestream includes NaN"

                del ref

        return
