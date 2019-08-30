#!/usr/bin/env python3

# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Test the ability to free memory from a toast.Cache.

This stores the following objects per detector in a Cache:

- Detector signal as float64
- Detector flags as uint8
- Detector pointing pixel numbers as int64
- Detector pointing weights as float32

It reports the memory available before and after this allocation.
Then it frees the buffers of a given type from all detectors and
compares the resulting change to what is expected.

"""
import os
import re
import sys
import argparse
import traceback
import psutil

import numpy as np

from toast.utils import Logger

from toast.cache import Cache


def main():
    log = Logger.get()

    parser = argparse.ArgumentParser(description="Allocate and free cache objects.")

    parser.add_argument(
        "--ndet", required=False, type=int, default=10, help="The number of detectors"
    )

    parser.add_argument(
        "--nobs", required=False, type=int, default=2, help="The number of observations"
    )

    parser.add_argument(
        "--obsminutes",
        required=False,
        type=int,
        default=60,
        help="The number of minutes in each observation.",
    )

    parser.add_argument(
        "--rate", required=False, type=float, default=37.0, help="The sample rate."
    )

    parser.add_argument(
        "--nloop",
        required=False,
        type=int,
        default=2,
        help="The number of allocate / free loops",
    )

    args = parser.parse_args()

    log.info("Input parameters:")
    log.info("  {} observations".format(args.nobs))
    log.info("  {} minutes per obs".format(args.obsminutes))
    log.info("  {} detectors per obs".format(args.ndet))
    log.info("  {}Hz sample rate".format(args.rate))

    nsampobs = int(args.obsminutes * 60 * args.rate)

    nsamptot = args.ndet * args.nobs * nsampobs

    log.info("{} total samples across all detectors and observations".format(nsamptot))

    bytes_sigobs = nsampobs * 8
    bytes_sigtot = nsamptot * 8
    bytes_flagobs = nsampobs * 1
    bytes_flagtot = nsamptot * 1
    bytes_pixobs = nsampobs * 8
    bytes_pixtot = nsamptot * 8
    bytes_wtobs = 3 * nsampobs * 4
    bytes_wttot = 3 * nsamptot * 4

    bytes_tot = bytes_sigtot + bytes_flagtot + bytes_pixtot + bytes_wttot
    bytes_tot_mb = bytes_tot / 2 ** 20
    log.info(
        "{} total bytes ({:0.2f}MB) of data expected".format(bytes_tot, bytes_tot_mb)
    )

    for lp in range(args.nloop):
        log.info("Allocation loop {:02d}".format(lp))
        vmem = psutil.virtual_memory()._asdict()
        avstart = vmem["available"]
        avstart_mb = avstart / 2 ** 20
        log.info("  Starting with {:0.2f}MB of available memory".format(avstart_mb))

        # The list of Caches, one per "observation"
        caches = list()

        # This structure holds external references to cache objects, to ensure that we
        # can destroy objects and free memory, even if external references are held.
        refs = list()

        for ob in range(args.nobs):
            ch = Cache()
            rf = dict()
            for det in range(args.ndet):
                dname = "{:04d}".format(det)
                cname = "{}_sig".format(dname)
                rf[cname] = ch.create(cname, np.float64, (nsampobs,))
                cname = "{}_flg".format(dname)
                rf[cname] = ch.create(cname, np.uint8, (nsampobs,))
                cname = "{}_pix".format(dname)
                rf[cname] = ch.create(cname, np.int64, (nsampobs,))
                cname = "{}_wgt".format(dname)
                rf[cname] = ch.create(cname, np.float32, (nsampobs, 3))
            caches.append(ch)
            refs.append(rf)

        vmem = psutil.virtual_memory()._asdict()
        avpost = vmem["available"]
        avpost_mb = avpost / 2 ** 20
        log.info("  After allocation, {:0.2f}MB of available memory".format(avpost_mb))

        diff = avstart_mb - avpost_mb
        diffperc = 100.0 * np.absolute(diff - bytes_tot_mb) / bytes_tot_mb
        log.info(
            "  Difference is {:0.2f}MB, expected {:0.2f}MB ({:0.2f}% residual)".format(
                diff, bytes_tot_mb, diffperc
            )
        )

        for suf in ["wgt", "pix", "flg", "sig"]:
            for ob, ch in zip(range(args.nobs), caches):
                for det in range(args.ndet):
                    dname = "{:04d}".format(det)
                    ch.destroy("{}_{}".format(dname, suf))

        vmem = psutil.virtual_memory()._asdict()
        avfinal = vmem["available"]
        avfinal_mb = avfinal / 2 ** 20
        log.info(
            "  After destruction, {:0.2f}MB of available memory".format(avfinal_mb)
        )

        diff = avfinal_mb - avpost_mb
        diffperc = 100.0 * np.absolute(diff - bytes_tot_mb) / bytes_tot_mb
        log.info(
            "  Difference is {:0.2f}MB, expected {:0.2f}MB ({:0.2f}% residual)".format(
                diff, bytes_tot_mb, diffperc
            )
        )

    return


if __name__ == "__main__":
    try:
        main()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print("".join(lines), flush=True)
