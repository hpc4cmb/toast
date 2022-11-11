
import os
import sys
import time

sys.path.append(os.getcwd())
import pyomptarget

# Import this **AFTER** the custom module above, since both will
# dlopen libgomp.
import numpy as np


def build_interval_dtype():
    dtdbl = np.dtype("double")
    dtll = np.dtype("longlong")
    fmts = [dtdbl.char, dtdbl.char, dtll.char, dtll.char]
    offs = [
        0,
        dtdbl.alignment,
        2 * dtdbl.alignment,
        2 * dtdbl.alignment + dtll.alignment,
    ]
    return np.dtype(
        {
            "names": ["start", "stop", "first", "last"],
            "formats": fmts,
            "offsets": offs,
        }
    )


def main():
    interval_dtype = build_interval_dtype()
    ndet = 2
    nsamp = 1000000

    focalplane = np.tile(
        np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        ndet,
    ).reshape((-1, 4))

    boresight = np.tile(
        np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        nsamp,
    ).reshape((-1, 4))

    quat_index = np.arange(ndet, dtype=np.int32)
    pixel_index = np.arange(ndet, dtype=np.int32)

    shared_flags = np.zeros(nsamp, dtype=np.uint8)

    nview = 1
    print(interval_dtype)
    intervals = np.zeros(nview, dtype=interval_dtype).view(np.recarray)
    intervals[0].first = 0
    intervals[0].last = nsamp - 1
    intervals[0].start = -1.0
    intervals[0].stop = -1.0

    quats = np.zeros((ndet, nsamp, 4), dtype=np.float64)
    pixels = np.zeros((ndet, nsamp), dtype=np.int64)

    mem = pyomptarget.stage_data(boresight, quats, intervals, shared_flags, pixels)

    # time.sleep(5)

    pyomptarget.pointing_detector(
        mem,
        focalplane,
        boresight,
        quat_index,
        quats,
        intervals,
        shared_flags,
    )

    nside = 1024
    nside_submap = 16
    n_pix_submap = 12 * nside_submap**2
    n_submap = (nside // nside_submap) ** 2
    hit_submaps = np.zeros(n_submap, dtype=np.uint8)

    pyomptarget.pixels_healpix_nest(
        mem,
        quat_index,
        quats,
        shared_flags,
        pixel_index,
        pixels,
        intervals,
        hit_submaps,
        n_pix_submap,
        nside,
    )

    pyomptarget.unstage_data(mem, quats, pixels)

    print(quats)
    print(pixels)


if __name__ == "__main__":
    main()
