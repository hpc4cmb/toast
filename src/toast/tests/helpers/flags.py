# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for creating fake flags."""

import numpy as np

from ...observation import default_values as defaults


def fake_flags(
    data,
    shared_name=defaults.shared_flags,
    shared_val=defaults.shared_mask_invalid,
    det_name=defaults.det_flags,
    det_val=defaults.det_mask_invalid,
    do_half=True,
    do_random=False,
):
    """Create fake flags.

    This will create shared flags at regular intervals and will optionally
    create both random detector flags as well as flagging the first half of
    each observation.

    Args:
        data (Data):  The data to process.
        shared_name (str):  The shared flag field.
        shared_val (int):  The shared flag value to set.
        det_name (str):  The det flag field.
        det_val (int):  The detector flag value to set.
        do_half (bool):  If True, flag the first half of each observation.
        do_random (bool):  If True, add random flagged samples.

    Returns:
        (None)

    """
    for ob in data.obs:
        ob.detdata.ensure(det_name, sample_shape=(), dtype=np.uint8)
        if shared_name not in ob.shared:
            ob.shared.create_column(
                shared_name,
                shape=(ob.n_local_samples,),
                dtype=np.uint8,
            )
        half = ob.n_local_samples // 2
        fshared = None
        if ob.comm_col_rank == 0:
            fshared = np.zeros(ob.n_local_samples, dtype=np.uint8)
            fshared[::100] = shared_val
            fshared |= ob.shared[shared_name].data
        ob.shared[shared_name].set(fshared, offset=(0,), fromrank=0)

        if do_half:
            for det in ob.local_detectors:
                ob.detdata[det_name][det, :half] |= det_val
        if do_random:
            for det in ob.local_detectors:
                starts = np.unique(
                    np.sort(
                        np.random.randint(
                            low=0,
                            high=ob.n_local_samples,
                            size=int(ob.n_local_samples // 50),
                        )
                    )
                )
                sizes = np.random.randint(low=1, high=10, size=len(starts))
                for strt, sz in zip(starts, sizes):
                    slc = slice(strt, strt + sz, 1)
                    ob.detdata[det_name][det, slc] |= det_val
