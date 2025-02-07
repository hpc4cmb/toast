# Copyright (c) 2025-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import os
import hashlib
import glob

from .utils import Logger


def observation_hash(obs):
    """Create a unique hash of an observation.
    
    The goal of this function is to ensure that if two observations are equal then
    they have the same unique hash, and if they differ then they have different
    hashes.

    Args:
        obs (Observation):  The observation to hash

    Returns:
        (str):  The unique hash string.
    
    """
    # FIXME:  Currently this function uses only information available on all
    # processes.  Instead, we should gather statistics per-detector across
    # processes and include that in the hashing.

    dhash = hashlib.md5()

    bytes = obs.name.encode("utf-8")
    dhash.update(bytes)

    bytes = str(obs.uid).encode("utf-8")
    dhash.update(bytes)

    bytes = obs.telescope.name.encode("utf-8")
    dhash.update(bytes)

    bytes = str(obs.session).encode("utf-8")
    dhash.update(bytes)

    bytes = str(obs.all_detectors).encode("utf-8")
    dhash.update(bytes)

    hex = dhash.hexdigest()

    return hex


def observation_cache(root_dir, obs, duplicates=None):
    """Get the cache directory for an observation.

    This builds the path to an observation cache directory, and also checks if the
    same observation exists with a different hash in the same root directory.  If
    `duplicates` is "warn", then a warning will be printed if any other observation
    cache directories exist for the same observation.  If `duplicates` is "fail",
    then the existance of duplicates with raise an exception.

    Args:
        root_dir (str):  The top level cache directory.
        obs (Observation):  The observation.
        duplicates (str):  The action to take for duplicates.

    Returns:
        (str):  The observation cache directory.

    """
    log = Logger.get()
    hsh = observation_hash(obs)

    obs_dir = f"{obs.name}_{hsh}"
    cache_dir = os.path.join(root_dir, obs_dir)

    if duplicates is not None:
        # We care about duplicate observations.  Check for those now.
        check = None
        if obs.comm.group_rank == 0:
            check_str = os.path.join(root_dir, f"{obs.name}_*")
            check = glob.glob(check_str)
        if obs.comm.comm_group is not None:
            check = obs.comm.comm_group.bcast(check, root=0)
        if len(check) != 0:
            # There are existing observation directories...
            check_dirs = ", ".join([f"'{os.path.basename(x)}'" for x in check])
            msg = f"{obs_dir}:  found existing cache dirs {check_dirs}"
            if duplicates == "warn":
                log.warning_rank(msg, comm=obs.comm.comm_group)
            elif duplicates == "fail":
                log.error_rank(msg, comm=obs.comm.comm_group)
                raise RuntimeError(msg)
            else:
                log.debug_rank(msg, comm=obs.comm.comm_group)
    
    return cache_dir
    
