# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


available = True
try:
    import tidas as tds
    from tidas.mpi import MPIVolume
except ImportError:
    available = False


def to_dict(td):
    meta = dict()
    for k in td.keys():
        tp = td.get_type(k)
        if tp == "d":
            meta[k] = td.get_float64(k)
        elif tp == "f":
            meta[k] = td.get_float32(k)
        elif tp == "l":
            meta[k] = td.get_int64(k)
        elif tp == "L":
            meta[k] = td.get_uint64(k)
        elif tp == "i":
            meta[k] = td.get_int32(k)
        elif tp == "I":
            meta[k] = td.get_uint32(k)
        elif tp == "h":
            meta[k] = td.get_int16(k)
        elif tp == "H":
            meta[k] = td.get_uint16(k)
        elif tp == "b":
            meta[k] = td.get_int8(k)
        elif tp == "B":
            meta[k] = td.get_uint8(k)
        else:
            meta[k] = td.get_string(k)
    return meta


def from_dict(meta):
    td = tds.Dictionary()
    for k, v in meta.items():
        if isinstance(v, float):
            td.put_float64(k, v)
        elif isinstance(v, int):
            td.put_int64(k, v)
        else:
            td.put_string(k, str(v))
    return td


def find_obs(vol, parent, name):
    """Return the TIDAS block representing an observation.

    If the block exists, a handle is returned.  Otherwise it is created.

    Args:
        vol (tidas.MPIVolume):  the volume.
        parent (str):  the path to the parent block.
        name (str):  the name of the observation block.

    Returns:
        (tidas.Block):  the block handle

    """
    # The root block
    root = vol.root()

    par = root
    if parent != "":
        # Descend tree to the parent node
        parentnodes = parent.split("/")
        for pn in parentnodes:
            if pn != "":
                par = par.block_get(pn)
    obs = None
    if name in par.block_names():
        obs = par.block_get(name)
    else:
        # Create the observation block
        obs = par.block_add(name, tds.Block())
    del par
    del root
    return obs
