# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for reading and writing a configuration dictionary to JSON.

The internal config dictionary stores all values as strings.  For
JSON format config documents we serialize the config dictionary
directly and keep all values as strings.  This makes it a trivial
format to dump, load, send, and archive.

"""

import json
from collections import OrderedDict

from ..utils import Environment
from .utils import merge_config


def _load_json_traits(tbl):
    """Load traits for a single TraitConfig object."""
    result = OrderedDict()
    for k in list(tbl.keys()):
        if k == "class":
            result[k] = tbl[k]
        else:
            # This is a trait
            result[k] = OrderedDict()
            result[k]["value"] = tbl[k]["value"]
            result[k]["type"] = tbl[k]["type"]
    return result


def load_json(file, input=None, comm=None):
    """Load a JSON config file.

    This loads the document into a config dictionary.  If input is specified, the file
    contents are merged into this dictionary.

    Args:
        file (str):  The file to load.
        input (dict):  Append to this dictionary.
        comm (MPI.Comm):  Optional communicator to broadcast across.

    Returns:
        (dict):  The result.

    """
    raw = None
    if comm is None or comm.rank == 0:
        with open(file, "r") as f:
            raw = json.load(f, object_pairs_hook=OrderedDict)
    if comm is not None:
        raw = comm.bcast(raw, root=0)

    if input is None:
        return raw

    # We need to merge results.
    merge_config(raw, input)

    return input


def dump_json(file, conf, comm=None):
    """Dump a configuration to a JSON file.

    This writes a config dictionary to a JSON file.

    Args:
        file (str):  The file to write.
        conf (dict):  The configuration to dump.
        comm (MPI.Comm):  Optional communicator to control which process writes.

    Returns:
        None

    """
    if comm is None or comm.rank == 0:
        env = Environment.get()
        versioned = OrderedDict()
        versioned["version"] = env.version()
        versioned.update(conf)

        with open(file, "w") as f:
            json.dump(versioned, f, indent=2)
