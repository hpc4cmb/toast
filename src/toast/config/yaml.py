# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for reading and writing a configuration dictionary to YAML.

The internal config dictionary stores all values as strings.  For
data types supported by the YAML standard, we convert these strings
back to their native values for ease of use and editing.

"""

from collections import OrderedDict

from astropy import units as u
from ruamel.yaml import YAML, CommentedMap

from ..trait_utils import string_to_trait, trait_to_string
from ..utils import Environment, Logger
from .utils import merge_config

yaml = YAML()


def _dump_yaml_element(elem):
    # Convert the config string into a value so that
    # we can encode it properly.
    val = string_to_trait(elem)
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    else:
        # Leave this as the string representation
        return elem


def _dump_yaml_trait(cm, name, value, typ, help):
    # Get the YAML-compatible value and insert into the commented map.
    val = _dump_yaml_element(value)
    # print(f"YAML dump {name}: {val} ({typ}) # {help}", flush=True)
    cm[name] = val
    if help is None or help == "None" or help == "":
        cm.yaml_add_eol_comment("(help string missing)", key=name)
    else:
        cm.yaml_add_eol_comment(help, key=name)


def dump_yaml(file, conf, comm=None):
    """Dump a configuration to a YAML file.

    This writes a config dictionary to a YAML file.

    Args:
        file (str):  The file to write.
        conf (dict):  The configuration to dump.
        comm (MPI.Comm):  Optional communicator to control which process writes.

    Returns:
        None

    """

    def convert_node(conf_root, cm_root):
        """Helper function to recursively convert dictionaries."""
        if isinstance(conf_root, (dict, OrderedDict)):
            # print("{}found dict".format(" " * indent_size))
            for k in list(conf_root.keys()):
                # print("{}  examine key {}".format(" " * indent_size, k))
                if isinstance(conf_root[k], (dict, OrderedDict)):
                    # print("{}  key is a dict".format(" " * indent_size))
                    if "value" in conf_root[k] and "type" in conf_root[k]:
                        # this is a trait
                        help = None
                        if "help" in conf_root[k]:
                            help = conf_root[k]["help"]
                        _dump_yaml_trait(
                            cm_root,
                            k,
                            conf_root[k]["value"],
                            conf_root[k]["type"],
                            help,
                        )
                    else:
                        cm_root[k] = CommentedMap()
                        convert_node(conf_root[k], cm_root[k])
                else:
                    cm_root[k] = conf_root[k]
        else:
            raise RuntimeError(f"Cannot convert config node {conf_root}")

    if comm is None or comm.rank == 0:
        env = Environment.get()
        doc = CommentedMap()
        doc.yaml_set_start_comment(
            f"TOAST config generated with version {env.version()}"
        )
        convert_node(conf, doc)
        # print(f"YAML dump final = {doc}", flush=True)
        with open(file, "w") as f:
            yaml.dump(doc, f)


def _load_yaml_element(elem):
    # See if we are loading one of the YAML supported scalar types
    if isinstance(elem, bool):
        if elem:
            return ("True", "bool")
        else:
            return ("False", "bool")
    if isinstance(elem, int):
        return (trait_to_string(elem), "int")
    if isinstance(elem, float):
        return (trait_to_string(elem), "float")
    if isinstance(elem, list):
        return (trait_to_string(elem), "list")

    # This is a string, which might represent a quantity or
    # some complicated container.  Convert to a value that
    # we can use to determine the type.
    val = string_to_trait(elem)
    if val is None:
        return ("None", "unknown")
    if isinstance(val, u.UnitBase):
        return (trait_to_string(val), "Unit")
    if isinstance(val, u.Quantity):
        return (trait_to_string(val), "Quantity")
    if isinstance(val, set):
        return (trait_to_string(val), "set")
    if isinstance(val, list):
        return (trait_to_string(val), "list")
    if isinstance(val, tuple):
        return (trait_to_string(val), "tuple")
    if isinstance(val, dict):
        return (trait_to_string(val), "dict")

    # This is just a string
    return (elem, "str")


def _load_yaml_traits(tbl):
    """Load traits for a single TraitConfig object."""
    result = OrderedDict()
    for k in list(tbl.keys()):
        if k == "class":
            # print(f"  found trait class '{tbl[k]}'")
            result[k] = tbl[k]
        else:
            # This is a trait
            result[k] = OrderedDict()
            result[k]["value"], result[k]["type"] = _load_yaml_element(tbl[k])
    return result


def load_yaml(file, input=None, comm=None):
    """Load a YAML config file.

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
            raw = yaml.load(f)
    if comm is not None:
        raw = comm.bcast(raw, root=0)

    # Parse the doc into a config dictionary
    def convert_node(raw_root, conf_root):
        """Helper function to recursively convert tables"""
        if isinstance(raw_root, dict):
            for k in list(raw_root.keys()):
                try:
                    subkeys = list(raw_root[k].keys())
                    # This element is table-like.
                    if "class" in subkeys:
                        conf_root[k] = _load_yaml_traits(raw_root[k])
                    else:
                        # This is just a dictionary
                        conf_root[k] = OrderedDict()
                        convert_node(raw_root[k], conf_root[k])
                except Exception as e:
                    # This element is not a sub-table, just copy.
                    conf_root[k] = raw_root[k]
                    raise
        else:
            raise RuntimeError("Cannot convert YAML node {}".format(raw_root))

    raw_config = OrderedDict()
    convert_node(raw, raw_config)

    if input is None:
        return raw_config

    # We need to merge results.
    merge_config(raw_config, input)

    return input
