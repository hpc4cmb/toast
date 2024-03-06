# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for reading and writing a configuration dictionary to TOML.

The internal config dictionary stores all values as strings.  For
data types supported by the TOML standard, we convert these strings
back to their native values for ease of use and editing.

"""

from collections import OrderedDict

import tomlkit
from astropy import units as u
from tomlkit import comment, document, dumps, loads, nl, table

from ..trait_utils import fix_quotes, string_to_trait, trait_to_string
from ..utils import Environment, Logger
from .utils import merge_config


def _load_toml_element(elem):
    log = Logger.get()
    # See if we are loading one of the TOML supported types
    if isinstance(elem, bool):
        if elem:
            return ("True", "bool")
        else:
            return ("False", "bool")
    if isinstance(elem, int):
        return (trait_to_string(elem), "int")
    if isinstance(elem, float):
        return (trait_to_string(elem), "float")

    # In old TOML files, lists were stored as TOMLKit arrays.
    # Here we cast those cases into strings to be parsed like
    # modern containers (which are stored as strings in the
    # the TOML files).  Remove this code after a suitable
    # deprecation period.
    if isinstance(elem, list):
        msg = f"Storing trait '{list(elem)}' as a TOML array is deprecated. "
        msg += f"All containers should be stored as a string representation."
        msg += f"You can use toast_config_verify to update your config files."
        log.warning(msg)
        elist = "["
        for el in elem:
            if isinstance(el, str):
                el = string_to_trait(el)
                if isinstance(el, str) and len(el) > 0:
                    elist += f"'{trait_to_string(el)}',"
                else:
                    elist += f"{trait_to_string(el)},"
            else:
                elist += f"{trait_to_string(el)},"
        elist += "]"
        elem = elist

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


def _load_toml_traits(tbl):
    """Load traits for a single TraitConfig object from the TOML."""
    # print("LOAD TraitConfig object {}".format(tbl), flush=True)
    result = OrderedDict()
    for k in list(tbl.keys()):
        if k == "class":
            # print(f"  found trait class '{tbl[k]}'")
            result[k] = tbl[k]
        else:
            # This is a trait
            result[k] = OrderedDict()
            result[k]["value"], result[k]["type"] = _load_toml_element(tbl[k])
            if k == "kernel_implementation":
                # In old TOML files this trait (which is the only Enum trait that
                # was used), was written as a string, rather than an integer.  Fix
                # that here.
                result[k]["value"] = str(int(fix_quotes(result[k]["value"])))
                result[k]["type"] = "enum"
        # print(f"  parsed as: {result[k]}")
    # print("LOAD toml result = {}".format(result))
    return result


def load_toml(file, input=None, comm=None):
    """Load a TOML config file.

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
            raw = loads(f.read())
    if comm is not None:
        raw = comm.bcast(raw, root=0)

    # Convert this TOML document into a dictionary

    def convert_node(raw_root, conf_root):
        """Helper function to recursively convert tables"""
        if isinstance(
            raw_root, (tomlkit.toml_document.TOMLDocument, tomlkit.items.Table)
        ):
            for k in list(raw_root.keys()):
                try:
                    subkeys = list(raw_root[k].keys())
                    # This element is table-like.
                    if "class" in subkeys:
                        # print("LOAD found traitconfig {}".format(k), flush=True)
                        conf_root[k] = _load_toml_traits(raw_root[k])
                    else:
                        # This is just a dictionary
                        conf_root[k] = OrderedDict()
                        convert_node(raw_root[k], conf_root[k])
                except Exception as e:
                    # This element is not a sub-table, just copy.
                    conf_root[k] = raw_root[k]
                    raise
        else:
            raise RuntimeError("Cannot convert TOML node {}".format(raw_root))

    raw_config = OrderedDict()
    convert_node(raw, raw_config)

    if input is None:
        return raw_config

    # We need to merge results.
    merge_config(raw_config, input)

    return input


def _dump_toml_element(elem):
    # Convert the config string into a value so that
    # we can encode it properly.
    val = string_to_trait(elem)
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    else:
        # Leave this as the string representation
        return elem


def _dump_toml_trait(tbl, indent, name, value, typ, help):
    # Get the TOML-compatible value
    val = _dump_toml_element(value)
    # print(f"dump[{indent}] {name} ({typ}): '{value}' --> |{val}|", flush=True)
    if typ == "bool":
        # Bools seem to have an issue adding comments.  To workaround, we
        # add the value as a string, add the comment, and then set it to
        # a real bool.
        tbl.add(name, "temp")
        if help is not None:
            tbl[name].comment(help)
        tbl[name].indent(indent)
        if val == "None":
            tbl[name] = "None"
        elif val:
            tbl[name] = True
        else:
            tbl[name] = False
    else:
        tbl.add(name, val)
        if help is not None:
            tbl[name].comment(help)
        tbl[name].indent(indent)


def dump_toml(file, conf, comm=None):
    """Dump a configuration to a TOML file.

    This writes a config dictionary to a TOML file.

    Args:
        file (str):  The file to write.
        conf (dict):  The configuration to dump.
        comm (MPI.Comm):  Optional communicator to control which process writes.

    Returns:
        None

    """

    def convert_node(conf_root, table_root, indent_size):
        """Helper function to recursively convert dictionaries to tables"""
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
                        _dump_toml_trait(
                            table_root,
                            indent_size,
                            k,
                            conf_root[k]["value"],
                            conf_root[k]["type"],
                            help,
                        )
                    else:
                        table_root[k] = table()
                        convert_node(conf_root[k], table_root[k], indent_size + 2)
                else:
                    table_root.add(k, conf_root[k])
                    table_root[k].indent(indent_size)
        else:
            raise RuntimeError("Cannot convert config node {}".format(conf_root))

    if comm is None or comm.rank == 0:
        env = Environment.get()
        doc = document()
        doc.add(comment("TOAST config"))
        doc.add(comment("Generated with version {}".format(env.version())))

        # Convert all top-level sections from the config dictionary into a TOML table.
        convert_node(conf, doc, 0)

        with open(file, "w") as f:
            f.write(dumps(doc))
