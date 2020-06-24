# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re

import copy

from collections import OrderedDict
from collections.abc import MutableMapping

import json

import numpy as np

import tomlkit
from tomlkit import comment, document, nl, table, loads, dumps

from astropy import units as u

from .utils import Environment, Logger

from .instrument import Focalplane, Telescope
from . import instrument

from .operator import Operator

from .traits import TraitConfig, build_config, add_config_args, args_update_config

from . import future_ops as ops


def parse_config(parser, operators=list()):
    """Load command line arguments associated with object properties.

    This function:

        * Adds a "--config" option to the parser which accepts multiple config file
          paths to load.

        * Adds arguments for all object parameters in the defaults for the specified
          classes.

        * Builds a config dictionary starting from the defaults, updating these using
          values from any config files, and then applies any overrides from the
          commandline.

    Args:
        parser (ArgumentParser):  The argparse parser.
        operators (list):  The operator classes to add to the commandline.  Note that
            if these are classes, then the commandline names will be the class names.
            If you pass a list of instances with the name attribute set, then the
            commandline names will use these.

    Returns:
        (dict):  The config dictionary.

    """

    # The default configuration
    defaults = build_config(operators)

    # Add commandline overrides for operators
    add_config_args(parser, defaults, "operators", ignore=["API"])

    # Add an option to load one or more config files.  These should have compatible
    # names for the operators used in defaults.
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        nargs="+",
        help="One or more input config files.",
    )

    # Parse commandline.
    args = parser.parse_args()

    # Load any config files.  This overrides default values with config file contents.
    config = copy.deepcopy(defaults)
    if args.config is not None:
        for conf in args.config:
            config = load_config(conf, input=config)

    # Parse operator commandline options.  These override any config file or default
    # values.
    remaining = args_update_config(args, config, defaults, "operators")

    # Remove the "config" option we created in this function
    del remaining.config

    return config, remaining


def _merge_config(loaded, original):
    for section, objs in loaded.items():
        if section in original.keys():
            # We have this section
            for objname, objprops in objs.items():
                if objname not in original[section]:
                    # This is a new object
                    original[section][objname] = objprops
                else:
                    for k, v in objprops.items():
                        original[section][objname][k] = v
        else:
            # A new section
            original[section] = objs


def _load_toml_trait(tbl):
    result = OrderedDict()
    for k in tbl.keys():
        if k == "class":
            result[k] = tbl[k]
        elif isinstance(tbl[k], str):
            if tbl[k] == "None":
                # Copy None values.  There is no way to determine the type in this case.
                # This is just a feature of how we are constructing the TOML files for
                # ease of use, rather than dumping the full trait info as sub-tables.
                # In practice these parameters will be ignored when constructing
                # TraitConfig objects and the defaults will be used anyway.
                result[k] = OrderedDict()
                result[k]["value"] = tbl[k]
                result[k]["type"] = "unknown"
                result[k]["unit"] = "None"
            else:
                result[k] = OrderedDict()
                # Is this string actually a Quantity?  We try to convert the first
                # element of the string to a float and the remainder to a Unit.
                parts = tbl[k].split()
                vstr = parts.pop(0)
                ustr = " ".join(parts)
                try:
                    v = float(vstr)
                    unit = u.Unit(ustr)
                    result[k]["value"] = "{:0.14e}".format(v)
                    result[k]["type"] = "Quantity"
                    result[k]["unit"] = str(unit)
                except Exception:
                    # Just a regular string
                    result[k]["value"] = tbl[k]
                    result[k]["type"] = "str"
                    result[k]["unit"] = "None"
        elif isinstance(tbl[k], bool):
            result[k] = OrderedDict()
            if tbl[k]:
                result[k]["value"] = "True"
            else:
                result[k]["value"] = "False"
            result[k]["type"] = "bool"
            result[k]["unit"] = "None"
        elif isinstance(tbl[k], int):
            result[k] = OrderedDict()
            result[k]["value"] = "{}".format(tbl[k])
            result[k]["type"] = "int"
            result[k]["unit"] = "None"
        elif isinstance(tbl[k], float):
            result[k] = OrderedDict()
            result[k]["value"] = "{:0.14e}".format(tbl[k])
            result[k]["type"] = "float"
            result[k]["unit"] = "None"
    return result


def load_toml(file, input=None):
    """Load a TOML config file.

    This loads the document into a config dictionary.  If input is specified, the file
    contents are merged into this dictionary.

    Args:
        file (str):  The file to load.
        input (dict):  Append to this dictionary.

    Returns:
        (dict):  The result.

    """
    raw = None
    with open(file, "r") as f:
        raw = loads(f.read())

    # Convert this TOML document into a dictionary

    def convert_node(raw_root, conf_root):
        """Helper function to recursively convert tables"""
        if isinstance(
            raw_root, (tomlkit.toml_document.TOMLDocument, tomlkit.items.Table)
        ):
            for k in raw_root.keys():
                try:
                    subkeys = raw_root[k].keys()
                    # This element is table-like.
                    if "class" in subkeys:
                        conf_root[k] = _load_toml_trait(raw_root[k])
                    else:
                        # This is just a dictionary
                        conf_root[k] = OrderedDict()
                        convert_node(raw_root[k], conf_root[k])
                except:
                    # This element is not a sub-table, just copy.
                    conf_root[k] = raw_root[k]
        else:
            raise RuntimeError("Cannot convert TOML node {}".format(raw_root))

    raw_config = OrderedDict()
    convert_node(raw, raw_config)

    if input is None:
        return raw_config

    # We need to merge results.
    _merge_config(raw_config, input)

    return input


def _dump_toml_trait(tbl, indent, name, value, unit, typ, help):
    if typ == "bool":
        # Bools seem to have an issue adding comments.  To workaround, we
        # add the value as a string, add the comment, and then set it to
        # a real bool.
        tbl.add(name, "temp")
        if help is not None:
            tbl[name].comment(help)
        tbl[name].indent(indent)
        if value == "None":
            tbl[name] = "None"
        elif value == "True":
            tbl[name] = True
        else:
            tbl[name] = False
    else:
        if typ == "Quantity":
            qval = "None"
            if value != "None":
                qval = "{} {}".format(value, unit)
            tbl.add(name, qval)
        elif typ in ["list", "set", "tuple"]:
            val = "None"
            if value != "None":
                val = list(value)
            tbl.add(name, val)
        elif typ == "dict":
            val = "None"
            if value != "None":
                val = table()
                subindent = indent_size + 2
                for k, v in value.items():
                    val.add(k, v)
                    val[k].indent(subindent)
            tbl.add(name, val)
        elif typ == "int":
            val = "None"
            if value != "None":
                val = int(value)
            tbl.add(name, val)
        elif typ == "float":
            val = "None"
            if value != "None":
                val = float(value)
            tbl.add(name, val)
        else:
            # Just leave the string representation.
            tbl.add(name, value)
        if help is not None:
            tbl[name].comment(help)
        tbl[name].indent(indent)


def dump_toml(file, conf):
    """Dump a configuration to a TOML file.

    This writes a config dictionary to a TOML file.

    Args:
        file (str):  The file to write.
        conf (dict):  The configuration to dump.

    Returns:
        None

    """
    env = Environment.get()
    doc = document()

    doc.add(comment("TOAST config"))
    doc.add(comment("Generated with version {}".format(env.version())))

    def convert_node(conf_root, table_root, indent_size):
        """Helper function to recursively convert dictionaries to tables"""
        if isinstance(conf_root, (dict, OrderedDict)):
            for k in list(conf_root.keys()):
                if isinstance(conf_root[k], (dict, OrderedDict)):
                    if "value" in conf_root[k] and "type" in conf_root[k]:
                        # this is a trait
                        unit = None
                        if "unit" in conf_root[k]:
                            unit = conf_root[k]["unit"]
                        help = None
                        if "help" in conf_root[k]:
                            help = conf_root[k]["help"]
                        _dump_toml_trait(
                            table_root,
                            indent_size,
                            k,
                            conf_root[k]["value"],
                            unit,
                            conf_root[k]["type"],
                            help,
                        )
                    else:
                        # descend tree
                        table_root[k] = table()
                        convert_node(conf_root[k], table_root[k], indent_size + 2)
                else:
                    table_root.add(k, conf_root[k])
                    table_root[k].indent(indent_size)
        else:
            raise RuntimeError("Cannot convert config node {}".format(conf_root))

    # Convert all top-level sections from the config dictionary into a TOML table.
    convert_node(conf, doc, 0)

    with open(file, "w") as f:
        f.write(dumps(doc))


def load_json(file, input=None):
    """Load a JSON config file.

    This loads the document into a config dictionary.  If input is specified, the file
    contents are merged into this dictionary.

    Args:
        file (str):  The file to load.
        input (dict):  Append to this dictionary.

    Returns:
        (dict):  The result.

    """
    raw = None
    with open(file, "r") as f:
        raw = json.load(f)

    if input is None:
        return raw

    # We need to merge results.
    _merge_config(raw_config, input)

    return input


def dump_json(file, conf):
    """Dump a configuration to a JSON file.

    This writes a config dictionary to a JSON file.

    Args:
        file (str):  The file to write.
        conf (dict):  The configuration to dump.

    Returns:
        None

    """
    env = Environment.get()
    versioned = OrderedDict()
    versioned["version"] = env.version()
    versioned.update(conf)

    with open(file, "w") as f:
        json.dump(versioned, f, indent=2)


def load_config(file, input=None):
    """Load a config file in a supported format.

    This loads the document into a config dictionary.  If input is specified, the file
    contents are merged into this dictionary.

    Args:
        file (str):  The file to load.
        input (dict):  Append to this dictionary.

    Returns:
        (dict):  The result.

    """
    ret = None
    try:
        ret = load_json(file, input=input)
    except Exception:
        ret = load_toml(file, input=input)
    return ret


def create(conf):
    """Instantiate classes in a configuration.

    This iteratively instantiates classes defined in the configuration, replacing
    object names with references to those objects.  References to other objects in the
    config are specified with the string '@config:' followed by a UNIX-style "path"
    where each element of the path is a dictionary key in the config.  For example:

        @config:/operators/pointing

    Would reference an object at conf["operators"]["pointing"].  Object references like
    this only work if the target of the reference is a built-in type (str, float, int,
    etc) or a class derived from TraitConfig.

    Args:
        conf (dict):  the configuration

    Returns:
        (dict):  The dictionary of instantiated classes

    """
    log = Logger.get()
    ref_prefix = "@config:"
    ref_pat = re.compile("^{}/(.*)".format(ref_prefix))

    # Helper functions

    def get_node(tree, cursor):
        node = None
        try:
            node = tree
            for c in cursor:
                parent = node
                node = parent[c]
            # We found it!
        except:
            node = None
        return node

    def find_object_ref(top, name):
        """
        Return same string if no match, None if matched but nonexistant, or
        the object itself.
        """
        found = name
        mat = ref_pat.match(name)
        if mat is not None:
            # See if the referenced object exists
            path = mat.group(1)
            path_keys = path.split("/")
            found = get_node(top, path_keys)
        return found

    def parse_tree(in_tree, out_tree, cursor):
        unresolved = 0
        # print("PARSE ------------------------")

        # The node at this cursor location
        # print("PARSE fetching node at cursor {}".format(cursor))
        in_node = get_node(in_tree, cursor)

        # print("PARSE at input {} got node {}".format(cursor, in_node))

        # The output parent node
        parent_cursor = list(cursor)
        node_name = parent_cursor.pop()
        out_parent = get_node(out_tree, parent_cursor)
        # print("PARSE at output parent {} got node {}".format(parent_cursor, out_parent))

        # The output node
        node_type = type(in_node)
        out_parent[node_name] = node_type()

        # In terms of this function, "nodes" are always dictionary-like
        for child_key, child_val in in_node.items():
            if isinstance(child_val, str):
                # print("PARSE child value {} is a string".format(child_val))
                # See if this string is an object reference and try to resolve it.
                check = find_object_ref(out_tree, child_val)
                if check is None:
                    unresolved += 1
                    out_parent[node_name][child_key] = child_val
                else:
                    out_parent[node_name][child_key] = check
            else:
                is_dict = None
                try:
                    subkeys = child_val.keys()
                    # Ok, this child is like a dictionary
                    is_dict = True
                except:
                    is_dict = False
                if is_dict:
                    child_cursor = list(cursor)
                    child_cursor.append(child_key)
                    # print(
                    #     "PARSE child value {} is a dict, descend with cursor {}".format(
                    #         child_val, child_cursor
                    #     )
                    # )
                    unresolved += parse_tree(in_tree, out_tree, child_cursor)
                else:
                    # Not a dictionary
                    try:
                        _ = len(child_val)
                        out_parent[node_name][child_key] = [
                            None for x in range(len(child_val))
                        ]

                        for elem in range(len(child_val)):
                            found = find_object_ref(out_tree, child_val[elem])
                            if found is None:
                                unresolved += 1
                                out_parent[node_name][child_key][elem] = child_val[elem]
                            else:
                                out_parent[node_name][child_key][elem] = found
                        # print("PARSE child value {} is a list".format(child_val))
                    except:
                        # Not a list / array, just leave it alone
                        # print("PARSE child value {} is not modified".format(child_val))
                        out_parent[node_name][child_key] = child_val

        # If this node is an object and all refs exist, then create it.  Otherwise
        # leave it alone.
        # print(
        #     "PARSE unresolved = {}, out_parent[{}] has class?  {}".format(
        #         unresolved, node_name, ("class" in out_parent[node_name])
        #     )
        # )
        if unresolved == 0 and "class" in out_parent[node_name]:
            # We have a TraitConfig object with all references resolved.
            # Instantiate it.
            # print("PARSE creating TraitConfig {}".format(node_name))
            obj = TraitConfig.from_config(node_name, out_parent[node_name])
            # print("PARSE instantiated {}".format(obj))
            out_parent[node_name] = obj

        # print("PARSE VERIFY parent[{}] = {}".format(node_name, out_parent[node_name]))
        # print("PARSE out_tree now:\n", out_tree, "\n--------------")
        return unresolved

    # Iteratively instantiate objects

    out = OrderedDict()

    done = False
    last_unresolved = None

    it = 0
    while not done:
        # print("PARSE iter ", it)
        done = True
        unresolved = 0
        for sect in list(conf.keys()):
            # print("PARSE  examine ", sect, "-->", type(conf[sect]))
            if not isinstance(conf[sect], (dict, OrderedDict)):
                continue
            out[sect] = OrderedDict()
            # print("PARSE   section ", sect)
            unresolved += parse_tree(conf, out, [sect])

        if last_unresolved is not None:
            if unresolved == last_unresolved:
                msg = "Cannot resolve all references in the configuration"
                log.error(msg)
                raise RuntimeError(msg)
        last_unresolved = unresolved
        if unresolved > 0:
            done = False
        it += 1

    return out
