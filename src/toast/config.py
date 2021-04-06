# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import types

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

from .traits import TraitConfig

from . import ops as ops


def build_config(objects):
    """Build a configuration of current values.

    Args:
        objects (list):  A list of class instances to add to the config.  These objects
            must inherit from the TraitConfig base class.

    Returns:
        (dict):  The configuration.

    """
    conf = OrderedDict()
    for o in objects:
        if not isinstance(o, TraitConfig):
            raise RuntimeError("The object list should contain TraitConfig instances")
        if o.name is None:
            raise RuntimeError("Cannot buid config from objects without a name")
        conf = o.get_config(input=conf)
    return conf


def add_config_args(
    parser, conf, section, ignore=list(), disabled=False, prefix="", separator="."
):
    """Add arguments to an argparser for each parameter in a config dictionary.

    Using a previously created config dictionary, add a commandline argument for each
    object parameter in a section of the config.  The type, units, and help string for
    the commandline argument come from the config, which is in turn built from the
    class traits of the object.  Boolean parameters are converted to store_true or
    store_false actions depending on their current value.

    Args:
        parser (ArgumentParser):  The parser to append to.
        conf (dict):  The configuration dictionary.
        section (str):  Process objects in this section of the config.
        ignore (list):  List of object parameters to ignore when adding args.
        disabled (bool):  If True, these objects are disabled by default.
        prefix (str):  Prepend this to the beginning of all options.
        separator (str):  Use this character between the class name and parameter.

    Returns:
        None

    """
    parent = conf
    if section is not None:
        path = section.split("/")
        for p in path:
            if p not in parent:
                msg = "section {} does not exist in config".format(section)
                raise RuntimeError(msg)
            parent = parent[p]
    for obj, props in parent.items():
        # Add option to enable / disable the object, so calling workflow can take
        # actions to conditionally use some objects.  This does not affect the parsing
        # of options.
        if disabled:
            # add option enable
            parser.add_argument(
                "--{}enable_{}".format(prefix, obj),
                required=False,
                default=False,
                action="store_true",
                help="Enable use of {}".format(obj),
            )
        else:
            # add option disable
            parser.add_argument(
                "--{}disable_{}".format(prefix, obj),
                required=False,
                default=False,
                action="store_true",
                help="Disable use of {}".format(obj),
            )
        for name, info in props.items():
            # print("examine options for {} = {}".format(name, info))
            if name in ignore:
                # Skip this as requested
                # print("  ignoring")
                continue
            if name == "class":
                # This is not a user-configurable parameter.
                # print("  skipping")
                continue
            if info["type"] not in ["bool", "int", "float", "str", "Quantity"]:
                # This is not something that we can get from parsing commandline
                # options.  Skip it.
                # print("  no type- skipping")
                continue
            if info["type"] == "bool":
                # special case for boolean
                option = "--{}{}{}{}".format(prefix, obj, separator, name)
                act = "store_true"
                if info["value"] == "True":
                    act = "store_false"
                    option = "--{}{}{}no_{}".format(prefix, obj, separator, name)
                # print("  add bool argument {}".format(option))
                parser.add_argument(
                    option,
                    required=False,
                    default=info["value"],
                    action=act,
                    help=info["help"],
                )
            else:
                option = "--{}{}{}{}".format(prefix, obj, separator, name)
                default = None
                typ = None
                hlp = info["help"]
                if info["type"] == "int":
                    typ = int
                    if info["value"] != "None":
                        default = int(info["value"])
                elif info["type"] == "float":
                    typ = float
                    if info["value"] != "None":
                        default = float(info["value"])
                elif info["type"] == "str":
                    typ = str
                    if info["value"] != "None":
                        default = info["value"]
                elif info["type"] == "Quantity":
                    typ = u.Quantity
                    if info["value"] != "None":
                        default = u.Quantity(
                            "{} {}".format(info["value"], info["unit"])
                        )
                # print("  add argument {}".format(option))
                parser.add_argument(
                    option,
                    required=False,
                    default=default,
                    type=typ,
                    help=hlp,
                )
    return


def args_update_config(args, conf, defaults, section, prefix="", separator="."):
    """Override options in a config dictionary from args namespace.

    Args:
        args (namespace):  The args namespace returned by ArgumentParser.parse_args()
        conf (dict):  The configuration to update.
        defaults (dict):  The starting default config, used to detect which options from
            argparse have been changed by the user.
        section (str):  Process objects in this section of the config.
        prefix (str):  Prepend this to the beginning of all options.
        separator (str):  Use this character between the class name and parameter.

    Returns:
        (namespace):  The un-parsed remaining arg vars.

    """
    remain = copy.deepcopy(args)
    parent = conf
    dparent = defaults
    if section is not None:
        path = section.split("/")
        for p in path:
            if p not in parent:
                msg = "section {} does not exist in config".format(section)
                raise RuntimeError(msg)
            parent = parent[p]
        for p in path:
            if p not in dparent:
                msg = "section {} does not exist in defaults".format(section)
                raise RuntimeError(msg)
            dparent = dparent[p]
    # Build the regex match of option names
    obj_pat = re.compile("{}(.*?){}(.*)".format(prefix, separator))
    for arg in vars(args):
        val = getattr(args, arg)
        obj_mat = obj_pat.match(arg)
        if obj_mat is not None:
            name = obj_mat.group(1)
            optname = obj_mat.group(2)
            if name not in parent:
                msg = (
                    "Parsing option '{}', config does not have object named {}".format(
                        arg, name
                    )
                )
                raise RuntimeError(msg)
            if name not in dparent:
                msg = "Parsing option '{}', defaults does not have object named {}".format(
                    arg, name
                )
                raise RuntimeError(msg)
            # Only update config options which are different than the default.
            # Otherwise we would be overwriting values from any config files with the
            # defaults from argparse.
            if val is None:
                val = "None"
            else:
                if dparent[name][optname]["unit"] != "None":
                    # This option is a quantity
                    val = "{:0.14e}".format(
                        val.to_value(u.Unit(dparent[name][optname]["unit"]))
                    )
                elif dparent[name][optname]["type"] == "float":
                    val = "{:0.14e}".format(val)
                else:
                    val = str(val)
            if val != dparent[name][optname]["value"]:
                parent[name][optname]["value"] = val
            # This arg was recognized, remove from the namespace.
            delattr(remain, arg)
    return remain


def parse_config(
    parser,
    operators_enabled=list(),
    operators_disabled=list(),
    templates_enabled=list(),
    templates_disabled=list(),
    prefix="",
):
    """Load command line arguments associated with object properties.

    This function:

        * Adds a "--config" option to the parser which accepts multiple config file
          paths to load.

        * Adds "--default_toml" and "--default_json" options to dump the default config
          options to files.

        * Adds a option "--job_group_size" to provide the commonly used option for
          setting the group size of the TOAST communicator.

        * Adds arguments for all object parameters in the defaults for the specified
          classes.

        * Builds a config dictionary starting from the defaults, updating these using
          values from any config files, and then applies any overrides from the
          commandline.

    Args:
        parser (ArgumentParser):  The argparse parser.
        operators_enabled (list):  The operator instances to configure from files or
            commandline.  These instances should have their "name" attribute set to
            something meaningful, since that name is used to construct the commandline
            options.  These are enabled by default and a "--disable_*" option is
            created for each.
        operators_disabled (list):  Same as above, but the operators are disabled by
            default and an "--enable_*" option is created for each.
        templates_enabled (list):  The template instances to add to the commandline.
            These instances should have their "name" attribute set to something
            meaningful, since that name is used to construct the commandline options.
            These are enabled by default and a "--disable_*" option is created for each.
        templates_disabled (list):  Same as above, but the templates are disabled by
            default and an "--enable_*" option is created for each.
        prefix (str):  Optional string to prefix all options by.

    Returns:
        (tuple):  The (config dictionary, args).  The args namespace contains all the
            remaining parameters after extracting the operator and template options.

    """

    # The default configuration
    defaults_op_enabled = build_config(operators_enabled)
    defaults_op_disabled = build_config(operators_disabled)
    defaults_tmpl_enabled = build_config(templates_enabled)
    defaults_tmpl_disabled = build_config(templates_disabled)

    # Add commandline overrides
    if len(operators_enabled) > 0:
        add_config_args(
            parser,
            defaults_op_enabled,
            "operators",
            ignore=["API"],
            disabled=False,
            prefix=prefix,
        )
    if len(operators_disabled) > 0:
        add_config_args(
            parser,
            defaults_op_disabled,
            "operators",
            ignore=["API"],
            disabled=True,
            prefix=prefix,
        )
    if len(templates_enabled) > 0:
        add_config_args(
            parser,
            defaults_tmpl_enabled,
            "templates",
            ignore=["API"],
            disabled=False,
            prefix=prefix,
        )
    if len(templates_disabled) > 0:
        add_config_args(
            parser,
            defaults_tmpl_disabled,
            "templates",
            ignore=["API"],
            disabled=True,
            prefix=prefix,
        )

    # Combine all the defaults
    defaults = OrderedDict()
    defaults["operators"] = OrderedDict()
    defaults["templates"] = OrderedDict()
    if "operators" in defaults_op_enabled:
        defaults["operators"].update(defaults_op_enabled["operators"])
    if "operators" in defaults_op_disabled:
        defaults["operators"].update(defaults_op_disabled["operators"])
    if "templates" in defaults_tmpl_enabled:
        defaults["templates"].update(defaults_tmpl_enabled["templates"])
    if "templates" in defaults_tmpl_disabled:
        defaults["templates"].update(defaults_tmpl_disabled["templates"])

    # Add an option to load one or more config files.  These should have compatible
    # names for the operators used in defaults.
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        nargs="+",
        help="One or more input config files.",
    )

    # Add options to dump default values
    parser.add_argument(
        "--default_toml",
        type=str,
        required=False,
        default=None,
        help="Dump default config values to a TOML file",
    )
    parser.add_argument(
        "--default_json",
        type=str,
        required=False,
        default=None,
        help="Dump default config values to a JSON file",
    )

    parser.add_argument(
        "--job_group_size",
        required=False,
        type=int,
        default=None,
        help="(Advanced) Size of the process groups",
    )

    parser.add_argument(
        "--job_node_mem",
        required=False,
        type=int,
        default=None,
        help="(Advanced) Override the detected memory per node in bytes",
    )

    # Parse commandline.
    args = parser.parse_args()

    # Dump default config values.
    if args.default_toml is not None:
        dump_toml(args.default_toml, defaults)
    if args.default_json is not None:
        dump_json(args.default_json, defaults)

    # Parse job args
    jobargs = types.SimpleNamespace(
        node_mem=args.job_node_mem,
        group_size=args.job_group_size,
    )
    del args.job_node_mem
    del args.job_group_size

    # Load any config files.  This overrides default values with config file contents.
    config = copy.deepcopy(defaults)
    if args.config is not None:
        for conf in args.config:
            config = load_config(conf, input=config)

    # Parse operator commandline options.  These override any config file or default
    # values.
    op_remaining = args_update_config(
        args, config, defaults, "operators", prefix=prefix
    )
    remaining = args_update_config(
        op_remaining, config, defaults, "templates", prefix=prefix
    )

    # Remove the options we created in this function
    del remaining.config
    del remaining.default_toml
    del remaining.default_json

    return config, remaining, jobargs


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


def _load_toml_traits(tbl):
    # print("LOAD TraitConfig object {}".format(tbl), flush=True)
    result = OrderedDict()
    for k in tbl.keys():
        if k == "class":
            result[k] = tbl[k]
        elif isinstance(tbl[k], tomlkit.items.Table):
            # This is a dictionary trait
            result[k] = OrderedDict()
            result[k]["value"] = OrderedDict()
            result[k]["type"] = "dict"
            result[k]["unit"] = "None"
            for tk, tv in tbl[k].items():
                result[k]["value"][str(tk)] = str(tv)
        elif isinstance(tbl[k], tomlkit.items.Array):
            # This is a list
            result[k] = OrderedDict()
            result[k]["value"] = list()
            result[k]["type"] = "list"
            result[k]["unit"] = "None"
            for it in tbl[k]:
                result[k]["value"].append(str(it))
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
    # print("LOAD toml result = {}".format(result))
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
                        # print("LOAD found traitconfig {}".format(k), flush=True)
                        conf_root[k] = _load_toml_traits(raw_root[k])
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
                subindent = indent + 2
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
            # print("{}found dict".format(" " * indent_size))
            for k in list(conf_root.keys()):
                # print("{}  examine key {}".format(" " * indent_size, k))
                if isinstance(conf_root[k], (dict, OrderedDict)):
                    # print("{}  key is a dict".format(" " * indent_size))
                    if "value" in conf_root[k] and "type" in conf_root[k]:
                        # print(
                        #    "{}  found value and type subkeys".format(" " * indent_size)
                        # )
                        # this is a trait
                        unit = None
                        if "unit" in conf_root[k]:
                            unit = conf_root[k]["unit"]
                        help = None
                        if "help" in conf_root[k]:
                            help = conf_root[k]["help"]
                        # print(
                        #     "{}  dumping trait {}, {}, {}".format(
                        #         " " * indent_size,
                        #         k,
                        #         conf_root[k]["value"],
                        #         conf_root[k]["type"],
                        #     )
                        # )
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
                        # print("{}  not a trait- descending".format(" " * indent_size))
                        # descend tree
                        table_root[k] = table()
                        convert_node(conf_root[k], table_root[k], indent_size + 2)
                else:
                    # print("{}  value = {}".format(" " * indent_size, conf_root[k]))
                    # print(
                    #     "{}  key is not a dict, add to table".format(" " * indent_size)
                    # )
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


def create_from_config(conf):
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
        (SimpleNamespace):  A namespace containing the sections and instantiated
            objects specified in the original config structure.

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
            # print("OBJREF checking {}".format(path_keys))
            found = get_node(top, path_keys)
            if found is not None:
                # It exists, but is this a TraitConfig object that has not yet been
                # created?
                if isinstance(found, (dict, OrderedDict)) and "class" in found:
                    # Yes...
                    found = None
            # print("OBJREF found = {}".format(found))
        return found

    def parse_tree(tree, cursor):
        unresolved = 0
        # print("PARSE ------------------------")

        # The node at this cursor location
        # print("PARSE fetching node at cursor {}".format(cursor))
        node = get_node(tree, cursor)

        # print("PARSE at cursor {} got node {}".format(cursor, node))

        # The output parent node
        parent_cursor = list(cursor)
        node_name = parent_cursor.pop()
        parent = get_node(tree, parent_cursor)
        # print("PARSE at parent {} got node {}".format(parent_cursor, parent))

        # In terms of this function, "nodes" are always dictionary-like
        for child_key in list(node.keys()):
            # We are modifying the tree in place, so we get a new reference to our
            # node each time.
            child_cursor = list(cursor)
            child_cursor.append(child_key)
            child_val = get_node(tree, child_cursor)

            if isinstance(child_val, TraitConfig):
                # This is an already-created object
                continue
            elif isinstance(child_val, str):
                # print("PARSE child value {} is a string".format(child_val))
                # See if this string is an object reference and try to resolve it.
                check = find_object_ref(tree, child_val)
                if check is None:
                    unresolved += 1
                else:
                    parent[node_name][child_key] = check
            else:
                is_dict = None
                try:
                    subkeys = child_val.keys()
                    # Ok, this child is like a dictionary
                    is_dict = True
                except:
                    is_dict = False
                if is_dict:
                    # print(
                    #     "PARSE child value {} is a dict, descend with cursor {}".format(
                    #         child_val, child_cursor
                    #     )
                    # )
                    unresolved += parse_tree(tree, child_cursor)
                else:
                    # Not a dictionary
                    is_list = None
                    try:
                        _ = len(child_val)
                        # It is a list
                        is_list = True
                    except:
                        is_list = False
                    if is_list:
                        for elem in range(len(child_val)):
                            found = find_object_ref(tree, child_val[elem])
                            # print(
                            #     "find_object {} --> {}".format(child_val[elem], found)
                            # )
                            if found is None:
                                unresolved += 1
                            else:
                                parent[node_name][child_key][elem] = found
                        # print("PARSE child value {} is a list".format(child_val))
                    else:
                        print("PARSE not modifying {}".format(child_val))

        # If this node is an object and all refs exist, then create it.  Otherwise
        # leave it alone.
        # print(
        #     "PARSE unresolved = {}, parent[{}] has class?  {}".format(
        #         unresolved, node_name, ("class" in parent[node_name])
        #     )
        # )
        if unresolved == 0 and "class" in parent[node_name]:
            # We have a TraitConfig object with all references resolved.
            # Instantiate it.
            # print("PARSE creating TraitConfig {}".format(node_name))
            obj = TraitConfig.from_config(node_name, parent[node_name])
            # print("PARSE instantiated {}".format(obj))
            parent[node_name] = obj

        # print("PARSE VERIFY parent[{}] = {}".format(node_name, parent[node_name]))
        # print("PARSE tree now:\n", tree, "\n--------------")
        return unresolved

    # Iteratively instantiate objects

    out = copy.deepcopy(conf)

    done = False
    last_unresolved = None

    it = 0
    while not done:
        # print("PARSE iter ", it)
        done = True
        unresolved = 0
        for sect in list(out.keys()):
            # print("PARSE  examine ", sect, "-->", type(out[sect]))
            if not isinstance(out[sect], (dict, OrderedDict)):
                continue
            # print("PARSE   section ", sect)
            unresolved += parse_tree(out, [sect])

        if last_unresolved is not None:
            if unresolved == last_unresolved:
                msg = "Cannot resolve all references in the configuration"
                log.error(msg)
                raise RuntimeError(msg)
        last_unresolved = unresolved
        if unresolved > 0:
            done = False
        it += 1

    # Convert this recursively into a namespace for easy use

    root_temp = dict()
    for sect in list(out.keys()):
        sect_ns = types.SimpleNamespace(**out[sect])
        root_temp[sect] = sect_ns

    out_ns = types.SimpleNamespace(**root_temp)

    return out_ns
