# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import ast
import copy
import json
import re
import sys
import types
from collections import OrderedDict
from collections.abc import MutableMapping

import numpy as np
import tomlkit
from astropy import units as u
from tomlkit import comment, document, dumps, loads, nl, table

from . import instrument
from .instrument import Focalplane, Telescope
from .traits import TraitConfig, trait_scalar_to_string, trait_string_to_scalar
from .utils import Environment, Logger


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


def add_config_args(parser, conf, section, ignore=list(), prefix="", separator="."):
    """Add arguments to an argparser for each parameter in a config dictionary.

    Using a previously created config dictionary, add a commandline argument for each
    object parameter in a section of the config.  The type, units, and help string for
    the commandline argument come from the config, which is in turn built from the
    class traits of the object.  Boolean parameters are converted to store_true or
    store_false actions depending on their current value.

    Containers in the config (list, set, tuple, dict) are parsed as a string to
    support easy passing on the commandline, and then later converted to the actual
    type.

    Args:
        parser (ArgumentParser):  The parser to append to.
        conf (dict):  The configuration dictionary.
        section (str):  Process objects in this section of the config.
        ignore (list):  List of object parameters to ignore when adding args.
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
            if info["type"] not in [
                "bool",
                "int",
                "float",
                "str",
                "Quantity",
                "Unit",
                "list",
                "dict",
                "tuple",
                "set",
            ]:
                # This is not something that we can get from parsing commandline
                # options.  Skip it.
                # print("  no type- skipping")
                continue
            if info["type"] == "bool":
                # Special case for boolean
                if name == "enabled":
                    # Special handling of the TraitConfig "enabled" trait.  We provide
                    # both an enable and disable option for every instance which later
                    # toggles that trait.
                    df = True
                    if info["value"] == "False":
                        df = False
                    parser.add_argument(
                        "--{}{}{}enable".format(prefix, obj, separator),
                        required=False,
                        default=df,
                        action="store_true",
                        help="Enable use of {}".format(obj),
                    )
                    parser.add_argument(
                        "--{}{}{}disable".format(prefix, obj, separator),
                        required=False,
                        default=(not df),
                        action="store_false",
                        help="Disable use of {}".format(obj),
                        dest="{}{}{}enable".format(prefix, obj, separator),
                    )
                else:
                    # General bool option
                    option = "--{}{}{}{}".format(prefix, obj, separator, name)
                    act = "store_true"
                    dflt = False
                    if info["value"] == "True":
                        act = "store_false"
                        dflt = True
                        option = "--{}{}{}no_{}".format(prefix, obj, separator, name)
                    # print("  add bool argument {}".format(option))
                    parser.add_argument(
                        option,
                        required=False,
                        default=dflt,
                        action=act,
                        help=info["help"],
                        dest="{}{}{}{}".format(prefix, obj, separator, name),
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
                elif info["type"] == "Unit":
                    typ = u.Unit
                    # The "value" for a Unit is always "unit"
                    if info["unit"] != "None":
                        default = u.Unit(info["unit"])
                elif info["type"] == "Quantity":
                    typ = u.Quantity
                    if info["value"] != "None":
                        default = u.Quantity(
                            "{} {}".format(info["value"], info["unit"])
                        )
                elif info["type"] == "list":
                    typ = str
                    if info["value"] != "None":
                        if len(info["value"]) == 0:
                            default = "[]"
                        else:
                            formatted = [
                                trait_scalar_to_string(x) for x in info["value"]
                            ]
                            default = f"{formatted}"
                elif info["type"] == "set":
                    typ = str
                    if info["value"] != "None":
                        if len(info["value"]) == 0:
                            default = "{}"
                        else:
                            formatted = set(
                                [trait_scalar_to_string(x) for x in info["value"]]
                            )
                            default = f"{formatted}"
                elif info["type"] == "tuple":
                    typ = str
                    if info["value"] != "None":
                        if len(info["value"]) == 0:
                            default = "()"
                        else:
                            formatted = tuple(
                                [trait_scalar_to_string(x) for x in info["value"]]
                            )
                            default = f"{formatted}"
                elif info["type"] == "dict":
                    typ = str
                    if info["value"] != "None":
                        if len(info["value"]) == 0:
                            default = "{}"
                        else:
                            formatted = {
                                x: trait_scalar_to_string(y)
                                for x, y in info["value"].items()
                            }
                            # print(f"dict formatted = {formatted}")
                            default = f"{info['value']}"
                else:
                    raise RuntimeError("Invalid trait type, should never get here!")
                # print(
                #     f"  add_argument: {option}, type={typ}, default={default} ({type(default)})"
                # )
                parser.add_argument(
                    option,
                    required=False,
                    default=default,
                    type=typ,
                    help=hlp,
                )
    return


def args_update_config(args, conf, useropts, section, prefix="", separator="\."):
    """Override options in a config dictionary from args namespace.

    Args:
        args (namespace):  The args namespace returned by ArgumentParser.parse_args()
        conf (dict):  The configuration to update.
        useropts (list):  The list of options actually specified by the user.
        section (str):  Process objects in this section of the config.
        prefix (str):  Prepend this to the beginning of all options.
        separator (str):  Use this character between the class name and parameter.

    Returns:
        (namespace):  The un-parsed remaining arg vars.

    """
    # Build the regex match of option names
    obj_pat = re.compile("{}(.*?){}(.*)".format(prefix, separator))
    user_pat = re.compile("--{}(.*?){}(.*)".format(prefix, separator))
    no_pat = re.compile("^no_(.*)")

    # Regex patterns for containers
    list_pat = re.compile(r"\[(.*)\]")
    dictset_pat = re.compile(r"\{(.*)\}")
    tuple_pat = re.compile(r"\((.*)\)")
    dictelem_pat = re.compile(r"(.*):(.*)")

    def _strip_quote(s):
        if s == "":
            return s
        else:
            return s.strip(" '\"")

    def _parse_list(parg):
        if parg is None:
            return "None"
        else:
            mat = list_pat.match(parg)
            if mat is None:
                msg = f"Argparse value '{parg}' is not a list"
                raise ValueError(msg)
            value = list()
            inner = mat.group(1)
            if inner == "[]" or inner == "":
                return value
            else:
                elems = inner.split(",")
                # print(f"_parse_list elems split = {elems}")
                for elem in elems:
                    # value.append(trait_string_to_scalar(_strip_quote(elem)))
                    value.append(_strip_quote(elem))
                return value

    def _parse_tuple(parg):
        if parg is None:
            return "None"
        else:
            mat = tuple_pat.match(parg)
            if mat is None:
                msg = f"Argparse value '{parg}' is not a tuple"
                raise ValueError(msg)
            value = list()
            inner = mat.group(1)
            if inner == "()" or inner == "":
                return tuple(value)
            else:
                elems = inner.split(",")
                for elem in elems:
                    value.append(_strip_quote(elem))
                    # value.append(trait_string_to_scalar(_strip_quote(elem)))
                return tuple(value)

    def _parse_set(parg):
        if parg is None:
            return "None"
        else:
            mat = dictset_pat.match(parg)
            if mat is None:
                msg = f"Argparse value '{parg}' is not a set"
                raise ValueError(msg)
            value = set()
            inner = mat.group(1)
            if inner == "{}" or inner == "":
                return value
            else:
                elems = inner.split(",")
                for elem in elems:
                    value.add(_strip_quote(elem))
                    # value.add(trait_string_to_scalar(_strip_quote(elem)))
                return value

    def _parse_dict(parg):
        if parg is None:
            return "None"
        else:
            mat = dictset_pat.match(parg)
            if mat is None:
                msg = f"Argparse value '{parg}' is not a dict"
                raise ValueError(msg)
            dstr = mat.group(1)
            value = dict()
            if dstr == "{}" or dstr == "":
                return value
            else:
                elems = dstr.split(",")
                for elem in elems:
                    el = _strip_quote(elem)
                    elem_mat = dictelem_pat.match(el)
                    if elem_mat is None:
                        msg = f"Argparse value '{parg}', element '{el}' "
                        msg += f" is not a dictionary key / value "
                        raise ValueError(msg)
                    elem_key = _strip_quote(elem_mat.group(1))
                    elem_val = _strip_quote(elem_mat.group(2))
                    value[elem_key] = elem_val
                    # value[elem_key] = trait_string_to_scalar(elem_val)
            return value

    # Parse the list of user options
    user = dict()
    for uo in useropts:
        user_mat = user_pat.match(uo)
        if user_mat is not None:
            name = user_mat.group(1)
            optname = user_mat.group(2)
            if name not in user:
                user[name] = set()
            # Handle mapping of option names
            if optname == "enable" or optname == "disable":
                optname = "enabled"
            no_mat = no_pat.match(optname)
            if no_mat is not None:
                optname = no_mat.group(1)
            user[name].add(optname)

    remain = copy.deepcopy(args)
    parent = conf
    if section is not None:
        path = section.split("/")
        for p in path:
            if p not in parent:
                msg = "section {} does not exist in config".format(section)
                raise RuntimeError(msg)
            parent = parent[p]

    # print(f"PARSER start config = {parent}")
    # print("PARSER = ", args)
    for arg, val in vars(args).items():
        obj_mat = obj_pat.match(arg)
        if obj_mat is not None:
            name = obj_mat.group(1)
            optname = obj_mat.group(2)
            if name not in parent:
                # This command line option is not part of this section
                continue

            if optname == "enable":
                # The actual trait name is "enabled"
                optname = "enabled"

            # For each option, convert the parsed value and compare with
            # the default config value.  If it differs, and the user
            # actively set this option, then change it.

            if parent[name][optname]["type"] == "bool":
                if isinstance(val, bool):
                    # Convert to str
                    if val:
                        val = "True"
                    else:
                        val = "False"
                elif val is None:
                    val = "None"
                else:
                    raise ValueError(f"value '{val}' is not bool or None")
                if (val != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = val
            elif parent[name][optname]["type"] == "Unit":
                # print(f"Parsing Unit:  {val} ({type(val)})")
                if isinstance(val, u.UnitBase):
                    unit = str(val)
                elif val is None:
                    unit = "None"
                else:
                    raise ValueError(f"value '{val}' is not a Unit or None")
                if (unit != parent[name][optname]["unit"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = "unit"
                    parent[name][optname]["unit"] = unit
            elif parent[name][optname]["type"] == "Quantity":
                # print(f"Parsing Quantity:  {val} ({type(val)})")
                if isinstance(val, u.Quantity):
                    value = f"{val.value:0.14e}"
                    unit = str(val.unit)
                elif val is None:
                    value = "None"
                    unit = parent[name][optname]["unit"]
                else:
                    raise ValueError(f"value '{val}' is not a Quantity or None")
                if (
                    value != parent[name][optname]["value"]
                    or unit != parent[name][optname]["unit"]
                ) and (name in user and optname in user[name]):
                    parent[name][optname]["value"] = value
                    parent[name][optname]["unit"] = unit
            elif parent[name][optname]["type"] == "float":
                if val is None:
                    val = "None"
                else:
                    val = f"{val:0.14e}"
                if (val != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = val
            elif parent[name][optname]["type"] == "int":
                if val is None:
                    val = "None"
                else:
                    val = f"{val}"
                if (val != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = val
            elif parent[name][optname]["type"] == "list":
                # For configs that are constructed from TOML, all sequence
                # containers are parsed as a list.  So try several types.
                try:
                    value = _parse_set(val)
                except ValueError:
                    try:
                        value = _parse_tuple(val)
                    except ValueError:
                        value = _parse_list(val)
                if (value != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = value
            elif parent[name][optname]["type"] == "tuple":
                value = _parse_tuple(val)
                if (value != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = value
            elif parent[name][optname]["type"] == "set":
                value = _parse_set(val)
                if (value != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = value
            elif parent[name][optname]["type"] == "dict":
                value = _parse_dict(val)
                if (value != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = value
            else:
                # This is a plain string
                if val is None:
                    val = "None"
                else:
                    val = str(val)
                if (val != parent[name][optname]["value"]) and (
                    name in user and optname in user[name]
                ):
                    parent[name][optname]["value"] = val
            delattr(remain, arg)
    return remain


def parse_config(
    parser,
    operators=list(),
    templates=list(),
    prefix="",
    opts=None,
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
        operators (list):  The operator instances to configure from files or
            commandline.  These instances should have their "name" attribute set to
            something meaningful, since that name is used to construct the commandline
            options.  An enable / disable option is created for each, which toggles the
            TraitConfig base class "disabled" trait.
        templates (list):  The template instances to add to the commandline.
            These instances should have their "name" attribute set to something
            meaningful, since that name is used to construct the commandline options.
            An enable / disable option is created for each, which toggles the
            TraitConfig base class "disabled" trait.
        prefix (str):  Optional string to prefix all options by.
        opts (list):  If not None, parse arguments from this list instead of sys.argv.

    Returns:
        (tuple):  The (config dictionary, args).  The args namespace contains all the
            remaining parameters after extracting the operator and template options.

    """

    # The default configuration
    defaults_op = build_config(operators)
    defaults_tmpl = build_config(templates)

    # Add commandline overrides
    if len(operators) > 0:
        add_config_args(
            parser,
            defaults_op,
            "operators",
            ignore=["API"],
            prefix=prefix,
        )
    if len(templates) > 0:
        add_config_args(
            parser,
            defaults_tmpl,
            "templates",
            ignore=["API"],
            prefix=prefix,
        )

    # Combine all the defaults
    defaults = OrderedDict()
    defaults["operators"] = OrderedDict()
    defaults["templates"] = OrderedDict()
    if "operators" in defaults_op:
        defaults["operators"].update(defaults_op["operators"])
    if "templates" in defaults_tmpl:
        defaults["templates"].update(defaults_tmpl["templates"])

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
        "--defaults_toml",
        type=str,
        required=False,
        default=None,
        help="Dump default config values to a TOML file",
    )
    parser.add_argument(
        "--defaults_json",
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

    # Parse commandline or list of options.
    if opts is None:
        opts = sys.argv[1:]
    args = parser.parse_args(args=opts)

    # print(f"DBG conf defaults = {defaults}")
    # print(f"DBG after parse_args:  {args}")

    # Dump default config values.
    if args.defaults_toml is not None:
        dump_toml(args.defaults_toml, defaults)
    if args.defaults_json is not None:
        dump_json(args.defaults_json, defaults)

    # Parse job args
    jobargs = types.SimpleNamespace(
        node_mem=args.job_node_mem,
        group_size=args.job_group_size,
    )
    del args.job_node_mem
    del args.job_group_size

    # Load any config files.  This overrides default values with config file contents.
    config = copy.deepcopy(defaults)

    # print(f"DBG conf before load = {config}")

    if args.config is not None:
        for conf in args.config:
            config = load_config(conf, input=config)

    # print(f"DBG conf after load = {config}")

    # Parse operator commandline options.  These override any config file or default
    # values.
    op_remaining = args_update_config(args, config, opts, "operators", prefix=prefix)
    remaining = args_update_config(
        op_remaining, config, opts, "templates", prefix=prefix
    )

    # Remove the options we created in this function
    del remaining.config
    del remaining.defaults_toml
    del remaining.defaults_json

    return config, remaining, jobargs


def _merge_config(loaded, original):
    log = Logger.get()
    for section, objs in loaded.items():
        if section in original.keys():
            # We have this section
            for objname, objprops in objs.items():
                if objname not in original[section]:
                    # This is a new object
                    original[section][objname] = objprops
                else:
                    # Only update the value and unit, while preserving
                    # any pre-existing type information.
                    for k, v in objprops.items():
                        if k == "class":
                            continue
                        if k in original[section][objname]:
                            # This key exists in the original object traits
                            cursor = original[section][objname][k]
                            cursor["value"] = v["value"]
                            cursor["unit"] = v["unit"]
                            if "type" not in cursor:
                                cursor["type"] = v["type"]
                        else:
                            # We have a config option that does not exist
                            # in the current object.  Warn user that this may
                            # indicate a stale config file.
                            msg = f"Object {objname} currently has no configuration"
                            msg += f" trait '{k}'.  This might be handled by the "
                            msg += f"class through API translation, but your config "
                            msg += f"file may be out of date."
                            log.warning(msg)
                            original[section][objname][k] = v
        else:
            # A new section
            original[section] = objs


def _load_toml_traits(tbl):
    # print("LOAD TraitConfig object {}".format(tbl), flush=True)
    result = OrderedDict()
    for k in list(tbl.keys()):
        # print(f"LOAD examine key {k}:  {type(tbl[k])}")
        if k == "class":
            # print(f"  found trait class '{tbl[k]}'")
            result[k] = tbl[k]
        elif isinstance(tbl[k], tomlkit.items.Table):
            # This is a dictionary trait
            # print(f"  found trait Dict '{tbl[k]}'")
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
                # print(f"  Array element '{str(it)}'")
                result[k]["value"].append(str(it))
        elif isinstance(tbl[k], str):
            # print(f"  Checking string '{tbl[k]}'")
            if len(tbl[k]) == 0:
                # Special handling for empty string
                # print(f"  found trait str empty '{tbl[k]}'")
                result[k] = OrderedDict()
                result[k]["value"] = ""
                result[k]["type"] = "str"
                result[k]["unit"] = "None"
            elif tbl[k] == "None":
                # Copy None values.  There is no way to determine the type in this case.
                # This is just a feature of how we are constructing the TOML files for
                # ease of use, rather than dumping the full trait info as sub-tables.
                # In practice these parameters will be ignored when constructing
                # TraitConfig objects and the defaults will be used anyway.
                # print(f"  found trait str None '{tbl[k]}'")
                result[k] = OrderedDict()
                result[k]["value"] = tbl[k]
                result[k]["type"] = "unknown"
                result[k]["unit"] = "None"
            else:
                # print("    Not empty or None")
                result[k] = OrderedDict()
                # Is this string actually a Quantity or Unit?
                try:
                    parts = tbl[k].split()
                    vstr = parts.pop(0)
                    ustr = " ".join(parts)
                    if vstr == "unit":
                        # this is a Unit
                        result[k]["value"] = "unit"
                        result[k]["type"] = "Unit"
                        if ustr == "None":
                            result[k]["unit"] = "None"
                        else:
                            result[k]["unit"] = str(u.Unit(ustr))
                        # print(f"  found trait Unit '{tbl[k]}'")
                    elif ustr == "":
                        raise ValueError("No units, just a string")
                    else:
                        result[k]["type"] = "Quantity"
                        if vstr == "None":
                            result[k]["value"] = "None"
                        else:
                            v = float(vstr)
                            result[k]["value"] = f"{v:0.14e}"
                        result[k]["unit"] = str(u.Unit(ustr))
                        # print(f"  found trait Quantity '{tbl[k]}'")
                except Exception as e:
                    # print("      failed... just a string")
                    # Just a regular string
                    # print(f"  found trait str '{tbl[k]}'")
                    result[k]["value"] = tbl[k]
                    result[k]["type"] = "str"
                    result[k]["unit"] = "None"
        elif isinstance(tbl[k], bool):
            # print(f"  found trait bool '{tbl[k]}'")
            result[k] = OrderedDict()
            if tbl[k]:
                result[k]["value"] = "True"
            else:
                result[k]["value"] = "False"
            result[k]["type"] = "bool"
            result[k]["unit"] = "None"
        elif isinstance(tbl[k], int):
            # print(f"  found trait int '{tbl[k]}'")
            result[k] = OrderedDict()
            result[k]["value"] = "{}".format(tbl[k])
            result[k]["type"] = "int"
            result[k]["unit"] = "None"
        elif isinstance(tbl[k], float):
            # print(f"  found trait float '{tbl[k]}'")
            result[k] = OrderedDict()
            result[k]["value"] = "{:0.14e}".format(tbl[k])
            result[k]["type"] = "float"
            result[k]["unit"] = "None"
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
                qval = f"{value} {unit}"
            tbl.add(name, qval)
        elif typ == "Unit":
            uval = "unit None"
            if unit != "None":
                uval = f"unit {unit}"
            tbl.add(name, uval)
        elif typ in ["list", "set", "tuple"]:
            val = "None"
            if value != "None":
                if isinstance(value, str):
                    val = ast.literal_eval(value)
                else:
                    val = list()
                    for elem in value:
                        if isinstance(elem, tuple):
                            # This is a quantity / unit
                            val.append(" ".join(elem))
                        else:
                            val.append(elem)
            tbl.add(name, val)
        elif typ == "dict":
            val = "None"
            if value != "None":
                if isinstance(value, str):
                    dval = ast.literal_eval(value)
                else:
                    dval = dict(value)
                val = table()
                subindent = indent + 2
                for k, v in dval.items():
                    if isinstance(v, tuple):
                        # This is a quantity
                        val.add(k, " ".join(v))
                    else:
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
        elif typ == "callable":
            # Just add None
            tbl.add(name, "None")
        else:
            # This must be a custom class or a str
            tbl.add(name, value)
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
    if comm is None or comm.rank == 0:
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
                            # this is a trait
                            unit = None
                            if "unit" in conf_root[k]:
                                unit = conf_root[k]["unit"]
                            help = None
                            if "help" in conf_root[k]:
                                help = conf_root[k]["help"]
                            # print(
                            #     "{}  trait {}, type {}, ({}): {}".format(
                            #         " " * indent_size,
                            #         conf_root[k]["value"],
                            #         conf_root[k]["type"],
                            #         unit,
                            #         help,
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
                            # descend tree
                            # print(
                            #     "{}  {} not a trait, descending".format(
                            #         " " * indent_size, k
                            #     )
                            # )
                            table_root[k] = table()
                            convert_node(conf_root[k], table_root[k], indent_size + 2)
                    else:
                        # print(
                        #     "{}  key {} not dict".format(" " * indent_size, k),
                        #     flush=True,
                        # )
                        table_root.add(k, conf_root[k])
                        table_root[k].indent(indent_size)
            else:
                raise RuntimeError("Cannot convert config node {}".format(conf_root))

        # Convert all top-level sections from the config dictionary into a TOML table.
        convert_node(conf, doc, 0)

        with open(file, "w") as f:
            f.write(dumps(doc))


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
            raw = json.load(f)
    if comm is not None:
        raw = comm.bcast(raw, root=0)

    if input is None:
        return raw

    # We need to merge results.
    _merge_config(raw, input)

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


def load_config(file, input=None, comm=None):
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
        ret = load_json(file, input=input, comm=comm)
    except Exception:
        ret = load_toml(file, input=input, comm=comm)
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
        # print(f"OBJREF get {name}", flush=True)
        if not isinstance(name, str):
            return name
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
            # print(f"PARSE child_key {child_key} = {child_val}")

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
                        # It is a list, tuple, or set
                        is_list = True
                    except:
                        is_list = False
                    if is_list:
                        parent[node_name][child_key] = list(child_val)
                        for elem in range(len(child_val)):
                            test_val = parent[node_name][child_key][elem]
                            found = find_object_ref(tree, test_val)
                            # print("find_object {} --> {}".format(test_val, found))
                            if found is None:
                                unresolved += 1
                            else:
                                parent[node_name][child_key][elem] = found
                        # print("PARSE child value {} is a list".format(child_val))
                    else:
                        # print("PARSE not modifying {}".format(child_val))
                        pass

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
