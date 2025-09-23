# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import ast
import copy
import os
import re
import sys
import types
from collections import OrderedDict
from collections.abc import MutableMapping
from json import JSONDecodeError

import numpy as np
from astropy import units as u
from tomlkit.exceptions import TOMLKitError

from ..trait_utils import scalar_to_string as trait_scalar_to_string
from ..trait_utils import string_to_scalar as trait_string_to_scalar
from ..trait_utils import string_to_trait, trait_to_string
from ..utils import Environment, Logger, import_from_name
from .json import dump_json, load_json
from .toml import dump_toml, load_toml
from .yaml import dump_yaml, load_yaml


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
        if not hasattr(o, "get_config"):
            raise RuntimeError("The object list should contain TraitConfig instances")
        if o.name is None:
            raise RuntimeError("Cannot buid config from objects without a name")
        conf = o.get_config(input=conf)
    return conf


def check_config_format(path, format=None):
    """Given a path name, return the config file format.

    Args:
        path (str):  The file name.
        format (str):  If specified, check that the file extension matches.

    Returns:
        (str):  The format ("toml", "json", "yaml")

    """
    log = Logger.get()

    # Determine the format from the file name
    _, rawext = os.path.splitext(path)
    ext = rawext.lower()
    if ext == ".toml" or ext == ".tml":
        file_format = "toml"
    elif ext == ".json" or ext == ".jsn":
        file_format = "json"
    elif ext == ".yaml" or ext == ".yml":
        file_format = "yaml"
    else:
        file_format = None

    if format is None:
        # No information from the calling code
        if file_format is None:
            # ... and we couldn't determine it.
            msg = "Cannot determine format (yaml, toml, json) from"
            msg += f" path '{path}'"
            raise RuntimeError(msg)
        else:
            # Return our guess
            return file_format
    else:
        # Calling code claims it is a particular format
        if file_format is None:
            # We could not determine format from path, trust the caller
            return format
        else:
            # Warn if they disagree, but still trust caller
            if file_format != format:
                msg = (
                    f"Config format '{format}' does not match filename '{file_format}'"
                )
                log.warning(msg)
            return format


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
    format = check_config_format(file)
    if format == "yaml":
        return load_yaml(file, input=input, comm=comm)
    elif format == "json":
        return load_json(file, input=input, comm=comm)
    else:
        return load_toml(file, input=input, comm=comm)


def dump_config(file, conf, format=None, comm=None):
    """Dump a config file to a supported format.

    Writes the configuration to a file in the specified format.  If `format` is
    None, the format with be determined from the file name.

    Args:
        file (str):  The file to write.
        conf (dict):  The configuration to dump.
        format (str):  The format ("toml", "json", "yaml")
        comm (MPI.Comm):  Optional communicator to control which process writes.

    Returns:
        None

    """
    real_format = check_config_format(file, format=format)
    if real_format == "yaml":
        dump_yaml(file, conf, comm=comm)
    elif real_format == "json":
        dump_json(file, conf, comm=comm)
    else:
        dump_toml(file, conf, comm=comm)


class TraitAction(argparse.Action):
    """Custom argparse action to check for valid use of None.

    Some traits support a None value even though they have a specific
    type.  This custom action checks for that None value and validates
    that it is an acceptable value.

    """

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        argtype=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        super(TraitAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=None,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )
        self.trait_type = argtype
        return

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            realval = list()
            for v in values:
                if v == "None" or v is None:
                    realval.append(None)
                else:
                    realval.append(v)
        else:
            if values == "None" or values is None:
                realval = None
            else:
                realval = values
        setattr(namespace, self.dest, realval)


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
    scalar_types = {
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "Quantity": u.Quantity,
        "Unit": u.Unit,
    }
    parsable_types = {
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
    }
    parsable_types.update(scalar_types)

    for obj, props in parent.items():
        for name, info in props.items():
            if name in ignore:
                # Skip this as requested
                continue
            if name == "class":
                # This is not a user-configurable parameter.
                continue
            if info["type"] not in parsable_types:
                # This is not something that we can get from parsing commandline
                # options.  Skip it.
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
                typ = str
                hlp = info["help"]

                # Scalar types are parsed directly.  Containers are left
                # as strings.
                if info["type"] in scalar_types:
                    typ = scalar_types[info["type"]]
                default = trait_to_string(info["value"])
                parser.add_argument(
                    option,
                    action=TraitAction,
                    required=False,
                    default=default,
                    argtype=typ,
                    help=hlp,
                )
    return


def args_update_config(args, conf, useropts, section, prefix="", separator=r"\."):
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
            #
            # NOTE: The TraitAction class correctly assigns empty set defaults to
            # the string "set()", but argparse seems to convert that to "{}".  So
            # here we fix that.
            if parent[name][optname]["type"] == "set" and val == "{}":
                val = "set()"

            value = trait_to_string(val)
            if (value != parent[name][optname]["value"]) and (
                name in user and optname in user[name]
            ):
                parent[name][optname]["value"] = value
            delattr(remain, arg)
    return remain


def add_job_parser_options(parser):
    # Add an option to load one or more config files.  These should have compatible
    # names for the operators used in defaults.
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        nargs="+",
        help="One or more input config files.",
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


def add_default_parser_options(parser, prefix, operators, templates):
    # Add options to dump default values
    parser.add_argument(
        "--defaults",
        type=str,
        required=False,
        default=None,
        help="Dump default config values to a file",
    )

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
    return defaults


def process_job_args(args):
    # Parse job args
    jobargs = types.SimpleNamespace(
        node_mem=args.job_node_mem,
        group_size=args.job_group_size,
    )
    del args.job_node_mem
    del args.job_group_size
    return jobargs


def process_default_args(args, defaults):
    # Dump default config values.
    if args.defaults is not None:
        dump_config(args.defaults, defaults)
    del args.defaults


def process_object_args(args, prefix, config, opts, operators, templates):
    # Parse operator and template commandline options.  These override any config
    # file or default values.
    if len(operators) > 0:
        op_remaining = args_update_config(
            args, config, opts, "operators", prefix=prefix
        )
    else:
        op_remaining = args
    if len(templates) > 0:
        remaining = args_update_config(
            op_remaining, config, opts, "templates", prefix=prefix
        )
    else:
        remaining = op_remaining
    return remaining


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
        (tuple):  The (config dictionary, other args, run args).  The other args
            namespace contains all the remaining parameters after extracting the
            operator and template options.

    """
    add_job_parser_options(parser)

    # Select commandline or list of options.
    if opts is None:
        opts = sys.argv[1:]

    # Add options for user-provided objects
    defaults = add_default_parser_options(parser, prefix, operators, templates)

    # Parse everything
    args = parser.parse_args(args=opts)

    # Process the default logging options
    process_default_args(args, defaults)

    # Extract the job args
    jobargs = process_job_args(args)

    # Load any config files.  This overrides default values with config file contents.
    config = copy.deepcopy(defaults)
    if args.config is not None:
        for conf in args.config:
            config = load_config(conf, input=config)
    del args.config

    # Now override defaults and config options with command line options
    remaining = process_object_args(args, prefix, config, opts, operators, templates)

    return config, remaining, jobargs


def run_config(
    parser,
    prefix="",
    opts=None,
):
    """Use config files to instantiate operators and templates.

    This function is similar to `parse_config()`, but the lists of operators and
    templates is read from the config files rather than being provided by the calling
    code.

    Args:
        parser (ArgumentParser):  The argparse parser.
        prefix (str):  Optional string to prefix all options by.
        opts (list):  If not None, parse arguments from this list instead of sys.argv.

    Returns:
        (tuple):  The (job namespace, config dictionary, other args, runtime args)
            from the result of parsing opts + config_files or command line and then
            configuring all operators and templates.

    """
    add_job_parser_options(parser)

    # Select commandline or list of options.
    if opts is None:
        opts = sys.argv[1:]

    # Parse the initial args, including the config file names
    initial_args, remaining_opts = parser.parse_known_args(args=opts)

    # Parse the job args
    jobargs = process_job_args(initial_args)

    # Get the config files
    if initial_args.config is None:
        raise RuntimeError("run_config() requires at least one config file.")
    config_files = list(initial_args.config)
    del initial_args.config

    # Make a pass through the config files, just to get the object names and types
    raw_config = dict()
    for conf in config_files:
        raw_config = load_config(conf, input=raw_config)
    operators = list()
    templates = list()
    for section, lst in [("operators", operators), ("templates", templates)]:
        if section not in raw_config:
            continue
        for objname, obj in raw_config[section].items():
            if "name" not in obj:
                msg = "config files used with run_config() must have a name for "
                msg += "each object"
                raise RuntimeError(msg)
            if "class" not in obj:
                msg = "config files used with run_config() must have a class for "
                msg += "each object"
                raise RuntimeError(msg)
            cls_path = obj["class"]
            cls = import_from_name(cls_path)
            lst.append(cls(name=obj["name"]["value"]))

    # Now we can build the config of default values for all objects.
    defaults = add_default_parser_options(parser, prefix, operators, templates)

    # Parse the remaining options now that we have our object options in the parser
    args = parser.parse_args(args=remaining_opts)

    # Parse the default logging options
    process_default_args(args, defaults)

    # Make a second pass through the config files.  This overrides default values with
    # config file contents.
    config = copy.deepcopy(defaults)
    for conf in config_files:
        config = load_config(conf, input=config)

    # Now override defaults and config options with command line options
    remaining = process_object_args(args, prefix, config, opts, operators, templates)

    other_dict = {**vars(remaining), **vars(initial_args)}
    otherargs = types.SimpleNamespace(**other_dict)

    return config, otherargs, jobargs
