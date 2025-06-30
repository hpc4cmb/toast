#!/usr/bin/env python3

# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Load two TOAST config files and report the differences between them."""

import argparse
import re
import sys
import traceback
from collections import OrderedDict

from astropy.units import Quantity, Unit
from toast.mpi import Comm, get_world
from toast.trait_utils import string_to_scalar
from toast.utils import Environment, Logger, import_from_name, object_fullname

import toast
from toast.config import dump_json, dump_toml, dump_yaml, load_config


def cropped_string(value, width):
    """Crop long strings, Add color to select strings"""
    try:
        value = string_to_scalar(value)
    except (SyntaxError, NameError):
        pass
    # Convert to string
    string = str(value)
    # If necessary, replace the middle with "..."
    if len(string) > width:
        w = width // 2 - 3
        string = string[:w] + "..." + string[-w:]
    baselength = len(string)
    # Use ANSI color sequences for highlighting.
    # Will not work on Windows
    if string == "True" or string == "ENABLED":
        string = "\033[32m" + string + "\033[0m"
    elif string == "False" or string == "DISABLED":
        string = "\033[31m" + string + "\033[0m"
    elif string == "NOT SET":
        string = "\033[33m" + string + "\033[0m"
    # Adding ANSI color sequences confuses Python string alignment
    extralength = len(string) - baselength
    return string, extralength


def compare_dictionaries(
    dict1,
    dict2,
    depth=0,
    msg=[],
    tab="  ",
    width=40,
):
    """Recursively compare two dictionaries and print out the
    differences

    """
    mismatched = False
    keys = set(dict1.keys())
    keys.update(dict2.keys())
    for key in sorted(keys):
        # for key, value1 in dict1.items():
        # Special case: if operator is only enabled in one config,
        # do not inspect any other keys
        if (
            "enabled" in dict1
            and dict1["enabled"] != dict2["enabled"]
            and key != "enabled"
        ):
            continue
        # Check if the key is present in both dictionaries
        # or just in one
        missing = False
        if key in dict1:
            value1 = dict1[key]
            if key in dict2:
                value2 = dict2[key]
            else:
                missing = True
                if value1["enabled"]["value"] == "True":
                    value1 = "ENABLED"
                else:
                    value1 = "DISABLED"
                value2 = "NOT SET"
                msg.append(tab * depth + f"{key}")
        else:
            missing = True
            if value2["enabled"]["value"] == "True":
                value2 = "ENABLED"
            else:
                value2 = "DISABLED"
            value1 = "NOT SET"
            msg.append(tab * depth + f"{key}")
        if isinstance(value1, OrderedDict) and not missing:
            mismatched |= compare_dictionaries(
                value1,
                value2,
                depth=depth + 1,
                msg=msg + [tab * depth + f"{key}"],
                tab=tab,
                width=width,
            )
        else:
            if value1 != value2:
                if msg[-1].strip() == "enabled":
                    # Special handling for the "enabled" key
                    if value1 == "True":
                        value1 = "ENABLED"
                    else:
                        value1 = "DISABLED"
                    if value2 == "True":
                        value2 = "ENABLED"
                    else:
                        value2 = "DISABLED"
                    msg.pop()
                w1, w2 = width
                w1 -= len(msg[-1])
                str1, extra1 = cropped_string(value1, w1)
                str2, extra2 = cropped_string(value2, w2)
                print("\n".join(msg) + f"{str1:>{w1+extra1}}{str2:>{w2+extra2}}")
                mismatched = True
        if mismatched:
            msg = []
    return mismatched


def main():
    env = Environment.get()
    log = Logger.get()

    mpiworld, procs, rank = get_world()
    if rank != 0:
        return

    if len(sys.argv) != 3:
        msg = f"Usage: toast_config_compare <config1> <config2>"
        raise RuntimeError(msg)

    path0 = sys.argv[0]
    path1 = sys.argv[1]
    path2 = sys.argv[2]

    configs = []
    for path in path1, path2:
        print(f"Parsing {path}")
        parser = argparse.ArgumentParser(description="Compare two TOAST configs")

        sys.argv = [path0, "--config", path]
        config_in, args, jobargs = toast.parse_config(parser)

        # Instantiate everything and then convert back to a config.
        # This will automatically prune stale traits, etc.
        run = toast.create_from_config(config_in)
        run_vars = vars(run)

        config_out = OrderedDict()
        for sect_key, sect_val in run_vars.items():
            sect_vars = vars(sect_val)
            obj_list = list()
            for obj_name, obj in sect_vars.items():
                obj_list.append(obj)
            config_out.update(toast.config.build_config(obj_list))
        configs.append(config_out)

    w1 = 70
    w2 = 60
    print("Comparing")
    print(f"{path1:>{w1}}{path2:>{w2}}")
    compare_dictionaries(*configs, tab="  ", width=(w1, w2))

    return


if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()
