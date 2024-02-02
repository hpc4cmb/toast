# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..utils import Environment, Logger


def merge_config(loaded, original):
    log = Logger.get()
    allowed = {
        "float": set(["float", "int", "bool"]),
        "int": set(["int", "bool"]),
        "bool": set(["int", "bool"]),
        "str": set(["str"]),
        "Quantity": set(["Quantity", "Unit"]),
        "Unit": set(["Unit"]),
        "list": set(["list", "tuple", "set"]),
        "tuple": set(["list", "tuple", "set"]),
        "set": set(["list", "tuple", "set"]),
        "dict": set(["dict"]),
        "enum": set(["enum", "int"]),
        "str": set(["str"]),
    }
    for section, objs in loaded.items():
        if section in original.keys():
            # We have this section
            if isinstance(original[section], str):
                # This is not a real section, but rather a string value of
                # top-level metadata.  Just copy it into place, overriding
                # the existing value.
                original[section] = objs
                continue
            for objname, objprops in objs.items():
                if objname not in original[section]:
                    # This is a new object
                    original[section][objname] = objprops
                else:
                    # Only update the value, while preserving
                    # any pre-existing type information.
                    for k, v in objprops.items():
                        if k == "class":
                            continue
                        if k in original[section][objname]:
                            # This key exists in the original object traits
                            cursor = original[section][objname][k]
                            cursor["value"] = v["value"]
                            if "type" not in cursor:
                                cursor["type"] = v["type"]
                            elif v["type"] is not None and v["type"] != "unknown":
                                # The loaded data has a type, check for consistency
                                if cursor["type"] not in allowed:
                                    # This is not one of the types we can check.  It
                                    # is likely the class name of an Instance trait.
                                    pass
                                elif v["type"] not in allowed[cursor["type"]]:
                                    msg = f"Loaded trait {v} has type not compatible "
                                    msg += f"with existing trait {cursor}"
                                    raise ValueError(msg)
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
