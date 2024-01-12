# Copyright (c) 2015-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..utils import Environment, Logger


def merge_config(loaded, original):
    log = Logger.get()
    for section, objs in loaded.items():
        if section in original.keys():
            # We have this section
            if isinstance(original[section], str):
                # Just some metadata
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
