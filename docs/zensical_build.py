#!/usr/bin/env python3
"""Zensical build wrapper which pre-imports the toast package.

Zensical renders pages in worker threads.  The dynamic API documentation
(mkdocstrings "force_inspection" option) imports the toast package during
the build, but importing toast from a non-main thread fails, since the
pshmem package registers signal handlers at import time.  Importing toast
here, before the build starts, means that the already-loaded module is
found in sys.modules by the build workers.
"""

import os
import sys

import toast  # noqa: F401  (must happen in the main thread)

from zensical import build

if __name__ == "__main__":
    config_file = None
    for candidate in ["zensical.toml", "mkdocs.yml", "mkdocs.yaml"]:
        if os.path.exists(candidate):
            config_file = candidate
            break
    else:
        raise RuntimeError("No config file found in the current folder.")

    # Match the CLI behavior of `zensical build` (no flags used by build.sh)
    build(os.path.abspath(config_file), {"clean": False, "strict": False})
