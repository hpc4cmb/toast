# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Ipython magic helpers."""

from argparse import ArgumentParser

from IPython.core.magic import (
    Magics,
    cell_magic,
    line_cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)

from .startup import start_parallel


@magics_class
class ToastMagics(Magics):
    def __init__(self, shell):
        # You must call the parent constructor
        super(ToastMagics, self).__init__(shell)

    def parse_args(self, line):
        parser = ArgumentParser(prog="TOAST Interactive")
        parser.add_argument(
            "-p", dest="procs", default=1, type=int, help="Number of processes to use"
        )
        parser.add_argument(
            "-t",
            dest="threads",
            default=1,
            type=int,
            help="Number of threads per process",
        )
        parser.add_argument(
            "-n",
            dest="nice",
            default=False,
            action="store_true",
            help="Run nice on all processes",
        )
        parser.add_argument(
            "-a",
            dest="auto",
            default=False,
            action="store_true",
            help="Enable %autopx for parallel cluster",
        )
        parser.add_argument("message", nargs="*")
        return parser.parse_args(line.split())

    @line_magic
    def toast(self, line):
        args = self.parse_args(line)
        return start_parallel(
            procs=args.procs,
            threads=args.threads,
            nice=args.nice,
            auto_mpi=args.auto,
            shell=self.shell,
            shell_line=line,
        )


def load_ipython_extension(ipython):
    magics = ToastMagics(ipython)
    ipython.register_magics(magics)
