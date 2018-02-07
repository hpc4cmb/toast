# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse

from . import timing

# Define some module-level variables that can be easily checked by other
# parts of the code

timing_enabled = False



def parse_args(parser):
    """Add internal TOAST command line options.

    Update an existing parser with command line options that control internal
    package state.  These are prefixed with "toast-" in order to separate them
    from user pipeline options.  This then parses all options (internal and
    user), strips the internal options, acts on those options, and passes
    back the remaining args object.

    Args:
        parser (argparse.ArgumentParser): the parser to update and parse.

    Returns:
        namespace: the usual args namespace with internal attributes removed.

    """
    prefix = "toast"

    # Add timing related options which will be used to control timemory
    # options if we are using that package.
    parser.add_argument( "--{}-timing-enable".format(prefix), required=False,
        default=False, action="store_true",
        help="Enable timing support" )

    parser.add_argument( "--{}-timing-enable-serialization".format(prefix),
        required=False, default=False, action="store_true",
        help="Enable timing serialization" )

    parser.add_argument( "--{}-timing-outfile".format(prefix),
        required=False, default="toast_timing", type=str,
        help="Timing output file when using a single file" )

    parser.add_argument( "--{}-timing-outdir".format(prefix),
        required=False, default=".", type=str,
        help="Timing output directory" )

    # Add other global package options here...


    # Parse all arguments

    args = parser.parse_args()

    # Did the user request to enable timing?

    do_timers = getattr(args, "{}_timing_enable".format(prefix))
    if do_timers:
        global timing_enabled
        timing_enabled = True

    # If we are using timemory, act on those options

    if timing.use_timemory:
        import timemory
        import timemory.options as opts

        do_ser = getattr(args, "{}_timing_enable_serialization".format(prefix))

        # Start with everything disabled
        timemory.toggle(False)
        opts.use_timers = False
        opts.report_file = False
        opts.serial_file = False

        if do_timers:
            # If timers are enabled, we enable them in timemory, and also
            # enable the standard report format.
            timemory.toggle(True)
            opts.use_timers = True
            opts.report_file = True

        # FIXME:  This does not seem to do the expected.  The serialized JSON
        # file seems to be always written.  Perhaps something to do with
        # MPI support?  This is something that can be changed / fixed in
        # this one function without impacting the rest of the codebase.

        if do_ser:
            # If we are doing serialization, then enable that option and
            # disable the standard report.
            opts.serial_file = True
            opts.report_file = False

        opts.output_dir = getattr(args, "{}_timing_outdir".format(prefix))
        opts.serial_fname = "{}.json".format(getattr(args,
            "{}_timing_outfile".format(prefix)))
        opts.report_fname = "{}.out".format(getattr(args,
            "{}_timing_outfile".format(prefix)))

    # Act on other global options here....



    # Clear all internal args

    delattr(args, "{}_timing_enable".format(prefix))
    delattr(args, "{}_timing_enable_serialization".format(prefix))
    delattr(args, "{}_timing_outfile".format(prefix))
    delattr(args, "{}_timing_outdir".format(prefix))

    # Return user args
    return args
