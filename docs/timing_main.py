#!/usr/bin/env python

import toast
import toast.timing as timing

def main():

    # This is the 2-level toast communicator.  By default,
    # there is just one group which spans MPI_COMM_WORLD.
    comm = toast.Comm()

    # Create an argparse and add custom arguments
    parser = argparse.ArgumentParser(description=“...")
    parser.add_argument('--groupsize',
                        required=False, type=np.int,
                        help='Size of a process group assigned to a CES')

    # pass the argparse object to timing module which will add timing
    # arguments and return "parse.parse_args() result after handling
    # the timing specific options
    args = timing.add_arguments_and_parse(parser, timing.FILE(noquotes=True))
    # create the primary auto timer for the entire script
    autotimer = timing.auto_timer(timing.FILE())

    # do the work...
    # etc...

if __name__ == '__main__':
    try:
        main()
        tman = timing.timing_manager()
        tman.report()
    except Exception as e:
        # etc...
