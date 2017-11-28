#!/usr/bin/env python3

import json
import sys
import argparse
import traceback

#import matplotlib
#matplotlib.use("qt4agg")
#import matplotlib.pyplot as plt


def plot(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="File input")

    args = parser.parse_args()

    print('Opening {}...'.format(args.file))
    f = open(args.file, "r")
    data_0 = json.load(f)

    print('Iterating over ranks...')
    for i in range(0, len(data_0['ranks'])):
        data_1 = data_0['ranks'][i]
        print('Iterating over timing manager timers...')
        for j in range(0, len(data_1['timing_manager']['timers'])):
            data_2 = data_1['timing_manager']['timers'][j]
            print("{}".format(data_2['timer.tag']))


if __name__ == "__main__":
    try:
        plot(sys.argv[1:])
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))

    print ('Done {}'.format(sys.argv[0]))
    sys.exit(0)
