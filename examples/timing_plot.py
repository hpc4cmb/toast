#!/usr/bin/env python3

from __future__ import division
import json
import sys
import argparse
import traceback
import collections
import numpy as np
import matplotlib.pyplot as plt

types = ('wall', 'sys', 'user', 'cpu', 'perc')
concurrency = 1
mpi_size = 1
min_time = 0.01

#=============================================================================#
def nested_dict():
    return collections.defaultdict(nested_dict)

#import matplotlib
#matplotlib.use("qt4agg")
#import matplotlib.pyplot as plt


class timing_data():

    def __init__(self):
        self.data = nested_dict()
        for key in types:
            self.data[key] = []

    def append(self, _data):
        n = 0
        for key in types:
            self.data[key].append(_data[n])
            n += 1

    def __add__(self, rhs):
        for key in types:
            self.data[key].extend(rhs.data[key])

    def reset(self):
        self.data = nested_dict()
        for key in types:
            self.data[key] = []

    def __getitem__(self, key):
        return self.data[key]


#=============================================================================#
class timing_function():

    def __init__(self):
        self.data = timing_data()

    def process(self, denom, start, stop):
        _wall = float(int(stop['wall']) - int(start['wall'])) / denom
        _user = float(int(stop['user']) - int(start['user'])) / denom
        _sys = float(int(stop['sys']) - int(start['sys'])) / denom
        _cpu = _user + _sys
        if _wall > 0.0:
            _perc = (_cpu / _wall) * 100.0
        else:
            _perc = 100.0
        if _wall > min_time:
            self.data.append([_wall, _sys, _user, _cpu, _perc])

    def __getitem__(self, key):
        return self.data[key]

    def length(self):
        return len(self.data['cpu'])


#=============================================================================#
def plot(filename, title, timing_data_dict):

    ntics = len(timing_data_dict)
    ytics = []

    if ntics == 0:
        print ('{} had no timing data less than the minimum time ({} s)'.format(filename,
                                                                            min_time))
        return()

    avgs = nested_dict()
    stds = nested_dict()
    for key in types:
        avgs[key] = []
        stds[key] = []

    for func, obj in timing_data_dict.items():
        ytics.append(func)
        for key in types:
            data = obj[key]
            avgs[key].append(np.mean(data))
            if len(data) > 1:
                stds[key].append(np.std(data))
            else:
                stds[key].append(0.0)

    # the x locations for the groups
    ind = np.arange(ntics)
    # the thickness of the bars: can also be len(x) sequence
    thickness = 0.8

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    f.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.90)

    ytics.reverse()
    for key in types:
        avgs[key].reverse()
        stds[key].reverse()

    iter_order = ('cpu', 'wall', 'sys')
    plots = []
    lk = None
    for key in iter_order:
        data = avgs[key]
        err = stds[key]
        p = None
        if lk is None:
            p = plt.barh(ind, data, thickness, xerr=err)
        else:
            p = plt.barh(ind, data, thickness, xerr=err, bottom=lk)
        #lk = avgs[key]
        plots.append(p)

    plt.grid()
    plt.xlabel('Time [seconds]')
    plt.title('Timing report for {}'.format(title))
    plt.yticks(ind, ytics, ha='left')
    plt.setp(ax.get_yticklabels(), fontsize='smaller')
    plt.legend(plots, iter_order)
    plt.show()


#=============================================================================#
def read(filename):

    print('Opening {}...'.format(filename))
    f = open(filename, "r")
    data_0 = json.load(f)
    global concurrency
    global mpi_size

    max_level = 0
    concurrency_sum = 0
    mpi_size = len(data_0['ranks'])
    for i in range(0, len(data_0['ranks'])):
        data_1 = data_0['ranks'][i]
        concurrency_sum += int(data_1['timing_manager']['omp_concurrency'])
        for j in range(0, len(data_1['timing_manager']['timers'])):
            data_2 = data_1['timing_manager']['timers'][j]
            nlaps = int(data_2['timer.ref']['laps'])
            indent = ""
            nlevel = int(data_2['timer.level'])
            max_level = max([max_level, nlevel])

    concurrency = concurrency_sum / mpi_size
    timing_functions = nested_dict()
    for i in range(0, len(data_0['ranks'])):
        data_1 = data_0['ranks'][i]
        for j in range(0, len(data_1['timing_manager']['timers'])):
            data_2 = data_1['timing_manager']['timers'][j]
            nlaps = int(data_2['timer.ref']['laps'])
            indent = ""
            nlevel = int(data_2['timer.level'])
            for n in range(0, nlevel):
                indent = ' {}'.format(indent)
            tag = '{} {} x {}'.format(indent, data_2['timer.tag'], nlaps)

            if not tag in timing_functions:
                timing_functions[tag] = timing_function()
            timing_func = timing_functions[tag]
            data_3 = data_2['timer.ref']

            for k in range(0, nlaps):
                data_4 = data_3['raw_history'][k]
                start = data_4['start']['time_since_epoch']['count']
                stop = data_4['stop']['time_since_epoch']['count']
                timing_func.process(data_3['to_seconds_ratio_den'], start, stop)

            if timing_func.length() == 0:
                del timing_functions[tag]

    return timing_functions


#=============================================================================#
def main(args):
    global concurrency
    global mpi_size

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs='*', help="File input")

    args = parser.parse_args()
    print('Files: {}'.format(args.files))

    file_data = dict()
    file_title = dict()
    for filename in args.files:
        print ('Reading {}...'.format(filename))
        file_data[filename] = read(filename)
        title = filename.replace('timing_report_', '')
        title = title.replace('json', 'py')
        title = '"{}"\n@ MPI procs = {}, Threads/proc = {}'.format(title, mpi_size,
                                                                int(concurrency))
        file_title[filename] = title

    for filename, data in file_data.items():
        print ('Plotting {}...'.format(filename))
        plot(filename, file_title[filename], data)


#=============================================================================#
if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))

    print ('Done - {}'.format(sys.argv[0]))
    sys.exit(0)
