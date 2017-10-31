#!/usr/bin/env python3

import sys
import os
import re

dirs = sys.argv[1:]

perf = [
    "pointing",
    "invnpp",
    "npp",
    "noise",
    "map"
]

mats = {}
mats["pointing"] = [
    re.compile(r"TODGround: Simulated scan in\s+(.*)\s+s.*"),
    re.compile(r"TODGround: Translated scan pointing in\s+(.*)\s+s.*"),
    re.compile(r"Simulated scans in\s+(.*)\s+seconds.*"),
    re.compile(r"Pointing generation took\s+(.*)\s+s.*"),
]
mats["invnpp"] = [
    re.compile(r" - distobjects accumulated in\s+(.*)\s+s.*"),
    re.compile(r" - distobjects reduced in\s+(.*)\s+s.*"),
]
mats["npp"] = [
    re.compile(r" - Inverting N_pp\^-1 took\s+(.*)\s+s.*"),
]
mats["atm"] = [
    re.compile(r"Atmosphere simulation took\s+(.*)\s+s.*"),
]
mats["noise"] = [
    re.compile(r"Noise simulation took\s+(.*)\s+s.*"),
]

# Note: the binned map making is double counted here.
mats["map"] = [
    re.compile(r"Mapmaking 0000 took\s+(.*)\s+s.*"),
]

with open("ground.csv", "w") as f:
    header = "job"
    for p in perf:
        header = "{},{}".format(header, p)
    f.write("{}\n".format(header))
    for d in dirs:
        if not os.path.isdir(d):
            continue
        logfile = os.path.join(d, "log")
        vals = {}
        for p in perf:
            vals[p] = 0.0
        with open(logfile, "r") as log:
            for line in log.readlines():
                for p in perf:
                    for mat in mats[p]:
                        result = mat.match(line)
                        if result is not None:
                            vals[p] += float(result.group(1))
        print(vals)
        outline = "{}".format(d)
        for p in perf:
            outline = "{},{}".format(outline, vals[p])
        f.write("{}\n".format(outline))


