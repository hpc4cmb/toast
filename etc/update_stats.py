#!/usr/bin/env python3

import os
import sys
import json


original = sys.argv[1]

data = dict()
do_backup = False
if os.path.isfile(original):
    with open(original, "r") as f:
        data = json.load(f)
    do_backup = True

for newfile in sys.argv[2:]:
    with open(newfile, "r") as f:
        newdata = json.load(f)
    for jobtype, props in newdata.items():
        if jobtype not in data:
            data[jobtype] = dict()
        for case, stats in props.items():
            data[jobtype][case] = stats

if do_backup:
    backup = f"{original}.bak"
    os.rename(original, backup)

with open(original, "w") as f:
    json.dump(data, f)
