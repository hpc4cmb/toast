#!/usr/bin/env python3

import os
import sys
import re
import subprocess as sp
import shutil
import stat

lesson_dir = os.path.dirname(sys.argv[0])
print("Scanning {}".format(lesson_dir))

nersc_dir = "/project/projectdirs/cmb/www/toast-tutorial"

for root, dirs, files in os.walk(lesson_dir):
    for d in dirs:
        if re.match(r"\d\d_.*", d) is None:
            continue
        ldir = os.path.join(lesson_dir, d)
        print("Scanning lesson dir {}".format(ldir))
        for lr, ld, lf in os.walk(ldir):
            for f in lf:
                if re.match(r".*\.ipynb", f) is None:
                    continue
                # We have a notebook!
                fpath = os.path.join(ldir, f)
                try:
                    print("Converting {} ...".format(fpath))
                    com = ["jupyter", "nbconvert", "--to", "html", fpath]
                    sp.check_call(com)
                    html = re.sub(r"\.ipynb", ".html", f)
                    infile = os.path.join(ldir, html)
                    outdir = os.path.join(nersc_dir, d)
                    outfile = os.path.join(outdir, html)
                    os.makedirs(outdir, exist_ok=True)
                    print("Move {} --> {}".format(infile, outfile))
                    shutil.copy2(infile, outfile)
                    os.chmod(
                        outfile,
                        stat.S_IROTH
                        | stat.S_IRGRP
                        | stat.S_IRUSR
                        | stat.S_IWGRP
                        | stat.S_IWUSR,
                    )
                    os.remove(infile)
                except:
                    print("Failed conversion")
                    raise
            break
    break
