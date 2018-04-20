import glob
import sys

import healpy as hp

run_type = sys.argv[1]

folder = "out_tiny_{}/".format(run_type)
ref = "ref_" + folder

for filename in glob.iglob(ref + "/**/*.fits", recursive=True):

    print("Compare", filename)

    m = hp.read_map(filename.replace(ref, folder), verbose=False)
    m_ref = hp.read_map(filename, verbose=False)
    m_diff_std = (m - m_ref).std()

    assert m_diff_std < 1e-7, "Maps differ, std {}".format(m_diff_std)

print("Test passed")
