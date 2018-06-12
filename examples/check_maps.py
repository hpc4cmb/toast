import glob
import sys

import healpy as hp

run_type = sys.argv[1]

folder = "out_tiny_{}/".format(run_type)
ref = "ref_" + folder

# recursive glob requires Python 3.5
filenames = glob.glob(ref + "/**/*.fits", recursive=True)
assert len(filenames) > 0, "No FITS files in the reference folder"
for filename in filenames:


    for i_pol, pol in enumerate("IQU"):
        print("Compare", filename, pol)
        try:
            m = hp.read_map(filename.replace(ref, folder), i_pol, verbose=False)
            m_ref = hp.read_map(filename, i_pol, verbose=False)
        except IndexError:
            # map is not polarized
            continue
        m_diff_std = (m - m_ref).std()

        assert m_diff_std < 1e-7, "{} Maps differ, std {}".format(pol, m_diff_std)

print("Test passed")
