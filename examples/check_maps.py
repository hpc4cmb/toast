import healpy as hp
import numpy as np

folder = "out_tiny_satellite/"
ref = "ref_out_tiny_satellite/"

h = hp.read_map(folder + "out_hits.fits")
h_ref = hp.read_map(ref + "out_hits.fits")

assert h.sum() == h_ref.sum(), "Total hits wrong"
assert np.sum(np.abs(h - h_ref))==0, "Hitmaps differ"

m = hp.read_map(folder + "out_000/binned.fits")
m_ref = hp.read_map(ref + "out_000/binned.fits")

m_diff_std = (m - m_ref).std()

assert m_diff_std < 1e-7, "Maps differ, std {}".format(m_diff_std)

print("Test passed")
