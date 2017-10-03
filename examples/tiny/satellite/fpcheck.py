
import sys

test = int(sys.argv[1]) - 1

nrings = 0
while (test - 6 * nrings) > 0:
    test -= 6 * nrings
    nrings += 1

npix = 1
for r in range(1, nrings+1):
    npix += 6 * r

print("npix = {}, ndet = {}".format(npix, npix*2))
