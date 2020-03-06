

import numpy as np

import toast.tod as tt

rate = 100.0
obssamp = 100000

# Constantly slewing precession axis
degday = 360.0 / 365.25

spinperiod = 10.0
spinangle = 30.0
precperiod = 50.0
precangle = 65.0

# obs 1

qprec = tt.slew_precession_axis(nsim=obssamp, 
    firstsamp=0, samplerate=rate, degday=degday)

boresight = tt.satellite_scanning(nsim=obssamp, firstsamp=0, qprec=qprec, samplerate=rate, spinperiod=spinperiod, spinangle=spinangle, precperiod=precperiod, precangle=precangle)

np.savetxt("custom_example_boresight_1.txt", boresight)

# obs 2

qprec = tt.slew_precession_axis(nsim=obssamp, 
    firstsamp=obssamp, samplerate=rate, degday=degday)

boresight = tt.satellite_scanning(nsim=obssamp, firstsamp=obssamp, qprec=qprec, samplerate=rate, spinperiod=spinperiod, spinangle=spinangle, precperiod=precperiod, precangle=precangle)

np.savetxt("custom_example_boresight_2.txt", boresight)

