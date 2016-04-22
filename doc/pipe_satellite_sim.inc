.. code-block:: console

  usage: toast_satellite_sim.py [-h] [--samplerate SAMPLERATE]
                                [--spinperiod SPINPERIOD]
                                [--spinangle SPINANGLE]
                                [--precperiod PRECPERIOD]
                                [--precangle PRECANGLE] [--hwprpm HWPRPM]
                                [--hwpstep HWPSTEP] [--hwpsteptime HWPSTEPTIME]
                                [--obs OBS] [--gap GAP] [--numobs NUMOBS]
                                [--obschunks OBSCHUNKS] [--outdir OUTDIR]
                                [--debug] [--nside NSIDE] [--baseline BASELINE]
                                [--noisefilter] [--madam] [--fp FP]

  Simulate satellite boresight pointing and make a noise map.

  optional arguments:
    -h, --help            show this help message and exit
    --samplerate SAMPLERATE
                          Detector sample rate (Hz)
    --spinperiod SPINPERIOD
                          The period (in minutes) of the rotation about the spin
                          axis
    --spinangle SPINANGLE
                          The opening angle (in degrees) of the boresight from
                          the spin axis
    --precperiod PRECPERIOD
                          The period (in minutes) of the rotation about the
                          precession axis
    --precangle PRECANGLE
                          The opening angle (in degrees) of the spin axis from
                          the precession axis
    --hwprpm HWPRPM       The rate (in RPM) of the HWP rotation
    --hwpstep HWPSTEP     For stepped HWP, the angle in degrees of each step
    --hwpsteptime HWPSTEPTIME
                          For stepped HWP, the the time in seconds between steps
    --obs OBS             Number of hours in one science observation
    --gap GAP             Cooler cycle time in hours between science obs
    --numobs NUMOBS       Number of complete observations
    --obschunks OBSCHUNKS
                          Number of chunks to subdivide each observation into
                          for data distribution
    --outdir OUTDIR       Output directory
    --debug               Write diagnostics
    --nside NSIDE         Healpix NSIDE
    --baseline BASELINE   Destriping baseline length (seconds)
    --noisefilter         Destripe with the noise filter enabled
    --madam               If specified, use libmadam for map-making
    --fp FP               Pickle file containing a dictionary of detector
                          properties. The keys of this dict are the detector
                          names, and each value is also a dictionary with keys
                          "quat" (4 element ndarray), "fwhm" (float, arcmin),
                          "fknee" (float, Hz), "alpha" (float), and "NET"
                          (float). For optional plotting, the key "color" can
                          specify a valid matplotlib color string.
