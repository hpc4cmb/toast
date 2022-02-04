```{code-block} console

usage: toast_satellite_schedule [-h] [--out OUT] [--start START]
                                [--num_obs NUM_OBS]
                                [--obs_minutes OBS_MINUTES]
                                [--gap_minutes GAP_MINUTES]
                                [--prec_minutes PREC_MINUTES]
                                [--spin_minutes SPIN_MINUTES]

Create a satellite observing schedule.

optional arguments:
  -h, --help            show this help message and exit
  --out OUT             The output schedule file.
  --start START         The start time of the mission as an ISO-8601 string.
  --num_obs NUM_OBS     The number of observations.
  --obs_minutes OBS_MINUTES
                        The length of each observation in minutes.
  --gap_minutes GAP_MINUTES
                        The length of the gaps between observations in
                        minutes.
  --prec_minutes PREC_MINUTES
                        The precession period in minutes.
  --spin_minutes SPIN_MINUTES
                        The spin period in minutes.

```
