```{code-block} console

usage: toast_ground_schedule [-h] [--site-name SITE_NAME]
                             [--telescope TELESCOPE] [--site-lon SITE_LON]
                             [--site-lat SITE_LAT] [--site-alt SITE_ALT]
                             [--scan-margin SCAN_MARGIN]
                             [--ra-period RA_PERIOD]
                             [--ra-amplitude-deg RA_AMPLITUDE_DEG]
                             [--dec-period DEC_PERIOD]
                             [--dec-amplitude-deg DEC_AMPLITUDE_DEG]
                             [--elevation-penalty-limit ELEVATION_PENALTY_LIMIT]
                             [--elevation-penalty-power ELEVATION_PENALTY_POWER]
                             [--equalize-area] [--equalize-time]
                             [--equalize-scans] --patch PATCH
                             [--patch-coord PATCH_COORD]
                             [--el-min-deg EL_MIN_DEG]
                             [--el-max-deg EL_MAX_DEG]
                             [--el-step-deg EL_STEP_DEG] [--alternate]
                             [--fp-radius-deg FP_RADIUS_DEG]
                             [--sun-avoidance-angle-deg SUN_AVOIDANCE_ANGLE_DEG]
                             [--moon-avoidance-angle-deg MOON_AVOIDANCE_ANGLE_DEG]
                             [--sun-el-max-deg SUN_EL_MAX_DEG]
                             [--boresight-angle-step-deg BORESIGHT_ANGLE_STEP_DEG]
                             [--boresight-angle-time-min BORESIGHT_ANGLE_TIME_MIN]
                             [--start START] [--stop STOP]
                             [--block-out BLOCK_OUT]
                             [--operational-days OPERATIONAL_DAYS]
                             [--timezone TIMEZONE] [--gap-s GAP_S]
                             [--gap-small-s GAP_SMALL_S]
                             [--time-step-s TIME_STEP_S] [--one-scan-per-day]
                             [--ces-max-time-s CES_MAX_TIME_S] [--debug]
                             [--polmap POLMAP] [--pol-min POL_MIN]
                             [--pol-max POL_MAX] [--delay-sso-check]
                             [--pole-mode]
                             [--pole-el-step-deg POLE_EL_STEP_DEG]
                             [--pole-ces-time-s POLE_CES_TIME_S] [--out OUT]
                             [--boresight-offset-el-deg BORESIGHT_OFFSET_EL_DEG]
                             [--boresight-offset-az-deg BORESIGHT_OFFSET_AZ_DEG]
                             [--elevations-deg ELEVATIONS_DEG]
                             [--partial-scans] [--no-partial-scans]
                             [--pole-raster-scan]
                             [--pole-raster-el-step-deg POLE_RASTER_EL_STEP_DEG]
                             [--az-rate-sky-deg AZ_RATE_SKY_DEG]
                             [--az-accel-mount-deg AZ_ACCEL_MOUNT_DEG]
                             [--el-rate-deg EL_RATE_DEG]
                             [--el-accel-deg EL_ACCEL_DEG]

Generate ground observation schedule.

optional arguments:
  -h, --help            show this help message and exit
  --site-name SITE_NAME
                        Observing site name
  --telescope TELESCOPE
                        Observing telescope name
  --site-lon SITE_LON   Observing site longitude [PyEphem string]
  --site-lat SITE_LAT   Observing site latitude [PyEphem string]
  --site-alt SITE_ALT   Observing site altitude [meters]
  --scan-margin SCAN_MARGIN
                        Random fractional margin [0..1] added to the scans to
                        smooth out edge effects
  --ra-period RA_PERIOD
                        Period of patch position oscillations in RA [visits]
  --ra-amplitude-deg RA_AMPLITUDE_DEG
                        Amplitude of patch position oscillations in RA [deg]
  --dec-period DEC_PERIOD
                        Period of patch position oscillations in DEC [visits]
  --dec-amplitude-deg DEC_AMPLITUDE_DEG
                        Amplitude of patch position oscillations in DEC [deg]
  --elevation-penalty-limit ELEVATION_PENALTY_LIMIT
                        Assign a penalty to observing elevations below this
                        limit [degrees]
  --elevation-penalty-power ELEVATION_PENALTY_POWER
                        Power in the elevation penalty function [> 0]
  --equalize-area       Adjust priorities to account for patch area
  --equalize-time       Modulate priority by integration time.
  --equalize-scans      Modulate priority by number of scans.
  --patch PATCH         Patch definition: name,weight,lon1,lat1,lon2,lat2 ...
                        OR name,weight,lon,lat,width
  --patch-coord PATCH_COORD
                        Sky patch coordinate system [C,E,G]
  --el-min-deg EL_MIN_DEG
                        Minimum elevation for a CES
  --el-max-deg EL_MAX_DEG
                        Maximum elevation for a CES
  --el-step-deg EL_STEP_DEG
                        Optional step to apply to minimum elevation
  --alternate           Alternate between rising and setting scans
  --fp-radius-deg FP_RADIUS_DEG
                        Focal plane radius [deg]
  --sun-avoidance-angle-deg SUN_AVOIDANCE_ANGLE_DEG
                        Minimum distance between the Sun and the bore sight
                        [deg]
  --moon-avoidance-angle-deg MOON_AVOIDANCE_ANGLE_DEG
                        Minimum distance between the Moon and the bore sight
                        [deg]
  --sun-el-max-deg SUN_EL_MAX_DEG
                        Maximum allowed sun elevation [deg]
  --boresight-angle-step-deg BORESIGHT_ANGLE_STEP_DEG
                        Boresight rotation step size [deg]
  --boresight-angle-time-min BORESIGHT_ANGLE_TIME_MIN
                        Boresight rotation step interval [minutes]
  --start START         UTC start time of the schedule
  --stop STOP           UTC stop time of the schedule
  --block-out BLOCK_OUT
                        Range of UTC calendar days to omit from scheduling in
                        format START_MONTH/START_DAY-END_MONTH/END_DAY or
                        START_YEAR/START_MONTH/START_DAY-
                        END_YEAR/END_MONTH/END_DAY where YEAR, MONTH and DAY
                        are integers. END days are inclusive
  --operational-days OPERATIONAL_DAYS
                        Number of operational days to schedule (empty days do
                        not count)
  --timezone TIMEZONE   Offset to apply to MJD to separate operational days
                        [hours]
  --gap-s GAP_S         Gap between CES:es [seconds]
  --gap-small-s GAP_SMALL_S
                        Gap between split CES:es [seconds]
  --time-step-s TIME_STEP_S
                        Time step after failed target acquisition [seconds]
  --one-scan-per-day    Pad each operational day to have only one CES
  --ces-max-time-s CES_MAX_TIME_S
                        Maximum length of a CES [seconds]
  --debug               Write diagnostics, including patch plots.
  --polmap POLMAP       Include polarization from map in the plotted patches
                        when --debug
  --pol-min POL_MIN     Lower plotting range for polarization map
  --pol-max POL_MAX     Upper plotting range for polarization map
  --delay-sso-check     Only apply SSO check during simulated scan.
  --pole-mode           Pole scheduling mode (no drift scan)
  --pole-el-step-deg POLE_EL_STEP_DEG
                        Elevation step in pole scheduling mode [deg]
  --pole-ces-time-s POLE_CES_TIME_S
                        Time to scan at constant elevation in pole mode
  --out OUT             Output filename
  --boresight-offset-el-deg BORESIGHT_OFFSET_EL_DEG
                        Optional offset added to every observing elevation
  --boresight-offset-az-deg BORESIGHT_OFFSET_AZ_DEG
                        Optional offset added to every observing azimuth
  --elevations-deg ELEVATIONS_DEG
                        Fixed observing elevations in a comma-separated list.
  --partial-scans       Allow partials scans when full scans are not
                        available.
  --no-partial-scans    Allow partials scans when full scans are not
                        available.
  --pole-raster-scan    Pole raster scan mode
  --pole-raster-el-step-deg POLE_RASTER_EL_STEP_DEG
                        Elevation step in pole raster scheduling mode [deg]
  --az-rate-sky-deg AZ_RATE_SKY_DEG
                        Azimuthal rate in pole raster scheduling mode [deg]
  --az-accel-mount-deg AZ_ACCEL_MOUNT_DEG
                        Azimuthal accleration in pole raster scheduling mode
                        [deg]
  --el-rate-deg EL_RATE_DEG
                        Elevation rate in pole raster scheduling mode [deg]
  --el-accel-deg EL_ACCEL_DEG
                        Elevation accleration in pole raster scheduling mode
                        [deg]

```
