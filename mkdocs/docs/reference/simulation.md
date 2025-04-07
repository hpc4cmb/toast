# Simulation Tools

TOAST contains a variety of Operators and other tools for simulating telescope
observations and different detector signals.

## Simulated Observing

When designing new telescopes or observing strategies the TOAST scheduler can be used to
create schedule files that can be passed to the `SimGround` and `SimSatellite`
operators.

### Ground-Based Schedules

#### Sky Patches

!!! note "To-Do"
    We can't add docs for the patch types, because they have no docstrings...

<!-- ::: toast.schedule_sim_ground.Patch
::: toast.schedule_sim_ground.SSOPatch
::: toast.schedule_sim_ground.CoolerCyclePatch
::: toast.schedule_sim_ground.HorizontalPatch
::: toast.schedule_sim_ground.WeightedHorizontalPatch
::: toast.schedule_sim_ground.SiderealPatch
::: toast.schedule_sim_ground.MaxDepthPatch -->

#### Scheduling Utilities

!!! note "To-Do"
    Do we want more of the low-level tools here?

::: toast.schedule_sim_ground.parse_patches
::: toast.schedule_sim_ground.build_schedule

#### Generating the Schedule

::: toast.schedule_sim_ground.run_scheduler

### Space-Based Schedules

Generating schedules for a satellite is conceptually simpler due to the constraints on spacecraft dynamics.

::: toast.schedule_sim_satellite.create_satellite_schedule

### Creating Observations

::: toast.ops.SimGround
::: toast.ops.SimSatellite

## Sky Signals

These operators generate detector data containing sources of power from outside the Earth's atmosphere.

::: toast.ops.SimDipole

### Beam-Convolved Sky

::: toast.ops.SimConviqt
::: toast.ops.SimTEBConviqt
::: toast.ops.SimWeightedConviqt
::: toast.ops.SimTotalconvolve

### Scanning a Healpix Map

::: toast.ops.ScanHealpixMap
::: toast.ops.ScanHealpixMask
::: toast.ops.InterpolateHealpixMap

### Scanning a WCS Projected Map

::: toast.ops.ScanWCSMap
::: toast.ops.ScanWCSMask

### Scanning an Arbitrary Map

::: toast.ops.ScanMap
::: toast.ops.ScanMask
::: toast.ops.ScanScale

### Point Sources

::: toast.ops.SimCatalog
<!--  ::: toast.ops.SimSSO -->

## Terrestrial Signals

These operators generate detector signal from the Earth's atmosphere and other sources of power outside a ground-based telescope.

::: toast.ops.WeatherModel
::: toast.ops.SimAtmosphere

::: toast.ops.SimScanSynchronousSignal

## Instrument Signals

These operators simulate instrumental effects from sources of power inside the telescope and receiver.

::: toast.ops.DefaultNoiseModel
::: toast.ops.ElevationNoise
::: toast.ops.SimNoise
::: toast.ops.CommonModeNoise

::: toast.ops.TimeConstant

::: toast.ops.InjectCosmicRays

::: toast.ops.GainDrifter
::: toast.ops.GainScrambler

::: toast.ops.PerturbHWP

<!-- Port HWPSS sim from sotodlib -->

::: toast.ops.CrossTalk

::: toast.ops.YieldCut

