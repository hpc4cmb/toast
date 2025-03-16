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

## Sky Signals



## Instrument Signals

