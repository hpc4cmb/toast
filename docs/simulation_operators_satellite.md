
# Simulated Satellite Observing

Simulating artificial observing from a satellite instrument consists of
two parts: building an observing \"schedule\" and then simulating
telescope motion using that schedule. A satellite schedule contains a
list of scans, which describe the overall telescope motion for some
length of time:

```{eval-rst}
.. autoclass:: toast.schedule.SatelliteScan
    :members:
```

```{eval-rst}
.. autoclass:: toast.schedule.SatelliteSchedule
    :members:
```

For generating large schedules, it is best to use the included
commandline tool:

```{include} simulation_operators_sat_sched.inc
```

Which writes the schedule to an ECSV file. This file can then be loaded
before passing it to the `SimSatellite` operator.
Alternatively, for small tests, you can build a schedule directly in
memory by calling the underlying function:

```{eval-rst}
.. autofunction:: toast.schedule_sim_satellite.create_satellite_schedule
```

After you have a `SatelliteSchedule` created or read from
disk, you can use the `SimSatellite` operator to actually
generate observations:

```{eval-rst}
.. autoclass:: toast.ops.SimSatellite
    :members:
```

This operator will append observations (using the schedule) to the
`Data` container passed to the `exec()` method.
The observations will contain simulated telescope pointing, and will
have the detector timestreams initialized to zero- ready for calling
other simulation operators to generate detector signals.
