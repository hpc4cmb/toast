
# Simulated Ground-Based Observing

Simulating observing with a ground-based telescope also consists of
generating an observing schedule and using that schedule to simulate
telescope pointing. A ground-based schedule contains a list of scans,
each of which describes the high-level motion of the telescope for some
length of time:

```{eval-rst}
.. autoclass:: toast.schedule.GroundScan
    :members:
```

```{eval-rst}
.. autoclass:: toast.schedule.GroundSchedule
    :members:
```

Because ground schedules are more complex, they are usually generated
with a commandline tool:

```{include} toast_ground_schedule.inc
```

This writes the schedule to a custom format that can later be loaded
before passing it to the `SimGround` operator:

```{eval-rst}
.. autoclass:: toast.ops.SimGround
    :members:
```
