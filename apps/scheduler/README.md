# scheduler

## Description

The goal is to find a fast, scalable way to determine whether multiple
applications may be run concurrently on the same node and stay within memory
limits.

## Current schedulers

- `scheduleHeuristic`:
  + Rudimentary approach of shifting runs forward in time.
  + Total memory used by all runs never exceeds the limit.
  + Only considers memory usage.
- `scheduleILP`:
  + More refined approach that minimizes the total runtime (the "makespan").
  + Keep total memory usage and CPU utilization under user-defined limits.
  + Considers CPU and memory utilization.
