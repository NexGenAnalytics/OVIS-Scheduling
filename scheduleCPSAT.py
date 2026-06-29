######################################################################## imports

from ortools.sat.python import cp_model
import numpy as np

############################################################# define constraints

MAX_MEMORY = 220 # GB
MAX_CPU = 125 # percent

###################################################################### scheduler

def schedule_direct_cp_sat(cpu_profiles, mem_profiles):

    print("Schedule with direct CP-SAT: ⏳")

    model = cp_model.CpModel()

    n_apps = len(cpu_profiles)
    print(f"Number of apps: {n_apps}")

    durations = [len(p) for p in cpu_profiles]
    print(f"Durations: {durations}")

    # Worst-case horizon (all jobs sequential)
    horizon = sum(durations)
    print(f"Horizon: {horizon}")

    ############################## x[i][s] = 1 if application i starts at time s

    x = {}
    starts = []
    ends = []

    for i in range(n_apps):

        possible = horizon - durations[i] + 1

        vars_i = []

        for s in range(possible):
            x[i, s] = model.NewBoolVar(f"x_{i}_{s}")
            vars_i.append(x[i, s])

        model.AddExactlyOne(vars_i)

        start = model.NewIntVar(0, horizon, f"start_{i}")
        model.Add(
            start ==
            sum(s * x[i, s] for s in range(possible))
        )

        end = model.NewIntVar(0, horizon + durations[i], f"end_{i}")
        model.Add(end == start + durations[i])

        starts.append(start)
        ends.append(end)

    ############################################################ cpu constraints

    for t in range(horizon):
        cpu_terms = []

        for i in range(n_apps):
            duration = durations[i]

            for s in range(horizon - duration + 1):
                local = t - s

                if 0 <= local < duration:
                    demand = int(round(cpu_profiles[i][local]))

                    if demand > 0:
                        cpu_terms.append(demand * x[i, s])

        if cpu_terms:
            model.Add(sum(cpu_terms) <= MAX_CPU)

    ######################################################### memory constraints

    for t in range(horizon):
        mem_terms = []

        for i in range(n_apps):
            duration = durations[i]

            for s in range(horizon - duration + 1):
                local = t - s

                if 0 <= local < duration:
                    demand = int(round(mem_profiles[i][local]))

                    if demand > 0:
                        mem_terms.append(demand * x[i, s])

        if mem_terms:
            model.Add(sum(mem_terms) <= MAX_MEMORY)

    ################################################################### makespan
    makespan = model.NewIntVar(0, horizon * 2, "makespan")
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)

    return model, starts, makespan

########################################################################### main

if __name__ == "__main__":
    # load profiles
    cpu_profiles = [
        np.loadtxt("./data/cpu_d0.csv"),
        np.loadtxt("./data/cpu_d1.csv"),
        np.loadtxt("./data/cpu_d2.csv"),
    ]

    mem_profiles = [
        np.loadtxt("./data/memory_d0.csv"),
        np.loadtxt("./data/memory_d1.csv"),
        np.loadtxt("./data/memory_d2.csv"),
    ]

    # run schedule
    model, starts, makespan = schedule_direct_cp_sat(
        cpu_profiles,
        mem_profiles,
    )

    # run solver
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    # print solution if any
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\nSolution")
        for i, start in enumerate(starts):
            print(
                f"Application {i}: "
                f"start={solver.Value(start)} "
                f"end={solver.Value(start)+len(cpu_profiles[i])}"
            )
        print(f"\nMakespan = {solver.Value(makespan)}")
    else:
        print("No solution found.")
