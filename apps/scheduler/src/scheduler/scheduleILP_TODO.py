"""Single-stage ILP: minimize makespan with memory and CPU constraints."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pulp

MAX_MEMORY = 220  # GB
MAX_CPU = 125     # percent
BASELINE = "d0"   # Dataset that must start at time 0

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'ilp')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load memory datasets
MEMORY_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith('memory_') and f.endswith('.csv')]
MEMORY_DATASETS = {
    f.split(".")[0].split("/")[-1].replace('memory_', ''): np.loadtxt(f, delimiter=',') for f in MEMORY_FILES
}

# Load CPU datasets
CPU_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith('cpu_') and f.endswith('.csv')]
CPU_DATASETS = {
    f.split(".")[0].split("/")[-1].replace('cpu_', ''): np.loadtxt(f, delimiter=',') for f in CPU_FILES
}

# Load timesteps
TIMESTEPS = np.loadtxt(os.path.join(DATA_DIR, 't.csv'), delimiter=',')

def plotSchedule(timesteps, memory_curves, cpu_curves, start_times):
    """Plots the scheduled memory and CPU curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    max_time = int(max(start_times.values())) + max(len(m) for m in memory_curves.values())
    global_time = np.arange(max_time)

    # Define consistent colors for each dataset
    colors = {'d0': 'C0', 'd1': 'C1', 'd2': 'C2'}

    # Plot memory curves
    for curve_name, curve_data in memory_curves.items():
        start_idx = int(start_times[curve_name])
        shifted_curve = np.zeros(max_time)
        shifted_curve[start_idx:start_idx + len(curve_data)] = curve_data
        ax1.plot(global_time, shifted_curve, label=f"Dataset {curve_name}",
                linewidth=2, color=colors.get(curve_name, None))

    # Plot summed memory curve
    summed_mem = np.zeros(max_time)
    for curve_name, curve_data in memory_curves.items():
        start_idx = int(start_times[curve_name])
        summed_mem[start_idx:start_idx + len(curve_data)] += curve_data

    ax1.plot(global_time, summed_mem, color='black', linestyle=':', linewidth=2.5, label='Summed Memory')
    ax1.axhline(y=MAX_MEMORY, color='r', linestyle='--', linewidth=2, label='Memory Ceiling')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Memory (GB)')
    ax1.set_title(f'ILP Schedule: Memory (Max: {MAX_MEMORY} GB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot CPU curves
    for curve_name, curve_data in cpu_curves.items():
        start_idx = int(start_times[curve_name])
        shifted_curve = np.zeros(max_time)
        shifted_curve[start_idx:start_idx + len(curve_data)] = curve_data
        ax2.plot(global_time, shifted_curve, label=f"Dataset {curve_name}",
                linewidth=2, color=colors.get(curve_name, None))

    # Plot summed CPU curve
    summed_cpu = np.zeros(max_time)
    for curve_name, curve_data in cpu_curves.items():
        start_idx = int(start_times[curve_name])
        summed_cpu[start_idx:start_idx + len(curve_data)] += curve_data

    ax2.plot(global_time, summed_cpu, color='black', linestyle=':', linewidth=2.5, label='Summed CPU')
    ax2.axhline(y=MAX_CPU, color='r', linestyle='--', linewidth=2, label='CPU Ceiling')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('CPU Utilization (%)')
    ax2.set_title(f'ILP Schedule: CPU (Max: {MAX_CPU}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'ilp_mem{MAX_MEMORY}_cpu{MAX_CPU}.png'), dpi=300)
    plt.close()
    print(f"Saved plot to {PLOTS_DIR}/ilp_mem{MAX_MEMORY}_cpu{MAX_CPU}.png")


def verifySchedule(memory_curves, cpu_curves, start_times):
    """Verify the schedule respects memory and CPU constraints."""
    max_time = int(max(start_times.values())) + max(len(m) for m in memory_curves.values())

    max_mem = 0
    max_cpu = 0
    mem_violations = 0
    cpu_violations = 0

    for t in range(max_time):
        mem_at_t = 0
        cpu_at_t = 0

        for curve, curve_data in memory_curves.items():
            start = int(start_times[curve])
            if start <= t < start + len(curve_data):
                mem_at_t += curve_data[t - start]

        for curve, curve_data in cpu_curves.items():
            start = int(start_times[curve])
            if start <= t < start + len(curve_data):
                cpu_at_t += curve_data[t - start]

        max_mem = max(max_mem, mem_at_t)
        max_cpu = max(max_cpu, cpu_at_t)

        if mem_at_t > MAX_MEMORY * 1.01:  # allow small tolerance for float noise
            mem_violations += 1
        if cpu_at_t > MAX_CPU * 1.01:
            cpu_violations += 1

    return max_mem, max_cpu, mem_violations, cpu_violations


def schedule():
    """Solve scheduling using makespan minimization with memory and CPU constraints."""
    print("Loading datasets...")
    timesteps = TIMESTEPS
    memory_curves = MEMORY_DATASETS
    cpu_curves = CPU_DATASETS

    print(f"Memory curves: {list(memory_curves.keys())}")
    print(f"CPU curves: {list(cpu_curves.keys())}")
    print(f"Curve lengths: {[len(m) for m in memory_curves.values()]}")

    # Time horizon: worst case is running everything back-to-back
    t_max = sum(len(m) for m in memory_curves.values())
    print(f"Time horizon (t_max): {t_max}")

    # Build model
    latest_start = {curve: t_max - len(mem) for curve, mem in memory_curves.items()}

    print("\nMinimizing makespan with memory and CPU constraints...")
    model = pulp.LpProblem("Memory_CPU_Scheduling", pulp.LpMinimize)

    # Start-time selector variables: y[curve, s] == 1 if curve starts at time s
    y = {}
    for curve, mem in memory_curves.items():
        for s in range(latest_start[curve] + 1):
            y[(curve, s)] = pulp.LpVariable(f"y_{curve}_{s}", cat="Binary")

    # Each curve must start exactly once
    for curve in memory_curves:
        model += pulp.lpSum(y[(curve, s)] for s in range(latest_start[curve] + 1)) == 1

    # Baseline must start at time 0
    if BASELINE in memory_curves:
        model += y[(BASELINE, 0)] == 1

    # Makespan variable
    makespan = pulp.LpVariable("makespan", lowBound=0, upBound=t_max, cat="Integer")

    # Link makespan to each curve's end time: makespan >= (s + L) * y[curve, s]
    for curve, mem in memory_curves.items():
        L = len(mem)
        for s in range(latest_start[curve] + 1):
            model += makespan >= (s + L) * y[(curve, s)]

    # Memory usage at each time t
    for t in range(t_max):
        mem_t_terms = []
        for curve, mem in memory_curves.items():
            L = len(mem)
            for s in range(latest_start[curve] + 1):
                idx = t - s
                if 0 <= idx < L:
                    mem_t_terms.append(mem[idx] * y[(curve, s)])
        if mem_t_terms:
            total_mem_t = pulp.lpSum(mem_t_terms)
            model += total_mem_t <= MAX_MEMORY

    # CPU usage at each time t
    for t in range(t_max):
        cpu_t_terms = []
        for curve, cpu in cpu_curves.items():
            L = len(cpu)
            for s in range(latest_start[curve] + 1):
                idx = t - s
                if 0 <= idx < L:
                    cpu_t_terms.append(cpu[idx] * y[(curve, s)])
        if cpu_t_terms:
            total_cpu_t = pulp.lpSum(cpu_t_terms)
            model += total_cpu_t <= MAX_CPU

    # Objective: minimize makespan only
    model += makespan

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=180)
    model.solve(solver)

    if pulp.LpStatus[model.status] != "Optimal":
        print(f"Solver status: {pulp.LpStatus[model.status]}")

    # Extract results
    start_times = {}
    for curve in memory_curves:
        for s in range(latest_start[curve] + 1):
            if y[(curve, s)].value() and y[(curve, s)].value() > 0.5:
                start_times[curve] = s
                break

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    for curve, start_time in sorted(start_times.items()):
        end_time = start_time + len(memory_curves[curve])
        print(f"{curve}: starts at minute {start_time}, ends at minute {end_time}")

    if makespan.value() is not None:
        print(f"\nTotal makespan: {int(makespan.value())} minutes")

    # Verify memory and CPU constraints
    print("\nVerifying solution...")
    max_mem, max_cpu, mem_violations, cpu_violations = verifySchedule(memory_curves, cpu_curves, start_times)
    print(f"Maximum memory used: {max_mem:.2f} GB (limit: {MAX_MEMORY} GB)")
    print(f"Maximum CPU used: {max_cpu:.2f}% (limit: {MAX_CPU}%)")

    if mem_violations > 0:
        print(f"WARNING: {mem_violations} timesteps violate memory constraint")
    else:
        print("✓ Memory constraint satisfied!")

    if cpu_violations > 0:
        print(f"WARNING: {cpu_violations} timesteps violate CPU constraint")
    else:
        print("✓ CPU constraint satisfied!")

    # Plot the result
    print("\nGenerating plot...")
    plotSchedule(timesteps, memory_curves, cpu_curves, start_times)

    return start_times


if __name__ == "__main__":
    schedule()
