"""This Python script generates sample memory and CPU utilization data for testing purposes."""
import os
import numpy as np
import matplotlib.pyplot as plt

# Constants
SAMPLE_FREQUENCY = 1     # samples per minute
DURATION = 80            # minutes (1 hr 20 minutes)
NOISE_LEVEL = 0.5        # standard deviation of Gaussian noise
CPU_NOISE_LEVEL = 2.0    # std dev of Gaussian noise for CPU (%)
CPU_SCALE_FACTOR = 0.45  # Scale CPU utilization

# Key features
SPIKES = [20, 40, 50, 60, 70]
SYNCS =  [(0, 12), (21, 23), (26, 31)]

def addSpike(curve, idx, height=None):
    """Adds a spike with ramp-up and ramp-down to the curve at the specified index."""
    spike_height = abs(np.random.normal(40, 5)) if height is None else height - curve[idx]

    # Add the main spike
    curve[idx] += spike_height

    # Ramp up: random percentage (30-60%) of the spike height
    if idx >= 1:
        ramp_up_pct = np.random.uniform(0.3, 0.6)
        curve[idx - 1] += spike_height * ramp_up_pct

    # Ramp down: different random percentage (30-60%) of the spike height
    if idx < len(curve) - 1:
        ramp_down_pct = np.random.uniform(0.3, 0.6)
        curve[idx + 1] += spike_height * ramp_down_pct

def generateMemoryBaseline():
    """Generates a baseline curve with noise."""
    t = np.arange(0, DURATION, SAMPLE_FREQUENCY)
    baseline = 5 * np.sqrt(0.9 * t)
    baseline += np.random.normal(0, NOISE_LEVEL, len(baseline))

    # Add spikes at defined locations
    for spike in SPIKES:
        addSpike(baseline, spike)

    # Big one at the end
    addSpike(baseline, len(baseline) - 1, height=115)

    return t, baseline

def generateAdditionalMemoryData(scales=[0.75, 0.4], baseline=None):
    """Generates additional curves based on scaling down the baseline."""

    if baseline is None:
        return

    datasets = []

    # Scale down the baseline for mean and min plots
    for scale in scales:
        memory = baseline * scale

        # Synchronize key regions to the baseline (non-spike regions)
        for i, j in SYNCS:
            region = slice(i, j)
            memory[region] = baseline[region]

        # Add run-up and spike synchronization
        # Spikes occur at spike + 1
        for spike in SPIKES:
            addSpike(memory, spike + 1)

        # Add the big final spike
        addSpike(memory, len(memory) - 1, height=baseline[-1])

        datasets.append(memory)

    return datasets

def generateCPUBaseline(t):
    """Generates a CPU utilization baseline (%) with high plateaus and brief dips/spikes."""
    n = len(t)
    cpu = np.zeros(n, dtype=float)

    # Define plateaus (start, end, level as fraction of 100) - scaled down by 20%
    plateaus = [
        (5, 20, 0.85 * CPU_SCALE_FACTOR),
        (20, 35, 0.88 * CPU_SCALE_FACTOR),
        (35, 45, 0.95 * CPU_SCALE_FACTOR),
        (45, 55, 0.60 * CPU_SCALE_FACTOR),  # I/O dip
        (55, 70, 0.92 * CPU_SCALE_FACTOR),
    ]

    # Startup ramp from 10% to first plateau
    ramp_end = plateaus[0][0]
    for i in range(0, min(ramp_end, n)):
        cpu[i] = (0.10 * CPU_SCALE_FACTOR) + (plateaus[0][2] - 0.10 * CPU_SCALE_FACTOR) * (i / max(ramp_end, 1))

    # Apply plateaus
    for start, end, level in plateaus:
        s = max(0, start)
        e = min(n, end)
        if s < e:
            cpu[s:e] = level

    # Cooldown tail
    for i in range(min(70, n), n):
        cpu[i] = max(0.5 * CPU_SCALE_FACTOR, cpu[i-1] - 0.02)

    # Brief spikes around defined memory SPIKES - also scaled down
    for s in SPIKES:
        if 0 <= s < n:
            cpu[s] = min(1.0, max(cpu[s], 0.98 * CPU_SCALE_FACTOR))
        if 0 <= s+1 < n:
            cpu[s+1] = min(1.0, max(cpu[s+1], 0.97 * CPU_SCALE_FACTOR))
        # small pre-spike dip
        if 0 <= s-1 < n:
            cpu[s-1] = max(0.5 * CPU_SCALE_FACTOR, cpu[s-1] - 0.10)

    # Add light noise and scale to percent
    cpu = 100.0 * cpu + np.random.normal(0, CPU_NOISE_LEVEL, n)
    cpu = np.clip(cpu, 0, 100)

    return cpu

    return cpu

def generateCPUAdditionalData(scales=[0.9, 0.7], baseline=None):
    """Generates additional CPU curves by scaling and synchronizing regions to the baseline."""

    if baseline is None:
        return []

    datasets = []
    n = len(baseline)

    for scale in scales:
        cpu = baseline * scale

        # Synchronize key regions to the baseline (keep high plateaus aligned)
        for i, j in SYNCS:
            s = max(0, i)
            e = min(n, j)
            if s < e:
                cpu[s:e] = baseline[s:e]

        # Add brief bursts around memory spikes (already scaled since baseline is scaled)
        for s in SPIKES:
            if 0 <= s < n:
                cpu[s] = min(100.0, max(cpu[s], 0.98 * CPU_SCALE_FACTOR * 100.0))
            if 0 <= s+1 < n:
                cpu[s+1] = min(100.0, max(cpu[s+1], 0.97 * CPU_SCALE_FACTOR * 100.0))

        # Light noise and clipping
        cpu += np.random.normal(0, CPU_NOISE_LEVEL, n)
        cpu = np.clip(cpu, 0, 100)

        datasets.append(cpu)

    return datasets

def generateData():
    """Generates sample memory and CPU data and saves to disk."""

    t, memory_baseline = generateMemoryBaseline()

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    timesteps_path = os.path.join(data_dir, "t.csv")
    np.savetxt(timesteps_path, t, delimiter=",")

    additional_memory_datasets = generateAdditionalMemoryData(baseline=memory_baseline)

    # Save baseline
    baseline_path = os.path.join(data_dir, 'memory_d0.csv')
    np.savetxt(baseline_path, memory_baseline, delimiter=',')
    print(f"Wrote baseline memory data to {baseline_path}")

    # Save additional datasets
    for i, memory_dataset in enumerate(additional_memory_datasets):
        dataset_path = os.path.join(data_dir, f'memory_d{i+1}.csv')
        np.savetxt(dataset_path, memory_dataset, delimiter=',')
        print(f"Wrote memory dataset to {dataset_path}")

    memory_list = [memory_baseline] + additional_memory_datasets
    memory_dict = {f'd{i}': dataset for i, dataset in enumerate(memory_list)}

    # Generate CPU datasets
    cpu_baseline = generateCPUBaseline(t)
    cpu_additional = generateCPUAdditionalData(baseline=cpu_baseline)

    # Save CPU datasets
    cpu_datasets = [cpu_baseline] + cpu_additional
    for i, cpu in enumerate(cpu_datasets):
        cpu_path = os.path.join(data_dir, f'cpu_d{i}.csv')
        np.savetxt(cpu_path, cpu, delimiter=',')
        print(f"Wrote CPU dataset to {cpu_path}")

    cpu_dict = {f'd{i}': dataset for i, dataset in enumerate(cpu_datasets)}

    data = {
        "memory": memory_dict,
        "cpu": cpu_dict
    }

    return t, data

def plotAndSave(times, datasets: dict, stat: str, withSum: bool = False):
    """Plots the generated memory data."""

    plts_dir = os.path.join(os.path.dirname(__file__), 'plots', 'data')
    os.makedirs(plts_dir, exist_ok=True)

    upper_stat = stat.capitalize()

    # First plot: individual datasets only
    plt.figure(figsize=(10, 4))

    for i, dataset in datasets.items():
        plt.plot(times, dataset, label=f"Dataset {i}")

    if withSum:
        summed = np.sum([dataset for dataset in datasets.values()], axis=0)
        plt.plot(times, summed, color='black', linestyle=':', linewidth=2.5, label=f'Summed {upper_stat}')

    stat_unit = "GB" if stat.lower() == "memory" else "%"
    with_sum_str = " (with Sum)" if withSum else ""
    plt.title(f'Synthetic Active {upper_stat} Across Job{with_sum_str}')
    plt.xlabel('Time (min)')
    plt.ylabel(f'{upper_stat} ({stat_unit})')
    plt.legend()
    plt.grid()

    save_sum_str = "Summed" if withSum else ""
    save_path = os.path.join(plts_dir, f'synthetic{upper_stat}{save_sum_str}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Wrote plot to {save_path}")

def main():

    t, data = generateData()
    for stat, dataset in data.items():
        plotAndSave(t, dataset, stat, withSum=False)
        plotAndSave(t, dataset, stat, withSum=True)

if __name__ == "__main__":
    main()
