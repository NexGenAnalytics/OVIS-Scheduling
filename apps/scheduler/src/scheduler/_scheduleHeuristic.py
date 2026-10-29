"""This script simulates the scheduling logic of the datasets in the data directory."""

import os
import numpy as np
import matplotlib.pyplot as plt

MAX_MEMORY = 220      # GB
BASELINE_LABEL = "d0" # or None, if all curves are flexible

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'heuristic')
os.makedirs(PLOTS_DIR, exist_ok=True)

MEMORY_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith('memory_') and f.endswith('.csv')]
MEMORY_DATASETS = {
    f.split(".")[0].split("/")[-1].replace('memory_', ''): np.loadtxt(f, delimiter=',') for f in MEMORY_FILES
}

TIMESTEPS = np.loadtxt(os.path.join(DATA_DIR, 't.csv'), delimiter=',')

def clearPlots():
    """Clears existing plots in the plots directory."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        return

    for f in os.listdir(PLOTS_DIR):
        if f.endswith('.png'):
            os.remove(os.path.join(PLOTS_DIR, f))

    print(f"Cleared existing plots in {PLOTS_DIR}")

def plotShift(timesteps, data, ceiling, iteration: int = 0):
    """Plots the current state of the data during shifting."""

    plt.figure(figsize=(10, 4))

    for i, dataset in data.items():
        plt.plot(timesteps, dataset, label=f"Dataset {i}")

    plt.axhline(y=ceiling, color='r', linestyle='--', label='Ceiling')

    summed_curve = sumAllCurves(data)
    plt.plot(timesteps, summed_curve, color='k', linestyle=':', label='Summed Curve')

    plt.title(f'Scheduled Active Memory Across Job')
    plt.xlabel('Time (min)')
    plt.ylabel('Memory (GB)')
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(PLOTS_DIR, f'heuristic_{MAX_MEMORY}_{iteration}.png'), dpi=300)
    plt.close()

    print(f"Wrote scheduling step plot to {PLOTS_DIR}")

    return

def sumAllCurves(data: dict):
    """Find the sum of all provided curves at each timestep."""
    summed_curve = np.sum([data[key] for key in data], axis=0)
    return summed_curve

def findHighestNonBaselineCurveAtPoint(data: dict, idx: int, baseline_label: str):
    max_val = -np.inf
    max_label = None
    for label, dataset in data.items():
        if label == baseline_label:
            continue
        if dataset[idx] > max_val:
            max_val = dataset[idx]
            max_label = label
    return max_label

"""
Need to rework this function to allow shifting the curves backwards and forwards
in time, as long as they never shift to BEFORE the baseline curve starts. The
assumption is always that the baseline curve starts first.

We also need to optimize the algorithm to keep the total timeline as small
as possible. By allowing us to add zeroes on the end, we open up to infinitely
shifting curves inch by inch forward in time.
"""
def findLowestPointAfterIndex(data: dict, lb_idx: int):

    # Handle case where the peak is at the end of the data
    if lb_idx >= len(sumAllCurves(data)) - 1:
        return lb_idx + 10, 0

    summed_curve = sumAllCurves(data)[lb_idx:]
    idx = np.argmin(summed_curve)
    min_val = summed_curve[idx]

    return idx + lb_idx, min_val

def findHighestPoint(data: dict):
    summed_curve = sumAllCurves(data)
    idx = np.argmax(summed_curve)
    max_val = summed_curve[idx]
    return idx, max_val

def shiftData(timesteps, data: dict, shift: tuple):
    # Determine which dataset to shift, and by how much
    shift_dataset = shift[0]
    shift_amount  = shift[1]

    # Perform the shift
    zeros = np.zeros(shift_amount)
    data[shift_dataset] = np.concatenate((zeros, data[shift_dataset]))

    # Add more timesteps to accomodate shift
    timesteps = np.concatenate((timesteps, np.arange(timesteps[-1]+1, timesteps[-1]+1+shift_amount)))

    # Append zeros to all other datasets to accomodate shift
    for other_dataset in data:
        if other_dataset != shift_dataset:
            data[other_dataset] = np.concatenate((data[other_dataset], np.zeros(shift_amount)))

    return timesteps, data

def recursiveMax(timesteps, data, ceiling, baseline: str = None, shift: tuple = None, n=1):
    # The baseline dataset will never be shifted
    baseline = baseline if baseline is not None else list(data.keys())[0]

    # If any of the curves pass the ceiling on their own, the problem is impossible
    for label, dataset in data.items():
        if np.max(dataset) >= ceiling:
            print(f"Dataset {label} exceeds ceiling on its own (max: {np.max(dataset):.2f} GB). Scheduling impossible.")
            return n, timesteps, data

    while n < 25:

        # Shift the data
        if shift is not None:
            timesteps, data = shiftData(timesteps, data, shift)

        plotShift(timesteps, data, ceiling, n)

        # Find the maximum QOI (active memory, for now) of each curve
        peak_idx, max_val = findHighestPoint(data)

        if max_val >= ceiling:

            # Find where the summed curves are lowest, and move the peak of the highest curve there
            low_idx, min_val = findLowestPointAfterIndex(data, peak_idx)
            print(f"-- Found lowest index: {low_idx}")
            lbl = findHighestNonBaselineCurveAtPoint(data, peak_idx, baseline)

            print(f"-- The highest curve at the summed peak is on dataset {lbl} (value: {data[lbl][peak_idx]:.2f}, t: {timesteps[peak_idx]} min) --")

            # Shift the highest curve to the lowest point
            print(f"-- Shifting dataset {lbl} from index {peak_idx} (summed value: {max_val:.2f}) to index {low_idx} (summedvalue: {min_val:.2f}) --")
            shift = (lbl, low_idx - peak_idx)

            # Iterate the counter
            n += 1

            # Run it back
            return recursiveMax(timesteps, data, ceiling, baseline, shift, n)

        return n, timesteps, data

def checkResults(timesteps, data: dict):
    """Checks the results of the scheduling for any issues."""
    print("\nRESULTS:")
    starts = {  label: np.nonzero(dataset)[0][0] for label, dataset in data.items() }
    ends = { label: np.nonzero(dataset)[0][-1] for label, dataset in data.items() }

    for lbl in data.keys():
        start_time = timesteps[starts[lbl]]
        end_time = timesteps[ends[lbl]]
        print(f"  - Dataset {lbl} runs from {start_time} min to {end_time} min")

        other_ends = [timesteps[ends[other_lbl]] for other_lbl in data.keys() if other_lbl != lbl]
        if other_ends and start_time > max(other_ends):
            print(f"      WARNING: {lbl} starts after all other runs have finished (latest end: {max(other_ends)} min)")

def schedule():
    clearPlots()
    memory_curves = {label: data for label, data in MEMORY_DATASETS.items()}
    n_iterations, shifted_timesteps, shifted_memory_curves = recursiveMax(TIMESTEPS, memory_curves, MAX_MEMORY, BASELINE_LABEL)
    print(f"Scheduling complete after {n_iterations} iterations.")

    checkResults(shifted_timesteps, shifted_memory_curves)

if __name__ == "__main__":
    schedule()
