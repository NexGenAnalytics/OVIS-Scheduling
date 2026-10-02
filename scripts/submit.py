#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import datetime as dt

import yaml

# -----------------------------
# Global configuration
# -----------------------------

# Configurable
METRICS = ["Active", "CPU"]

# SLURM
ACCOUNT   = "fy250066"
PARTITION = "batch"
NODES     = 1          # this can change
NTASKS    = 4          # this can change
TIME      = "01:00:00" # this can change

# OpenMP
OMP_NUM_THREADS = 1

# Directories
PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIRECTORY    = os.path.join(PROJECT_DIRECTORY, "data")

DECKS_DIR     = os.path.join(DATA_DIRECTORY, "input-decks", "lammps")
OUTPUT_DIR    = os.path.join(DATA_DIRECTORY, "ldms")
MANIFESTS_DIR = os.path.join()

# Derived
RUNNER   = # pass the path to this
TODAY    = dt.datetime.now().strftime("%Y%m%d")
MANIFEST = os.path.join(MANIFESTS_DIR, f"manifest_{TODAY}.yaml")

# -----------------------------
# Helpers
# -----------------------------

def problem_from_deck(deck_path):
    name = os.path.basename(deck_path)

    if not name.startswith("in."):
        raise ValueError(f"Deck does not start with 'in.': {deck_path}")

    return name[len("in."):]


def find_decks(decks_dir, problems=None):
    if problems:
        decks = []
        for problem in problems:
            deck = os.path.join(decks_dir, f"in.{problem}")
            if not os.path.exists(deck):
                raise FileNotFoundError(f"Missing deck: {deck}")
            decks.append(deck)
        return decks

    decks = []
    for name in sorted(os.listdir(decks_dir)):
        if name.startswith("in."):
            decks.append(os.path.join(decks_dir, name))

    return decks


def read_nodelist_file(path, max_nodes=None):
    nodes = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Accept either:
            #   mz1330
            # or:
            #   mz1330 5101330
            node = line.split()[0]

            if node.startswith("mz"):
                nodes.append(node)

            if max_nodes is not None and len(nodes) >= max_nodes:
                break

    if not nodes:
        raise RuntimeError(f"No nodes found in {path}")

    return ",".join(nodes)


def submit_job(args, problem):
    job_output_dir = os.path.join(OUTPUT_DIR, problem)
    os.makedirs(job_output_dir, exist_ok=True)

    export_vars = [
        "ALL",
        f"BUILD={args.build}",
        f"PROBLEM={problem}",
        f"BASE_DIR={PROJECT_DIRECTORY}",
        f"NTASKS={NTASKS}",
        f"OMP_NUM_THREADS={OMP_NUM_THREADS}",
    ]

    cmd = [
        "sbatch",
        "--parsable",
        "--account", ACCOUNT,
        "--partition", PARTITION,
        "--job-name", problem,
        "--nodes", str(NODES),
        "--ntasks", str(NTASKS),
        "--time", TIME,
        "--export", ",".join(export_vars),
        "--output", os.path.join(job_output_dir, f"{problem}-%j.out"),
        "--error", os.path.join(job_output_dir, f"{problem}-%j.err")
    ]

    cmd.append(RUNNER)

    print("Submitting:")
    print("    " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if result.returncode != 0:
        print("sbatch failed")
        print("Return code:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError("sbatch submission failed")

    job_id = result.stdout.strip().split(";")[0]

    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")

    return int(job_id)


def write_manifest(path, metrics, jobs):
    manifest = {
        "metrics": metrics,
        "jobs": jobs,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"\nWrote manifest: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit LAMMPS jobs and write LDMS query YAML manifest."
    )

    parser.add_argument(
        "--build", "-b",
        default="cpu_mpi_omp",
        help="LAMMPS build name.",
    )

    parser.add_argument(
        "--problem", "-p",
        action="append",
        dest="problems",
        help="Problem name to submit. Can be used multiple times. Default: all decks.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    decks = find_decks(DECKS_DIR, args.problems)

    if not decks:
        raise RuntimeError(f"No decks found in {DECKS_DIR}")

    confirmation = input(f"This will schedule {len(decks)} jobs with slurm. Continue? [y]/n: ")
    if confirmation.strip().lower() in ["n", "no"]:
        print("Exiting.")
        sys.exit()

    jobs = {}

    for deck in decks:
        problem = problem_from_deck(deck)
        job_id = submit_job(args, problem)

        jobs[job_id] = {
            "build": args.build,
            "problem": problem,
            "input_deck": os.path.basename(deck),
            "ntasks": NTASKS,
            "nodes": NODES,
            "omp_num_threads": OMP_NUM_THREADS,
            "time": TIME,
        }

        print(f"Submitted {problem}: job_id={job_id}")

    write_manifest(MANIFEST, METRICS, jobs)

if __name__ == "__main__":
    main()

