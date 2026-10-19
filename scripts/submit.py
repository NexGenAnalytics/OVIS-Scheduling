#!/usr/bin/env python3

"""Submit one Slurm job per entry in the LAMMPS run plan and record them in a manifest."""

import argparse
import datetime as dt
import fnmatch
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ACCOUNT = "fy250066"
PARTITION = "batch"
METRICS = ["mem_active_kb", "cpu_cores_used"]

REPO = Path(__file__).resolve().parents[1]
DECKS_DIR = REPO / "data" / "input-decks" / "lammps"
PLAN = DECKS_DIR / "plan.yaml"
MANIFESTS_DIR = REPO / "data" / "manifests"
RUNNER = REPO / "scripts" / "run.sh"
RECORDER = REPO / "scripts" / "record_particles.py"
ROOT = Path("/gpfs/cwschil/Scheduling")

PLAN_KEYS = {"build", "nodes", "ntasks_per_node", "omp_num_threads", "time"}


def hours(walltime):
    h, m, s = (int(part) for part in walltime.split(":"))
    return h + m / 60 + s / 3600


def load_plan(path):
    plan = yaml.safe_load(path.read_text()) or {}
    return plan.get("defaults") or {}, plan.get("decks") or {}


def resolve_jobs(defaults, overrides, problems):
    """One resolved run config per (deck, plan entry); decks absent from the plan use the defaults."""
    unknown = sorted(set(overrides) - set(problems))
    if unknown:
        raise ValueError(f"plan lists decks that do not exist: {', '.join(unknown)}")

    jobs = []
    for problem in problems:
        entries = overrides.get(problem) or [{}]
        for entry in entries if isinstance(entries, list) else [entries]:
            bad = set(entry) - PLAN_KEYS
            if bad:
                raise ValueError(f"{problem}: unknown plan keys {', '.join(sorted(bad))}")
            job = {**defaults, **entry, "problem": problem, "input_deck": f"in.{problem}"}
            job["ntasks"] = job["nodes"] * job["ntasks_per_node"]
            if isinstance(job["time"], int):  # YAML reads unquoted 12:00:00 as base-60 seconds
                job["time"] = "{}:{:02d}:{:02d}".format(job["time"] // 3600, job["time"] // 60 % 60, job["time"] % 60)
            jobs.append(job)
    return jobs


def select(jobs, patterns, max_time):
    if patterns:
        jobs = [job for job in jobs if any(fnmatch.fnmatch(job["problem"], p) for p in patterns)]
    if max_time:
        jobs = [job for job in jobs if hours(job["time"]) <= hours(max_time)]
    return jobs


def run_dir(job, root, stamp):
    tag = f"{stamp}_{job['nodes']}n{job['ntasks_per_node']}p{job['omp_num_threads']}t"
    return root / "runs" / job["build"] / job["problem"] / tag


def sbatch_cmd(job, rundir, root):
    lmp = root / job["build"] / "install" / "bin" / "lmp"
    return [
        "sbatch", "--parsable",
        "--account", ACCOUNT,
        "--partition", PARTITION,
        "--job-name", job["problem"],
        "--nodes", str(job["nodes"]),
        "--ntasks-per-node", str(job["ntasks_per_node"]),
        "--cpus-per-task", str(job["omp_num_threads"]),
        "--time", job["time"],
        "--chdir", str(rundir),
        "--output", "slurm-%j.out",
        "--error", "slurm-%j.err",
        "--export", f"ALL,LMP={lmp},DECK={job['input_deck']}",
        str(RUNNER),
    ]


def submit(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    job_id = result.stdout.strip().split(";")[0]
    if result.returncode != 0 or not job_id.isdigit():
        raise RuntimeError(f"sbatch failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    return int(job_id)


def collector_cmd(manifest, job_ids, root):
    """Job that records particle counts into the manifest once every run has ended."""
    return [
        "sbatch", "--parsable",
        "--account", ACCOUNT,
        "--partition", PARTITION,
        "--job-name", "record_particles",
        "--nodes", "1",
        "--ntasks", "1",
        "--time", "00:15:00",
        "--dependency", "afterany:" + ":".join(str(job_id) for job_id in job_ids),
        "--chdir", str(root / "runs"),
        "--output", "record_particles-%j.out",
        "--wrap", f"{sys.executable} {RECORDER} {manifest}",
    ]


def write_manifest(path, jobs, metrics=METRICS):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"metrics": metrics, "jobs": jobs}, sort_keys=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Submit LAMMPS jobs from the run plan and write a query manifest.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("-p", "--problem", action="append", help="Deck name or glob to submit (repeatable). Default: all.")
    parser.add_argument("--max-time", help="Only submit jobs whose planned walltime is at most HH:MM:SS.")
    parser.add_argument("--dry-run", action="store_true", help="Print the sbatch commands without submitting.")
    parser.add_argument("--root", type=Path, default=ROOT, help=f"Scheduling root on the cluster (default: {ROOT}).")
    return parser.parse_args()


def main():
    args = parse_args()
    problems = sorted(p.name[3:] for p in DECKS_DIR.glob("in.*"))
    jobs = select(resolve_jobs(*load_plan(PLAN), problems), args.problem, args.max_time)
    if not jobs:
        sys.exit("No jobs match.")

    for build in {job["build"] for job in jobs}:
        lmp = args.root / build / "install" / "bin" / "lmp"
        if not args.dry_run and not lmp.exists():
            sys.exit(f"Missing LAMMPS binary: {lmp}")

    node_hours = sum(job["nodes"] * hours(job["time"]) for job in jobs)
    prompt = f"You're about to submit {len(jobs)} jobs ({node_hours:.1f} node-hours requested). Continue? [y/N]: "
    if not args.yes and input(prompt).strip().lower() not in ("y", "yes"):
        sys.exit("Exiting.")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = MANIFESTS_DIR / f"manifest_{stamp}.yaml"
    submitted = {}

    for job in jobs:
        deck = DECKS_DIR / job["input_deck"]
        rundir = run_dir(job, args.root, stamp)
        cmd = sbatch_cmd(job, rundir, args.root)
        if args.dry_run:
            print(" ".join(cmd))
            submitted[len(submitted)] = job  # placeholder IDs so the collector command can be shown
            continue

        rundir.mkdir(parents=True, exist_ok=True)
        shutil.copy(deck, rundir)
        job_id = submit(cmd)
        submitted[job_id] = {**job, "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
                             "run_dir": str(rundir), "submitted": dt.datetime.now().isoformat(timespec="seconds")}
        # Rewrite after every job so a failure part-way through keeps the IDs already submitted.
        write_manifest(manifest, submitted)
        print(f"Submitted {job['problem']} ({job['nodes']}x{job['ntasks_per_node']}x{job['omp_num_threads']}): {job_id}")

    if not submitted:
        return
    collector = collector_cmd(manifest, submitted, args.root)
    if args.dry_run:
        print(" ".join(collector))
        return
    print(f"\nWrote manifest: {manifest}")
    print(f"Submitted particle-count recorder: {submit(collector)}. Commit the manifest after it completes.")


if __name__ == "__main__":
    main()
