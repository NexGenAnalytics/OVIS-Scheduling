#!/usr/bin/env python3

"""Record each job's actual particle counts from its log.lammps into the manifest."""

import argparse
import re
import sys
from pathlib import Path

import yaml

from submit import write_manifest

CREATED = re.compile(r"^Created (\d+) atoms")  # create_atoms adds to the total
TOTALS = [
    re.compile(r"^Deleted \d+ atoms, new total = (\d+)"),  # delete_atoms
    re.compile(r"^\s+(\d+) atoms$"),  # replicate / read_data
]
LOOP = re.compile(r"^Loop time of .* with (\d+) atoms")


def parse_counts(log_text):
    """(peak, final, complete) atom counts while runs execute; counts are None if no atoms were made."""
    current, simulated = 0, []
    for line in log_text.splitlines():
        created = CREATED.match(line)
        total = next((m for m in (p.match(line) for p in TOTALS) if m), None)
        loop = LOOP.match(line)
        if created:
            current += int(created.group(1))
        elif total:
            current = int(total.group(1))
        elif loop:
            # A run starts at the setup total and ends at the loop total (deposit grows, evaporate shrinks).
            simulated += [current, int(loop.group(1))]
            current = simulated[-1]
    complete = "Total wall time" in log_text
    if not complete and current:
        simulated.append(current)  # killed mid-run: the in-progress run started with the current total
    peak = max(simulated) if simulated else None
    final = simulated[-1] if simulated else None
    return peak, final, complete


def record(manifest_path):
    manifest = yaml.safe_load(manifest_path.read_text())
    print(f"{manifest_path}")
    for job_id, job in manifest["jobs"].items():
        log = Path(job["run_dir"]) / "log.lammps" if job.get("run_dir") else None
        if log is None or not log.exists():
            print(f"  {job_id} {job['problem']}: skipped (no {'log' if log else 'run_dir'})")
            continue
        peak, final, complete = parse_counts(log.read_text(errors="replace"))
        job["particle_count"], job["particle_count_final"] = peak, final
        status = "ok" if complete else "WARNING: run did not finish"
        print(f"  {job_id} {job['problem']}: peak={peak} final={final} {status}")
    write_manifest(manifest_path, manifest["jobs"], manifest["metrics"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    for path in parser.parse_args().manifests:
        if not path.exists():
            sys.exit(f"Missing manifest: {path}")
        record(path)


if __name__ == "__main__":
    main()
