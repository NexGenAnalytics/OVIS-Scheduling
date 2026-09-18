#!/usr/bin/env python3

import os
import argparse
import traceback
from pathlib import Path

import yaml
import pandas as pd
from sosdb import Sos

# -----------------------------
# Global configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data", "ldms")

SOS_DATABASE = "/storage/manzano/sos/database"
SOS_CONFIG   = "/opt/ovis/dsos/config/manzano.conf"

MEM_SCHEMA = "meminfo_toss4"
CPU_SCHEMA = "procstat_96"

MERGE_TOLERANCE_SECONDS = 5.0

# -----------------------------
# Helpers
# -----------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_query(cont, sql):
    query = cont.query(1024 * 1024)
    query.select(sql)

    chunks = []
    df = query.next()

    while df is not None:
        chunks.append(df.copy(deep=True))
        df = query.next()

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def timestamp_to_seconds(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64") / 1.0e9
    return series.astype(float)


def get_run_dir(job_id, build, problem):
    return os.path.join(
        BASE_DIR,
        "runs",
        build,
        problem,
        str(job_id),
    )

# -----------------------------
# Metric fetchers
# -----------------------------

def fetch_active(cont, job_id):
    sql = f"""
select timestamp, job_id, component_id, Active
from {MEM_SCHEMA}
where job_id == {job_id}
order_by job_time_comp
"""

    df = fetch_query(cont, sql)

    if df.empty:
        return df

    df = df.copy()
    df["timestamp"]    = timestamp_to_seconds(df["timestamp"])
    df["job_id"]       = df["job_id"].astype(int)
    df["component_id"] = df["component_id"].astype(int)
    df["Active"]       = df["Active"].astype(float)

    df = df.sort_values(["component_id", "timestamp"])

    return df

def fetch_cpu(cont, job_id):
    sql = f"""
select timestamp, job_id, component_id, user, sys
from {CPU_SCHEMA}
where job_id == {job_id}
order_by job_time_comp
"""

    df = fetch_query(cont, sql)

    if df.empty:
        return df

    df = df.copy()
    df["timestamp"]    = timestamp_to_seconds(df["timestamp"])
    df["job_id"]       = df["job_id"].astype(int)
    df["component_id"] = df["component_id"].astype(int)
    df["user"]         = df["user"].astype(float)
    df["sys"]          = df["sys"].astype(float)

    df = df.sort_values(["component_id", "timestamp"])

    df["cpu_counter"]       = df["user"] + df["sys"]
    df["cpu_counter_delta"] = df.groupby("component_id")["cpu_counter"].diff()
    df["time_delta_s"]      = df.groupby("component_id")["timestamp"].diff()

    df["cpu_cores_used"] = (
        df["cpu_counter_delta"] / df["time_delta_s"] / 100.0
    )

    df["cpu_cores_used"] = df["cpu_cores_used"].fillna(0.0)

    return df[
        [
            "timestamp",
            "job_id",
            "component_id",
            "user",
            "sys",
            "cpu_counter_delta",
            "time_delta_s",
            "cpu_cores_used",
        ]
    ].copy()

# -----------------------------
# Combine metrics
# -----------------------------

def merge_active_cpu(active_df, cpu_df):
    merged_parts = []

    for component_id, active_part in active_df.groupby("component_id"):
        cpu_part = cpu_df[cpu_df["component_id"] == component_id].copy()

        if cpu_part.empty:
            continue

        active_part = active_part.sort_values("timestamp")
        cpu_part = cpu_part.sort_values("timestamp")

        merged = pd.merge_asof(
            active_part,
            cpu_part,
            on="timestamp",
            by=["job_id", "component_id"],
            direction="nearest",
            tolerance=MERGE_TOLERANCE_SECONDS,
        )

        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    return pd.concat(merged_parts, ignore_index=True)


def build_result_dataframe(metric_data):
    has_active = "Active" in metric_data and not metric_data["Active"].empty
    has_cpu = "CPU" in metric_data and not metric_data["CPU"].empty

    if has_active and has_cpu:
        res = merge_active_cpu(metric_data["Active"], metric_data["CPU"])
    elif has_active:
        res = metric_data["Active"].copy()
    elif has_cpu:
        res = metric_data["CPU"].copy()
    else:
        return pd.DataFrame()

    if res.empty:
        return res

    res = res.sort_values(["component_id", "timestamp"])
    res["time_rel_s"] = res["timestamp"] - res["timestamp"].min()

    return res

# -----------------------------
# Per-job processing
# -----------------------------

def process_job(cont, job_id, job_cfg, metrics):
    build   = job_cfg["build"]
    problem = job_cfg["problem"]

    run_dir = get_run_dir(job_id, build, problem)
    os.makedirs(run_dir, exist_ok=True)

    metric_data = {}

    if "Active" in metrics:
        metric_data["Active"] = fetch_active(cont, job_id)

    if "CPU" in metrics:
        metric_data["CPU"] = fetch_cpu(cont, job_id)

    unsupported = sorted(set(metrics) - {"Active", "CPU"})
    for metric in unsupported:
        print(f"WARNING: unsupported metric '{metric}' ignored for job {job_id}")

    res = build_result_dataframe(metric_data)

    if res.empty:
        raise RuntimeError(f"No LDMS data found for job_id={job_id}")

    out_csv = os.path.join(run_dir, "ldms_metrics.csv")
    res.to_csv(out_csv, index=False)

    return {
        "job_id": job_id,
        "build": build,
        "problem": problem,
        "run_dir": run_dir,
        "out_csv": out_csv,
        "rows": len(res),
        "status": "ok",
    }

# -----------------------------
# Main Driver
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch LDMS metrics for jobs listed in a YAML file."
    )

    parser.add_argument(
        "yaml_file",
        help="YAML file containing metrics and jobs.",
    )

    args = parser.parse_args()
    manifest_file = args.yaml_file

    cfg_id = manifest_file.split("/")[-1].split("_")[-1].split(".")[0]
    cfg = load_yaml(args.yaml_file)

    metrics = cfg.get("metrics", [])
    jobs = cfg.get("jobs", {})

    if not metrics:
        raise RuntimeError("YAML file must contain a non-empty 'metrics' list")

    if not jobs:
        raise RuntimeError("YAML file must contain a non-empty 'jobs' mapping")

    sess = Sos.Session(SOS_CONFIG)
    cont = sess.open(SOS_DATABASE)

    summaries = []

    for job_id_raw, job_cfg in jobs.items():
        job_id = int(job_id_raw)

        print(f"\n=== Processing job {job_id} ===")

        try:
            summary = process_job(cont, job_id, job_cfg, metrics)
            summaries.append(summary)

            print(f"Wrote {summary['out_csv']}")
            print(f"Rows: {summary['rows']}")

        except Exception as e:
            print(f"ERROR processing job {job_id}: {e}")
            traceback.print_exc()

            summaries.append(
                {
                    "job_id": job_id,
                    "build": job_cfg.get("build", ""),
                    "problem": job_cfg.get("problem", ""),
                    "run_dir": "",
                    "out_csv": "",
                    "rows": 0,
                    "status": "failed",
                }
            )

    summary_df = pd.DataFrame(summaries)

    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, f"ldms_{cfg_id}.csv")
    summary_df.to_csv(output_file, index=False)

    print(f"\nWrote summary: {output_file}")
    print(summary_df)


if __name__ == "__main__":
    main()
