#!/usr/bin/env python3

"""Fetch per-job LDMS metric time series from DSOS, one un-aggregated CSV per job."""

import argparse
import traceback
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
LDMS_DATA_DIR = BASE_DIR / "data" / "ldms"

SOS_CONFIG = "/opt/ovis/dsos/config/all.conf"
SOS_DATABASE = "/storage/manzano/sos/database"

MEM_SCHEMA = "meminfo_toss4"
CPU_SCHEMA = "procstat_96"

CHUNK_ROWS = 1024 * 1024

SAMPLE_COLUMNS = ["timestamp", "job_id", "component_id", "metric", "unit", "value"]
TIDY_COLUMNS = ["timestamp", "time_rel_s", *SAMPLE_COLUMNS[1:]]
RESOURCE_COLUMNS = [
    "nodes", "ntasks", "ntasks_per_node", "omp_num_threads", "time", "deck_sha256",
    "particle_count", "particle_count_final",
]
MANIFEST_COLUMNS = ["build", "problem", *RESOURCE_COLUMNS]
SUMMARY_COLUMNS = ["job_id", *MANIFEST_COLUMNS, "components", "rows", "out_csv", "status"]
BASELINE_COLUMNS = ["job_id", "component_id", "metric", "unit", "value", "samples"]

# LDMS tags samples taken while no job runs on the node with this job_id.
IDLE_JOB_ID = 0
# How far before the job starts (or after it ends) idle samples count toward the node baseline.
BASELINE_WINDOW_S = 600.0

# Turns one component's time-ordered samples into the metric's value column.
Deriver = Callable[[pd.DataFrame], pd.Series]

# Runs one DSOS SQL statement and returns every matching row.
QueryFn = Callable[[str], pd.DataFrame]


class MetricSpec(NamedTuple):
    schema: str
    columns: Tuple[str, ...]
    derive: Deriver
    unit: str


def gauge(*columns: str) -> Deriver:
    """Read columns that already hold the value, such as memory in use, summing them if several."""
    return lambda df: sum(df[column].astype(float) for column in columns)


def difference(minuend: str, subtrahend: str) -> Deriver:
    """Subtract one gauge column from another, such as MemTotal - MemAvailable."""
    return lambda df: df[minuend].astype(float) - df[subtrahend].astype(float)


def rate(*columns: str, scale: float = 1.0) -> Deriver:
    """Differentiate monotonic counters, such as CPU jiffies, into a per-second rate."""
    def derive(df: pd.DataFrame) -> pd.Series:
        total = sum(df[column].astype(float) for column in columns)
        return total.diff() / df["timestamp"].diff() / scale
    return derive


# Add a metric by adding a row here; nothing downstream needs to change.
METRICS: Dict[str, MetricSpec] = {
    "mem_active_kb": MetricSpec(MEM_SCHEMA, ("Active",), gauge("Active"), "kB"),
    # Node-wide meminfo: anon + shmem approximates the job's resident memory on an exclusive node.
    "mem_anon_kb": MetricSpec(MEM_SCHEMA, ("AnonPages",), gauge("AnonPages"), "kB"),
    "mem_anon_huge_kb": MetricSpec(MEM_SCHEMA, ("AnonHugePages",), gauge("AnonHugePages"), "kB"),
    "mem_shmem_kb": MetricSpec(MEM_SCHEMA, ("Shmem",), gauge("Shmem"), "kB"),
    "mem_used_kb": MetricSpec(MEM_SCHEMA, ("MemTotal", "MemAvailable"), difference("MemTotal", "MemAvailable"), "kB"),
    "mem_kernel_kb": MetricSpec(MEM_SCHEMA, ("PageTables", "SUnreclaim"), gauge("PageTables", "SUnreclaim"), "kB"),
    "mem_committed_kb": MetricSpec(MEM_SCHEMA, ("Committed_AS",), gauge("Committed_AS"), "kB"),
    "mem_cached_kb": MetricSpec(MEM_SCHEMA, ("Cached",), gauge("Cached"), "kB"),
    # procstat counts USER_HZ jiffies, so a scale of 100 gives cores in use.
    "cpu_cores_used": MetricSpec(CPU_SCHEMA, ("user", "sys"), rate("user", "sys", scale=100.0), "cores"),
}

# Metric names used by older manifests.
ALIASES = {"Active": "mem_active_kb", "CPU": "cpu_cores_used"}


def resolve(name: str) -> str:
    canonical = ALIASES.get(name, name)
    if canonical not in METRICS:
        raise KeyError(f"unknown metric '{name}'; known metrics: {', '.join(sorted(METRICS))}")
    return canonical


def load_manifest(path: Path) -> Tuple[List[str], Dict[int, dict]]:
    """Validated manifest as (canonical metric names, {job_id: job config})."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    metrics = cfg.get("metrics") or []
    jobs = cfg.get("jobs") or {}

    if not metrics:
        raise ValueError(f"{path}: 'metrics' must be a non-empty list")
    if not jobs:
        raise ValueError(f"{path}: 'jobs' must be a non-empty mapping")

    parsed = {}
    for job_id, job_cfg in jobs.items():
        job_cfg = job_cfg or {}
        missing = [key for key in ("build", "problem") if not job_cfg.get(key)]
        if missing:
            raise ValueError(f"{path}: job {job_id} is missing {', '.join(missing)}")
        parsed[int(job_id)] = job_cfg

    # Resolve up front so a typo fails before the first query runs.
    return [resolve(metric) for metric in metrics], parsed


def run_query(cont, sql: str) -> pd.DataFrame:
    """Drain a DSOS query into one dataframe."""
    query = cont.query(CHUNK_ROWS)
    query.select(sql)

    chunks = []

    chunk = query.next()
    while chunk is not None:
        chunks.append(chunk.copy(deep=True))
        chunk = query.next()

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def build_select(spec: MetricSpec, job_id: int) -> str:
    columns = ", ".join(("timestamp", "component_id", *spec.columns))
    return f"select {columns} from {spec.schema} where job_id == {job_id} order_by job_time_comp"


def timestamp_to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64") / 1.0e9
    return series.astype(float)


def fetch_metric(query: QueryFn, job_id: int, name: str) -> pd.DataFrame:
    """Every sample of one metric for one job, one row per (timestamp, component)."""
    spec = METRICS[resolve(name)]
    raw = query(build_select(spec, job_id))

    if raw.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)

    df = raw.copy()
    df["timestamp"] = timestamp_to_seconds(df["timestamp"])
    df["component_id"] = df["component_id"].astype(int)
    df = df.sort_values(["component_id", "timestamp"], ignore_index=True)

    # Derive per component so a counter rate never spans a node boundary.
    groups = df.groupby("component_id", sort=False)[["timestamp", *spec.columns]]
    df["value"] = pd.concat([spec.derive(group) for _, group in groups])

    df["job_id"] = job_id
    df["metric"] = resolve(name)
    df["unit"] = spec.unit

    # A rate is undefined at a component's first sample; drop it rather than invent a zero.
    return df.dropna(subset=["value"])[SAMPLE_COLUMNS]


def collect_job(query: QueryFn, job_id: int, metrics: Iterable[str]) -> pd.DataFrame:
    """Full un-aggregated time series for one job, tidy and stacked across metrics."""
    frames = [fetch_metric(query, job_id, name) for name in metrics]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.sort_values(["metric", "component_id", "timestamp"], ignore_index=True)

    # Shared origin so every metric's curve lines up on one axis.
    tidy["time_rel_s"] = tidy["timestamp"] - tidy["timestamp"].min()

    return tidy[TIDY_COLUMNS]


def build_node_select(columns: Iterable[str], component_id: int) -> str:
    columns = ", ".join(("timestamp", "job_id", *columns))
    return f"select {columns} from {MEM_SCHEMA} where component_id == {component_id} order_by comp_time_job"


def idle_window(node: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Idle samples just before the job, or just after it when the node was busy beforehand."""
    idle = node[node["job_id"].astype(int) == IDLE_JOB_ID]
    before = idle[(idle["timestamp"] >= start - BASELINE_WINDOW_S) & (idle["timestamp"] < start)]
    if not before.empty:
        return before
    return idle[(idle["timestamp"] > end) & (idle["timestamp"] <= end + BASELINE_WINDOW_S)]


def collect_baseline(query: QueryFn, job_id: int, tidy: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Median idle-node value of each memory metric, per component, to subtract from the job's samples."""
    memory = [name for name in metrics if METRICS[name].schema == MEM_SCHEMA]
    columns = sorted({column for name in memory for column in METRICS[name].columns})
    rows = []

    for component_id, samples in tidy.groupby("component_id"):
        node = query(build_node_select(columns, component_id)) if memory else pd.DataFrame()
        if node.empty:
            continue
        node["timestamp"] = timestamp_to_seconds(node["timestamp"])
        window = idle_window(node, samples["timestamp"].min(), samples["timestamp"].max())
        if window.empty:
            print(f"No idle samples within {BASELINE_WINDOW_S:.0f} s of job {job_id} on component {component_id}")
            continue
        for name in memory:
            spec = METRICS[name]
            value = float(spec.derive(window).median())
            rows.append({"job_id": job_id, "component_id": component_id, "metric": name,
                         "unit": spec.unit, "value": value, "samples": len(window)})

    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


def manifest_fields(job_cfg: dict) -> dict:
    return {column: job_cfg.get(column, "") for column in MANIFEST_COLUMNS}


def refresh_summary(summary: pd.DataFrame, jobs: Dict[int, dict]) -> pd.DataFrame:
    """Re-copy manifest fields (such as newly recorded particle counts) into an existing summary."""
    fields = pd.DataFrame([{"job_id": job_id, **manifest_fields(cfg)} for job_id, cfg in jobs.items()])
    kept = summary.drop(columns=[column for column in MANIFEST_COLUMNS if column in summary])
    return kept.merge(fields, on="job_id", how="left")[SUMMARY_COLUMNS]


def summary_path(data_dir: Path, manifest: Path) -> Path:
    stamp = manifest.stem[len("manifest_"):] if manifest.stem.startswith("manifest_") else manifest.stem
    return data_dir / f"summary_{stamp}.csv"


def summarize(job_id: int, job_cfg: dict, tidy: pd.DataFrame, out_csv, status: str) -> dict:
    return {
        "job_id": job_id,
        **manifest_fields(job_cfg),
        "components": tidy["component_id"].nunique() if not tidy.empty else 0,
        "rows": len(tidy),
        "out_csv": str(out_csv),
        "status": status,
    }


def process_job(query: QueryFn, job_id: int, job_cfg: dict, metrics: List[str], data_dir: Path) -> dict:
    tidy = collect_job(query, job_id, metrics)

    if tidy.empty:
        return summarize(job_id, job_cfg, tidy, "", "no_data")

    out_csv = data_dir / job_cfg["build"] / job_cfg["problem"] / str(job_id) / "ldms_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(tidy)} rows)")

    baseline = collect_baseline(query, job_id, tidy, metrics)
    if not baseline.empty:
        baseline_csv = out_csv.with_name("ldms_baseline.csv")
        baseline.to_csv(baseline_csv, index=False)
        print(f"Wrote {baseline_csv} ({len(baseline)} rows)")

    return summarize(job_id, job_cfg, tidy, out_csv, "ok")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write one un-aggregated CSV of LDMS metrics per job in a manifest.",
    )
    parser.add_argument("manifest", type=Path, help="YAML manifest listing metrics and jobs.")
    parser.add_argument("--data-dir", type=Path, default=LDMS_DATA_DIR)
    parser.add_argument("--sos-config", default=SOS_CONFIG)
    parser.add_argument("--sos-database", default=SOS_DATABASE)
    parser.add_argument(
        "--refresh-summary", action="store_true",
        help="Update an existing summary CSV's manifest columns without querying DSOS.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    metrics, jobs = load_manifest(args.manifest)
    summary_csv = summary_path(args.data_dir, args.manifest)

    if args.refresh_summary:
        refresh_summary(pd.read_csv(summary_csv), jobs).to_csv(summary_csv, index=False)
        print(f"Refreshed summary: {summary_csv}")
        return

    from sosdb import Sos  # only installed on the DSOS cluster

    cont = Sos.Session(args.sos_config).open(args.sos_database)
    query = partial(run_query, cont)

    summaries = []
    for job_id, job_cfg in jobs.items():
        print(f"\n=== job {job_id} ===")
        try:
            summaries.append(process_job(query, job_id, job_cfg, metrics, args.data_dir))
        except Exception as error:
            print(f"ERROR processing job {job_id}: {error}")
            traceback.print_exc()
            summaries.append(summarize(job_id, job_cfg, pd.DataFrame(), "", "failed"))

    args.data_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nWrote summary: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
