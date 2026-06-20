#!/usr/bin/env python3
"""
Parallel SREGym benchmark orchestrator.

Shards the benchmark's problem list across N independent workers that run
simultaneously, each owning its own Kubernetes cluster (selected via KUBECONFIG)
and a distinct set of host ports. When every worker finishes, the per-worker
result CSVs are merged into one combined CSV.

This is a thin layer on top of `main.py`: each worker is just a normal
`python main.py ...` invocation with a problem shard, a unique results label,
and non-colliding ports. The sequential conductor logic is unchanged — the
isolation boundary is "one cluster per worker".

Examples
--------
Two workers on two local kind clusters:

    python run_parallel.py \
        --agent stratus --model gpt-5 \
        --kind-clusters pbench0,pbench1

Two workers against explicit kubeconfig files, 4 workers' worth of shards:

    python run_parallel.py \
        --agent stratus --model anthropic/claude-sonnet-4-6 \
        --kubeconfigs /tmp/kc0,/tmp/kc1

Restrict to a subset of problems (handy for validation):

    python run_parallel.py --agent stratus --model gpt-5 \
        --kind-clusters pbench0,pbench1 \
        --problems misconfig_app_hotel_res,missing_service_hotel_reservation
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Per-worker port bases. Worker i gets base + i. Spacing of 1 is fine because
# each worker uses exactly one port from each base.
API_PORT_BASE = 8000
MCP_PORT_BASE = 9954
PROXY_PORT_BASE = 16443


def discover_problem_ids() -> list[str]:
    """Return the default benchmark problem list (same source main.py uses)."""
    code = (
        "from sregym.conductor.problems.registry import ProblemRegistry;"
        "print('\\n'.join(ProblemRegistry().get_problem_ids()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(
            "❌ Could not discover problem IDs by importing the registry "
            f"(needs a valid kubeconfig). Pass --problems or --problems-file instead.\n{out.stderr}"
        )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def round_robin_shards(problems: list[str], n: int) -> list[list[str]]:
    """Split problems into n near-equal shards (round-robin for balance)."""
    shards: list[list[str]] = [[] for _ in range(n)]
    for idx, pid in enumerate(problems):
        shards[idx % n].append(pid)
    return shards


def kubeconfig_for_kind(cluster: str, dest_dir: Path) -> str:
    """Export a kind cluster's kubeconfig to its own file and return the path."""
    dest = dest_dir / f"kubeconfig-{cluster}"
    res = subprocess.run(
        ["kind", "get", "kubeconfig", "--name", cluster],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        sys.exit(f"❌ `kind get kubeconfig --name {cluster}` failed:\n{res.stderr}")
    dest.write_text(res.stdout)
    return str(dest)


def merge_results(worker_dirs: list[Path], agent: str, out_path: Path) -> int:
    """Concatenate each worker's <agent>_ALL_results.csv into one CSV.

    Adds a `worker` column. Returns the number of result rows written.
    """
    rows: list[dict] = []
    for i, wdir in enumerate(worker_dirs):
        if wdir is None:
            continue
        csv_path = wdir / f"{agent}_ALL_results.csv"
        if not csv_path.exists():
            print(f"⚠️  No results CSV for worker{i} at {csv_path} (worker may have crashed)")
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                row["worker"] = f"worker{i}"
                rows.append(row)

    if not rows:
        print("⚠️  No result rows found across workers; nothing to merge.")
        return 0

    fieldnames = sorted({k for row in rows for k in row})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def newest_worker_dir(label: str, after_ts: float) -> Path | None:
    """Find the results/<timestamp>_<label> dir created during this run."""
    candidates = sorted(
        (p for p in (REPO_ROOT / "results").glob(f"*_{label}") if p.is_dir() and p.stat().st_mtime >= after_ts - 5),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SREGym benchmark in parallel across sharded workers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", required=True, help="Agent to run (e.g. stratus, claudecode)")
    parser.add_argument("--model", required=True, help="LiteLLM model string")
    parser.add_argument("--judge-model", default=None, help="Judge model (defaults to --model)")
    parser.add_argument("--n-attempts", type=int, default=1, help="Attempts per problem (default: 1)")
    parser.add_argument("--agent-timeout", type=int, default=1800, help="Per-attempt agent timeout seconds")

    cluster_group = parser.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument(
        "--kind-clusters",
        help="Comma-separated kind cluster names, one per worker (e.g. pbench0,pbench1).",
    )
    cluster_group.add_argument(
        "--kubeconfigs",
        help="Comma-separated kubeconfig file paths, one per worker.",
    )

    problem_group = parser.add_mutually_exclusive_group()
    problem_group.add_argument("--problems", help="Comma-separated problem IDs to run (overrides default list).")
    problem_group.add_argument("--problems-file", help="File with newline-delimited problem IDs to run.")

    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Pass --force-build to the first worker so the agent image is (re)built before others start.",
    )
    args = parser.parse_args()

    # 1) Resolve per-worker kubeconfigs.
    run_root = REPO_ROOT / "results" / f"parallel_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    if args.kind_clusters:
        clusters = [c.strip() for c in args.kind_clusters.split(",") if c.strip()]
        kubeconfigs = [kubeconfig_for_kind(c, run_root) for c in clusters]
    else:
        kubeconfigs = [p.strip() for p in args.kubeconfigs.split(",") if p.strip()]
        for kc in kubeconfigs:
            if not Path(kc).exists():
                sys.exit(f"❌ Kubeconfig not found: {kc}")

    n_workers = len(kubeconfigs)
    if n_workers < 1:
        sys.exit("❌ Need at least one cluster/kubeconfig.")

    # 2) Resolve and shard the problem list.
    if args.problems:
        problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    elif args.problems_file:
        problems = [
            line.strip()
            for line in Path(args.problems_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        problems = discover_problem_ids()

    if len(problems) < n_workers:
        print(f"ℹ️  Only {len(problems)} problems for {n_workers} workers; some workers will be idle.")
    shards = round_robin_shards(problems, n_workers)

    print(f"🧩 {len(problems)} problems sharded across {n_workers} workers:")
    for i, shard in enumerate(shards):
        print(f"   worker{i}: {len(shard)} problems  (cluster: {kubeconfigs[i]})")

    # 3) Launch workers.
    start_ts = time.time()
    procs = []
    log_files = []
    for i, (kc, shard) in enumerate(zip(kubeconfigs, shards)):
        if not shard:
            procs.append(None)
            log_files.append(None)
            continue

        shard_file = run_root / f"worker{i}.problems"
        shard_file.write_text("\n".join(shard) + "\n")

        cmd = [
            sys.executable,
            "main.py",
            "--agent", args.agent,
            "--model", args.model,
            "--n-attempts", str(args.n_attempts),
            "--agent-timeout", str(args.agent_timeout),
            "--problems-file", str(shard_file),
            "--run-label", f"worker{i}",
            "--api-port", str(API_PORT_BASE + i),
            "--mcp-port", str(MCP_PORT_BASE + i),
            "--proxy-port", str(PROXY_PORT_BASE + i),
        ]
        if args.judge_model:
            cmd += ["--judge-model", args.judge_model]
        # Only the first worker forces an image (re)build; the shared image is
        # then reused by the rest. Build happens before the agent runs.
        if args.force_build and i == 0:
            cmd += ["--force-build"]

        env = os.environ.copy()
        env["KUBECONFIG"] = kc

        log_path = run_root / f"worker{i}.log"
        log_fh = open(log_path, "w")
        log_files.append(log_fh)
        print(f"🚀 worker{i}: launching ({len(shard)} problems) → {log_path}")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append(proc)

        # Stagger image build: let worker0 build the image before the rest start
        # so they don't all race to build the same Docker image concurrently.
        if args.force_build and i == 0:
            print("⏳ Letting worker0 build the agent image before starting the rest...")
            time.sleep(90)

    # 4) Wait for all workers.
    print("⏳ Waiting for all workers to finish...")
    exit_codes = []
    for i, proc in enumerate(procs):
        if proc is None:
            exit_codes.append(None)
            continue
        rc = proc.wait()
        exit_codes.append(rc)
        status = "✅" if rc == 0 else "❌"
        print(f"{status} worker{i} exited with code {rc}")
    for fh in log_files:
        if fh:
            fh.close()

    # 5) Merge results.
    worker_dirs = [newest_worker_dir(f"worker{i}", start_ts) if procs[i] else None for i in range(n_workers)]
    combined = run_root / "combined_ALL_results.csv"
    n_rows = merge_results(worker_dirs, args.agent, combined)

    elapsed = time.time() - start_ts
    print(f"\n⏱️  Total wall-clock: {elapsed / 60:.1f} min")
    if n_rows:
        print(f"✅ Merged {n_rows} result rows → {combined}")
    print(f"📂 Per-worker logs, shards, and results under: {run_root}")

    if any(rc not in (0, None) for rc in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
