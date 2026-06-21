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
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Per-worker port bases. Worker i gets base + i. Spacing of 1 is fine because
# each worker uses exactly one port from each base.
API_PORT_BASE = 8000
MCP_PORT_BASE = 9954
PROXY_PORT_BASE = 16443
ENGINE_PORT_BASE = 8080  # cerebral mode: worker i reaches its engine at localhost:8080+i

DEPLOY_STACK_SH = REPO_ROOT / "scripts" / "cerebral" / "deploy_stack.sh"


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


def mem_available_mb() -> int:
    """Host MemAvailable in MB (from /proc/meminfo); -1 if unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def start_mem_guard(procs: list, threshold_mb: int) -> threading.Event:
    """Watchdog: if host MemAvailable drops below threshold_mb, kill all workers.

    A safety backstop so a runaway problem (e.g. a metastable load generator) can
    never OOM a shared box — we abort the run instead. Returns a 'tripped' event.
    """
    tripped = threading.Event()

    def _watch():
        breaches = 0
        while not tripped.is_set():
            if all(p is None or p.poll() is not None for p in procs):
                return  # all workers finished
            avail = mem_available_mb()
            if 0 <= avail < threshold_mb:
                breaches += 1
                print(f"⚠️  mem-guard: MemAvailable {avail}MB < {threshold_mb}MB (strike {breaches}/2)")
                if breaches >= 2:  # two consecutive low reads → abort
                    print(f"🛑 mem-guard TRIPPED — killing all workers to protect the box (avail={avail}MB)")
                    tripped.set()
                    for p in procs:
                        if p is not None and p.poll() is None:
                            p.kill()
                    return
            else:
                breaches = 0
            time.sleep(15)

    threading.Thread(target=_watch, daemon=True).start()
    return tripped


def deploy_cerebral_stack(cluster: str, log_path: Path) -> None:
    """Deploy the cerebral stack into a kind cluster via deploy_stack.sh."""
    if "DEEPSEEK_API_KEY" not in os.environ:
        sys.exit("❌ --cerebral needs DEEPSEEK_API_KEY in the environment (source /root/.env).")
    print(f"🧠 deploying cerebral stack into {cluster} → {log_path}")
    # deploy_stack.sh selects the cluster via `kubectl --context kind-<name>`, so it
    # needs the default kubeconfig (which has every kind context). Drop any KUBECONFIG
    # the orchestrator set (e.g. for problem discovery), which may point at one cluster.
    deploy_env = os.environ.copy()
    deploy_env.pop("KUBECONFIG", None)
    with open(log_path, "w") as fh:
        res = subprocess.run(
            ["bash", str(DEPLOY_STACK_SH), cluster],
            cwd=REPO_ROOT,
            env=deploy_env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    if res.returncode != 0:
        sys.exit(f"❌ cerebral stack deploy failed for {cluster} (see {log_path})")


class EnginePortForward:
    """Keepalive `kubectl port-forward` to a cluster's cerebral-engine svc.

    Restarts the forward if it drops, so a long worker run never loses the engine
    connection. Call stop() to tear it down.
    """

    def __init__(self, kubeconfig: str, host_port: int, log_path: Path) -> None:
        self.kubeconfig = kubeconfig
        self.host_port = host_port
        self.log_path = log_path
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        env = os.environ.copy()
        env["KUBECONFIG"] = self.kubeconfig
        with open(self.log_path, "w") as fh:
            while not self._stop.is_set():
                self._proc = subprocess.Popen(
                    ["kubectl", "-n", "cerebral", "port-forward",
                     "svc/cerebral-engine", f"{self.host_port}:8080"],
                    env=env, stdout=fh, stderr=subprocess.STDOUT,
                )
                self._proc.wait()  # returns if the forward drops
                if not self._stop.is_set():
                    fh.write(f"\n[keepalive] port-forward dropped, restarting on :{self.host_port}\n")
                    fh.flush()
                    time.sleep(2)

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def wait_engine_healthy(host_port: int, timeout: float = 120.0) -> bool:
    """Poll the engine /api/health via the port-forward until ok or timeout."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{host_port}/api/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SREGym benchmark in parallel across sharded workers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", default=None, help="Agent to run (e.g. stratus, claudecode). Defaults to 'cerebral' with --cerebral.")
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
        "--exclude",
        default="",
        help="Comma-separated problem IDs to skip (e.g. the metastable load-generator problems "
        "load_spike_rpc_retry_storm,gc_capacity_degradation,capacity_decrease_rpc_retry_storm that "
        "spawn a ~20GB workload generator and can OOM a shared box).",
    )
    parser.add_argument(
        "--mem-guard-mb",
        type=int,
        default=0,
        help="If >0, abort the whole run (kill all workers) when host MemAvailable drops below this "
        "many MB — a safety backstop against OOM-ing a shared box. 0 disables.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Pass --force-build to the first worker so the agent image is (re)built before others start.",
    )
    parser.add_argument(
        "--cerebral",
        action="store_true",
        help="Cerebral mode: deploy the cerebral dataplane+engine into each worker cluster, "
        "port-forward its engine, and run the benchmark with --agent cerebral. Requires "
        "--kind-clusters and DEEPSEEK_API_KEY in the environment.",
    )
    args = parser.parse_args()

    if args.cerebral:
        if not args.kind_clusters:
            sys.exit("❌ --cerebral requires --kind-clusters (cluster names are needed to deploy the stack).")
        args.agent = "cerebral"
    if not args.agent:
        sys.exit("❌ --agent is required (or use --cerebral).")

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

    excluded = {p.strip() for p in args.exclude.split(",") if p.strip()}
    if excluded:
        hits = sorted(p for p in problems if p in excluded)
        problems = [p for p in problems if p not in excluded]
        print(f"🚫 Excluding {len(hits)} problem(s): {hits}")

    if len(problems) < n_workers:
        print(f"ℹ️  Only {len(problems)} problems for {n_workers} workers; some workers will be idle.")
    shards = round_robin_shards(problems, n_workers)

    print(f"🧩 {len(problems)} problems sharded across {n_workers} workers:")
    for i, shard in enumerate(shards):
        print(f"   worker{i}: {len(shard)} problems  (cluster: {kubeconfigs[i]})")

    # 2b) Cerebral mode: deploy the stack into each worker cluster and start a
    # keepalive engine port-forward (worker i → localhost:ENGINE_PORT_BASE+i).
    engine_pfs: list[EnginePortForward] = []
    engine_urls: dict[int, str] = {}
    if args.cerebral:
        for i, cluster in enumerate(clusters):
            if not shards[i]:
                continue
            deploy_cerebral_stack(cluster, run_root / f"worker{i}.cerebral-deploy.log")
            port = ENGINE_PORT_BASE + i
            pf = EnginePortForward(kubeconfigs[i], port, run_root / f"worker{i}.engine-pf.log")
            pf.start()
            engine_pfs.append(pf)
            engine_urls[i] = f"http://127.0.0.1:{port}"
            if wait_engine_healthy(port):
                print(f"🧠 worker{i}: engine healthy at {engine_urls[i]}")
            else:
                print(f"⚠️  worker{i}: engine /api/health not ok yet at {engine_urls[i]} (continuing)")

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
        # Ensure the interpreter dir is on PATH so a non-containerized agent's
        # kickoff_command ("python -m clients.<agent>.driver") resolves to this
        # same (venv) python, which has the deps. Without this the launcher's
        # shell can't find `python` and the agent exits 127.
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        if args.cerebral and i in engine_urls:
            env["CEREBRAL_ENGINE_URL"] = engine_urls[i]
            # Keep the cerebral stack alive across the conductor's per-problem reconcile.
            env["SREGYM_PROTECTED_NAMESPACES"] = "cerebral"

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

    # 4) Wait for all workers (with optional memory-guard watchdog).
    mem_guard = start_mem_guard(procs, args.mem_guard_mb) if args.mem_guard_mb > 0 else None
    if mem_guard is not None:
        print(f"🛡️  mem-guard active: will abort if MemAvailable < {args.mem_guard_mb}MB")
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
    if mem_guard is not None and mem_guard.is_set():
        print("🛑 Run was ABORTED by the memory guard — partial results only. Free up the box / reduce workers.")

    # Tear down cerebral engine port-forwards (the in-cluster stacks are left
    # running so the clusters can be reused; deploy_stack.sh --teardown removes them).
    for pf in engine_pfs:
        pf.stop()

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
