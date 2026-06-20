"""Cerebral × SREGym adapter — dataplane-mode result relay.

A minimal SREGym agent client that carries **no** perception or reasoning of its
own. The Cerebral engine investigates autonomously: the in-cluster dataplane
detects the injected fault, the engine's relay triggers a full pipeline run, and
the result lands as an incident. This adapter only waits for that incident, then
submits the engine's answer to SREGym in the shape each stage scores.

It mirrors `cerebral-monorepo/benchmarks/aiopslab/cerebral_agent.py` (same
incident-relay logic), adapted to SREGym's driver contract:

    GET  /status      -> {"stage": "setup"|"diagnosis"|"mitigation"|"tearing_down"|"done"}
    GET  /get_app     -> {"app_name","namespace","namespaces","descriptions"}
    GET  /get_problem -> {"problem_id"}
    POST /submit      -> {"solution": "<text>"}   (grades current stage, advances)

Per stage:
  * diagnosis  -> POST a natural-language root-cause description from the incident
  * mitigation -> apply the engine's remediation command (kubectl), then POST ""

Anti-hang/stuck/crash: every wait is bounded (settle + pipeline timeout); if no
incident forms or the engine is unreachable, the adapter submits the honest empty
and moves on. It never raises out of the driver — SREGym's --agent-timeout is the
outer backstop, but the adapter is designed to always reach a clean submit.

Env (all optional):
  CEREBRAL_ENGINE_URL              default http://localhost:8080
  CEREBRAL_ENGINE_TIMEOUT          default 60    (per-request seconds)
  CEREBRAL_PIPELINE_TIMEOUT        default 240   (max wait for an incident)
  CEREBRAL_PIPELINE_POLL_INTERVAL  default 5     (sleep between polls)
  CEREBRAL_SETTLE_SECONDS          default 45    (ignore incidents during deploy churn)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Add SREGym root to path (mirrors the other clients)
sregym_root = Path(__file__).resolve().parents[2]
if str(sregym_root) not in sys.path:
    sys.path.insert(0, str(sregym_root))

from logger import init_logger  # noqa: E402

init_logger()
logger = logging.getLogger("all.cerebral.driver")

ENGINE_URL = os.getenv("CEREBRAL_ENGINE_URL", "http://localhost:8080").rstrip("/")
ENGINE_TIMEOUT = float(os.getenv("CEREBRAL_ENGINE_TIMEOUT", "60"))
PIPELINE_TIMEOUT = float(os.getenv("CEREBRAL_PIPELINE_TIMEOUT", "240"))
POLL_INTERVAL = float(os.getenv("CEREBRAL_PIPELINE_POLL_INTERVAL", "5"))
SETTLE_SECONDS = float(os.getenv("CEREBRAL_SETTLE_SECONDS", "45"))

SUBMIT_STAGES = {"diagnosis", "mitigation"}
TERMINAL_STAGES = {"done", "tearing_down"}


# --------------------------------------------------------------------------- #
# SREGym conductor API
# --------------------------------------------------------------------------- #
def api_base() -> str:
    host = os.getenv("API_HOSTNAME", "localhost")
    port = os.getenv("API_PORT", "8000")
    # API_HOSTNAME may be 0.0.0.0 (bind addr); talk to it over loopback.
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def get_status() -> str | None:
    try:
        r = requests.get(f"{api_base()}/status", timeout=15)
        r.raise_for_status()
        return r.json().get("stage")
    except Exception as e:
        logger.debug(f"/status error: {e}")
        return None


def get_app_info() -> dict[str, Any]:
    r = requests.get(f"{api_base()}/get_app", timeout=30)
    r.raise_for_status()
    return r.json()


def get_problem_id() -> str:
    try:
        r = requests.get(f"{api_base()}/get_problem", timeout=30)
        r.raise_for_status()
        return r.json().get("problem_id", "unknown")
    except Exception:
        return "unknown"


def submit(solution: str) -> bool:
    """POST a solution for the current stage. Returns True on a graded response."""
    try:
        r = requests.post(f"{api_base()}/submit", json={"solution": solution}, timeout=300)
        logger.info(f"submit -> {r.status_code}: {r.text[:200]}")
        return r.status_code in (200, 201)
    except Exception as e:
        logger.warning(f"submit failed: {e}")
        return False


def wait_for_submit_stage(timeout: float = 600.0) -> str | None:
    """Wait until the conductor reaches a submission stage (or terminal)."""
    start = time.time()
    while time.time() - start < timeout:
        stage = get_status()
        if stage in SUBMIT_STAGES or stage in TERMINAL_STAGES:
            return stage
        time.sleep(2)
    return get_status()


def wait_until_stage_leaves(current: str, timeout: float = 300.0) -> str | None:
    """After a submit, wait for the conductor to advance past `current`."""
    start = time.time()
    while time.time() - start < timeout:
        stage = get_status()
        if stage != current:
            return stage
        time.sleep(2)
    return get_status()


# --------------------------------------------------------------------------- #
# Cerebral engine incident relay (ported from benchmarks/aiopslab/cerebral_agent.py)
# --------------------------------------------------------------------------- #
def _engine_get(path: str) -> dict[str, Any]:
    url = f"{ENGINE_URL}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=ENGINE_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _bare(node_id: str) -> str:
    return node_id.rsplit("/", 1)[-1] if node_id else node_id


def _ts(ts: Any) -> float:
    if not isinstance(ts, str) or not ts:
        return 0.0
    try:
        p = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if p.tzinfo is None:
            p = p.replace(tzinfo=timezone.utc)
        return p.timestamp()
    except ValueError:
        return 0.0


class CerebralRelay:
    """Waits for and reads this episode's engine incident for a namespace.

    A fresh adapter process runs per SREGym problem, so the process start time is
    the episode start: only incidents created after it count, which excludes
    incidents from prior problems on the same long-lived engine.
    """

    def __init__(self, namespaces: list[str]) -> None:
        self.namespaces = set(namespaces)
        self.started = time.time()
        self.created_after = self.started
        self.run_id: str | None = None
        self._done: dict[str, Any] | None = None  # cached terminal incident

    def incident(self) -> dict[str, Any]:
        """Return the engine's terminal answer for this episode (cached).

        Bounded by PIPELINE_TIMEOUT; returns {"anomaly": False} as the honest
        empty if nothing forms or the engine is unreachable. Never raises.
        """
        if self._done is not None:
            return self._done
        # Let deploy churn + LLM investigation pass before giving up.
        deadline = self.started + SETTLE_SECONDS + PIPELINE_TIMEOUT
        while time.time() < deadline:
            try:
                ev = self._advance()
            except (urllib.error.URLError, OSError, ValueError) as e:
                logger.warning(f"engine unreachable ({e}); submitting honest empty")
                self._done = {"anomaly": False}
                return self._done
            if ev is not None:
                self._done = ev
                return ev
            time.sleep(POLL_INTERVAL)
        logger.info("pipeline timeout — no incident; honest empty")
        self._done = {"anomaly": False}
        return self._done

    def _advance(self) -> dict[str, Any] | None:
        if not self.run_id:
            # Ignore everything during the initial deploy-churn window.
            if time.time() - self.started < SETTLE_SECONDS:
                return None
            doc = self._latest_incident()
            if doc:
                self.run_id = doc.get("run_id")
                return self._to_done(doc)
            return None
        doc = _engine_get(f"/api/incidents/{urllib.parse.quote(self.run_id)}")
        return self._to_done(doc)

    def _latest_incident(self) -> dict[str, Any] | None:
        listing = _engine_get("/api/incidents?limit=50")
        candidates = []
        for doc in listing.get("incidents") or []:
            ns = doc.get("namespace")
            if self.namespaces and ns not in self.namespaces:
                continue
            if _ts(doc.get("created_at") or doc.get("updated_at")) < self.created_after - 2:
                continue
            candidates.append(doc)
        if not candidates:
            return None
        # Deploy-churn incidents form first; the injected fault's incident forms
        # later, so the latest post-episode incident is the fault.
        candidates.sort(key=lambda d: _ts(d.get("created_at") or d.get("updated_at")), reverse=True)
        return candidates[0]

    def _to_done(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        status = doc.get("status")
        # Still investigating — keep waiting (bounded by incident()'s deadline).
        if status in {"active", "", None}:
            return None
        if status == "no_trigger":
            return {"anomaly": False}

        decision = doc.get("decision") or {}
        rca = self._rca_from_events(doc.get("events") or [])
        service = doc.get("service") or ""
        localized = _bare(rca.get("component") or service)
        taxonomy = doc.get("taxonomy") or {}
        if rca.get("system_level") and rca.get("fault_type"):
            taxonomy = {"system_level": rca["system_level"], "fault_type": rca["fault_type"]}
        done = {
            "anomaly": status == "completed" or bool(service),
            "service": service,
            "localized": localized,
            "decision_type": decision.get("decision_type"),
            "reasoning": decision.get("reasoning"),
            "rca": rca or None,
            "taxonomy": taxonomy,
            "remediation": self._remediation_from_decision(decision),
        }
        return done

    @staticmethod
    def _rca_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
        for ev in reversed(events):
            if ev.get("event_type") == "rca_synthesis":
                return ev.get("summary") or {}
        return {}

    @staticmethod
    def _remediation_from_decision(decision: dict[str, Any]) -> str:
        plan = decision.get("execution_plan") or {}
        # The engine already validated this as a single, safe kubectl invocation.
        return (plan.get("command") or "").strip()

    # --- map the incident to SREGym submissions ----------------------------- #
    @staticmethod
    def diagnosis_text(done: dict[str, Any]) -> str:
        if not done.get("anomaly"):
            return "No anomaly detected in the application."
        parts: list[str] = []
        comp = done.get("localized") or done.get("service")
        if comp:
            parts.append(f"The faulty component is '{comp}'.")
        rca = done.get("rca") or {}
        summary = rca.get("summary") or rca.get("root_cause") or done.get("reasoning")
        if summary:
            parts.append(f"Root cause: {summary}")
        tax = done.get("taxonomy") or {}
        if tax.get("system_level") or tax.get("fault_type"):
            parts.append(f"Fault taxonomy: system_level={tax.get('system_level')}, fault_type={tax.get('fault_type')}.")
        if done.get("decision_type"):
            parts.append(f"Recommended action: {done['decision_type']}.")
        return " ".join(parts) or f"An anomaly was detected affecting '{comp or 'the application'}'."


# --------------------------------------------------------------------------- #
# Mitigation: apply the engine's remediation command
# --------------------------------------------------------------------------- #
def apply_remediation(command: str) -> None:
    """Run the engine's remediation command (kubectl) with the launcher KUBECONFIG.

    Bounded and best-effort: a failed fix still leads to a submit so the problem
    is graded (as failed mitigation) rather than hanging.
    """
    if not command:
        logger.info("no remediation command from engine; submitting without a fix")
        return
    logger.info(f"applying remediation: {command}")
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        logger.info(f"remediation rc={r.returncode} stdout={r.stdout[:300]} stderr={r.stderr[:300]}")
    except Exception as e:
        logger.warning(f"remediation command failed: {e}")


# --------------------------------------------------------------------------- #
# Optional preflight (not invoked for non-containerized agents, but provided)
# --------------------------------------------------------------------------- #
def run_preflight() -> None:
    try:
        h = _engine_get("/api/health")
        ok = h.get("status") == "ok"
        print(f"engine health: {h}")
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"engine unreachable at {ENGINE_URL}: {e}")
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    logger.info("=" * 70)
    logger.info("Cerebral × SREGym adapter (dataplane-mode relay)")
    logger.info(f"engine={ENGINE_URL} api={api_base()} settle={SETTLE_SECONDS}s timeout={PIPELINE_TIMEOUT}s")
    logger.info("=" * 70)

    stage = wait_for_submit_stage(timeout=600)
    if stage in TERMINAL_STAGES or stage is None:
        logger.warning(f"conductor not in a submission stage (stage={stage}); nothing to do")
        return

    app_info = get_app_info()
    namespaces = app_info.get("namespaces") or [app_info.get("namespace")]
    namespaces = [n for n in namespaces if n]
    problem_id = get_problem_id()
    logger.info(f"problem={problem_id} namespaces={namespaces}")

    relay = CerebralRelay(namespaces)

    # Drive each submission stage the conductor presents until it's done.
    guard = 0
    while guard < 10:
        guard += 1
        stage = get_status()
        logger.info(f"stage={stage}")
        if stage in TERMINAL_STAGES or stage is None:
            break

        if stage == "diagnosis":
            done = relay.incident()
            text = relay.diagnosis_text(done)
            logger.info(f"diagnosis: {text}")
            submit(text)
            stage = wait_until_stage_leaves("diagnosis")

        elif stage == "mitigation":
            done = relay.incident()
            apply_remediation(done.get("remediation", ""))
            submit("")
            stage = wait_until_stage_leaves("mitigation")

        else:  # setup or any transient stage
            time.sleep(3)

    logger.info(f"adapter done (final stage={get_status()})")


if __name__ == "__main__":
    main()
