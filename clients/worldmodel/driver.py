"""SREGym adapter for Cerebral's perception-world-model runtime.

Adapter responsibilities are deliberately narrow:
  - connect to SREGym's API and MCP servers
  - expose raw MCP tools to the world-model runner
  - submit the runner's diagnosis/mitigation answers
  - persist raw runner observations for verification
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

sregym_root = Path(__file__).resolve().parents[2]
if str(sregym_root) not in sys.path:
    sys.path.insert(0, str(sregym_root))

from clients.stratus.configs.langgraph_tool_configs import LanggraphToolConfig  # noqa: E402
from logger import init_logger  # noqa: E402

world_model_agents_path = (
    Path(os.environ.get("CEREBRAL_WORLD_MODEL_PATH", ""))
    if os.environ.get("CEREBRAL_WORLD_MODEL_PATH")
    else Path("/root/perception-world-model/agents")
)
local_world_model_agents_path = Path(__file__).resolve().parents[3] / "perception-world-model" / "agents"
for candidate in (world_model_agents_path, local_world_model_agents_path):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cerebral_agents.sregym_runtime import (  # noqa: E402
    SREGymMcpTool,
    SREGymObservation,
    SREGymStageResult,
    SREGymWorldModelRuntime,
    provider_from_env,
)

init_logger()
logger = logging.getLogger("all.worldmodel.driver")


class SreGymMcp:
    """Raw SREGym MCP gateway used by the world-model runner."""

    def __init__(self) -> None:
        config = LanggraphToolConfig()
        self.session_id = str(uuid.uuid4())
        self.urls = {
            "kubectl": config.kubectl_mcp_url,
            "prometheus": config.prometheus_mcp_url,
            "loki": f"http://{os.getenv('API_HOSTNAME', 'localhost')}:{os.getenv('MCP_SERVER_PORT', '9954')}/loki/sse",
            "jaeger": config.jaeger_mcp_url,
            "submit": config.submit_mcp_url,
        }

    async def list_tools(self, endpoint: str) -> list[SREGymMcpTool]:
        return _static_tools(endpoint)

    async def call_tool(self, endpoint: str, tool: str, args: dict[str, Any]) -> str:
        if endpoint == "kubernetes":
            return await self._call_kubernetes_tool(tool, args)
        endpoint, tool, args = _map_otel_tool(endpoint, tool, args)
        return await self._call_tool(endpoint, tool, args)

    async def _call_kubernetes_tool(self, tool: str, args: dict[str, Any]) -> str:
        if tool == "get_previous_rollback":
            return await self._call_tool(
                "kubectl",
                "get_previous_rollbackable_cmd",
                {
                    "deployment": _required(args, "deployment"),
                    "namespace": args.get("namespace") or "",
                },
            )
        cmd = _kubectl_command(tool, args)
        return await self._call_tool("kubectl", "exec_kubectl_cmd_safely", {"cmd": cmd})

    async def _call_tool(self, endpoint: str, tool: str, args: dict[str, Any]) -> str:
        timeout = _mcp_timeout_seconds(endpoint, tool, args)
        started = time.monotonic()
        logger.info("worldmodel MCP call start endpoint=%s tool=%s timeout=%.0fs args=%s", endpoint, tool, timeout, _log_args(args))
        try:
            result = await asyncio.wait_for(self._call_tool_once(endpoint, tool, args), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "worldmodel MCP call timeout endpoint=%s tool=%s elapsed=%.2fs args=%s",
                endpoint,
                tool,
                time.monotonic() - started,
                _log_args(args),
            )
            raise TimeoutError(f"MCP tool {endpoint}.{tool} timed out after {timeout:.0f}s") from None
        logger.info("worldmodel MCP call done endpoint=%s tool=%s elapsed=%.2fs", endpoint, tool, time.monotonic() - started)
        return result

    async def _call_tool_once(self, endpoint: str, tool: str, args: dict[str, Any]) -> str:
        async with AsyncExitStack() as stack:
            session = await self._session(endpoint, stack)
            result = await session.call_tool(tool, arguments=args or {})
        return _mcp_text(result)

    async def submit(self, answer: str) -> None:
        await self.call_tool("submit", "submit", {"ans": answer})

    async def _session(self, endpoint: str, stack: AsyncExitStack) -> ClientSession:
        url = self.urls[endpoint]
        kwargs: dict[str, Any] = {"url": url}
        if endpoint != "submit":
            kwargs["headers"] = {"sregym_ssid": self.session_id}
        try:
            read_stream, write_stream = await stack.enter_async_context(sse_client(**kwargs))
        except TypeError:
            kwargs.pop("headers", None)
            read_stream, write_stream = await stack.enter_async_context(sse_client(**kwargs))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    async def close(self) -> None:
        return None


def _load_env() -> None:
    for path in (Path("/root/.env"), Path.cwd() / ".env"):
        if path.exists():
            load_dotenv(path, override=False)
    if os.environ.get("DEEPSEEK_API_KEY"):
        os.environ.setdefault("CEREBRAL_GENERIC_API_KEY", os.environ["DEEPSEEK_API_KEY"])
        os.environ.setdefault("OPENAI_API_KEY", os.environ["DEEPSEEK_API_KEY"])
    os.environ.setdefault("CEREBRAL_GENERIC_BASE_URL", os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    os.environ.setdefault("CEREBRAL_WORLD_MODEL_PATH", str(world_model_agents_path))


def get_api_base_url() -> str:
    host = os.getenv("API_HOSTNAME", "localhost")
    port = os.getenv("API_PORT", "8000")
    return f"http://{host}:{port}"


def api_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{get_api_base_url()}{path}", timeout=10)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def wait_for_stage(stages: set[str], timeout: int = 300) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            stage = api_json("/status").get("stage")
            if stage in stages:
                return str(stage)
            if stage in {"done", "tearing_down"}:
                return str(stage)
        except Exception as exc:
            logger.debug("status poll failed: %s", exc)
        time.sleep(1)
    raise TimeoutError(f"conductor did not reach {sorted(stages)} within {timeout}s")


def save_results(logs_dir: Path, problem_id: str, payload: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"worldmodel_results_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("saved results to %s", path)


def _schema_for_tool(tool: Any) -> dict[str, Any]:
    for attr in ("inputSchema", "input_schema", "args_schema"):
        value = getattr(tool, attr, None)
        if isinstance(value, dict):
            return value
    return {}


def _static_tools(endpoint: str) -> list[SREGymMcpTool]:
    """Expose raw SREGym MCP tool names without a discovery round-trip."""
    if endpoint == "kubectl":
        return [
            _mcp_tool(
                "kubectl",
                "exec_kubectl_cmd_safely",
                "Execute a kubectl command through SREGym's safety proxy. Diagnosis should use read-only kubectl commands.",
                {"cmd": "Full kubectl command, starting with kubectl."},
            ),
        ]
    if endpoint == "prometheus":
        return [
            _mcp_tool("prometheus", "get_metrics", "Query Prometheus with PromQL.", {"query": "PromQL query."}),
            _mcp_tool("prometheus", "get_alerts", "Return currently firing Prometheus alerts.", {}),
        ]
    if endpoint == "loki":
        return [
            _mcp_tool(
                "loki",
                "get_logs",
                "Query Loki logs with LogQL.",
                {"query": "LogQL query.", "last_n_minutes": "Minutes to look back."},
            ),
            _mcp_tool("loki", "get_labels", "Return available Loki label names.", {}),
            _mcp_tool("loki", "get_label_values", "Return values for a Loki label.", {"label": "Label name."}),
        ]
    if endpoint == "jaeger":
        return [
            _mcp_tool("jaeger", "get_services", "Return traced service names.", {}),
            _mcp_tool("jaeger", "get_operations", "Return traced operations for a service.", {"service": "Service name."}),
            _mcp_tool(
                "jaeger",
                "get_traces",
                "Return recent traces for a service.",
                {"service": "Service name.", "last_n_minutes": "Minutes to look back."},
            ),
            _mcp_tool(
                "jaeger",
                "get_dependency_graph",
                "Return Jaeger service dependencies.",
                {"last_n_minutes": "Minutes to look back."},
            ),
        ]
    return []


def _mcp_tool(
    endpoint: str,
    name: str,
    description: str,
    properties: dict[str, Any],
) -> SREGymMcpTool:
    return SREGymMcpTool(
        endpoint=endpoint,
        name=name,
        description=description,
        args_schema={
            "type": "object",
            "properties": {
                key: value if isinstance(value, dict) else {"description": str(value)}
                for key, value in properties.items()
            },
        },
    )


def _world_model_endpoints() -> tuple[str, ...]:
    configured = os.getenv("SREGYM_WORLDMODEL_ENDPOINTS")
    if configured:
        return tuple(endpoint.strip() for endpoint in configured.split(",") if endpoint.strip())
    return ("kubectl", "prometheus", "loki", "jaeger")


def _map_otel_tool(endpoint: str, tool: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if endpoint == "prometheus":
        return endpoint, {"query": "get_metrics", "alerts": "get_alerts"}.get(tool, tool), args
    if endpoint == "loki":
        return endpoint, {"query_logs": "get_logs"}.get(tool, tool), args
    if endpoint == "jaeger":
        return endpoint, {
            "list_services": "get_services",
            "list_operations": "get_operations",
            "dependency_graph": "get_dependency_graph",
        }.get(tool, tool), args
    return endpoint, tool, args


def _kubectl_command(tool: str, args: dict[str, Any]) -> str:
    namespace = str(args.get("namespace") or "").strip()
    if tool == "get_pods":
        cmd = ["kubectl", "get", "pods"]
        if args.get("all_namespaces"):
            cmd.append("-A")
        else:
            cmd.extend(["-n", _required(args, "namespace")])
        if args.get("selector"):
            cmd.extend(["-l", str(args["selector"])])
        cmd.extend(["-o", "wide"])
        return _join(cmd)
    if tool == "get_deployment":
        return _join(["kubectl", "get", "deployment", _required(args, "deployment"), "-n", _required(args, "namespace"), "-o", "yaml"])
    if tool == "describe_resource":
        return _join(["kubectl", "describe", str(_required(args, "kind")), str(_required(args, "name")), "-n", _required(args, "namespace")])
    if tool == "get_events":
        cmd = ["kubectl", "get", "events"]
        if namespace:
            cmd.extend(["-n", namespace])
        if args.get("field_selector"):
            cmd.extend(["--field-selector", str(args["field_selector"])])
        cmd.append("--sort-by=.lastTimestamp")
        return _join(cmd)
    if tool == "get_logs":
        target = f"pod/{args['pod']}" if args.get("pod") else f"deploy/{_required(args, 'deployment')}"
        cmd = ["kubectl", "logs", target, "-n", _required(args, "namespace")]
        if args.get("container"):
            cmd.extend(["-c", str(args["container"])])
        cmd.extend(["--tail", str(args.get("tail_lines") or 120)])
        return _join(cmd)
    if tool == "get_service":
        return _join(["kubectl", "get", "service", _required(args, "service"), "-n", _required(args, "namespace"), "-o", "yaml"])
    if tool == "get_endpoints":
        return _join(["kubectl", "get", "endpoints", _required(args, "service"), "-n", _required(args, "namespace"), "-o", "yaml"])
    if tool == "rollout_status":
        return _join([
            "kubectl",
            "rollout",
            "status",
            f"deploy/{_required(args, 'deployment')}",
            "-n",
            _required(args, "namespace"),
            f"--timeout={args.get('timeout') or '120s'}",
        ])
    if tool == "rollout_undo":
        cmd = ["kubectl", "rollout", "undo", f"deploy/{_required(args, 'deployment')}", "-n", _required(args, "namespace")]
        if args.get("to_revision"):
            cmd.append(f"--to-revision={args['to_revision']}")
        return _join(cmd)
    if tool == "restart_deployment":
        return _join(["kubectl", "rollout", "restart", f"deploy/{_required(args, 'deployment')}", "-n", _required(args, "namespace")])
    if tool == "set_image":
        return _join([
            "kubectl",
            "set",
            "image",
            f"deploy/{_required(args, 'deployment')}",
            f"{_required(args, 'container')}={_required(args, 'image')}",
            "-n",
            _required(args, "namespace"),
        ])
    if tool == "set_env":
        return _join([
            "kubectl",
            "set",
            "env",
            f"deploy/{_required(args, 'deployment')}",
            f"{_required(args, 'name')}={_required(args, 'value')}",
            "-n",
            _required(args, "namespace"),
        ])
    if tool == "unset_env":
        return _join([
            "kubectl",
            "set",
            "env",
            f"deploy/{_required(args, 'deployment')}",
            f"{_required(args, 'name')}-",
            "-n",
            _required(args, "namespace"),
        ])
    if tool == "patch_deployment":
        patch = args.get("patch")
        patch_text = json.dumps(patch) if isinstance(patch, (dict, list)) else str(_required(args, "patch"))
        return _join([
            "kubectl",
            "patch",
            "deployment",
            _required(args, "deployment"),
            "-n",
            _required(args, "namespace"),
            "--type",
            str(args.get("patch_type") or "merge"),
            "-p",
            patch_text,
        ])
    if tool == "scale_deployment":
        return _join([
            "kubectl",
            "scale",
            f"deploy/{_required(args, 'deployment')}",
            "-n",
            _required(args, "namespace"),
            f"--replicas={int(_required(args, 'replicas'))}",
        ])
    if tool == "delete_pod":
        return _join(["kubectl", "delete", "pod", _required(args, "pod"), "-n", _required(args, "namespace")])
    raise ValueError(f"unknown Kubernetes tool: {tool}")


def _required(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required arg: {key}")
    return str(value)


def _join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _mcp_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = getattr(part, "text", None)
            parts.append(str(text if text is not None else part))
        return "\n".join(parts)
    return str(result)


def _mcp_timeout_seconds(endpoint: str, tool: str, args: dict[str, Any]) -> float:
    if endpoint == "kubectl" and tool == "exec_kubectl_cmd_safely":
        cmd = str(args.get("cmd") or "")
        if " rollout status " in f" {cmd} " or cmd.startswith("kubectl wait "):
            return float(os.getenv("SREGYM_MCP_ROLLOUT_TIMEOUT_SECONDS", "75"))
    return float(os.getenv("SREGYM_MCP_CALL_TIMEOUT_SECONDS", "45"))


def _log_args(args: dict[str, Any]) -> str:
    return json.dumps(args, default=str)[:1000]


def _stage_timeout_seconds(stage: str) -> float:
    if stage == "diagnosis":
        return float(os.getenv("SREGYM_WORLDMODEL_DIAGNOSIS_TIMEOUT_SECONDS", os.getenv("SREGYM_WORLDMODEL_STAGE_TIMEOUT_SECONDS", "240")))
    return float(os.getenv("SREGYM_WORLDMODEL_MITIGATION_TIMEOUT_SECONDS", os.getenv("SREGYM_WORLDMODEL_STAGE_TIMEOUT_SECONDS", "420")))


async def _bounded_stage(stage: str, task: Any) -> SREGymStageResult:
    timeout = _stage_timeout_seconds(stage)
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except TimeoutError:
        message = f"StageTimeoutError: {stage} exceeded {timeout:.0f}s"
        logger.exception(message)
        return SREGymStageResult(
            answer="" if stage == "mitigation" else message,
            observations=[SREGymObservation("worldmodel.adapter", {"stage": stage}, message)],
        )


async def amain() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="Run Cerebral perception-world-model on SREGym")
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL_ID", "deepseek/deepseek-chat"))
    parser.add_argument("--logs-dir", default=os.getenv("AGENT_LOGS_DIR", "./logs/worldmodel"))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("WORLDMODEL_MAX_STEPS", "40")))
    args = parser.parse_args()

    wait_for_stage({"diagnosis", "mitigation"})
    app_info = api_json("/get_app")
    problem = api_json("/get_problem")
    problem_id = str(problem.get("problem_id") or "unknown")

    mcp = SreGymMcp()
    try:
        runner = SREGymWorldModelRuntime(
            provider=provider_from_env(args.model),
            mcp=mcp,
            max_steps=args.max_steps,
        )

        diagnosis = await _bounded_stage("diagnosis", runner.diagnose(app_info, problem))
        await mcp.submit(diagnosis.answer)

        mitigation = None
        if wait_for_stage({"mitigation", "done", "tearing_down"}) == "mitigation":
            mitigation = await _bounded_stage("mitigation", runner.mitigate(app_info, problem, diagnosis.answer))
            await mcp.submit(mitigation.answer)

        save_results(
            Path(args.logs_dir),
            problem_id,
            {
                "problem_id": problem_id,
                "adapter_role": "mcp_connect_metadata_submit_artifacts",
                "world_model_path": os.environ.get("CEREBRAL_WORLD_MODEL_PATH"),
                "world_model_runner": "cerebral_agents.sregym_runtime.SREGymWorldModelRuntime",
                "diagnosis": diagnosis.answer,
                "diagnosis_observations": [obs.__dict__ for obs in diagnosis.observations],
                "mitigation_answer": "" if mitigation is None else mitigation.answer,
                "mitigation_observations": [] if mitigation is None else [obs.__dict__ for obs in mitigation.observations],
            },
        )
    finally:
        await mcp.close()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
