#!/usr/bin/env bash
# Deploy (or tear down) the Cerebral stack — mongodb + dataplane + engine — into
# one kind cluster, wired for SREGym. One durable stack per worker cluster,
# reused across that worker's whole problem shard.
#
# The repo manifests target ghcr/production; this script overrides them to the
# locally-built images (kind-loaded) and points the engine at SREGym's app
# namespaces. Idempotent: safe to re-run.
#
# Usage:
#   DEEPSEEK_API_KEY=... bash deploy_stack.sh <kind-cluster-name>
#   bash deploy_stack.sh <kind-cluster-name> --teardown
#
# Env:
#   DEEPSEEK_API_KEY            required for deploy (engine LLM/RCA via DeepSeek)
#   CEREBRAL_MONOREPO           default /root/cerebral-monorepo (holds deploy/*.yaml)
#   CEREBRAL_ENGINE_IMAGE       default cerebral-engine:obs-test
#   CEREBRAL_DATAPLANE_IMAGE    default cerebral-dataplane:local
#   CEREBRAL_WATCH_NAMESPACES   default = SREGym app namespaces (comma list)
set -euo pipefail

CLUSTER="${1:-}"
[ -z "$CLUSTER" ] && { echo "usage: deploy_stack.sh <kind-cluster-name> [--teardown]"; exit 2; }
MODE="${2:-deploy}"

MONOREPO="${CEREBRAL_MONOREPO:-/root/cerebral-monorepo}"
ENGINE_IMAGE="${CEREBRAL_ENGINE_IMAGE:-cerebral-engine:obs-test}"
DATAPLANE_IMAGE="${CEREBRAL_DATAPLANE_IMAGE:-cerebral-dataplane:local}"
WATCH_NS="${CEREBRAL_WATCH_NAMESPACES:-hotel-reservation,astronomy-shop,social-network,train-ticket,blueprint-hotel-reservation,fleetcast,tidb-cluster}"
OBSERVER_MODEL="${CEREBRAL_OBSERVER_MODEL:-}"   # optional MESH_OBSERVER_MODEL override (e.g. deepseek-v4-pro)
OBSERVER_BASE_URL="${CEREBRAL_OBSERVER_BASE_URL:-}"   # optional MESH_OBSERVER_BASE_URL override (e.g. https://openrouter.ai/api)
OBSERVER_TIMEOUT="${CEREBRAL_OBSERVER_TIMEOUT:-}"   # optional MESH_OBSERVER_TIMEOUT_SECONDS override (slow providers)
CTX="kind-${CLUSTER}"
DEPLOY_DIR="${MONOREPO}/deploy"

k() { kubectl --context "$CTX" "$@"; }

if [ "$MODE" = "--teardown" ]; then
  echo "==> Tearing down cerebral namespace on $CTX"
  k delete namespace cerebral --ignore-not-found --wait=false || true
  exit 0
fi

[ -z "${DEEPSEEK_API_KEY:-}" ] && { echo "❌ DEEPSEEK_API_KEY not set"; exit 2; }

echo "==> [1/6] kind load images into $CLUSTER"
kind load docker-image "$ENGINE_IMAGE" "$DATAPLANE_IMAGE" --name "$CLUSTER"

echo "==> [2/6] namespace + manifests"
k create namespace cerebral --dry-run=client -o yaml | k apply -f -
# Apply dataplane (creates RBAC) + engine + mongodb. Namespace already exists,
# so the earlier apply-order race can't happen.
k apply -f "$DEPLOY_DIR/cerebral-dataplane.yaml" \
        -f "$DEPLOY_DIR/cerebral-engine.yaml" \
        -f "$DEPLOY_DIR/cerebral-mongodb.yaml"

echo "==> [3/6] deepseek secret"
k -n cerebral create secret generic cerebral-secrets \
  --from-literal=deepseek-api-key="$DEEPSEEK_API_KEY" \
  --dry-run=client -o yaml | k apply -f -

echo "==> [4/6] point deployments at local images (preserves env)"
k -n cerebral set image deploy/cerebral-engine engine="$ENGINE_IMAGE"
k -n cerebral set image deploy/cerebral-dataplane dataplane="$DATAPLANE_IMAGE"
# Strategic merge (by container name) so env/ports/probes survive; just flip
# imagePullPolicy so kind-loaded images aren't re-pulled from ghcr.
k -n cerebral patch deploy cerebral-engine \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","imagePullPolicy":"IfNotPresent"}]}}}}'
k -n cerebral patch deploy cerebral-dataplane \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"dataplane","imagePullPolicy":"IfNotPresent"}]}}}}'

echo "==> [5/6] watch SREGym app namespaces: $WATCH_NS"
k -n cerebral set env deploy/cerebral-engine MESH_KUBERNETES_ALLOWED_NAMESPACES="$WATCH_NS"
if [ -n "$OBSERVER_MODEL" ]; then
  echo "    observer model: $OBSERVER_MODEL"
  k -n cerebral set env deploy/cerebral-engine MESH_OBSERVER_MODEL="$OBSERVER_MODEL"
fi
if [ -n "$OBSERVER_BASE_URL" ]; then
  echo "    observer base_url: $OBSERVER_BASE_URL"
  k -n cerebral set env deploy/cerebral-engine MESH_OBSERVER_BASE_URL="$OBSERVER_BASE_URL"
fi
if [ -n "$OBSERVER_TIMEOUT" ]; then
  echo "    observer timeout: ${OBSERVER_TIMEOUT}s"
  k -n cerebral set env deploy/cerebral-engine MESH_OBSERVER_TIMEOUT_SECONDS="$OBSERVER_TIMEOUT"
fi

echo "==> [6/6] wait for rollouts"
k -n cerebral rollout status deploy/cerebral-mongodb --timeout=120s
k -n cerebral rollout status deploy/cerebral-dataplane --timeout=120s
k -n cerebral rollout status deploy/cerebral-engine --timeout=180s

echo "✅ cerebral stack ready on $CTX"
k -n cerebral get pods -o wide
