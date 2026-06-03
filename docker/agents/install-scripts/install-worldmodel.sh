#!/usr/bin/env bash
set -euo pipefail

echo "[$(date -Iseconds)] Preparing Cerebral World Model adapter..."
python3 -m pip install --break-system-packages --upgrade openai
echo "[$(date -Iseconds)] Cerebral World Model MCP adapter is ready"
