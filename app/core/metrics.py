"""Process-wide Prometheus metrics.

In single-worker mode the default registry is used directly.  For multi-worker
uvicorn deployments, set PROMETHEUS_MULTIPROC_DIR to a writable directory;
prometheus_client will aggregate per-worker .db files when the /metrics endpoint
calls generate_latest(registry).

Usage:
    from app.core.metrics import tool_cap_hits, make_registry
    tool_cap_hits.inc()
"""
from __future__ import annotations

import os

from prometheus_client import CollectorRegistry, Counter, multiprocess

tool_cap_hits: Counter = Counter(
    "flight_agent_tool_cap_hits_total",
    "Agentic turns that exhausted the tool iteration cap without producing a text response",
)


def make_registry() -> CollectorRegistry:
    """Return a registry that aggregates all worker metrics when in multiprocess mode."""
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    from prometheus_client import REGISTRY
    return REGISTRY
