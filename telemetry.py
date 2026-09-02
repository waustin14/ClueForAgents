"""OpenTelemetry tracing setup and the shared tracer.

Instrumented code (game/engine.py, agents/base.py) imports `tracer` from
here and uses it unconditionally. The OpenTelemetry API provides a no-op
tracer by default, so spans are inert (no export, negligible overhead)
until `configure_tracing()` is called — tracing is a pure add-on that
nothing about the core game or agent logic depends on.
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import trace

tracer = trace.get_tracer("clueforagents")


def configure_tracing(service_name: str = "clueforagents", **resource_attributes: Any) -> None:
    """Installs an OTLP-exporting TracerProvider as the global default.

    Reads the collector endpoint/protocol/headers from the standard
    OTEL_EXPORTER_OTLP_* environment variables (OTEL_EXPORTER_OTLP_ENDPOINT
    defaults to http://localhost:4317 if unset) rather than defining a
    parallel configuration surface on top of what the SDK already reads.

    Requires the optional `otel` dependency group (`uv sync --extra otel`)
    — raises ImportError if it isn't installed. Callers that want tracing
    to be a soft opt-in should catch that rather than depending on the
    SDK being present.
    """
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name, **resource_attributes})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def configure_langsmith_tracing(project_name: str = "clueforagents") -> None:
    """Turns on LangSmith tracing for every LangChain-backed LLM call.

    Every provider in `agents/langchain_client.py` is a LangChain
    `BaseChatModel`, so once `langsmith` is installed and tracing is
    enabled, each `model.ainvoke(...)` call inside `LangChainLLMClient.complete`
    is captured automatically through LangChain's global callback manager —
    no per-provider or per-call-site instrumentation needed, the same
    "instrument once at the root" shape as `configure_tracing()` above, just
    for LLM I/O (prompts, completions, token usage, latency) instead of
    game/agent spans.

    Only fills in `LANGSMITH_TRACING` and `LANGSMITH_PROJECT` when unset, so
    an operator's existing environment always wins. `LANGSMITH_API_KEY` and
    `LANGSMITH_ENDPOINT` are read straight from the environment rather than
    accepted as parameters here, for the same reason `configure_tracing()`
    doesn't wrap `OTEL_EXPORTER_OTLP_*`: no parallel configuration surface.

    Requires the optional `langsmith` dependency group (`uv sync --extra
    langsmith`) — raises ImportError if it isn't installed. Callers that
    want tracing to be a soft opt-in should catch that rather than
    depending on the SDK being present.
    """
    import langsmith  # noqa: F401  (import only to assert the extra is installed)

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project_name)
