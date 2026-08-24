"""OpenTelemetry tracing setup and the shared tracer.

Instrumented code (game/engine.py, agents/base.py) imports `tracer` from
here and uses it unconditionally. The OpenTelemetry API provides a no-op
tracer by default, so spans are inert (no export, negligible overhead)
until `configure_tracing()` is called — tracing is a pure add-on that
nothing about the core game or agent logic depends on.
"""

from __future__ import annotations

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
