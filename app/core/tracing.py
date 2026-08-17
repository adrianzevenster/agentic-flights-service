"""OpenTelemetry setup.

Tracing is fully disabled (no-op) when OTEL_EXPORTER_ENDPOINT is blank,
so no spans are emitted and there is zero overhead in local dev.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NonRecordingSpan

from app.core.config import settings

_tracer: trace.Tracer | None = None


def configure_tracing(app=None) -> None:
    """Initialise the global OTel tracer and optionally instrument FastAPI.

    Args:
        app: FastAPI application instance. When provided and an OTLP endpoint
             is configured, HTTP requests are automatically instrumented.
    """
    global _tracer

    if not settings.OTEL_EXPORTER_ENDPOINT:
        _tracer = trace.get_tracer(__name__)  # returns a no-op tracer
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{settings.OTEL_EXPORTER_ENDPOINT}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(settings.OTEL_SERVICE_NAME)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)


def get_tracer() -> trace.Tracer:
    """Return the configured tracer (no-op when tracing is disabled).

    Returns:
        The process-wide OTel Tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(__name__)
    return _tracer
