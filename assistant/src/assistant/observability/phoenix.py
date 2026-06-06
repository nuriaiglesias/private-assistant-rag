from __future__ import annotations

"""Observability helpers for initializing OpenTelemetry tracing with Phoenix."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from assistant.core.config import AppConfig

_CONFIGURED = False


def configure_tracer(config: AppConfig) -> trace.Tracer:
    """Configure and return an OpenTelemetry tracer if Phoenix is enabled."""
    global _CONFIGURED
    if not config.phoenix_enabled:
        return trace.get_tracer("assistant.rag")

    if not _CONFIGURED:
        endpoint = f"http://{config.phoenix_host}:{config.phoenix_port}"
        resource = Resource.create({"service.name": config.phoenix_project})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _CONFIGURED = True

    return trace.get_tracer("assistant.rag")
