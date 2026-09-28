"""An OpenTelemetry collector on a local port, over OTLP/HTTP or OTLP/gRPC, that keeps what it gets.

`serving_http_collector()` answers `/v1/traces`, `/v1/metrics` and `/v1/logs` the way a collector's
OTLP/HTTP receiver does; `serving_grpc_collector()` runs the three OTLP gRPC services. Both decode
every export with the protobuf messages opentelemetry-proto ships and yield a `Collector`:
`spans()`, `metric_points(name)` and `log_records()` read what arrived as plain values, and
`headers` keeps the headers (gRPC metadata) of each export. `http_collector_env` and
`grpc_collector_env` are the environment of an instance exporting all three signals there, with
the batch and export intervals cut so a test waits well under a second for each.
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import pytest

from harness.instance import free_port
from harness.listener import ReceivedRequest, listening

SERVICE_NAME = "harbour-webui"
EXPORT_WAIT = 30.0
QUICK_EXPORTS = {
    "OTEL_BSP_SCHEDULE_DELAY": "200",
    "OTEL_BLRP_SCHEDULE_DELAY": "200",
    "OTEL_METRICS_EXPORT_INTERVAL_MILLIS": "500",
}


def _plain(value) -> Any:
    kind = value.WhichOneof("value")
    return getattr(value, kind) if kind else None


def _attributes(pairs) -> dict[str, Any]:
    return {pair.key: _plain(pair.value) for pair in pairs}


@dataclass
class Span:
    scope: str
    name: str
    trace_id: str
    attributes: dict[str, Any]
    status: int  # 0 unset, 1 ok, 2 error
    resource: dict[str, Any]


@dataclass
class LogRecord:
    scope: str
    body: Any
    trace_id: str
    attributes: dict[str, Any]


@dataclass
class Collector:
    endpoint: str = ""
    exports: list[tuple[str, Any]] = field(default_factory=list)  # (signal, decoded request)
    headers: list[dict[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, signal: str, message, headers: dict[str, str]) -> None:
        with self.lock:
            self.exports.append((signal, message))
            self.headers.append({name.lower(): value for name, value in headers.items()})

    def _of(self, signal: str) -> list:
        with self.lock:
            return [message for kind, message in self.exports if kind == signal]

    def spans(self) -> list[Span]:
        return [
            Span(
                scope=scoped.scope.name,
                name=span.name,
                trace_id=span.trace_id.hex(),
                attributes=_attributes(span.attributes),
                status=span.status.code,
                resource=_attributes(batch.resource.attributes),
            )
            for export in self._of("traces")
            for batch in export.resource_spans
            for scoped in batch.scope_spans
            for span in scoped.spans
        ]

    def metric_points(self, name: str) -> list[tuple[dict[str, Any], float]]:
        """(attributes, value) of every data point exported for the metric, oldest first."""
        points = []
        for export in self._of("metrics"):
            for batch in export.resource_metrics:
                for scoped in batch.scope_metrics:
                    for metric in scoped.metrics:
                        if metric.name != name:
                            continue
                        kind = metric.WhichOneof("data")
                        for point in getattr(metric, kind).data_points:
                            value = point.count if kind == "histogram" else _number(point)
                            points.append((_attributes(point.attributes), value))
        return points

    def log_records(self) -> list[LogRecord]:
        return [
            LogRecord(
                scope=scoped.scope.name,
                body=_plain(record.body),
                trace_id=record.trace_id.hex(),
                attributes=_attributes(record.attributes),
            )
            for export in self._of("logs")
            for batch in export.resource_logs
            for scoped in batch.scope_logs
            for record in scoped.log_records
        ]

    def wait_for(self, found: Callable[[], Any], what: str) -> Any:
        """Poll until `found()` returns something truthy, and return it."""
        deadline = time.monotonic() + EXPORT_WAIT
        while time.monotonic() < deadline:
            result = found()
            if result:
                return result
            time.sleep(0.2)
        pytest.fail(f"the collector never got {what}")


def _number(point) -> float:
    kind = point.WhichOneof("value")
    return getattr(point, kind) if kind else 0


def _messages():
    trace = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    metrics = pytest.importorskip("opentelemetry.proto.collector.metrics.v1.metrics_service_pb2")
    logs = pytest.importorskip("opentelemetry.proto.collector.logs.v1.logs_service_pb2")
    return {
        "traces": (trace.ExportTraceServiceRequest, trace.ExportTraceServiceResponse),
        "metrics": (metrics.ExportMetricsServiceRequest, metrics.ExportMetricsServiceResponse),
        "logs": (logs.ExportLogsServiceRequest, logs.ExportLogsServiceResponse),
    }


@contextlib.contextmanager
def serving_http_collector() -> Iterator[Collector]:
    messages = _messages()
    collector = Collector()
    with listening() as service:
        for signal, (request_type, _) in messages.items():

            def receive(request: ReceivedRequest, signal=signal, request_type=request_type):
                collector.record(signal, request_type.FromString(request.body), request.headers)
                return 200, {"Content-Type": "application/x-protobuf"}, b""

            service.route("POST", f"/v1/{signal}", receive)
        collector.endpoint = service.base_url
        yield collector


@contextlib.contextmanager
def serving_grpc_collector() -> Iterator[Collector]:
    grpc = pytest.importorskip("grpc")
    from opentelemetry.proto.collector.logs.v1 import logs_service_pb2_grpc
    from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2_grpc
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc

    messages = _messages()
    collector = Collector()

    def servicer(signal: str, base: type):
        response_type = messages[signal][1]

        class Receiver(base):
            def Export(self, request, context):  # noqa: N802
                collector.record(signal, request, dict(context.invocation_metadata()))
                return response_type()

        return Receiver()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(
        servicer("traces", trace_service_pb2_grpc.TraceServiceServicer), server
    )
    metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(
        servicer("metrics", metrics_service_pb2_grpc.MetricsServiceServicer), server
    )
    logs_service_pb2_grpc.add_LogsServiceServicer_to_server(
        servicer("logs", logs_service_pb2_grpc.LogsServiceServicer), server
    )
    port = free_port()
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    collector.endpoint = f"http://127.0.0.1:{port}"
    try:
        yield collector
    finally:
        server.stop(grace=None)


def http_collector_env(collector: Collector) -> dict[str, str]:
    return {
        "ENABLE_OTEL": "true",
        "ENABLE_OTEL_TRACES": "true",
        "ENABLE_OTEL_METRICS": "true",
        "ENABLE_OTEL_LOGS": "true",
        "OTEL_SERVICE_NAME": SERVICE_NAME,
        "OTEL_OTLP_SPAN_EXPORTER": "http",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"{collector.endpoint}/v1/traces",
        "OTEL_METRICS_EXPORTER_OTLP_ENDPOINT": f"{collector.endpoint}/v1/metrics",
        "OTEL_LOGS_EXPORTER_OTLP_ENDPOINT": f"{collector.endpoint}/v1/logs",
        **QUICK_EXPORTS,
    }


def grpc_collector_env(collector: Collector, username: str, password: str) -> dict[str, str]:
    return {
        "ENABLE_OTEL": "true",
        "ENABLE_OTEL_TRACES": "true",
        "ENABLE_OTEL_METRICS": "true",
        "ENABLE_OTEL_LOGS": "true",
        "OTEL_SERVICE_NAME": SERVICE_NAME,
        "OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint,
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "OTEL_BASIC_AUTH_USERNAME": username,
        "OTEL_BASIC_AUTH_PASSWORD": password,
        **QUICK_EXPORTS,
    }
