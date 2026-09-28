"""An OpenTelemetry collector's trace service over gRPC, the exporter Open WebUI uses by default.

`serving_trace_collector()` yields a `TraceCollector` answering the OTLP `TraceService.Export`
call. Every export it takes is kept with the time it arrived; `span_names()` lists the spans
they carried. `refuse_next(retry_after)` makes the next export fail the way a busy collector
does: `UNAVAILABLE`, with a `google.rpc.RetryInfo` in the trailing metadata that says when to
try again, which the exporter reads with googleapis-common-protos. `refusals` holds the times
of the refused exports. `trace_collector_env(collector)` is the environment of an instance that
exports its request traces there.
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import Iterator

import grpc
from google.protobuf.duration_pb2 import Duration
from google.rpc.error_details_pb2 import RetryInfo
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc

RETRY_INFO_KEY = "google.rpc.retryinfo-bin"


@dataclass
class TraceCollector(trace_service_pb2_grpc.TraceServiceServicer):
    address: str = ""
    exports: list[tuple[float, trace_service_pb2.ExportTraceServiceRequest]] = field(
        default_factory=list
    )
    refusals: list[float] = field(default_factory=list)
    pending_refusals: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def refuse_next(self, retry_after: float) -> None:
        with self.lock:
            self.pending_refusals.append(retry_after)

    def span_names(self) -> list[str]:
        with self.lock:
            return [
                span.name
                for _, request in self.exports
                for resource_spans in request.resource_spans
                for scope_spans in resource_spans.scope_spans
                for span in scope_spans.spans
            ]

    def first_export_after(self, moment: float) -> float | None:
        with self.lock:
            return next((arrived for arrived, _ in self.exports if arrived >= moment), None)

    def Export(self, request, context):
        arrived = time.monotonic()
        with self.lock:
            retry_after = self.pending_refusals.pop(0) if self.pending_refusals else None
            if retry_after is None:
                self.exports.append((arrived, request))
            else:
                self.refusals.append(arrived)
        if retry_after is not None:
            seconds = int(retry_after)
            delay = Duration(seconds=seconds, nanos=int((retry_after - seconds) * 1e9))
            context.set_trailing_metadata(
                ((RETRY_INFO_KEY, RetryInfo(retry_delay=delay).SerializeToString()),)
            )
            context.abort(grpc.StatusCode.UNAVAILABLE, "the collector is busy")
        return trace_service_pb2.ExportTraceServiceResponse()


@contextlib.contextmanager
def serving_trace_collector() -> Iterator[TraceCollector]:
    collector = TraceCollector()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(collector, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    collector.address = f"http://127.0.0.1:{port}"
    try:
        yield collector
    finally:
        server.stop(grace=None)


def trace_collector_env(collector: TraceCollector) -> dict[str, str]:
    return {
        "ENABLE_OTEL": "true",
        "ENABLE_OTEL_TRACES": "true",
        "OTEL_OTLP_SPAN_EXPORTER": "grpc",
        "OTEL_EXPORTER_OTLP_ENDPOINT": collector.address,
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "OTEL_BSP_SCHEDULE_DELAY": "200",
    }
