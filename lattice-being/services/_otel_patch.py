from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def init(service_name: str):
	provider = TracerProvider()
	processor = BatchSpanProcessor(OTLPSpanExporter())
	provider.add_span_processor(processor)
	trace.set_tracer_provider(provider)
	return trace.get_tracer(service_name)
