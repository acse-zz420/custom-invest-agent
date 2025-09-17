from phoenix.otel import register
from opentelemetry import trace


print("--- Checking for OpenTelemetry Environment Variables ---")

collector_endpoint = "http://localhost:6006/v1/traces"

tracer_provider = register(
    endpoint=collector_endpoint,
    protocol="http/protobuf",
    project_name="Custom_Rag_Pipeline",
    batch=True,
    auto_instrument=True
)
tracer = trace.get_tracer("rag.pipline.tracer")

def shutdown_tracer():
    print("\nShutting down TracerProvider to send all pending traces...")
    tracer_provider.shutdown()
    print("All traces have been sent to Phoenix.")