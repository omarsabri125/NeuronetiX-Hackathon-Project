from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Define HTTP metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint'])

# Define ML metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total ML Predictions', ['prediction'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'ML Prediction Latency')
CHURN_PROBABILITY = Histogram('ml_churn_probability', 'Churn Probability Distribution', buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PREDICTION_ERRORS = Counter('ml_prediction_errors_total', 'Total ML Prediction Errors', ['error_type'])
# Feature value gauges
FEATURE_TENURE = Gauge('ml_feature_tenure', 'Current Tenure Value')
FEATURE_MONTHLY_CHARGES = Gauge('ml_feature_monthly_charges', 'Current Monthly Charges Value')
FEATURE_TOTAL_CHARGES = Gauge('ml_feature_total_charges', 'Current Total Charges Value')

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        endpoint = request.url.path

        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()

        return response


def setup_metrics(app: FastAPI):

    app.add_middleware(PrometheusMiddleware)

    @app.get("/TrhBVe_m5gg2002_E5VVqS", include_in_schema=False)
    def metrics():
        return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
