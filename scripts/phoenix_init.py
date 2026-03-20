"""
Phoenix Observability Initialization
=====================================

Lazy initialization of Arize Phoenix for RAG observability.
Import this module and call get_tracer() to instrument code.

USAGE
-----
    from phoenix_init import get_tracer, launch_phoenix
    
    # Start Phoenix dashboard (optional - auto-starts on first trace)
    launch_phoenix()
    
    # Get tracer for instrumentation
    tracer = get_tracer()
    
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("query", "test")
        # ... your code ...

CONFIGURATION
-------------
    PHOENIX_ENABLED (env): Set to "false" to disable tracing
    PHOENIX_PORT (env): Dashboard port (default: 6006)
"""

import os
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration
PHOENIX_ENABLED = os.environ.get("PHOENIX_ENABLED", "true").lower() != "false"
PHOENIX_PORT = int(os.environ.get("PHOENIX_PORT", "6006"))

# Global state
_phoenix_session = None
_tracer = None


class NoOpSpan:
    """No-op span for when Phoenix is disabled."""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def set_attribute(self, key, value):
        pass
    
    def set_status(self, status):
        pass
    
    def record_exception(self, exception):
        pass


class NoOpTracer:
    """No-op tracer for when Phoenix is disabled."""
    def start_as_current_span(self, name, **kwargs):
        return NoOpSpan()
    
    def start_span(self, name, **kwargs):
        return NoOpSpan()


def launch_phoenix(port: int = None) -> Optional[object]:
    """
    Launch Phoenix dashboard.
    
    Args:
        port: Dashboard port (default: PHOENIX_PORT env or 6006)
    
    Returns:
        Phoenix session object, or None if disabled/failed
    """
    global _phoenix_session
    
    if not PHOENIX_ENABLED:
        logger.info("[PHOENIX] Disabled via PHOENIX_ENABLED=false")
        return None
    
    if _phoenix_session is not None:
        logger.debug("[PHOENIX] Already running")
        return _phoenix_session
    
    port = port or PHOENIX_PORT
    
    try:
        import phoenix as px
        
        _phoenix_session = px.launch_app(port=port)
        logger.info(f"[PHOENIX] Dashboard started at http://localhost:{port}")
        return _phoenix_session
        
    except ImportError:
        logger.warning("[PHOENIX] arize-phoenix not installed. Run: pip install arize-phoenix")
        return None
    except Exception as e:
        logger.warning(f"[PHOENIX] Failed to launch: {e}")
        return None


@lru_cache(maxsize=1)
def _init_tracer():
    """Initialize OpenTelemetry tracer with Phoenix exporter."""
    if not PHOENIX_ENABLED:
        return NoOpTracer()
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        
        # Phoenix provides OTLP exporter
        import phoenix as px
        from phoenix.otel import register
        
        # Register Phoenix as the trace provider
        tracer_provider = register(
            project_name="alexandria",
            endpoint=f"http://localhost:{PHOENIX_PORT}/v1/traces"
        )
        
        # Get tracer
        tracer = trace.get_tracer("alexandria.rag")
        logger.info("[PHOENIX] Tracer initialized")
        return tracer
        
    except ImportError as e:
        logger.warning(f"[PHOENIX] Missing dependency: {e}. Using no-op tracer.")
        return NoOpTracer()
    except Exception as e:
        logger.warning(f"[PHOENIX] Tracer init failed: {e}. Using no-op tracer.")
        return NoOpTracer()


def get_tracer():
    """
    Get the OpenTelemetry tracer for Phoenix.
    
    Lazy initialization - first call will set up tracing.
    Returns no-op tracer if Phoenix is disabled or unavailable.
    
    Returns:
        OpenTelemetry Tracer or NoOpTracer
    """
    global _tracer
    
    if _tracer is None:
        _tracer = _init_tracer()
    
    return _tracer


def is_phoenix_enabled() -> bool:
    """Check if Phoenix tracing is enabled."""
    return PHOENIX_ENABLED


def get_phoenix_url() -> str:
    """Get Phoenix dashboard URL."""
    return f"http://localhost:{PHOENIX_PORT}"


# Convenience decorator for tracing functions
def traced(span_name: str = None):
    """
    Decorator to trace a function.
    
    Usage:
        @traced("my_operation")
        def my_function():
            ...
    """
    def decorator(func):
        name = span_name or func.__name__
        
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Test Phoenix initialization
    print(f"Phoenix enabled: {PHOENIX_ENABLED}")
    print(f"Phoenix port: {PHOENIX_PORT}")
    
    session = launch_phoenix()
    if session:
        print(f"Dashboard: {get_phoenix_url()}")
        
        # Test trace
        tracer = get_tracer()
        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test_key", "test_value")
            print("Test span created!")
    else:
        print("Phoenix not available")
