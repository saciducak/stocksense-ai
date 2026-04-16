"""FastAPI middleware for logging, error handling, and CORS.

Provides production-grade request processing pipeline with:
- Request/response logging with timing
- Structured error handling
- CORS configuration for dashboard access
"""

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_cors(app: FastAPI, origins: list[str] | None = None) -> None:
    """Configure CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance.
        origins: Allowed origin URLs. Defaults to localhost ports.
    """
    allowed_origins = origins or [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8080",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def request_logging_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """Log every request with method, path, status, and duration.

    Provides production visibility into API usage patterns
    and performance characteristics.
    """
    start_time = time.perf_counter()

    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        duration = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"❌ {request.method} {request.url.path} → 500 "
            f"({duration:.1f}ms) Error: {e}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(e),
                "status_code": 500,
            },
        )

    # Calculate duration
    duration = (time.perf_counter() - start_time) * 1000

    # Log with appropriate level based on status
    status = response.status_code
    if status >= 500:
        log_fn = logger.error
        emoji = "❌"
    elif status >= 400:
        log_fn = logger.warning
        emoji = "⚠️"
    else:
        log_fn = logger.info
        emoji = "✅"

    log_fn(
        f"{emoji} {request.method} {request.url.path} → {status} "
        f"({duration:.1f}ms)"
    )

    # Add timing header
    response.headers["X-Process-Time-Ms"] = f"{duration:.2f}"
    return response
