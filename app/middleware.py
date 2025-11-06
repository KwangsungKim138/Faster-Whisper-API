import time
import uuid
import logging
from fastapi import Request, Response

logger = logging.getLogger("app.timing")

async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response | None = None

    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception:
        logging.getLogger("app.error").exception(
            "EXC req_id=%s method=%s path=%s", req_id, request.method, request.url.path
        )
        raise
    finally:
        dur_ms = (time.perf_counter() - start) * 1000.0
        status = getattr(response, "status_code", 500)
        logger.info(
            "TIMING req_id=%s method=%s path=%s status=%s dur_ms=%.1f",
            req_id, request.method, request.url.path, status, dur_ms
        )
