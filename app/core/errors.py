"""Единый формат ошибок и глобальные обработчики исключений."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging_config import logger

HTTP_ERROR_CODES = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}

UNION_BRANCHES = ("FeatureVectorChurn", "list[FeatureVectorChurn]")


class ChurnError(Exception):
    """Ожидаемая ошибка сервиса: HTTP-статус + код + сообщение + подробности."""

    def __init__(self, status_code: int, code: str, message: str, details: Any = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_response(status_code: int, code: str, message: str, details: Any = None):
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "details": details},
    )


def clean_validation_errors(exc: RequestValidationError) -> list[dict]:
    """Превращает ошибки Pydantic в короткий список {field, message, type}."""
    errors = exc.errors()
    branch = None
    if isinstance(exc.body, dict):
        branch = UNION_BRANCHES[0]
    elif isinstance(exc.body, list):
        branch = UNION_BRANCHES[1]
    if branch and any(len(e["loc"]) > 1 and e["loc"][1] == branch for e in errors):
        errors = [e for e in errors if len(e["loc"]) > 1 and e["loc"][1] == branch]

    result = []
    for e in errors:
        loc = e["loc"]
        skip = 2 if len(loc) > 1 and loc[1] in UNION_BRANCHES else 1
        path = ""
        for part in loc[skip:]:
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
        if e["type"] == "json_invalid":
            path = ""
        result.append({"field": path or None, "message": e["msg"], "type": e["type"]})
    return result


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ChurnError)
    async def churn_error_handler(request: Request, exc: ChurnError):
        # Серверные ошибки (5xx) логируем как ERROR, клиентские (4xx) — как WARNING.
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        logger.log(level, "%s %s -> %s %s: %s", request.method, request.url.path,
                   exc.status_code, exc.code, exc.message)
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code = HTTP_ERROR_CODES.get(exc.status_code, f"HTTP_{exc.status_code}")
        logger.warning("%s %s -> %s %s", request.method, request.url.path, exc.status_code, code)
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        details = clean_validation_errors(exc)
        logger.warning("%s %s -> 422 VALIDATION_ERROR: %s", request.method, request.url.path, details)
        return error_response(
            422,
            "VALIDATION_ERROR",
            f"Входные данные не прошли проверку (ошибок: {len(details)})",
            details,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.error("%s %s -> 500 INTERNAL_ERROR", request.method, request.url.path, exc_info=exc)
        return error_response(500, "INTERNAL_ERROR", "Внутренняя ошибка сервиса")
