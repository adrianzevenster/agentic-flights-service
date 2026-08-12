from __future__ import annotations

import contextvars
import json
import logging

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get("")
        if rid:
            data["request_id"] = rid
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
