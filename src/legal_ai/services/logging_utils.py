import json
import logging
from typing import Any, Dict

class JsonLogger:
    def __init__(self, name: str = "legal_rag"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg: str, **kwargs: Any) -> None:
        record: Dict[str, Any] = {"level": "info", "message": msg}
        record.update(kwargs)
        self.logger.info(json.dumps(record, ensure_ascii=False))

    def error(self, msg: str, **kwargs: Any) -> None:
        record: Dict[str, Any] = {"level": "error", "message": msg}
        record.update(kwargs)
        self.logger.error(json.dumps(record, ensure_ascii=False))
