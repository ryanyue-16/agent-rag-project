from __future__ import annotations

from enum import StrEnum


class FailureCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    DEPENDENCY_ERROR = "dependency_error"
    NOT_READY = "not_ready"
    INTERNAL_ERROR = "internal_error"


STATUS_FAILURES: dict[int, tuple[FailureCode, bool]] = {
    500: (FailureCode.INTERNAL_ERROR, False),
    502: (FailureCode.DEPENDENCY_ERROR, True),
    404: (FailureCode.NOT_FOUND, False),
    409: (FailureCode.CONFLICT, False),
    413: (FailureCode.VALIDATION_ERROR, False),
    422: (FailureCode.VALIDATION_ERROR, False),
    429: (FailureCode.RATE_LIMITED, True),
    503: (FailureCode.NOT_READY, True),
    504: (FailureCode.TIMEOUT, True),
}
