"""Error-text and HTTP-status classification shared by both adapters.

Adapters check their structured native signals first and fall back to this. The
TypeScript ``classify`` in ``providers/common.ts`` uses the same patterns and order;
``docs/fixtures/error-classification-v1.json`` holds the cases both must agree on.
"""

from __future__ import annotations

import re

TRANSIENT = "transient_api_error"

# JavaScript semantics, so non-ASCII text classifies as in TypeScript: ASCII \b, \d and
# case folding (re.ASCII), JavaScript's \s, and . stopping at JavaScript line terminators.
_FLAGS = re.ASCII | re.IGNORECASE
_SPACE = r"[\t\n\v\f\r    -     　﻿]"
_ANY = r"[^\n\r  ]"

# Status codes only count next to an HTTP marker, never as bare numbers.
_STATUS_PATTERNS = (
    re.compile(
        rf"\b(?:status(?: code)?|HTTP(?: status)?|API Error){_SPACE}*:?{_SPACE}*(\d{{3}})\b",
        _FLAGS,
    ),
    re.compile(
        r"\b(\d{3}) (?:Bad Request|Unauthorized|Payment Required|Forbidden|Not Found"
        r"|Too Many Requests|Internal Server Error|Bad Gateway|Service Unavailable"
        r"|Gateway Timeout)\b",
        _FLAGS,
    ),
)
# "upgrade to Plus" is Codex's text for a ChatGPT plan without Codex access.
_USAGE_LIMIT = re.compile(
    r"\busage limits?\b|\bquota exceeded\b|\binsufficient_quota\b"
    r"|\bexceeded your current quota\b|\bupgrade to (?:Plus|Pro)\b",
    _FLAGS,
)
_CONTEXT_WINDOW = re.compile(
    r"\bprompt is too long\b|\bcontext[_ ]length[_ ]exceeded\b|\bcontext[ _-]?window\b"
    r"|\bmaximum context length\b",
    _FLAGS,
)
_BILLING = re.compile(r"\bcredit balance\b|\bbilling\b", _FLAGS)
_AUTHENTICATION = re.compile(
    r"\bunauthorized\b|\bauthentication(?:_error)?\b|\binvalid[_ ](?:x-)?api[_ -]?key\b"
    r"|\bincorrect api key\b|\bnot logged in\b|\bmissing api key\b",
    _FLAGS,
)
_PERMISSION = re.compile(r"\bforbidden\b|\bpermission denied\b|\bpermission_error\b", _FLAGS)
# "Model provider `x` not found" is a configuration error, not a missing model.
_MODEL_NOT_FOUND = re.compile(
    r"\bmodel_not_found\b|\bunknown model\b"
    rf"|\bmodel\b(?! provider){_ANY}{{0,80}}?\b(?:not found|does not exist|is not supported)\b",
    _FLAGS,
)
_INVALID_REQUEST = re.compile(
    r"\binvalid_request_error\b|\binvalid prompt\b|\bbad request\b", _FLAGS
)
_TRANSIENT = re.compile(
    r"\brate[ _-]?limit|\boverloaded(?:_error)?\b|\bhigh (?:demand|load)\b"
    r"|\btemporarily unavailable\b|\bat capacity\b|\bserver (?:is )?busy\b|\bstream disconnected\b"
    r"|\b(?:connection|request) timed out\b|\bconnection (?:refused|reset|error)\b"
    r"|\bconnection closed before message completed\b"
    r"|\bConnectionRefused\b|\bECONNRESET\b|\bECONNREFUSED\b|\bETIMEDOUT\b",
    _FLAGS,
)


def status_in(message: str) -> int | None:
    for pattern in _STATUS_PATTERNS:
        match = pattern.search(message)
        if match:
            return int(match.group(1))
    return None


def classify(message: str, status: int | None = None) -> str | None:
    """Return an ``error_type`` for a failure message, or None when nothing matches.

    Quota, context and billing text outrank the status, since those arrive as 400 or
    429. Otherwise the status decides, then the text.
    """

    code = status if status is not None else status_in(message)
    if _USAGE_LIMIT.search(message):
        return "usage_limit_exceeded"
    if _CONTEXT_WINDOW.search(message):
        return "context_window_exceeded"
    if _BILLING.search(message):
        return "billing_error"
    if code is not None:
        if code in (408, 409, 429) or code >= 500:
            return TRANSIENT
        if code == 401:
            return "authentication_failed"
        if code == 402:
            return "billing_error"
        if code == 403:
            return "permission_denied"
    if _MODEL_NOT_FOUND.search(message):
        return "model_not_found"
    if _AUTHENTICATION.search(message):
        return "authentication_failed"
    if code is not None:
        return "invalid_request" if code in (400, 422) else f"api_error_{code}"
    if _PERMISSION.search(message):
        return "permission_denied"
    if _INVALID_REQUEST.search(message):
        return "invalid_request"
    if _TRANSIENT.search(message):
        return TRANSIENT
    return None
