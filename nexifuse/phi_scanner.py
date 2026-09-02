"""Standalone PHI safety scanner.

Extracted so both the inference server and the MCP server can reuse it. Deliberately has no
heavy dependencies (just ``re``): it runs offline, with no model or FastAPI stack required.
"""

from __future__ import annotations

import re

# Patterns that catch code which would log/print patient identifiers in plaintext.
PHI_UNSAFE_PATTERNS = [
    re.compile(r'console\.log\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'print\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'logger\.\w+\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'System\.out\.println\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'log\.(info|debug|warn|error)\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'printf?\([^)]*patient\.(ssn|name|mrn|dob|social|address|phone)', re.IGNORECASE),
    re.compile(r'console\.log\([^)]*\b(ssn|socialSecurity|social_security)\b', re.IGNORECASE),
]

PHI_WARNING = (
    "PHI Safety Warning: this code may log or print patient identifiers in plaintext. "
    "Always redact()/mask() PHI fields (SSN, MRN, name, DOB) in logs, error messages, "
    "and API responses."
)


def scan_phi(text: str) -> dict:
    """Structured scan.

    Returns ``{"safe": bool, "violations": [{"pattern": str, "count": int}], "warning": str|None}``.
    """
    violations = []
    for pattern in PHI_UNSAFE_PATTERNS:
        found = pattern.findall(text or "")
        if found:
            violations.append({"pattern": pattern.pattern[:80], "count": len(found)})
    safe = len(violations) == 0
    return {"safe": safe, "violations": violations, "warning": None if safe else PHI_WARNING}


def annotate_phi(response: str) -> tuple[bool, str]:
    """Back-compat helper for the inference server.

    Returns ``(is_safe, response)`` where ``response`` is annotated with a warning when unsafe.
    """
    result = scan_phi(response)
    if result["safe"]:
        return True, response
    return False, response + "\n\n> **" + PHI_WARNING + "**"
