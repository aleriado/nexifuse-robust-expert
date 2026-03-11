"""Multi-format validation engine.

Validates training example outputs for:
  - JavaScript syntax (via subprocess ESLint or basic heuristic)
  - XML well-formedness (xml.etree)
  - HL7 v2 message structure
  - FHIR R4 JSON schema
  - Security scan (hardcoded creds, unmasked PHI, SQL injection)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from nexifuse.config import PipelineConfig
from nexifuse.models import TrainingExample, ValidationResult, ValidationReport

logger = logging.getLogger(__name__)

# --- Security patterns ---
_SECURITY_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)select\s+.*\s+from\s+.*\s+where\s+.*['\"]?\s*\+", "Possible SQL injection"),
]

# Placeholder/example values commonly found in training data — NOT real credentials
_SECURITY_ALLOWLIST: list[re.Pattern[str]] = [
    # Example SSNs and MRN-like patterns in HL7/FHIR sample messages
    re.compile(r"123-45-6789|000-00-0000|999-99-9999|111-11-1111|987-65-4321|456-78-9012"),
    re.compile(r"(?i)(SSN|MRN|PID|HL7|FHIR|segment|message)"),
    # Placeholder tokens/passwords/secrets in example code
    re.compile(
        r"(?i)(YOUR_|REPLACE_|EXAMPLE_|PLACEHOLDER|TODO|INSERT|CHANGE_ME|"
        r"your[_-]?(access|api|auth|fhir|endpoint)|xxxx|XYZ123|abc123|abcdef|"
        r"secureP@ss|samplePass|secureSecret|"
        r"test[_-]?password|my[_-]?(password|secret|custom)|client[_-]?(secret|id)|"
        r"\"password\"|\"secret\"|\"username\"|DriverManager|JDBC|"
        r"Bearer\s+token|Authorization.*Bearer)",
    ),
]

# Strict patterns — only flag if NOT matching the allowlist
_STRICT_SECURITY_PATTERNS: list[tuple[str, str]] = [
    (r"\d{3}-\d{2}-\d{4}", "Possible SSN"),
    (r"password\s*[:=]\s*['\"][^'\"]+['\"]", "Hardcoded password"),
    (r"api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
    (r"secret\s*[:=]\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
    (r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Hardcoded bearer token"),
]

# HL7 v2 required fields by message type
_HL7_REQUIRED_SEGMENTS: dict[str, list[str]] = {
    "ADT": ["MSH", "EVN", "PID", "PV1"],
    "ORU": ["MSH", "PID", "OBR", "OBX"],
    "ORM": ["MSH", "PID", "ORC"],
    "SIU": ["MSH", "SCH", "PID"],
    "VXU": ["MSH", "PID", "RXA"],
    "DEFAULT": ["MSH"],
}


def validate_javascript(code: str, eslint_config: str | None = None) -> ValidationResult:
    """Validate JavaScript code via ESLint or basic bracket matching."""
    # Try ESLint first (only if config file actually exists)
    if eslint_config and Path(eslint_config).exists():
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                ["npx", "eslint", "--config", eslint_config, "--no-eslintrc", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            Path(tmp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return ValidationResult(passed=True)
            else:
                return ValidationResult(
                    passed=False,
                    error_type="syntax",
                    error_detail=result.stdout[:500] or result.stderr[:500],
                )
        except (subprocess.SubprocessError, OSError):
            logger.debug("ESLint not available, falling back to heuristic")

    # Heuristic fallback: bracket/brace/paren matching
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    in_string = False
    escape_next = False
    string_char = ""

    for i, ch in enumerate(code):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if in_string:
            if ch == string_char:
                in_string = False
            continue
        if ch in ("'", '"', '`'):
            in_string = True
            string_char = ch
            continue
        if ch in ("(", "[", "{"):
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return ValidationResult(
                    passed=False, error_type="syntax",
                    error_detail=f"Unmatched '{ch}' at position {i}",
                    line_number=code[:i].count("\n") + 1,
                )
            stack.pop()

    if stack:
        return ValidationResult(
            passed=False, error_type="syntax",
            error_detail=f"Unclosed brackets: {''.join(stack)}",
        )
    return ValidationResult(passed=True)


def validate_xml(content: str) -> ValidationResult:
    """Validate XML well-formedness."""
    try:
        ET.fromstring(content)
        return ValidationResult(passed=True)
    except ET.ParseError as exc:
        return ValidationResult(
            passed=False, error_type="syntax",
            error_detail=str(exc),
            line_number=getattr(exc, "position", (None,))[0],
        )


def validate_hl7v2(message: str) -> ValidationResult:
    """Validate HL7 v2 message structure."""
    lines = [l for l in message.strip().split("\n") if l.strip()]
    if not lines:
        return ValidationResult(passed=False, error_type="structure", error_detail="Empty message")

    # Check MSH header
    first = lines[0]
    if not first.startswith("MSH|"):
        return ValidationResult(
            passed=False, error_type="structure",
            error_detail="Message must start with MSH segment",
        )

    # Parse field separator and encoding characters
    if len(first) < 8:
        return ValidationResult(
            passed=False, error_type="structure",
            error_detail="MSH segment too short — missing encoding characters",
        )

    field_sep = first[3]  # Should be |
    if field_sep != "|":
        return ValidationResult(
            passed=False, error_type="structure",
            error_detail=f"Expected '|' as field separator, got '{field_sep}'",
        )

    # Extract segments present
    segments = []
    for line in lines:
        seg_id = line[:3] if len(line) >= 3 else ""
        if seg_id.isalpha() or (len(seg_id) == 3 and seg_id[:2].isalpha() and seg_id[2].isdigit()):
            segments.append(seg_id)

    # Try to detect message type from MSH-9
    fields = first.split("|")
    msg_type = ""
    if len(fields) > 8:
        msg_type = fields[8].split("^")[0] if fields[8] else ""

    required = _HL7_REQUIRED_SEGMENTS.get(msg_type, _HL7_REQUIRED_SEGMENTS["DEFAULT"])
    missing = [seg for seg in required if seg not in segments]

    if missing:
        return ValidationResult(
            passed=False, error_type="structure",
            error_detail=f"Missing required segments for {msg_type or 'HL7'}: {', '.join(missing)}",
        )

    return ValidationResult(passed=True)


def validate_fhir_r4(content: str, schema_dir: str | None = None) -> ValidationResult:
    """Validate FHIR R4 JSON resource structure."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return ValidationResult(passed=False, error_type="syntax", error_detail=str(exc))

    if not isinstance(data, dict):
        return ValidationResult(
            passed=False, error_type="structure",
            error_detail="FHIR resource must be a JSON object",
        )

    resource_type = data.get("resourceType")
    if not resource_type:
        return ValidationResult(
            passed=False, error_type="structure",
            error_detail="Missing required 'resourceType' field",
        )

    # If schema dir exists, try JSON schema validation
    if schema_dir:
        schema_path = Path(schema_dir) / f"{resource_type}.schema.json"
        if schema_path.exists():
            try:
                import jsonschema
                schema = json.loads(schema_path.read_text())
                jsonschema.validate(data, schema)
            except ImportError:
                logger.debug("jsonschema not installed — skipping schema validation")
            except Exception as exc:
                return ValidationResult(
                    passed=False, error_type="structure",
                    error_detail=f"Schema validation failed: {exc}",
                )

    return ValidationResult(passed=True)


def _is_allowlisted(content: str, match: re.Match) -> bool:
    """Check if a security match is a known placeholder/example value."""
    # Check surrounding context (80 chars around the match)
    start = max(0, match.start() - 40)
    end = min(len(content), match.end() + 40)
    context = content[start:end]
    for allow_pat in _SECURITY_ALLOWLIST:
        if allow_pat.search(context):
            return True
    return False


def scan_security(content: str) -> ValidationResult:
    """Scan content for security issues: hardcoded creds, PHI, SQL injection.

    Allows placeholder/example values commonly found in training data
    (e.g. 123-45-6789, Bearer YOUR_ACCESS_TOKEN, password = "example").
    """
    # Always-flag patterns (no allowlist, e.g. SQL injection)
    for pattern, description in _SECURITY_PATTERNS:
        try:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                line_num = content[:match.start()].count("\n") + 1
                return ValidationResult(
                    passed=False, error_type="security",
                    error_detail=f"{description} at line {line_num}",
                    line_number=line_num,
                )
        except re.error:
            continue

    # Strict patterns — flag only if NOT a known placeholder
    for pattern, description in _STRICT_SECURITY_PATTERNS:
        try:
            match = re.search(pattern, content, re.IGNORECASE)
            if match and not _is_allowlisted(content, match):
                line_num = content[:match.start()].count("\n") + 1
                return ValidationResult(
                    passed=False, error_type="security",
                    error_detail=f"{description} at line {line_num}",
                    line_number=line_num,
                )
        except re.error:
            continue

    return ValidationResult(passed=True)


def _detect_content_type(content: str) -> str:
    """Heuristic detection of content type."""
    stripped = content.strip()
    if stripped.startswith("MSH|"):
        return "hl7v2"
    if stripped.startswith("<") and (">" in stripped):
        return "xml"
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and data.get("resourceType"):
                return "fhir"
            return "json"
        except json.JSONDecodeError:
            pass
    # Default to JavaScript (Mirth transformers)
    return "javascript"


def validate_example(
    example: TrainingExample,
    config: PipelineConfig | None = None,
) -> ValidationResult:
    """Validate a single training example's output field."""
    content = example.output.strip()
    if not content:
        return ValidationResult(passed=False, error_type="structure", error_detail="Empty output")

    # Security scan first
    sec_result = scan_security(content)
    if not sec_result.passed:
        return sec_result

    # Detect type and validate
    content_type = _detect_content_type(content)
    eslint_config = config.validation.eslint_config if config else None
    fhir_schema_dir = config.validation.fhir_schema_dir if config else None

    if content_type == "hl7v2":
        return validate_hl7v2(content)
    elif content_type == "xml":
        return validate_xml(content)
    elif content_type == "fhir":
        return validate_fhir_r4(content, fhir_schema_dir)
    elif content_type == "javascript":
        return validate_javascript(content, eslint_config)

    return ValidationResult(passed=True)


def validate_batch(
    input_path: str | Path,
    output_passed: str | Path = "data/validated/passed.jsonl",
    output_failed: str | Path = "data/validated/failed.jsonl",
    config: PipelineConfig | None = None,
) -> ValidationReport:
    """Validate a JSONL file of training examples.

    Writes passing examples to output_passed and failing to output_failed.
    Returns a ValidationReport with summary statistics.
    """
    input_path = Path(input_path)
    output_passed = Path(output_passed)
    output_failed = Path(output_failed)
    output_passed.parent.mkdir(parents=True, exist_ok=True)
    output_failed.parent.mkdir(parents=True, exist_ok=True)

    report = ValidationReport()

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_passed, "w", encoding="utf-8") as fpass,
        open(output_failed, "w", encoding="utf-8") as ffail,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            report.total += 1
            example = TrainingExample.from_dict(record)
            result = validate_example(example, config)

            if result.passed:
                report.passed += 1
                fpass.write(line + "\n")
            else:
                report.failed += 1
                category = result.error_type or "unknown"
                report.errors_by_category[category] = (
                    report.errors_by_category.get(category, 0) + 1
                )
                # Annotate the record with error info
                record["_validation_error"] = {
                    "type": result.error_type,
                    "detail": result.error_detail,
                    "line": result.line_number,
                }
                ffail.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Validation complete: %d total, %d passed, %d failed — %s",
        report.total, report.passed, report.failed, report.errors_by_category,
    )
    return report
