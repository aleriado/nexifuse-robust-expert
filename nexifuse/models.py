"""Shared data models for the NexiFuse pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""

    instruction: str
    input: str = ""
    output: str = ""
    cot_trace: str | None = None
    domain: str = ""
    source_standard: str = ""
    version: str = ""

    def to_dict(self) -> dict:
        d = {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "domain": self.domain,
            "source_standard": self.source_standard,
            "version": self.version,
        }
        if self.cot_trace is not None:
            d["cot_trace"] = self.cot_trace
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainingExample:
        return cls(
            instruction=d.get("instruction", ""),
            input=d.get("input", ""),
            output=d.get("output", ""),
            cot_trace=d.get("cot_trace"),
            domain=d.get("domain", ""),
            source_standard=d.get("source_standard", ""),
            version=d.get("version", ""),
        )


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    role: str  # "user" or "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> ConversationTurn:
        return cls(role=d.get("role", ""), content=d.get("content", ""))


@dataclass
class ConversationExample:
    """A multi-turn conversation for interactive training."""

    turns: list[ConversationTurn] = field(default_factory=list)
    domain: str = ""
    scenario_type: str = ""
    source_standard: str = ""
    version: str = ""

    def to_dict(self) -> dict:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "domain": self.domain,
            "scenario_type": self.scenario_type,
            "source_standard": self.source_standard,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConversationExample:
        turns = [ConversationTurn.from_dict(t) for t in d.get("turns", [])]
        return cls(
            turns=turns,
            domain=d.get("domain", ""),
            scenario_type=d.get("scenario_type", ""),
            source_standard=d.get("source_standard", ""),
            version=d.get("version", ""),
        )


@dataclass
class DPOPair:
    """A preference pair for Direct Preference Optimization."""

    prompt: str
    chosen: str
    rejected: str

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DPOPair:
        return cls(
            prompt=d.get("prompt", ""),
            chosen=d.get("chosen", ""),
            rejected=d.get("rejected", ""),
        )


@dataclass
class ScrapedExample:
    """A code example scraped from a GitHub repository."""

    instruction: str
    input: str = ""
    output: str = ""
    source_repo: str = ""
    file_path: str = ""
    domain: str = ""

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source_repo": self.source_repo,
            "file_path": self.file_path,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScrapedExample:
        return cls(
            instruction=d.get("instruction", ""),
            input=d.get("input", ""),
            output=d.get("output", ""),
            source_repo=d.get("source_repo", ""),
            file_path=d.get("file_path", ""),
            domain=d.get("domain", ""),
        )


@dataclass
class ValidationResult:
    """Result of validating a single item."""

    passed: bool
    error_type: str | None = None  # "syntax", "structure", "security", None
    error_detail: str | None = None
    line_number: int | None = None


@dataclass
class ValidationReport:
    """Summary report from a batch validation run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors_by_category: dict[str, int] = field(default_factory=dict)


@dataclass
class CleaningStats:
    """Statistics from a data cleaning run."""

    input_rows: int = 0
    output_rows: int = 0
    duplicates_removed: int = 0
    malformed_skipped: int = 0
    empty_discarded: int = 0
    identity_filtered: int = 0
