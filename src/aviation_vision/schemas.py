from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ClassificationResult:
    label: str
    confidence: float
    second_label: str | None = None
    second_confidence: float | None = None
    accepted: bool = True
    ambiguous: bool = False

    @property
    def margin(self) -> float | None:
        if self.second_confidence is None:
            return None
        return self.confidence - self.second_confidence


@dataclass(slots=True)
class VisionObject:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float] | None = None
    mask_area_ratio: float | None = None


@dataclass(slots=True)
class AnalysisReport:
    requested_mode: str
    executed_model: str | None
    task: str | None
    classification: ClassificationResult | None = None
    objects: list[VisionObject] = field(default_factory=list)
    summary: str = ""
    knowledge: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    processing_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

