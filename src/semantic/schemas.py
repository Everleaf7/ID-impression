from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EXPRESSION_TYPES = {"object", "character", "symbolic", "scene", "fantasy_hybrid"}


@dataclass
class Concept:
    id: str
    tokens: list[str]
    literal_meanings: list[str]
    abstract_concepts: list[str]
    associations: list[str]
    expression_type: str
    association_methods: list[str]
    association_distance: int
    detail_level: int
    main_concept: str
    main_subjects: list[str]
    secondary_elements: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    accent_colors: list[str] = field(default_factory=list)
    composition: str = ""
    image_prompt: str = ""

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("id must not be empty")
        if self.expression_type not in EXPRESSION_TYPES:
            raise ValueError(f"unsupported expression_type: {self.expression_type}")
        if not 0 <= self.association_distance <= 3:
            raise ValueError("association_distance must be in [0, 3]")
        if not 1 <= self.detail_level <= 4:
            raise ValueError("detail_level must be in [1, 4]")
        if not self.main_concept.strip():
            raise ValueError("main_concept must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Concept":
        allowed = set(cls.__dataclass_fields__)
        concept = cls(**{key: value for key, value in raw.items() if key in allowed})
        concept.validate()
        return concept


def validate_dataset_record(raw: dict[str, Any]) -> None:
    Concept.from_dict(raw)
    image_path = raw.get("image_path")
    if image_path is not None and not isinstance(image_path, str):
        raise ValueError("image_path must be a string or null")
    review_status = raw.get("review_status")
    if review_status is not None and not isinstance(review_status, str):
        raise ValueError("review_status must be a string")
