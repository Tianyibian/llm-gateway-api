from typing import Annotated, Literal
from pydantic import Field, model_validator
from app.models.graphrag import GraphContract


class EvidenceSelection(GraphContract):
    relevant: bool
    excerpt_ids: list[Annotated[int, Field(strict=True, ge=1)]] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent_selection(self):
        if self.relevant != bool(self.excerpt_ids):
            raise ValueError("Relevant evidence requires excerpt IDs")
        return self


class EvidenceMap(GraphContract):
    relevant: bool
    quotes: list[str] = Field(max_length=8)

    @model_validator(mode="after")
    def bounded_quotes(self):
        if self.relevant != bool(self.quotes):
            raise ValueError("Relevant evidence requires quotes")
        if any(not quote.strip() for quote in self.quotes) or sum(map(len, self.quotes)) > 2000:
            raise ValueError("Invalid quote budget")
        return self


class AnswerParagraph(GraphContract):
    text: str = Field(min_length=1, max_length=1800)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)


class ReducedAnswer(GraphContract):
    status: Literal["answered", "insufficient"]
    paragraphs: list[AnswerParagraph] = Field(max_length=12)

    @model_validator(mode="after")
    def consistent_answer(self):
        if (self.status == "answered") != bool(self.paragraphs):
            raise ValueError("Only answered results contain paragraphs")
        if any(not paragraph.text.strip() for paragraph in self.paragraphs):
            raise ValueError("Answer paragraphs cannot be blank")
        return self


class GeneratedGraphAnswer(GraphContract):
    answer: str
    status: Literal["complete", "insufficient", "unavailable"]
    reason_code: str
    mapped_evidence: int = 0
    cited_evidence_ids: list[str] = Field(default_factory=list)
    limited: bool = False
