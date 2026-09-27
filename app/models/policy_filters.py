"""Explicit narrowing filters, never client-controlled access permissions."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200, strict=True)]
SourcePath = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024, strict=True)]


class PolicyMetadataFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    categories: list[Category] | None = Field(default=None, min_length=1, max_length=20)
    source_types: list[Literal["faq", "html", "pdf", "docx"]] | None = Field(default=None, min_length=1, max_length=4)
    source_paths: list[SourcePath] | None = Field(default=None, min_length=1, max_length=20)
