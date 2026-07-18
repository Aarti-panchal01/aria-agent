"""
Pydantic schemas for ARIA's structured LLM outputs.

Using structured output (via ``llm.with_structured_output``) instead of
regex parsing makes the critic's judgement machine-readable and reliable.
"""

from pydantic import BaseModel, Field


class CriticScore(BaseModel):
    """
    Multi-dimensional quality assessment of a single research finding.

    Produced by the critic node via structured output. The ``overall`` score
    drives replanning decisions; the per-dimension scores and ``reasoning``
    are stored in the reasoning trace for transparency.
    """

    relevance: int = Field(
        ..., ge=0, le=10, description="How well the finding answers the subtask."
    )
    specificity: int = Field(
        ..., ge=0, le=10, description="How concrete and detailed the finding is."
    )
    source_quality: int = Field(
        ..., ge=0, le=10, description="Credibility and authority of the sources."
    )
    completeness: int = Field(
        ..., ge=0, le=10, description="How fully the subtask is covered."
    )
    overall: int = Field(
        ..., ge=0, le=10, description="Holistic 0-10 quality score."
    )
    reasoning: str = Field(
        ..., description="One-sentence justification for the scores."
    )
    replan_needed: bool = Field(
        ...,
        description="True if this subtask should be re-researched with a better query.",
    )
    replan_instruction: str = Field(
        default="",
        description=(
            "Specific, actionable instruction for how to improve the search "
            "for this subtask. Empty when replan_needed is False."
        ),
    )
