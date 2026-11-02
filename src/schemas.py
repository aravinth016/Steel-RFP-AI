"""Strict schemas for normalized supplier evaluation results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


CriterionStatus = Literal[
    "COMPLIANT",
    "PARTIALLY_COMPLIANT",
    "NON_COMPLIANT",
    "NOT_PROVIDED",
    "NOT_VERIFIABLE",
    "NOT_APPLICABLE",
    "PENDING_REVIEW",
    "NOT_COMPARABLE",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceCitation(StrictModel):
    source_id: StrictStr
    page: StrictInt = Field(gt=0)
    quote: StrictStr = Field(min_length=1)


class CriterionEvaluation(StrictModel):
    criterion_id: StrictStr
    score: StrictFloat = Field(ge=0)
    status: CriterionStatus
    justification: StrictStr
    evidence: StrictStr | None
    source_id: StrictStr | None
    page: StrictInt | None
    risks: list[StrictStr]
    clarifications: list[StrictStr]


class GateEvaluation(StrictModel):
    id: StrictStr
    status: Literal["Pass", "Fail", "Pending"]
    requirement: StrictStr
    reason: StrictStr
    evidence: StrictStr | None
    source_id: StrictStr | None
    page: StrictInt | None


class TechnicalOffer(StrictModel):
    material_description: StrictStr | None
    offered_grade: StrictStr | None
    offered_specification: StrictStr | None
    offered_quantity_mt: StrictFloat | None
    deviations: list[StrictStr]
    compliance_status: CriterionStatus
    evidence: list[EvidenceCitation]


class QualityDocument(StrictModel):
    document_type: StrictStr
    provided: StrictBool | None
    heat_or_batch_reference: StrictStr | None
    source_id: StrictStr | None
    page: StrictInt | None
    evidence_quote: StrictStr | None
    verification_status: Literal["NOT_INDEPENDENTLY_VERIFIED"]


class CommercialQuote(StrictModel):
    currency: StrictStr | None
    basic_rate_per_mt: StrictFloat | None
    freight_per_mt: StrictFloat | None
    tax_amount: StrictFloat | None
    loading_unloading_amount: StrictFloat | None
    other_charges: StrictFloat | None
    total_quoted_amount: StrictFloat | None
    delivery_basis: StrictStr | None
    quote_valid_until: StrictStr | None
    payment_terms: StrictStr | None
    exclusions: list[StrictStr]
    comparability_status: Literal[
        "COMPARABLE", "NOT_COMPARABLE", "PENDING_REVIEW", "NOT_PROVIDED"
    ]
    evidence: list[EvidenceCitation]


class DeliveryDetails(StrictModel):
    quantity_available_mt: StrictFloat | None
    first_delivery_date: StrictStr | None
    completion_date: StrictStr | None
    delivery_phases: list[StrictStr]
    delivery_location: StrictStr | None
    evidence: list[EvidenceCitation]


class EvaluationOutput(StrictModel):
    summary: StrictStr
    gates: list[GateEvaluation]
    criteria: list[CriterionEvaluation]
    technical_offer: TechnicalOffer
    quality_documents: list[QualityDocument]
    commercial_quote: CommercialQuote
    delivery: DeliveryDetails
