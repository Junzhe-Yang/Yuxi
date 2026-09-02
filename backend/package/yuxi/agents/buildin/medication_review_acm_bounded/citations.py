from __future__ import annotations

import hashlib
import re
from typing import Any

from yuxi.agents.buildin.medication_review_lite.models import EvidenceItem

from .models import (
    CitationClaimRecord,
    CitationEvidenceSnapshot,
    CitationVerificationRecord,
)

CITATION_REHYDRATION_VERSION = "bounded-citation-rehydration-v1"
_CITATION_PATTERN = re.compile(r"\[(EV-[A-Z0-9-]+)\]", re.IGNORECASE)


def build_citation_verification(
    draft: str,
    state: dict[str, Any],
) -> CitationVerificationRecord:
    evidence_store: dict[str, EvidenceItem] = {}
    for evidence_id, raw in (state.get("evidence_store") or {}).items():
        try:
            evidence_store[str(evidence_id).upper()] = (
                raw if isinstance(raw, EvidenceItem) else EvidenceItem.model_validate(raw)
            )
        except Exception:  # noqa: BLE001 - corrupt evidence remains auditable elsewhere
            continue

    cited_ids = list(dict.fromkeys(value.upper() for value in _CITATION_PATTERN.findall(draft)))
    unknown_ids = [value for value in cited_ids if value not in evidence_store]
    claims: list[CitationClaimRecord] = []
    for line in draft.splitlines():
        line_ids = list(dict.fromkeys(value.upper() for value in _CITATION_PATTERN.findall(line)))
        if not line_ids:
            continue
        claim_text = _CITATION_PATTERN.sub("", line).strip()
        if not claim_text:
            continue
        claims.append(
            CitationClaimRecord(
                claim_id=f"CLAIM-{len(claims) + 1:03d}",
                claim_text=claim_text,
                evidence_ids=line_ids,
            )
        )

    snapshots = [
        CitationEvidenceSnapshot(
            evidence_id=evidence_id,
            content_hash=evidence_store[evidence_id].content_hash,
            raw_text_sha256=hashlib.sha256(evidence_store[evidence_id].raw_text.encode("utf-8")).hexdigest(),
            raw_text=evidence_store[evidence_id].raw_text,
            source_document=evidence_store[evidence_id].source_document,
            file_id=evidence_store[evidence_id].file_id,
            chunk_id=evidence_store[evidence_id].chunk_id,
            chunk_index=evidence_store[evidence_id].chunk_index,
        )
        for evidence_id in cited_ids
        if evidence_id in evidence_store
    ]
    status = "unknown_evidence" if unknown_ids else "ready" if cited_ids else "no_citations"
    return CitationVerificationRecord(
        draft_hash=hashlib.sha256(draft.encode("utf-8")).hexdigest(),
        status=status,
        cited_evidence_ids=cited_ids,
        unknown_evidence_ids=unknown_ids,
        claims=claims,
        evidence_snapshots=snapshots,
    )
