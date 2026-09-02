from __future__ import annotations

from dataclasses import dataclass

from .models import (
    EvidenceCandidate,
    EvidenceItemV3,
    EvidenceOccurrenceV3,
    EvidenceSelectionAudit,
)


@dataclass(frozen=True)
class EvidenceMergeResult:
    evidence: list[EvidenceItemV3]
    candidate_evidence_ids: list[str]
    new_evidence_ids: list[str]
    duplicate_ratio: float


def _occurrence_key(item: EvidenceOccurrenceV3) -> tuple:
    return (
        item.query_id,
        tuple(item.linked_question_ids),
        tuple(item.linked_element_ids),
        tuple(item.linked_patient_fact_ids),
        item.rank,
    )


def merge_evidence(
    *,
    existing: list[EvidenceItemV3],
    candidates: list[EvidenceCandidate],
) -> EvidenceMergeResult:
    ordered = list(existing)
    by_hash = {item.content_hash: item for item in ordered}
    next_number = max(
        [
            int(item.evidence_id[2:])
            for item in ordered
            if item.evidence_id.startswith("EV") and item.evidence_id[2:].isdigit()
        ]
        or [0]
    ) + 1
    candidate_ids: list[str] = []
    new_ids: list[str] = []
    duplicate_count = 0

    for candidate in candidates:
        current = by_hash.get(candidate.content_hash)
        if current is not None:
            duplicate_count += 1
            seen = {_occurrence_key(value) for value in current.occurrences}
            occurrences = list(current.occurrences)
            for occurrence in candidate.occurrences:
                if _occurrence_key(occurrence) not in seen:
                    occurrences.append(occurrence)
                    seen.add(_occurrence_key(occurrence))
            replacement = current.model_copy(update={"occurrences": occurrences})
            ordered[ordered.index(current)] = replacement
            by_hash[candidate.content_hash] = replacement
            candidate_ids.append(replacement.evidence_id)
            continue

        evidence_id = f"EV{next_number:03d}"
        next_number += 1
        parent_id = None
        if candidate.parent_content_hash:
            parent = by_hash.get(candidate.parent_content_hash)
            parent_id = parent.evidence_id if parent else None
        item = EvidenceItemV3(
            **candidate.model_dump(exclude={"parent_content_hash"}),
            evidence_id=evidence_id,
            parent_content_hash=candidate.parent_content_hash,
            parent_evidence_id=parent_id,
        )
        ordered.append(item)
        by_hash[item.content_hash] = item
        candidate_ids.append(evidence_id)
        new_ids.append(evidence_id)

    return EvidenceMergeResult(
        evidence=ordered,
        candidate_evidence_ids=list(dict.fromkeys(candidate_ids)),
        new_evidence_ids=new_ids,
        duplicate_ratio=(duplicate_count / len(candidates) if candidates else 0.0),
    )


def _estimated_tokens(text: str) -> int:
    compact = " ".join(text.split())
    return max((len(compact) + 1) // 2, 1)


def select_evidence(
    *,
    evidence: list[EvidenceItemV3],
    priority_evidence_ids: list[str],
    max_evidence: int,
    max_tokens: int,
) -> EvidenceSelectionAudit:
    by_id = {item.evidence_id: item for item in evidence}
    warnings: list[str] = []
    selected: list[str] = []
    selected_set: set[str] = set()
    token_count = 0

    def add(evidence_id: str) -> None:
        nonlocal token_count
        if evidence_id in selected_set or len(selected) >= max_evidence:
            return
        item = by_id.get(evidence_id)
        if item is None:
            warnings.append(f"忽略不存在的优先 Evidence ID：{evidence_id}")
            return
        item_tokens = _estimated_tokens(item.raw_text)
        if item_tokens > max_tokens:
            warnings.append(f"{evidence_id} 单片段超过最终证据预算，已跳过")
            return
        if token_count + item_tokens > max_tokens:
            return
        selected.append(evidence_id)
        selected_set.add(evidence_id)
        token_count += item_tokens

    for evidence_id in priority_evidence_ids:
        add(evidence_id)

    opened_ids = [
        item.evidence_id for item in evidence if item.source_method == "open"
    ]
    for evidence_id in opened_ids:
        add(evidence_id)

    query_ids: list[str] = []
    by_query: dict[str, list[tuple[int, str]]] = {}
    for item in evidence:
        for occurrence in item.occurrences:
            if occurrence.query_id not in by_query:
                by_query[occurrence.query_id] = []
                query_ids.append(occurrence.query_id)
            by_query[occurrence.query_id].append((occurrence.rank, item.evidence_id))
    for values in by_query.values():
        values.sort(key=lambda value: (value[0], value[1]))

    cursor = 0
    while len(selected) < max_evidence:
        progressed = False
        for query_id in query_ids:
            values = by_query[query_id]
            if cursor >= len(values):
                continue
            add(values[cursor][1])
            progressed = True
        if not progressed:
            break
        cursor += 1

    skipped = [
        item.evidence_id for item in evidence if item.evidence_id not in selected_set
    ]
    return EvidenceSelectionAudit(
        selected_evidence_ids=selected,
        priority_evidence_ids=[
            value for value in priority_evidence_ids if value in by_id
        ],
        opened_evidence_ids=opened_ids,
        skipped_evidence_ids=skipped,
        estimated_tokens=token_count,
        max_evidence=max_evidence,
        max_tokens=max_tokens,
        warnings=warnings,
    )
