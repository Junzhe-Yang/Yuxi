# Executive assessment

Your proposal is **well aligned with the current design direction of mainstream agent frameworks**. Its central ideas have strong precedents:

1. A complete, durable execution record is retained for audit and recovery.
2. Each model call receives a newly constructed, bounded view of that record.
3. Typed application state serves as the operational source of truth.
4. Tool implementations obtain controller-known data through hidden runtime context.
5. The model only supplies variables that genuinely require semantic judgment.
6. Tool availability, retries, output budgets, and termination conditions are controlled outside free-form model reasoning.

The proposal therefore constitutes a **domain-specific control layer built on Yuxi and LangGraph**, rather than an ungrounded general-purpose agent framework.

The scale of the modification is also justified by the observed failures. Your trace shows model input growing from 8,666 to 135,533 tokens, 84,299 tokens of accumulated historical model output before the pathological turn, and a single 9,991-token response dominated by repetitive pre-tool language. The tool problem is similarly structural. Of 78 named calls, 32 were rejected at the business-semantic level, with most failures arising from the model reproducing controller-known state or reconstructing evidence ownership and provenance.

The older trace also contains a particularly important observability defect. A judgment tool returns `ok: false` with an `EvidenceContractError`, while the outer tool-call record still reports `status: success`. This makes a structured semantic outcome layer necessary.

My principal recommendations are:

- Keep the independent bounded agent and freeze the existing ACM-PRIM agent.
- Keep the Audit Transcript, Evidence Store, Working Ledger, Bounded Model View, ActionDirective, narrow tools, structured outcomes, generation guard, and citation rehydration.
- Replace the proposed “recent 3 to 5 interactions” rule with a relevance-first, scope-keyed selection policy.
- Treat the 256k context as provider capacity. Use much smaller phase-specific operational budgets.
- Use static narrow tool families with hidden runtime binding. Generate dynamic enums only for small candidate sets and only after provider compatibility testing.
- Keep Yuxi platform services, checkpointing, conversation persistence, Milvus, Atlas, and the existing evidence store unchanged.

------

# 1. What mainstream frameworks currently do

## 1.1 LangChain and LangGraph

LangChain now explicitly separates **transient model context** from **persistent tool and lifecycle context**. Messages, prompts, available tools, model selection, and output format can be changed for one model call without rewriting the persisted state. Middleware is the standard extension surface for this operation. Tools can read state, runtime context, and stores through runtime injection. Dynamic tool filtering is also a first-class pattern. ([Docs by LangChain](https://docs.langchain.com/oss/python/langchain/context-engineering))

Its prebuilt middleware includes summarization, context editing, tool selection, model and tool call limits, tool error conversion, and bounded retry behavior. Tool errors can be represented as `ToolMessage(status="error")`, while retry middleware can be limited by tool and exception category. ([Docs by LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in))

This maps directly to your design:

| Your component          | LangChain/LangGraph analogue                             |
| ----------------------- | -------------------------------------------------------- |
| Bounded Model View      | Transient model context through `wrap_model_call`        |
| Working Ledger          | Typed graph state                                        |
| Evidence Store          | State/store/artifact-backed context                      |
| ActionDirective         | Controller-derived state used by middleware              |
| Dynamic legal tools     | Model-request tool filtering                             |
| Prebound arguments      | `ToolRuntime` and runtime state                          |
| Typed state transition  | `Command` or reducer updates                             |
| Structured tool failure | Error `ToolMessage` and tool middleware                  |
| Context budget record   | Model-request instrumentation and token-aware middleware |

Yuxi already relies on `create_agent`, middleware, graph state, and checkpointing. Its generic loop lets the model decide whether to answer or invoke tools, so hard behavioral constraints naturally belong in middleware, tools, and the custom agent state layer.

## 1.2 OpenAI Agents SDK

OpenAI Agents SDK sessions preserve full conversation history by default. Before each run, stored history is prepended to the new turn. The SDK provides `session_input_callback` for filtering, reordering, or selecting historical items without rewriting the stored session. It also exposes a `call_model_input_filter` that operates on the fully prepared input immediately before each model call. ([OpenAI GitHub](https://openai.github.io/openai-agents-python/sessions/))

The SDK includes a `ToolOutputTrimmer` that keeps recent turns at full fidelity and replaces bulky older tool outputs with compact previews. This is close to your deterministic receipt design, although your clinical workflow needs a richer typed receipt because provenance and support state matter. ([OpenAI GitHub](https://openai.github.io/openai-agents-python/ref/extensions/tool_output_trimmer/))

OpenAI also separates local application context from LLM-visible history. The local context object is available to tools and lifecycle hooks and remains outside the model prompt. Capability visibility can be controlled dynamically, while authorization and argument validation remain inside tool implementations or guardrails. ([OpenAI GitHub](https://openai.github.io/openai-agents-python/context/))

Tool use can be set to `auto`, `required`, `none`, or a specific named tool. The SDK resets forced tool choice after a call by default to reduce tool-use loops. ([OpenAI GitHub](https://openai.github.io/openai-agents-python/agents/))

This supports four parts of your proposal:

- Controller-known identifiers belong in local runtime context.
- The model-facing schema should carry only semantically open variables.
- Action rounds may expose one legal tool and force its use.
- Historical tool results may be replaced with a compact, replay-safe representation.

## 1.3 Anthropic Claude Platform

Anthropic recommends server-side compaction as the primary long-horizon strategy and provides context editing for more selective control. Its documentation explicitly treats context as a finite resource and warns that irrelevant material degrades model focus. ([Claude Platform](https://platform.claude.com/docs/en/build-with-claude/context-editing))

For tool-heavy agents, Anthropic can clear old tool results once a token threshold is reached. Cleared results are replaced by placeholders. Recent tool-use and result pairs remain intact, and selected tools can be excluded from clearing. Its default configuration retains recent interactions rather than deleting the entire tool history indiscriminately. ([Claude Platform](https://platform.claude.com/docs/en/build-with-claude/context-editing))

Anthropic also offers on-demand tool search, allowing a large catalog to remain outside the immediate model context until relevant definitions are discovered. This follows the same principle as your phase-specific tool surface. ([Claude Platform](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool))

Your evidence rehydration design follows the same broader pattern. Large artifacts remain in durable storage, while the model receives exact content only when the active action requires it.

## 1.4 Google Agent Development Kit

Google ADK collects user instructions, retrieved data, tool responses, and generated model content as session context. Its compaction system supports token-triggered and turn-triggered reduction, with token-based compaction taking priority. A configurable tail of recent events remains in raw form after older events are compacted. ([Adk](https://adk.dev/context/compaction/))

ADK also exposes callbacks before model and tool operations. These callbacks receive session or tool context and can modify, intercept, or replace operations. State changes are persisted through the event and session model. ([Adk](https://adk.dev/callbacks/types-of-callbacks/))

The important correspondence is that ADK treats compaction, state, model callbacks, and tool callbacks as separate concerns. Your proposal makes the same separation through the Working Ledger, Context View, Controller, and ToolOutcome.

## 1.5 Microsoft AutoGen

AutoGen offers several interchangeable model-context implementations. These include an unbounded context, a last-N-message buffer, a token-limited recent context, and a head-and-tail context. It also permits custom context implementations. ([Microsoft GitHub](https://microsoft.github.io/autogen/stable/reference/python/autogen_core.model_context.html))

AutoGen demonstrates that mainstream frameworks usually provide a **context policy interface**, while the application selects an appropriate policy. Its built-in policies remain largely structural, based on recency or token count. Clinical relevance, evidence ownership, and investigation-specific selection remain application responsibilities.

## 1.6 Microsoft Semantic Kernel

Semantic Kernel provides an `IChatHistoryReducer` interface and built-in truncation and summarization reducers. These reducers preserve system messages and support configurable target and trigger sizes. ([Microsoft Learn](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/chat-history))

Semantic Kernel also includes experimental contextual function selection. It embeds recent conversation context and tool descriptions, then advertises only the most relevant functions to the model. The context used for function selection can itself be filtered or rewritten. ([Microsoft Learn](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-contextual-function-selection))

For your workflow, deterministic phase routing remains preferable to embedding-based tool selection because the legal action follows directly from typed state. Semantic tool search becomes useful only if the tool catalog later grows substantially.

## 1.7 Pydantic AI

Pydantic AI provides a `ProcessHistory` capability that intercepts message history before each model request. It supports recency filters, context-aware filters, and model-generated summaries. Its documentation explicitly warns that tool calls and corresponding results must remain paired after slicing or summarization. ([Pydantic Docs](https://ai.pydantic.dev/message-history/))

This supports your proposal to project AI and Tool messages as atomic interaction units and to keep malformed or aborted outputs outside subsequent model views.

------

# 2. The central architectural pattern

Across these frameworks, the strongest convergence can be summarized as four planes:

```text
Durable Audit Plane
  Complete user, model, tool, controller, retry, rejection, and abort events

Operational State Plane
  Typed task state, obligations, budgets, ownership, provenance, and next actions

Artifact and Evidence Plane
  Large immutable results, exact source text, hashes, files, and rehydration handles

Transient Model Plane
  A bounded, action-specific projection assembled immediately before each model call
```

Your proposed architecture already follows this pattern almost exactly. The document distinguishes the Audit Transcript, Evidence Store, Working Ledger, and Bounded Model View, with the model view reconstructed for each call.

A useful conceptual distinction is:

### Decision context

This is the information the model receives while deciding what semantic action or tool argument to produce. It includes the current directive, relevant ledger state, selected historical interactions, evidence, and currently available tool schemas.

### Execution context

This is the information available to the tool implementation after a call has been emitted. It includes the active investigation, exact obligation, candidate ownership, provenance graph, budgets, backend clients, and controller state.

Mainstream frameworks increasingly place execution context in code-local runtime objects. Consequently, a tool should rarely require the model to reproduce the full state that authorizes and parameterizes its execution. LangChain’s runtime model and OpenAI’s local context model both support this separation. ([Docs by LangChain](https://docs.langchain.com/oss/python/langchain/context-engineering))

This distinction explains why your narrow-tool proposal is appropriate. Fields such as `investigation_id`, exact obligation text, candidate ownership, derived resolved aspects, and probe lineage belong to execution context. Query wording, evidence interpretation, clinical conclusion, and residual uncertainty remain model judgments.

------

# 3. Does the proposal overhaul Yuxi excessively?

## 3.1 Overall judgment

The modification is substantial at the **medication-review agent layer** and remains appropriately bounded at the **Yuxi platform layer**.

The independent agent is justified because the proposal changes several externally observable contracts:

- Model-visible message history
- Available tool set
- Tool schemas
- Tool-result semantics
- State transition authority
- Retry behavior
- Generation budgets
- Termination rules
- Trace schema
- Final citation verification

Trying to introduce all of these changes inside the existing `MedicationReviewAcmPrimAgent` would make historical reproduction and causal evaluation harder. Freezing the old implementation and introducing `MedicationReviewAcmBoundedAgent` provides a defensible experimental boundary. Your plan already specifies this separation and preserves the current retrieval and evidence infrastructure.

## 3.2 Components to retain or revise

| Proposal element                | Recommendation                         | Reason                                                       |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| Independent bounded agent       | **Retain**                             | Clear behavioral and experimental boundary                   |
| Frozen ACM-PRIM implementation  | **Retain**                             | Reproducibility and ablation                                 |
| Audit Transcript                | **Retain**                             | Full observability and recovery                              |
| Existing Evidence Store         | **Retain and reuse**                   | Avoid evidence duplication and divergent IDs                 |
| Working Ledger as authority     | **Retain**                             | Standard typed-state pattern                                 |
| Bounded Model View              | **Retain**                             | Core remedy for context growth                               |
| ActionDirective                 | **Retain**                             | Explicit legal-action contract                               |
| Narrow tools                    | **Retain**                             | Removes state echo and provenance reconstruction             |
| Structured ToolOutcome          | **Retain**                             | Corrects transport and semantic status mismatch              |
| Forced tool choice              | **Retain with compatibility fallback** | Appropriate when the controller already knows the action class |
| Generation circuit breaker      | **Retain**                             | Tool-call limits cannot interrupt repetition within one generated response |
| Citation rehydration            | **Retain**                             | Exact evidence can be loaded only for cited claims           |
| 256k provider declaration       | **Retain as capacity metadata**        | Useful for admission control and deployment verification     |
| 120k to 160k routine target     | **Revise downward**                    | Excessive for most action rounds                             |
| Recent 3 to 5 interactions      | **Use only as a fallback tail**        | Recency alone ignores investigation relevance                |
| Generic LLM history summary     | **Restrict to conversational residue** | Clinical state and evidence provenance require typed representation |
| Dynamic tool schemas everywhere | **Simplify**                           | Static narrow schemas plus small aliases reduce provider complexity |
| Stream-time cancellation        | **Defer**                              | Post-response isolation and output limits deliver most early benefit |
| Subagents                       | **Continue to defer**                  | Shared patient state and provenance make orchestration more complex |

The plan’s own risk discussion reaches compatible conclusions, particularly around context-window complacency, controller overconstraint, evidence rehydration, tool-choice compatibility, and premature subagent introduction.

------

# 4. How historical turns should actually be selected

## 4.1 Select context for the model action, not for the tool executor

A typical agent loop makes the model call first, then executes the selected tool. Therefore, the relevant selection point is the **model call that produces the tool invocation**. The tool executor then receives:

```text
model-generated semantic arguments
+
controller-prebound runtime state
+
backend dependencies
```

The tool executor does not need the full conversation transcript.

This suggests a `ModelViewAssembler` keyed by `ActionDirective`.

## 4.2 Represent history as typed context atoms

Each durable event should be indexed as a typed atom:

```python
class ContextAtom:
    atom_id: str
    kind: str
    phase: str | None
    investigation_id: str | None
    obligation_id: str | None
    probe_id: str | None
    recovery_id: str | None
    evidence_ids: tuple[str, ...]
    tool_call_id: str | None
    state_version: int
    material_state_change: bool
    outcome: str | None
    token_count: int
    rehydration_handle: str | None
```

Recommended atom kinds include:

```text
USER_CASE
ACTION_DIRECTIVE
LEDGER_SNAPSHOT
TOOL_INTERACTION
STATE_TRANSITION
EVIDENCE_ARTIFACT
EVIDENCE_RECEIPT
REJECTION
GENERATION_ABORT
AUDIT_GAP
DRAFT
CITATION_CHECK
```

This supports deterministic relevance selection and makes every inclusion auditable.

## 4.3 Use a lexicographic retention policy

A clinical workflow benefits from ordered retention classes rather than one opaque weighted similarity score.

| Priority | Retention class                                              | Treatment                                     |
| -------- | ------------------------------------------------------------ | --------------------------------------------- |
| P0       | System policy, patient case, current directive, active ledger excerpt | Always include                                |
| P0       | Unfinished tool-call and result frontier                     | Always include as a complete pair             |
| P1       | Events for the active obligation, probe, or recovery         | Include before any recency-based history      |
| P1       | Exact candidate and selected evidence required by the current judgment | Include in raw form                           |
| P2       | Material state changes from the active investigation         | Include as compact receipts or ledger entries |
| P2       | Provenance ancestors of currently visible evidence           | Include compact lineage                       |
| P3       | Latest relevant rejection and its repair instruction         | Include until repaired or exhausted           |
| P3       | Latest phase transition or audit gap leading to the directive | Include compactly                             |
| P4       | Recent completed interactions                                | Include only while budget remains             |
| P5       | Completed unrelated investigations and pure process narration | Omit from the model view                      |
| P5       | Repetitive or aborted model output                           | Retain only in Audit Transcript               |

Recency should break ties inside a relevance class. It should not outrank exact obligation or provenance relevance.

This is the main change I would make to your current historical projection rule. Your plan already protects the original case, unfinished pairs, selected recent interactions, deterministic receipts, and abnormal-output isolation. The “recent 3 to 5 interactions” component should become the P4 safety tail.

## 4.4 Preserve tool-call/result pairs

Treat an assistant tool call and its resulting ToolMessage as one atomic projection unit:

```text
ToolInteractionAtom
  assistant_tool_call
  tool_result
  resulting_state_delta
  semantic_outcome
```

Projection may transform the representation while preserving pairing:

1. Current unresolved pair receives full representation.
2. Recent and directly relevant pair receives essential arguments and full result.
3. Older state-changing pair receives essential arguments and a deterministic receipt.
4. Unrelated completed pair may be omitted when its state effect is already present in the ledger.
5. Orphaned calls and orphaned results are rejected during model-view construction.

Pydantic AI explicitly warns that message slicing and summarization must preserve tool-call/result pairing. ([Pydantic Docs](https://ai.pydantic.dev/message-history/)) Anthropic’s context editing similarly retains recent tool-use and result pairs while clearing older result payloads. ([Claude Platform](https://platform.claude.com/docs/en/build-with-claude/context-editing))

## 4.5 Use deterministic receipts

A completed interaction receipt could use the following form:

```json
{
  "receipt_version": "TOOL_RECEIPT_V1",
  "call_id": "call_...",
  "action": "search_active_obligation",
  "outcome": "SUCCESS",
  "reason_code": null,
  "state_changed": true,
  "state_version_after": 27,
  "artifacts": ["search:Q014"],
  "evidence_refs": ["E3", "E7"],
  "support_updates": ["INV004:OBL002"],
  "next_legal_actions": ["REVIEW_ACTIVE_OBLIGATION"]
}
```

The receipt should avoid duplicated clinical prose. The authoritative conclusion, support relation, and remaining obligations already live in the Working Ledger.

## 4.6 Phase-specific context matrix

| Action phase            | Historical context to retain                                 | Evidence representation                                      |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Propose agenda**      | Patient case, extraction warnings, current coverage dimensions, recent trigger event | Usually none, with parent trigger evidence when extending agenda |
| **Search obligation**   | Active obligation, same-obligation query receipts, routes already attempted, latest query validation outcome | Evidence IDs and small digests only when query reformulation requires them |
| **Review evidence**     | Search/open calls producing current candidates, candidate ownership, probe lineage, prior support decisions for the same obligation | Exact candidate raw text and already bound support evidence  |
| **Close investigation** | All obligation outcomes in the active investigation, relevant rejection history, residual gaps | Exact raw evidence for support, contradiction, conditions, and insufficiency |
| **Global audit**        | Investigation conclusions, support matrix, uncovered dimensions, recovery status | IDs and deterministic digests, with raw evidence only for disputed gaps |
| **Draft final**         | Closed-investigation conclusions, claim-to-evidence mappings, residual uncertainty | Evidence IDs and compact evidence digests                    |
| **Verify citations**    | Draft claims and cited IDs                                   | Exact cited raw text, source, location, and hash only        |

This closely follows your proposed evidence rehydration table.

## 4.7 Deterministic assembly algorithm

```python
def build_model_view(state: BoundedState) -> ModelView:
    directive = controller.next_directive(state)

    atoms = []
    atoms.extend(mandatory_atoms(state, directive))
    atoms.extend(unresolved_tool_frontier(state))
    atoms.extend(events_for_active_obligation(state, directive))
    atoms.extend(events_for_active_investigation(state, directive))
    atoms.extend(provenance_ancestors(state, directive))
    atoms.extend(latest_relevant_rejection(state, directive))
    atoms.extend(latest_phase_transition(state, directive))
    atoms.extend(recent_fallback_tail(state, limit=3))

    atoms = deduplicate_atoms(atoms)
    atoms = enforce_tool_pair_integrity(atoms)
    atoms = choose_representation(
        atoms,
        raw_for_current=True,
        receipt_for_completed=True,
        omit_process_narration=True,
    )

    evidence = evidence_policy.select(state, directive)
    view = budgeter.admit(
        system=build_system_prompt(directive),
        patient_case=state.patient_case,
        directive=directive,
        ledger=ledger_projection(state, directive),
        history=atoms,
        evidence=evidence,
        tools=tool_policy.tools_for(directive),
    )

    trace.context_manifest.append(view.manifest)
    return view
```

------

# 5. Add a context manifest

Every model call should record why each component was included:

```python
class ContextManifestEntry:
    source_id: str
    component: str
    representation: Literal["raw", "receipt", "digest", "reference"]
    retention_reason: str
    scope_keys: dict[str, str]
    token_count: int
    priority: int
    rehydration_handle: str | None
```

Example manifest:

```json
{
  "model_call_id": "mc_027",
  "directive_id": "dir_027",
  "entries": [
    {
      "source_id": "patient_case",
      "component": "USER_CASE",
      "representation": "raw",
      "retention_reason": "MANDATORY_CASE",
      "token_count": 1820,
      "priority": 0
    },
    {
      "source_id": "Q014",
      "component": "TOOL_INTERACTION",
      "representation": "receipt",
      "retention_reason": "SAME_ACTIVE_OBLIGATION",
      "token_count": 143,
      "priority": 1
    },
    {
      "source_id": "evidence:E7",
      "component": "EVIDENCE_ARTIFACT",
      "representation": "raw",
      "retention_reason": "CURRENT_REVIEW_CANDIDATE",
      "token_count": 1278,
      "priority": 1
    }
  ],
  "excluded_counts": {
    "process_narration": 18,
    "unrelated_investigation": 9,
    "aborted_generation": 1
  }
}
```

This creates a testable answer to the question “why did the model see this history during this action?” It also permits offline replay of alternate selection policies.

------

# 6. Revise the context budget strategy

The 256k declaration should remain part of the provider profile and trace. The proposed 192k absolute admission boundary is also defensible as an emergency ceiling. The routine target of 120k to 160k remains too generous for most action rounds.

Anthropic explicitly observes diminishing returns from irrelevant context, while OpenAI, LangChain, ADK, and AutoGen all provide mechanisms that compact or filter context well before an absolute provider limit. ([Claude Platform](https://platform.claude.com/docs/en/build-with-claude/context-editing))

I recommend phase-specific initial targets. These are engineering calibration defaults, with final values selected from replay and live trace distributions.

| Phase                        | Preferred input range | Phase hard limit |
| ---------------------------- | --------------------- | ---------------- |
| Agenda creation or extension | 12k to 24k            | 40k              |
| Search action                | 16k to 32k            | 48k              |
| Evidence review              | 32k to 56k            | 80k              |
| Investigation closure        | 40k to 64k            | 96k              |
| Global audit                 | 32k to 64k            | 96k              |
| Final draft                  | 64k to 96k            | 128k             |
| Citation verification        | 48k to 72k            | 96k              |
| Absolute provider admission  | N/A                   | 192k             |

A call approaching 192k should be exceptional and produce a trace warning. Frequent calls above 100k would indicate that the evidence selector or ledger projection remains too broad.

## 6.1 Reduction order

When a phase budget is exceeded, apply reductions in a fixed order:

1. Remove pure model narration.
2. Replace completed raw ToolMessages with receipts.
3. Remove completed interactions from unrelated investigations.
4. Replace secondary evidence raw text with deterministic digests.
5. Reduce same-investigation fallback history.
6. Restrict candidates to those owned by the active obligation and legal provenance routes.
7. Rehydrate only the evidence required for the current semantic decision.
8. Reject model admission if the phase hard limit remains exceeded.

The following components should survive every reduction:

- Current ActionDirective
- Lossless patient facts
- Active obligation and investigation
- Unresolved tool frontier
- Current candidate ownership
- Evidence directly required for the active judgment
- Applicable safety and citation rules

------

# 7. Tool invocation design

## 7.1 Keep model-facing arguments semantically narrow

The ActionDirective and controller should prebind:

```text
active investigation
active obligation
active recovery
allowed route set
candidate evidence set
candidate file set
probe lineage
current state version
remaining budgets
retry policy
```

The model-facing search tool then becomes:

```python
search_active_obligation(
    query_text: str,
    route_choice: Literal["GLOBAL", "DOCUMENT"] | None = None,
)
```

When only one route is legal, omit `route_choice` and bind the route in code.

The support tool becomes:

```python
record_active_obligation_support(
    evidence_aliases: list[str],
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
    rationale: str,
)
```

The controller resolves aliases such as `E1`, `E2`, and `E3` to immutable evidence IDs. This reduces token usage and transcription risk from long hashes.

The closing tool becomes:

```python
close_active_investigation(
    conclusion: str,
    status: Literal["SUPPORTED", "CONDITIONAL", "INSUFFICIENT"],
    residual_uncertainty: str | None = None,
)
```

The controller derives:

```text
resolved aspects
remaining aspects
selected evidence union
active investigation ID
kind-specific legal fields
support matrix
closure eligibility
```

Your proposal already identifies this split and correctly transfers derived fields to the controller.

## 7.2 Expose only currently legal tools

For each directive:

```text
PROPOSE_INITIAL_AGENDA
  expose propose_initial_agenda

SEARCH_ACTIVE_OBLIGATION
  expose search_active_obligation
  optionally expose read_selected_evidence

REVIEW_ACTIVE_OBLIGATION
  expose record_active_obligation_support
  expose search_active_obligation only when another route remains legal
  expose read_selected_evidence

CLOSE_ACTIVE_INVESTIGATION
  expose close_active_investigation

AUDIT_COVERAGE
  expose audit_coverage

DRAFT_FINAL
  use structured model output

VERIFY_CITATIONS
  use structured model output
```

LangChain supports state-based tool filtering, OpenAI supports dynamic capability visibility and forced tool choice, and Semantic Kernel supports contextual function advertisement. ([Docs by LangChain](https://docs.langchain.com/oss/python/langchain/context-engineering))

For your small action set, deterministic phase routing is more appropriate than an LLM tool-selector call. An extra tool-selection model call would add latency and another failure point without resolving genuine ambiguity.

## 7.3 Keep static schemas where possible

Dynamic schemas introduce provider and serialization risks. A safer hierarchy is:

1. Static narrow tool definition
2. Hidden controller binding
3. Short per-call candidate aliases
4. Small enum only when it materially prevents invalid selection
5. Runtime validation as final authority

For example, expose candidate aliases `E1` through `E8` in the current directive and validate them in the tool. Avoid regenerating an entire Pydantic class for every state unless schema-level enum enforcement has been tested against the exact OpenAI-compatible endpoint.

------

# 8. Structured ToolOutcome

Use two distinct status dimensions:

```python
transport_status: Literal["COMPLETED", "FAILED"]
semantic_outcome: Literal[
    "SUCCESS",
    "NO_RESULT",
    "REJECTED",
    "INVALID_ARGUMENT",
    "NEEDS_INPUT",
    "RETRYABLE_ERROR",
    "FATAL_ERROR",
]
```

Recommended envelope:

```json
{
  "call_id": "call_027",
  "directive_id": "dir_027",
  "tool_name": "record_active_obligation_support",
  "transport_status": "COMPLETED",
  "semantic_outcome": "REJECTED",
  "reason_code": "EVIDENCE_OUTSIDE_CANDIDATE_SET",
  "retryable_by_model": true,
  "technical_retryable": false,
  "state_changed": false,
  "executed_backend": false,
  "state_version_before": 26,
  "state_version_after": 26,
  "recovery_action": "REPAIR_EVIDENCE_SELECTION_ONCE",
  "message_for_model": "Choose evidence aliases from E1, E2, or E4.",
  "next_legal_actions": ["RECORD_ACTIVE_OBLIGATION_SUPPORT"]
}
```

The outer framework status should also reflect failure where supported. LangChain’s middleware can produce an error-status ToolMessage for tool failures. ([Docs by LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in))

The controller behavior should be finite:

| Outcome            | Controller response                                          |
| ------------------ | ------------------------------------------------------------ |
| `SUCCESS`          | Apply state delta and advance directive                      |
| `NO_RESULT`        | Record a genuine probe and choose another legal route or insufficiency |
| `REJECTED`         | Remove the identical illegal action from the reachable set   |
| `INVALID_ARGUMENT` | Permit one schema or query repair                            |
| `NEEDS_INPUT`      | Add the required missing input to the next directive         |
| `RETRYABLE_ERROR`  | Execute bounded technical retry in code                      |
| `FATAL_ERROR`      | Disable the capability and enter explicit fallback           |

This is consistent with your proposed closed outcome set and recovery rules.

------

# 9. Prevent repeated illegal actions

Add an action fingerprint:

```python
fingerprint = hash(
    directive_id,
    state_version,
    tool_name,
    canonicalize(model_arguments),
)
```

Store:

```text
attempt count
semantic outcome
reason code
state version
whether the state changed
```

Controller rules:

- A rejected fingerprint cannot be reached again under the same state version.
- A repaired action must have different normalized arguments.
- A state transition permits a new fingerprint.
- Two consecutive no-state-change failures exhaust the model repair path.
- Technical retries retain the same fingerprint and are counted separately from semantic retries.

This closes an important gap left by ordinary call-count limits. Call limits cap total calls, while fingerprints prevent same-state semantic loops.

------

# 10. Generation protection remains necessary

Your pathological trace arose inside one model generation before a legal tool call was emitted. Tool-call limits cannot interrupt that behavior.

The proposed protections are therefore appropriate:

1. Action rounds receive a small output budget, around 1,024 to 1,536 tokens, with a 2,048-token absolute cap.
2. Final synthesis uses a separate budget.
3. The current legal tool is forced when provider support is reliable.
4. A structured action-output fallback is used when forced tool calling is unreliable.
5. Repetition detectors inspect n-grams, periodic suffixes, unique-token ratios, and tool-call absence.
6. Aborted raw output enters the Audit Transcript only.
7. One forced, smaller-budget retry is permitted.
8. A second failure enters controller fallback.

OpenAI’s tool-choice controls and LangChain’s model/tool limits support this general strategy. ([OpenAI GitHub](https://openai.github.io/openai-agents-python/agents/)) Your plan appropriately separates response-level isolation from the later possibility of stream-time cancellation.

------

# 11. Recommended Yuxi integration boundary

Yuxi’s current architecture uses a generic `create_agent` loop and middleware-generated capabilities. The earlier analysis also notes that checkpoints and the business database maintain separate forms of conversation state.

A clean integration should use:

```text
MedicationReviewAcmBoundedAgent
  create_agent
    AcmBoundedControllerMiddleware
    AcmBoundedModelViewMiddleware
    AcmGenerationGuardMiddleware
    narrow bounded tools
    existing checkpointer
```

### Keep unchanged

```text
BaseAgent invocation infrastructure
chat_service and HTTP API
thread and checkpoint identity
conversation persistence
frontend message history
Milvus retrieval implementation
Corpus Atlas
Top-25 background and Top-10 visible behavior
Evidence ID and hash generation
raw Evidence Store
legacy ACM-PRIM agent
legacy traces and batch variants
```

### Implement inside the new agent module

```text
controller.py
  ActionDirective
  legal phase routing
  action fingerprinting
  retry reachability

context_view.py
  ContextAtom indexing
  relevance-first selection
  pair-safe projection
  evidence rehydration policy
  token admission
  context manifest

tools.py
  narrow model-facing schemas
  hidden runtime binding
  structured ToolOutcome

harness.py
  pre-model message projection
  phase-specific tools
  forced-tool settings
  generation isolation

models.py
  bounded state
  context manifest
  tool outcomes
  generation abort records
  controller versioning
```

The model-view middleware should modify only the model request copy. The durable checkpoint and trace retain full events. LangChain explicitly supports transient message and tool overrides without altering persisted state. ([Docs by LangChain](https://docs.langchain.com/oss/python/langchain/context-engineering))

This boundary gives you the modern architecture you need while preserving Yuxi’s core services.

------

# 12. Validation strategy

## 12.1 Use staged ablation

| Variant | Change                                             |
| ------- | -------------------------------------------------- |
| A       | Frozen existing ACM-PRIM                           |
| B       | A plus bounded model-view projection               |
| C       | B plus ActionDirective and narrow tools            |
| D       | C plus structured ToolOutcome and finite repair    |
| E       | D plus generation guard                            |
| F       | E plus final citation rehydration and verification |

Keep the following controlled across variants:

```text
model and provider configuration
knowledge base
Milvus settings
Atlas
retrieval Top-K
case set
evidence IDs
gold annotations
answer formatting
```

This will show whether performance changes arise from context projection, tool interfaces, controller routing, or final verification.

## 12.2 Context-management metrics

Record per model call:

```text
input tokens by component
projected total tokens
raw transcript token count
projection reduction ratio
number of raw interactions
number of receipts
number of evidence digests
number of exact evidence rehydrations
phase target and hard limit
admission rejection count
excluded atom counts by reason
```

Report p50, p90, p95, p99, and maximum by phase.

## 12.3 Tool metrics

```text
semantic acceptance rate
semantic rejection rate by phase and reason
transport/semantic status mismatch count
state-echo error count
scheduler-target error count
derived-state submission error count
candidate-ownership error count
provenance error count
query-shape rejection count
same-fingerprint repetition count
repair-exhaustion count
technical retry count
state-changing call rate
```

Errors removed by interface construction should be reported separately from remaining semantic errors. Exact obligation copying, active-investigation selection, and derived-state echo should approach zero. Query quality and clinical evidence interpretation will remain probabilistic.

## 12.4 Long-horizon liveness metrics

```text
completion rate
same-state loop count
phase transition count
actions per completed investigation
unresolved obligations at termination
pending recovery at termination
generation abort rate
forced-tool compliance rate
action-output p95 and p99
```

## 12.5 Existing retrieval and answer metrics

Continue using the project’s formal three-level metric specification, which applies to PRIM-RAG, DA-PRIM, baseline RAG, and subsequent compatible experiments.

Particularly relevant process metrics include:

- Autonomous Recall AUC, which rewards early evidence discovery and penalizes unproductive tails.
- Productive Call Rate and New Targets per Call, which measure whether successive searches actually close new evidence targets.
- Document and chunk recall
- Document-conditioned chunk recall
- `judgment_score`
- `coverage_score`
- paired bootstrap confidence intervals

The specification currently treats Faithfulness, Semantic Context Recall, and Evidence Consistency as future metrics. Citation verification can supply the infrastructure for these measures, though they should enter formal reporting only after their definitions and evaluator implementations have been frozen.

------

# 13. Recommended acceptance criteria

## Architecture

- Old and new agents coexist with distinct IDs and versions.
- Full events remain auditable.
- Every model call has a ContextManifest.
- Evidence raw text, hashes, locations, and occurrences remain immutable.
- No model-facing context is created through provider-side silent truncation.

## Context

- Agenda and search calls ordinarily remain below 32k.
- Evidence-review calls ordinarily remain below 56k.
- Final drafting ordinarily remains below 96k.
- No call exceeds its phase hard limit.
- No call exceeds the 192k absolute ceiling.
- Aborted outputs contribute zero tokens to subsequent views.

## Tool correctness

- Exact obligation echo errors equal zero.
- Scheduler-target errors equal zero.
- Derived resolved/remaining echo errors equal zero.
- Kind-specific parameter misuse equals zero.
- Candidate and provenance violations are blocked before state mutation.
- Identical rejected fingerprints cannot recur under the same state version.
- Transport and semantic outcomes agree in every exported trace.
- Overall semantic rejection falls below 5 percent, with phase-specific reporting.

## Generation

- Action outputs remain below 2,048 tokens.
- Repetitive pre-tool generation no longer reaches thousands of tokens.
- One guarded retry is the maximum after repetition abort.
- Same-state free sampling cannot continue indefinitely.

## Method quality

- Unprobed obligations and pending recovery continue blocking finalization.
- Search count remains determined by unresolved evidence obligations.
- Core document recall shows no systematic paired decline.
- Chunk target recall shows no systematic paired decline.
- Productive Call Rate and Autonomous Recall AUC remain equal or improve.
- `judgment_score` and `coverage_score` remain equal or improve.
- Any observed difference is accompanied by paired confidence intervals.

------

# Final recommendation

Proceed with the independent `MedicationReviewAcmBoundedAgent`.

The proposal’s strongest elements already match modern framework practice:

```text
durable transcript
typed working state
external evidence artifacts
transient model projection
runtime-injected tool context
phase-specific tool exposure
finite retries
structured outcomes
on-demand rehydration
```

The most important implementation change concerns **context selection**. Historical interactions should be selected through active obligation, active investigation, provenance ancestry, material state transition, and relevant repair history. A recent-message tail can provide secondary continuity.

The second important change concerns **context size**. The 256k declaration should remain a provider capability and emergency margin. Most action calls should operate far below that ceiling.

The third important change concerns **tool interfaces**. The model should generate clinical semantics and retrieval expressions. The controller should supply IDs, ownership, provenance, scheduling state, route legality, budgets, and derived fields.

Implemented within the new agent and middleware layer, this architecture preserves Yuxi’s platform foundations and gives ACM-PRIM a much stronger long-horizon reliability model.