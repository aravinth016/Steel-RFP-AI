# PROJECT SPECIFICATION

## Agentic RFP Proposal Evaluation

**Project format:** Agentic AI + RFP Evaluation classroom mini project  
**Target stack:** Python, Streamlit, SQLite, any configured JSON-capable LLM  
**Primary goal:** Compare supplier proposals against the same RFP requirements, explain every score with evidence, and identify the highest-ranked eligible proposal under a configurable weighted evaluation rubric.

**Current domain adaptation:** TMT reinforcement-steel procurement. The uploaded buyer RFP remains authoritative; the application must not invent grade, standard, test limit, quantity, dates, delivery basis, or commercial terms.

---
 
## 1. Product vision

Build an AI-assisted procurement evaluation application that reads an RFP and supporting requirements documents alongside supplier proposals. It extracts and compares information, evaluates supplier proposals against a configurable rubric, calculates weighted scores deterministically, and presents a transparent leaderboard with reasons and evidence.

The application is a decision-support tool. It must not claim that a supplier is legally or contractually awarded a tender. The final award decision remains with authorized human reviewers and must follow the procurement policy for that project.

### Core outcome

The main result should make it easy to answer:

> Which eligible supplier proposal best meets the RFP requirements and ranks highest by the deterministic PPI policy, and why?

Show the highest-ranked eligible supplier prominently, while retaining absolute weighted scores, benchmarks, PPI, mandatory-gate outcomes, missing documentation, risks, reviewer decisions, and evidence for auditability.

## 2. Scope

The evaluation workflow supports one mode: **RFP Proposal Evaluation**. Compare supplier proposals for the same RFP against mandatory eligibility gates and a weighted technical/commercial rubric. Do not compare suppliers across unrelated RFPs or differing requirements. The same RFP documents and rubric version should be used for every supplier in a run.

## 3. User interface requirements

Create an uncluttered multi-section Streamlit application with the following pages or tabs.

### 3.1 Home / New RFP proposal evaluation
- Create an RFP proposal evaluation run with an automatically generated `RUN_ID` and timestamp.
- Explain the difference between mandatory pass/fail gates and weighted scoring.
- Offer a **Demo Run** for new users that works without an API key and uses fictional sample data plus a deterministic mock evaluator.
- Clearly label demo results as synthetic and not real procurement advice.

### 3.2 Document upload
- Accept multiple supplier proposal PDFs and shared RFP/supporting documents.
- Shared documents can include RFP requirements, project specifications, contract excerpts, approved material/manufacturer lists, standards excerpts supplied by the user, drawings converted to readable text where supported, and required testing/certification requirements.
- Per supplier, capture supplier name, proposal submission date, historical experience rating (if used by the rubric), proposed amount/currency (if applicable), and optional notes.
- Show uploaded filename, file size, supplier mapping, upload status, and extraction status.
- Provide clear controls to remove a supplier package or add another supplier before the run is locked.
- Prevent duplicate supplier names in the same run, or explicitly ask the user to confirm if duplicates are allowed.
- Validate file type and reasonable file-size/page limits; handle encrypted, scanned, unreadable, empty, or corrupt PDFs with clear messages.

### 3.3 Criteria and weight configuration
- Display criterion name, description, weight, maximum score, score rubric, whether it is mandatory/pass-fail, and active status.
- Allow authorized users to add, edit, activate, or deactivate criteria before starting an evaluation.
- Validate that active weighted criteria total exactly 100% before the run starts.
- Keep mandatory pass/fail gates separate from the 100% weighted criteria.
- Snapshot the criteria, weights, and scoring policy at run creation/start. Subsequent edits must not silently alter a past run.
- Display a warning that the starter weights are examples and must be reviewed against the project's tender/procurement policy.

### 3.4 LLM settings
- Provide settings for the selected LLM provider with a password-masked API key field and a **Test connection** button.
- Never display, log, commit, or persist the raw API key in SQLite or result JSON.
- For a locally entered key, retain it only for the current Streamlit session where feasible. For deployment, support `st.secrets` configuration. Explain that a key entered in a public multi-user application needs careful handling; do not build a shared public key vault into the demo.
- Provide clear connection states: Not configured, Testing, Connected, Invalid key, Rate limited, Network/API error.
- Allow the user to select from supported model IDs for the selected provider rather than hard-coding an obsolete model name.
- If no API key is available, allow Demo Mode and any local deterministic tests to run; do not pretend the real LLM was called.
- Show estimated/request count and friendly errors where possible, but do not expose secrets or raw sensitive document text in error logs.

### 3.5 Evaluation progress
- Show stages: file validation, PDF extraction, retrieval/indexing (if RAG enabled), LLM evaluation, schema validation, human review, deterministic scoring, ranking, persistence.
- Preserve completed supplier outputs when another supplier fails; persist failed supplier status and warnings so the batch remains auditable.
- Record normalization warnings and provider/evaluation failures in the saved run.

### 3.6 Leaderboard and recommended winner
- Display final rank, supplier name, eligibility/gate status, AI-proposed score, human-approved final score, PPI/peer benchmark metric, submission date, experience rating, proposed price when provided, and review status.
- Make the top-ranked **eligible** proposal clear and explain why it ranks first.
- Show ineligible or mandatory-gate-failing proposals in the leaderboard for transparency, but mark them ineligible and do not recommend them as the winner unless the human reviewer explicitly records an approved exception under the relevant policy.
- Let users sort/filter by supplier, eligibility, score, review status, or missing documentation.
- Include JSON export for the complete run and CSV export for the leaderboard where practical.

### 3.7 Detailed comparison
- Provide side-by-side supplier comparison by criterion.
- For every score/status, show the AI's justification, exact supporting evidence snippet, source document filename, page number where available, section/clause reference where available, and retrieval/citation details.
- Clearly differentiate **document evidence**, **LLM explanation**, **reviewer comment**, and **system-calculated metric**.
- Include documented risks, deviations, missing information, and clarification requests.
- Include a section for reviewer comments and an audit trail of changes.

### 3.8 Spider / radar plots
- Display a radar (spider) chart comparing selected suppliers across the active evaluation criteria.
- Normalize criteria to a common 0–100 scale for charting; document the normalization rule.
- Allow users to select suppliers and criteria to avoid overcrowded charts.
- Provide an accessible table alongside the chart because radar charts can be difficult to read precisely.
- Do not plot missing or invalid data as if it were a genuine zero without clearly indicating the missing value.
- Keep any chart optional; the criterion comparison table remains the precise, accessible scorecard.

### 3.9 Human review / adjudication
- Support a human-in-the-loop review before finalization.
- For every supplier and criterion, show the AI-proposed score, status, justification, evidence, and warnings.
- Allow a reviewer to accept the AI score or override the status and/or score, add a mandatory reason for an override, and optionally request re-evaluation.
- Preserve the original AI result even when a reviewer changes it.
- Calculate the official final weighted score and final ranking using the accepted/reviewer-approved scores, not silently from the original AI scores.
- Show both AI-proposed and human-approved totals in the UI and in exported JSON.
- Use review states such as `Pending Review`, `Reviewed`, and `Finalized`.
- Only mark a run finalized when all required suppliers/criteria have been reviewed, or show clearly which reviews are incomplete. Make any exception explicit.
- Save reviewer identity only if the application has an appropriate authentication mechanism. Otherwise, record a reviewer-entered display name/role as an unauthenticated label and do not present it as verified identity.

### 3.10 Q&A Bot
- Provide a dedicated **Q&A Bot** page in the sidebar for asking questions about a saved evaluation run.
- Let the user search shared project requirements only or shared requirements plus one selected supplier's submissions. Never mix supplier-specific evidence across suppliers.
- Chunk extracted pages while preserving source ID, filename, scope, supplier ID, and page number.
- Retrieve with BM25 and a configured embedding bi-encoder where supported, then fuse rankings using reciprocal rank fusion (RRF).
- Ground answers only in the retrieved excerpts, validate cited source labels, and show the original filename and page for each citation.
- If no relevant excerpt is retrieved, report that the information does not appear in the uploaded documents.
- Obtain explicit Q&A consent before sending document chunks and questions to the selected embedding provider and before sending the question and retrieved excerpts to the selected answer provider.
- Store extracted page text locally for saved-run Q&A, separately from run JSON exports. Do not retain original PDFs or API keys.

## 4. Document-grounded evaluation rules

The evaluation assistant must follow evidence-grounding principles:

- Use only uploaded submittals, specifications, contract excerpts, approved lists, and user-supplied code/standard excerpts as evidence for project-specific findings.
- Do not invent project requirements, facts, certificate details, or conflicts.
- Use an objective, technical, neutral tone.
- If only a submittal is present and no specification is uploaded, summarize the submittal and state that full compliance checking is not possible.
- If only a specification is present, summarize requirements and state that no proposal/submittal content is available for comparison.
- If information is absent, say: **“This information does not appear in the uploaded documents.”**
- If requirements conflict or are ambiguous, identify the exact references and request clarification instead of choosing a governing requirement without evidence.
- Distinguish absence of evidence from evidence of non-compliance.
- Do not infer that a supplier is approved merely because its name appears in a proposal; compare against the uploaded approved list when one exists.
- Do not assume an external code/standard applies unless it is referenced by uploaded project documentation or the user explicitly selects it as an evaluation source.

## 5. Suggested starter evaluation rubric

There is no single universal set of weights suitable for all RFPs. The application must support configurable, policy-approved criteria and weights. Provide the following as a clearly labelled **starter template only**, not as an official or industry-mandated standard.

### 5.1 Mandatory pass/fail gates (outside the weighted score)
Examples to configure per procurement package:
- Required proposal and mandatory attachments submitted.
- Supplier meets stated eligibility requirements.
- Mandatory scope and technical requirements are met, or there is an explicitly allowed exception process.
- Required statutory, safety, quality, or code documentation is provided where required by the uploaded procurement documents.
- No unresolved disqualifying condition identified in the tender rules.

A failed mandatory gate makes a supplier `Ineligible` by default. The reason and evidence must be visible. Do not make a legal eligibility decision from an LLM guess; the gate must be defined in the configured rules and reviewed by a human when ambiguous.

### 5.2 Weighted criteria (example sum = 100%)

| Criterion | Example weight | Evidence to inspect |
|---|---:|---|
| Technical and grade compliance | 30% | Offered grade, stated specification, deviations, product form, and evidence |
| Quality and test documentation | 15% | MTCs, heat/batch traceability, test results, and quality records |
| Commercial value | 20% | Total quoted amount and itemized rates, freight, taxes, charges, and exclusions |
| Quantity and delivery capability | 15% | Offered quantity, availability, delivery phases, dates, and destination |
| Supplier experience and references | 10% | Relevant steel-supply history, references, and supporting evidence |
| Warranty and replacement support | 5% | Stated warranty, rejection, and replacement terms |
| Documentation and quotation completeness | 5% | Quote validity, forms, declarations, and required attachments |
| **Total** | **100%** | |

Do not score a criterion that is not applicable without an explicit configured rule.

### 5.3 Scoring scales

- Use a consistent criterion score scale, such as 0–5 or 0–10, configured per criterion.
- Each criterion must include a written scoring rubric defining what each score means. For example, do not leave a generic score of 8/10 without a policy-specific definition.
- Normalize each score to a 0–100 percentage before applying its weight:

`normalized_score_pct = (criterion_score / criterion_max_score) * 100`

`weighted_contribution = normalized_score_pct * (criterion_weight_pct / 100)`

`final_weighted_score_pct = sum(weighted_contribution for active weighted criteria)`

- Mandatory pass/fail gate results are separate from weighted scoring. A strong weighted score must not hide a failed mandatory gate.
- Missing/not-stated information must have its own status. Any numeric treatment for missing evidence (zero, penalty, or pending review) must be an explicit configurable policy, displayed to users, and covered by tests. Never silently treat a missing value as a verified failure.
- For a price criterion where lower comparable cost is better, a possible formula is:

`price_score_pct = (lowest_evaluated_price_among_eligible_bids / supplier_evaluated_price) * 100`

Use this only if the procurement rules allow it and the evaluated prices are on a comparable basis. Handle zero, missing, non-positive, different-currency, tax, scope, and non-comparable prices safely. Do not compare headline prices when the scopes or inclusions differ.

For steel quotation evaluation, a total-price comparison additionally requires a human confirmation that the RFP permits the method, a positive quoted total for every supplier, the same currency, matching offered quantity and delivery basis, and comparable included cost scope. Missing freight, taxes, or other charges must not be inferred. If those conditions are not established, do not calculate the commercial price score.

### 5.4 Winner and tie-break policy

The assignment ranking policy is deterministic:
1. Apply the mandatory eligibility review; eligible suppliers rank ahead of ineligible or pending suppliers.
2. Within each eligibility group, sort by higher PPI first.
3. Break ties by earlier submission date, then higher historical experience rating, then supplier name ascending.
4. Assign sequential ranks only after the stable sort.

The absolute weighted score is reported independently and is not an extra tie-break. The active policy is stored with and displayed for every run. The LLM never calculates totals, benchmarks, PPI, or rank.

## 6. Retrieval-Augmented Generation (RAG)

RAG is supported and recommended for evaluations involving long specifications or multiple supporting documents. It improves access to relevant evidence but does not guarantee correctness; citations and human review remain necessary.

### Proposed RAG workflow
1. Extract text from PDFs and retain provenance metadata: run ID, supplier ID, file name, document category, and page number.
2. Split text into manageable overlapping chunks while preserving the provenance metadata. Start with approximately 400 words per chunk and tune with testing; keep tables and specification clauses intact where possible.
3. Use local BM25 and, for Q&A providers with an embedding endpoint, an embedding bi-encoder; combine rankings with reciprocal rank fusion. Evaluation may use local keyword retrieval.
4. For each criterion, retrieve relevant chunks from both the common project requirements and the relevant supplier proposal. Keep supplier evidence and requirement evidence distinctly labelled.
5. Optionally rerank retrieved chunks when a configured provider supports reranking.
6. Send the LLM only the relevant retrieved excerpts and the evaluation instructions. Request structured JSON with source references and evidence snippets.
7. Validate the output and verify that referenced source IDs/pages correspond to actual retrieved chunks.
8. Display evidence in the comparison view and allow reviewers to open the related file/page where available.

### RAG controls and safeguards
- Provide a **Use RAG** toggle and show which corpus/documents are in scope.
- Default to the documents uploaded for the current run; do not search the public internet by default.
- Do not mix one supplier's proposal text into another supplier's evidence. Shared specifications can be used for all suppliers; supplier-specific evidence must be scoped to the correct supplier.
- Mark each statement as supported, partly supported, unsupported, or ambiguous where feasible.
- If retrieval finds no relevant evidence, return a missing-information warning rather than inventing an answer.
- Preserve page/clause metadata through chunking; a citation must be traceable to the original file and location, not merely a vector index position.
- For interactive Q&A, include source filename and page citations, reject labels not in the retrieved set, and require explicit consent for both embedding and answer-generation providers.
- Add tests for wrong-supplier retrieval, relevant evidence retrieval, no-result retrieval, and incorrect source references.

## 7. LLM provider integration

- Use the supported SDK or API for the selected provider and current API version documented at implementation time.
- Keep model identifiers configurable in settings; do not assume a model identifier will remain current forever.
- Implement an `LLMProvider` interface so demo mode and a future provider can be substituted without rewriting scoring/ranking code.
- The live provider should use the selected configured provider for document-based evaluation. RAG mode may use that provider's embedding API where available.
- Do not let the model calculate the final weighted score, peer benchmark, PPI, tie-break ordering, or final rank. Python owns those calculations.
- Add timeout, retry/backoff for recoverable failures, rate-limit messages, malformed-response handling, and token/document-size safeguards.
- Never send user documents to a provider without clear user acknowledgement and a description of what will be transmitted.

## 8. Human-in-the-loop scoring process

Use the following states:

`Uploaded → AI Evaluated → Awaiting Human Review → Reviewed → Finalized`

For each criterion, store:
- AI score, AI compliance status, justification, evidence and source references.
- Reviewer decision: accept AI score / override score or status / request more information.
- Final approved score and status.
- Reviewer comment and mandatory override reason when overridden.
- Timestamps for evaluation and reviewer action.

The reviewer must be able to compare AI and human scores side by side. A difference report should show criterion-level disagreements and how they affect the final weighted score and rank. The reviewer can re-run the AI evaluation after replacing or adding documents; the earlier results remain in the audit trail and are marked as superseded, not silently overwritten.

Human review should be required for ambiguous conflicts, potentially disqualifying non-compliance, missing mandatory evidence, unsupported findings, and material AI/reviewer disagreement. Define exactly what constitutes material disagreement in a configurable rule, for example a score difference of two or more points on a 0–10 scale or a change in pass/fail status.

## 9. Required output structure

For each supplier, generate:

1. **Proposal/Submittal Summary** — product/material or scope, manufacturer, model/type, intended location, technical data, price where applicable, declared standards, delivery/warranty information where present.
2. **Requirements Summary** — criterion/requirement, specification or contract reference, required value/condition, and source document.
3. **Supplier vs Requirement Comparison** — requirement, supplier value, status, AI score, evidence snippet, source file/page/section, justification, and reviewer decision.
4. **Mandatory Gate Results** — pass/fail/pending, rule, evidence, and consequence for eligibility.
5. **Scoring Breakdown** — raw criterion score, normalized score, weight, weighted contribution, scoring rationale, evidence, and warnings.
6. **Missing Information and Risks** — explicit missing/not-stated items, ambiguous clauses, documentation gaps, and unresolved concerns.
7. **Review Checklist** — requirement-driven checks for the human reviewer.
8. **Recommended Comments** — objective, evidence-based boilerplate comments and requests for clarification/resubmission.
9. **Clarification Points** — only actual conflicts or ambiguities supported by uploaded documents.
10. **Human Review Log** — AI score versus reviewer score, decision, reason for override, reviewer-entered name/role if provided, and timestamp.
11. **Final Result** — eligibility, AI-proposed score, human-approved weighted score, PPI, final rank, ranking explanation, and finalization status.

## 10. JSON output schema (illustrative)

```json
{
  "run_id": "RFP-2026-0001",
  "supplier_name": "Example Construction Materials",
  "document_version": 1,
  "eligibility": {
    "status": "Eligible",
    "gates": [
      {
        "gate_id": "G001",
        "status": "Pass",
        "requirement": "Required test certificate supplied",
        "evidence": "Certificate listed in the submitted package",
        "source_file": "supplier_submittal.pdf",
        "page": 4
      }
    ]
  },
  "criteria": [
    {
      "criterion_id": "C001",
      "criterion_name": "Technical/specification compliance",
      "weight_pct": 30,
      "max_score": 10,
      "ai_score": 8,
      "ai_status": "Partially compliant",
      "ai_justification": "The submitted value meets two documented requirements; one parameter is not stated.",
      "evidence": [
        {
          "quote": "Illustrative source excerpt",
          "source_file": "supplier_submittal.pdf",
          "page": 2,
          "section": "Product data"
        }
      ],
      "reviewer_action": "Override",
      "reviewer_score": 7,
      "reviewer_status": "Pending clarification",
      "override_reason": "Reviewer identified an additional unresolved requirement.",
      "final_score": 7,
      "final_status": "Pending clarification",
      "warnings": []
    }
  ],
  "ai_weighted_score_pct": 82.0,
  "human_approved_weighted_score_pct": 78.0,
  "ppi": 94.5,
  "final_rank": 2,
  "review_status": "Reviewed",
  "risks": [],
  "overall_summary": "Illustrative only; scores and evidence are not real."
}
```

The application must validate JSON against an explicit schema, ensure every active criterion has exactly one result, reject or normalize out-of-range scores under a documented policy, and record warnings. Example numbers and source text above are fictitious.

## 11. SQLite persistence

- `evaluation_criteria`: criterion ID, name, description, weight, maximum score, rubric, active flag, and display order.
- `runs` and `rfp_runs`: run ID, timestamp, status, criteria snapshot, ranking policy, selected model ID (never API key), warnings, and complete result payload.
- `supplier_results`: run/supplier IDs, supplier name, submission date, experience rating, absolute score, PPI, rank, and complete supplier result JSON.
- `rag_pages`: extracted page text retained locally for run-scoped Q&A.
- `review_events`: reviewer decisions, timestamps, and reasons.
- `app_settings`: non-secret provider/model preferences.

Use parameterized SQL, transactions, migrations or reproducible initialization scripts, and stable IDs. Do not store API secrets in SQLite.

## 12. Ranking, benchmarking, and explanation

- Criterion benchmark: highest valid comparable score among the suppliers in the same run.
- Criterion gap: supplier score minus the benchmark score.
- Relative-performance percentage: supplier score / benchmark score × 100, with explicit safe handling when the benchmark is zero or the criterion is missing.
- PPI: weighted average of criterion relative-performance percentages, based on configured weights. Missing criteria and zero/unavailable benchmarks are excluded and the remaining weights are renormalized; PPI is undefined if no criterion has a relative percentage.
- Ranking: eligibility group first, then PPI descending, earlier submission date, higher experience rating, and supplier name ascending.
- Every tie-break must be deterministic and displayed in the run's ranking explanation.
- Do not allow the LLM to reorder the leaderboard or choose the winner outside the deterministic ranking tool.

## 13. Demo mode and synthetic data

Provide a demo mode that does not require an API key and cannot be mistaken for a real evaluation.

- Include at least four synthetic suppliers with intentionally different strengths, prices, delivery schedules, experience, evidence quality, missing fields, and compliance issues.
- Include one synthetic RFP and at least four synthetic supplier proposals, clearly watermarked or labelled `SYNTHETIC DEMO DATA`.
- If creating PDFs automatically is supported, generate fictional PDF samples. Otherwise, include small prepared demo PDFs in `sample_data/`.
- Use a deterministic mock evaluator for demo runs and automated tests. Do not describe mock results as live provider results.
- Provide a sample pack containing one fictional RFP and four proposals with intentionally different strengths, prices, schedules, experience, evidence, and missing information.
- Make it possible for a new user to explore the leaderboard, comparison table, chart, and human review workflow without configuring secrets.

## 14. Testing requirements

Write automated tests for:
- Criteria weights adding to 100%; inactive criteria; invalid weights and max scores.
- Score normalization and weighted calculations using hand-calculated expected values.
- Mandatory gate pass, fail, and pending states.
- Price scoring with comparable prices, missing price, zero or negative price, currencies, and unequal scope.
- PPI, zero benchmark, missing criterion, score ties, and deterministic tie-break ordering.
- Required criterion missing from LLM JSON, invalid JSON, scores outside range, duplicate criterion IDs, unsupported evidence references, and malformed evidence.
- PDF with readable text, empty/scanned PDF, corrupt/encrypted PDF, and extraction warning.
- RAG retrieval preserving source file/page/section and supplier isolation.
- Human score override updating the approved score/rank while preserving the original AI result and audit event.
- Demo mode without API key, provider connection errors, and API rate limits.
- End-to-end run from upload through leaderboard and JSON export.

Do not report tests as passed unless the tests have actually been run.

## 15. Recommended implementation phases

1. **Plan and scaffold:** review this file, propose architecture, create folder structure, requirements, and tests.
2. **Rules first:** implement criteria validation, score normalization, gates, weighted scoring, PPI, deterministic ranking, and unit tests without calling an LLM.
3. **Document intake:** implement PDF extraction, supplier metadata, and extraction tests.
4. **Evaluation and validation:** implement mock evaluation, schema validation, and configured LLM providers.
5. **RAG:** implement chunking, retrieval, optional reranking, source metadata, citations, and isolation tests.
6. **Human review:** implement accept/override workflow, audit log, recalculation, and disagreement indicators.
7. **Persistence and UI:** implement SQLite and Streamlit pages, leaderboard, comparison table, radar chart, run details, and JSON/CSV exports.
8. **Demo and end-to-end testing:** verify the app works without an API key and with the selected provider configured.
9. **Deployment:** document local setup, GitHub publication, secrets management, and Streamlit Community Cloud limitations.

## 16. Repository and deployment requirements

Suggested structure:

```text
construction-proposal-ranker/
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── src/
│   ├── document_extractor.py
│   ├── schemas.py
│   ├── evaluator.py
│   ├── llm_providers.py
│   ├── rag.py
│   ├── validation.py
│   ├── scoring.py
│   ├── ranking.py
│   ├── human_review.py
│   └── database.py
├── scripts/
│   └── seed_db.py
├── tests/
├── sample_data/
│   ├── specifications/
│   └── suppliers/
└── .streamlit/
    └── config.toml
```

- Include a `requirements.txt` and reproducible local setup instructions.
- Ignore `.venv/`, `__pycache__/`, `.env`, `.streamlit/secrets.toml`, local databases containing real evaluation data, and private uploaded files.
- Never commit API keys, secrets, real supplier confidential documents, or real procurement results to public GitHub.
- For Streamlit Community Cloud, use deployment secrets for provider keys. Do not rely on a local SQLite file as durable shared storage across restarts; use external persistent storage if long-term history is required. Demo data may be recreated from a seed script.
- Include privacy and data handling notices explaining that documents may be sent to the selected provider when the live AI feature is enabled.

## 17. Definition of done

The application is ready for demonstration when:
- Criteria are loaded from SQLite, configurable, and validated to an active total weight of 100%.
- One RFP run accepts multiple supplier PDFs with names, submission dates, and experience ratings.
- Each supplier is evaluated independently against the active criteria and produces evidence-grounded JSON.
- Missing, malformed, and out-of-range criterion results are normalized with saved warnings before scoring.
- Python computes absolute weighted scores, criterion benchmarks and gaps, relative percentages, PPI, and stable ranks.
- Ranks apply eligibility first, followed by PPI, submission date, experience rating, and supplier name.
- Complete results, including failed supplier states, persist to SQLite; leaderboard and comparison views expose results.
- A complete run can be downloaded as JSON, and at least four labelled synthetic supplier PDFs are available.
- The README documents assumptions, formulas, setup, and known limitations.
