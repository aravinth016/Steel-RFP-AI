# Forge Steel RFP AI — Application Overview

## Purpose

Forge Steel RFP AI is a Streamlit decision-support application for comparing reinforcement-steel supplier quotations against a shared Request for Proposal (RFP). It combines PDF text extraction, a per-supplier LangGraph workflow, strict Pydantic validation, evidence verification, deterministic scoring and ranking, and SQLite history. Reviewers make procurement decisions; the application does not implement a separate gate or score-approval workflow.

The application does not award contracts or make legal decisions. AI-proposed criterion scores feed weighted totals and ranking directly. Users should review evidence and findings and make the final procurement decision under the RFP policy.

## Application workflow

1. **Configure criteria** — Review and edit the active evaluation criteria and weights. Active weighted criteria must total 100%.
2. **Set up an RFP run** — Select the LLM provider and model in LLM Settings, upload the buyer RFP and supporting steel documents, and enter only optional buyer requirements stated in those documents.
3. **Enter supplier information** — Provide each supplier's name, submission date, experience rating, and PDFs for technical offer, quotation, MTC, certificates, and support evidence.
4. **Authorize provider use** — Confirm that uploaded document excerpts, buyer requirements, and supplier metadata may be sent to the selected provider.
5. **Estimate and authorize evaluation** — After local PDF extraction and prompt preparation, the app estimates input tokens using a rough four-characters-per-token heuristic and output tokens using an approximate per-criterion allowance. It displays per-supplier estimates and asks the user to explicitly proceed. Cost remains unavailable unless a verified current rate for the exact provider/model is configured.
6. **Evaluate quotations** — LangGraph processes each supplier independently through preparation, model evaluation, strict response validation, evidence verification, deterministic score calculation, and result assembly. A validated cached result skips the provider call. Local keyword retrieval is scoped to the RFP and current supplier. Evidence quotes are verified against source pages; supplier failures are recorded without stopping the remaining suppliers.
7. **Compare and review** — Inspect the steel-specific leaderboard, quote breakdown, technical offer, quality evidence, delivery, criterion scores, risks, clarifications, and evidence. What-if analysis recalculates hypothetical scores/ranks from edited weights and reports seeded winner sensitivity without changing saved run results.
8. **Recalculate and rank** — Python calculates weighted scores from AI-proposed criterion scores, peer benchmarks, PPI, commercial price score when safely comparable, and deterministic rank. There is no separate eligibility gate or score approval workflow.
9. **Inspect history, ask questions, and export** — Review saved runs, scorecards, warnings, audit events, workflow traces, and a tamper-evident hash chain in History; use the run-scoped Q&A Bot or export a selected run as JSON.

## Main areas

- **Home** — Explains the decision-support workflow, includes a node-by-node StateGraph overview with normal and failure/cached-result paths, and offers a completed synthetic walkthrough that does not call an LLM.
- **New steel RFP evaluation** — Uses the provider and model configured in LLM Settings (without a separate provider selector), accepts the shared RFP and steel quotation packages, captures buyer requirements, requests data-sharing consent, presents the approximate prompt-token estimate, and waits for explicit confirmation before calling the selected LLM.
- **Criteria** — Edits criterion names, descriptions, weights, maximum scores, rubrics, and active states. The saved template is stored in SQLite and each run snapshots its criteria.
- **Leaderboard** — Displays supplier rank, AI weighted scores, PPI, estimated and provider-reported token usage, grade, offered quantity, quoted total, MTC count, warnings, and export controls.
- **Compare & review** — Shows AI criterion results and citations, technical/commercial/delivery extractions, quality documents, risks, clarifications, and a normalized radar chart. It also includes an explicitly hypothetical weight editor and deterministic sensitivity analysis; scenario results do not mutate saved results.
- **History** — Reads saved run payloads and dedicated SQLite trace, warning, audit, and criterion tables; verifies the append-only hash chain and does not call an LLM.
- **Q&A Bot** — Searches shared RFP requirements and at most one supplier's proposal at a time; it requires consent before provider calls.
- **LLM Settings** — Selects and persists the provider/model, captures the provider API key, offers a connection test, and configures Q&A embeddings when available. The new evaluation page displays the configured provider and confirms which one will receive the run; there is no separate provider selector or automatic fallback.

## Starter evaluation rubric

The editable starter template uses a 0–10 criterion score range:

| Criterion | Weight |
|---|---:|
| Technical and grade compliance | 30% |
| Quality and test documentation | 15% |
| Commercial value | 20% |
| Quantity and delivery capability | 15% |
| Supplier experience and references | 10% |
| Warranty and replacement support | 5% |
| Documentation and quotation completeness | 5% |
| **Total** | **100%** |

These steel-focused values are examples, not an official procurement policy. Confirm criteria, weights, scoring rules, and price-comparability rules against the buyer's actual RFP.

## Evaluation and validation

- Supplier PDFs are extracted page by page. Supported files are text-searchable PDFs up to 15 MB and 100 pages. Encrypted, empty, corrupt, or textless/scanned PDFs are rejected with an error; OCR is not included.
- The model is asked to return JSON containing an overall summary, criterion-wise scores and justifications, evidence citations, and steel offer details.
- Normalized responses are checked against strict Pydantic models, the active criterion set, and criterion-specific score maxima before deterministic scoring.
- The application checks that cited document IDs and pages belong to the run, and verifies evidence quotes against source page text.
- Missing criterion results are added with a score of zero and a warning. Malformed results are normalized with warnings; scores outside the allowed range are clipped. These AI-proposed scores feed the weighted score; users should inspect the evidence and findings.
- Failed supplier evaluations are retained in the run with an error status and prevent a winner recommendation from an incomplete comparison.
- Extracted unknown steel fields remain null. Quality-document presence does not establish authenticity. Statuses distinguish missing, uncertain, non-compliant, and non-comparable information.
- Provider keys are not written to SQLite or run exports. The app can read provider-specific environment variables or Streamlit secrets; locally entered keys are held in the Streamlit session.
- Estimated input/output tokens are displayed before the request and do not limit generation. Actual input/output usage is displayed separately when provider metadata is available; absent values remain unavailable. Cost is not estimated without verified provider/model pricing.

## Scoring and ranking

For each active criterion:

```text
absolute weighted score = sum((criterion score / criterion maximum) × criterion weight)
criterion benchmark = highest valid supplier score for that criterion
criterion gap = supplier score - criterion benchmark
relative performance (%) = supplier score / criterion benchmark × 100
PPI = weighted average of available criterion relative-performance percentages
```

If a benchmark is zero or unavailable, relative performance is undefined and excluded from PPI; the remaining weights are renormalized. PPI is undefined when no criterion has a valid relative percentage.

Suppliers are ordered deterministically by higher PPI, earlier submission date, higher experience rating, and supplier name ascending. Absolute weighted score is reported, but is not an extra tie-break. There is no separate eligibility gate or gate-review workflow.

Commercial price scoring is only applied when the user confirms that the RFP permits it and all supplier quotes have positive totals, common currency, matching offered quantity and delivery basis, and explicit comparability. Unstated freight, taxes, and charges are never inferred; non-comparable commercial results remain unscored for human resolution.

## Providers

Live proposal evaluation supports the configured providers and models in LLM Settings:

- Claude
- Cohere.ai
- Gemini
- ChatGPT
- OpenRouter

OpenRouter includes configured free-model options, whose access and capacity are controlled by OpenRouter and may change. A key for one provider does not authenticate to another provider's direct API.

## Q&A and data storage

Q&A scopes evidence to shared RFP requirements and either project requirements alone or one selected supplier. It uses BM25 keyword retrieval and, where supported, embeddings from the selected provider; Claude uses BM25 only. Document embeddings may be cached for the current app session. Answer generation makes a separate LLM request.

The local SQLite database is `data/evaluations.sqlite3`. It stores criteria, run and supplier result payloads, criterion evaluations, workflow traces, evaluation warnings, audit events, non-secret app settings, and extracted page text used by saved-run Q&A. Trace records contain stage metadata only—not API keys, prompts, raw model responses, or original uploaded PDFs. Do not upload confidential documents unless the provider and local data retention have been approved. The selected provider/model has an optional output-token limit control; `0` keeps provider defaults and explicit limits are checked against a conservative provider/model ceiling.

## Synthetic sample materials

Synthetic steel RFP and quotation PDFs are available in the repository's [`sample_data/`](./sample_data/) directory for local upload testing. The app does not offer a download of synthetic files from the New evaluation page. The completed demo on Home remains deterministic and does not call an LLM.

The PDFs contain extractable text and are labelled `SYNTHETIC DEMO DATA`. Live evaluation of sample files requires a configured LLM provider.

The PDFs can be regenerated with:

```bash
python sample_data/generate_samples.py
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Set the selected provider's key in **LLM Settings** or configure it through the local ignored `.env` file / Streamlit secrets. Never commit API keys, real supplier proposals, or confidential evaluation data.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

Run the unit tests to validate the current implementation; the suite includes tests for strict schemas, evidence verification, workflow failure isolation, and SQLite run history.

## Current limitations

- No OCR for scanned PDFs.
- No authentication or role-based permissions; reviewer identity is a user-entered label and is not verified.
- No general automatic retry of failed supplier evaluations; Cohere receives at most one compact retry after an output-token-limit failure.
- A changed proposal should be evaluated in a new run; document version replacement/invalidation is not implemented.
- The local SQLite database is not durable shared storage for a hosted multi-user deployment.
- Q&A uses provider embedding requests where supported; proposal evaluation retrieval is local keyword matching, not embedding-based RAG.
- Live provider calls, rate limits, free-model availability, and model behavior depend on the selected provider and account.
