# Forge Steel RFP AI — Application Overview

## Purpose

Forge Steel RFP AI is a Streamlit decision-support application for comparing reinforcement-steel supplier quotations against a shared Request for Proposal (RFP). It combines PDF text extraction, steel-focused structured extraction, evidence validation, deterministic scoring and ranking, SQLite persistence, and human review.

The application does not award contracts or make legal decisions. AI-proposed criterion scores feed weighted totals and ranking directly. Users should review evidence and findings and make the final procurement decision under the RFP policy.

## Application workflow

1. **Configure criteria** — Review and edit the active evaluation criteria and weights. Active weighted criteria must total 100%.
2. **Set up an RFP run** — Select the LLM provider and model in LLM Settings, upload the buyer RFP and supporting steel documents, and enter only optional buyer requirements stated in those documents.
3. **Enter supplier information** — Provide each supplier's name, submission date, experience rating, and PDFs for technical offer, quotation, MTC, certificates, and support evidence.
4. **Authorize provider use** — Confirm that uploaded document excerpts, buyer requirements, and supplier metadata may be sent to the selected provider.
5. **Evaluate quotations** — The selected model assesses each supplier independently, extracting technical offer, quality documents, commercial quote, delivery, criterion findings, risks, and clarifications. Local keyword retrieval is scoped to the RFP and current supplier. Evidence quotes are verified against source pages.
6. **Compare and review** — Inspect the steel-specific leaderboard, quote breakdown, technical offer, quality evidence, delivery, criterion scores, risks, clarifications, and evidence.
7. **Recalculate and rank** — Python calculates weighted scores from AI-proposed criterion scores, peer benchmarks, PPI, commercial price score when safely comparable, and deterministic rank. There is no separate eligibility gate or score approval workflow.
8. **Ask questions and export** — Use the run-scoped Q&A Bot and download a leaderboard CSV or complete run JSON.

## Main areas

- **Home** — Explains the decision-support workflow and offers a completed, deterministic synthetic walkthrough that does not call an LLM.
- **New steel RFP evaluation** — Creates a run, accepts the shared RFP and steel quotation packages, captures buyer requirements, requests data-sharing consent, and calls the selected LLM.
- **Criteria** — Edits criterion names, descriptions, weights, maximum scores, rubrics, and active states. The saved template is stored in SQLite and each run snapshots its criteria.
- **Leaderboard** — Displays supplier rank, AI weighted scores, PPI, grade, offered quantity, quoted total, MTC count, warnings, and export controls.
- **Compare & review** — Shows AI criterion results and citations, technical/commercial/delivery extractions, quality documents, risks, clarifications, and a normalized radar chart.
- **Q&A Bot** — Searches shared RFP requirements and at most one supplier's proposal at a time; it requires consent before provider calls.
- **LLM Settings** — Selects and persists the provider/model, captures the provider API key, offers a connection test, and configures Q&A embeddings when available. The new evaluation page also exposes the selected provider and confirms which one will receive the run; there is no automatic provider fallback.

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
- The application checks that cited document IDs and pages belong to the run, and verifies evidence quotes against source page text.
- Missing criterion results are added with a score of zero and a warning. Malformed results are normalized with warnings; scores outside the allowed range are clipped. These AI-proposed scores feed the weighted score; users should inspect the evidence and findings.
- Failed supplier evaluations are retained in the run with an error status and prevent a winner recommendation from an incomplete comparison.
- Extracted unknown steel fields remain null. Quality-document presence does not establish authenticity. Statuses distinguish missing, uncertain, non-compliant, and non-comparable information.
- Provider keys are not written to SQLite or run exports. The app can read provider-specific environment variables or Streamlit secrets; locally entered keys are held in the Streamlit session.

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

The local SQLite database is `data/evaluations.sqlite3`. It stores criteria, run and supplier result payloads, review audit events, non-secret app settings, and extracted page text used by saved-run Q&A. Original uploaded PDFs and API keys are not stored in the database. Do not upload confidential documents unless the provider and local data retention have been approved.

## Synthetic sample materials

The **New steel RFP evaluation** page offers a ZIP download containing:

- One fictional, two-page RFP.
- Four fictional, two-page steel quotations: Apex Steels, Bright Metals, Nexa Steelworks, and Orbit Alloys.

Each PDF contains extractable text and is labelled `SYNTHETIC DEMO DATA`. The sample pack is for upload/evaluation testing and requires a configured LLM provider for a live evaluation. It is distinct from the completed synthetic walkthrough on Home, which is deterministic and makes no provider request.

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

The current implementation was last validated with **60 passing unit tests**. The generated sample PDFs were also checked for readable text, two-page length, and the synthetic-data label.

## Current limitations

- No OCR for scanned PDFs.
- No authentication or role-based permissions; reviewer identity is a user-entered label and is not verified.
- No automatic retry of a failed supplier evaluation within the same run.
- A changed proposal should be evaluated in a new run; document version replacement/invalidation is not implemented.
- The local SQLite database is not durable shared storage for a hosted multi-user deployment.
- Q&A uses provider embedding requests where supported; proposal evaluation retrieval is local keyword matching, not embedding-based RAG.
- Live provider calls, rate limits, free-model availability, and model behavior depend on the selected provider and account.
