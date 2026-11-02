# Forge Steel RFP AI

Forge Steel RFP AI is a local Streamlit decision-support application for comparing supplier quotations for reinforcement steel against the buyer's RFP and supporting requirements. Live evaluations support Claude, Cohere.ai, Gemini, ChatGPT, and OpenRouter. The completed walkthrough is synthetic and works without an API key.

The app does not make a contract award or legal eligibility decision. AI-proposed criterion scores feed weighted scoring and ranking directly; users should review the evidence, risks, and clarifications before making procurement decisions. The app does not implement a separate pass/fail gate or score-approval workflow.

## Run locally

```bash
cd "mini project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

In **LLM Settings**, select the provider and model, enter that provider's API key, and test the connection. The selected provider and model are saved locally and remain active for new evaluations and Q&A until changed; **New steel RFP evaluation** uses that configured selection and does not provide a separate provider selector. Each new-run page confirms the provider that will be called; the app does not silently switch or fall back to another provider. The same provider key is used for evaluation, Q&A answers, and embeddings where that provider offers an embedding API. For local OpenRouter setup, put the key after `OPENROUTER_API_KEY=` in the ignored `.env` file and restart the app. OpenRouter embeddings use `openai/text-embedding-3-small` by default. Claude has no first-party embeddings endpoint, so its Q&A retrieval uses local BM25 keyword search and Claude generates the answer with the same key. Alternatively, configure provider secrets through Streamlit secrets. Do not put keys in source control. API keys entered in the UI are held in the Streamlit session and are not persisted to SQLite or exports.

If every supplier evaluation with Cohere times out, test the connection in **LLM Settings**, then retry with a supported faster model or another provider. Evaluation requests allow up to 180 seconds for Cohere responses; persistent timeouts can still indicate provider or network latency.

## Demo walkthrough

Choose **Explore the completed demo** on Home. It opens a fully evaluated sample run with four fictional suppliers, different prices and evidence strengths, a ranked leaderboard, comparison table, and chart. The synthetic suppliers and results are clearly marked and never call an LLM.

The synthetic RFP and quotation PDFs are in [`sample_data/`](./sample_data/). Regenerate them locally with:

```bash
python sample_data/generate_samples.py
```

All sample organizations, values, dates, and certificates are fictional. Sample MTCs are not valid certificates.

## Live evaluation

1. Review the criteria and weights on **Criteria**. Criteria are loaded from and saved to SQLite. Active weighted criteria must total exactly 100%; these example weights are not a procurement policy.
2. Upload the buyer's RFP/enquiry and supporting steel specification, BOQ, delivery schedule, and commercial terms. Optionally enter grade, quantity, delivery location/date, currency, and material requirements as stated in those documents. Upload each supplier's technical offer, quotation, MTC, certificates, and other PDFs. Enter name, submission date, and experience rating. PDFs must contain extractable text and be no larger than 15 MB / 100 pages. OCR of scanned documents is not included.
3. Confirm the provider-specific data-sharing notice. The uploaded PDF text, buyer requirements, and supplier metadata are sent to the selected LLM for evaluation. It extracts steel grade, quantity, quality documents, commercial breakdown, delivery, evidence, and criterion assessments. Local keyword retrieval is enabled by default and remains scoped to shared requirements and the individual supplier's own PDFs.
4. The response is parsed and validated before scoring. Missing criterion rows are filled with a zero score and `NOT_PROVIDED`; malformed scores are set to zero and marked `NOT_VERIFIABLE`; out-of-range scores are clipped to the criterion's valid range with a warning. Evidence citations must match an uploaded source and verifiable page text.
5. Review the evidence, scores, risks, and clarifications in **Compare & review**. Python calculates weighted totals from AI-proposed scores, per-criterion peer benchmarks, score gaps, relative percentages, PPI, and deterministic ranking. There is no separate eligibility filter or gate-review/finalization flow.
6. Each supplier is processed independently by a LangGraph workflow: prepare, evaluate, strict Pydantic validation, citation verification, deterministic scoring, and result assembly. A failed supplier is recorded without stopping the remaining suppliers. Cohere's existing compact-response retry remains limited to one retry after an output-token-limit failure.

Before any evaluation request is sent, the app displays per-supplier approximate input and output token estimates and asks you to explicitly proceed. Input uses a rough four-characters-per-token prompt heuristic; output uses an approximate per-criterion allowance. These estimates are not provider billing data. When available, actual input/output usage reported by the provider is shown separately after evaluation; providers may omit usage metadata. Cost is shown as unavailable unless a verified current rate for the exact provider/model is configured. Token estimates do not limit document text or model output.

**Compare & review** also includes a what-if analysis. Adjust active criterion weights to see hypothetical weighted scores, ranking movement, and seeded winner sensitivity. These results are labeled non-official and never modify or save the original run.

The **Leaderboard** and **History** pages export evaluations as CSV and PDF; the leaderboard and History retain complete-run JSON exports. The **Q&A Bot** can export a chat as CSV or PDF with source citations. New evaluations receive sequential IDs (`RFP-0001`, `RFP-0002`, …); existing run IDs remain unchanged.

The **History** page reads saved runs from SQLite without invoking an LLM. It shows supplier scorecards, criteria, warnings, audit events, LangGraph stage traces, and a tamper-evident hash chain. The chain links hashes of saved run snapshots and audit events, detects later edits to those snapshots, and rejects updates/deletions through SQLite triggers. It is not tamper-proof against an administrator who can replace the local database. Prompts, API keys, raw model responses, and original PDF files are not included in trace records.

## Q&A Bot

Open **Q&A Bot** from the sidebar after selecting a saved live run. Choose **Project requirements only** or one supplier. Retrieval always includes shared requirements and, at most, the selected supplier's submissions; other suppliers' documents are excluded.

The bot splits extracted pages into overlapping chunks and combines BM25 rankings with embeddings from the currently selected LLM provider (Cohere, OpenRouter, ChatGPT, or Gemini), using reciprocal rank fusion. Claude uses BM25 keyword retrieval:

`RRF(chunk) = 1 / (60 + BM25 rank) + 1 / (60 + bi-encoder rank)`

Only the highest-ranked evidence chunks are sent to the run's selected LLM. Citations use source labels validated against the retrieved set and display the original filename and page. If no evidence passes retrieval, the bot reports that the information does not appear in the uploaded documents.

Before each Q&A scope, confirm that document chunks and questions may be sent to the currently selected provider for embeddings where supported, and that the question plus retrieved excerpts may be sent to that provider for an answer. The app stores extracted page text in the local SQLite database, separate from run JSON exports, so saved runs remain available to the bot; original PDFs and API keys are not stored. Do not use this with confidential documents unless the provider and this local retention are approved.

The five weighted starter categories follow the project brief's examples: Technical Capability (30%), Implementation Plan (20%), Commercial Value (20%), Security & Compliance (20%), and Support & Experience (10%). Their descriptions and rubrics apply these broad categories to the steel-supplier example. The weights remain examples, not procurement policy; review them against the actual procurement policy. No separate pass/fail gate workflow is used. A commercial score is calculated only after the user confirms that the RFP permits lowest-total-price scoring and the extracted quotes have positive totals, common currency, matching quantity and delivery basis, and comparable scope. Missing charges are never inferred; otherwise commercial scoring is marked `NOT_COMPARABLE`. Certificates are not treated as independently authenticated.

The app stores run metadata, extracted page text for Q&A, criterion evaluations, workflow traces, warnings, audit events, and cited snippets in `data/evaluations.sqlite3`; it does not store original uploaded PDFs or API keys. The local database is ignored by git. Local SQLite is not durable shared storage for a hosted multi-user deployment.

## Scoring and ranking

`absolute weighted score = sum((criterion score / criterion maximum) × criterion weight)`

`criterion benchmark = highest valid supplier score for that criterion`

`criterion gap = supplier score - criterion benchmark`

`relative performance (%) = supplier score / criterion benchmark × 100`

`PPI = weighted average of available criterion relative-performance percentages`

Suppliers are ranked by PPI descending, then earlier submission date, higher experience rating, and supplier name ascending. Absolute weighted score is reported but is not an extra tie-break. If a criterion benchmark is zero or unavailable, its relative percentage is undefined and excluded from PPI; PPI is reweighted over the criteria with defined relative percentages. The leaderboard and run details show the ranking policy.

The evaluation prompt uses only the uploaded buyer and current supplier documents, rejects citations to unknown files/pages, and validates exactly one score per active criterion regardless of provider. AI-proposed criterion scores are used directly for weighted totals and ranking.

## Current MVP boundaries

- The demo is a deterministic synthetic walkthrough; it does not call an LLM.
- Evaluation-time retrieval uses local keyword matching; Q&A uses BM25 plus embeddings from the selected provider where supported, fused with reciprocal rank fusion. Claude uses BM25 only.
- PDF files are not retained. Extracted page text for live runs is stored locally to support saved-run Q&A. To replace a document, start a new run and upload the revised file; old run results remain unchanged.
- Failed supplier evaluations and their warnings are persisted and block a winner recommendation for that run. Start a new run after correcting the issue; only Cohere's single compact retry for output-token-limit failures is automatic.
- OCR for scanned PDFs, authentication/role permissions, remote hosting, and durable multi-user storage are not included.

## Privacy and deployment

Document excerpts are transmitted to the selected LLM only after an explicit confirmation for each live run. Do not upload confidential documents unless the organization has approved the provider and data handling. For Streamlit Community Cloud, set the relevant provider key in deployment secrets. Use an approved persistent database/storage service for durable multi-user history; do not rely on an ephemeral local SQLite file.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
# Steel-RFP-AI
# Steel-RFP-AI
