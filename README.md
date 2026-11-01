# Forge Steel RFP AI

Forge Steel RFP AI is a local Streamlit decision-support application for comparing supplier quotations for reinforcement steel against the buyer's RFP and supporting requirements. Live evaluations support Claude, Cohere.ai, Gemini, ChatGPT, and OpenRouter. The completed walkthrough is synthetic and works without an API key.

The app does not make a contract award or legal eligibility decision. AI-proposed criterion scores feed the weighted score and ranking directly. Human reviewers must validate evidence and confirm mandatory gates under the project's procurement policy.

## Run locally

```bash
cd "mini project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

In **LLM Settings** or directly on **New steel RFP evaluation**, select the provider and model, enter that provider's API key, and test the connection. The selected provider and model are saved locally and remain active for new evaluations and Q&A until changed. Each new-run page confirms the provider that will be called; the app does not silently switch or fall back to another provider. The same provider key is used for evaluation, Q&A answers, and embeddings where that provider offers an embedding API. For local OpenRouter setup, put the key after `OPENROUTER_API_KEY=` in the ignored `.env` file and restart the app. OpenRouter embeddings use `openai/text-embedding-3-small` by default. Claude has no first-party embeddings endpoint, so its Q&A retrieval uses local BM25 keyword search and Claude generates the answer with the same key. Alternatively, configure provider secrets through Streamlit secrets. Do not put keys in source control. API keys entered in the UI are held in the Streamlit session and are not persisted to SQLite or exports.

If every supplier evaluation with Cohere times out, test the connection in **LLM Settings**, then retry with a supported faster model or another provider. Evaluation requests allow up to 180 seconds for Cohere responses; persistent timeouts can still indicate provider or network latency.

## Demo walkthrough

Choose **Explore the completed demo** on Home. It opens a fully evaluated, ranked and reviewed sample run with four fictional suppliers, different prices and evidence strengths, two failed mandatory gates, a comparison table, chart, JSON/CSV downloads, and a review audit trail. The synthetic suppliers and results are clearly marked and never call an LLM.

For an end-to-end live upload test, use **Download synthetic steel RFP and four quotations** on **New steel RFP evaluation**. The ZIP contains one fictional RFP and four fictional, three-page supplier quotations. Each supplier PDF includes a clearly labeled C01–C05 evidence schedule aligned to the five example categories in the project brief: technical capability, implementation plan, commercial value, security and compliance, and support and experience. These are evidence statements, not pre-assigned criterion scores. Intentional gaps and the Orbit Fe 500 grade deviation remain explicit; missing evidence must not be inferred. Extract the PDFs and upload the RFP as a shared document and each supplier document under its supplier. All files are labelled `SYNTHETIC DEMO DATA`.

For a more detailed, corporate-style walkthrough, use **Download corporate-style tender and two supplier pack** on the same page. This ZIP contains a fictional invitation to tender and technical requirements/evaluation matrix for shared documents, plus technical proposals, deviations/clarifications, commercial offers, and sample MTCs for Apex Steels and Nexa Steelworks. The page also offers a styled, searchable combined submission PDF for each of the four demo suppliers: Apex Steels and Nexa Steelworks use their four corporate-pack documents, while Bright Metals and Orbit Alloys use their available synthetic two-page quotation. Each combined PDF has a cover and bookmarks for its included document sections; the quotation-only files do not invent missing attachments. A styled, combined tender PDF and a separate **Download complete corporate pack as one PDF** button are also available. For a live multi-supplier evaluation, upload the tender-side PDFs as shared requirements, then upload each supplier's source PDFs only in its own supplier section; do not upload an all-in-one PDF as a shared document, as that would mix supplier evidence. The pack deliberately includes a delivery-date exception in Nexa's response to exercise gate and evidence review. Every organization, value, date, and certificate is synthetic; sample MTCs are not valid certificates. The structure is inspired by the supplied document categories but does not reproduce their text or assert real procurement policy.

The corporate-style PDFs, including the Bright Metals and Orbit Alloys template-structured response packs, can be regenerated locally with:

```bash
python sample_data1/generate_corporate_pack.py
```

## Live evaluation

1. Review the criteria and weights on **Criteria**. Criteria are loaded from and saved to SQLite. Active weighted criteria must total exactly 100%; these example weights are not a procurement policy.
2. Upload the buyer's RFP/enquiry and supporting steel specification, BOQ, delivery schedule, and commercial terms. Optionally enter grade, quantity, delivery location/date, currency, and material requirements as stated in those documents. Configure pass/fail gates only from requirements explicitly marked mandatory in the RFP. Upload each supplier's technical offer, quotation, MTC, certificates, and other PDFs. Enter name, submission date, and experience rating. PDFs must contain extractable text and be no larger than 15 MB / 100 pages. OCR of scanned documents is not included.
3. Confirm the provider-specific data-sharing notice. The uploaded PDF text, buyer requirements, configured gates, and supplier metadata are sent to the selected LLM for evaluation. It extracts steel grade, quantity, quality documents, commercial breakdown, delivery, evidence, and criterion assessments. Local keyword retrieval is enabled by default and remains scoped to shared requirements and the individual supplier's own PDFs.
4. The response is parsed and validated before scoring. Missing criterion rows are filled with a zero score and `NOT_PROVIDED`; malformed scores are set to zero and marked `NOT_VERIFIABLE`; out-of-range scores are clipped to the criterion's valid range with a warning. Evidence citations must match an uploaded source and verifiable page text.
5. Review the evidence and each mandatory gate in **Compare & review**. Human reviewers confirm gate outcomes; they do not approve or override criterion scores. Python calculates weighted totals from the AI-proposed scores, per-criterion peer benchmarks, score gaps, relative percentages, PPI, and deterministic ranking.
6. Finalize only after every supplier's mandatory-gate outcomes are reviewed and resolved.

## Q&A Bot

Open **Q&A Bot** from the sidebar after selecting a saved live run. Choose **Project requirements only** or one supplier. Retrieval always includes shared requirements and, at most, the selected supplier's submissions; other suppliers' documents are excluded.

The bot splits extracted pages into overlapping chunks and combines BM25 rankings with embeddings from the currently selected LLM provider (Cohere, OpenRouter, ChatGPT, or Gemini), using reciprocal rank fusion. Claude uses BM25 keyword retrieval:

`RRF(chunk) = 1 / (60 + BM25 rank) + 1 / (60 + bi-encoder rank)`

Only the highest-ranked evidence chunks are sent to the run's selected LLM. Citations use source labels validated against the retrieved set and display the original filename and page. If no evidence passes retrieval, the bot reports that the information does not appear in the uploaded documents.

Before each Q&A scope, confirm that document chunks and questions may be sent to the currently selected provider for embeddings where supported, and that the question plus retrieved excerpts may be sent to that provider for an answer. The app stores extracted page text in the local SQLite database, separate from run JSON exports, so saved runs remain available to the bot; original PDFs and API keys are not stored. Do not use this with confidential documents unless the provider and this local retention are approved.

The five weighted starter categories follow the project brief's examples: Technical Capability (30%), Implementation Plan (20%), Commercial Value (20%), Security & Compliance (20%), and Support & Experience (10%). Their descriptions and rubrics apply these broad categories to the steel-supplier example. The weights remain examples, not procurement policy; review them against the actual procurement policy. Mandatory RFP pass/fail gates remain separate from weighted criteria. A commercial score is calculated only after the reviewer confirms that the RFP permits lowest-total-price scoring and the extracted quotes have positive totals, common currency, matching quantity and delivery basis, and comparable scope. Missing charges are never inferred; otherwise commercial scoring is marked `NOT_COMPARABLE` pending human resolution. Certificates are not treated as independently authenticated.

The app stores run metadata, extracted page text for Q&A, scores, cited snippets and review audit events in `data/evaluations.sqlite3`; it does not store original uploaded PDFs or API keys. The local database is ignored by git. Local SQLite is not durable shared storage for a hosted multi-user deployment.

## Scoring and ranking

`absolute weighted score = sum((criterion score / criterion maximum) × criterion weight)`

`criterion benchmark = highest valid supplier score for that criterion`

`criterion gap = supplier score - criterion benchmark`

`relative performance (%) = supplier score / criterion benchmark × 100`

`PPI = weighted average of available criterion relative-performance percentages`

Mandatory gates are separate from weighted scoring. A supplier with a failed gate is ineligible; pending gates do not qualify for a recommendation. Eligible suppliers are ranked by PPI descending, earlier submission date, higher experience rating, then supplier name ascending. Ineligible suppliers follow eligible ones and use the same ordering. Absolute weighted score is reported but is not an extra tie-break. If a criterion benchmark is zero or unavailable, its relative percentage is undefined and excluded from PPI; PPI is reweighted over the criteria with defined relative percentages. The leaderboard and run details show the ranking policy.

The evaluation prompt uses only the uploaded buyer and current supplier documents, rejects citations to unknown files/pages, and validates exactly one score per active criterion regardless of provider. AI-proposed criterion scores are used directly for weighted totals and ranking; mandatory-gate outcomes remain subject to human review.

## Current MVP boundaries

- The demo is a deterministic synthetic walkthrough; it does not call an LLM.
- Evaluation-time retrieval uses local keyword matching; Q&A uses BM25 plus embeddings from the selected provider where supported, fused with reciprocal rank fusion. Claude uses BM25 only.
- PDF files are not retained. Extracted page text for live runs is stored locally to support saved-run Q&A. To replace a document, start a new run and upload the revised file; old run results remain unchanged.
- Failed supplier evaluations and their warnings are persisted and block a winner recommendation for that run. Start a new run after correcting the issue; automatic in-run retry is not included yet.
- OCR for scanned PDFs, authentication/role permissions, remote hosting, and durable multi-user storage are not included.

## Privacy and deployment

Document excerpts are transmitted to the selected LLM only after an explicit confirmation for each live run. Do not upload confidential documents unless the organization has approved the provider and data handling. For Streamlit Community Cloud, set the relevant provider key in deployment secrets. Use an approved persistent database/storage service for durable multi-user history; do not rely on an ephemeral local SQLite file.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
# Steel-RFP-AI
