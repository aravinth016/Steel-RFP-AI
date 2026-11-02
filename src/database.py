"""SQLite persistence for evaluation outputs and extracted Q&A evidence text."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluations.sqlite3"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_criteria (
                criterion_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                weight REAL NOT NULL,
                max_score INTEGER NOT NULL,
                rubric TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                display_order INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                is_demo INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rfp_runs (
                rfp_run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS supplier_results (
                rfp_run_id TEXT NOT NULL REFERENCES rfp_runs(rfp_run_id) ON DELETE CASCADE,
                supplier_id TEXT NOT NULL,
                supplier_name TEXT NOT NULL,
                submission_date TEXT,
                experience_rating REAL,
                absolute_score REAL,
                ppi REAL,
                final_rank INTEGER,
                result_json TEXT NOT NULL,
                PRIMARY KEY (rfp_run_id, supplier_id)
            );
            CREATE TABLE IF NOT EXISTS review_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                supplier_id TEXT NOT NULL,
                criterion_id TEXT,
                event_at TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rag_pages (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                source_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                scope TEXT NOT NULL,
                supplier_id TEXT,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (run_id, source_id, page)
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflow_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                supplier_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                node TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                details_json TEXT NOT NULL,
                UNIQUE(run_id, supplier_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS evaluation_warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                supplier_id TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, supplier_id, message)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                supplier_id TEXT,
                event_type TEXT NOT NULL,
                event_at TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS criterion_evaluations (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                supplier_id TEXT NOT NULL,
                criterion_id TEXT NOT NULL,
                score REAL,
                status TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                PRIMARY KEY (run_id, supplier_id, criterion_id)
            );
            CREATE TABLE IF NOT EXISTS audit_chain (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS audit_chain_no_update
            BEFORE UPDATE ON audit_chain
            BEGIN
                SELECT RAISE(ABORT, 'audit_chain is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS audit_chain_no_delete
            BEFORE DELETE ON audit_chain
            BEGIN
                SELECT RAISE(ABORT, 'audit_chain is append-only');
            END;
            """
        )
        from src.scoring import STARTER_CRITERIA

        existing_rows = db.execute(
            """
            SELECT criterion_id, name, description, weight, max_score, rubric, is_active
            FROM evaluation_criteria
            """
        ).fetchall()
        existing_ids = {row["criterion_id"] for row in existing_rows}
        legacy_template = {
            "commercial": "Commercial Value",
            "implementation": "Implementation Plan",
            "security": "Security & Compliance",
            "support": "Support & Experience",
            "technical": "Technical Capability",
        }
        old_steel_template = {
            "C01": (
                "Technical and grade compliance",
                "Offered steel grade, requested grade, dimensions or product form, stated specification, deviations, and compliance evidence.",
                30.0,
                "9-10: stated RFP requirements fully addressed with clear evidence; 7-8: meets requirements with minor documented gaps; 4-6: partially meets or needs clarification; 1-3: material deviation shown; 0: no usable evidence. Missing facts must be marked NOT_PROVIDED, not assumed non-compliant.",
            ),
            "C02": (
                "Quality and test documentation",
                "Mill Test Certificate availability, heat/batch traceability, test results, quality certificates, inspection details, and document validity.",
                15.0,
                "9-10: applicable quality/test documents are complete, traceable and clearly cited; 7-8: strong evidence with minor gaps; 4-6: partial evidence or verification needed; 1-3: major gaps shown; 0: no usable evidence. Document presence does not prove authenticity.",
            ),
            "C03": (
                "Commercial value",
                "Comparable price per MT, total quoted amount, freight, taxes, unloading, other charges, exclusions, and payment terms.",
                20.0,
                "9-10: transparent, complete and comparable evaluated cost under the RFP method; 7-8: clear quote with minor gaps; 4-6: partially itemized or comparability needs review; 1-3: material commercial uncertainty; 0: no usable quote evidence. Do not infer missing charges.",
            ),
            "C04": (
                "Quantity and delivery capability",
                "Quantity offered against required quantity, availability, delivery dates, delivery phases, and delivery to the required site.",
                15.0,
                "9-10: quantity and delivery commitments clearly meet the RFP with traceable evidence; 7-8: credible commitments with minor gaps; 4-6: partial availability or date clarification needed; 1-3: material shortfall or uncertainty shown; 0: no usable evidence. Apply mandatory consequences only when explicitly configured from the RFP.",
            ),
            "C05": (
                "Supplier experience and references",
                "Relevant steel supply experience, similar project references, capacity claims, and supporting evidence.",
                10.0,
                "9-10: multiple directly relevant, well-supported references and capacity evidence; 7-8: credible relevant experience with minor gaps; 4-6: limited or partly verifiable experience; 1-3: weak evidence; 0: no usable evidence.",
            ),
            "C06": (
                "Warranty and replacement support",
                "Warranty terms, rejection/replacement process, response commitments, and support obligations.",
                5.0,
                "9-10: clear, comprehensive commitments aligned with RFP; 7-8: adequate terms with minor gaps; 4-6: partial or ambiguous terms; 1-3: significant limitations; 0: no usable evidence. Do not infer terms.",
            ),
            "C07": (
                "Documentation and quotation completeness",
                "Required forms, signed quotation, validity period, commercial breakdown, declarations, and traceable supporting documents.",
                5.0,
                "9-10: requested documents and quote details complete and traceable; 7-8: minor omissions; 4-6: material gaps needing clarification; 1-3: major incompleteness; 0: no usable documents. Missing documents are NOT_PROVIDED, not proof of test failure.",
            ),
        }
        is_legacy_template = (
            len(existing_rows) == len(legacy_template)
            and {row["criterion_id"]: row["name"] for row in existing_rows}
            == legacy_template
        )
        is_old_steel_template = (
            len(existing_rows) == len(old_steel_template)
            and {
                row["criterion_id"]: (
                    row["name"],
                    row["description"],
                    float(row["weight"]),
                    int(row["max_score"]),
                    row["rubric"],
                    int(row["is_active"]),
                )
                for row in existing_rows
            }
            == {
                criterion_id: (name, description, weight, 10, rubric, 1)
                for criterion_id, (name, description, weight, rubric) in old_steel_template.items()
            }
        )
        should_seed = not existing_ids or is_legacy_template or is_old_steel_template
        if should_seed:
            db.execute("DELETE FROM evaluation_criteria")
            db.executemany(
                """
                INSERT INTO evaluation_criteria(
                    criterion_id, name, description, weight, max_score, rubric, is_active, display_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["id"],
                        item["name"],
                        item["description"],
                        float(item["weight"]),
                        int(item["max_score"]),
                        item["rubric"],
                        int(bool(item.get("active", True))),
                        index,
                    )
                    for index, item in enumerate(STARTER_CRITERIA)
                ],
            )


def load_criteria() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT criterion_id, name, description, weight, max_score, rubric, is_active
            FROM evaluation_criteria ORDER BY display_order, criterion_id
            """
        ).fetchall()
    return [
        {
            "id": row["criterion_id"],
            "name": row["name"],
            "description": row["description"],
            "weight": float(row["weight"]),
            "max_score": int(row["max_score"]),
            "rubric": row["rubric"],
            "active": bool(row["is_active"]),
        }
        for row in rows
    ]


def save_criteria(criteria: list[dict[str, Any]]) -> None:
    with connect() as db:
        db.execute("DELETE FROM evaluation_criteria")
        db.executemany(
            """
            INSERT INTO evaluation_criteria(
                criterion_id, name, description, weight, max_score, rubric, is_active, display_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(item["id"]),
                    str(item["name"]),
                    str(item.get("description", "")),
                    float(item["weight"]),
                    int(item["max_score"]),
                    str(item["rubric"]),
                    int(bool(item.get("active", True))),
                    index,
                )
                for index, item in enumerate(criteria)
            ],
        )


def get_app_setting(key: str) -> str | None:
    with connect() as db:
        row = db.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            (key,),
        ).fetchone()
    return str(row["value"]) if row else None


def save_app_setting(key: str, value: str) -> None:
    with connect() as db:
        db.execute(
            """
            INSERT INTO app_settings(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )


def reserve_next_run_id() -> str:
    sequence_key = "_run_id_sequence"
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        setting = db.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            (sequence_key,),
        ).fetchone()
        stored_sequence = int(setting["value"]) if setting else 0
        existing_ids = db.execute("SELECT run_id FROM runs").fetchall()
        existing_sequence = max(
            (
                int(match.group(1))
                for row in existing_ids
                if (match := re.fullmatch(r"RFP-(\d+)", row["run_id"]))
            ),
            default=0,
        )
        next_sequence = max(stored_sequence, existing_sequence) + 1
        db.execute(
            """
            INSERT INTO app_settings(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (sequence_key, str(next_sequence)),
        )
    return f"RFP-{next_sequence:04d}"


def save_run(payload: dict[str, Any]) -> None:
    with connect() as db:
        db.execute(
            """
            INSERT INTO runs(run_id, created_at, mode, status, is_demo, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status, payload=excluded.payload
            """,
            (
                payload["run_id"],
                payload["created_at"],
                payload["mode"],
                payload["status"],
                int(payload.get("is_demo", False)),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        db.execute(
            """
            INSERT INTO rfp_runs(rfp_run_id, created_at, status) VALUES (?, ?, ?)
            ON CONFLICT(rfp_run_id) DO UPDATE SET
                created_at=excluded.created_at, status=excluded.status
            """,
            (payload["run_id"], payload["created_at"], payload["status"]),
        )
        db.execute("DELETE FROM supplier_results WHERE rfp_run_id = ?", (payload["run_id"],))
        db.executemany(
            """
            INSERT INTO supplier_results(
                rfp_run_id, supplier_id, supplier_name, submission_date, experience_rating,
                absolute_score, ppi, final_rank, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    payload["run_id"],
                    supplier["supplier_id"],
                    supplier["name"],
                    supplier.get("submission_date"),
                    supplier.get("experience"),
                    supplier.get("absolute_score"),
                    supplier.get("ppi"),
                    supplier.get("rank"),
                    json.dumps(supplier, ensure_ascii=False),
                )
                for supplier in payload.get("suppliers", [])
            ],
        )
        for table in (
            "workflow_traces",
            "evaluation_warnings",
            "audit_events",
            "criterion_evaluations",
        ):
            db.execute(f"DELETE FROM {table} WHERE run_id = ?", (payload["run_id"],))
        db.executemany(
            """
            INSERT INTO workflow_traces(
                run_id, supplier_id, sequence, node, status, started_at, completed_at, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    payload["run_id"],
                    trace["supplier_id"],
                    int(trace["sequence"]),
                    trace["node"],
                    trace["status"],
                    trace["started_at"],
                    trace["completed_at"],
                    json.dumps(trace.get("details", {}), ensure_ascii=False),
                )
                for trace in payload.get("workflow_traces", [])
            ],
        )
        db.executemany(
            """
            INSERT INTO audit_events(run_id, supplier_id, event_type, event_at, details_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    payload["run_id"],
                    event.get("supplier_id"),
                    event["event_type"],
                    event["event_at"],
                    json.dumps(event.get("details", {}), ensure_ascii=False),
                )
                for event in payload.get("audit_events", [])
            ],
        )
        suppliers_by_name = {
            str(supplier.get("name", "")).casefold(): str(supplier["supplier_id"])
            for supplier in payload.get("suppliers", [])
            if supplier.get("name") and supplier.get("supplier_id")
        }
        warning_rows = []
        for warning in payload.get("warnings", []):
            message = str(warning)
            supplier_id = next(
                (
                    supplier_id
                    for name, supplier_id in suppliers_by_name.items()
                    if message.casefold().startswith(f"{name}:")
                ),
                None,
            )
            warning_rows.append(
                (
                    payload["run_id"],
                    supplier_id,
                    message,
                    payload["created_at"],
                )
            )
        db.executemany(
            """
            INSERT OR IGNORE INTO evaluation_warnings(run_id, supplier_id, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            warning_rows,
        )
        db.executemany(
            """
            INSERT INTO criterion_evaluations(
                run_id, supplier_id, criterion_id, score, status, evaluation_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    payload["run_id"],
                    supplier["supplier_id"],
                    item["criterion_id"],
                    (
                        float(item.get("ai_score", item.get("score")))
                        if item.get("ai_score", item.get("score")) is not None
                        else None
                    ),
                    str(item.get("status", item.get("ai_status", "UNKNOWN"))),
                    json.dumps(item, ensure_ascii=False),
                )
                for supplier in payload.get("suppliers", [])
                for item in supplier.get("criteria_results", [])
                if item.get("criterion_id") is not None
            ],
        )
        snapshot_json = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        _append_audit_entry(
            db,
            event_key=hashlib.sha256(
                f"run_snapshot:{payload['run_id']}:{snapshot_json}".encode("utf-8")
            ).hexdigest(),
            run_id=payload["run_id"],
            event_type="run_snapshot_saved",
            event_at=payload["created_at"],
            payload_hash=hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest(),
        )
        for event in payload.get("audit_events", []):
            event_json = json.dumps(
                event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            _append_audit_entry(
                db,
                event_key=hashlib.sha256(
                    f"{payload['run_id']}:{event_json}".encode("utf-8")
                ).hexdigest(),
                run_id=payload["run_id"],
                event_type=str(event["event_type"]),
                event_at=str(event["event_at"]),
                payload_hash=hashlib.sha256(event_json.encode("utf-8")).hexdigest(),
            )


def _append_audit_entry(
    db: sqlite3.Connection,
    *,
    event_key: str,
    run_id: str,
    event_type: str,
    event_at: str,
    payload_hash: str,
) -> None:
    if db.execute(
        "SELECT 1 FROM audit_chain WHERE event_key = ?", (event_key,)
    ).fetchone():
        return
    latest = db.execute(
        "SELECT entry_hash FROM audit_chain ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    previous_hash = latest["entry_hash"] if latest else "0" * 64
    fields = {
        "event_key": event_key,
        "run_id": run_id,
        "event_type": event_type,
        "event_at": event_at,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
    }
    entry_hash = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    db.execute(
        """
        INSERT INTO audit_chain(
            event_key, run_id, event_type, event_at, payload_hash, previous_hash, entry_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_key,
            run_id,
            event_type,
            event_at,
            payload_hash,
            previous_hash,
            entry_hash,
        ),
    )


def load_run(run_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


def save_rag_corpus(run_id: str, documents: list[dict[str, Any]]) -> None:
    """Persist extracted page text locally for later, run-scoped Q&A."""
    with connect() as db:
        db.execute("DELETE FROM rag_pages WHERE run_id = ?", (run_id,))
        db.executemany(
            """
            INSERT INTO rag_pages(run_id, source_id, filename, scope, supplier_id, page, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    document["id"],
                    document["filename"],
                    document.get("scope", "supplier"),
                    document.get("supplier_id"),
                    int(page["page"]),
                    str(page.get("text", "")),
                )
                for document in documents
                for page in document.get("pages", [])
                if str(page.get("text", "")).strip()
            ],
        )


def load_rag_corpus(run_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT source_id, filename, scope, supplier_id, page, text
            FROM rag_pages
            WHERE run_id = ?
            ORDER BY scope, supplier_id, filename, page
            """,
            (run_id,),
        ).fetchall()
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        document = documents.setdefault(
            row["source_id"],
            {
                "id": row["source_id"],
                "filename": row["filename"],
                "scope": row["scope"],
                "supplier_id": row["supplier_id"],
                "pages": [],
            },
        )
        document["pages"].append({"page": row["page"], "text": row["text"]})
    return list(documents.values())


def list_runs() -> list[dict[str, str]]:
    with connect() as db:
        rows = db.execute(
            "SELECT run_id, created_at, mode, status, is_demo FROM runs ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "mode": row["mode"],
            "status": row["status"],
            "is_demo": bool(row["is_demo"]),
        }
        for row in rows
    ]


def record_review(
    run_id: str,
    supplier_id: str,
    criterion_id: str | None,
    reviewer: str,
    decision: str,
    reason: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        db.execute(
            """
            INSERT INTO review_events(run_id, supplier_id, criterion_id, event_at, reviewer, decision, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, supplier_id, criterion_id, timestamp, reviewer, decision, reason),
        )


def review_events(run_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT supplier_id, criterion_id, event_at, reviewer, decision, reason "
            "FROM review_events WHERE run_id = ? ORDER BY id DESC",
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def workflow_traces(run_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT supplier_id, sequence, node, status, started_at, completed_at, details_json
            FROM workflow_traces
            WHERE run_id = ?
            ORDER BY supplier_id, sequence
            """,
            (run_id,),
        ).fetchall()
    return [
        {**dict(row), "details": json.loads(row["details_json"])}
        for row in rows
    ]


def evaluation_warnings(run_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT supplier_id, message, created_at
            FROM evaluation_warnings
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def audit_events(run_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT supplier_id, event_type, event_at, details_json
            FROM audit_events
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()
    return [
        {**dict(row), "details": json.loads(row["details_json"])}
        for row in rows
    ]


def criterion_evaluations(run_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT supplier_id, criterion_id, score, status, evaluation_json
            FROM criterion_evaluations
            WHERE run_id = ?
            ORDER BY supplier_id, criterion_id
            """,
            (run_id,),
        ).fetchall()
    return [
        {**dict(row), "evaluation": json.loads(row["evaluation_json"])}
        for row in rows
    ]


def verify_audit_chain() -> tuple[bool, int | None]:
    """Verify hashes and persisted run snapshots; return the first invalid sequence."""
    previous_hash = "0" * 64
    with connect() as db:
        rows = db.execute(
            """
            SELECT sequence, event_key, run_id, event_type, event_at, payload_hash,
                   previous_hash, entry_hash
            FROM audit_chain
            ORDER BY sequence
            """
        ).fetchall()
        snapshots = db.execute(
            """
            SELECT run_id, payload FROM runs
            """
        ).fetchall()
        latest_snapshots = db.execute(
            """
            SELECT chain.run_id, chain.sequence, chain.payload_hash
            FROM audit_chain AS chain
            JOIN (
                SELECT run_id, MAX(sequence) AS sequence
                FROM audit_chain
                WHERE event_type = 'run_snapshot_saved'
                GROUP BY run_id
            ) AS latest
              ON latest.run_id = chain.run_id AND latest.sequence = chain.sequence
            """
        ).fetchall()
    for row in rows:
        fields = {
            "event_key": row["event_key"],
            "run_id": row["run_id"],
            "event_type": row["event_type"],
            "event_at": row["event_at"],
            "payload_hash": row["payload_hash"],
            "previous_hash": previous_hash,
        }
        expected_hash = hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if row["previous_hash"] != previous_hash or row["entry_hash"] != expected_hash:
            return False, int(row["sequence"])
        previous_hash = row["entry_hash"]
    snapshot_hashes = {
        row["run_id"]: row["payload_hash"] for row in latest_snapshots
    }
    for snapshot in snapshots:
        canonical_payload = json.dumps(
            json.loads(snapshot["payload"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_payload_hash = hashlib.sha256(
            canonical_payload.encode("utf-8")
        ).hexdigest()
        if (
            snapshot["run_id"] in snapshot_hashes
            and snapshot_hashes[snapshot["run_id"]] != expected_payload_hash
        ):
            run_row = next(
                (
                    row for row in rows
                    if row["run_id"] == snapshot["run_id"]
                    and row["event_type"] == "run_snapshot_saved"
                ),
                None,
            )
            return False, int(run_row["sequence"]) if run_row else None
    return True, None


def list_audit_chain(run_id: str | None = None) -> list[dict[str, Any]]:
    with connect() as db:
        if run_id is None:
            rows = db.execute(
                """
                SELECT sequence, run_id, event_type, event_at, payload_hash,
                       previous_hash, entry_hash
                FROM audit_chain ORDER BY sequence
                """
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT sequence, run_id, event_type, event_at, payload_hash,
                       previous_hash, entry_hash
                FROM audit_chain WHERE run_id = ? ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
    return [dict(row) for row in rows]
