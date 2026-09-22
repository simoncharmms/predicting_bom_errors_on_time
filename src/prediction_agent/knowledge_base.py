"""
Shared knowledge base: a temporal knowledge graph plus a pattern registry.

Backed by SQLite (stdlib only, no extra dependency) so that several agents can
attach to the same file. Three rules make a multi-agent KB usable:

  1. Append-only with provenance. Every row records the writing agent, its model
     version, a confidence and a timestamp. Agents may contradict each other;
     the graph keeps both statements and the consumer resolves.
  2. An agent may only retract its own assertions (`retract`), never overwrite
     another agent's.
  3. Confidence decays with age, so a pattern nobody confirms drops out of the
     high-precision path instead of lingering forever.

Everything the Prediction Agent learns that is worth sharing - confirmed
patterns, component risk history, engineer verdicts - lives here, which is what
lets a new vehicle generation start warm instead of cold.
"""

import json
import sqlite3
import time
from dataclasses import dataclass

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id   TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,
    props     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src       TEXT NOT NULL,
    rel       TEXT NOT NULL,
    dst       TEXT NOT NULL,
    phase     REAL,
    props     TEXT NOT NULL,
    agent     TEXT NOT NULL,
    written_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_rel ON edges(rel);
CREATE TABLE IF NOT EXISTS assertions (
    assertion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject   TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object    TEXT NOT NULL,
    confidence REAL NOT NULL,
    agent     TEXT NOT NULL,
    model_version TEXT NOT NULL,
    written_at REAL NOT NULL,
    retracted  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS patterns (
    signature TEXT PRIMARY KEY,
    support   INTEGER NOT NULL,
    hits      INTEGER NOT NULL,
    precision REAL NOT NULL,
    lift      REAL NOT NULL,
    generation TEXT,
    confirmed INTEGER NOT NULL DEFAULT 0,
    dismissed INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    part      REAL, component REAL, phase REAL,
    verdict   TEXT NOT NULL,
    signature TEXT,
    agent     TEXT NOT NULL,
    written_at REAL NOT NULL
);
"""


@dataclass
class Assertion:
    subject: str
    predicate: str
    object: str
    confidence: float


class KnowledgeBase:
    def __init__(self, path: str, agent: str = "prediction_agent",
                 model_version: str = "0.1.0", half_life_days: float = 180.0):
        self.path = path
        self.agent = agent
        self.model_version = model_version
        self.half_life_days = half_life_days
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- graph -------------------------------------------------------------
    def upsert_nodes(self, rows: list[tuple[str, str, dict]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO nodes(node_id, kind, props) VALUES (?,?,?)",
            [(nid, kind, json.dumps(props)) for nid, kind, props in rows])
        self.conn.commit()

    def add_edges(self, rows: list[tuple[str, str, str, float, dict]]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT INTO edges(src, rel, dst, phase, props, agent, written_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [(s, r, d, phase, json.dumps(p), self.agent, now)
             for s, r, d, phase, p in rows])
        self.conn.commit()

    def neighbours(self, node_id: str, rel: str | None = None) -> list[tuple]:
        query = "SELECT src, rel, dst, phase, props FROM edges WHERE src = ?"
        params: list = [node_id]
        if rel:
            query += " AND rel = ?"
            params.append(rel)
        return self.conn.execute(query, params).fetchall()

    # -- assertions --------------------------------------------------------
    def assert_many(self, assertions: list[Assertion]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT INTO assertions(subject, predicate, object, confidence, "
            "agent, model_version, written_at) VALUES (?,?,?,?,?,?,?)",
            [(a.subject, a.predicate, a.object, a.confidence, self.agent,
              self.model_version, now) for a in assertions])
        self.conn.commit()

    def retract(self, subject: str, predicate: str) -> int:
        """Rule 2: an agent may only retract its own assertions."""
        cur = self.conn.execute(
            "UPDATE assertions SET retracted = 1 WHERE subject = ? AND "
            "predicate = ? AND agent = ? AND retracted = 0",
            (subject, predicate, self.agent))
        self.conn.commit()
        return cur.rowcount

    def read(self, predicate: str, include_foreign: bool = True) -> pd.DataFrame:
        query = ("SELECT subject, predicate, object, confidence, agent, "
                 "model_version, written_at FROM assertions "
                 "WHERE predicate = ? AND retracted = 0")
        params: list = [predicate]
        if not include_foreign:
            query += " AND agent = ?"
            params.append(self.agent)
        df = pd.read_sql_query(query, self.conn, params=params)
        if len(df):
            df["decayed_confidence"] = df.apply(
                lambda r: self.decay(r.confidence, r.written_at), axis=1)
        return df

    def decay(self, confidence: float, written_at: float) -> float:
        """Rule 3: exponential decay with a configurable half-life."""
        age_days = max(0.0, (time.time() - written_at) / 86400.0)
        return float(confidence * 0.5 ** (age_days / self.half_life_days))

    # -- pattern registry --------------------------------------------------
    def put_patterns(self, frame: pd.DataFrame) -> None:
        """frame: signature, support, hits, precision, lift, generation."""
        now = time.time()
        self.conn.executemany(
            "INSERT INTO patterns(signature, support, hits, precision, lift, "
            "generation, updated_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(signature) DO UPDATE SET support=excluded.support, "
            "hits=excluded.hits, precision=excluded.precision, "
            "lift=excluded.lift, updated_at=excluded.updated_at",
            [(r.signature, int(r.support), int(r.hits), float(r.precision),
              float(r.lift), str(r.get("generation", "all")), now)
             for _, r in frame.iterrows()])
        self.conn.commit()

    def get_patterns(self, min_support: int = 1) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM patterns WHERE support >= ?", self.conn,
            params=[min_support])

    # -- human feedback ----------------------------------------------------
    def record_feedback(self, rows: list[tuple[float, float, float, str, str]]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT INTO feedback(part, component, phase, verdict, signature, "
            "agent, written_at) VALUES (?,?,?,?,?,?,?)",
            [(p, c, ph, v, sig, self.agent, now) for p, c, ph, v, sig in rows])
        # A verdict updates the registry counters, which is what makes
        # precision a learnable quantity rather than a static evaluation.
        for _, _, _, verdict, sig in rows:
            column = "confirmed" if verdict == "confirmed" else "dismissed"
            self.conn.execute(
                f"UPDATE patterns SET {column} = {column} + 1 WHERE signature = ?",
                (sig,))
        self.conn.commit()

    def feedback_precision(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT signature, confirmed, dismissed, "
            "CAST(confirmed AS REAL) / NULLIF(confirmed + dismissed, 0) "
            "AS human_precision FROM patterns "
            "WHERE confirmed + dismissed > 0", self.conn)

    def stats(self) -> dict:
        def count(table):
            return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {t: count(t) for t in
                ("nodes", "edges", "assertions", "patterns", "feedback")}

    def close(self) -> None:
        self.conn.close()
