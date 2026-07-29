"""
db.py

Phase 3. Where a call, a lead and a booking are persisted.

    from db import Database
    db = Database()
    await db.connect()                       # creates the file and schema
    call_id = await db.start_call(transport="webrtc")
    await db.upsert_lead(call_id, budget_max=400000, area="Cedar Park")
    await db.upsert_lead(call_id, bedrooms=3)      # merges, does not overwrite
    await db.end_call(call_id)

WRITE STRATEGY: PROGRESSIVE UPSERT
----------------------------------
One lead row per call, updated as the caller reveals things, rather than one
write at the end of qualification.

The reason is the failure mode. This is a phone line, and people hang up
partway through: distracted, interrupted, changed their mind. A caller who said
"around four hundred thousand, three bedrooms, near Cedar Park" and then left
is a genuinely valuable lead. An agent can call them back. Under a write-once
design that caller leaves nothing behind at all, and the most common real world
outcome produces the emptiest database row.

The cost is real and worth stating: every tool call is an extra LLM round trip,
and Gemini's TTFB tail on this project has been measured up to 4.6s, so turns
that trigger a save can feel slower. That is why `upsert_lead` merges rather
than replaces, so the model can call it with only the fields it just learned
instead of restating the whole lead every time, and why the prompt asks for a
save when NEW information arrives rather than every turn.

EVERY LEAD FIELD IS NULLABLE
-----------------------------
Not laziness, it is what progressive upsert means. At the moment of the first
write the agent knows one thing. A NOT NULL on `budget_max` would reject the
partial lead this whole design exists to keep.

The one thing that is NOT nullable is `call_id`. A lead with no call is
unreachable and meaningless.

POSTGRES PORTABILITY
--------------------
The handoff says a Postgres move should be easy, so:

- Timestamps are TEXT holding ISO-8601 UTC, not SQLite's `datetime()` output
  and not Unix ints. `2026-07-29T01:34:12Z` parses identically in both, and
  moving to a real `timestamptz` later is a column type change, not a data
  rewrite.
- Money is INTEGER, in whole currency units. No floats anywhere near a price.
- No SQLite-only syntax beyond `INTEGER PRIMARY KEY` (the rowid alias) and
  `ON CONFLICT`, both of which have direct Postgres equivalents (`GENERATED AS
  IDENTITY` and the same `ON CONFLICT`).
- Every query is parameterised. Aside from injection, `?` versus `%s` is a
  mechanical swap only if nothing is being string-formatted in the first place.

CONCURRENCY
-----------
aiosqlite runs SQLite on a worker thread, so `await` never blocks the event
loop. That matters more here than it looks: this process is also streaming
audio, and a synchronous write that stalls the loop for even 50ms is an audible
gap in the caller's ear. The handoff calls this out explicitly.

Each call gets its own connection via `connect()`, and WAL is enabled so a
reader (a future dashboard, or you poking at the file with sqlite3) never
blocks the writer.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

DEFAULT_DB_PATH = Path(__file__).parent / "cedar_grove.db"

# Fields the agent is allowed to write to a lead. Anything not in here is
# rejected rather than silently ignored: a model that invents a field name
# should produce a loud error during development, not a write that quietly
# drops half the caller's answer.
LEAD_FIELDS = (
    "name",
    "phone",
    "intent",
    "budget_min",
    "budget_max",
    "area",
    "bedrooms",
    "property_type",
    "timeline",
    "financing_status",
    "notes",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY,
    call_sid    TEXT,
    stream_sid  TEXT,
    transport   TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT
);

CREATE TABLE IF NOT EXISTS leads (
    id               INTEGER PRIMARY KEY,
    call_id          INTEGER NOT NULL REFERENCES calls(id),
    name             TEXT,
    phone            TEXT,
    intent           TEXT,
    budget_min       INTEGER,
    budget_max       INTEGER,
    area             TEXT,
    bedrooms         INTEGER,
    property_type    TEXT,
    timeline         TEXT,
    financing_status TEXT,
    notes            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- One lead per call. This is what makes upsert_lead an UPSERT rather than an
-- append: without it, a caller answering six questions produces six lead rows
-- and an agent calling back has to guess which is current.
CREATE UNIQUE INDEX IF NOT EXISTS idx_leads_call_id ON leads(call_id);

CREATE TABLE IF NOT EXISTS bookings (
    id                INTEGER PRIMARY KEY,
    lead_id           INTEGER NOT NULL REFERENCES leads(id),
    listing_ref       TEXT,
    scheduled_for     TEXT,
    confirmation_code TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bookings_lead_id ON bookings(lead_id);
"""


def _now() -> str:
    """ISO-8601 UTC with a Z suffix. One timestamp format, everywhere."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UnknownLeadField(ValueError):
    """Raised when a write names a column that does not exist.

    Deliberately loud. The alternative is filtering unknown keys out, which
    turns a model hallucinating `budget` instead of `budget_max` into a lead
    that silently loses its budget, and you find out weeks later looking at an
    empty column.
    """


class Database:
    """Async SQLite access for calls, leads and bookings."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.getenv("DATABASE_PATH") or DEFAULT_DB_PATH)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the connection and make sure the schema exists."""
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        # WAL so a reader never blocks the writer. Without it, opening the file
        # in a sqlite3 shell mid-call can lock out the agent's own writes.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # Enforce the call_id and lead_id references. SQLite leaves this OFF by
        # default, which means a REFERENCES clause is documentation rather than
        # a constraint unless you ask for it, per connection, every time.
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info(f"Database ready at {self.path}")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() was never awaited.")
        return self._conn

    # -- calls -------------------------------------------------------------

    async def start_call(
        self,
        transport: str,
        call_sid: str | None = None,
        stream_sid: str | None = None,
    ) -> int:
        """Record that a call began. Returns the call id everything else hangs off."""
        cursor = await self._db.execute(
            "INSERT INTO calls (call_sid, stream_sid, transport, started_at) "
            "VALUES (?, ?, ?, ?)",
            (call_sid, stream_sid, transport, _now()),
        )
        await self._db.commit()
        call_id = cursor.lastrowid
        logger.info(f"Call {call_id} started on {transport}")
        return int(call_id)

    async def end_call(self, call_id: int) -> None:
        """Stamp the call as finished. Safe to call twice."""
        await self._db.execute(
            "UPDATE calls SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (_now(), call_id),
        )
        await self._db.commit()

    # -- leads -------------------------------------------------------------

    async def upsert_lead(self, call_id: int, **fields: Any) -> int:
        """Create or update the single lead attached to this call.

        Merges: only the fields passed are written, everything else is left
        alone. So `upsert_lead(id, area="Cedar Park")` followed by
        `upsert_lead(id, bedrooms=3)` leaves a lead holding both, which is the
        whole point of progressive capture.

        Passing None for a field is treated as "no information" and skipped
        rather than as "erase this". The model returns nulls for slots it has
        not filled, and a later call must not wipe an earlier answer.

        ONE STATEMENT, DELIBERATELY. The obvious implementation is SELECT, then
        INSERT or UPDATE depending on what came back. That has a window between
        the read and the write, and Gemini can emit parallel function calls, so
        the window is reachable. Measured on 2026-07-29 with four concurrent
        first-writes on a fresh call: one row survived, three raised
        IntegrityError against the unique index, and the caller's bedrooms,
        intent and budget were dropped. The index protected the data's shape
        and lost its content, which is the worse half of the problem.

        `ON CONFLICT ... DO UPDATE` collapses both branches into a single
        atomic statement, so there is no window to race in.
        """
        unknown = set(fields) - set(LEAD_FIELDS)
        if unknown:
            raise UnknownLeadField(
                f"Not lead columns: {sorted(unknown)}. Known: {list(LEAD_FIELDS)}"
            )

        known = {k: v for k, v in fields.items() if v is not None}
        now = _now()

        columns = ("call_id", *known, "created_at", "updated_at")
        # `excluded` is the row the INSERT tried to add, so this writes exactly
        # the fields passed in and leaves every other column untouched.
        updates = ", ".join(f"{name} = excluded.{name}" for name in known)
        updates = f"{updates}, updated_at = excluded.updated_at" if updates else (
            "updated_at = excluded.updated_at"
        )

        rows = await self._db.execute_fetchall(
            f"INSERT INTO leads ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))}) "
            f"ON CONFLICT(call_id) DO UPDATE SET {updates} "
            f"RETURNING id",
            (call_id, *known.values(), now, now),
        )
        await self._db.commit()
        lead_id = int(rows[0]["id"])
        logger.info(f"Lead {lead_id} on call {call_id} captured: {sorted(known)}")
        return lead_id

    async def get_lead(self, call_id: int) -> dict[str, Any] | None:
        """The current state of this call's lead, or None if nothing captured yet."""
        rows = await self._db.execute_fetchall(
            "SELECT * FROM leads WHERE call_id = ?", (call_id,)
        )
        return dict(rows[0]) if rows else None

    async def missing_fields(self, call_id: int, required: tuple[str, ...]) -> list[str]:
        """Which of `required` this lead still has no answer for.

        Exists so the agent can be told what is still outstanding rather than
        having to remember across a long call, which is the thing models are
        worst at.
        """
        lead = await self.get_lead(call_id)
        if lead is None:
            return list(required)
        return [name for name in required if lead.get(name) is None]

    # -- bookings ----------------------------------------------------------

    async def create_booking(
        self,
        lead_id: int,
        listing_ref: str | None = None,
        scheduled_for: str | None = None,
        confirmation_code: str | None = None,
    ) -> int:
        """Record a viewing. Phase 4 wires the agent-facing tool for this."""
        cursor = await self._db.execute(
            "INSERT INTO bookings (lead_id, listing_ref, scheduled_for, "
            "confirmation_code, created_at) VALUES (?, ?, ?, ?, ?)",
            (lead_id, listing_ref, scheduled_for, confirmation_code, _now()),
        )
        await self._db.commit()
        booking_id = int(cursor.lastrowid)
        logger.info(f"Booking {booking_id} created for lead {lead_id}")
        return booking_id
