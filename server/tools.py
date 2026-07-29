"""
tools.py

Phase 3. The agent's first tool: save_lead_details.

The schema and its handler live here rather than in bot.py so that adding
Phase 4's search_listings and book_viewing is an edit to one file, and so the
handlers can be unit tested without building a pipeline. Tonight's lesson from
ENGINEERING-LOG.md 016 was that checking an artifact is not checking a
behaviour, and a handler buried in a closure inside run_bot can only be tested
by running a whole conversation.

HOW THE TOOL REACHES THE DATABASE
----------------------------------
Through `PipelineWorker(app_resources=...)`, which pipecat hands to every tool
handler as `FunctionCallParams.app_resources`. It is passed by reference, so
one `CallResources` is created per conversation and every handler in that
conversation sees the same database connection and the same call_id.

A closure over `run_bot`'s locals would also work and be shorter. This is
better because it makes the handler a plain module-level function taking data,
which can be called directly in a test.

WHY THIS TOOL DOES NOT BLOCK THE CONVERSATION
----------------------------------------------
Registered with `cancel_on_interruption=False`, which despite the name is what
makes a tool ASYNCHRONOUS in pipecat: "the LLM continues the conversation
immediately without waiting for the result, and the result is injected later
via a developer message" (llm_service.py).

That matters here more than usual. The whole objection to progressive capture
is that every tool call costs a round trip, and Gemini's TTFB on this project
has a measured tail up to 4.6s. If the agent had to wait for the save before
speaking, saving on every new fact would undo the entire Phase 5.5 latency
win. It does not have to wait: nothing the agent says next depends on whether
the row was written. So the caller hears the reply while the write happens
behind it.

The tradeoff accepted in exchange: if a write fails, the agent has already
moved on and cannot mention it. That is the right call for a lead capture
tool. The caller cannot help with a database error, and stopping the
conversation to report one would be worse than losing the row. Failures are
logged loudly and returned in the result so they land in the transcript.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from db import Database, UnknownLeadField

# What the agent is trying to collect. Used to tell it what is still
# outstanding, which is the thing models are worst at tracking over a long
# call. Order is roughly the order a natural conversation surfaces them.
REQUIRED_SLOTS = (
    "intent",
    "budget_max",
    "area",
    "bedrooms",
    "property_type",
    "timeline",
    "financing_status",
)


@dataclass
class CallResources:
    """Everything a tool handler needs, scoped to one conversation."""

    db: Database
    call_id: int


SAVE_LEAD_DETAILS_SCHEMA = FunctionSchema(
    name="save_lead_details",
    description=(
        "Save what you have learned about the caller's requirements. Call this "
        "as soon as the caller tells you something new, without waiting for the "
        "conversation to finish. Pass only the fields you just learned; "
        "previously saved values are kept automatically. Never guess a value "
        "the caller did not give you."
    ),
    properties={
        "intent": {
            "type": "string",
            "enum": ["buy", "rent"],
            "description": "Whether the caller wants to buy or rent.",
        },
        "budget_max": {
            "type": "integer",
            "description": (
                "Top of the caller's budget in whole dollars, so 400000 for "
                "four hundred thousand. No commas, no currency symbol."
            ),
        },
        "budget_min": {
            "type": "integer",
            "description": "Bottom of the budget in whole dollars, if they gave a range.",
        },
        "area": {
            "type": "string",
            "description": "Neighbourhood, suburb or area the caller named.",
        },
        "bedrooms": {"type": "integer", "description": "Number of bedrooms wanted."},
        "property_type": {
            "type": "string",
            "description": "For example house, condo, townhome, apartment.",
        },
        "timeline": {
            "type": "string",
            "description": "How soon they want to move, in the caller's own words.",
        },
        "financing_status": {
            "type": "string",
            "description": (
                "Their mortgage situation, for example pre-approved, not yet "
                "pre-approved, paying cash."
            ),
        },
        "name": {"type": "string", "description": "The caller's name."},
        "phone": {"type": "string", "description": "A callback number."},
        "notes": {
            "type": "string",
            "description": (
                "Anything else useful to the agent calling back that does not "
                "fit the other fields."
            ),
        },
    },
    # Nothing is required. That is the point of progressive capture: the agent
    # should be able to save one fact the moment it hears it. Marking anything
    # required would push the model toward waiting until it has a full set,
    # which is exactly the write-once behaviour this design rejected.
    required=[],
)


async def save_lead_details(params: FunctionCallParams) -> None:
    """Merge whatever the caller just revealed into this call's lead row."""
    resources: CallResources | None = params.app_resources
    if resources is None:
        # Misconfiguration, not a caller problem. Loud, because a silently
        # discarded lead is the failure this whole phase exists to prevent.
        logger.error(
            "save_lead_details called with no app_resources. The lead was NOT "
            "saved. Pass CallResources via PipelineWorker(app_resources=...)."
        )
        await params.result_callback({"saved": False, "error": "storage unavailable"})
        return

    fields: dict[str, Any] = {k: v for k, v in dict(params.arguments).items() if v is not None}

    try:
        lead_id = await resources.db.upsert_lead(resources.call_id, **fields)
    except UnknownLeadField as e:
        # The model invented a field name. Save what is valid rather than
        # losing the whole turn's information to one bad key.
        logger.warning(f"save_lead_details got an unknown field: {e}")
        known = {k: v for k, v in fields.items() if k in SAVE_LEAD_DETAILS_SCHEMA.properties}
        lead_id = await resources.db.upsert_lead(resources.call_id, **known)
        fields = known
    except Exception as e:
        logger.exception(f"save_lead_details failed to write: {e!r}")
        await params.result_callback({"saved": False, "error": "could not save"})
        return

    still_missing = await resources.db.missing_fields(resources.call_id, REQUIRED_SLOTS)
    logger.info(
        f"save_lead_details wrote {sorted(fields)} to lead {lead_id}; "
        f"still missing {still_missing}"
    )

    # Telling the model what is outstanding is the useful half of the result.
    # It cannot reliably track seven slots across a ten turn call, and this
    # turns that from a memory problem into a lookup.
    await params.result_callback(
        {
            "saved": True,
            "saved_fields": sorted(fields),
            "still_missing": still_missing,
        }
    )
