"""
latency.py

Phase 5.5. Per-stage latency, one line per turn, a summary at hangup, and a
file you can compare across models.

    uv run bot.py -t webrtc          instrumentation is on, watch the log
    uv run latency.py                compare every run recorded so far
    uv run latency.py --clear        start a fresh comparison

WHAT THIS IS FOR
----------------
HANDOFF.md's Phase 3 question is which model to run, and it says to decide on
"tool-call correctness AND latency variance, not just median speed". The
numbers backing that up were these, read out of a log by hand:

    Deepgram TTFB     0.283 to 0.294s    4% spread
    ElevenLabs TTFA   0.169 to 0.282s    fine
    Gemini TTFB       0.356 to 2.023s    5.7x spread, the problem

Four samples, transcribed by eye, from one run. That is enough to form a
suspicion and not enough to decide anything. This makes the same measurement
automatic, and writes it somewhere two models can be compared.

WHY THERE IS ALMOST NO MEASUREMENT CODE HERE
---------------------------------------------
Pipecat already does the measuring. `UserBotLatencyObserver` watches frames go
past and emits three events:

    on_first_bot_speech_latency   client connect -> first bot audio
    on_latency_measured           user stopped speaking -> bot started speaking
    on_latency_breakdown          per-service TTFB, user turn, function calls

It needs `enable_metrics=True` in PipelineParams, which bot.py already sets.
So this file is a recorder and a reporting layer, not a stopwatch. Writing our
own frame timing would mean re-deriving what the library already computes, and
getting the VAD stop_secs adjustment subtly wrong.

WHAT THE NUMBERS MEAN, AND THE ONE THAT LIES
---------------------------------------------
`end-to-end` is what the caller experiences: they stop talking, and this long
later they hear a voice. Everything else is a component of it.

The summary leads with SPREAD (max divided by min) rather than the median, on
purpose. A median hides the tail, and on a phone call the tail is the whole
experience: nobody remembers the six replies that came back in half a second,
they remember the one that took two and a half and made them say "hello?".
Any stage over 3x is flagged.

WHERE IT DOES NOT WORK
----------------------
Text-mode evals. `on_latency_measured` fires on BotStartedSpeakingFrame, and
text mode skips TTS entirely, so no audio frame is ever produced and no
measurement happens. That is correct rather than broken: there is no
user-perceived latency in a test that never speaks. Measure on webrtc.

ORDERING CAVEAT
---------------
`on_latency_measured` and `on_latency_breakdown` are dispatched as separate
asyncio tasks from the same function, created in that order. In practice
measured arrives first. Nothing guarantees it, so this records whichever
arrives and emits the turn when the breakdown lands, treating a missing
end-to-end as absent rather than as zero.
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

DEFAULT_SAMPLES_PATH = Path(__file__).parent / "latency.jsonl"

# A stage varying by more than this between its best and worst sample gets
# flagged. 3x is not a standard, it is the point past which a caller starts
# noticing that replies do not arrive at a consistent pace.
SPREAD_FLAG = 3.0

# ...but only if its worst sample is actually big enough to feel. The first
# live run flagged ElevenLabs text aggregation at 3.7x, ranging from 0.020s to
# 0.073s. Both ends are inaudible; the ratio is real and the finding is
# useless. A flag column that fires on noise trains you to stop reading it,
# which costs you the one row that mattered (Gemini at 14.2x on the same run).
# Ratios need a magnitude gate before they mean anything.
SPREAD_FLAG_MIN_SECS = 0.25


@dataclass
class TurnLatency:
    """One measured turn."""

    turn: int
    kind: str  # "greeting" or "reply"
    end_to_end_secs: float | None
    user_turn_secs: float | None
    stages: dict[str, float] = field(default_factory=dict)
    function_calls: dict[str, float] = field(default_factory=dict)

    def as_line(self) -> str:
        """One log line, dense enough to scan a whole call at a glance."""
        parts = [f"turn {self.turn} ({self.kind})"]
        if self.end_to_end_secs is not None:
            parts.append(f"end-to-end {self.end_to_end_secs:.3f}s")
        if self.user_turn_secs is not None:
            # VAD silence detection plus STT finalisation plus any turn
            # analyzer wait. When end-to-end looks bad and every service TTFB
            # looks fine, this is usually where the time went.
            parts.append(f"user turn {self.user_turn_secs:.3f}s")
        parts += [f"{name} {secs:.3f}s" for name, secs in self.stages.items()]
        parts += [f"{name}() {secs:.3f}s" for name, secs in self.function_calls.items()]
        return "LATENCY  " + "  |  ".join(parts)


class LatencyRecorder:
    """Collects TurnLatency rows from a UserBotLatencyObserver.

    Attach it to an observer, run a conversation, then call `summary()`.
    """

    def __init__(
        self,
        model: str,
        transport: str,
        config: dict | None = None,
        samples_path: Path | None = None,
    ):
        """
        Args:
            config: Tuning knobs in effect for this call, e.g.
                {"stop_secs": 1.5}. Recorded alongside the samples and used to
                GROUP them in `compare`, which is the whole reason it exists:
                two runs of the same model with different settings must not be
                pooled, or the comparison silently averages away the thing you
                changed. Any knob you add here becomes comparable for free.
        """
        self.model = model
        self.transport = transport
        self.config = config or {}
        self.samples_path = samples_path or DEFAULT_SAMPLES_PATH
        self.turns: list[TurnLatency] = []
        self._pending_end_to_end: float | None = None
        self._pending_first_speech: float | None = None

    def attach(self, observer) -> None:
        """Register handlers on a pipecat UserBotLatencyObserver.

        Handlers receive the observer as their first argument; that is
        pipecat's convention for every event handler, see
        `BaseObject._run_handler`.
        """

        @observer.event_handler("on_first_bot_speech_latency")
        async def _on_first_speech(_observer, latency: float):
            self._pending_first_speech = latency

        @observer.event_handler("on_latency_measured")
        async def _on_measured(_observer, latency: float):
            self._pending_end_to_end = latency

        @observer.event_handler("on_latency_breakdown")
        async def _on_breakdown(_observer, breakdown):
            self._record(breakdown)

    def _record(self, breakdown) -> None:
        """Turn a LatencyBreakdown plus whatever stash we have into a row."""
        # The greeting has no preceding user speech, so `on_latency_measured`
        # never fires for it and the only meaningful number is how long the
        # caller waited after connecting. Labelling it rather than reporting a
        # blank keeps it out of the reply statistics, where it would drag the
        # numbers around for reasons that have nothing to do with the model.
        is_greeting = self._pending_end_to_end is None and self._pending_first_speech is not None
        end_to_end = self._pending_end_to_end or self._pending_first_speech

        row = TurnLatency(
            turn=len(self.turns),
            kind="greeting" if is_greeting else "reply",
            end_to_end_secs=end_to_end,
            user_turn_secs=getattr(breakdown, "user_turn_secs", None),
            stages={t.processor: t.duration_secs for t in breakdown.ttfb},
            function_calls={
                fc.function_name: fc.duration_secs for fc in breakdown.function_calls
            },
        )

        aggregation = getattr(breakdown, "text_aggregation", None)
        if aggregation is not None:
            # Sentence aggregation before TTS. Small, but it sits directly in
            # the path between the model finishing a sentence and the caller
            # hearing it, so it belongs in the breakdown rather than nowhere.
            row.stages[f"{aggregation.processor} aggregation"] = aggregation.duration_secs

        self.turns.append(row)
        self._pending_end_to_end = None
        self._pending_first_speech = None
        logger.info(row.as_line())

    # -- reporting ---------------------------------------------------------

    def _reply_rows(self) -> list[TurnLatency]:
        return [t for t in self.turns if t.kind == "reply"]

    def summary(self) -> str:
        """A per-stage table for this call. Empty string if nothing measured."""
        replies = self._reply_rows()
        if not replies:
            if self.turns:
                return (
                    "LATENCY: only the greeting was measured, so there is nothing "
                    "to summarise. Hold a few turns of conversation."
                )
            return (
                "LATENCY: nothing measured. Expected on a text-mode eval, which "
                "never produces bot audio. Measure on -t webrtc."
            )

        series: dict[str, list[float]] = {"end-to-end": []}
        for row in replies:
            if row.end_to_end_secs is not None:
                series["end-to-end"].append(row.end_to_end_secs)
            if row.user_turn_secs is not None:
                series.setdefault("user turn", []).append(row.user_turn_secs)
            for name, secs in row.stages.items():
                series.setdefault(name, []).append(secs)

        config = "  ".join(f"{k}={v}" for k, v in sorted(self.config.items()))
        return _format_table(
            f"LATENCY SUMMARY  {len(replies)} replies  model={self.model}  "
            f"transport={self.transport}" + (f"  {config}" if config else ""),
            series,
        )

    def write_samples(self) -> None:
        """Append this call's rows so runs can be compared later."""
        replies = self._reply_rows()
        if not replies:
            return
        record = {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "transport": self.transport,
            "config": self.config,
            "turns": [
                {
                    "turn": t.turn,
                    "end_to_end_secs": t.end_to_end_secs,
                    "user_turn_secs": t.user_turn_secs,
                    "stages": t.stages,
                    "function_calls": t.function_calls,
                }
                for t in replies
            ],
        }
        with self.samples_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            f"LATENCY: appended {len(replies)} replies to {self.samples_path.name}. "
            f"`uv run latency.py` compares every run recorded so far."
        )


def _format_table(title: str, series: dict[str, list[float]]) -> str:
    """Render stages as a fixed-width table, worst spread first.

    Sorted by spread rather than by name or by median, because the whole point
    of this table is finding the stage that is inconsistent, and a stage that
    is slow but steady is a much easier problem than one that is usually fast.
    """
    rows = []
    for name, values in series.items():
        if not values:
            continue
        lo, hi = min(values), max(values)
        spread = (hi / lo) if lo > 0 else float("inf")
        rows.append((name, len(values), lo, statistics.median(values), hi, spread))

    if not rows:
        return ""

    rows.sort(key=lambda r: r[5], reverse=True)
    width = max(len(r[0]) for r in rows)

    out = ["", "=" * max(len(title), width + 42), title, "=" * max(len(title), width + 42)]
    out.append(f"{'stage'.ljust(width)}   n    min    p50    max   spread")
    for name, n, lo, mid, hi, spread in rows:
        inconsistent = spread >= SPREAD_FLAG and hi >= SPREAD_FLAG_MIN_SECS
        flag = "  <-- inconsistent" if inconsistent else ""
        spread_text = "inf" if spread == float("inf") else f"{spread:.1f}x"
        out.append(
            f"{name.ljust(width)}  {n:>2}  {lo:5.3f}  {mid:5.3f}  {hi:5.3f}  "
            f"{spread_text:>6}{flag}"
        )
    out.append(
        f"\nSpread is max/min. Flagged at {SPREAD_FLAG:.0f}x or worse, but only when the "
        f"max is at least\n{SPREAD_FLAG_MIN_SECS:.2f}s, since a big ratio between two "
        "inaudible numbers means nothing. A stage\nthat is slow but steady is a far "
        "easier problem than one that is usually fast."
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# `uv run latency.py`: compare every run recorded so far, grouped by model.
# This is the Phase 3 decision tool. Record a few conversations on one model,
# change GEMINI_MODEL in .env, record a few more, then run this.
# --------------------------------------------------------------------------


def compare(samples_path: Path) -> str:
    if not samples_path.is_file():
        return (
            f"No samples at {samples_path}.\n"
            "Hold a conversation on `uv run bot.py -t webrtc` first; the bot "
            "appends a record per call."
        )

    # Grouped by model AND config, not model alone. Two runs of the same model
    # with different tuning are different experiments, and pooling them would
    # average away the exact effect you changed the setting to see.
    by_group: dict[str, list[dict]] = {}
    calls: dict[str, int] = {}
    for line in samples_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        config = record.get("config") or {}
        # Records written before config was captured say so rather than
        # silently joining whichever group they resemble.
        label = "  ".join(f"{k}={v}" for k, v in sorted(config.items())) or "config unrecorded"
        group = f"{record.get('model', 'unknown')}   {label}"
        by_group.setdefault(group, []).extend(record.get("turns", []))
        calls[group] = calls.get(group, 0) + 1

    if not by_group:
        return f"{samples_path} has no usable records."

    sections = []
    for model, turns in sorted(by_group.items()):
        series: dict[str, list[float]] = {"end-to-end": []}
        for turn in turns:
            if turn.get("end_to_end_secs") is not None:
                series["end-to-end"].append(turn["end_to_end_secs"])
            if turn.get("user_turn_secs") is not None:
                series.setdefault("user turn", []).append(turn["user_turn_secs"])
            for name, secs in (turn.get("stages") or {}).items():
                series.setdefault(name, []).append(secs)
        sections.append(
            _format_table(
                f"{model}   {len(turns)} replies across {calls[model]} call(s)", series
            )
        )

    if len(by_group) > 1:
        sections.append(
            "\nLook at the spread column before the median. A configuration that is "
            "300ms\nslower on p50 and half as variable is the better phone agent, "
            "because nobody\nremembers the six fast replies, they remember the one "
            "that made them say hello."
        )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recorded latency across runs.")
    parser.add_argument("--file", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument(
        "--clear", action="store_true", help="delete the samples file and start over"
    )
    args = parser.parse_args()

    if args.clear:
        if args.file.is_file():
            args.file.unlink()
            print(f"Deleted {args.file}")
        else:
            print(f"Nothing to delete at {args.file}")
        return

    print(compare(args.file))


if __name__ == "__main__":
    main()
