"""
focus_eval.py

Replay the whole conversation, grade only the turns you name.

    uv run focus_eval.py 5 6            grade turns 5 and 6
    uv run focus_eval.py 3-6            a range
    uv run focus_eval.py 5 6 && ./run-eval.sh evals/_focus.yaml

WHY THIS EXISTS
---------------
The judge's free tier limit is requests per DAY, confirmed on 2026-07-29 by
check_judge_models.py: `limit: 20`, metric
`generativelanguage.googleapis.com/generate_content_free_tier_requests`.

One full pass of phase2_acceptance.yaml is seven judge calls, one per turn. So
a 20/day model affords two complete runs, and every run after that reports
failures that are really an empty quota. When you are iterating on turn 6, five
of those seven calls are spent re-confirming turns you already know pass, which
is most of a day's budget spent on regression you were not asking about.

The saving comes from dropping `eval:`, NOT from dropping `expect:`.

    expect:
      - event: response          <- kept. the harness WAITS here. free.
        eval: >-                 <- removed. this is the judge call.
          the bot does not ...

`_evaluate_aggregate` in pipecat/evals/harness.py ends with `return ("pass",
"")`, so an expectation carrying an `event:` and no `eval:` passes the moment
the first response arrives and never constructs a judge call. That is the
pacing primitive this script needs, and it is free.

THE VERSION OF THIS THAT WAS WRONG, 2026-07-29
-----------------------------------------------
The first cut deleted the whole `expect:` block from ungraded turns, on the
reading that `expect:` is optional (scenario.py: "omit it for a turn that only
sends input or only waits") and that a turn would still drive the conversation.
It does still send its utterance. It does not wait for the answer.

With nothing to wait on, all five ungraded turns fired inside a few
milliseconds, each one interrupting the bot mid-reply, so no assistant message
was ever committed to the context. By turn 5 the bot had almost no history and
opened with its greeting, four turns late. Turn 6 then failed to recall a
budget it had never been told, which is what sent me looking.

The tell was in the bot log, and it is the fingerprint HANDOFF.md already
names: two consecutive `user` messages with no `model` between them.

    user   "Around 400k, three bedrooms, somewhere near Cedar Park."
    user   "What do you have available right now?"

Worse than the broken run was that turn 5 PASSED on it. Its criterion asks only
that the bot not answer an off-topic question and steer back to the home
search, and a greeting does technically satisfy that. A criterion phrased
entirely as an absence can be satisfied by a bot that has lost the thread
completely.

So: never strip the wait. Strip the grading.

Grading turns 5 and 6 costs 2 judge calls instead of 7, on a conversation that
is byte-identical to the full run's.

WHY GENERATE IT INSTEAD OF KEEPING A SECOND FILE
------------------------------------------------
Two scenario files holding the same conversation drift, and a stale copy of a
test is worse than no copy, because it passes. There is one source of truth,
this reads it, and the output is gitignored and disposable. If you edit a
criterion, regenerate rather than hand-editing the generated file.

WHAT IS LOST
------------
Comments. The output is machine-dumped YAML, so the reasoning in the source
scenario does not survive into it. That is fine for a throwaway, and it is the
reason the generated file carries a header pointing back at the original.
Never commit it, and never treat a focused pass as a green suite: a full run
is the only thing that proves the whole scenario.
"""

import argparse
import sys
from pathlib import Path

import yaml

DEFAULT_SOURCE = Path("evals/phase2_acceptance.yaml")
DEFAULT_OUTPUT = Path("evals/_focus.yaml")


def parse_turn_spec(tokens: list[str]) -> set[int]:
    """Turn ["3-5", "0"] into {0, 3, 4, 5}. Ranges are inclusive."""
    wanted: set[int] = set()
    for token in tokens:
        if "-" in token:
            start, _, end = token.partition("-")
            try:
                lo, hi = int(start), int(end)
            except ValueError:
                sys.exit(f"Bad range {token!r}. Expected something like 3-6.")
            if lo > hi:
                sys.exit(f"Bad range {token!r}. Start is after the end.")
            wanted.update(range(lo, hi + 1))
        else:
            try:
                wanted.add(int(token))
            except ValueError:
                sys.exit(f"Bad turn {token!r}. Expected a number or a range.")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a scenario that replays every turn but grades only some.",
    )
    parser.add_argument(
        "turns",
        nargs="+",
        help="Turn numbers to grade. Accepts ranges: 0 3 5-6",
    )
    parser.add_argument("--from", dest="source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", dest="output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.source.is_file():
        sys.exit(f"No scenario at {args.source}. Run this from server/.")

    try:
        scenario = yaml.safe_load(args.source.read_text())
    except yaml.YAMLError as e:
        # safe_load does not know pipecat's !include tag. If the scenario grows
        # one, this is where you will find out, rather than in a confusing
        # failure three steps later.
        sys.exit(f"Could not parse {args.source}: {e}")

    turns = scenario.get("turns") or []
    if not turns:
        sys.exit(f"{args.source} has no turns.")

    wanted = parse_turn_spec(args.turns)
    out_of_range = sorted(n for n in wanted if n >= len(turns))
    if out_of_range:
        sys.exit(
            f"Turn(s) {out_of_range} do not exist. {args.source.name} has "
            f"{len(turns)}, numbered 0 to {len(turns) - 1}."
        )

    # Only an expectation carrying `eval:` costs a judge call. Counting whole
    # expectations would overstate the saving on any scenario that uses a bare
    # `event:` or a `text_contains:`, both of which are graded locally.
    def judge_calls(turn: dict) -> int:
        return sum(1 for e in (turn.get("expect") or []) if e.get("eval") is not None)

    full_cost = sum(judge_calls(t) for t in turns)
    kept = 0

    for index, turn in enumerate(turns):
        if index in wanted:
            kept += judge_calls(turn)
            continue

        expectations = turn.get("expect") or []
        paced = []
        for expectation in expectations:
            # An `absent: true` expectation asserts that nothing arrives, so it
            # deliberately holds the turn open for its whole window. As a pacing
            # step that is pure waiting, and it is not what we are here to
            # check, so drop it.
            if expectation.get("absent"):
                continue
            # Strip the graded parts, keep the wait. `event:` and `within_ms:`
            # stay so the harness still blocks until the bot answers, which is
            # the entire reason the conversation stays coherent.
            for graded_key in ("eval", "text_contains"):
                expectation.pop(graded_key, None)
            paced.append(expectation)

        if expectations and not paced:
            # Every expectation on this turn was `absent:`, so there is nothing
            # left to wait on and the next turn would interrupt this one.
            paced = [{"event": "response"}]

        if paced:
            turn["expect"] = paced
        else:
            turn.pop("expect", None)

    # The bug this script shipped with, turned into an assertion.
    #
    # The invariant is not "I did not remove a wait", it is "every turn that has
    # another turn behind it can be waited on". Checking the weaker version
    # would pass a source scenario that was already missing a wait, and produce
    # exactly the silent breakage this guard exists to stop. A turn with nothing
    # to wait on lets the next turn fire immediately and interrupt the bot
    # mid-reply, and the graded turn then runs against a conversation that never
    # happened. The last turn is exempt: nothing follows it to do the
    # interrupting.
    original_turns = yaml.safe_load(args.source.read_text()).get("turns") or []
    for index, turn in enumerate(turns[:-1]):
        if turn.get("expect"):
            continue
        had_one = bool((original_turns[index] or {}).get("expect"))
        blame = (
            "this script stripped it, which is a bug in this script"
            if had_one
            else f"{args.source.name} has no `expect:` on that turn either"
        )
        sys.exit(
            f"Refusing to write: turn {index} has nothing to wait on, and {blame}.\n"
            f"Every turn before the last needs an `event:` expectation, or the "
            f"harness sends the next turn immediately and interrupts the bot "
            f"before it can answer. The graded turns then see a conversation "
            f"that never took place, and can pass on it."
        )

    if kept == 0:
        sys.exit(
            "That selection grades nothing, so the run would prove nothing and "
            "still start a bot. Pick a turn that has an `expect:` block."
        )

    scenario["name"] = f"{scenario.get('name', 'scenario')}_focus"

    header = (
        "# GENERATED by focus_eval.py. Do not edit, do not commit.\n"
        f"# Source: {args.source}\n"
        f"# Grading turns: {sorted(wanted)}.\n"
        "#\n"
        "# Every other turn keeps its `event:` expectation and loses its `eval:`.\n"
        "# It still runs and the harness still WAITS for the bot's reply, which is\n"
        "# what keeps the conversation coherent, it just is not graded and so\n"
        "# costs no judge call.\n"
        "#\n"
        "# A focused pass is not a green suite. Re-run the full scenario before\n"
        "# you believe a phase is done.\n"
    )
    args.output.write_text(header + yaml.safe_dump(scenario, sort_keys=False))

    saved = full_cost - kept
    print(f"Wrote {args.output}")
    print(f"  grading turns {sorted(wanted)}, replaying the other {len(turns) - len(wanted)}")
    print(f"  judge calls: {kept} instead of {full_cost}, saving {saved} per run")
    print(f"\n  ./run-eval.sh {args.output}")


if __name__ == "__main__":
    main()
