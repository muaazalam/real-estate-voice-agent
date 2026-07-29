"""
test_guard.py

Frame-sequence tests for EmptyResponseGuard.

    uv run test_guard.py

WHY THIS FILE EXISTS. The guard has been wrong three times (log 019, log 020),
and every single time it looked correct on the page. Attempt 2's bug was
"cancel the pending timer when text arrives", which cancelled a timer that had
not been created yet, and no amount of rereading found it. A table of frame
sequences found it on the first run. This is the third entry in the engineering
log where a small table test caught something reading did not, so the table now
lives in the repo instead of in a shell history.

It costs nothing to run. No API keys are exercised, no audio, no judge, no
quota. It uses pipecat's own `run_test` harness, so the guard runs inside a
real pipeline with the real FrameProcessor machinery rather than a mock of it.

Importing bot.py runs validate_config(), so server/.env must be present and
complete. That is deliberate: a test that quietly runs against a half-configured
module is worse than one that refuses to start.

READING A FAILURE. Each row prints the sequence it sent and what came back.
"expected stall, got none" means the guard stayed silent through dead air, which
is the 2026-07-29 bug. "expected none, got empty" means it interrupted a healthy
turn, which is the 2026-07-28 bug. They are opposite failures and the guard has
shipped both, so the table deliberately contains cases for each.

WHY THERE ARE SLEEPS IN THE MIDDLE OF SEQUENCES. Pipecat processes SystemFrames
immediately, ahead of the ordered queue that ControlFrames and DataFrames wait
in. Of the frames used here:

    UserStoppedSpeakingFrame   SystemFrame    jumps the queue
    InterruptionFrame          SystemFrame    jumps the queue
    FunctionCallsStartedFrame  SystemFrame    jumps the queue
    LLMFullResponseStart/End   ControlFrame   queued
    LLMTextFrame               DataFrame      queued

So a sequence sent with no delay does NOT reach the processor in the order it
was written. `new_utterance_rearms` originally sent two utterances back to back
and both UserStoppedSpeakingFrames arrived before any of the first turn's LLM
frames, which is an interleaving that cannot occur in a live call, because a
real second utterance is separated from the first by someone actually speaking.
It failed for a reason that could never happen on a phone line.

Where a system frame is meant to arrive AFTER preceding control or data frames,
there is a short sleep in front of it to let the queue drain. Those sleeps are
load-bearing. Removing one does not make the test faster, it makes it test a
different thing.
"""

import asyncio
import sys

from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from bot import EmptyResponseGuard

# Short deadlines so the whole table runs in a few seconds. The RATIO is what
# the tests depend on, never the absolute values: silence must be shorter than
# stall, and every sleep below is chosen relative to these two. Production
# values are 2.0 and STALL_FILLER_SECS (6.0) in bot.py.
SILENCE_SECS = 0.3
STALL_SECS = 0.7

EMPTY_FILLER = "EMPTY_FILLER"
STALL_FILLER = "STALL_FILLER"


def user_stopped():
    """Import lazily so an import error in bot.py surfaces before this runs."""
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    return UserStoppedSpeakingFrame()


def fcs():
    """The guard only type-checks this frame, so the call list can be empty."""
    return FunctionCallsStartedFrame(function_calls=[])


# Each case is (name, frames, expected, why).
#
# `expected` is None, "empty" or "stall", naming which filler should be spoken.
# `why` is printed on failure and is the actual point of the row: a red line
# with no explanation is just as unhelpful in a test as it is in a log.
CASES = [
    (
        "normal_reply",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Are you looking to buy or rent?"),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "A healthy turn must be silent. Attempt 2 fired here on every reply.",
    ),
    (
        "normal_reply_streamed",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Are you looking"),
            LLMTextFrame(" to buy or rent?"),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "Streaming arrives as many small deltas; one is enough to prove life.",
    ),
    (
        "whitespace_is_not_an_answer",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("   "),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        "empty",
        "Whitespace-only deltas are common mid-stream and are not a reply.",
    ),
    (
        "zero_token_window",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        "empty",
        "THE ORIGINAL BUG, 2026-07-28: completion tokens 0, line went silent.",
    ),
    (
        "function_call_only",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            SleepFrame(sleep=0.05),
            fcs(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "A textless window is normal once tools exist. Log 019, false fire 1.",
    ),
    (
        "function_call_then_text",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            SleepFrame(sleep=0.05),
            fcs(),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Got it, noted."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "The speak-after-tool path: two windows, one utterance, no filler.",
    ),
    (
        "empty_window_then_function_call",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.15),
            LLMFullResponseStartFrame(),
            fcs(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "ATTEMPT 1's BUG: the real answer lands one window after an empty one.",
    ),
    (
        "empty_window_then_text",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.15),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Sure, one bedroom."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "Same as above but the late answer is text rather than a tool call.",
    ),
    (
        "interruption_after_empty_window",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "Barge-in makes an empty response CORRECT. Filling would talk over them. "
        "The sleep matters: it makes the interruption cancel an ARMED timer "
        "rather than arriving before the window ever closed.",
    ),
    (
        "interruption_before_any_window",
        lambda: [
            user_stopped(),
            InterruptionFrame(),
            SleepFrame(sleep=0.9),
        ],
        None,
        "The stall timer must also respect barge-in, not just the empty path.",
    ),
    (
        "hung_window_never_closes",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            SleepFrame(sleep=0.9),
        ],
        "stall",
        "TODAY'S BUG, 2026-07-29: request never returned, attempt 2 stayed mute.",
    ),
    (
        "no_llm_activity_at_all",
        lambda: [
            user_stopped(),
            SleepFrame(sleep=0.9),
        ],
        "stall",
        "The degenerate case: nothing downstream ever came back at all.",
    ),
    (
        "slow_but_alive",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            SleepFrame(sleep=0.5),
            LLMTextFrame("Sorry, busy morning."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        None,
        "A slow turn that still works must not be interrupted. Guards the "
        "STALL_FILLER_SECS choice: text at 0.5 beats a 0.7 deadline.",
    ),
    (
        "late_empty_window_cannot_extend_the_stall",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            SleepFrame(sleep=0.5),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        "stall",
        "Re-arming must only move a deadline EARLIER. Closing empty at 0.5 "
        "would push a duration-based timer out to 0.8, past the 0.7 stall.",
    ),
    (
        "one_filler_per_utterance",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        "empty",
        "Two dead windows, one utterance, ONE apology. Eight in a row is the "
        "failure that started log 019.",
    ),
    (
        "new_utterance_rearms",
        lambda: [
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Certainly."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.15),
            user_stopped(),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.5),
        ],
        "empty",
        "A good turn must not immunise the next one against the guard. The "
        "0.15s sleep is the whole point: without it the second (system) "
        "UserStoppedSpeakingFrame overtakes the first turn's queued text.",
    ),
    (
        "teardown_cancels_a_pending_stall",
        lambda: [
            user_stopped(),
            SleepFrame(sleep=0.2),
        ],
        None,
        "run_test sends EndFrame here. A filler must never fire into a "
        "pipeline that is already closing.",
    ),
]


def classify(frames):
    """Return the list of fillers spoken, as 'empty' / 'stall' labels."""
    spoken = []
    for frame in frames:
        if isinstance(frame, TTSSpeakFrame):
            if frame.text == EMPTY_FILLER:
                spoken.append("empty")
            elif frame.text == STALL_FILLER:
                spoken.append("stall")
            else:
                spoken.append(f"unknown({frame.text!r})")
    return spoken


async def run_case(name, build_frames, expected, why):
    guard = EmptyResponseGuard(
        filler=EMPTY_FILLER,
        stall_filler=STALL_FILLER,
        silence_secs=SILENCE_SECS,
        stall_secs=STALL_SECS,
    )
    down, _up = await run_test(guard, frames_to_send=build_frames())
    spoken = classify(down)

    want = [] if expected is None else [expected]
    ok = spoken == want

    got = "none" if not spoken else ", ".join(spoken)
    exp = "none" if expected is None else expected
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {name:<42}  expected {exp:<5}  got {got}")
    if not ok:
        print(f"        why this case exists: {why}")
    return ok


async def main():
    print(f"EmptyResponseGuard: {len(CASES)} frame sequences")
    print(f"silence_secs={SILENCE_SECS}  stall_secs={STALL_SECS}\n")

    results = []
    for name, build_frames, expected, why in CASES:
        try:
            results.append(await run_case(name, build_frames, expected, why))
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {name:<42}  {type(exc).__name__}: {exc}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
