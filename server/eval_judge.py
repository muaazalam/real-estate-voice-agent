"""
eval_judge.py

Judge factory for the pipecat eval harness. Referenced from a scenario as:

    judge:
      eval:
        factory: eval_judge.gemini_judge

`run-eval.sh` puts server/ on PYTHONPATH so `importlib.import_module` can find
this module. Running `pipecat eval run` directly will fail to import it.

WHY THIS FILE EXISTS
--------------------
The obvious config, `service: openai` with `model: gemini-3.6-flash`, gets you
an authenticated judge that then fails on every turn with:

    judge said no: could not parse judge response: '{"verdict'

The verdict JSON is truncated after nine characters. The cause is a collision
between two things:

1. `EvalJudge.__init__` hardcodes `max_tokens=200`, and `from_config` does not
   expose it. Worse, `run_inference` applies that 200 AFTER
   `build_chat_completion_params`, so no setting can raise it. It is absolute.

2. Gemini 3.x models think, and thinking tokens are billed against that same
   200 token output budget. gemini-3.6-flash benchmarked at 2564ms per call
   against roughly 500ms for the lite models, which is the same thinking
   showing up as latency. It burns the budget reasoning and gets cut off
   mid-key before it can emit the answer.

The fix is not a bigger budget, it is less thinking. `reasoning_effort` maps to
Gemini's `thinking_level` through the OpenAI compatibility layer, and pipecat
forwards unknown keys via `Settings.extra`, which
`build_chat_completion_params` merges last. A judge deciding whether a sentence
satisfies a plain-English criterion does not need deep reasoning, so capping it
costs nothing and buys back the whole token budget.

Reasoning cannot be disabled entirely on Gemini 3 models, only lowered.

THE SECOND BUG, AND WHY response_format IS HERE
-----------------------------------------------
With reasoning capped the judge started returning proper JSON, but only most of
the time. One turn came back as prose and the harness reported:

    judge said no: (unstructured no)

That is `_parse_verdict`'s fallback, and the fallback is a raw substring scan:

    if "no" in lowered and "yes" not in lowered:  -> verdict NO

"no" is a substring of "not", "cannot", "know", "none" and "neighborhood". So a
prose reply explaining why the bot PASSED can be scored as a failure, silently,
with a reason string that tells you nothing. Any non-JSON reply is close to a
coin flip weighted toward NO.

`response_format={"type": "json_object"}` removes the fallback from the picture
entirely rather than trying to make prose parse reliably.

THE THIRD BUG, AND WHY THE JUDGE RETRIES
----------------------------------------
2026-07-29. With both of the above fixed, turns 0 through 5 passed and turn 6
failed with:

    judge said no: judge call failed: RateLimitError

That is not a verdict about the bot. It is the free tier quota, reported in the
same slot as a real judgement, which is exactly the failure shape entry 012 was
about: infrastructure trouble wearing the costume of a result. `EvalJudge`
catches every exception from `run_inference` and returns `verdict="no"`, so any
transient API problem reads as a bot regression.

Seven turns is seven judge calls in about 26 seconds, and Gemini free tier
limits are per project, not per key, so the bot's own traffic shares the
budget. Google no longer publishes the free tier RPM numbers in the docs; they
are per project at https://aistudio.google.com/rate-limit.

The OpenAI SDK does retry a 429 on its own, and its retry is structurally
incapable of helping here. `openai/_constants.py`:

    DEFAULT_MAX_RETRIES = 2
    INITIAL_RETRY_DELAY = 0.5
    MAX_RETRY_DELAY = 8.0

Two retries roughly 0.5s and 1s apart. A per-MINUTE quota needs up to 60
seconds to clear, so the SDK gives up after about a second and a half of
waiting, and worse, each of those retries is itself a request counted against
the same quota. The default is tuned for a brief server-side blip, not for a
rolling window.

So `_RetryingJudgeLLM` below turns the SDK's retries OFF and does its own, with
delays sized to the window that actually has to pass. It honors the server's
retry-after or Gemini's retryDelay when either is present, and falls back to
8s, 20s, 40s. If it still fails it raises `JudgeRateLimitedNotABotFailure`, so
the harness line reads `judge call failed: JudgeRateLimitedNotABotFailure`
rather than naming a generic error that looks like the bot's fault.

If evals start feeling slow, grep the run for "judge rate limited". Every wait
is logged with its duration.

THE FOURTH BUG: THE BUDGET WAS NEVER ACTUALLY FIXED
----------------------------------------------------
2026-07-29, straight after moving the judge to gemini-3.5-flash for quota
reasons. Turn 3 failed with:

    judge said no: could not parse judge response: 'Here is the JSON requested'

Six tokens, out of a 200 token budget. The other ~194 went to reasoning. This
is the truncation from the first section, back again, because
`reasoning_effort: low` means something different to gemini-3.5-flash than it
meant to gemini-3.6-flash. Capping reasoning was never a fix, it was a way of
fitting under a ceiling, and it only held for the one model it was tuned on.

Note also that `response_format={"type": "json_object"}` did not prevent a
prose preamble here, so whatever the compatibility layer does with it for this
model, it is not constrained decoding. Worth knowing before relying on it.

The ceiling turned out not to be a ceiling. The earlier note in this file said
200 "cannot be raised" and called it absolute, on the grounds that
`EvalJudge.__init__` hardcodes it, `from_config` does not expose it, and
`run_inference` applies it after `build_chat_completion_params` so no setting
survives. All true, and all about pipecat's CONFIG surface. But EvalJudge
reaches the model by calling `self._service.run_inference(max_tokens=...)`, and
`self._service` is this class. The number arrives as an ordinary keyword
argument to a method we already override. `_RetryingJudgeLLM.run_inference`
now raises it to JUDGE_MAX_TOKENS.

That makes truncation structurally impossible rather than tuned-around, and it
means `reasoning_effort` goes back to being a latency knob instead of load-
bearing correctness.

The general shape: "this value is not configurable" is a statement about the
configuration surface. If you are the object being called, the argument is
still just an argument.

IF THIS BREAKS
--------------
Knobs, in the order worth trying:

- If verdicts truncate again, raise JUDGE_MAX_TOKENS. It is a plain constant
  now. Do not reach for `reasoning_effort` first: that treats a symptom, and
  the last two judge models needed different values of it.
- If the call errors on an unexpected parameter, drop `response_format`. It is
  not doing much on Gemini anyway, see above, and it is the most likely
  parameter to be unsupported by the compatibility layer.
- `reasoning_effort: minimal` in the scenario's judge block if the judge is too
  slow. This is now a speed setting, not a correctness one.
- Drop `temperature` last. It is set to 0 so verdicts are stable run to run,
  which matters for a regression suite.

An auth error means run-eval.sh did not export the credentials. An import error
means PYTHONPATH did not include server/.
"""

import asyncio
import os
import random
import re
from typing import Any

from loguru import logger
from openai import RateLimitError
from pipecat.services.openai.llm import OpenAILLMService

# Gemini speaks OpenAI's protocol here, so pipecat's OpenAI client works
# unmodified against the free Gemini key this project already uses. No second
# provider, no credit card, and nothing competing for 8 GB of RAM the way a
# local 9B model did.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_JUDGE_MODEL = "gemini-3.6-flash"
DEFAULT_REASONING_EFFORT = "low"

# Output budget for a judge call, replacing the 200 that EvalJudge hardcodes.
#
# The original note in this file said that 200 "cannot be raised" and called it
# absolute. That was true of pipecat's CONFIG surface: `from_config` does not
# expose max_tokens, and `run_inference` applies the 200 after
# `build_chat_completion_params`, so no setting reaches it. It was not true of
# the call itself. EvalJudge passes the number as a keyword argument, and this
# class is the thing being called, so the value is ours to override. See the
# fourth section of the module docstring.
#
# 1500 rather than a snug fit. The output is a one-line JSON verdict; all of
# this headroom exists for thinking tokens, which vary with how hard the
# criterion is and are not worth predicting per model. The free tier limit that
# actually binds here is requests per day, not tokens, so unused budget costs
# nothing.
JUDGE_MAX_TOKENS = 1500

# Waits between judge retries, in seconds, used only when the server does not
# tell us how long to wait. Sized against a per-minute quota: the point is to
# let the rolling window age out, so these are tens of seconds, not the SDK's
# tenths. Worst case a genuinely exhausted quota costs about 70 seconds on one
# turn, and fail-fast means only one turn per run can pay it.
JUDGE_RETRY_DELAYS_SECS = (8.0, 20.0, 40.0)

# Cap on anything the server asks for, so a bad header cannot hang a run.
JUDGE_RETRY_MAX_WAIT_SECS = 90.0

# Ceiling on the SUM of the waits. Without this, honouring the server's own
# retryDelay three times ran a single turn for 118 seconds on 2026-07-29 and
# still failed. Past about this point you are not waiting out a burst, you are
# out of quota, and the run should say so instead of stalling.
JUDGE_RETRY_TOTAL_BUDGET_SECS = 75.0

# Substrings that identify a DAILY quota in a 429. Matched against the error
# after `_error_blob` strips spaces, hyphens and underscores, so all of
# "PerDay", "per_day", "per day" and "Per-Day" collapse onto "perday" and one
# marker covers every spelling the API has been seen to use. Real examples:
# quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier", quotaMetric
# "generate_requests_per_day", and the prose "Quota exceeded: requests per day".
#
# Deliberately not matching the abbreviation "rpd". It is three characters and
# would collide with any id that happens to contain them, and a false positive
# here reports a transient burst as an all-day outage.
DAILY_QUOTA_MARKERS = ("perday", "dailylimit", "dailyquota")


class JudgeRateLimitedNotABotFailure(RuntimeError):
    """Raised when the judge is still rate limited after every retry.

    The name is the whole point. `EvalJudge._call_judge` catches everything and
    reports `judge call failed: {e.__class__.__name__}` in a slot that otherwise
    holds a verdict about the bot, so the class name is the only thing standing
    between you and a run that looks like a behavioural regression when the real
    problem is quota. Do not rename this to something tidy.
    """


class JudgeDailyQuotaExhaustedNotABotFailure(RuntimeError):
    """Raised when the 429 names a per-day quota, which waiting cannot fix.

    Same naming rule as the class above: this string is what lands in the
    harness summary where a verdict about the bot normally goes.
    """


def _error_blob(error: RateLimitError) -> str:
    """Everything the 429 carried, lowercased and despaced, for substring checks.

    Quota identifiers arrive in different places depending on whether the
    request went through the native endpoint or the OpenAI-compatible one, so
    flatten the whole error rather than walking a path that moves.
    """
    parts = [
        str(getattr(error, "body", "") or ""),
        str(getattr(error, "message", "") or ""),
        str(error),
    ]
    flat = " ".join(parts).lower()
    # Strip every separator the same field has been seen written with. Missing
    # the underscore here made "generate_requests_per_day" read as retryable,
    # which is the exact case this function exists to catch.
    for separator in (" ", "-", "_"):
        flat = flat.replace(separator, "")
    return flat


def _is_daily_quota(error: RateLimitError) -> bool:
    """True when the 429 is a requests-per-DAY limit.

    Worth separating from a per-minute limit because the correct response is
    the opposite. A per-minute quota is a burst you wait out. A per-day quota
    resets at midnight Pacific, so retrying does nothing except spend more of
    the quota you just ran out of, which is what happened on 2026-07-29: the
    retry made the next run fail one turn EARLIER than the run before it.
    """
    blob = _error_blob(error)
    return any(marker in blob for marker in DAILY_QUOTA_MARKERS)


def _server_requested_wait(error: RateLimitError) -> float | None:
    """Pull a wait duration out of a 429, if the server supplied one.

    Two places to look. The standard `retry-after` header, and Gemini's own
    RetryInfo in the error body, which is a duration string like "27s".
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after")
    if raw is not None:
        try:
            return min(float(raw), JUDGE_RETRY_MAX_WAIT_SECS)
        except (TypeError, ValueError):
            pass

    # Gemini puts RetryInfo in the error payload rather than the header. The
    # shape moves around between the native and the OpenAI-compatible endpoint,
    # so match the duration wherever it sits instead of walking a fixed path.
    #
    # Both quote styles are matched deliberately. `error.body` is usually an
    # already-parsed dict, and `str()` on a dict is Python's repr, which uses
    # SINGLE quotes. A JSON-only pattern looks right, passes review, and never
    # matches a single real payload. That was the first version of this line.
    body = getattr(error, "body", None)
    if body is not None:
        match = re.search(r"""['"]retryDelay['"]\s*:\s*['"]([0-9.]+)s['"]""", str(body))
        if match:
            try:
                return min(float(match.group(1)), JUDGE_RETRY_MAX_WAIT_SECS)
            except ValueError:
                pass
    return None


class _RetryingJudgeLLM(OpenAILLMService):
    """OpenAILLMService that waits out a 429 instead of failing the turn.

    See "THE THIRD BUG" in this module's docstring for why the SDK's own retry
    does not cover this case.
    """

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Build the client with the SDK's retries disabled.

        `max_retries=0` looks backwards in a class about retrying. It is not.
        The SDK's two retries land 0.5s and 1s after the 429, far too early for
        a per-minute window to have moved, and each one spends another request
        against the very quota we are over. Turning them off makes the retry
        budget ours to spend on waits long enough to work.
        """
        client = super().create_client(api_key=api_key, base_url=base_url, **kwargs)
        return client.with_options(max_retries=0)

    async def run_inference(self, *args, **kwargs):
        """Run one judge call, retrying through rate limits.

        Also raises the output budget. EvalJudge passes max_tokens=200 as a
        keyword argument, and a thinking model spends most of that reasoning and
        gets cut off mid-verdict. Intercepting it here is the only place the
        number is reachable, since pipecat's config surface never exposes it.
        See JUDGE_MAX_TOKENS.

        Only RateLimitError is retried. Everything else propagates on the first
        failure, because a bad request or an auth error will not improve with
        waiting and a fast, honest traceback beats a slow one.
        """
        # Raise, never lower. If a caller ever asks for more than this, they
        # know something about their criterion that this constant does not.
        if kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = max(kwargs["max_tokens"], JUDGE_MAX_TOKENS)
        else:
            kwargs["max_tokens"] = JUDGE_MAX_TOKENS

        last_error: RateLimitError | None = None
        spent_secs = 0.0

        for attempt in range(len(JUDGE_RETRY_DELAYS_SECS) + 1):
            try:
                return await super().run_inference(*args, **kwargs)
            except RateLimitError as e:
                last_error = e

                # Log the raw payload once. Without this the only evidence of
                # WHICH quota was hit is the wall clock, and reconstructing a
                # quota type from run durations is not a thing anyone should
                # have to do twice.
                logger.warning(
                    "judge 429 on attempt {}: {}",
                    attempt + 1,
                    str(getattr(e, "body", None) or e)[:600],
                )

                if _is_daily_quota(e):
                    raise JudgeDailyQuotaExhaustedNotABotFailure(
                        "the judge's DAILY request quota is gone, so this turn "
                        "was never graded and the result says nothing about the "
                        "bot. Waiting will not help: per-day quota resets at "
                        "midnight Pacific. Quotas are per project per model, so "
                        "the fastest unblock is a different judge model, which "
                        "has its own daily bucket. Run `uv run "
                        "check_judge_models.py` to see which are alive. "
                        f"Raw error: {e}"
                    ) from e

                if attempt == len(JUDGE_RETRY_DELAYS_SECS):
                    break

                delay = _server_requested_wait(e) or JUDGE_RETRY_DELAYS_SECS[attempt]
                # Jitter so that a future parallel runner does not resynchronise
                # every worker onto the same retry instant.
                delay += random.uniform(0, 1.0)

                if spent_secs + delay > JUDGE_RETRY_TOTAL_BUDGET_SECS:
                    logger.warning(
                        "judge retry budget of {:.0f}s is spent, giving up rather "
                        "than waiting another {:.1f}s",
                        JUDGE_RETRY_TOTAL_BUDGET_SECS,
                        delay,
                    )
                    break

                logger.warning(
                    "judge rate limited (attempt {}/{}), waiting {:.1f}s before "
                    "retrying, {:.1f}s of {:.0f}s budget spent",
                    attempt + 1,
                    len(JUDGE_RETRY_DELAYS_SECS) + 1,
                    delay,
                    spent_secs,
                    JUDGE_RETRY_TOTAL_BUDGET_SECS,
                )
                await asyncio.sleep(delay)
                spent_secs += delay

        raise JudgeRateLimitedNotABotFailure(
            f"the judge was rate limited and stayed that way after "
            f"{spent_secs:.0f}s of waiting, so this turn was never actually "
            f"graded and the result says nothing about the bot. Free tier quota "
            f"is per project, not per key, so the bot's own calls share it. "
            f"Check https://aistudio.google.com/rate-limit, and run `uv run "
            f"check_judge_models.py` to find a judge model with headroom. "
            f"Last error: {last_error}"
        ) from last_error


def gemini_judge(config: dict[str, Any]) -> OpenAILLMService:
    """
    Build the judge LLM service. Called by pipecat with the scenario's
    `judge.eval` block, so `model` and `reasoning_effort` are both settable
    from YAML without touching this file.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set, so the eval judge cannot authenticate. "
            "Run evals through ./run-eval.sh, which exports it from .env. "
            "The pipecat eval CLI does not read .env by itself."
        )

    return _RetryingJudgeLLM(
        api_key=api_key,
        base_url=GEMINI_OPENAI_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=config.get("model") or DEFAULT_JUDGE_MODEL,
            # A judge that changes its mind between runs turns a regression
            # suite into a coin flip. Pin it.
            temperature=0,
            extra={
                "reasoning_effort": config.get(
                    "reasoning_effort", DEFAULT_REASONING_EFFORT
                ),
                # Force valid JSON. Without this the model complies with the
                # "respond only with JSON" instruction most of the time but not
                # always, and the harness's fallback for non-JSON is a raw
                # substring scan: any reply containing "no" and not "yes"
                # becomes a NO. "not", "cannot", "know" and "neighborhood" all
                # contain "no", so a prose reply explaining why the bot PASSED
                # can be scored as a failure. Observed on 2026-07-28 as
                # `judge said no: (unstructured no)`.
                "response_format": {"type": "json_object"},
            },
        ),
    )
