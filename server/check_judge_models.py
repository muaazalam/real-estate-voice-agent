"""
check_judge_models.py

Which models can actually judge, and if one cannot, exactly why.

    uv run check_judge_models.py
    uv run check_judge_models.py gemini-3.6-flash gemini-3.5-flash
    uv run check_judge_models.py --list

WHY THIS EXISTS
---------------
Two problems this project keeps paying for.

First, from HANDOFF.md: Gemini's ListModels advertises models that
generateContent then rejects with 404. A model string is only real once a
completion call has succeeded with it, so this probes with a real one-token
completion rather than trusting a catalogue. `--list` will show you the
catalogue, with that caveat attached.

Second, and the reason this file was written on 2026-07-29: the eval judge
started failing on rate limits, and the only evidence of WHICH quota was
exhausted was the wall clock. A per-minute quota and a per-day quota look
identical in the harness output, and they call for opposite responses. A
per-minute limit is a burst you wait out. A per-day limit resets at midnight
Pacific, and retrying it just spends more of what you ran out of.

Quotas are per project PER MODEL, so a judge on a different model has its own
untouched daily bucket. That makes "which model still has headroom" the
question that actually unblocks a run, and this script answers it in one call
per model.

READING THE OUTPUT
------------------
    OK        completion succeeded, usable as a judge right now
    RATE      alive but out of quota, and the quota id says which kind
    404       model string is not real for this account, do not use it
    AUTH      the key is wrong or lacks access
    ERROR     anything else, printed in full

The bot's own model is marked BOT. Do not pick it as the judge: the point of a
separate judge model is that it does not share the model-under-test's blind
spots. That decision is recorded in HANDOFF.md, and this script only marks it,
it does not enforce it.

Each probe is ONE request with retries disabled, because a diagnostic for a
quota problem must not itself spend meaningful quota.
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import (
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

# Same endpoint the judge uses, so a pass here means a pass there. Probing the
# native Gemini endpoint instead would test a different code path and could
# report a model as healthy that the OpenAI compatibility layer rejects.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.6-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

QUOTA_HINTS = [
    ("perday", "PER DAY. Resets midnight Pacific. Waiting today will not help."),
    ("perminute", "per minute. A burst limit, clears within about 60s."),
    ("tokens", "token based rather than request based."),
]


def _load_key() -> tuple[str, str | None]:
    """Read the key and the bot's model from server/.env.

    An explicit path, not bare load_dotenv(), per ENGINEERING-LOG.md entry 002:
    bare load_dotenv resolves from the calling file's directory rather than cwd,
    which is a quiet way to load the wrong config.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            f"GOOGLE_API_KEY is not set. Looked in {env_path}\n"
            "This script reads the same .env the bot does."
        )
    return key, os.environ.get("GEMINI_MODEL")


def _quota_kind(error: Exception) -> str:
    """Describe which quota a 429 refers to, in plain words."""
    blob = str(getattr(error, "body", "") or "") + str(error)
    flat = blob.lower().replace(" ", "").replace("_", "").replace("-", "")
    for marker, explanation in QUOTA_HINTS:
        if marker in flat:
            return explanation
    return "quota type not stated in the error, see the raw body below"


def probe(client: OpenAI, model: str) -> tuple[str, str]:
    """One real completion against one model. Returns (status, detail)."""
    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=200,
            temperature=0,
        )
    except NotFoundError:
        return "404", "model string rejected by the completions endpoint"
    except AuthenticationError as e:
        return "AUTH", str(e)[:200]
    except RateLimitError as e:
        detail = _quota_kind(e)
        raw = str(getattr(e, "body", None) or e)[:400]
        return "RATE", f"{detail}\n        raw: {raw}"
    except APIStatusError as e:
        return "ERROR", f"HTTP {e.status_code}: {str(e)[:200]}"
    except Exception as e:
        return "ERROR", f"{e.__class__.__name__}: {str(e)[:200]}"

    elapsed = time.monotonic() - started
    text = (response.choices[0].message.content or "").strip()
    if not text:
        # A thinking model can spend the whole output budget reasoning and
        # return nothing. That is survivable for a chat turn and fatal for a
        # judge, whose entire output is a short JSON verdict. See entry 011.
        return "OK", f"{elapsed:.2f}s but returned EMPTY text, risky as a judge"
    return "OK", f"{elapsed:.2f}s, said {text[:40]!r}"


def list_catalogue(client: OpenAI) -> None:
    """Print what the API advertises, with the standing warning attached."""
    print("Advertised by the models endpoint:\n")
    try:
        for model in sorted(m.id for m in client.models.list()):
            print(f"  {model}")
    except Exception as e:
        print(f"  could not list: {e.__class__.__name__}: {e}")
    print(
        "\nThis is a catalogue, not a guarantee. Models listed here have been\n"
        "observed returning 404 from generateContent. Confirm with a probe\n"
        "before putting one in a scenario."
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    key, bot_model = _load_key()
    client = OpenAI(
        api_key=key,
        base_url=GEMINI_OPENAI_BASE_URL,
        # No SDK retries. A 429 is the RESULT here, not an obstacle, and a
        # diagnostic that silently triples its own request count while
        # investigating a quota problem is worse than useless.
        max_retries=0,
    )

    if "--list" in sys.argv:
        list_catalogue(client)
        return

    candidates = args or DEFAULT_CANDIDATES
    print(f"Probing {len(candidates)} model(s) with one real completion each.")
    print(f"Bot model from .env: {bot_model or 'unset'}\n")

    usable = []
    for model in candidates:
        marker = "  BOT" if model == bot_model else "     "
        status, detail = probe(client, model)
        print(f"{marker}  {status:<5}  {model:<26}  {detail}")
        if status == "OK" and "EMPTY" not in detail and model != bot_model:
            usable.append(model)

    print()
    if usable:
        print("Usable as a judge right now:")
        for model in usable:
            print(f"  {model}")
        print(
            f"\nTo switch, set `model:` under `judge.eval` in\n"
            f"  evals/phase2_acceptance.yaml\n"
            f"to one of the above. Nothing in eval_judge.py needs to change;\n"
            f"the factory already reads the model from the scenario."
        )
    else:
        print(
            "Nothing here is usable as a judge.\n"
            "If everything says RATE with a per-day quota, the project is out\n"
            "for the day and resets at midnight Pacific. Your live limits are\n"
            "at https://aistudio.google.com/rate-limit"
        )


if __name__ == "__main__":
    main()
