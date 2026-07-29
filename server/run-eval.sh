#!/usr/bin/env bash
#
# run-eval.sh - start the bot on the eval transport, run a scenario, clean up.
#
#   ./run-eval.sh                              runs evals/phase2_acceptance.yaml
#   ./run-eval.sh evals/starter_text.yaml      runs a specific scenario
#
# Exists for three reasons, all of them things that went wrong on 2026-07-28:
#
# 1. The bot and the harness are two processes that must overlap. Running them
#    sequentially in one terminal means the harness dials a socket nothing is
#    listening on and reports "failed to connect", which looks like a bug in
#    the transport and is not.
# 2. `pipecat eval run` does NOT read .env, so the judge's credentials have to
#    be exported first. Doing that by hand every run is how secrets end up in
#    shell history.
# 3. The judge is Gemini via its OpenAI-compatible endpoint. pipecat's OpenAI
#    client passes api_key=None and base_url=None to the OpenAI SDK, which then
#    falls back to these two environment variables. Mapping GOOGLE_API_KEY onto
#    OPENAI_API_KEY here is the whole trick.

set -euo pipefail
cd "$(dirname "$0")"

SCENARIO="${1:-evals/phase2_acceptance.yaml}"

if [ ! -f .env ]; then
    echo "No .env in $(pwd). Copy .env.example to .env and fill it in." >&2
    exit 1
fi

# set -a exports everything sourced, so the judge inherits it without any
# key ever being typed on the command line or landing in shell history.
set -a
# shellcheck disable=SC1091
. ./.env
set +a

if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "GOOGLE_API_KEY missing from .env, the judge cannot authenticate." >&2
    exit 1
fi

export OPENAI_API_KEY="$GOOGLE_API_KEY"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

# The scenario's judge block uses `factory: eval_judge.gemini_judge`, resolved
# with importlib. Console scripts do not put the working directory on sys.path,
# so without this the factory import fails no matter where you run from.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting bot on the eval transport, logging to eval-bot.log"
uv run bot.py -t eval > eval-bot.log 2>&1 &
BOT=$!

# Kill the bot however this script exits, including Ctrl-C and a failed run.
# Without this a crashed harness leaves an orphan holding port 7860, and the
# next run fails with a confusing bind error.
cleanup() {
    kill "$BOT" 2>/dev/null || true
    wait "$BOT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the port rather than sleeping a fixed amount. Startup loads the
# Silero VAD model, which is not instant on a cold filesystem cache.
echo -n "Waiting for ws://localhost:7860 "
for i in $(seq 1 40); do
    if nc -z localhost 7860 2>/dev/null; then
        echo "ready"
        break
    fi
    if ! kill -0 "$BOT" 2>/dev/null; then
        echo
        echo "Bot exited during startup. Last 20 lines of eval-bot.log:" >&2
        tail -20 eval-bot.log >&2
        exit 1
    fi
    echo -n "."
    sleep 0.5
done

if ! nc -z localhost 7860 2>/dev/null; then
    echo
    echo "Bot never opened port 7860. See eval-bot.log." >&2
    exit 1
fi

echo
uv run pipecat eval run "$SCENARIO" -v
