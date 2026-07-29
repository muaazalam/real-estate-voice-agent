# Cedar Grove Realty: an inbound voice agent

A real-time phone agent that answers a call, qualifies a real estate lead across
seven fields, and writes it to a database while the conversation is still going.
Built on [Pipecat](https://github.com/pipecat-ai/pipecat) 1.6 with a cascade
pipeline: Deepgram for speech, Gemini for reasoning, ElevenLabs for voice.

Median response time is **1.04 seconds** end to end, measured rather than
estimated. See [Latency](#latency-measured-not-estimated).

> Cedar Grove Realty is fictional. This is a portfolio project, built as working
> software rather than a no-code platform configuration.

---

## What it does

A caller dials in and has an ordinary conversation. Underneath, the agent is:

- **Qualifying the lead** across seven fields: intent, budget, area, bedrooms,
  property type, timeline, and financing status.
- **Persisting as it goes.** Every fact is saved the moment the caller says it,
  one field at a time, not batched at the end of the call. A caller who hangs
  up halfway through still leaves a usable lead.
- **Refusing to invent.** It will not quote a listing, a price, or an
  availability it does not have, and it will not promise to look something up
  that it cannot finish before it stops speaking.
- **Staying interruptible.** The caller can talk over it and it stops.

A representative call captured all seven fields across ten tool calls without
the caller ever being asked to repeat themselves.

## How it works

```
  caller audio  ──▶  Deepgram STT  ──▶  user context aggregator
                                                  │
                                                  ▼
                                            Gemini LLM  ──▶  save_lead_details
                                                  │                  │
                                          EmptyResponseGuard      SQLite
                                                  │              (aiosqlite)
                                                  ▼
                                        ElevenLabs TTS  ──▶  caller audio
                                                  │
                                          assistant context aggregator
```

Three transports share one bot: **WebRTC** for browser testing, **Twilio Media
Streams** for the phone line, and a headless **eval** transport that the test
harness drives with scripted conversations.

### Stack, and why

| Layer | Choice | Reasoning |
|---|---|---|
| Orchestration | Pipecat 1.6, cascade mode | Separate STT/LLM/TTS gives a text-mode test loop that speech-to-speech cannot. Iterating on prompts costs no audio credits and needs no microphone. |
| STT | Deepgram | Streaming, and it handles 8 kHz telephony audio. |
| LLM | Gemini `3.5-flash-lite` | Free tier, no credit card, and small enough not to compete with the pipeline for memory. |
| TTS | ElevenLabs `eleven_flash_v2_5` | The low-latency model, and half the credits per character of the standard one. |
| Storage | SQLite via `aiosqlite` | Async so a database write never blocks the audio loop. Schema written so a Postgres move is a driver change. |

**Sample rates are deliberately split.** Audio in is 8 kHz, matching what a
phone line actually delivers, so speech recognition accuracy stays
representative of the real thing rather than flattered by studio-quality input.
Audio out is 16 kHz, the lowest rate ElevenLabs offers.

## Latency, measured not estimated

Instrumentation was built before optimisation, because the first guess about
where the time goes is usually wrong. Per-stage timings are recorded on every
call and written to `latency.jsonl`; `uv run latency.py` compares runs grouped
by model and configuration.

Ten replies over WebRTC:

| stage | min | p50 | max | spread |
|---|---|---|---|---|
| user turn detection | 0.287s | 0.363s | 1.699s | 5.9x |
| **end to end** | **0.800s** | **1.039s** | **2.863s** | **3.6x** |
| Gemini time to first byte | 0.338s | 0.416s | 1.101s | 3.3x |
| ElevenLabs time to first audio | 0.124s | 0.139s | 0.165s | 1.3x |
| Deepgram time to first byte | 0.285s | 0.298s | 0.338s | 1.2x |

**The instrumentation paid for itself immediately.** It showed that two of six
turns were spending the full 3000ms ceiling in end-of-turn detection, waiting on
a caller who had already finished speaking. Tuning that one parameter to 1.5s
took end-to-end p50 from 2.512s to 1.039s and the worst case from 8.103s to
2.863s, without touching the model.

The value was chosen by testing the edge rather than guessing: a deliberate
two-second pause mid-sentence still gets cut off at 1.0s and does not at 1.5s.

**Read the spread column, not the median.** Gemini's free tier is the least
predictable stage by a wide margin, and a 4.7s outlier on a working turn is
normal. Adding the first tool moved p50 to 2.029s, because the model emits the
function call and then needs a second generation to speak.

## Testing

Voice agents cannot be eyeballed, so behaviour is asserted two ways.

**Behavioural evals.** Scripted conversations drive the running bot headless,
with an LLM judge scoring natural-language criteria. The seven-turn acceptance
scenario covers greeting, slot capture, memory across turns, refusing to invent
a listing, and redirecting off-topic questions. It passes 7/7.

```bash
./run-eval.sh evals/phase2_acceptance.yaml
```

The judge is a different model from the bot, deliberately, so they do not share
blind spots. Getting reliable verdicts out of it took four rounds of hardening,
documented in `eval_judge.py`.

**Frame-sequence tests.** The pipeline's failure modes are about ordering and
timing, which is exactly what reading code is worst at catching. Seventeen frame
sequences run the empty-response guard inside a real pipeline in a few seconds,
with no API calls:

```bash
uv run test_guard.py
```

## Engineering notes

A few problems worth reading the code for.

**The model sometimes says nothing, and sometimes says nothing for a long
time.** Gemini occasionally returns a completion with zero tokens. No error, no
exception, just a silent phone line while the caller says "Hello?" into the
void. `EmptyResponseGuard` in [`bot.py`](server/bot.py) watches for it and
speaks a filler. Getting it right took three attempts: the first two decided at
a frame boundary, and the condition is actually about elapsed time. The third
handles both the model answering with nothing and the model not answering at
all, which are indistinguishable to the caller and were not to the guard.

**Progressive saves are one atomic statement, not read-then-write.** The caller
reveals facts one at a time and each triggers a save. Reading the row, merging,
and writing it back opens a window where two saves in flight lose one of them.
`upsert_lead` in [`db.py`](server/db.py) is a single `ON CONFLICT` statement
instead.

**A tool that reports what is still missing.** `save_lead_details` returns the
list of fields not yet captured, so the model does not have to track seven slots
across a long conversation from context alone.

**Naming a tool that does not exist deadlocks the call.** An earlier prompt told
the model it could look up listings before that tool was written. The model
would announce it was checking and then stall forever, because it was waiting on
a function that was never going to resolve. The rule that replaced it, and which
survives into the current prompt, is that the agent never promises an action it
cannot finish before it stops speaking.

## Running it

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
cd server
uv sync
cp .env.example .env      # then add your keys
```

`GOOGLE_API_KEY`, `GEMINI_MODEL`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` and
`ELEVENLABS_VOICE_ID` are all required. There are no code defaults: the bot
exits at boot naming every missing one, rather than failing later inside a call
with something harder to read.

```bash
uv run bot.py -t webrtc    # browser mic and speakers, no telephony needed
uv run bot.py -t twilio    # the phone line
uv run bot.py -t eval      # headless, for the eval harness
```

With `-t webrtc`, open <http://localhost:7860> and talk to it.

## Project structure

```
server/
├── bot.py            pipeline, config validation, EmptyResponseGuard, transports
├── prompt.py         system prompts, with import-time asserts so the tested
│                     prompt and the shipped one cannot drift apart
├── db.py             calls, leads and bookings; async, atomic progressive upsert
├── tools.py          save_lead_details and the required-slot definition
├── latency.py        per-stage instrumentation and cross-run comparison
├── eval_judge.py     hardened LLM judge for the eval harness
├── test_guard.py     frame-sequence tests, no API calls
└── evals/            behavioural scenarios
```

## Status

Working: the conversation, lead qualification, persistence, grounding
guardrails, the eval suite, and the latency instrumentation.

Next: listing search and viewing bookings, then silence handling and human
handoff.

The telephony transport is implemented and the TwiML is written, but the Twilio
trial account it was built against never provisioned a number, so the demo runs
over WebRTC. Telnyx is the fallback and is a transport parameter change, since
the Pipecat runner supports it natively.
