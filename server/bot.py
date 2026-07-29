"""
bot.py

Phase 2: the Cedar Grove Realty inbound voice agent running its real persona.

Phase 1 proved audio moves in both directions. Phase 2 swaps in the full
real estate intake prompt and verifies the agent holds a coherent multi-turn
conversation: short replies, memory of earlier turns, no invented listings.
Still no tools and no database. Those are Phase 3 and 4.

Three transports are wired:

    uv run bot.py -t webrtc     browser mic and speakers, no Twilio, no phone
    uv run bot.py -t twilio     the real phone line, once Twilio is sorted
    uv run bot.py -t eval       headless, driven by the pipecat eval harness

Develop against webrtc. It exercises the same pipeline, the same VAD, and the
same interruption path as the phone, so Phase 5 barge-in work is testable
today rather than after the Twilio account is unblocked.

Use eval for regression runs. It needs no browser and no microphone, and in
text mode it spends no ElevenLabs credits:

    uv run bot.py -t eval                                     # terminal 1
    uv run pipecat eval run evals/phase2_acceptance.yaml -v   # terminal 2

Add -v for verbose logging to see the per-service TTFB metrics. Do NOT use it
on the bot itself, where it enables TRACE and buries everything useful.
"""

import asyncio
import os
import sys
import time

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from db import Database
from latency import LatencyRecorder
from prompt import PHASE_3_SYSTEM_PROMPT
from tools import (
    REQUIRED_SLOTS,
    SAVE_LEAD_DETAILS_SCHEMA,
    CallResources,
    save_lead_details,
)

# The FastAPI websocket params class moved during the 1.x reorganisation.
# Once check_api.py tells you which path your install uses, delete the branch
# you do not need and keep a single plain import.
try:
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
except ImportError:
    from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams


# ---------------------------------------------------------------------------
# Configuration, resolved and validated before anything else happens.
#
# find_dotenv walks up from the directory of the file that calls it, which is
# this one, NOT from the shell's working directory. That is how a .env at the
# repo root and a .env in server/ can both exist with server/ silently winning
# and the root file never being opened. Resolving the path explicitly and
# logging it means the file actually in use is never in question again.
#
# override=True also means these values beat anything exported in the shell, so
# `GEMINI_MODEL=... uv run bot.py` would NOT take effect. Edit the file instead.
# ---------------------------------------------------------------------------

DOTENV_PATH = find_dotenv(usecwd=False)
load_dotenv(DOTENV_PATH, override=True)


# Required configuration, with the reason each one exists. No code-level
# defaults, deliberately. A hardcoded fallback lets a missing value fail
# quietly and far from its cause: an absent GEMINI_MODEL once fell through to a
# stale hardcoded model id and surfaced as a mid-conversation 404 from Gemini,
# which reads like a service outage rather than a config typo. Fail here, at
# boot, naming the key and the file.
REQUIRED_ENV_VARS = {
    "GOOGLE_API_KEY": "Google AI Studio key, https://aistudio.google.com/apikey",
    "GEMINI_MODEL": "model id verified with a real generateContent call, ListModels lies",
    "DEEPGRAM_API_KEY": "Deepgram key, https://console.deepgram.com",
    "ELEVENLABS_API_KEY": "ElevenLabs key, https://elevenlabs.io/app/settings/api-keys",
    "ELEVENLABS_VOICE_ID": "voice id ADDED to your account, GET /v1/voices lists yours",
}

# Twilio credentials are only needed on the telephony path, so they are checked
# in create_transport's branch rather than here. Requiring them at boot would
# block the webrtc transport, which is the whole development loop.
TWILIO_ENV_VARS = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")


def validate_config(transport_name: str | None = None) -> None:
    """
    Fail fast and loudly on missing configuration.

    Reports every missing key at once rather than one per run, and names the
    exact file to edit. Called at import, before the server binds a port.
    """
    if DOTENV_PATH:
        logger.info(f"Config loaded from {DOTENV_PATH}")
    else:
        logger.warning(
            "No .env file found. Falling back to the ambient environment. "
            "Expected one at server/.env, next to bot.py."
        )

    required = dict(REQUIRED_ENV_VARS)
    if transport_name == "twilio":
        required.update(
            {name: "required by the twilio transport" for name in TWILIO_ENV_VARS}
        )

    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        width = max(len(name) for name in missing)
        target = DOTENV_PATH or "server/.env (create it from server/.env.example)"
        report = "\n".join(
            [
                "",
                "Missing required configuration. Not starting.",
                "",
                f"  file to edit: {target}",
                "",
                *(f"  {name.ljust(width)}   {required[name]}" for name in missing),
                "",
                "Add each key above to that exact file, then run again.",
                "",
            ]
        )
        logger.error(report)
        sys.exit(1)

    logger.info(
        f"Config OK. model={os.environ['GEMINI_MODEL']} "
        f"voice={os.environ['ELEVENLABS_VOICE_ID']}"
    )


validate_config()


# Twilio Media Streams carries 8 kHz mono mu-law. Running the browser transport
# at the same rate is deliberate: it means the VAD thresholds you tune and the
# STT accuracy you observe in development are the ones you get on the phone.
# Browser audio will sound noticeably thin as a result. That is the point.
#
# Set AUDIO_IN_SAMPLE_RATE=16000 in .env temporarily if you want to hear how
# the agent sounds without the telephony bandwidth limit. Do not tune against it.
# Deepgram still receives 8 kHz so STT accuracy stays representative of the
# phone. Output must be a rate ElevenLabs actually supports: their PCM formats
# are 16000, 22050, 24000 and 44100. The only 8 kHz option is ulaw_8000, which
# is what the TwilioFrameSerializer requests on the telephony path.
AUDIO_IN_SAMPLE_RATE = int(os.getenv("AUDIO_IN_SAMPLE_RATE", "8000"))
AUDIO_OUT_SAMPLE_RATE = int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "16000"))

# How long the smart turn analyzer will wait, once the caller has gone quiet,
# before ending their turn anyway. Pipecat's default is 3 seconds
# (base_smart_turn.STOP_SECS).
#
# Measured on 2026-07-29, first instrumented call, six replies. Two of them hit
# the ceiling exactly:
#
#     End of Turn complete due to stop_secs. Silence in ms: 3000.0
#
# and those two turns show `user turn 3.201s` and `user turn 3.202s` in the
# latency log against a median of 0.356s. The analyzer returned
# EndOfTurnState.INCOMPLETE on four of eight analyses, so it genuinely was
# unsure rather than misfiring; the cost is that being unsure is expensive.
# Turn 6 stacked a 3.202s wait onto a 4.651s Gemini response for 8.1 seconds of
# silence, which on a phone line reads as a dropped call.
#
# THE TRADEOFF, which is the whole point of this being a knob. Lower means the
# analyzer's "not finished yet" judgement gets overruled sooner, so replies come
# faster AND callers who pause mid-sentence get interrupted more. Higher is the
# reverse. There is no correct value, only the one that sounds right for this
# demo, and the only way to know is to listen.
#
# Change it in .env, not here, and compare runs with `uv run latency.py`. The
# number to watch is the `user turn` row: its max should drop toward this value
# and its spread should collapse.
SMART_TURN_STOP_SECS = float(os.getenv("SMART_TURN_STOP_SECS", "1.5"))

# How long the caller may sit in silence after they stop speaking before
# EmptyResponseGuard holds the line for them. This is a DEAD AIR budget, not a
# latency target: it has to sit above the slowest turn that still works, or it
# will interrupt healthy conversations.
#
# 6.0 is chosen from measurements, not taste. Gemini TTFB p50 is 0.416s and the
# worst TTFB on a turn that completed correctly was 4.743s (run-phase3b.log).
# 5.0 would have fired on that turn. Raise this before suspecting anything else
# if you hear the hold phrase during a normal-sounding call.
STALL_FILLER_SECS = float(os.getenv("STALL_FILLER_SECS", "6.0"))

# Every tool the agent is told about. Phase 4 appends search_listings and
# book_viewing here. Defined once so the pipeline and the latency record cannot
# disagree about how many tools were live on a given run.
TOOL_SCHEMAS = (SAVE_LEAD_DETAILS_SCHEMA,)

# Transport configuration, keyed by the -t flag. create_transport picks the
# matching entry. Note there is no vad_analyzer here: in Pipecat 1.x the VAD
# analyzer belongs to the user aggregator, not the transport. Some reference
# docstrings still show the old placement.
def _twilio_params() -> FastAPIWebsocketParams:
    """
    Built by create_transport only when -t twilio is selected, which makes it
    the right place to require the Twilio credentials. Checking them at import
    would block the webrtc transport, and webrtc is the entire dev loop.
    """
    validate_config(transport_name="twilio")
    return FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # Twilio wants a raw mu-law stream, not a WAV file with a header on
        # the front. Leaving this True is a classic cause of audio that sounds
        # like static or does not play at all. The runner sets this and the
        # serializer automatically, but being explicit costs nothing.
        add_wav_header=False,
    )


class EmptyResponseGuard(FrameProcessor):
    """
    Speak a filler when the LLM produces a turn with no text at all.

    Observed on 2026-07-28, run-speech.log around 20:09:03. The caller said
    "Seven", the turn analyzer returned EndOfTurnState.INCOMPLETE and stalled
    3000ms before firing anyway, and Gemini then answered with:

        GoogleLLMService#0 prompt tokens: 525, completion tokens: 0

    Zero completion tokens means no LLMTextFrame, so TTS never ran and the line
    went silent. The caller waited roughly twelve seconds and said "Hello?".
    Nothing in the log flagged it: the whole run had one warning and it was the
    usual teardown one. A crash would have been kinder than this.

    The retry a second later had a near-identical context, 528 prompt tokens
    against 525, and produced 43 tokens. So this is model non-determinism, not a
    malformed context, and it can land on any turn.

    BROKEN AND FIXED BY PHASE 3, 2026-07-29. The original version reset its
    flags on every LLMFullResponseStartFrame, so its unit of judgement was one
    LLM response window. That was correct while the agent had no tools, and
    wrong the moment it had one, because a response window with no text stopped
    meaning "the model failed" and started meaning one of three things:

      1. The model emitted a FUNCTION CALL instead of text. Entirely normal.
         Log: `completion tokens: 15` and `Function call: save_lead_details`,
         with no LLMTextFrame anywhere.
      2. An async tool result came back and triggered a generation that did
         nothing at all. Log: `prompt tokens: 0, completion tokens: 0`.
      3. The model genuinely returned nothing, which is the only case this
         class exists for.

    Firing on 1 and 2 meant the caller heard "Sorry, I did not catch that"
    after almost every sentence, and worse, TTSSpeakFrame appends to context,
    so the transcript filled with apologies the model then had to reason
    around. Eight fires in one short conversation.

    THE SECOND ATTEMPT, AND WHY IT ALSO FAILED. Making the unit of judgement
    one user utterance instead of one response window fixed most of it, 8 false
    fires down to 3, and was still wrong, because it still DECIDED at the close
    of a response window:

        User started speaking
        prompt tokens: 0, completion tokens: 0     <- empty window
        EmptyResponseGuard fires                    <- premature
        Function call: save_lead_details            <- the real answer, later

    An empty window is not the end of the turn. With async tools the model's
    actual response can arrive one or two windows after an empty one, and no
    amount of flag bookkeeping fixes a decision made too early.

    THE ACTUAL FIX: WAIT.

    The condition this class cares about is "the caller is sitting in silence",
    and that is a fact about TIME, not about frame order. So an empty window
    schedules the filler rather than speaking it, and anything that proves the
    model is alive cancels it. If the model was merely slow or between windows,
    the timer never expires. If the turn really is dead, the caller hears the
    filler a beat later than they would have. The original incident had them
    waiting twelve seconds, so a two second delay costs nothing that matters.

    ATTEMPT 3, AND THE HOLE ATTEMPT 2 LEFT. 2026-07-29, run-phase3c.log.
    Google's free tier degraded mid-morning: gemini-3.5-flash-lite went from a
    0.416s p50 TTFB to 30s on a one-word probe, and gemini-3.5-flash returned
    HTTP 503. The caller said "I'm looking to buy", heard nothing, said
    "Hello?", heard nothing, said "Are you there?", and hung up after twelve
    seconds of silence. THE GUARD NEVER FIRED.

    It could not have. Attempt 2 arms its timer on LLMFullResponseEndFrame, so
    its precondition is "the model finished and said nothing". Here the model
    had not finished. A request that never comes back never closes a window,
    never reaches the arming point, and never starts a timer. The guard slept
    through the exact silence it exists to fill.

    Two failures, identical from the caller's chair, and attempt 2 only saw
    one:

        window CLOSED with no text   the model answered with nothing
        window never CLOSED          the model has not answered yet

    So arm at the START of the wait rather than at a point inside it. A user
    utterance is the only event guaranteed to precede every possible silence,
    so UserStoppedSpeakingFrame arms a long STALL deadline, and a window that
    closes empty SHORTENS that deadline to the old two seconds, because a
    closed empty window is positive evidence rather than an absence of news.
    A deadline is only ever moved earlier, never later.

        UserStoppedSpeakingFrame    new utterance: clear, then ARM (stall)
        LLMTextFrame                model spoke, cancel
        FunctionCallsStartedFrame   model acted, cancel
        InterruptionFrame           barge-in, an empty response is CORRECT
        LLMFullResponseEndFrame     nothing yet? SHORTEN to silence_secs
        EndFrame / CancelFrame      teardown, cancel

    The two cases also deserve different words. "Sorry, I did not catch that"
    asks the caller to repeat themselves, which is right when the model
    produced nothing and wrong when it is merely slow: it heard them fine. A
    stall gets a neutral hold instead, which buys time without lying and
    without asking for anything back.

    TTSSpeakFrame defaults to append_to_context=True so the assistant
    aggregator records what was said, which also stops the context
    accumulating two consecutive user messages, the fingerprint of the bug
    this class was written for. run-phase3c.log shows that fingerprint in its
    purest form: three consecutive 'user' parts and no 'model' between them.
    """

    def __init__(
        self,
        filler: str,
        stall_filler: str,
        silence_secs: float = 2.0,
        stall_secs: float = STALL_FILLER_SECS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._filler = filler
        self._stall_filler = stall_filler
        self._silence_secs = silence_secs
        self._stall_secs = stall_secs
        self._responding = False
        self._saw_text = False
        self._saw_function_call = False
        self._interrupted = False
        self._fired_this_utterance = False
        self._pending: asyncio.Task | None = None
        self._deadline: float = 0.0

    async def _cancel_pending(self) -> None:
        if self._pending is not None:
            await self.cancel_task(self._pending)
            self._pending = None
            self._deadline = 0.0

    async def _arm(self, delay: float, text: str, direction: FrameDirection) -> None:
        """
        Schedule the filler for `delay` seconds from now.

        Monotonic ABSOLUTE deadlines, not durations, and a deadline is only
        ever moved EARLIER. Both matter. The empty-window path re-arms an
        already-running stall timer, and re-arming with a duration would let a
        window that closed late push the filler further out than the stall
        timer had it, which is backwards: closing empty is worse news than not
        closing at all, so it must never buy the model more time.
        """
        if self._fired_this_utterance or self._interrupted:
            return
        deadline = time.monotonic() + delay
        if self._pending is not None:
            if deadline >= self._deadline:
                return
            await self._cancel_pending()
        self._deadline = deadline
        self._pending = self.create_task(
            self._speak_at(deadline, text, direction),
            name="empty-response-filler",
        )

    async def _speak_at(
        self, deadline: float, text: str, direction: FrameDirection
    ) -> None:
        """Wait out the silence, then fill it if nothing has arrived."""
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        if self._interrupted or self._fired_this_utterance:
            return
        logger.warning(
            f"EmptyResponseGuard: silence with no text and no tool call, "
            f"speaking filler: {text!r}"
        )
        self._fired_this_utterance = True
        await self.push_frame(TTSSpeakFrame(text), direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Two mechanisms, and both are needed. The FLAGS stop a timer being
        # started when the model already answered inside this window. The TIMER
        # covers the case the flags cannot see, where the answer lands in a
        # LATER window. An earlier version had only the timer, and cancelling
        # on text did nothing because the timer is not created until the end
        # frame, which is after the text: every normal reply got a filler.
        if isinstance(frame, UserStoppedSpeakingFrame):
            # New utterance. Whatever was pending belonged to the last one.
            await self._cancel_pending()
            self._saw_text = False
            self._saw_function_call = False
            self._interrupted = False
            self._fired_this_utterance = False
            # Arm the stall watchdog. This is the only arming point that is
            # guaranteed to be reached, because it does not depend on the LLM
            # doing anything at all. See "ATTEMPT 3" in the class docstring.
            await self._arm(self._stall_secs, self._stall_filler, direction)

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._responding = True

        elif isinstance(frame, LLMTextFrame):
            # Whitespace-only deltas are common while streaming and must not
            # count as a real answer.
            if frame.text and frame.text.strip():
                self._saw_text = True
                await self._cancel_pending()

        elif isinstance(frame, FunctionCallsStartedFrame):
            # Broadcast downstream from run_function_calls, which the LLM
            # service calls BEFORE pushing LLMFullResponseEndFrame. The model
            # is working; the caller is not being abandoned.
            self._saw_function_call = True
            await self._cancel_pending()

        elif isinstance(frame, InterruptionFrame):
            # The caller barged in, so the truncated or empty response is the
            # intended outcome. Speaking a filler here would talk over them.
            self._interrupted = True
            await self._cancel_pending()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Never leave a timer running into teardown, or the filler fires
            # into a pipeline that is already closing.
            await self._cancel_pending()

        elif isinstance(frame, LLMFullResponseEndFrame):
            # A window that closed with nothing in it is positive evidence,
            # not merely an absence of news, so it SHORTENS the stall deadline
            # to silence_secs. _arm refuses to move a deadline later, so a
            # window that closes empty after the stall timer is already most
            # of the way through cannot reprieve it.
            nothing_yet = not self._saw_text and not self._saw_function_call
            if self._responding and nothing_yet:
                await self._arm(self._silence_secs, self._filler, direction)
            self._responding = False

        await self.push_frame(frame, direction)


def _transport_label(runner_args: RunnerArguments) -> str:
    """Name the transport, for the latency record.

    There is no `transport` field on RunnerArguments; only
    WebSocketRunnerArguments carries `transport_type`, and it is None until
    create_transport auto-detects the telephony provider. What always differs
    per -t flag is the SUBCLASS the runner instantiates, so derive from that.

    Getting this wrong is quiet rather than loud: every latency sample would be
    filed under "unknown" and the whole point of the file is comparing runs.
    """
    explicit = getattr(runner_args, "transport_type", None)
    if explicit:
        return str(explicit)
    name = type(runner_args).__name__.removesuffix("RunnerArguments")
    return {
        "SmallWebRTC": "webrtc",
        "Eval": "eval",
        "WebSocket": "websocket",
        "Daily": "daily",
        "LiveKit": "livekit",
        "Vonage": "vonage",
    }.get(name, name.lower() or "unknown")


def _eval_params():
    """
    Transport for `pipecat eval run`. The harness connects as an RTVI client
    over a plain WebSocket and drives the conversation from a YAML scenario.

    No RTVIProcessor or RTVIObserver is added here on purpose. PipelineWorker
    creates both automatically unless it finds them already in the pipeline
    (enable_rtvi defaults to True), and adding them by hand only earns a
    "skipping default ones" warning. The webrtc run already proves this: the
    logs show RTVIProcessor#0 linked into the pipeline with nothing in this
    file creating it.

    Imported lazily so the webrtc and twilio paths do not pay for the evals
    extra at startup, matching how pipecat's own runner does it.

    audio_in/out are both enabled so one transport serves both scenario modes.
    Text mode sends RTVI send-text and skips TTS regardless; audio mode needs
    these on, and the harness enables its virtual mic per connection.
    """
    from pipecat.evals.transport import EvalTransportParams

    return EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": _twilio_params,
    "eval": _eval_params,
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Assemble and run the voice pipeline for a single inbound conversation."""

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    # Speech to text. Deepgram streams interim and final transcripts as the
    # caller speaks rather than waiting for them to finish, which is most of
    # why this feels responsive instead of walkie-talkie.
    speech_to_text = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
    )

    # The language model. Gemini Flash is the right class of model here: your
    # replies are one or two sentences, so time to first token matters far
    # more than reasoning depth. PHASE_2_SYSTEM_PROMPT carries the Cedar Grove
    # persona, the qualification slots, and the no-invented-listings guardrail.
    # No default model id here. The previous "gemini-2.5-flash" fallback is
    # exactly what made a missing GEMINI_MODEL look like a Gemini outage.
    language_model = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            model=os.environ["GEMINI_MODEL"],
            system_instruction=PHASE_3_SYSTEM_PROMPT,
        ),
    )

    # Registered explicitly rather than by attaching the handler to the schema,
    # because the schema route registers with default options and the option
    # that matters here is not the default.
    #
    # cancel_on_interruption=False is pipecat's name for "asynchronous": the
    # LLM continues talking immediately instead of waiting for the result. That
    # is what makes saving on every new fact affordable. Nothing the agent says
    # next depends on whether the row was written, so making the caller wait
    # for a disk write plus a second round trip would spend the entire Phase
    # 5.5 latency win on nothing.
    language_model.register_function(
        "save_lead_details",
        save_lead_details,
        cancel_on_interruption=False,
    )

    # One database and one call row per conversation. Opened here rather than
    # at import so a failure to open is scoped to this call instead of taking
    # the whole server down.
    database = Database()
    await database.connect()
    call_id = await database.start_call(transport=_transport_label(runner_args))
    call_resources = CallResources(db=database, call_id=call_id)

    # Catches both ways the caller ends up in silence: the model returning zero
    # completion tokens, and the model not returning at all. See the class
    # docstring for the exact log lines that motivated each.
    #
    # The two fillers are deliberately different. Asking someone to repeat
    # themselves is right when the model produced nothing and wrong when it is
    # just slow, because it heard them perfectly well. Neither line promises an
    # action, which is the rule from log 013.
    empty_response_guard = EmptyResponseGuard(
        filler="Sorry, I did not catch that. Could you say it again?",
        stall_filler="One moment, please.",
    )

    # Text to speech. eleven_flash_v2_5 is both the low latency model and the
    # one that costs half the credits per character, which matters a lot on a
    # 10k credit monthly budget.
    text_to_speech = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(
            voice=os.environ["ELEVENLABS_VOICE_ID"],
            model="eleven_flash_v2_5",
        ),
    )

    # Conversation memory. The context object holds the running message list.
    # The aggregator pair writes into it: the user aggregator appends what the
    # caller said, the assistant aggregator appends what the bot said, and they
    # sit on opposite sides of the LLM in the pipeline.
    #
    # LLMContextAggregatorPair is an object with .user() and .assistant()
    # accessors. It is not a 2-tuple and does not unpack.
    # The tools list is what ADVERTISES the function to Gemini. Registering the
    # handler above only says what to run if it is called; without the schema
    # here the model never learns the tool exists and will never call it.
    context = LLMContext(tools=list(TOOL_SCHEMAS))
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Voice activity detection lives here in Pipecat 1.x. This is the
            # component that decides when the caller has stopped talking, and
            # it is what you tune in Phase 5 to get barge-in feeling right.
            vad_analyzer=SileroVADAnalyzer(),
            # VAD and the turn analyzer are two different things and both cost
            # time. VAD decides whether there is speech at all; the analyzer
            # decides whether an UTTERANCE is finished, and it is the one that
            # was spending 3 seconds a turn. Only `stop` is overridden here, so
            # `start` keeps pipecat's defaults (VAD plus transcription).
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=SMART_TURN_STOP_SECS),
                        ),
                    ),
                ],
            ),
        ),
    )
    logger.info(f"Smart turn analyzer: stop_secs={SMART_TURN_STOP_SECS} (pipecat default 3)")

    # Order matters. Audio has to be transcribed before the model can read it,
    # and text has to be synthesised before it can be played. The assistant
    # aggregator sits after transport.output() so that an interrupted reply is
    # recorded as what was actually spoken, not what was generated.
    pipeline = Pipeline(
        [
            transport.input(),
            speech_to_text,
            aggregators.user(),
            language_model,
            # Between LLM and TTS on purpose: it needs to see the response
            # window close (LLMFullResponseEndFrame) while it can still inject
            # a TTSSpeakFrame that the TTS below will actually synthesise.
            empty_response_guard,
            text_to_speech,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    # PipelineWorker cancels itself after IDLE_TIMEOUT_SECS, 300 by default.
    # On a live call that is correct behaviour: a caller silent for five
    # minutes should be hung up on. Under eval it is actively destructive.
    #
    # The idle timer only resets on BotSpeakingFrame or UserSpeakingFrame, and
    # a text-mode scenario emits NEITHER, because text mode skips TTS and sends
    # no audio. So the clock starts at StartFrame and never resets, and any
    # text eval lasting longer than five minutes kills its own bot mid-run.
    #
    # Observed 2026-07-28. Bot up at 20:44:16, then at exactly 20:49:16:
    #     _idle_timeout_detected - ...and cancelling the runner.
    # The harness was still waiting on turn 0's judge, and the next turn died
    # on ConnectionClosedOK: received 1001 (going away). The traceback points
    # at the harness, which is misleading; the bot had shut itself down.
    #
    # Only relaxed for eval. webrtc and twilio keep the default.
    worker_kwargs = {}
    if isinstance(runner_args, EvalRunnerArguments):
        worker_kwargs["idle_timeout_secs"] = None
        logger.info("Eval transport: idle timeout disabled for this run")

    # Phase 5.5 instrumentation. UserBotLatencyObserver does the measuring;
    # latency.py records what it emits, logs a line per turn and prints a
    # per-stage table at hangup. See latency.py for what the numbers mean.
    #
    # Passed in explicitly rather than relying on PipelineWorker. The worker
    # builds one of these itself, but only when enable_tracing is on AND
    # OpenTelemetry is available (worker.py, the `if self._enable_tracing and
    # self._turn_tracking_observer` branch), which is not this project. An
    # observer sits outside the pipeline and cannot alter frames, so this
    # cannot affect the conversation. That matters: instrumentation that can
    # change behaviour is worse than no instrumentation.
    #
    # Requires enable_metrics=True below, which is where the TTFB numbers the
    # observer reads actually come from.
    latency_observer = UserBotLatencyObserver()
    latency_recorder = LatencyRecorder(
        model=os.environ["GEMINI_MODEL"],
        transport=_transport_label(runner_args),
        # Anything tuned per run belongs here, or `uv run latency.py` pools two
        # different experiments under one heading and averages away the effect.
        # `tools` is in here because adding one changes what the model has to
        # generate on a turn, so pre-tool and post-tool runs are not comparable
        # and must not share a heading.
        config={"stop_secs": SMART_TURN_STOP_SECS, "tools": len(TOOL_SCHEMAS)},
    )
    latency_recorder.attach(latency_observer)

    agent = PipelineWorker(
        pipeline,
        name="cedar-grove-intake",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
        ),
        observers=[latency_observer],
        # Handed to every tool handler as FunctionCallParams.app_resources, by
        # reference, so all of this call's tools share one connection and one
        # call_id.
        app_resources=call_resources,
        **worker_kwargs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Caller is on the line. Make the agent speak first."""
        logger.info("Client connected, starting conversation")
        # Queueing an LLMRunFrame with no preceding user turn is what makes the
        # agent greet first instead of sitting in silence. On an inbound line
        # the agent always opens.
        await agent.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Caller hung up. Report latency, then cancel so we do not leak."""
        logger.info("Client disconnected, cancelling runner")

        # Before the cancel, not after. Cancelling tears the worker down and
        # anything queued behind it is not guaranteed to run.
        #
        # Wrapped because a crash in reporting must never be the reason a
        # session fails to close cleanly. Instrumentation is allowed to be
        # wrong; it is not allowed to be load-bearing.
        try:
            summary = latency_recorder.summary()
            if summary:
                logger.info(summary)
            latency_recorder.write_samples()
        except Exception as e:
            logger.warning(f"Latency reporting failed, continuing teardown: {e!r}")

        # Close the call row and the connection. Separately wrapped from the
        # latency block above: a reporting failure must not be the reason a
        # lead's call never gets stamped as finished.
        try:
            lead = await database.get_lead(call_id)
            logger.info(
                f"Call {call_id} captured lead: {lead['id'] if lead else 'NONE'}"
                + (f", still missing {await database.missing_fields(call_id, REQUIRED_SLOTS)}"
                   if lead else "")
            )
            await database.end_call(call_id)
            await database.close()
        except Exception as e:
            logger.warning(f"Database teardown failed: {e!r}")

        await runner.cancel()

    await runner.add_workers(agent)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """
    Entry point called by Pipecat's development runner.

    The runner owns the FastAPI server. For twilio it accepts the incoming
    WebSocket, parses the Twilio start message for the stream SID and call SID,
    and builds the TwilioFrameSerializer. For webrtc it serves the prebuilt
    client UI and negotiates the peer connection. Either way it hands back a
    ready transport, which is why there is no hand-written webhook route or
    serializer wiring in this file.
    """
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
