"""
bot.py

Phase 2: the Cedar Grove Realty inbound voice agent running its real persona.

Phase 1 proved audio moves in both directions. Phase 2 swaps in the full
real estate intake prompt and verifies the agent holds a coherent multi-turn
conversation: short replies, memory of earlier turns, no invented listings.
Still no tools and no database. Those are Phase 3 and 4.

Two transports are wired:

    uv run bot.py -t webrtc     browser mic and speakers, no Twilio, no phone
    uv run bot.py -t twilio     the real phone line, once Twilio is sorted

Develop against webrtc. It exercises the same pipeline, the same VAD, and the
same interruption path as the phone, so Phase 5 barge-in work is testable
today rather than after the Twilio account is unblocked.

Add -v for verbose logging to see the per-service TTFB metrics.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from prompt import PHASE_2_SYSTEM_PROMPT

# The FastAPI websocket params class moved during the 1.x reorganisation.
# Once check_api.py tells you which path your install uses, delete the branch
# you do not need and keep a single plain import.
try:
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
except ImportError:
    from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams


load_dotenv(override=True)


# Twilio Media Streams carries 8 kHz mono mu-law. Running the browser transport
# at the same rate is deliberate: it means the VAD thresholds you tune and the
# STT accuracy you observe in development are the ones you get on the phone.
# Browser audio will sound noticeably thin as a result. That is the point.
#
# Set AUDIO_SAMPLE_RATE=16000 in .env temporarily if you want to hear how the
# agent sounds without the telephony bandwidth limit. Do not tune against it.
# Deepgram still receives 8 kHz so STT accuracy stays representative of the
# phone. Output must be a rate ElevenLabs actually supports: their PCM formats
# are 16000, 22050, 24000 and 44100. The only 8 kHz option is ulaw_8000, which
# is what the TwilioFrameSerializer requests on the telephony path.
AUDIO_IN_SAMPLE_RATE = int(os.getenv("AUDIO_IN_SAMPLE_RATE", "8000"))
AUDIO_OUT_SAMPLE_RATE = int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "16000"))

# Transport configuration, keyed by the -t flag. create_transport picks the
# matching entry. Note there is no vad_analyzer here: in Pipecat 1.x the VAD
# analyzer belongs to the user aggregator, not the transport. Some reference
# docstrings still show the old placement.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # Twilio wants a raw mu-law stream, not a WAV file with a header on
        # the front. Leaving this True is a classic cause of audio that sounds
        # like static or does not play at all. The runner sets this and the
        # serializer automatically, but being explicit costs nothing.
        add_wav_header=False,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Assemble and run the voice pipeline for a single inbound conversation."""

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    # Speech to text. Deepgram streams interim and final transcripts as the
    # caller speaks rather than waiting for them to finish, which is most of
    # why this feels responsive instead of walkie-talkie.
    speech_to_text = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    # The language model. Gemini Flash is the right class of model here: your
    # replies are one or two sentences, so time to first token matters far
    # more than reasoning depth. PHASE_2_SYSTEM_PROMPT carries the Cedar Grove
    # persona, the qualification slots, and the no-invented-listings guardrail.
    language_model = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            system_instruction=PHASE_2_SYSTEM_PROMPT,
        ),
    )

    # Text to speech. eleven_flash_v2_5 is both the low latency model and the
    # one that costs half the credits per character, which matters a lot on a
    # 10k credit monthly budget.
    text_to_speech = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
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
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Voice activity detection lives here in Pipecat 1.x. This is the
            # component that decides when the caller has stopped talking, and
            # it is what you tune in Phase 5 to get barge-in feeling right.
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

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
            text_to_speech,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    agent = PipelineWorker(
        pipeline,
        name="cedar-grove-intake",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
        ),
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
        """Caller hung up. Cancel the runner so we do not leak the worker."""
        logger.info("Client disconnected, cancelling runner")
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
