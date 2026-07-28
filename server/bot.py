"""
bot.py

Phase 1: the bare call loop for the Cedar Grove Realty inbound voice agent.

Caller dials the Twilio number, Twilio opens a Media Streams WebSocket to this
process, audio flows through Deepgram to Gemini to ElevenLabs and back out to
the caller. No tools, no database, no slot filling yet. The only thing this
file proves is that audio moves in both directions with acceptable latency.

Run it with:
    uv run bot.py -t twilio

BEFORE YOU RUN THIS: run 'uv run python check_api.py' first. Pipecat 1.x moved
several classes between modules and the import block below reflects the layout
verified against the official examples. If check_api.py reports a different
path for anything, use its output, not this file.

THE ONE BLOCK YOU MAY NEED TO RECONCILE is marked TRANSPORT BLOCK below. Your
scaffolded bot.py already contains a transport setup that is correct for your
exact installed version. If this file's version misbehaves, paste yours in.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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
from pipecat.transports.base_transport import BaseTransport

from prompt import PHASE_1_SYSTEM_PROMPT

# Small compatibility shim. The FastAPI websocket params class moved during the
# 1.x reorganisation. Once check_api.py tells you which path your install uses,
# delete the branch you do not need and keep a single plain import.
try:
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
except ImportError:
    from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams


load_dotenv(override=True)


# Twilio Media Streams carries 8 kHz mono audio. Declaring that here means
# Pipecat does not resample up and back down for nothing, which costs latency
# and audio quality for zero benefit.
TELEPHONY_SAMPLE_RATE = 8000


async def run_bot(transport: BaseTransport, handle_sigint: bool) -> None:
    """Assemble and run the voice pipeline for a single inbound call."""

    # Speech to text. Deepgram streams interim and final transcripts as the
    # caller speaks rather than waiting for them to finish, which is most of
    # why this feels responsive instead of walkie-talkie.
    speech_to_text = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    # The language model. Gemini Flash is the right class of model here: your
    # replies are one or two sentences, so time to first token matters far more
    # than reasoning depth.
    language_model = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            system_instruction=PHASE_1_SYSTEM_PROMPT,
        ),
    )

    # Text to speech. eleven_flash_v2_5 is both the low latency model and the
    # one that costs half the credits per character, which matters a lot on the
    # free tier. It is already the ElevenLabs service default, set explicitly so
    # nobody has to go read the source to find out.
    text_to_speech = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID"),
            model="eleven_flash_v2_5",
        ),
    )

    # Conversation memory. The context object holds the running message list.
    # The aggregator pair is what actually writes into it: the user aggregator
    # appends what the caller said, the assistant aggregator appends what the
    # bot said, and they sit on opposite sides of the LLM in the pipeline.
    conversation_context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        conversation_context,
        user_params=LLMUserAggregatorParams(
            # Voice activity detection lives here in Pipecat 1.x, not on the
            # transport as older tutorials show. This is the component that
            # decides when the caller has stopped talking, and it is also what
            # you tune in Phase 5 to get barge-in feeling right.
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
            user_aggregator,
            language_model,
            text_to_speech,
            transport.output(),
            assistant_aggregator,
        ]
    )

    pipeline_task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=TELEPHONY_SAMPLE_RATE,
            audio_out_sample_rate=TELEPHONY_SAMPLE_RATE,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Caller is on the line. Make the agent speak first."""
        logger.info("Caller connected, starting conversation")
        # Queueing an LLMRunFrame with an empty user turn is what makes the
        # agent greet the caller instead of sitting in silence waiting for
        # them to speak first. On an inbound line, the agent always opens.
        await pipeline_task.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Caller hung up. Tear the pipeline down so we do not leak the task."""
        logger.info("Caller disconnected, cancelling pipeline")
        await pipeline_task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(pipeline_task)


async def bot(runner_args: RunnerArguments):
    """
    Entry point called by Pipecat's development runner.

    The runner owns the FastAPI server, accepts the incoming Twilio WebSocket,
    parses the Twilio start message for the stream SID and call SID, builds the
    TwilioFrameSerializer, and hands you a ready transport. This is why there is
    no hand-written webhook route or serializer wiring in this file.
    """

    # ------------------------------------------------------------------
    # TRANSPORT BLOCK
    # If anything about the connection misbehaves, this is the block to
    # replace with the one from your scaffolded bot.py.
    # ------------------------------------------------------------------
    transport_params = {
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Twilio wants a raw mu-law stream, not a WAV file with a header
            # bolted on the front. Leaving this True is a classic cause of
            # audio that sounds like static or does not play at all.
            add_wav_header=False,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    # ------------------------------------------------------------------
    # END TRANSPORT BLOCK
    # ------------------------------------------------------------------

    await run_bot(transport, runner_args.handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
