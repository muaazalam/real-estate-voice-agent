"""
check_api.py

Pipecat 1.x moved a lot of classes between modules, and not every example in the
wild has caught up. Run this before editing bot.py. It probes YOUR installed
version, reports which import paths actually resolve, and prints an import block
you can paste straight into bot.py.

Run from the project root:
    uv run python check_api.py
"""

import importlib
import sys


def try_import(module_path, class_name):
    """Attempt to import class_name from module_path. Return True if it worked."""
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return False
    return hasattr(module, class_name)


def probe(label, candidates):
    """
    Try each (module_path, class_name) candidate in order.
    Print the first one that resolves, or a failure line if none do.
    Returns the winning import line as a string, or None.
    """
    for module_path, class_name in candidates:
        if try_import(module_path, class_name):
            import_line = "from {} import {}".format(module_path, class_name)
            print("  OK    {:<28} {}".format(label, import_line))
            return import_line
    print("  FAIL  {:<28} none of {} candidates resolved".format(label, len(candidates)))
    for module_path, class_name in candidates:
        print("          tried: from {} import {}".format(module_path, class_name))
    return None


def main():
    print()
    print("Python:", sys.version.split()[0])

    try:
        import pipecat
        installed_version = getattr(pipecat, "__version__", "unknown")
        print("Pipecat:", installed_version)
    except ImportError:
        print("Pipecat is not installed in this environment.")
        print("Are you running this with 'uv run python check_api.py' from the project root?")
        return

    print()
    print("Probing import paths")
    print("-" * 70)

    resolved_import_lines = []

    # The pipeline execution API. This is the one that differs between the
    # examples repo (PipelineTask / PipelineRunner) and the current quickstart
    # docs (PipelineWorker / WorkerRunner).
    task_import = probe("pipeline task", [
        ("pipecat.pipeline.task", "PipelineWorker"),
        ("pipecat.pipeline.task", "PipelineTask"),
    ])
    runner_import = probe("pipeline runner", [
        ("pipecat.pipeline.runner", "WorkerRunner"),
        ("pipecat.pipeline.runner", "PipelineRunner"),
    ])

    params_import = probe("pipeline params", [
        ("pipecat.pipeline.task", "PipelineParams"),
    ])
    pipeline_import = probe("pipeline", [
        ("pipecat.pipeline.pipeline", "Pipeline"),
    ])

    # Context and aggregators.
    context_import = probe("llm context", [
        ("pipecat.processors.aggregators.llm_context", "LLMContext"),
    ])
    aggregator_import = probe("context aggregator", [
        ("pipecat.processors.aggregators.llm_response_universal", "LLMContextAggregatorPair"),
    ])
    user_params_import = probe("user aggregator params", [
        ("pipecat.processors.aggregators.llm_response_universal", "LLMUserAggregatorParams"),
    ])

    # Voice activity detection. In 1.x this attaches to the user aggregator,
    # not to the transport.
    vad_import = probe("silero vad", [
        ("pipecat.audio.vad.silero", "SileroVADAnalyzer"),
    ])

    # Services.
    stt_import = probe("deepgram stt", [
        ("pipecat.services.deepgram.stt", "DeepgramSTTService"),
        ("pipecat.services.deepgram", "DeepgramSTTService"),
    ])
    llm_import = probe("google llm", [
        ("pipecat.services.google.llm", "GoogleLLMService"),
        ("pipecat.services.google", "GoogleLLMService"),
    ])
    tts_import = probe("elevenlabs tts", [
        ("pipecat.services.elevenlabs.tts", "ElevenLabsTTSService"),
        ("pipecat.services.elevenlabs", "ElevenLabsTTSService"),
    ])

    # Runner plumbing. create_transport is what wires the Twilio websocket and
    # serializer for you, so you do not hand-build FastAPIWebsocketTransport.
    runner_types_import = probe("runner arguments", [
        ("pipecat.runner.types", "RunnerArguments"),
    ])
    create_transport_import = probe("create_transport", [
        ("pipecat.runner.utils", "create_transport"),
    ])

    # Frames.
    frame_import = probe("llm run frame", [
        ("pipecat.frames.frames", "LLMRunFrame"),
    ])

    # Twilio serializer. You may not import this directly if create_transport
    # handles it, but its presence confirms the twilio extra is installed.
    probe("twilio serializer", [
        ("pipecat.serializers.twilio", "TwilioFrameSerializer"),
    ])

    print("-" * 70)
    print()

    # Check whether the service classes expose the nested Settings dataclass,
    # which replaced flat keyword arguments in recent versions.
    print("Checking settings style")
    print("-" * 70)
    service_checks = [
        ("pipecat.services.google.llm", "GoogleLLMService"),
        ("pipecat.services.elevenlabs.tts", "ElevenLabsTTSService"),
        ("pipecat.services.deepgram.stt", "DeepgramSTTService"),
    ]
    for module_path, class_name in service_checks:
        try:
            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)
        except Exception:
            print("  SKIP  {} not importable".format(class_name))
            continue
        if hasattr(service_class, "Settings"):
            print("  OK    {} has a nested Settings class".format(class_name))
            print("          use: {}(api_key=..., settings={}.Settings(...))".format(
                class_name, class_name))
        else:
            print("  NOTE  {} has NO nested Settings class".format(class_name))
            print("          use flat keyword arguments instead")

    print("-" * 70)
    print()
    print("Paste-ready import block for bot.py")
    print("-" * 70)
    ordered_imports = [
        vad_import,
        frame_import,
        pipeline_import,
        runner_import,
        params_import,
        task_import,
        context_import,
        aggregator_import,
        user_params_import,
        runner_types_import,
        create_transport_import,
        stt_import,
        llm_import,
        tts_import,
    ]
    for import_line in ordered_imports:
        if import_line is not None:
            print(import_line)
    print("-" * 70)
    print()

    failure_count = sum(1 for import_line in ordered_imports if import_line is None)
    if failure_count == 0:
        print("All probes resolved. bot.py as written should import cleanly.")
    else:
        print("{} probe(s) failed.".format(failure_count))
        print("Most likely cause: a missing extra. Check that you installed")
        print('pipecat-ai with [deepgram,google,elevenlabs,silero,websocket,runner].')


if __name__ == "__main__":
    main()
