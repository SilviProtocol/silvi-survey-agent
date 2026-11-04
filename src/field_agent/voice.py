"""Pipecat voice pipeline for the field agent (optional ``voice`` extra).

Pipecat 1.14 already ships ``MetaSTTService`` for Muse Voice Transcribe and
OpenAI-compatible LLM services for Baseten and Fireworks, so this module
adapts and configures upstream rather than reimplementing transports:

- ``FieldMetaSTTService``: upstream streaming STT, fixed to ENDPOINTING
  (model-detected end of speech), biased with survey keywords, gated and
  metered by the spend guard.
- ``FieldAgentProcessor``: turns final transcripts into checked wire replies
  through :class:`field_agent.agent.FieldAgent` and pushes speech downstream
  as text frames. No TTS here; "start of reply" is the first speech frame.
- ``build_pipecat_llm_service``: upstream Pipecat LLM services for each
  selectable model, for the later full voice stack. The pipeline here routes
  through FieldAgent instead, so the spend guard and tool-call checker always
  apply.

Replay has no audio, so nothing here invents any: without live input the STT
service simply has nothing to transcribe.
"""

from __future__ import annotations

import asyncio
import time

try:
    from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.baseten.llm import BasetenLLMService
    from pipecat.services.fireworks.llm import FireworksLLMService
    from pipecat.services.meta.stt import MetaSTTService
    from pipecat.services.openai.llm import OpenAILLMService
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "field_agent.voice needs the optional voice extra: uv sync --extra dev --extra voice"
    ) from exc

from field_agent.agent import FieldAgent
from field_agent.providers import ProviderRegistry
from field_agent.spend import SpendGuard

DEFAULT_KEYWORDS = [
    "Silvi",
    "waypoint",
    "crosshair",
    "seedling",
    "planting",
    "plot corner",
    "canopy",
    "survey",
]


class FieldMetaSTTService(MetaSTTService):
    """Muse Voice Transcribe with survey biasing and spend metering."""

    def __init__(
        self,
        *,
        api_key: str,
        spend_guard: SpendGuard | None = None,
        keywords: list[str] | None = None,
        **kwargs,
    ):
        settings = MetaSTTService.Settings(
            mode="ENDPOINTING",
            keywords=list(keywords) if keywords is not None else list(DEFAULT_KEYWORDS),
        )
        super().__init__(api_key=api_key, settings=settings, **kwargs)
        self._spend_guard = spend_guard
        self._audio_seconds_sent = 0.0
        self.last_speech_end_at: float | None = None

    @property
    def audio_seconds_sent(self) -> float:
        return self._audio_seconds_sent

    async def setup(self, setup):
        if self._spend_guard is not None:
            self._spend_guard.check("meta")  # refuse the session at the cap
        await super().setup(setup)

    async def run_stt(self, audio: bytes):
        if audio and self.sample_rate:
            # 16-bit mono PCM in: two bytes per sample.
            self._audio_seconds_sent += (len(audio) // 2) / float(self.sample_rate)
        async for frame in super().run_stt(audio):
            yield frame

    async def _handle_message(self, message: dict):
        if message.get("type") in ("speechEnd", "speechComplete"):
            self.last_speech_end_at = time.perf_counter()
        await super()._handle_message(message)

    async def _disconnect(self):
        try:
            if self._spend_guard is not None and self._audio_seconds_sent > 0:
                self._spend_guard.record_stt(audio_seconds=self._audio_seconds_sent)
                self._audio_seconds_sent = 0.0
        finally:
            await super()._disconnect()


class FieldAgentProcessor(FrameProcessor):
    """Final transcript -> checked wire replies -> speech text downstream."""

    def __init__(self, agent: FieldAgent, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text.strip()
        ):
            speech_end_at = time.perf_counter()
            await self.push_frame(frame, direction)
            try:
                outputs = await asyncio.to_thread(
                    self._agent.handle_turn, frame.text.strip(), speech_end_at=speech_end_at
                )
            except Exception as exc:  # noqa: BLE001 - surface as pipeline error, keep running
                await self.push_error(error_msg=f"field agent turn failed: {exc}", exception=exc)
                return
            for message in outputs:
                if message["kind"] == "speech":
                    await self.push_frame(TextFrame(message["payload"]["text"]))
            return
        await self.push_frame(frame, direction)


def build_stt_service(
    registry: ProviderRegistry,
    spend_guard: SpendGuard | None = None,
    *,
    keywords: list[str] | None = None,
) -> FieldMetaSTTService:
    return FieldMetaSTTService(
        api_key=registry.require_key("meta"), spend_guard=spend_guard, keywords=keywords
    )


def build_pipecat_llm_service(registry: ProviderRegistry, selector: str):
    """Upstream Pipecat LLM service for a selectable model (future stack)."""
    model = registry.model(selector)
    key = registry.require_key(model.provider)
    if model.provider == "baseten":
        return BasetenLLMService(
            api_key=key, settings=BasetenLLMService.Settings(model=model.model_id)
        )
    if model.provider == "fireworks":
        return FireworksLLMService(
            api_key=key, settings=FireworksLLMService.Settings(model=model.model_id)
        )
    # Meta Model API is OpenAI-compatible; Pipecat has no Meta LLM wrapper.
    return OpenAILLMService(
        api_key=key,
        base_url=registry.provider("meta").base_url,
        settings=OpenAILLMService.Settings(model=model.model_id),
    )


class _TranscriptCollector(FrameProcessor):
    """Sink that keeps the words a streamed transcription produced."""

    def __init__(self, heard: _Heard, **kwargs):
        super().__init__(**kwargs)
        self._heard = heard

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import InterimTranscriptionFrame

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._heard.final_text = frame.text.strip()
            self._heard.final_event.set()
        elif isinstance(frame, InterimTranscriptionFrame) and frame.text.strip():
            self._heard.last_partial = frame.text.strip()
        await self.push_frame(frame, direction)


class _Heard:
    """Words heard so far, shared between the pipeline and finalise."""

    def __init__(self):
        import threading

        self.final_text: str | None = None
        self.last_partial: str | None = None
        self.final_event = threading.Event()


_END = object()
_CANCEL = object()


class MetaBufferedTranscriber:
    """One push-to-talk utterance, transcribed while it is spoken.

    The speech pipeline starts at construction (the press), so
    connecting overlaps the hold. Frames are fed as they arrive;
    anything that shows up before the realtime service reports itself
    connected waits in the queue and goes out in order — the service
    drops audio sent earlier, and it finalises an utterance only when
    it hears trailing silence, so ``finalise`` sends a breath of PCM
    silence and returns on the first final transcription. No final
    inside the budget means the last partial stands, else "".

    The pipeline runs on its own thread; ``feed``/``finalise``/
    ``cancel`` are the synchronous surface the wire server drives.
    Any failure sounds like silence: an empty string, and the caller
    sends the empty final.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        spend_guard: SpendGuard | None = None,
        keywords: list[str] | None = None,
        sample_rate: int = 16000,
        service_factory=None,
        connect_timeout_s: float = 5.0,
        final_timeout_s: float = 4.0,
        silence_s: float = 1.2,
    ):
        import queue
        import threading

        self._api_key = api_key
        self._spend_guard = spend_guard
        self._keywords = list(keywords) if keywords else None
        self._sample_rate = sample_rate
        self._service_factory = service_factory
        self._connect_timeout_s = connect_timeout_s
        self._final_timeout_s = final_timeout_s
        self._silence_s = silence_s
        self._queue: queue.Queue = queue.Queue()
        self._heard = _Heard()
        self._cancelled = False
        self._finished = False
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="field-agent-stt", daemon=True
        )
        self._thread.start()

    # -- wire surface -----------------------------------------------------------
    def feed(self, data: bytes) -> None:
        if not self._cancelled and not self._finished and data:
            self._queue.put(bytes(data))

    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put(_CANCEL)
        self._thread.join(timeout=1.0)

    def finalise(self) -> str:
        if self._cancelled:
            return ""
        self._finished = True
        self._queue.put(_END)
        self._done.wait(timeout=self._final_timeout_s + self._connect_timeout_s + 3.0)
        self._thread.join(timeout=1.0)
        return self._heard.final_text or self._heard.last_partial or ""

    @property
    def closed(self) -> bool:
        return self._done.is_set() and not self._thread.is_alive()

    # -- pipeline thread ----------------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._amain())
        except Exception:  # noqa: BLE001, S110 - silence over a crashed speech path
            pass
        finally:
            self._done.set()

    def _make_service(self):
        if self._service_factory is not None:
            return self._service_factory(self._keywords)
        return FieldMetaSTTService(
            api_key=self._api_key, spend_guard=self._spend_guard, keywords=self._keywords
        )

    async def _amain(self) -> None:
        from pipecat.frames.frames import EndFrame, InputAudioRawFrame
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask

        service = self._make_service()
        connected = asyncio.Event()
        register = getattr(service, "add_event_handler", None)
        if callable(register):
            register("on_connected", lambda *a: connected.set())
        else:  # a service with nothing to connect is ready at once
            connected.set()
        task = PipelineTask(
            Pipeline([service, _TranscriptCollector(self._heard)]),
            params=PipelineParams(
                audio_in_sample_rate=self._sample_rate,
                audio_out_sample_rate=self._sample_rate,
            ),
        )
        # This runs on its own thread, where signal handlers are not
        # ours to install.
        runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
        loop = asyncio.get_running_loop()
        try:
            # Hold every frame until the service is connected; it drops
            # audio sent before that. A service that never connects
            # still gets its chance after the timeout.
            try:
                await asyncio.wait_for(connected.wait(), timeout=self._connect_timeout_s)
            except TimeoutError:
                pass
            while True:
                item = await loop.run_in_executor(None, self._queue.get)
                if item is _CANCEL:
                    break
                if item is _END:
                    await self._send_silence(task, InputAudioRawFrame)
                    # The endpoint fires on the silence; take the first
                    # final, but never wait past the budget.
                    try:
                        await loop.run_in_executor(
                            None, self._heard.final_event.wait, self._final_timeout_s
                        )
                    except Exception:  # noqa: BLE001, S110
                        pass
                    break
                await task.queue_frame(
                    InputAudioRawFrame(
                        audio=item, sample_rate=self._sample_rate, num_channels=1
                    )
                )
        finally:
            try:
                await task.queue_frame(EndFrame())
                await asyncio.wait_for(runner, timeout=5.0)
            except Exception:  # noqa: BLE001 - closing must not hang the utterance
                runner.cancel()

    async def _send_silence(self, task, frame_cls) -> None:
        total = int(self._sample_rate * 2 * self._silence_s)
        step = 1280  # 40 ms of PCM16 mono at 16 kHz
        sent = 0
        while sent < total:
            n = min(step, total - sent)
            await task.queue_frame(
                frame_cls(audio=b"\x00" * n, sample_rate=self._sample_rate, num_channels=1)
            )
            sent += n


def make_meta_stt_factory(registry: ProviderRegistry, spend_guard: SpendGuard | None = None):
    """A ``ServerCore`` stt_factory backed by Muse Voice Transcribe.

    Returns None when no Meta key is set, so the server simply never
    advertises ``audio_in`` and phones keep their on-device speech.
    """
    try:
        key = registry.require_key("meta")
    except Exception:  # noqa: BLE001 - no key, no server speech
        return None

    def factory(keywords: list[str]) -> MetaBufferedTranscriber:
        return MetaBufferedTranscriber(
            api_key=key, spend_guard=spend_guard, keywords=keywords or None
        )

    return factory


def build_pipeline(
    agent: FieldAgent,
    *,
    registry: ProviderRegistry | None = None,
    spend_guard: SpendGuard | None = None,
    include_stt: bool = False,
    keywords: list[str] | None = None,
    extra_processors: list | None = None,
) -> Pipeline:
    """Assemble [STT?] -> field agent -> [downstream processors]."""
    processors: list = []
    if include_stt:
        if registry is None:
            raise ValueError("include_stt needs a ProviderRegistry for the Meta key")
        processors.append(build_stt_service(registry, spend_guard, keywords=keywords))
    processors.append(FieldAgentProcessor(agent))
    processors.extend(extra_processors or [])
    return Pipeline(processors)
