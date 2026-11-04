"""Text-only chat clients for Spark, Spark Open and Glimmer.

All three providers expose OpenAI-compatible chat-completions endpoints
(Meta Model API, Baseten Model APIs, Fireworks inference). One client speaks
that shape; the registry supplies base URL, model id and prices, and the
spend guard gates and logs every call. Requests are rejected if any message
carries image, video or audio content: models get text and transcripts only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from field_agent.providers import ProviderRegistry
from field_agent.spend import SpendGuard

_NON_TEXT_PARTS = {
    "image",
    "image_url",
    "input_image",
    "video",
    "input_video",
    "audio",
    "input_audio",
    "file",
}

Transport = Callable[[str, str, dict, dict, float], tuple[int, dict]]
# Streaming transport: same request, returns (status, iterable of raw SSE lines).
StreamTransport = Callable[[str, str, dict, dict, float], tuple[int, object]]


class ProviderError(RuntimeError):
    """A provider call failed. Carries status, never a credential.

    ``status`` is the HTTP status when there was one, and None for a
    connection failure — fallback logic keys off it.
    """

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def assert_text_only(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            if content.startswith(("data:image", "data:video")):
                raise ValueError("non-text payload refused: data URI in message content")
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise TypeError("message content parts must be text objects")
                ptype = str(part.get("type", "text"))
                if ptype in _NON_TEXT_PARTS:
                    raise ValueError(f"non-text content part refused: {ptype}")
                if ptype != "text":
                    raise ValueError(f"unsupported content part refused: {ptype}")
            continue
        if content is not None:
            raise ValueError("message content must be text")


def _urllib_transport(method: str, url: str, headers: dict, body: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError, OSError):
            payload = {"error": {"message": str(exc)}}
        return exc.code, payload
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ProviderError(f"connection failed: {exc}", status=None) from exc


def _urllib_stream_transport(method: str, url: str, headers: dict, body: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json", "Accept": "text/event-stream"},
        method=method,
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc.code, [exc.read().decode("utf-8", "replace")]
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ProviderError(f"connection failed: {exc}", status=None) from exc

    def lines():
        with response:
            for raw in response:
                yield raw.decode("utf-8", "replace")

    return response.status, lines()


def provider_error_message(payload) -> str:
    """The provider's own words, whether ``error`` is a string or an object."""
    if not isinstance(payload, dict):
        return "request failed"
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "request failed"


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    usage_estimated: bool = False
    latency_ms: float = 0.0
    model_id: str = ""
    # Streaming only: ms from request start to the first token/tool delta.
    first_token_ms: float | None = None
    streamed: bool = False
    # The model spent its whole reply thinking (reasoning content only,
    # no visible text). Treated as an empty reply; the reasoning itself
    # is never carried here and never spoken.
    reasoning_only: bool = False


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class OpenAICompatibleChatClient:
    def __init__(
        self,
        registry: ProviderRegistry,
        selector: str,
        spend_guard: SpendGuard | None = None,
        *,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        timeout_s: float = 30.0,
    ):
        self._registry = registry
        self.selector = selector
        self._model = registry.model(selector)
        self._guard = spend_guard
        self._transport = transport or _urllib_transport
        self._stream_transport = stream_transport or _urllib_stream_transport
        self._timeout = timeout_s
        # Set when this provider answered 400 to a call carrying
        # reasoning_effort and the retry without it was the way through.
        self.effort_dropped = False

    def _require_available(self) -> None:
        if self._model.unavailable:
            provider = self._registry.provider(self._model.provider)
            raise ProviderError(
                f"{provider.display_name} model {self.selector} is unavailable: "
                f"{self._model.unavailable}"
            )

    def _body(self, messages, tools, max_tokens, *, with_effort: bool = True) -> dict:
        body: dict = {"model": self._model.model_id, "messages": messages}
        if tools:
            body["tools"] = tools
            # Meta supports only "auto"; it is also the valid default choice
            # on the OpenAI-compatible Baseten and Fireworks endpoints.
            body["tool_choice"] = "auto"
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if with_effort and not self.effort_dropped and self._model.reasoning_effort:
            body["reasoning_effort"] = self._model.reasoning_effort
        return body

    @property
    def model_id(self) -> str:
        return self._model.model_id

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        assert_text_only(messages)
        self._require_available()
        provider = self._registry.provider(self._model.provider)
        if self._guard is not None:
            self._guard.check(self._model.provider)
        key = self._registry.require_key(self._model.provider)

        body = self._body(messages, tools, max_tokens)
        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        started = time.perf_counter()
        status, payload = self._transport("POST", url, headers, body, self._timeout)
        if status == 400 and "reasoning_effort" in body:
            # This provider will not take the knob; one retry without it,
            # and no effort on later calls from this client either.
            self.effort_dropped = True
            body = self._body(messages, tools, max_tokens, with_effort=False)
            status, payload = self._transport("POST", url, headers, body, self._timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if status < 200 or status >= 300:
            raise ProviderError(
                f"{provider.display_name} HTTP {status}: {provider_error_message(payload)}",
                status=status,
            )

        result = self._parse(payload, messages, latency_ms)
        if self._guard is not None:
            self._guard.record_chat(
                selector=self.selector,
                input_tokens=result.input_tokens,
                cached_input_tokens=result.cached_input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=round(latency_ms, 1),
            )
        return result

    def _request_parts(self, messages, tools, max_tokens):
        assert_text_only(messages)
        self._require_available()
        provider = self._registry.provider(self._model.provider)
        if self._guard is not None:
            self._guard.check(self._model.provider)
        key = self._registry.require_key(self._model.provider)
        body = self._body(messages, tools, max_tokens)
        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        return provider, url, {"Authorization": f"Bearer {key}"}, body

    def _record(self, result: ChatResult, latency_ms: float) -> None:
        if self._guard is not None:
            self._guard.record_chat(
                selector=self.selector,
                input_tokens=result.input_tokens,
                cached_input_tokens=result.cached_input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=round(latency_ms, 1),
            )

    def complete_stream(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        """Stream one completion (SSE). ``on_token`` fires per content/tool
        delta as it arrives, so callers can timestamp the first token; the
        assembled result is returned at the end. Spend guard checks before
        the call and records after it — including a mid-stream failure,
        which is recorded with estimated usage before raising."""
        provider, url, headers, body = self._request_parts(messages, tools, max_tokens)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        status, lines = self._stream_transport("POST", url, headers, body, self._timeout)
        if status == 400 and "reasoning_effort" in body:
            self.effort_dropped = True
            body = self._body(messages, tools, max_tokens, with_effort=False)
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            status, lines = self._stream_transport("POST", url, headers, body, self._timeout)
        if status < 200 or status >= 300:
            text = "".join(str(line) for line in lines)  # error body, if any
            try:
                message = provider_error_message(json.loads(text))
            except ValueError:
                message = "request failed"
            raise ProviderError(f"{provider.display_name} HTTP {status}: {message}", status=status)

        text_parts: list[str] = []
        tool_slots: dict[int, dict] = {}
        usage: dict = {}
        model_id = self._model.model_id
        first_token_ms: float | None = None
        saw_reasoning = False

        def note_token(chunk: str) -> None:
            nonlocal first_token_ms
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - started) * 1000.0, 1)
            if on_token is not None:
                on_token(chunk)

        try:
            for raw_line in lines:
                line = str(raw_line).strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue  # keep-alive / partial line; never guess
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    # Some gateways deliver the failure as a 200 stream
                    # carrying an error event; the words still count.
                    raise ProviderError(
                        f"{provider.display_name} stream error: "
                        f"{provider_error_message(chunk)}",
                        status=None,
                    )
                if chunk.get("model"):
                    model_id = str(chunk["model"])
                if chunk.get("usage"):
                    usage = chunk["usage"] or {}
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") if choices else {}) or {}
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    saw_reasoning = True
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)
                    note_token(content)
                for frag in delta.get("tool_calls") or []:
                    idx = int(frag.get("index") or 0)
                    slot = tool_slots.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    if frag.get("id"):
                        slot["id"] = frag["id"]
                    fn = frag.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += str(fn["name"])
                    if fn.get("arguments"):
                        slot["arguments"] += str(fn["arguments"])
                    note_token(str(fn.get("arguments") or fn.get("name") or ""))
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            partial = ChatResult(
                text="".join(text_parts),
                input_tokens=sum(_estimate_tokens(str(m.get("content") or "")) for m in messages),
                output_tokens=_estimate_tokens("".join(text_parts)) if text_parts else 0,
                usage_estimated=True,
                latency_ms=round(latency_ms, 1),
                model_id=model_id,
                first_token_ms=first_token_ms,
                streamed=True,
            )
            self._record(partial, latency_ms)
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(
                f"{provider.display_name} stream failed mid-reply: {exc}", status=None
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(text_parts)
        tool_calls = [tool_slots[i] for i in sorted(tool_slots)]
        estimated = "prompt_tokens" not in usage
        if estimated:
            input_tokens = sum(_estimate_tokens(str(m.get("content") or "")) for m in messages)
            output_tokens = _estimate_tokens(text + "".join(t["arguments"] for t in tool_calls))
            cached = 0
        else:
            details = usage.get("prompt_tokens_details") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            cached = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
        result = ChatResult(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            usage_estimated=estimated,
            latency_ms=round(latency_ms, 1),
            model_id=model_id,
            first_token_ms=first_token_ms,
            streamed=True,
            reasoning_only=bool(saw_reasoning and not text and not tool_calls),
        )
        self._record(result, latency_ms)
        return result

    def _parse(self, payload: dict, messages: list[dict], latency_ms: float) -> ChatResult:
        if isinstance(payload, dict) and payload.get("error"):
            # A 200 that carries an error object is still an error, and
            # its words are the ones worth showing.
            provider = self._registry.provider(self._model.provider)
            raise ProviderError(
                f"{provider.display_name} error: {provider_error_message(payload)}",
                status=None,
            )
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        text = message.get("content") or ""
        if not isinstance(text, str):
            text = "".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
        tool_calls: list[dict] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            tool_calls.append(
                {
                    "id": raw.get("id"),
                    "name": fn.get("name") or raw.get("name", ""),
                    "arguments": fn.get("arguments", raw.get("arguments", "{}")),
                }
            )
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        estimated = "prompt_tokens" not in usage
        if estimated:
            input_tokens = sum(_estimate_tokens(str(m.get("content") or "")) for m in messages)
            output_tokens = _estimate_tokens(text)
            cached = 0
        else:
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            cached = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
        reasoning_only = bool(
            not text
            and not tool_calls
            and (message.get("reasoning_content") or message.get("reasoning"))
        )
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            usage_estimated=estimated,
            latency_ms=round(latency_ms, 1),
            model_id=str(payload.get("model") or self._model.model_id),
            reasoning_only=reasoning_only,
        )


class MockChatClient:
    """Deterministic stand-in used whenever no key is available.

    Give it a queue of ChatResults / dicts, or a handler(messages, tools).
    It enforces the same text-only rule as the live client and records calls.
    """

    def __init__(self, responses=None, *, handler=None, model_id: str = "mock"):
        self._responses = list(responses or [])
        self._handler = handler
        self.model_id = model_id
        self.selector = "mock"
        self.calls: list[dict] = []

    def complete(self, messages, *, tools=None, max_tokens=None) -> ChatResult:
        assert_text_only(messages)
        self.calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens})
        if self._handler is not None:
            out = self._handler(messages, tools)
            return out if isinstance(out, ChatResult) else ChatResult(**out)
        if not self._responses:
            raise AssertionError("MockChatClient ran out of scripted responses")
        out = self._responses.pop(0)
        return out if isinstance(out, ChatResult) else ChatResult(**out)

    def complete_stream(self, messages, *, tools=None, max_tokens=None, on_token=None):
        result = self.complete(messages, tools=tools, max_tokens=max_tokens)
        if on_token is not None:
            chunks = [result.text] if result.text else [
                str(c.get("arguments") or "") for c in result.tool_calls
            ]
            for chunk in chunks:
                if chunk:
                    on_token(chunk)
        result.streamed = True
        result.first_token_ms = 0.0
        return result


_FALLBACK_STATUSES = {402, 404}


def _worth_falling_back(exc: BaseException) -> bool:
    if not isinstance(exc, ProviderError):
        return False
    return exc.status is None or exc.status in _FALLBACK_STATUSES or exc.status >= 500


class FallbackChatClient:
    """One retry on a second seat when the first cannot be reached.

    Payment, not-found, server and connection failures move the call to
    the fallback client once; anything else (a 400 the caller caused,
    say) surfaces as-is. ``fallback_events`` records each hop so the
    agent can note it in the turn trace.
    """

    def __init__(self, primary, fallback, *, primary_selector: str, fallback_selector: str):
        self._primary = primary
        self._fallback = fallback
        self.primary_selector = primary_selector
        self.fallback_selector = fallback_selector
        self.fallback_events: list[dict] = []

    def _hop(self, exc: BaseException) -> None:
        self.fallback_events.append(
            {"from": self.primary_selector, "to": self.fallback_selector, "reason": str(exc)}
        )

    def complete(self, messages, *, tools=None, max_tokens=None):
        try:
            return self._primary.complete(messages, tools=tools, max_tokens=max_tokens)
        except ProviderError as exc:
            if not _worth_falling_back(exc):
                raise
            self._hop(exc)
            return self._fallback.complete(messages, tools=tools, max_tokens=max_tokens)

    def complete_stream(self, messages, *, tools=None, max_tokens=None, on_token=None):
        try:
            return self._primary.complete_stream(
                messages, tools=tools, max_tokens=max_tokens, on_token=on_token
            )
        except ProviderError as exc:
            if not _worth_falling_back(exc):
                raise
            self._hop(exc)
            return self._fallback.complete_stream(
                messages, tools=tools, max_tokens=max_tokens, on_token=on_token
            )


def make_chat_client(
    registry: ProviderRegistry,
    selector: str,
    spend_guard: SpendGuard | None = None,
    *,
    mock: bool = False,
    mock_handler=None,
    transport: Transport | None = None,
    stream_transport: StreamTransport | None = None,
):
    """Build the client for a selectable model. ``mock=True`` never networks."""
    if mock:
        return MockChatClient(handler=mock_handler, model_id=registry.model(selector).model_id)
    info = registry.model(selector)
    if info.unavailable:
        raise ProviderError(
            f"model {selector} is unavailable: {info.unavailable}"
        )
    primary = OpenAICompatibleChatClient(
        registry, selector, spend_guard, transport=transport, stream_transport=stream_transport
    )
    if info.fallback and info.fallback != selector:
        fallback_info = registry.model(info.fallback)
        if not fallback_info.unavailable:
            fallback = OpenAICompatibleChatClient(
                registry,
                info.fallback,
                spend_guard,
                transport=transport,
                stream_transport=stream_transport,
            )
            return FallbackChatClient(
                primary,
                fallback,
                primary_selector=selector,
                fallback_selector=info.fallback,
            )
    return primary
