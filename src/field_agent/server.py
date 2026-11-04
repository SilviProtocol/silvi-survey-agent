"""Live agent server: the phone's endpoint.

A WebSocket server speaking the wire contract. Each frame is one JSON
envelope (a frame may also carry several newline-separated envelopes).
Phone -> agent messages feed one ``FieldAgent`` per session; the agent's
validated replies return on the same socket.

Sequence rules (full text in docs/PHONE-CLIENT.md):

- Each direction has its own data seq space, starting at 0.
- Control kinds (resume / resumed / error) do not consume a seq; their
  envelope seq is the sender's next data seq.
- A dropped connection loses nothing: the session (agent, last contiguous
  client seq, every output sent) survives the socket. The phone reconnects,
  sends ``resume`` with the last agent seq it received, and the server
  answers ``resumed`` plus every missed output, in order. The phone then
  resends anything after the server's last client seq; duplicates at or
  below it are dropped, never re-fed to the agent.

The transport-free core (``ServerCore`` / ``Connection``) is synchronous and
carries all protocol decisions, so tests can drive it without a socket.
The asyncio wrapper below only moves bytes, one message at a time, which
keeps ordering exact and lets TCP/WebSocket buffering do backpressure.

Binds 127.0.0.1 by default. A non-loopback bind requires an explicit LAN
opt-in in the CLI: there is no authentication or TLS in v1.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from jsonschema import ValidationError

from field_agent.contracts import validate_wire

CONTROL_KINDS = {
    "resume",
    "resumed",
    "error",
    "capabilities",
    "voice_start",
    "voice_end",
    "voice_cancel",
}
CLIENT_DATA_KINDS = {
    "session_start",
    "pose",
    "plant_sighting",
    "plant_snapshot",
    "coverage",
    "transcript",
    "clip_ref",
    "session_end",
}
# Wire v2: accepted only on a connection the server has already sent
# ``capabilities {wire: 2}`` on. Until then the phone speaks v1 only,
# so a v2 kind arriving early is a protocol error, never a surprise.
CLIENT_V2_KINDS = {"anchor", "mode", "evidence_photo", "plant_view"}
AGENT_DATA_KINDS = {
    "speech",
    "waypoint",
    "polygon",
    "gap",
    "evidence_request",
    "view_request",
    "tree_status",
}
AGENT_V2_KINDS = {"evidence_request", "view_request", "tree_status"}
VOICE_CONTROL_KINDS = {"voice_start", "voice_end", "voice_cancel"}


def _voice_unconfigured(core: ServerCore, conn: Connection, message: dict):
    return core._error(
        conn,
        "voice_not_configured",
        "server speech is not enabled on this server",
        session_id=message["session_id"],
        mono=float(message["t_mono_ms"]),
    )

MAX_FRAME_BYTES = 4 * 1024 * 1024
DEFAULT_SESSION_TTL_S = 3600.0

# WebSocket close codes this server uses (documented in PHONE-CLIENT.md).
CLOSE_BAD_DATA = 1003  # not JSON, or not the wire envelope
CLOSE_POLICY = 1008  # protocol violation (seq gap, wrong first message, ...)
CLOSE_SUPERSEDED = 4001  # a newer connection resumed this session


@dataclass
class Session:
    session_id: str
    agent: object  # FieldAgent
    wire_v2: bool = False
    last_client_seq: int = -1
    duplicates_dropped: int = 0
    ended: bool = False
    active: Connection | None = None
    last_seen_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def next_agent_seq(self) -> int:
        return self.agent.next_output_seq

    def outputs_after(self, last_agent_seq: int) -> list[dict]:
        outputs = self.agent.outputs
        return [m for m in outputs if m["seq"] > last_agent_seq]


class Connection:
    """One socket's binding to at most one session. Transport-free."""

    def __init__(self, core: ServerCore):
        self._core = core
        self.session: Session | None = None
        self.superseded = False
        self._last_mono = 0.0
        # The v2 gate is per connection: it opens when this server sends
        # capabilities on this socket and closes with the socket.
        self.wire2_open = False
        # Server speech: at most one streamed utterance per connection.
        self._voice: dict | None = None
        self.stray_binary_frames = 0

    def feed_binary(self, data: bytes) -> None:
        """One binary frame of PCM for the open utterance, if any."""
        self._core._feed_binary(self, data)

    def feed_line(self, line: str) -> tuple[list[dict], int | None]:
        """Process one raw envelope. Returns (replies, close_code | None)."""
        return self._core._feed_line(self, line)


class ServerCore:
    def __init__(
        self,
        agent_factory: Callable[[], object],
        *,
        session_ttl_s: float = DEFAULT_SESSION_TTL_S,
        clock=time.monotonic,
        wire_v2: bool = True,
        audio_in: dict | None = None,
        stt_factory: Callable[[list[str]], object] | None = None,
    ):
        self._factory = agent_factory
        self._ttl = session_ttl_s
        self._clock = clock
        self._wire_v2 = wire_v2
        self._stt_factory = stt_factory
        if audio_in is None and stt_factory is not None:
            audio_in = {"sample_rate": 16000, "encoding": "pcm_s16le", "channels": 1}
        self._audio_in = dict(audio_in) if audio_in else None
        self.sessions: dict[str, Session] = {}

    def _capabilities(self, conn: Connection, session: Session, mono: float) -> dict | None:
        """Open this connection's v2 gate; None when v2 is disabled."""
        if not self._wire_v2:
            return None
        conn.wire2_open = True
        session.wire_v2 = True
        agent = session.agent
        if hasattr(agent, "wire_v2"):
            agent.wire_v2 = True
        payload: dict = {"wire": 2, "max_frame_bytes": MAX_FRAME_BYTES}
        if self._audio_in is not None:
            payload["audio_in"] = dict(self._audio_in)
        return self._control("capabilities", session, session.session_id, mono, payload)

    def new_connection(self) -> Connection:
        return Connection(self)

    def disconnect(self, conn: Connection) -> None:
        # A dropped socket drops any open utterance with it; binary
        # audio is never resent, so there is nothing to resume.
        self._cancel_voice(conn)
        session = conn.session
        if session is not None and session.active is conn:
            session.active = None
            session.last_seen_at = self._clock()
        conn.session = None

    # -- helpers ------------------------------------------------------------

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            sid
            for sid, s in self.sessions.items()
            if s.active is None and now - s.last_seen_at > self._ttl
        ]
        for sid in expired:
            del self.sessions[sid]

    def _control(
        self, kind: str, session: Session | None, session_id: str, mono: float, payload: dict
    ) -> dict:
        message = {
            "v": "1",
            "kind": kind,
            "seq": session.next_agent_seq if session is not None else 0,
            "session_id": session_id,
            "t_mono_ms": float(mono),
            "t_wall_ms": None,
            "payload": payload,
        }
        validate_wire(message)
        return message

    def _error(
        self,
        conn: Connection,
        code: str,
        text: str,
        *,
        close_code: int = CLOSE_POLICY,
        expected_seq: int | None = None,
        session_id: str | None = None,
        mono: float = 0.0,
    ) -> tuple[list[dict], int]:
        session = conn.session
        sid = session_id or (session.session_id if session is not None else "unknown")
        payload: dict = {"code": code, "message": text}
        if expected_seq is not None:
            payload["expected_seq"] = expected_seq
        return [self._control("error", session, sid, mono, payload)], close_code

    def _bind(self, conn: Connection, session: Session) -> None:
        old = session.active
        if old is not None and old is not conn:
            old.superseded = True
            old.session = None
        session.active = conn
        session.last_seen_at = self._clock()
        conn.session = session
        conn.superseded = False

    # -- server speech ---------------------------------------------------------
    #
    # The phone streams one utterance as voice_start, binary PCM16
    # frames, then voice_end (the release ends the turn; no waiting on
    # endpointing) or voice_cancel. The controls carry no data seq.
    # A transcriber is anything with feed(bytes), finalise() -> str and
    # cancel(); production wires Muse Voice Transcribe, tests a fake.

    def _cancel_voice(self, conn: Connection) -> None:
        voice = conn._voice
        conn._voice = None
        if voice is not None:
            try:
                voice["stt"].cancel()
            except Exception:  # noqa: BLE001, S110 - a dead transcriber must not kill the link
                pass

    def _feed_binary(self, conn: Connection, data: bytes) -> None:
        voice = conn._voice
        if voice is None or not conn.wire2_open:
            # Binary outside an open utterance (or before capabilities)
            # is a protocol error, but a soft one: log and drop, keep
            # the socket.
            conn.stray_binary_frames += 1
            return
        if voice["bytes"] >= voice["cap_bytes"]:
            voice["truncated"] = True
            return
        take = bytes(data[: voice["cap_bytes"] - voice["bytes"]])
        voice["bytes"] += len(take)
        voice["stt"].feed(take)

    def _voice_control(self, conn: Connection, message: dict):
        session = conn.session
        assert session is not None
        mono = float(message["t_mono_ms"])
        kind = message["kind"]
        payload = message["payload"]
        if not conn.wire2_open:
            return self._error(
                conn,
                "wire_v2_not_open",
                f"{kind} is a wire v2 control; it is accepted only after capabilities",
                session_id=session.session_id,
                mono=mono,
            )
        if kind == "voice_cancel":
            if conn._voice is not None and conn._voice["id"] == payload["utterance_id"]:
                self._cancel_voice(conn)
            return [], None
        if self._stt_factory is None:
            return _voice_unconfigured(self, conn, message)
        if kind == "voice_start":
            # A new press replaces a hold still waiting.
            self._cancel_voice(conn)
            keywords: list[str] = []
            agent = session.agent
            if hasattr(agent, "stt_keywords"):
                keywords = list(agent.stt_keywords())
            rate = int(payload.get("sample_rate") or 16000)
            conn._voice = {
                "id": str(payload["utterance_id"]),
                "stt": self._stt_factory(keywords),
                "focus": payload.get("focus"),
                "bytes": 0,
                "cap_bytes": rate * 2 * 30,  # one utterance, 30 s of PCM16
                "truncated": False,
            }
            return [], None
        # voice_end: flush and finalise now, whatever was heard.
        voice = conn._voice
        if voice is None or voice["id"] != payload["utterance_id"]:
            return [], None
        conn._voice = None
        try:
            text = str(voice["stt"].finalise() or "")
        except Exception:  # noqa: BLE001 - a failed transcription is an empty final, not a dead link
            text = ""
        agent = session.agent
        with session._lock:
            before = len(agent.outputs)
            agent.handle_server_transcript(voice["id"], text, focus=voice.get("focus"))
            return list(agent.outputs[before:]), None

    def _feed_agent(self, session: Session, message: dict) -> list[dict]:
        before = len(session.agent.outputs)
        session.agent(message)
        return list(session.agent.outputs[before:])

    # -- protocol -------------------------------------------------------------

    def _feed_line(self, conn: Connection, line: str) -> tuple[list[dict], int | None]:
        try:
            message = json.loads(line)
        except (ValueError, TypeError):
            return self._error(
                conn, "bad_json", "frame is not one JSON object", close_code=CLOSE_BAD_DATA
            )
        if not isinstance(message, dict):
            return self._error(
                conn, "bad_json", "frame is not a JSON object", close_code=CLOSE_BAD_DATA
            )
        try:
            validate_wire(message)
        except ValidationError as exc:
            return self._error(
                conn,
                "invalid_message",
                f"wire validation failed: {exc.message}",
                close_code=CLOSE_BAD_DATA,
                session_id=str(message.get("session_id") or "unknown"),
            )

        kind = message["kind"]
        mono = float(message["t_mono_ms"])
        conn._last_mono = mono
        if kind in AGENT_DATA_KINDS or kind in ("resumed", "error", "capabilities"):
            return self._error(
                conn,
                "wrong_direction",
                f"{kind} travels agent -> phone only",
                session_id=message["session_id"],
                mono=mono,
            )

        if conn.session is None:
            return self._first_message(conn, message)
        return self._bound_message(conn, message)

    def _first_message(self, conn: Connection, message: dict) -> tuple[list[dict], int | None]:
        self._purge_expired()
        kind = message["kind"]
        sid = message["session_id"]
        mono = float(message["t_mono_ms"])
        if kind == "session_start":
            if sid in self.sessions:
                return self._error(
                    conn,
                    "session_exists",
                    "session already live; reconnect with resume or use a new session_id",
                    session_id=sid,
                    mono=mono,
                )
            if int(message["seq"]) != 0:
                return self._error(
                    conn,
                    "seq_gap",
                    "a fresh session starts at client seq 0",
                    expected_seq=0,
                    session_id=sid,
                    mono=mono,
                )
            session = Session(session_id=sid, agent=self._factory(), last_seen_at=self._clock())
            self.sessions[sid] = session
            self._bind(conn, session)
            with session._lock:
                replies = self._feed_agent(session, message)
                session.last_client_seq = 0
            caps = self._capabilities(conn, session, mono)
            return ([caps] if caps is not None else []) + replies, None
        if kind == "resume":
            session = self.sessions.get(sid)
            if session is None:
                return self._error(
                    conn,
                    "unknown_session",
                    "no live session with that session_id; start a new session",
                    session_id=sid,
                    mono=mono,
                )
            bad = self._check_resume(conn, session, message)
            if bad is not None:
                return bad
            self._bind(conn, session)
            return self._resume_replies(conn, session, message), None
        return self._error(
            conn,
            "expected_session_start",
            "first message on a connection must be session_start or resume",
            session_id=sid,
            mono=mono,
        )

    def _check_resume(self, conn: Connection, session: Session, message: dict):
        last_agent_seq = message["payload"].get("last_agent_seq")
        last_agent_seq = -1 if last_agent_seq is None else int(last_agent_seq)
        if last_agent_seq >= session.next_agent_seq:
            return self._error(
                conn,
                "resume_ahead",
                "last_agent_seq is beyond this session's outputs",
                session_id=session.session_id,
                mono=float(message["t_mono_ms"]),
            )
        return None

    def _resume_replies(self, conn: Connection, session: Session, message: dict) -> list[dict]:
        mono = float(message["t_mono_ms"])
        last_agent_seq = message["payload"].get("last_agent_seq")
        last_agent_seq = -1 if last_agent_seq is None else int(last_agent_seq)
        resumed = self._control(
            "resumed",
            session,
            session.session_id,
            mono,
            {
                "last_client_seq": session.last_client_seq,
                "next_agent_seq": session.next_agent_seq,
            },
        )
        # A lost session closed the phone's v2 gate; reopen it here,
        # before any replayed output.
        caps = self._capabilities(conn, session, mono)
        return [resumed] + ([caps] if caps is not None else []) + session.outputs_after(last_agent_seq)

    def _bound_message(self, conn: Connection, message: dict) -> tuple[list[dict], int | None]:
        session = conn.session
        assert session is not None
        kind = message["kind"]
        mono = float(message["t_mono_ms"])
        if message["session_id"] != session.session_id:
            return self._error(
                conn,
                "session_mismatch",
                "every message on a connection carries its bound session_id",
                session_id=session.session_id,
                mono=mono,
            )
        if kind == "resume":
            bad = self._check_resume(conn, session, message)
            if bad is not None:
                return bad
            return self._resume_replies(conn, session, message), None
        if kind in CLIENT_V2_KINDS and not conn.wire2_open:
            return self._error(
                conn,
                "wire_v2_not_open",
                f"{kind} is a wire v2 kind; it is accepted only after capabilities",
                session_id=session.session_id,
                mono=mono,
            )
        if kind in VOICE_CONTROL_KINDS:
            return self._voice_control(conn, message)
        if kind == "session_start":
            return self._error(
                conn,
                "unexpected_session_start",
                "session_start is only the first message of a new session",
                session_id=session.session_id,
                mono=mono,
            )
        seq = int(message["seq"])
        with session._lock:
            if seq <= session.last_client_seq:
                session.duplicates_dropped += 1
                return [], None
            if seq != session.last_client_seq + 1:
                return self._error(
                    conn,
                    "seq_gap",
                    "client data seq must increase by one per message",
                    expected_seq=session.last_client_seq + 1,
                    session_id=session.session_id,
                    mono=mono,
                )
            replies = self._feed_agent(session, message)
            session.last_client_seq = seq
            session.last_seen_at = self._clock()
            if kind == "session_end":
                session.ended = True
        return replies, None


# -- asyncio WebSocket wrapper -------------------------------------------------


def _dumps(message: dict) -> str:
    return json.dumps(message, separators=(",", ":"))


async def _handle_socket(ws, core: ServerCore, sockets_by_session: dict) -> None:
    from websockets.exceptions import ConnectionClosed

    conn = core.new_connection()
    bound_sid: str | None = None
    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                # Binary frames are streamed utterance audio (PCM16),
                # never JSON. Anything stray is dropped in the core.
                await asyncio.to_thread(conn.feed_binary, raw)
                continue
            for line in str(raw).splitlines():
                if not line.strip():
                    continue
                replies, close_code = await asyncio.to_thread(conn.feed_line, line)
                if conn.session is not None and conn.session.session_id != bound_sid:
                    bound_sid = conn.session.session_id
                    old = sockets_by_session.get(bound_sid)
                    sockets_by_session[bound_sid] = ws
                    if old is not None and old is not ws:
                        await old.close(code=CLOSE_SUPERSEDED, reason="session resumed elsewhere")
                for reply in replies:
                    await ws.send(_dumps(reply))
                if close_code is not None:
                    await ws.close(code=close_code)
                    return
                if conn.superseded:
                    await ws.close(code=CLOSE_SUPERSEDED, reason="session resumed elsewhere")
                    return
    except (ConnectionClosed, OSError):
        pass  # the phone went away; the session survives for resume
    finally:
        core.disconnect(conn)
        if bound_sid is not None and sockets_by_session.get(bound_sid) is ws:
            del sockets_by_session[bound_sid]


async def serve_agent(host: str | None, port: int | None, core: ServerCore, **serve_kwargs):
    """Start the WebSocket server. Returns the websockets server object.

    Extra kwargs pass to ``websockets.asyncio.server.serve`` (e.g. a Unix
    socket ``path=`` with ``host=port=None``, used by tests where TCP bind
    is unavailable).
    """
    from websockets.asyncio.server import serve, unix_serve

    sockets_by_session: dict = {}

    async def handler(ws):
        await _handle_socket(ws, core, sockets_by_session)

    common = {"max_size": MAX_FRAME_BYTES, "ping_interval": 20, "ping_timeout": 20}
    if "path" in serve_kwargs:
        return await unix_serve(handler, **common, **serve_kwargs)
    return await serve(handler, host, port, **common, **serve_kwargs)


def run_server(host: str, port: int, core: ServerCore) -> None:
    """Blocking entry point used by the CLI."""

    async def main() -> None:
        server = await serve_agent(host, port, core)
        bound = ", ".join(str(s.getsockname()) for s in server.sockets or [])
        print(f"field-agent server listening on {bound} (Ctrl-C to stop)", flush=True)
        await asyncio.Future()  # serve until cancelled / Ctrl-C

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
