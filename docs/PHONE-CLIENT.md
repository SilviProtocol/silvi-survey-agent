# Phone client: talking to the field-agent server

For whoever builds the phone side. You do not need to read the server.
Types: `bridge/field-agent.d.ts` (generated from `schemas/wire.schema.json`).

## Connection

- WebSocket, one JSON envelope per text frame. A frame may also carry
  several newline-separated envelopes; the server answers one per frame.
- `field-agent serve` binds `127.0.0.1:8766` by default. LAN use needs
  `--allow-lan --host 0.0.0.0` on the server. There is **no auth or TLS in
  v1**: only run LAN mode on a trusted network.
- Max frame 4 MiB. Server pings every 20 s; answer pings (any ws library
  does) or the socket is dropped — that is safe, see Reconnects.
- Text only over this socket too: no image/audio payloads in any field.

## Messages

Envelope: `{v:"1", kind, seq, session_id, t_mono_ms, t_wall_ms, payload}`.
`t_mono_ms` is your monotonic clock; `t_wall_ms` wall time or `null`.
Positions are site-local metres unless a geo anchor is present.

Phone -> agent (data): `session_start`, `pose`, `plant_sighting`,
`plant_snapshot`, `coverage`, `transcript`, `clip_ref`, `session_end`.
Agent -> phone (data): `speech`, `waypoint`, `polygon`, `gap`.
Control (either side, see below): `resume`, `resumed`, `error`.
Sending an agent-direction kind to the server is an error (`wrong_direction`).

## Sequence numbers

- Each direction has its own data seq space, starting at 0, +1 per message.
- Your first message on the first socket is `session_start` with seq 0.
  Pick a `session_id` unique per walk and keep it for the whole walk.
- Control kinds do **not** consume a seq. Their envelope `seq` carries the
  sender's *next* data seq (informational; the payload fields govern).
- A gap in your seq is fatal to the socket: server sends `error`
  (`seq_gap`, with `expected_seq`) and closes with 1008. Fix by resending
  from `expected_seq` on a resumed socket — or never gap: keep an outbox.
- Duplicates (seq at or below what the server has processed) are silently
  dropped, never re-fed to the agent. Resending is always safe.

## Reconnects (dropping loses nothing)

The session — agent state, last contiguous client seq, every output sent —
survives the socket (default 1 h after last contact).

1. Keep every sent data message in an outbox indexed by seq, and remember
   the highest agent data seq you have received (`last_agent_seq`, -1 if none).
2. On a new socket, send as the **first** message:
   `resume` with payload `{"last_agent_seq": N}` (or `null` for none).
3. Server replies `resumed` with `{"last_client_seq": M, "next_agent_seq": K}`,
   then replays every agent output with seq > N, in order.
4. Resend your outbox entries with seq > M, then carry on. Ignore any
   received data seq you already have.
5. Only one live socket per session: resuming closes the old one (4001).
6. `resume` for an unknown/expired session gets `error`
   (`unknown_session`); start a new session with a new `session_id`.
   `session_start` for a live session gets `session_exists`; resume instead.
7. Use capped exponential backoff between attempts (client uses
   0.4 s doubling to 2 s, 5 tries); the server imposes no rate limit in v1.

## Backpressure

- The server processes one message at a time per socket, in order, and
  writes replies before reading the next frame. If you send faster than
  the agent thinks (a transcript turn can take ~1 s), your send buffer
  fills; that *is* the flow control. Do not drop queued poses silently —
  if you must shed load, shed before assigning a seq, never after.
- Never renumber. A message that has a seq keeps it for every resend.
- Agent replies are few and small; read continuously so the server's
  writes never block on you.

## Errors and close codes

`error` payload: `{code, message, expected_seq?}`; the server closes after
sending it. Codes: `bad_json` (1003), `invalid_message` (1003, schema),
`seq_gap`, `expected_session_start`, `unexpected_session_start`,
`session_exists`, `unknown_session`, `session_mismatch`, `wrong_direction`,
`resume_ahead` (all 1008), superseded socket (4001, no error frame).

## Minimal session

```
-> session_start seq 0
-> pose seq 1 ... transcript seq N
<- speech seq 0 (agent direction)
-> session_end seq N+1
```

## Wire v2: the version handshake

Everything above is v1 and keeps working unchanged. v2 adds kinds
(`anchor`, `mode`, `evidence_photo`, `plant_view` from the phone;
`evidence_request`, `view_request`, `tree_status` from the agent) and
fields (`plant_snapshot.display_no`, `transcript.focus`, `pose.geo`).
They travel only after a handshake, so a new phone survives an old
server and an old phone never hears a v2 kind:

- Right after `session_start` binds, and again right after `resumed`
  (before any replayed output), the server sends one control message:
  `capabilities {wire: 2, max_frame_bytes}`. Like the other controls
  it spends no agent data seq; its envelope seq is the next one.
- Until you have seen `capabilities` on this socket, send v1 only.
  A v2 kind that arrives earlier is a protocol error
  (`wire_v2_not_open`, 1008). When the gate opens, re-send every plant
  snapshot with `display_no`, then the current `mode`.
- A dropped socket closes the gate. The server reopens it on the next
  connection's `resumed`.
- `display_no` is the number drawn on screen; speech always uses it,
  and a merge keeps the lower number. `transcript.focus` and
  `voice_start.focus` name the crosshair plant for that utterance and
  win over the server's computed crosshair for that turn.

### Evidence and the two rings

`evidence_photo {plant_id, keyframe_seq, distance_m, jpeg, request_id?}`
(base64 JPEG, long side ≤ 1280 px, ≤ 400 KB) goes to the species
service behind the server's plug; photos never enter the model's
messages. `evidence_request {request_id, plant_id, want}` (`want` one
of bark, leaf, whole, flower_or_fruit, base, side) asks for the next
shot; echo its `request_id` on the answering photo. `view_request
{plant_id, max_px}` is answered by `plant_view` (a crop) from a
keyframe that sees the plant. `tree_status {trees: [{plant_id,
number, what_ring, short_name}]}` carries species-ring changes
only — at most two batches a second, one per tree per second, names
at most 18 characters. The what ring is the species service's
confidence.

### Speech on the server

When the server can transcribe, `capabilities` also carries
`audio_in {sample_rate: 16000, encoding: "pcm_s16le", channels: 1}` —
exactly that object. Without it, keep on-device speech. With it, one
push-to-talk hold is:

```
-> voice_start {utterance_id, sample_rate, encoding, channels, focus?}   (control, no seq)
-> binary frames: raw PCM16, any even length (1280 bytes ≈ 40 ms is fine)
-> voice_end {utterance_id}                                              (control, no seq)
<- transcript {utterance_id, text, final: true}                          (agent data)
<- speech ...   (the turn runs on the final text, as for a sent transcript)
```

Rules: the release ends the turn — the server finalises on
`voice_end`, it does not wait for silence detection. One final per
utterance, sent even when nothing was heard (empty text), within
10 s. Send a fresh `pose` and snapshot before `voice_start` and
before `voice_end`, as before a typed transcript, so the turn sees
where the surveyor stands at the release. An utterance is capped at
30 s of audio. `voice_cancel` abandons the hold; a new press replaces
a hold still waiting; a dropped socket drops the open utterance and
binary frames are never resent. Binary frames with no open utterance
are dropped, not fatal. This server sends no partials; if a future one
does, they arrive as control (no agent seq) and only the final is data.
