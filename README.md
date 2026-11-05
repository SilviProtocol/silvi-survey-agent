# Silvi Field Agent

A voice field-survey agent. A surveyor walks a planting site with a
phone; the phone streams what it sees over an open wire contract, and
this agent talks back — naming trees, asking for the next useful
photo, keeping count, and writing survey records as the walk goes.

Built during a hackathon starting 27 October 2026. The phone app and
its positioning are closed prior work and are not in this repo; the
agent reaches them only through the wire contract in `schemas/` and
`docs/PHONE-CLIENT.md`. Species answers come through a plug
(`src/field_agent/species/`): Silvi runs a closed species service
behind it, and this repo ships a plain local BioCLIP 2 fallback.

Built with Muse Spark Open. Model seats: Muse Spark for the spoken
turns, Muse Voice Transcribe for speech on the server, and Muse
Spark Open weights served through the provider config. Built with
Muse Code.

## What's here

- The agent: spoken turns over a tool-calling model, with writes
  gated in code so a question can never change a record.
- The wire: versioned JSON envelopes (`schemas/wire.schema.json`),
  a live WebSocket server with reconnect-safe sessions, and journal
  replay for recorded walks.
- Records: `schemas/observation.schema.json` — one 3D-anchored
  observation per plant, validated on the way in.
- The species plug and fallback, and a spend-guarded provider layer
  (`config/providers.json`).

## Run it

```bash
uv sync --extra dev
uv run field-agent demo            # synthetic journal, mock model, writes data/demo/records.json
uv run field-agent doctor          # provider keys and spend state (no live calls)
uv run field-agent serve           # live server on 127.0.0.1:8766 (needs a model key)
uv run pytest
```

Copy `.env.example` to `.env` and add a provider key to serve live.
Server-side speech needs the voice extra and a Meta key
(`uv sync --extra voice`); the local species fallback needs the
species extra and BioCLIP 2 weights at `BIOCLIP2_WEIGHTS`
(`uv sync --extra species`), or set `SPECIES_SERVICE_URL` to use a
species service over HTTP.

## Models

| Seat | Model | Notes |
|---|---|---|
| `spark` | Muse Spark | the spoken turns, reasoning effort minimal |
| `spark-open` | Muse Spark Open (Baseten) | open weights seat |
| `spark-open-fw` | Muse Spark Open (Fireworks) | fallback seat |
| speech | Muse Voice Transcribe | optional server-side transcription |
| species fallback | BioCLIP 2 | local, one photo at a time |

The language model is sent words, not photos. Photos go only to the
species plug.

## Prior work

See `PRIOR-WORK.md` for what predates the hackathon and stays closed.

## Licence

AGPL-3.0-only. See `LICENSE`.
