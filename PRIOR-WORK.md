# Prior work

The phone app, its on-phone detection and positioning, and the
species service existed before 27 October 2026. They are closed and
are not part of this repository. This code reaches them only through
two seams: the wire contract (the phone streams poses, plant
snapshots, pins and speech; the agent answers) and the species plug
(evidence photos out, names and confidences back).

What is new here, built during the hackathon week from 27 October:
the voice agent and its tool wiring, the survey world and records,
the live server and journal replay, server-side speech, the species
plug with its plain local fallback, and the schemas.
