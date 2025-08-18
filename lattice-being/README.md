Lattice‑Being Monorepo Scaffold v0.1

Pragmatic, buildable scaffold for the augmented stack: Choppy → LIMPS → MatrixProcessor → Time‑Crystals → Emotion Layer → Quantum‑Coherence → Exo‑Lattice → Control Panel.

Dev quickstart

- Python services: create venvs and install deps (fastapi, uvicorn, pydantic, numpy, httpx). Or use Docker Compose.
- Run with Docker: `make up` (from repo root). Stop with `make down`.
- Control Panel: `cd apps/control-panel && npm i && npm run dev`.

Ports

- Choppy: 8001
- LIMPS: 8002
- MatrixProcessor: 8003
- Time‑Crystals: 8004
- Emotion: 8005
- Exo‑Lattice: 8080
- NATS: 4222 (client), 8222 (monitor)

Notes

- Julia Quantum‑Sim runs locally via `julia services/quantum-sim/server.jl` (not yet in compose).
- JSON Schemas under `libs/schemas`.
- SDK stubs in `libs/python-sdk` and `libs/ts-sdk`.
