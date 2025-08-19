CCL ↔ LIMPS Bridge (with al-ULS)

A compact, top‑down workflow you can drop into Cursor or any editor.

What you get

- CCL (Categorical Coherence Linter) to detect "ghosts" in code.
- Julia Bridge exposing:
  - POST /coherence → compute coherence/ghost score from a CCL report.
  - POST /optimize  → optimize a pipeline adjacency matrix via al-ULS (local Julia pkg or HTTP proxy).
- Mock al-ULS FastAPI for local e2e testing.
- Adjacency builder: turn CCL pipelines into a matrix for /optimize.
- One-command dev with make dev.
- Tests to validate JSON and the adjacency builder.

> Note: JSON does not support comments. Do not put # or // comments in .json files in examples/.



Quick start

```bash
# 1) Python env (>=3.10)
python -m venv .venv && . .venv/bin/activate
pip install fastapi uvicorn pydantic

# 2) Julia deps
cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..

# 3) Run everything (mock al-ULS + Julia bridge)
make dev  # (spawns two terminals in most shells)

# 4) Run CCL on your code and build adjacency
python python/ccl.py path/to/your_pkg --report report.json
python python/ccl_build_adjacency.py report.json --out adjacency.json

# 5) Send to bridge
python python/ccl_julia_client.py report.json --julia-url http://localhost:8099 --adj adjacency.json

# 6) Run tests
python -m tests.run
```

Top‑down layout

```
.
├── README.md
├── Makefile
├── scripts/
│   └── dev.sh
├── julia/
│   ├── Project.toml
│   └── LIMPSBridge.jl
├── python/
│   ├── ccl.py                   # entropy‑driven ghost detector (single file)
│   ├── ccl_build_adjacency.py   # build matrix from CCL pipelines
│   ├── ccl_julia_client.py      # post to /coherence and /optimize
│   └── mock_al_uls_server.py    # FastAPI stub for al‑ULS
├── examples/
│   ├── adjacency.json
│   └── report.min.json
└── tests/
    └── run.py
```

al‑ULS integration

- Local Julia package: expose `AL_ULS.optimize(adj; mode, beta)` on `JULIA_LOAD_PATH`.
- HTTP proxy: export `AL_ULS_URL` (e.g., `http://localhost:8000`) and run the FastAPI.