SHELL := /bin/bash
export PYTHONPATH := $(PWD)/python

.PHONY: dev bridge mock clean test

dev:
	@bash scripts/dev.sh

bridge:
	@cd julia && AL_ULS_URL=$${AL_ULS_URL:-http://localhost:8000} julia --project=. -e 'using LIMPSBridge; LIMPSBridge.serve()'

mock:
	@. .venv/bin/activate 2>/dev/null || true; \
	python -m uvicorn python.mock_al_uls_server:app --host 0.0.0.0 --port 8000

test:
	@python -m tests.run

clean:
	rm -f report.json adjacency.json examples/adjacency.test.out.json