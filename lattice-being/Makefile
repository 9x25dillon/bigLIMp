.PHONY: up down build fmt lint test zip mesh otel

up:
	docker compose -f infra/docker-compose.yml up --build -d

down:
	docker compose -f infra/docker-compose.yml down -v

build:
	docker compose -f infra/docker-compose.yml build

test:
	pytest -q || true

zip:
	cd .. && zip -r lattice-being.zip lattice-being -x "*/node_modules/*" "*/__pycache__/*" "*/dist/*"

fmt:
	@echo "(todo) add ruff/black/prettier config"

lint:
	@echo "(todo) add flake8/eslint configs"

mesh:
	docker compose -f infra/docker-compose.yml up --build -d mesh quantum

otel:
	@echo "Set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 before running services"

import-multi-branch:
	python3 scripts/import_multi_branch_repos.py --user $(USER)


