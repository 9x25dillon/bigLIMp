.PHONY: up down build fmt lint test zip

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
