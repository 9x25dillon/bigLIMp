from sqlalchemy import create_engine, text
import os

DB_URL = os.getenv("POSTGRES_URL", "")
engine = create_engine(DB_URL) if DB_URL else None


def migrate():
	if not engine:
		return
	with engine.begin() as cx:
		cx.execute(
			text(
				"""
				CREATE TABLE IF NOT EXISTS crystals (
				  id TEXT PRIMARY KEY,
				  signature TEXT NOT NULL,
				  period DOUBLE PRECISION NOT NULL,
				  stability DOUBLE PRECISION NOT NULL,
				  last_observed TIMESTAMP NOT NULL
				);
				"""
			)
		)
