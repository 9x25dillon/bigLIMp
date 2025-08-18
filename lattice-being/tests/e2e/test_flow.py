import httpx
import pytest

BASE = "http://localhost"


@pytest.mark.asyncio
async def test_mesh_roundtrip():
	async with httpx.AsyncClient() as cl:
		text = "Entropy is a river that learns to sing."
		ch = (await cl.post(f"{BASE}:8001/ingest", json={"text": text, "size": 40, "overlap": 10})).json()
		assert ch and isinstance(ch, list)
		sc = (await cl.post(f"{BASE}:8002/score", json={"chunks": ch})).json()
		assert "plan" in sc
		emo = (await cl.post(f"{BASE}:8005/affect", json={"text": text})).json()
		assert "vector" in emo
		mesh = (await cl.post(f"{BASE}:8006/respond", json={"prompt": text})).json()
		assert "winner" in mesh
