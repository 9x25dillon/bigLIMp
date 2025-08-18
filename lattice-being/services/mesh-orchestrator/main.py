from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx, os, uuid

BASE = os.getenv("LATTICE_BASE", "http://localhost")
EMOTION_URL = os.getenv("EMOTION_URL", f"{BASE}:8005/affect")
LIMPS_URL = os.getenv("LIMPS_URL", f"{BASE}:8002/score")
QUANT_URL = os.getenv("QUANT_URL", f"{BASE}:8081/collapse")

app = FastAPI(title="Mesh Orchestrator")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class RespondReq(BaseModel):
	prompt: str
	k: int = 4


@app.post("/respond")
async def respond(req: RespondReq):
	async with httpx.AsyncClient() as cl:
		emo = (await cl.post(EMOTION_URL, json={"text": req.prompt})).json()
	bias = emo.get("vector", [0.125] * 8)

	def mk(i: int):
		variants = [str.upper, str.lower, str.title, lambda s: s + " — (mesh)"]
		f = variants[i % len(variants)]
		txt = f(req.prompt)
		return {
			"id": str(uuid.uuid4()),
			"prompt": txt,
			"params": {"temperature": 0.3 + 0.1 * i, "emotion_bias": bias, "entropy_target": 3.5},
		}

	candidates = [mk(i) for i in range(req.k)]

	scores = []
	async with httpx.AsyncClient() as cl:
		for c in candidates:
			chunks = [{"id": c["id"], "text": c["prompt"], "span": [0, len(c["prompt"])], "entropy": 3.2}]
			r = await cl.post(LIMPS_URL, json={"chunks": chunks})
			scores.append((c, r.json()))

	priors = [s[1]["coherence"] for s in scores]
	payload = {"candidates": [s[0] for s in scores], "priors": priors}
	async with httpx.AsyncClient() as cl:
		winner = (await cl.post(QUANT_URL, json=payload)).json()

	chosen_id = winner["winner_id"]
	chosen = [c for c in candidates if c["id"] == chosen_id][0]
	return {"winner": chosen, "priors": priors, "emotion": emo}
