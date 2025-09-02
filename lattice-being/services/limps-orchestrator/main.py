from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os


app = FastAPI(title="LIMPS Orchestrator")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class Chunk(BaseModel):
	id: str
	text: str
	span: List[int]
	entropy: float


class ScoreReq(BaseModel):
	chunks: List[Chunk]
	constraints: Dict[str, float] | None = None


@app.post("/score")
def score(req: ScoreReq):
	if not req.chunks:
		return {"ghost_score": 0, "coherence": 0, "plan": []}
	avg_entropy = sum(c.entropy for c in req.chunks) / len(req.chunks)
	coverage = (req.chunks[-1].span[1] - req.chunks[0].span[0]) / max(1, sum(len(c.text) for c in req.chunks))
	plan = [
		{"action": "mesh.generate", "k": 4, "entropy_target": avg_entropy},
		{"action": "matrix.optimize", "method": "chebyshev"},
	]
	return {"ghost_score": ghost_score, "coherence": coherence, "plan": plan}


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
