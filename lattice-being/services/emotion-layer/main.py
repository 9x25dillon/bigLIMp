from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import os

app = FastAPI(title="EmotionLayer")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

EMOTIONS = ["calm", "joy", "awe", "focus", "sad", "anger", "fear", "surprise"]


class AffectReq(BaseModel):
	text: str
	context: str | None = None


@app.post("/affect")
def affect(req: AffectReq):
	h = abs(hash(req.text + (req.context or "")))
	rng = np.random.default_rng(h % (2**32))
	vec = rng.random(8)
	vec /= vec.sum()
	entropy_bias = float(-(vec * np.log(vec + 1e-9)).sum())
	return {"emotions": EMOTIONS, "vector": vec.tolist(), "entropy_bias": entropy_bias}


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
