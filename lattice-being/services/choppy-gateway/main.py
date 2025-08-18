from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import math
import uuid
import os

app = FastAPI(title="Choppy Gateway")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class IngestReq(BaseModel):
	text: str
	size: int = 600
	overlap: int = 90


class Chunk(BaseModel):
	id: str
	text: str
	span: Tuple[int, int]
	entropy: float


def shannon_entropy(s: str) -> float:
	if not s:
		return 0.0
	from collections import Counter
	c = Counter(s)
	total = len(s)
	return -sum((n/total) * math.log2(n/total) for n in c.values())


@app.post("/ingest", response_model=List[Chunk])
def ingest(req: IngestReq):
	t = req.text
	k, r = req.size, req.overlap
	chunks: List[Chunk] = []
	i = 0
	while i < len(t):
		j = min(len(t), i + k)
		segment = t[i:j]
		chunks.append(Chunk(id=str(uuid.uuid4()), text=segment, span=(i, j), entropy=shannon_entropy(segment)))
		if j == len(t):
			break
		i = j - r if j - r > i else j
	return chunks


class EntropyReq(BaseModel):
	text: str


@app.post("/entropy")
def entropy(req: EntropyReq):
	return {"value": shannon_entropy(req.text)}


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
