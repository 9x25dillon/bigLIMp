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
	adaptive: bool | None = True


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


def _init_tracing():
	try:
		from opentelemetry import trace
		from opentelemetry.sdk.trace import TracerProvider
		from opentelemetry.sdk.trace.export import BatchSpanProcessor
		from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
		provider = TracerProvider()
		processor = BatchSpanProcessor(OTLPSpanExporter())
		provider.add_span_processor(processor)
		trace.set_tracer_provider(provider)
		return trace.get_tracer("choppy-gateway")
	except Exception:
		return None

tracer = _init_tracing()


@app.post("/ingest", response_model=List[Chunk])
def ingest(req: IngestReq):
	t = req.text
	k, r = req.size, req.overlap
	chunks: List[Chunk] = []
	i = 0
	if req.adaptive and len(t) > 0:
		target = 3.5
		while i < len(t):
			best_j = min(len(t), i + k)
			best_diff = 1e9
			for step in (int(k*0.5), int(k*0.75), k, int(k*1.25), int(k*1.5)):
				j = min(len(t), i + max(64, step))
				seg = t[i:j]
				e = shannon_entropy(seg)
				d = abs(e - target)
				if d < best_diff:
					best_diff = d
					best_j = j
			segment = t[i:best_j]
			chunks.append(Chunk(id=str(uuid.uuid4()), text=segment, span=(i, best_j), entropy=shannon_entropy(segment)))
			if best_j == len(t):
				break
			i = best_j - r if best_j - r > i else best_j
	else:
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