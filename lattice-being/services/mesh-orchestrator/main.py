from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import httpx, os, uuid, asyncio

import httpx, os, uuid


BASE = os.getenv("LATTICE_BASE", "http://localhost")
EMOTION_URL = os.getenv("EMOTION_URL", f"{BASE}:8005/affect")
LIMPS_URL = os.getenv("LIMPS_URL", f"{BASE}:8002/score")
QUANT_URL = os.getenv("QUANT_URL", f"{BASE}:8081/collapse")


# Model server config (OpenAI-compatible or OpenRouter/vLLM)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "vllm")  # openai|openrouter|vllm
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", f"{BASE}:8007/v1")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")


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
        return trace.get_tracer("mesh-orchestrator")
    except Exception:
        return None

tracer = _init_tracing()


async def generate_candidates(prompt: str, k: int, bias: List[float]):
    temps = [0.4 + 0.15 * i for i in range(k)]
    top_ps = [0.8 - 0.05 * i for i in range(k)]

    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"

    async def one(i: int):
        temperature = max(0.1, min(2.0, temps[i]))
        top_p = max(0.1, min(1.0, top_ps[i]))
        body = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        url = f"{MODEL_BASE_URL}/chat/completions"
        async with httpx.AsyncClient(timeout=60) as cl:
            r = await cl.post(url, json=body, headers=headers)
            r.raise_for_status()
            js = r.json()
            text = js.get("choices", [{}])[0].get("message", {}).get("content", prompt)
        return {
            "id": str(uuid.uuid4()),
            "prompt": text,
            "params": {"temperature": temperature, "top_p": top_p, "emotion_bias": bias, "entropy_target": 3.5},
        }

    return await asyncio.gather(*[one(i) for i in range(k)])


@app.post("/respond")
async def respond(req: RespondReq):
	async with httpx.AsyncClient() as cl:
		emo = (await cl.post(EMOTION_URL, json={"text": req.prompt})).json()
	bias = emo.get("vector", [0.125] * 8)


	candidates = await generate_candidates(req.prompt, req.k, bias)

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
