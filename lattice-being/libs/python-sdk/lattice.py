import httpx, os


class LatticeClient:
	def __init__(self, base: str = None):
		self.base = base or os.getenv("LATTICE_BASE", "http://localhost")

	async def chunks(self, text: str):
		async with httpx.AsyncClient() as cl:
			r = await cl.post(f"{self.base}:8001/ingest", json={"text": text, "size":600, "overlap":90})
			r.raise_for_status()
			return r.json()

	async def score(self, chunks):
		async with httpx.AsyncClient() as cl:
			r = await cl.post(f"{self.base}:8002/score", json={"chunks": chunks})
			r.raise_for_status()
			return r.json()
