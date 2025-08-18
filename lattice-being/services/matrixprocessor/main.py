from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import os

def _xp():
	try:
		import cupy as cp  # type: ignore
		return cp
	except Exception:
		return np

def _svd(A):
	try:
		import torch
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		T = torch.tensor(A, dtype=torch.float32, device=device)
		u, s, vt = torch.linalg.svd(T, full_matrices=False)
		return u.cpu().numpy(), s.cpu().numpy(), vt.cpu().numpy()
	except Exception:
		return np.linalg.svd(A, full_matrices=False)

app = FastAPI(title="MatrixProcessor")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class ChebyshevReq(BaseModel):
	coeffs: List[float]
	degree: int


@app.post("/chebyshev")
def chebyshev(req: ChebyshevReq):
	xp = _xp()
	x = xp.linspace(-1, 1, 256)
	T = [xp.ones_like(x), x]
	for n in range(2, req.degree + 1):
		T.append(2 * x * T[-1] - T[-2])
	basis = xp.vstack(T[: req.degree + 1])
	coeffs = xp.array(req.coeffs[: req.degree + 1])
	y = (coeffs[:, None] * basis).sum(axis=0)
	try:
		return {"projected": y.get().tolist()}
	except Exception:
		return {"projected": xp.asnumpy(y).tolist() if hasattr(xp, "asnumpy") else y.tolist()}


class OptimizeReq(BaseModel):
	matrix: List[List[float]]
	method: str = "gd"


@app.post("/optimize")
def optimize(req: OptimizeReq):
	A = np.array(req.matrix, dtype=float)
	u, s, vt = _svd(A)
	return {
		"solution": {"u": u[:, 0].tolist(), "sigma": float(s[0]), "v": vt[0, :].tolist()},
		"metrics": {"condition": float(np.linalg.cond(A))},
	}


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)