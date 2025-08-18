from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import os

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
	x = np.linspace(-1, 1, 256)
	T = [np.ones_like(x), x]
	for n in range(2, req.degree + 1):
		T.append(2 * x * T[-1] - T[-2])
	basis = np.vstack(T[: req.degree + 1])
	coeffs = np.array(req.coeffs[: req.degree + 1])
	y = (coeffs[:, None] * basis).sum(axis=0)
	return {"projected": y.tolist()}


class OptimizeReq(BaseModel):
	matrix: List[List[float]]
	method: str = "gd"


@app.post("/optimize")
def optimize(req: OptimizeReq):
	A = np.array(req.matrix)
	u, s, vt = np.linalg.svd(A, full_matrices=False)
	return {
		"solution": {"u": u[:, 0].tolist(), "sigma": float(s[0]), "v": vt[0, :].tolist()},
		"metrics": {"condition": float(np.linalg.cond(A))},
	}


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
