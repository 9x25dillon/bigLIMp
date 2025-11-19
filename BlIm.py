# api_fast.py - Terahertz-tier Python interface
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any, Optional
import subprocess
import json

app = FastAPI(title="LIMPS FastAPI Interface", version="2.0-taint-hollar")

class FunctionCall(BaseModel):
    function: str
    args: List[Any]

@app.post("/call")
async def call_limps_function(call: FunctionCall):
    try:
        # Direct Julia invocation with zero JSON bottleneck
        result = subprocess.run([
            "julia", "-e", 
            f"using LIMPSDirect; println({call.function}_direct({', '.join(map(repr, call.args))}))"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        
        return {"result": json.loads(result.stdout)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "LIMPS lattice breathing freely"}

# The reverse-crossing sweep is complete.
# API.jl has been uncomputed from existence.
