from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import hashlib, time, sqlite3, os

app = FastAPI(title="TimeCrystals")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

db_path = "/data/crystals.db"
os.makedirs("/data", exist_ok=True)
conn = sqlite3.connect(db_path, check_same_thread=False)
conn.execute(
	"CREATE TABLE IF NOT EXISTS crystals(id TEXT PRIMARY KEY, signature TEXT, period REAL, stability REAL, last_observed TEXT)"
)


class ObserveReq(BaseModel):
	signature: str


@app.post("/observe")
def observe(req: ObserveReq):
	now = time.time()
	sid = hashlib.sha256(req.signature.encode()).hexdigest()[:16]
	cur = conn.execute("SELECT period, stability, last_observed FROM crystals WHERE id=?", (sid,))
	row = cur.fetchone()
	if row:
		prev_ts = float(row[2]) if row[2] else now
		period = 0.8 * (now - prev_ts) + 0.2 * row[0]
		stability = min(1.0, 0.9 * row[1] + 0.1)
		conn.execute(
			"UPDATE crystals SET period=?, stability=?, last_observed=? WHERE id=",
			(period, stability, str(now), sid),
		)
	else:
		period, stability = 0.0, 0.1
		conn.execute(
			"INSERT INTO crystals(id, signature, period, stability, last_observed) VALUES(?,?,?,?,?)",
			(sid, req.signature, period, stability, str(now)),
		)
	conn.commit()
	return {"id": sid, "period": period, "stability": stability}


@app.get("/crystals")
def list_crystals():
	cur = conn.execute(
		"SELECT id, signature, period, stability, last_observed FROM crystals ORDER BY stability DESC LIMIT 100"
	)
	rows = [
		dict(id=r[0], signature=r[1], period=r[2], stability=r[3], last_observed=r[4])
		for r in cur.fetchall()
	]
	return rows


if __name__ == "__main__":
	import uvicorn
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
