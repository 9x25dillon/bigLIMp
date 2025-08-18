export async function chunks(text: string, base = "http://localhost") {
	const r = await fetch(`${base}:8001/ingest`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ text, size:600, overlap:90 }) });
	return await r.json();
}
