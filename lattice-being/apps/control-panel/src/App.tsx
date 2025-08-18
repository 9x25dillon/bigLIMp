import React, { useState } from "react";

export default function App() {
	const [text, setText] = useState("");
	const [chunks, setChunks] = useState<any[]>([]);
	const [score, setScore] = useState<any | null>(null);
	const [emotion, setEmotion] = useState<any | null>(null);
	const [mesh, setMesh] = useState<any | null>(null);

	async function analyze() {
		const ch = await fetch("http://localhost:8001/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, size: 600, overlap: 90 }) }).then(r => r.json());
		setChunks(ch);
		const sc = await fetch("http://localhost:8002/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ chunks: ch }) }).then(r => r.json());
		setScore(sc);
		const emo = await fetch("http://localhost:8005/affect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }).then(r => r.json());
		setEmotion(emo);
	}

	async function meshRespond() {
		const m = await fetch("http://localhost:8006/respond", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: text, k: 4 }) }).then(r => r.json());
		setMesh(m);
	}

	return (
		<div style={{ padding: 24, fontFamily: "ui-sans-serif" }}>
			<h1 style={{ fontSize: 24, fontWeight: 700 }}>Lattice Control Panel</h1>
			<textarea value={text} onChange={e => setText(e.target.value)} placeholder="Paste text…" style={{ width: "100%", height: 120, marginTop: 12 }} />
			<div style={{ display: "flex", gap: 12, marginTop: 12 }}>
				<button onClick={analyze} style={{ padding: "8px 16px" }}>Analyze</button>
				<button onClick={meshRespond} style={{ padding: "8px 16px" }}>Mesh Respond</button>
			</div>

			{emotion && (
				<div style={{ marginTop: 24 }}>
					<h2>Emotion Vector</h2>
					<pre>{JSON.stringify(emotion, null, 2)}</pre>
				</div>
			)}

			{chunks.length > 0 && (
				<div style={{ marginTop: 24 }}>
					<h2>Chunks ({chunks.length})</h2>
					<ul>
						{chunks.map(c => (
							<li key={c.id}>[{c.span.join("-")}] H={c.entropy.toFixed(2)} — {c.text.slice(0, 60)}…</li>
						))}
					</ul>
				</div>
			)}

			{score && (
				<div style={{ marginTop: 24 }}>
					<h2>Plan</h2>
					<pre>{JSON.stringify(score, null, 2)}</pre>
				</div>
			)}

			{mesh && (
				<div style={{ marginTop: 24 }}>
					<h2>Mesh Winner</h2>
					<pre>{JSON.stringify(mesh, null, 2)}</pre>
				</div>
			)}
		</div>
	);
}
