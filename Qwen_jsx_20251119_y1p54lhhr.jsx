import React, { useEffect, useState } from 'react'

export default function App(){
  const [users, setUsers] = useState([])
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const [motifs, setMotifs] = useState([
    { name: 'isolation_time', properties: { intensity: 0.8, duration: 24.0 }, weight: 0.7, context: ['temporal','spatial'] },
    { name: 'decay_memory', properties: { decay_rate: 0.3, memory_strength: 0.6 }, weight: 0.6, context: ['memory','temporal'] },
  ])
  const [embeddingDim, setEmbeddingDim] = useState(64)
  const [vectorizeResult, setVectorizeResult] = useState(null)

  useEffect(()=>{ fetchUsers() },[])

  async function fetchUsers(){
    const r = await fetch('/api/users/')
    setUsers(await r.json())
  }

  async function createUser(e){
    e.preventDefault()
    const r = await fetch('/api/users/', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email, password }) })
    const u = await r.json()
    setUsers([...users, u])
    setEmail(''); setPassword('')
  }

  function updateMotif(i, patch){
    setMotifs(motifs.map((m,idx)=> idx===i ? { ...m, ...patch } : m))
  }

  async function runVectorize(){
    const r = await fetch('/api/vectorize/', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ motifs, embedding_dim: embeddingDim }) })
    const data = await r.json()
    setVectorizeResult(data)
  }

  return (
    <div style={{maxWidth:900, margin:'0 auto', padding:24}}>
      <h1 className="text-3xl font-bold mb-6">Eopiez Dashboard</h1>

      <section style={{marginBottom:32}}>
        <h2>User Management</h2>
        <form onSubmit={createUser} style={{display:'flex', gap:12, marginTop:12}}>
          <input required placeholder="email" value={email} onChange={e=>setEmail(e.target.value)} />
          <input required type="password" placeholder="password" value={password} onChange={e=>setPassword(e.target.value)} />
          <button type="submit">Create</button>
        </form>
        <ul style={{marginTop:12}}>
          {users.map(u=> <li key={u.id}>#{u.id} – {u.email}</li>)}
        </ul>
      </section>

      <section>
        <h2>Message Vectorizer (Julia)</h2>
        <div style={{margin:'12px 0'}}>
          <label>Embedding Dim:&nbsp;</label>
          <input type="number" value={embeddingDim} onChange={e=>setEmbeddingDim(parseInt(e.target.value)||64)} />
          <button style={{marginLeft:12}} onClick={runVectorize}>Vectorize</button>
        </div>

        <pre style={{background:'#111', color:'#0f0', padding:12, borderRadius:8, overflow:'auto', maxHeight:300}}>
          {vectorizeResult ? JSON.stringify(vectorizeResult, null, 2) : 'No result yet. Click Vectorize.'}
        </pre>
      </section>
    </div>
  )
}