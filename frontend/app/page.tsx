'use client'

import { useState } from 'react'

export default function Home() {
  const [message, setMessage] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)

  const sendMessage = async () => {
    if (!message.trim()) return

    setLoading(true)
    setAnswer('')

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })

      const data = await res.json()
      setAnswer(data.answer || 'No answer returned.')
    } catch (error) {
      setAnswer('Something went wrong while contacting the backend.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ maxWidth: 800, margin: '40px auto', padding: 20 }}>
      <h1>RAG Chatbot</h1>
      <p>Ask questions from the indexed documents.</p>

      <textarea
        rows={6}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your question here..."
        style={{ width: '100%', marginBottom: 12, padding: 10 }}
      />

      <button onClick={sendMessage} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>

      <div style={{ marginTop: 24, whiteSpace: 'pre-wrap' }}>
        {answer}
      </div>
    </main>
  )
}