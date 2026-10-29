import { useState, useEffect, useRef } from 'react'

const API_BASE = 'http://localhost:8000'

export default function ChatBot() {
  const [isOpen, setIsOpen] = useState(false)
  const [message, setMessage] = useState('')
  const [chatHistory, setChatHistory] = useState([
    { role: 'assistant', content: 'Hello! I am your EngineIQ AI Advisor. How can I help you with your Cat C18 diagnostics today?' }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [chatHistory, isLoading])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!message.trim() || isLoading) return

    const userMessage = { role: 'user', content: message }
    setChatHistory(prev => [...prev, userMessage])
    setMessage('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE}/api/chat/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          history: chatHistory
        })
      })

      if (!response.ok) throw new Error('Chat API error')
      
      const data = await response.json()
      setChatHistory(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error connecting to the AI advisor.' }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={`chatbot-wrapper ${isOpen ? 'open' : ''}`}>
      <button className="chatbot-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '×' : '💬'}
      </button>

      <div className="chatbot-window card">
        <div className="chatbot-header">
          <div className="chatbot-header-info">
            <div className="chatbot-icon">🤖</div>
            <div>
              <h3>AI Advisor</h3>
              <p>EngineIQ Logic</p>
            </div>
          </div>
          <div className="chatbot-status">
            <span className="chat-status-dot"></span>
            Online
          </div>
        </div>

        <div className="chatbot-messages" ref={scrollRef}>
          {chatHistory.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-bubble">{msg.content}</div>
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <div className="message-bubble loading">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          )}
        </div>

        <form className="chatbot-input" onSubmit={handleSend}>
          <input
            type="text"
            placeholder="Ask about sensors, leaks..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>🚀</button>
        </form>
      </div>
    </div>
  )
}
