import { useCallback, useEffect, useRef, useState } from 'react'
import { useChat } from '../hooks/useChat'
import { useChatStore } from '../store/chatStore'
import { ChatMessageItem } from './ChatMessageItem'
import { useContextMenuClose } from '../hooks/useContextMenuClose'
import type { ChatMessage } from '../types'

const NO_MESSAGES: ChatMessage[] = []

type ContextMenu = { x: number; y: number; convId: string }

export function ChatPanel() {
  const conversations = useChatStore((s) => s.conversations)
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const activeConv = useChatStore((s) => s.conversations.find((c) => c.id === s.activeConversationId))
  const messages = activeConv?.messages ?? NO_MESSAGES
  const newConversation = useChatStore((s) => s.newConversation)
  const selectConversation = useChatStore((s) => s.selectConversation)
  const deleteConversation = useChatStore((s) => s.deleteConversation)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const { sendMessage } = useChat()

  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const [contextMenu, setContextMenu] = useState<ContextMenu | null>(null)
  const closeContextMenu = useCallback(() => setContextMenu(null), [])
  useContextMenuClose(closeContextMenu)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    setInput('')
    sendMessage(trimmed)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div
      style={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderLeft: '1px solid #333',
        background: '#1e1e1e',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {/* ── Conversation list ─────────────────────────────────── */}
      <div style={{ borderBottom: '1px solid #333', flexShrink: 0 }}>
        {/* Header */}
        <div
          style={{
            padding: '8px 12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: '#888',
              textTransform: 'uppercase',
              letterSpacing: 1,
            }}
          >
            Conversations
          </span>
          <button
            onClick={() => newConversation()}
            title="New conversation"
            className="conv-new-btn"
          >
            +
          </button>
        </div>

        {/* List */}
        <div style={{ maxHeight: 160, overflowY: 'auto' }}>
          {conversations.length === 0 && (
            <p style={{ color: '#555', fontSize: 12, padding: '4px 12px 8px' }}>
              Aucune conversation
            </p>
          )}
          {conversations.map((conv) => {
            const isActive = conv.id === activeConversationId
            return (
              <div
                key={conv.id}
                onClick={() => { void selectConversation(conv.id) }}
                onContextMenu={(e) => {
                  e.preventDefault()
                  setContextMenu({ x: e.clientX, y: e.clientY, convId: conv.id })
                }}
                className={`conv-item${isActive ? ' active' : ''}`}
              >
                <span style={{ opacity: isActive ? 1 : 0, fontSize: 10, flexShrink: 0 }}>▶</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{conv.title}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Active conversation header ─────────────────────────── */}
      <div
        style={{
          padding: '7px 12px',
          borderBottom: '1px solid #333',
          fontWeight: 600,
          fontSize: 13,
          color: '#cccccc',
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          minHeight: 34,
        }}
      >
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {activeConv?.title ?? 'Mathesis'}
        </span>
        {isStreaming && (
          <span style={{ opacity: 0.5, fontSize: 11, flexShrink: 0 }}>en train d'écrire…</span>
        )}
      </div>

      {/* ── Messages ──────────────────────────────────────────── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 12px 0' }}>
        {!activeConv && (
          <p style={{ color: '#555', fontSize: 13, textAlign: 'center', marginTop: 40 }}>
            Créez une conversation ou envoyez un message…
          </p>
        )}
        {activeConv && messages.length === 0 && (
          <p style={{ color: '#555', fontSize: 13, textAlign: 'center', marginTop: 40 }}>
            Demandez à Mathesis de créer un cours…
          </p>
        )}
        {messages.map((msg) => (
          <ChatMessageItem key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* ── Input ─────────────────────────────────────────────── */}
      <div style={{ padding: 10, borderTop: '1px solid #333', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Écrivez un message… (Entrée pour envoyer)"
            disabled={isStreaming}
            rows={3}
            style={{
              flex: 1,
              background: '#2d2d2d',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#cccccc',
              padding: '8px 10px',
              fontSize: 13,
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
            }}
          />
          <button
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            style={{
              background: isStreaming || !input.trim() ? '#333' : '#0e639c',
              border: 'none',
              borderRadius: 6,
              color: '#fff',
              padding: '0 14px',
              cursor: isStreaming || !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: 13,
              fontWeight: 600,
              transition: 'background 0.15s',
            }}
          >
            ↑
          </button>
        </div>
      </div>

      {/* ── Context menu ──────────────────────────────────────── */}
      {contextMenu && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'fixed',
            top: contextMenu.y,
            left: contextMenu.x,
            background: '#2d2d2d',
            border: '1px solid #444',
            borderRadius: 6,
            zIndex: 1000,
            minWidth: 150,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            overflow: 'hidden',
          }}
        >
          <div
            onClick={() => { deleteConversation(contextMenu.convId); setContextMenu(null) }}
            className="conv-menu-delete"
          >
            Supprimer
          </div>
        </div>
      )}
    </div>
  )
}

