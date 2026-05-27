import { useCallback, useEffect, useRef, useState } from 'react'
import { useChat } from '../hooks/useChat'
import { useChatStore } from '../store/chatStore'
import { ChatMessageItem } from './ChatMessageItem'
import { ctxMenuStyle } from './ctxMenuStyle'
import { useContextMenuClose } from '../hooks/useContextMenuClose'
import type { ChatMessage } from '../types'

const NO_MESSAGES: ChatMessage[] = []

type ConvCtxMenu = { x: number; y: number; convId: string }
type MsgCtxMenu = { x: number; y: number; msgIndex: number }

export function ChatPanel() {
  const conversations = useChatStore((s) => s.conversations)
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const activeConv = conversations.find((c) => c.id === activeConversationId)
  const messages = activeConv?.messages ?? NO_MESSAGES
  const variantRuns = activeConv?.variantRuns ?? null
  const newConversation = useChatStore((s) => s.newConversation)
  const selectConversation = useChatStore((s) => s.selectConversation)
  const deleteConversation = useChatStore((s) => s.deleteConversation)
  const renameConversation = useChatStore((s) => s.renameConversation)
  const forkConversation = useChatStore((s) => s.forkConversation)
  const navigateVariant = useChatStore((s) => s.navigateVariant)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const { sendMessage, rerunLastMessage } = useChat()

  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const [convCtxMenu, setConvCtxMenu] = useState<ConvCtxMenu | null>(null)
  const [msgCtxMenu, setMsgCtxMenu] = useState<MsgCtxMenu | null>(null)
  const [renamingConvId, setRenamingConvId] = useState<string | null>(null)
  const [renameConvValue, setRenameConvValue] = useState('')
  const convRenameInputRef = useRef<HTMLInputElement>(null)

  const closeAllMenus = useCallback(() => { setConvCtxMenu(null); setMsgCtxMenu(null) }, [])
  useContextMenuClose(closeAllMenus)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (renamingConvId) convRenameInputRef.current?.focus()
  }, [renamingConvId])

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

  const startRenameConv = (convId: string, currentTitle: string) => {
    setConvCtxMenu(null)
    setRenamingConvId(convId)
    setRenameConvValue(currentTitle)
  }

  const commitRenameConv = async () => {
    if (!renamingConvId || !renameConvValue.trim()) { setRenamingConvId(null); return }
    await renameConversation(renamingConvId, renameConvValue.trim())
    setRenamingConvId(null)
  }

  const canRerun = !isStreaming && messages.some((m) => m.role === 'user')

  return (
    <div
      style={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderLeft: '1px solid var(--border)',
        background: 'var(--bg-1)',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {/* ── Conversation list ─────────────────────────────────── */}
      <div style={{ borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <div style={{ padding: '8px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 1 }}>
            Conversations
          </span>
          <button onClick={() => newConversation()} title="New conversation" className="conv-new-btn">+</button>
        </div>

        <div style={{ maxHeight: 160, overflowY: 'auto' }}>
          {conversations.length === 0 && (
            <p style={{ color: 'var(--text-3)', fontSize: 12, padding: '4px 12px 8px' }}>No conversations</p>
          )}
          {conversations.map((conv) => {
            const isActive = conv.id === activeConversationId
            const isRenaming = renamingConvId === conv.id
            return (
              <div
                key={conv.id}
                onClick={() => !isRenaming && selectConversation(conv.id)}
                onDoubleClick={() => startRenameConv(conv.id, conv.title)}
                onContextMenu={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setConvCtxMenu({ x: e.clientX, y: e.clientY, convId: conv.id })
                }}
                className={`conv-item${isActive ? ' active' : ''}${convCtxMenu?.convId === conv.id ? ' ctx-target' : ''}`}
              >
                <span style={{ opacity: isActive ? 1 : 0, fontSize: 10, flexShrink: 0 }}>▶</span>
                {isRenaming ? (
                  <input
                    ref={convRenameInputRef}
                    value={renameConvValue}
                    onChange={(e) => setRenameConvValue(e.target.value)}
                    onBlur={commitRenameConv}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitRenameConv()
                      if (e.key === 'Escape') setRenamingConvId(null)
                      e.stopPropagation()
                    }}
                    onClick={(e) => e.stopPropagation()}
                    style={{ flex: 1, background: 'var(--bg-3)', border: '1px solid #0e639c', color: 'var(--text-bright)', fontSize: 12, padding: '0 4px', outline: 'none', minWidth: 0 }}
                  />
                ) : (
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{conv.title}</span>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Active conversation header ─────────────────────────── */}
      <div
        style={{
          padding: '7px 12px',
          borderBottom: '1px solid var(--border)',
          fontWeight: 600,
          fontSize: 13,
          color: 'var(--text-1)',
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
        {isStreaming && <span style={{ opacity: 0.5, fontSize: 11, flexShrink: 0 }}>typing…</span>}
      </div>

      {/* ── Messages ──────────────────────────────────────────── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 12px 0' }}>
        {!activeConv && (
          <p style={{ color: 'var(--text-3)', fontSize: 13, textAlign: 'center', marginTop: 40 }}>
            Start a conversation or send a message…
          </p>
        )}
        {activeConv && messages.length === 0 && (
          <p style={{ color: 'var(--text-3)', fontSize: 13, textAlign: 'center', marginTop: 40 }}>
            Ask Mathesis to create a course…
          </p>
        )}
        {messages.map((msg, idx) => (
          <ChatMessageItem
            key={msg.id}
            message={msg}
            highlighted={msgCtxMenu?.msgIndex === idx}
            onContextMenu={(e) => {
              e.preventDefault()
              e.stopPropagation()
              setMsgCtxMenu({ x: e.clientX, y: e.clientY, msgIndex: idx })
            }}
          />
        ))}

        {/* Variant navigator — shown below the last assistant message */}
        {variantRuns && variantRuns.runs.length >= 2 && !isStreaming && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0 8px', justifyContent: 'flex-start' }}>
            <button
              onClick={() => activeConversationId && navigateVariant(activeConversationId, -1)}
              disabled={variantRuns.activeIndex === 0}
              style={variantNavBtnStyle}
            >‹</button>
            <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
              {variantRuns.activeIndex + 1} / {variantRuns.runs.length}
            </span>
            <button
              onClick={() => activeConversationId && navigateVariant(activeConversationId, 1)}
              disabled={variantRuns.activeIndex === variantRuns.runs.length - 1}
              style={variantNavBtnStyle}
            >›</button>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input ─────────────────────────────────────────────── */}
      <div style={{ padding: 10, borderTop: '1px solid var(--border)', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Write a message… (Enter to send)"
            disabled={isStreaming}
            rows={3}
            style={{
              flex: 1,
              background: 'var(--bg-3)',
              border: '1px solid var(--border-2)',
              borderRadius: 6,
              color: 'var(--text-1)',
              padding: '8px 10px',
              fontSize: 13,
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <button
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              style={{
                flex: 1,
                background: isStreaming || !input.trim() ? 'var(--bg-3)' : '#0e639c',
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
            <button
              onClick={rerunLastMessage}
              disabled={!canRerun}
              title="Rerun last message"
              style={{
                background: 'var(--bg-3)',
                border: '1px solid var(--border-2)',
                borderRadius: 6,
                color: 'var(--text-2)',
                padding: '0 10px',
                height: 28,
                cursor: canRerun ? 'pointer' : 'not-allowed',
                fontSize: 13,
                transition: 'background 0.15s',
              }}
            >
              ↺
            </button>
          </div>
        </div>
      </div>

      {/* ── Conversation context menu ──────────────────────────── */}
      {convCtxMenu && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={ctxMenuStyle(convCtxMenu.x, convCtxMenu.y)}
        >
          {[
            { label: 'Rename', action: () => { const c = conversations.find((x) => x.id === convCtxMenu.convId); if (c) startRenameConv(c.id, c.title) } },
            null,
            { label: 'Delete', action: () => { deleteConversation(convCtxMenu.convId); setConvCtxMenu(null) }, danger: true },
          ].map((item, i) =>
            item === null ? (
              <div key={i} style={{ height: 1, background: 'var(--border)', margin: '3px 0' }} />
            ) : (
              <button
                key={item.label}
                onClick={() => { setConvCtxMenu(null); item.action() }}
                style={{ display: 'block', width: '100%', textAlign: 'left', background: 'none', border: 'none', cursor: 'pointer', padding: '5px 14px', color: item.danger ? '#f47067' : 'var(--text-1)', fontSize: 13 }}
              >
                {item.label}
              </button>
            ),
          )}
        </div>
      )}

      {/* ── Message context menu ───────────────────────────────── */}
      {msgCtxMenu && activeConversationId && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={ctxMenuStyle(msgCtxMenu.x, msgCtxMenu.y)}
        >
          <button
            onClick={() => {
              setMsgCtxMenu(null)
              forkConversation(activeConversationId, msgCtxMenu.msgIndex)
            }}
            style={{ display: 'block', width: '100%', textAlign: 'left', background: 'none', border: 'none', cursor: 'pointer', padding: '5px 14px', color: 'var(--text-1)', fontSize: 13 }}
          >
            Fork from here
          </button>
        </div>
      )}
    </div>
  )
}

const variantNavBtnStyle: React.CSSProperties = {
  background: 'none',
  border: '1px solid var(--border-2)',
  borderRadius: 3,
  color: 'var(--text-2)',
  cursor: 'pointer',
  width: 22,
  height: 22,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: 16,
  padding: 0,
}