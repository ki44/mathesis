import type { ChatMessage } from '../types'
import { MarkdownRenderer } from './MarkdownRenderer'

interface Props {
  message: ChatMessage
  onContextMenu?: (e: React.MouseEvent) => void
  highlighted?: boolean
}

export function ChatMessageItem({ message, onContextMenu, highlighted }: Props) {
  const isUser = message.role === 'user'
  const isToolCall = message.role === 'tool_call'

  if (isUser) {
    return (
      <div
        className={`msg-wrapper${highlighted ? ' highlighted' : ''}`}
        style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 10, paddingLeft: 40 }}
        onContextMenu={onContextMenu}
      >
        <div
          style={{
            padding: '10px 14px',
            borderRadius: '18px 18px 4px 18px',
            background: 'rgba(127, 109, 242, 0.15)',
            border: '1px solid rgba(127, 109, 242, 0.3)',
            color: 'var(--text-bright)',
            fontSize: 14,
            lineHeight: 1.6,
            wordBreak: 'break-word',
            whiteSpace: 'pre-wrap',
          }}
        >
          {message.content || <span style={{ opacity: 0.4 }}>…</span>}
        </div>
      </div>
    )
  }

  if (isToolCall) {
    return (
      <div className={`msg-wrapper${highlighted ? ' highlighted' : ''}`} style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 6 }} onContextMenu={onContextMenu}>
        <div
          style={{
            padding: '4px 10px',
            borderRadius: 6,
            background: 'var(--bg-3)',
            color: 'var(--text-2)',
            fontSize: 12,
            fontStyle: 'italic',
            wordBreak: 'break-word',
          }}
        >
          {message.content || <span style={{ opacity: 0.4 }}>…</span>}
        </div>
      </div>
    )
  }

  // assistant
  return (
    <div
      className={`msg-wrapper${highlighted ? ' highlighted' : ''}`}
      style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 10, paddingRight: 16 }}
      onContextMenu={onContextMenu}
    >
      <div style={{ color: 'var(--text-1)', fontSize: 14, lineHeight: 1.6, wordBreak: 'break-word', width: '100%' }}>
        {message.content
          ? <MarkdownRenderer content={message.content} compact />
          : <span style={{ opacity: 0.4, padding: '8px 12px', display: 'block' }}>…</span>}
      </div>
    </div>
  )
}

