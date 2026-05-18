import type { ChatMessage } from '../types'

interface Props {
  message: ChatMessage
}

export function ChatMessageItem({ message }: Props) {
  const isUser = message.role === 'user'
  const isToolCall = message.role === 'tool_call'

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: 8,
      }}
    >
      <div
        style={{
          maxWidth: '85%',
          padding: '8px 12px',
          borderRadius: 8,
          background: isUser ? '#0e639c' : isToolCall ? '#2d2d2d' : '#252526',
          color: isToolCall ? '#888' : '#cccccc',
          fontSize: isToolCall ? 12 : 14,
          fontStyle: isToolCall ? 'italic' : 'normal',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          lineHeight: 1.5,
        }}
      >
        {message.content || <span style={{ opacity: 0.4 }}>…</span>}
      </div>
    </div>
  )
}
