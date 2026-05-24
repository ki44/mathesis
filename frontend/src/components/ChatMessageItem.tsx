import type { ChatMessage } from '../types'
import { MarkdownRenderer } from './MarkdownRenderer'

interface Props {
  message: ChatMessage
}

export function ChatMessageItem({ message }: Props) {
  const isUser = message.role === 'user'
  const isToolCall = message.role === 'tool_call'
  const isAssistant = message.role === 'assistant'

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
          padding: isAssistant ? '4px 4px' : '8px 12px',
          borderRadius: 8,
          background: isUser ? '#0e639c' : isToolCall ? 'var(--bg-3)' : 'var(--bg-2)',
          color: isToolCall ? 'var(--text-2)' : 'var(--text-1)',
          fontSize: isToolCall ? 12 : 14,
          fontStyle: isToolCall ? 'italic' : 'normal',
          whiteSpace: isAssistant ? undefined : 'pre-wrap',
          wordBreak: 'break-word',
          lineHeight: 1.5,
        }}
      >
        {isAssistant
          ? (message.content
              ? <MarkdownRenderer content={message.content} compact />
              : <span style={{ opacity: 0.4, padding: '8px 12px', display: 'block' }}>…</span>)
          : (message.content || <span style={{ opacity: 0.4 }}>…</span>)
        }
      </div>
    </div>
  )
}
