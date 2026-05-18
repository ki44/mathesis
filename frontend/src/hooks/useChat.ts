import { useCallback } from 'react'
import { useChatStore } from '../store/chatStore'
import { useCourseStore } from '../store/courseStore'

export function useChat() {
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const newConversation = useChatStore((s) => s.newConversation)
  const addMessage = useChatStore((s) => s.addMessage)
  const appendDelta = useChatStore((s) => s.appendDelta)
  const setIsStreaming = useChatStore((s) => s.setIsStreaming)
  const fetchProposals = useCourseStore((s) => s.fetchProposals)

  const sendMessage = useCallback(
    async (text: string) => {
      const convId = activeConversationId ?? newConversation()
      addMessage(convId, { role: 'user', content: text })
      const asstId = addMessage(convId, { role: 'assistant', content: '' })
      setIsStreaming(true)

      try {
        const res = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, conversation_id: convId }),
        })

        if (!res.body) throw new Error('No response body')

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          // SSE messages are separated by double newlines
          const parts = buffer.split('\n\n')
          buffer = parts.pop() ?? ''

          for (const part of parts) {
            const lines = part.trim().split('\n')
            let event = ''
            let data = ''
            for (const line of lines) {
              if (line.startsWith('event: ')) event = line.slice(7)
              else if (line.startsWith('data: ')) data = line.slice(6)
            }
            if (!event || !data) continue

            const payload = JSON.parse(data) as Record<string, string>

            if (event === 'delta') {
              appendDelta(convId, asstId, payload.text ?? '')
            } else if (event === 'tool_call') {
              addMessage(convId, { role: 'tool_call', content: `⚙ ${payload.name}` })
            } else if (event === 'done') {
              await fetchProposals()
            } else if (event === 'error') {
              appendDelta(convId, asstId, `\n\n⚠ Erreur : ${payload.message}`)
            }
          }
        }
      } catch (err) {
        appendDelta(convId, asstId, `\n\n⚠ Erreur réseau : ${String(err)}`)
      } finally {
        setIsStreaming(false)
      }
    },
    [activeConversationId, newConversation, addMessage, appendDelta, setIsStreaming, fetchProposals],
  )

  return { sendMessage }
}

