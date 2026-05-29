import { useCallback } from 'react'
import { useChatStore } from '../store/chatStore'
import { useCourseStore } from '../store/courseStore'

export function useChat() {
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const newConversation = useChatStore((s) => s.newConversation)
  const addMessage = useChatStore((s) => s.addMessage)
  const appendDelta = useChatStore((s) => s.appendDelta)
  const setIsStreaming = useChatStore((s) => s.setIsStreaming)
  const startRerun = useChatStore((s) => s.startRerun)
  const finalizeRerun = useChatStore((s) => s.finalizeRerun)
  const clearVariantRuns = useChatStore((s) => s.clearVariantRuns)
  const fetchProposals = useCourseStore((s) => s.fetchProposals)
  const fetchFiles = useCourseStore((s) => s.fetchFiles)

  const stream = useCallback(
    async (text: string, convId: string, rerun: boolean, variantOverride: Array<{ role: string; content: string }> | null = null) => {
      let asstId = addMessage(convId, { role: 'assistant', content: '' })
      let needNewAsst = false
      setIsStreaming(true)

      try {
        const res = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, conversation_id: convId, rerun, variant_override: variantOverride }),
        })

        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

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
              if (needNewAsst) {
                asstId = addMessage(convId, { role: 'assistant', content: '' })
                needNewAsst = false
              }
              appendDelta(convId, asstId, payload.text ?? '')
            } else if (event === 'tool_call') {
              addMessage(convId, { role: 'tool_call', content: `${payload.name}` })
              needNewAsst = true
            } else if (event === 'done') {
              await Promise.all([fetchProposals(), fetchFiles()])
            } else if (event === 'error') {
              appendDelta(convId, asstId, `\n\n⚠ Error: ${payload.message}`)
            }
          }
        }
      } catch (err) {
        appendDelta(convId, asstId, `\n\n⚠ Network error: ${String(err)}`)
      } finally {
        setIsStreaming(false)
      }
    },
    [addMessage, appendDelta, setIsStreaming, fetchProposals, fetchFiles],
  )

  const sendMessage = useCallback(
    async (text: string) => {
      const convId = activeConversationId ?? newConversation()
      addMessage(convId, { role: 'user', content: text })

      // If the user is viewing an older variant (not the most recent run), capture its
      // assistant messages so the backend can continue from the correct history context.
      const conv = useChatStore.getState().conversations.find((c) => c.id === convId)
      const vr = conv?.variantRuns
      const variantOverride =
        vr && vr.activeIndex < vr.runs.length - 1
          ? vr.runs[vr.activeIndex]
              .filter((m) => m.role === 'assistant')
              .map((m) => ({ role: 'assistant', content: m.content }))
          : null

      clearVariantRuns(convId)
      await stream(text, convId, false, variantOverride)
    },
    [activeConversationId, newConversation, addMessage, clearVariantRuns, stream],
  )

  const rerunLastMessage = useCallback(async () => {
    const convId = activeConversationId
    if (!convId) return
    const lastUserText = startRerun(convId)
    if (!lastUserText) return
    try {
      await stream(lastUserText, convId, true)
    } finally {
      finalizeRerun(convId)
    }
  }, [activeConversationId, startRerun, finalizeRerun, stream])

  return { sendMessage, rerunLastMessage }
}

