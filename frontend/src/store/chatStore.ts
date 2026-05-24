import { create } from 'zustand'
import type { ChatMessage } from '../types'

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  messagesLoaded: boolean
}

interface ChatState {
  conversations: Conversation[]
  activeConversationId: string | null
  isStreaming: boolean

  fetchConversations: () => Promise<void>
  newConversation: () => string
  selectConversation: (id: string) => Promise<void>
  deleteConversation: (id: string) => Promise<void>
  addMessage: (convId: string, msg: Omit<ChatMessage, 'id'>) => string
  appendDelta: (convId: string, msgId: string, delta: string) => void
  setIsStreaming: (value: boolean) => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  activeConversationId: null,
  isStreaming: false,

  fetchConversations: async () => {
    const res = await fetch('/api/conversations')
    if (!res.ok) return
    const summaries = await res.json() as Array<{ id: string; title: string; created_at: string; updated_at: string }>
    const conversations: Conversation[] = summaries.map((s) => ({
      id: s.id,
      title: s.title,
      messages: [],
      messagesLoaded: false,
    }))
    const firstId = conversations[0]?.id ?? null
    set({ conversations, activeConversationId: firstId })
    if (firstId) {
      await get().selectConversation(firstId)
    }
  },

  newConversation: () => {
    const id = crypto.randomUUID()
    const conv: Conversation = { id, title: 'New conversation', messages: [], messagesLoaded: true }
    set((state) => ({
      conversations: [conv, ...state.conversations],
      activeConversationId: id,
    }))
    return id
  },

  selectConversation: async (id: string) => {
    set({ activeConversationId: id })
    const conv = get().conversations.find((c) => c.id === id)
    if (!conv || conv.messagesLoaded) return
    const res = await fetch(`/api/conversations/${id}/messages`)
    if (!res.ok) return
    const msgs = await res.json() as Array<{ role: string; content: string }>
    set((state) => ({
      conversations: state.conversations.map((c) =>
        c.id !== id
          ? c
          : {
              ...c,
              messagesLoaded: true,
              messages: msgs.map((m) => ({
                id: crypto.randomUUID(),
                role: m.role as ChatMessage['role'],
                content: m.content,
              })),
            },
      ),
    }))
  },

  deleteConversation: async (id: string) => {
    const snapshot = get().conversations
    set((state) => {
      const conversations = state.conversations.filter((c) => c.id !== id)
      const activeConversationId =
        state.activeConversationId === id
          ? (conversations[0]?.id ?? null)
          : state.activeConversationId
      return { conversations, activeConversationId }
    })
    const res = await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
    if (!res.ok) set({ conversations: snapshot })
  },

  addMessage: (convId, msg) => {
    const msgId = crypto.randomUUID()
    set((state) => ({
      conversations: state.conversations.map((c) => {
        if (c.id !== convId) return c
        const messages = [...c.messages, { ...msg, id: msgId }]
        const title =
          c.title === 'New conversation' && msg.role === 'user'
            ? msg.content.slice(0, 40) + (msg.content.length > 40 ? '…' : '')
            : c.title
        return { ...c, messages, title }
      }),
    }))
    return msgId
  },

  appendDelta: (convId, msgId, delta) =>
    set((state) => ({
      conversations: state.conversations.map((c) =>
        c.id !== convId
          ? c
          : {
              ...c,
              messages: c.messages.map((m) =>
                m.id === msgId ? { ...m, content: m.content + delta } : m,
              ),
            },
      ),
    })),

  setIsStreaming: (value) => set({ isStreaming: value }),
}))

