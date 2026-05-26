import { create } from 'zustand'
import type { ChatMessage } from '../types'

export interface VariantRuns {
  runs: ChatMessage[][]  // each entry = assistant messages from one run
  activeIndex: number
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  variantRuns: VariantRuns | null  // set when the last user message has been rerun
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
  renameConversation: (id: string, title: string) => Promise<void>
  forkConversation: (convId: string, msgIndex: number) => Promise<void>
  addMessage: (convId: string, msg: Omit<ChatMessage, 'id'>) => string
  appendDelta: (convId: string, msgId: string, delta: string) => void
  setIsStreaming: (value: boolean) => void
  startRerun: (convId: string) => string  // saves current tail as run[0], returns last user msg text
  finalizeRerun: (convId: string) => void  // called after streaming; moves new tail into runs
  navigateVariant: (convId: string, direction: 1 | -1) => void
  clearVariantRuns: (convId: string) => void
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
      variantRuns: null,
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
    const conv: Conversation = { id, title: 'New conversation', messages: [], variantRuns: null, messagesLoaded: true }
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
              variantRuns: null,
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
    const { conversations: snapshot, activeConversationId: activeSnapshot } = get()
    set((state) => {
      const conversations = state.conversations.filter((c) => c.id !== id)
      const activeConversationId =
        state.activeConversationId === id
          ? (conversations[0]?.id ?? null)
          : state.activeConversationId
      return { conversations, activeConversationId }
    })
    const res = await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
    if (!res.ok) set({ conversations: snapshot, activeConversationId: activeSnapshot })
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

  renameConversation: async (id, title) => {
    // Optimistic update
    set((state) => ({
      conversations: state.conversations.map((c) => (c.id === id ? { ...c, title } : c)),
    }))
    const res = await fetch(`/api/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    })
    if (!res.ok) {
      // Revert on failure — refetch
      await get().fetchConversations()
    }
  },

  forkConversation: async (convId, msgIndex) => {
    const res = await fetch(`/api/conversations/${convId}/fork`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message_index: msgIndex }),
    })
    if (!res.ok) return
    const summary = await res.json() as { id: string; title: string; created_at: string; updated_at: string }
    const newConv: Conversation = {
      id: summary.id,
      title: summary.title,
      messages: [],
      variantRuns: null,
      messagesLoaded: false,
    }
    set((state) => ({
      conversations: [newConv, ...state.conversations],
      activeConversationId: newConv.id,
    }))
    await get().selectConversation(newConv.id)
  },

  // Captures the current assistant tail as run[0] before streaming a rerun.
  // Returns the text of the last user message so useChat can re-send it.
  startRerun: (convId) => {
    let lastUserText = ''
    set((state) => ({
      conversations: state.conversations.map((c) => {
        if (c.id !== convId) return c
        // Find the last user message index
        let lastUserIdx = -1
        for (let i = c.messages.length - 1; i >= 0; i--) {
          if (c.messages[i].role === 'user') { lastUserIdx = i; break }
        }
        if (lastUserIdx === -1) return c
        lastUserText = c.messages[lastUserIdx].content
        const tail = c.messages.slice(lastUserIdx + 1)
        const runs = c.variantRuns ? c.variantRuns.runs : [tail]
        return {
          ...c,
          // Trim messages back to user message + everything before; new assistant messages stream in
          messages: c.messages.slice(0, lastUserIdx + 1),
          variantRuns: { runs, activeIndex: runs.length },  // next slot = runs.length
        }
      }),
    }))
    return lastUserText
  },

  // Called after streaming completes; saves the newly streamed messages as the latest run.
  finalizeRerun: (convId) =>
    set((state) => ({
      conversations: state.conversations.map((c) => {
        if (c.id !== convId || !c.variantRuns) return c
        let lastUserIdx = -1
        for (let i = c.messages.length - 1; i >= 0; i--) {
          if (c.messages[i].role === 'user') { lastUserIdx = i; break }
        }
        const tail = c.messages.slice(lastUserIdx + 1)
        const runs = [...c.variantRuns.runs]
        runs[c.variantRuns.activeIndex] = tail
        return { ...c, variantRuns: { runs, activeIndex: c.variantRuns.activeIndex } }
      }),
    })),

  navigateVariant: (convId, direction) =>
    set((state) => ({
      conversations: state.conversations.map((c) => {
        if (c.id !== convId || !c.variantRuns) return c
        const { runs, activeIndex } = c.variantRuns
        const nextIndex = activeIndex + direction
        if (nextIndex < 0 || nextIndex >= runs.length) return c
        // Find last user message
        let lastUserIdx = -1
        for (let i = c.messages.length - 1; i >= 0; i--) {
          if (c.messages[i].role === 'user') { lastUserIdx = i; break }
        }
        const base = c.messages.slice(0, lastUserIdx + 1)
        return {
          ...c,
          messages: [...base, ...runs[nextIndex]],
          variantRuns: { runs, activeIndex: nextIndex },
        }
      }),
    })),

  clearVariantRuns: (convId) =>
    set((state) => ({
      conversations: state.conversations.map((c) => (c.id === convId ? { ...c, variantRuns: null } : c)),
    })),
}))

