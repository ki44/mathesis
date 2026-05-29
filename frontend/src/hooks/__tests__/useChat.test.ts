import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useChatStore } from '../../store/chatStore'
import type { ChatMessage } from '../../types'
import { useChat } from '../useChat'

const CONV_ID = 'test-conv'

function makeChatMessage(role: ChatMessage['role'], content: string, id = crypto.randomUUID()): ChatMessage {
  return { id, role, content }
}

function emptyStream() {
  return new Response(new ReadableStream({ start: (c) => c.close() }), { status: 200 })
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(emptyStream()))
  useChatStore.setState({
    conversations: [{ id: CONV_ID, title: 'Test', messages: [], variantRuns: null, messagesLoaded: true }],
    activeConversationId: CONV_ID,
    isStreaming: false,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('sendMessage – variant_override', () => {
  it('sends null variant_override when there are no variant runs', async () => {
    // variantRuns is null (set in beforeEach)
    const { result } = renderHook(() => useChat())
    await act(async () => { await result.current.sendMessage('hello') })

    const body = JSON.parse((vi.mocked(global.fetch).mock.calls[0][1] as RequestInit).body as string)
    expect(body.variant_override).toBeNull()
  })

  it('sends null variant_override when the latest variant is active', async () => {
    const v1 = [makeChatMessage('assistant', 'Answer 1')]
    const v2 = [makeChatMessage('assistant', 'Answer 2')]
    useChatStore.setState({
      conversations: [{
        id: CONV_ID,
        title: 'Test',
        messages: [makeChatMessage('user', 'Turn 5'), ...v2],
        variantRuns: { runs: [v1, v2], activeIndex: 1 }, // latest variant selected
        messagesLoaded: true,
      }],
      activeConversationId: CONV_ID,
      isStreaming: false,
    })

    const { result } = renderHook(() => useChat())
    await act(async () => { await result.current.sendMessage('new message') })

    const body = JSON.parse((vi.mocked(global.fetch).mock.calls[0][1] as RequestInit).body as string)
    expect(body.variant_override).toBeNull()
  })

  it('sends the selected variant messages (assistant only) when an older variant is active', async () => {
    const v1: ChatMessage[] = [
      makeChatMessage('assistant', 'Answer 1'),
      makeChatMessage('tool_call', 'some_tool'), // should be filtered out
    ]
    const v2 = [makeChatMessage('assistant', 'Answer 2')]
    useChatStore.setState({
      conversations: [{
        id: CONV_ID,
        title: 'Test',
        messages: [makeChatMessage('user', 'Turn 5'), ...v1],
        variantRuns: { runs: [v1, v2], activeIndex: 0 }, // older variant selected
        messagesLoaded: true,
      }],
      activeConversationId: CONV_ID,
      isStreaming: false,
    })

    const { result } = renderHook(() => useChat())
    await act(async () => { await result.current.sendMessage('new message') })

    const body = JSON.parse((vi.mocked(global.fetch).mock.calls[0][1] as RequestInit).body as string)
    // Only the assistant message from v1; tool_call is excluded
    expect(body.variant_override).toEqual([{ role: 'assistant', content: 'Answer 1' }])
  })
})
