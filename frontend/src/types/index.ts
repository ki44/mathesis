export interface CourseFile {
  filename: string
  content: string
  updated_at: string
}

export interface Proposal {
  filename: string
  proposed_content: string
  description: string
  created_at: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'tool_call'
  content: string
}

export interface HunkDecision {
  hunkIndex: number
  accepted: boolean | null
}
