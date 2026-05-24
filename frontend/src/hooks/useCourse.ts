import { useEffect } from 'react'
import { useChatStore } from '../store/chatStore'
import { useCourseStore } from '../store/courseStore'

export function useCourse() {
  const fetchFiles = useCourseStore((s) => s.fetchFiles)
  const fetchProposals = useCourseStore((s) => s.fetchProposals)
  const fetchConversations = useChatStore((s) => s.fetchConversations)

  // Load initial data on mount
  useEffect(() => {
    fetchFiles()
    fetchProposals()
    fetchConversations()
  }, [fetchFiles, fetchProposals, fetchConversations])
}
