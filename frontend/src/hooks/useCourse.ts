import { useEffect } from 'react'
import { useChatStore } from '../store/chatStore'
import { useCourseStore } from '../store/courseStore'

export function useCourse() {
  const fetchFiles = useCourseStore((s) => s.fetchFiles)
  const fetchFolders = useCourseStore((s) => s.fetchFolders)
  const fetchProposals = useCourseStore((s) => s.fetchProposals)
  const fetchConversations = useChatStore((s) => s.fetchConversations)

  useEffect(() => {
    fetchFiles()
    fetchFolders()
    fetchProposals()
    fetchConversations()
  }, [fetchFiles, fetchFolders, fetchProposals, fetchConversations])
}
