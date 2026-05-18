import { create } from 'zustand'
import type { CourseFile, Proposal } from '../types'

interface CourseState {
  files: CourseFile[]
  activeFilename: string | null
  proposals: Record<string, Proposal>
  fileRevisions: Record<string, number>

  setActiveFilename: (filename: string | null) => void
  fetchFiles: () => Promise<void>
  fetchProposals: () => Promise<void>
  saveFile: (filename: string, content: string) => Promise<void>
  applyChanges: (filename: string, content: string) => Promise<void>
  rejectProposal: (filename: string) => Promise<void>
  deleteFile: (filename: string) => Promise<void>
}

export const useCourseStore = create<CourseState>((set, get) => ({
  files: [],
  activeFilename: null,
  proposals: {},
  fileRevisions: {},

  setActiveFilename: (filename) => set({ activeFilename: filename }),

  fetchFiles: async () => {
    const res = await fetch('/api/courses')
    if (!res.ok) return
    const files: CourseFile[] = await res.json()
    set({ files })
  },

  fetchProposals: async () => {
    const res = await fetch('/api/proposals')
    if (!res.ok) return
    const list: Proposal[] = await res.json()
    const proposals: Record<string, Proposal> = {}
    for (const p of list) proposals[p.filename] = p
    set({ proposals })
    // Also refresh file list so sidebar stays in sync
    get().fetchFiles()
  },

  applyChanges: async (filename, content) => {
    const res = await fetch(`/api/courses/${encodeURIComponent(filename)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })
    if (!res.ok) throw new Error('Failed to apply changes')
    const updated: CourseFile = await res.json()
    set((state) => {
      const proposals = { ...state.proposals }
      delete proposals[filename]
      return {
        files: state.files.map((f) => (f.filename === filename ? updated : f)),
        proposals,
        fileRevisions: { ...state.fileRevisions, [filename]: (state.fileRevisions[filename] ?? 0) + 1 },
      }
    })
  },

  saveFile: async (filename, content) => {
    const res = await fetch(`/api/courses/${encodeURIComponent(filename)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })
    if (!res.ok) throw new Error('Failed to save')
    const updated: CourseFile = await res.json()
    set((state) => ({
      files: state.files.map((f) => (f.filename === filename ? updated : f)),
    }))
  },

  rejectProposal: async (filename) => {
    await fetch(`/api/proposals/${encodeURIComponent(filename)}`, { method: 'DELETE' })
    set((state) => {
      const proposals = { ...state.proposals }
      delete proposals[filename]
      return { proposals }
    })
  },

  deleteFile: async (filename) => {
    const res = await fetch(`/api/courses/${encodeURIComponent(filename)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error('Failed to delete')
    set((state) => {
      const proposals = { ...state.proposals }
      delete proposals[filename]
      return {
        files: state.files.filter((f) => f.filename !== filename),
        proposals,
        activeFilename: state.activeFilename === filename ? null : state.activeFilename,
      }
    })
  },
}))
