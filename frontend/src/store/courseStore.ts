import { create } from 'zustand'
import type { ClipboardItem, CourseFile, FolderEntry, Proposal } from '../types'

interface CourseState {
  files: CourseFile[]
  folders: FolderEntry[]
  activeFilename: string | null
  openFiles: string[]
  pinnedFiles: string[]
  clipboard: ClipboardItem | null
  proposals: Record<string, Proposal>
  fileRevisions: Record<string, number>

  // ── selection ──────────────────────────────────────────────────────────────
  setActiveFilename: (filename: string | null) => void
  openFile: (filename: string) => void
  closeFile: (filename: string) => void
  closeMultiple: (filenames: string[]) => void
  pinFile: (filename: string) => void
  unpinFile: (filename: string) => void

  // ── fetch ──────────────────────────────────────────────────────────────────
  fetchFiles: () => Promise<void>
  fetchFolders: () => Promise<void>
  fetchProposals: () => Promise<void>

  // ── file CRUD ──────────────────────────────────────────────────────────────
  createFile: (filename: string, content?: string) => Promise<void>
  saveFile: (filename: string, content: string) => Promise<void>
  applyChanges: (filename: string, content: string) => Promise<void>
  rejectProposal: (filename: string) => Promise<void>
  deleteFile: (filename: string) => Promise<void>
  renameFile: (oldFilename: string, newFilename: string) => Promise<void>
  copyFile: (filename: string, newFilename?: string) => Promise<CourseFile>

  // ── folder CRUD ────────────────────────────────────────────────────────────
  createFolder: (path: string) => Promise<void>
  deleteFolder: (path: string) => Promise<void>
  renameFolder: (oldPath: string, newPath: string) => Promise<void>

  // ── clipboard ──────────────────────────────────────────────────────────────
  setClipboard: (item: ClipboardItem | null) => void
  pasteItem: (targetFolder: string | null) => Promise<void>

  // ── undo ───────────────────────────────────────────────────────────────────
  undoStack: Array<{ kind: 'file'; filename: string; content: string } | { kind: 'folder'; path: string; files: Array<{ filename: string; content: string }> }>
  undoDelete: () => Promise<void>
}

export const useCourseStore = create<CourseState>((set, get) => ({
  files: [],
  folders: [],
  activeFilename: null,
  openFiles: [],
  pinnedFiles: [],
  clipboard: null,
  proposals: {},
  fileRevisions: {},
  undoStack: [],

  // ── selection ──────────────────────────────────────────────────────────────

  setActiveFilename: (filename) => {
    set({ activeFilename: filename })
    if (filename) {
      get().openFile(filename)
      localStorage.setItem('mathesis:activeFilename', filename)
    } else {
      localStorage.removeItem('mathesis:activeFilename')
    }
  },

  openFile: (filename) =>
    set((state) => ({
      openFiles: state.openFiles.includes(filename) ? state.openFiles : [...state.openFiles, filename],
    })),

  closeFile: (filename) =>
    set((state) => {
      const openFiles = state.openFiles.filter((f) => f !== filename)
      const activeFilename =
        state.activeFilename === filename
          ? ([...openFiles].reverse().find((f) => state.pinnedFiles.includes(f)) ?? openFiles[openFiles.length - 1] ?? null)
          : state.activeFilename
      return { openFiles, activeFilename }
    }),

  pinFile: (filename) =>
    set((state) => ({
      pinnedFiles: state.pinnedFiles.includes(filename) ? state.pinnedFiles : [...state.pinnedFiles, filename],
    })),

  unpinFile: (filename) =>
    set((state) => ({ pinnedFiles: state.pinnedFiles.filter((f) => f !== filename) })),

  closeMultiple: (filenames) =>
    set((state) => {
      const toClose = new Set(filenames)
      const openFiles = state.openFiles.filter((f) => !toClose.has(f))
      const activeFilename = toClose.has(state.activeFilename ?? '')
        ? (openFiles[openFiles.length - 1] ?? null)
        : state.activeFilename
      return { openFiles, activeFilename }
    }),

  // ── fetch ──────────────────────────────────────────────────────────────────

  fetchFiles: async () => {
    const res = await fetch('/api/courses')
    if (!res.ok) return
    const files: CourseFile[] = await res.json()
    set({ files })
    const saved = localStorage.getItem('mathesis:activeFilename')
    if (saved && files.some((f) => f.filename === saved)) get().setActiveFilename(saved)
  },

  fetchFolders: async () => {
    const res = await fetch('/api/folders')
    if (!res.ok) return
    const folders: FolderEntry[] = await res.json()
    set({ folders })
  },

  fetchProposals: async () => {
    const res = await fetch('/api/proposals')
    if (!res.ok) return
    const list: Proposal[] = await res.json()
    const proposals: Record<string, Proposal> = {}
    for (const p of list) proposals[p.filename] = p
    set({ proposals })
  },

  // ── file CRUD ──────────────────────────────────────────────────────────────

  createFile: async (filename, content = '') => {
    const res = await fetch('/api/courses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, content }),
    })
    if (!res.ok) throw new Error(await res.text())
    const file: CourseFile = await res.json()
    set((state) => ({ files: [...state.files, file].sort((a, b) => a.filename.localeCompare(b.filename)) }))
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
    const res = await fetch(`/api/proposals/${encodeURIComponent(filename)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error('Failed to reject proposal')
    set((state) => {
      const proposals = { ...state.proposals }
      delete proposals[filename]
      return { proposals }
    })
  },

  deleteFile: async (filename) => {
    const snapshot = get()
    const content = snapshot.files.find((f) => f.filename === filename)?.content ?? ''
    set((state) => {
      const files = state.files.filter((f) => f.filename !== filename)
      const openFiles = state.openFiles.filter((f) => f !== filename)
      const pinnedFiles = state.pinnedFiles.filter((f) => f !== filename)
      const proposals = { ...state.proposals }
      delete proposals[filename]
      const activeFilename =
        state.activeFilename === filename ? (openFiles[openFiles.length - 1] ?? null) : state.activeFilename
      return { files, openFiles, pinnedFiles, proposals, activeFilename, undoStack: [...state.undoStack, { kind: 'file' as const, filename, content }] }
    })
    const res = await fetch(`/api/courses/${encodeURIComponent(filename)}`, { method: 'DELETE' })
    if (!res.ok) {
      set({
        files: snapshot.files,
        openFiles: snapshot.openFiles,
        pinnedFiles: snapshot.pinnedFiles,
        proposals: snapshot.proposals,
        activeFilename: snapshot.activeFilename,
        undoStack: snapshot.undoStack,
      })
    }
  },

  renameFile: async (oldFilename, newFilename) => {
    const res = await fetch('/api/file-ops/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_filename: oldFilename, new_filename: newFilename }),
    })
    if (!res.ok) throw new Error(await res.text())
    const updated: CourseFile = await res.json()
    set((state) => {
      const proposals = { ...state.proposals }
      if (proposals[oldFilename]) {
        proposals[newFilename] = { ...proposals[oldFilename], filename: newFilename }
        delete proposals[oldFilename]
      }
      return {
        files: state.files.map((f) => (f.filename === oldFilename ? updated : f)),
        openFiles: state.openFiles.map((f) => (f === oldFilename ? newFilename : f)),
        pinnedFiles: state.pinnedFiles.map((f) => (f === oldFilename ? newFilename : f)),
        activeFilename: state.activeFilename === oldFilename ? newFilename : state.activeFilename,
        proposals,
      }
    })
  },

  copyFile: async (filename, newFilename) => {
    const res = await fetch('/api/file-ops/copy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, new_filename: newFilename ?? null }),
    })
    if (!res.ok) throw new Error(await res.text())
    const created: CourseFile = await res.json()
    set((state) => ({
      files: [...state.files, created].sort((a, b) => a.filename.localeCompare(b.filename)),
    }))
    return created
  },

  // ── folder CRUD ────────────────────────────────────────────────────────────

  createFolder: async (path) => {
    const res = await fetch('/api/folders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    })
    if (!res.ok) throw new Error(await res.text())
    const folder: FolderEntry = await res.json()
    set((state) => ({ folders: [...state.folders, folder].sort((a, b) => a.path.localeCompare(b.path)) }))
  },

  deleteFolder: async (path) => {
    const prefix = path + '/'
    const filesToSave = get().files
      .filter((f) => f.filename.startsWith(prefix))
      .map((f) => ({ filename: f.filename, content: f.content }))
    set((state) => ({
      undoStack: [...state.undoStack, { kind: 'folder' as const, path, files: filesToSave }],
      folders: state.folders.filter((f) => f.path !== path && !f.path.startsWith(prefix)),
      files: state.files.filter((f) => !f.filename.startsWith(prefix)),
    }))
    const res = await fetch(`/api/folders/${encodeURIComponent(path)}`, { method: 'DELETE' })
    if (!res.ok) {
      await get().fetchFiles()
      await get().fetchFolders()
      throw new Error(await res.text())
    }
  },

  renameFolder: async (oldPath, newPath) => {
    const res = await fetch('/api/folder-ops/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_path: oldPath, new_path: newPath }),
    })
    if (!res.ok) throw new Error(await res.text())
    const updatedFiles: CourseFile[] = await res.json()
    const prefix = oldPath + '/'
    set((state) => {
      const proposals = { ...state.proposals }
      for (const f of updatedFiles) {
        const oldName = oldPath + '/' + f.filename.slice(newPath.length + 1)
        if (proposals[oldName]) {
          proposals[f.filename] = { ...proposals[oldName], filename: f.filename }
          delete proposals[oldName]
        }
      }
      return {
        folders: state.folders.map((f) =>
          f.path === oldPath
            ? { ...f, path: newPath }
            : f.path.startsWith(prefix)
              ? { ...f, path: newPath + '/' + f.path.slice(prefix.length) }
              : f,
        ),
        files: state.files.map((f) => {
          if (!f.filename.startsWith(prefix)) return f
          const updated = updatedFiles.find((u) => u.filename === newPath + '/' + f.filename.slice(prefix.length))
          return updated ?? f
        }),
        openFiles: state.openFiles.map((f) => (f.startsWith(prefix) ? newPath + '/' + f.slice(prefix.length) : f)),
        pinnedFiles: state.pinnedFiles.map((f) =>
          f.startsWith(prefix) ? newPath + '/' + f.slice(prefix.length) : f,
        ),
        activeFilename: state.activeFilename?.startsWith(prefix)
          ? newPath + '/' + state.activeFilename.slice(prefix.length)
          : state.activeFilename,
        proposals,
      }
    })
  },

  // ── undo ───────────────────────────────────────────────────────────────────

  undoDelete: async () => {
    const { undoStack, createFile, createFolder } = get()
    if (undoStack.length === 0) return
    const entry = undoStack[undoStack.length - 1]
    set((state) => ({ undoStack: state.undoStack.slice(0, -1) }))
    if (entry.kind === 'file') {
      await createFile(entry.filename, entry.content)
    } else {
      await createFolder(entry.path)
      for (const f of entry.files) await createFile(f.filename, f.content)
    }
  },

  // ── clipboard ──────────────────────────────────────────────────────────────

  setClipboard: (item) => set({ clipboard: item }),

  pasteItem: async (targetFolder) => {
    const { clipboard, files, copyFile, renameFile, renameFolder, createFolder } = get()
    if (!clipboard) return

    const buildTarget = (sourcePath: string): string => {
      const basename = sourcePath.includes('/') ? sourcePath.split('/').pop()! : sourcePath
      return targetFolder ? `${targetFolder}/${basename}` : basename
    }

    if (clipboard.kind === 'file') {
      const target = buildTarget(clipboard.path)
      if (clipboard.type === 'copy') {
        await copyFile(clipboard.path, target === clipboard.path ? undefined : target)
      } else {
        if (target !== clipboard.path) await renameFile(clipboard.path, target)
        set({ clipboard: null })
      }
    } else {
      const target = buildTarget(clipboard.path)
      if (clipboard.type === 'copy') {
        const prefix = clipboard.path + '/'
        const toCopy = files.filter((f) => f.filename.startsWith(prefix))
        const subFolders = get().folders.filter(
          (f) => f.path !== clipboard.path && f.path.startsWith(prefix),
        )
        await createFolder(target).catch(() => {})
        for (const sf of subFolders) {
          await createFolder(target + '/' + sf.path.slice(prefix.length)).catch(() => {})
        }
        for (const f of toCopy) {
          await copyFile(f.filename, `${target}/${f.filename.slice(prefix.length)}`)
        }
      } else {
        await renameFolder(clipboard.path, target)
        set({ clipboard: null })
      }
    }
  },
}))
