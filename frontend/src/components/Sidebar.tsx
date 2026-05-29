import { useCallback, useEffect, useRef, useState } from 'react'
import { useCourseStore } from '../store/courseStore'
import { useContextMenuClose } from '../hooks/useContextMenuClose'
import { useThemeStore } from '../store/themeStore'
import { ctxMenuStyle } from './ctxMenuStyle'

// ─── Types ────────────────────────────────────────────────────────────────────

interface TreeNode {
  kind: 'folder' | 'file'
  name: string
  path: string
  children?: TreeNode[]
}

type CtxMenu = { x: number; y: number; node: TreeNode }
type Creating = { parentPath: string | null; kind: 'file' | 'folder' } | null

// ─── Tree prop types ─────────────────────────────────────────────────────────

interface TreeState {
  collapsedFolders: Set<string>
  activeFilename: string | null
  focusedPath: string | null
  ctxMenuPath: string | null
  renamingPath: string | null
  renameValue: string
  proposals: Record<string, unknown>
  clipboard: { type: string; kind: string; path: string } | null
  dragPath: string | null
  dragOverPath: string | null
  creating: Creating
  createValue: string
}

interface TreeHandlers {
  toggleFolder: (path: string) => void
  setFocusedPath: (p: string | null) => void
  setCtxMenu: (m: CtxMenu | null) => void
  setActiveFilename: (f: string | null) => void
  setRenameValue: (v: string) => void
  renameInputRef: React.RefObject<HTMLInputElement | null>
  commitRename: () => void
  setRenamingPath: (p: string | null) => void
  handleDragStart: (e: React.DragEvent, path: string) => void
  handleDragOver: (e: React.DragEvent, node: TreeNode) => void
  handleDrop: (e: React.DragEvent, node: TreeNode) => void
  handleDragEnd: () => void
  setCreateValue: (v: string) => void
  createInputRef: React.RefObject<HTMLInputElement | null>
  commitCreate: () => void
  cancelCreate: () => void
}

// ─── Tree builder ─────────────────────────────────────────────────────────────

function buildTree(filePaths: string[], folderPaths: string[]): TreeNode[] {
  const root: TreeNode[] = []
  const folderMap = new Map<string, TreeNode>()

  const ensureFolder = (path: string): TreeNode => {
    if (folderMap.has(path)) return folderMap.get(path)!
    const segments = path.split('/')
    const name = segments[segments.length - 1]
    const node: TreeNode = { kind: 'folder', name, path, children: [] }
    folderMap.set(path, node)

    if (segments.length === 1) {
      root.push(node)
    } else {
      const parentPath = segments.slice(0, -1).join('/')
      const parent = ensureFolder(parentPath)
      parent.children!.push(node)
    }
    return node
  }

  // Register explicit empty folders first
  for (const fp of [...folderPaths].sort()) ensureFolder(fp)

  // Add files
  for (const filePath of [...filePaths].sort()) {
    const segments = filePath.split('/')
    const name = segments[segments.length - 1]
    const fileNode: TreeNode = { kind: 'file', name, path: filePath }
    if (segments.length === 1) {
      root.push(fileNode)
    } else {
      const parentPath = segments.slice(0, -1).join('/')
      const parent = ensureFolder(parentPath)
      parent.children!.push(fileNode)
    }
  }

  // Sort children: folders first, then files, alphabetically
  const sortChildren = (nodes: TreeNode[]) => {
    nodes.sort((a, b) => {
      if (a.kind !== b.kind) return a.kind === 'folder' ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    for (const n of nodes) if (n.children) sortChildren(n.children)
  }
  sortChildren(root)

  return root
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function findNode(nodes: TreeNode[], path: string): TreeNode | null {
  for (const n of nodes) {
    if (n.path === path) return n
    if (n.children) {
      const found = findNode(n.children, path)
      if (found) return found
    }
  }
  return null
}

// ─── Icons ──────────────────────────────────────────────────────────────────

const IcoNewFile = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="12" y1="18" x2="12" y2="12" />
    <line x1="9" y1="15" x2="15" y2="15" />
  </svg>
)

const IcoNewFolder = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
    <line x1="12" y1="11" x2="12" y2="17" />
    <line x1="9" y1="14" x2="15" y2="14" />
  </svg>
)

const IcoSun = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
)

const IcoMoon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
  </svg>
)

const FILE_ICON_MAP: Record<string, [string, string]> = {
  md:       ['MD',  '#519aba'],
  mdx:      ['MDX', '#519aba'],
  ts:       ['TS',  '#3178c6'],
  tsx:      ['TSX', '#3178c6'],
  js:       ['JS',  '#cbcb41'],
  jsx:      ['JSX', '#cbcb41'],
  json:     ['{}',  '#d4bb6e'],
  css:      ['CSS', '#7b5ea7'],
  html:     ['HTM', '#e44d26'],
  py:       ['PY',  '#3572a5'],
  yaml:     ['YML', '#cc3e44'],
  yml:      ['YML', '#cc3e44'],
  txt:      ['TXT', '#888888'],
  _default: ['•',   '#888888'],
}

function FileIcon({ filename }: { filename: string }) {
  const dot = filename.lastIndexOf('.')
  const ext = dot === -1 ? '' : filename.slice(dot + 1).toLowerCase()
  const [label, color] = FILE_ICON_MAP[ext] ?? FILE_ICON_MAP._default
  return (
    <span style={{ width: 15, flexShrink: 0, fontSize: 9, fontWeight: 700, letterSpacing: '-0.3px', color, lineHeight: 1, textAlign: 'center', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
      {label}
    </span>
  )
}

function getParent(path: string): string | null {
  const idx = path.lastIndexOf('/')
  return idx === -1 ? null : path.slice(0, idx)
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

export function Sidebar() {
  const files = useCourseStore((s) => s.files)
  const folders = useCourseStore((s) => s.folders)
  const proposals = useCourseStore((s) => s.proposals)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const clipboard = useCourseStore((s) => s.clipboard)
  const setActiveFilename = useCourseStore((s) => s.setActiveFilename)
  const deleteFile = useCourseStore((s) => s.deleteFile)
  const deleteFolder = useCourseStore((s) => s.deleteFolder)
  const renameFile = useCourseStore((s) => s.renameFile)
  const renameFolder = useCourseStore((s) => s.renameFolder)
  const createFile = useCourseStore((s) => s.createFile)
  const createFolder = useCourseStore((s) => s.createFolder)
  const copyFile = useCourseStore((s) => s.copyFile)
  const setClipboard = useCourseStore((s) => s.setClipboard)
  const pasteItem = useCourseStore((s) => s.pasteItem)
  const undoLast = useCourseStore((s) => s.undoLast)
  const undoStack = useCourseStore((s) => s.undoStack)

  const { theme, toggle } = useThemeStore()

  const [collapsedFolders, setCollapsedFolders] = useState<Set<string>>(new Set())
  const [ctxMenu, setCtxMenu] = useState<CtxMenu | null>(null)
  const [focusedPath, setFocusedPath] = useState<string | null>(null)
  const [renamingPath, setRenamingPath] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const renameInputRef = useRef<HTMLInputElement>(null)
  const [dragPath, setDragPath] = useState<string | null>(null)
  const [dragOverPath, setDragOverPath] = useState<string | null>(null)
  const [creating, setCreating] = useState<Creating>(null)
  const [createValue, setCreateValue] = useState('')
  const createInputRef = useRef<HTMLInputElement>(null)
  const [pendingDelete, setPendingDelete] = useState<TreeNode | null>(null)

  const closeCtxMenu = useCallback(() => setCtxMenu(null), [])
  useContextMenuClose(closeCtxMenu)

  useEffect(() => {
    if (renamingPath) renameInputRef.current?.focus()
  }, [renamingPath])

  useEffect(() => {
    if (creating) setTimeout(() => createInputRef.current?.focus(), 30)
  }, [creating])

  const tree = buildTree(files.map((f) => f.filename), folders.map((f) => f.path))

  const toggleFolder = (path: string) =>
    setCollapsedFolders((prev) => {
      const next = new Set(prev)
      next.has(path) ? next.delete(path) : next.add(path)
      return next
    })

  const startRename = (node: TreeNode) => {
    setCtxMenu(null)
    setRenamingPath(node.path)
    setRenameValue(node.name)
  }

  const commitRename = async () => {
    if (!renamingPath || !renameValue.trim()) { setRenamingPath(null); return }
    const node = renamingPath
    const dir = node.includes('/') ? node.slice(0, node.lastIndexOf('/') + 1) : ''
    const newPath = dir + renameValue.trim()
    if (newPath !== node) {
      const isFolder = folders.some((f) => f.path === node)
      try {
        if (isFolder) await renameFolder(node, newPath)
        else await renameFile(node, newPath)
      } catch (err) { console.error(err) }
    }
    setRenamingPath(null)
  }

  const requestDelete = (node: TreeNode) => {
    setCtxMenu(null)
    setPendingDelete(node)
  }

  const confirmDelete = async () => {
    if (!pendingDelete) return
    const node = pendingDelete
    setPendingDelete(null)
    if (node.kind === 'folder') await deleteFolder(node.path).catch(() => {})
    else await deleteFile(node.path).catch(() => {})
  }

  const startCreating = (parentPath: string | null, kind: 'file' | 'folder') => {
    setCtxMenu(null)
    if (parentPath) setCollapsedFolders((prev) => { const next = new Set(prev); next.delete(parentPath); return next })
    setCreating({ parentPath, kind })
    setCreateValue('')
  }

  const commitCreate = async () => {
    if (!createValue.trim() || !creating) { setCreating(null); return }
    const path = creating.parentPath ? `${creating.parentPath}/${createValue.trim()}` : createValue.trim()
    try {
      if (creating.kind === 'folder') await createFolder(path)
      else await createFile(path)
    } catch (err) { console.error(err) }
    setCreating(null)
  }

  const cancelCreate = () => setCreating(null)

  const handleDuplicate = (node: TreeNode) => {
    setCtxMenu(null)
    copyFile(node.path).catch(() => {})
  }

  const handlePaste = (targetFolder: string | null) => {
    setCtxMenu(null)
    pasteItem(targetFolder).catch(() => {})
  }

  // ── Drag and drop ──────────────────────────────────────────────────────────

  const handleDragStart = (e: React.DragEvent, path: string) => {
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', path)
    setDragPath(path)
  }

  const handleDragOver = (e: React.DragEvent, node: TreeNode) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDragOverPath(node.kind === 'folder' ? node.path : getParent(node.path))
  }

  const handleDrop = (e: React.DragEvent, node: TreeNode) => {
    e.preventDefault()
    const sourcePath = e.dataTransfer.getData('text/plain') || dragPath
    if (!sourcePath) { setDragPath(null); setDragOverPath(null); return }
    const targetFolder = node.kind === 'folder' ? node.path : getParent(node.path)
    const basename = sourcePath.split('/').pop()!
    const newPath = targetFolder ? `${targetFolder}/${basename}` : basename
    if (newPath !== sourcePath && !newPath.startsWith(sourcePath + '/')) {
      const isFolder = folders.some((f) => f.path === sourcePath)
      if (isFolder) renameFolder(sourcePath, newPath).catch(() => {})
      else renameFile(sourcePath, newPath).catch(() => {})
    }
    setDragPath(null); setDragOverPath(null)
  }

  const handleDragEnd = () => { setDragPath(null); setDragOverPath(null) }

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────

  useEffect(() => {
    const isEditable = (el: EventTarget | null) =>
      el instanceof HTMLElement && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        if (isEditable(e.target)) return
        if (undoStack.length > 0) { e.preventDefault(); undoLast(); return }
      }
      if (e.key === 'Escape') { setPendingDelete(null); return }

      if (!focusedPath || isEditable(e.target)) return
      const node = findNode(tree, focusedPath)
      if (!node) return

      if (e.key === 'F2') {
        e.preventDefault(); startRename(node)
      } else if (e.key === 'Delete') {
        e.preventDefault(); requestDelete(node)
      } else if (e.key === 'c' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault(); setClipboard({ type: 'copy', kind: node.kind, path: node.path })
      } else if (e.key === 'x' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault(); setClipboard({ type: 'cut', kind: node.kind, path: node.path })
      } else if (e.key === 'v' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        const targetFolder = node.kind === 'folder' ? node.path : getParent(node.path)
        handlePaste(targetFolder)
      } else if (e.key === 'd' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        if (node.kind === 'file') handleDuplicate(node)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [focusedPath, tree, clipboard, undoStack]) // eslint-disable-line react-hooks/exhaustive-deps

  const state: TreeState = {
    collapsedFolders,
    activeFilename,
    focusedPath,
    ctxMenuPath: ctxMenu?.node.path ?? null,
    renamingPath, renameValue,
    proposals, clipboard,
    dragPath, dragOverPath,
    creating, createValue,
  }

  const handlers: TreeHandlers = {
    toggleFolder,
    setFocusedPath,
    setCtxMenu,
    setActiveFilename,
    setRenameValue, renameInputRef, commitRename, setRenamingPath,
    handleDragStart, handleDragOver, handleDrop, handleDragEnd,
    setCreateValue, createInputRef, commitCreate, cancelCreate,
  }

  return (
    <div style={{ width: '100%', background: 'var(--bg-2)', display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', fontWeight: 600, fontSize: 12, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
          <span>Courses</span>
          <div style={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <button onClick={() => startCreating(null, 'file')} title="New file" className="icon-btn"><IcoNewFile /></button>
            <button onClick={() => startCreating(null, 'folder')} title="New folder" className="icon-btn"><IcoNewFolder /></button>
            <button onClick={toggle} title="Toggle theme" className="icon-btn">
              {theme === 'dark' ? <IcoSun /> : <IcoMoon />}
            </button>
          </div>
        </div>

        {/* Tree */}
        <div
          style={{ flex: 1, overflowY: 'auto', paddingTop: 4 }}
          onClick={() => setFocusedPath(null)}
          onDragOver={(e) => { e.preventDefault(); setDragOverPath(null) }}
          onDrop={(e) => {
            e.preventDefault()
            const sourcePath = e.dataTransfer.getData('text/plain') || dragPath
            if (!sourcePath) return
            const basename = sourcePath.split('/').pop()!
            if (basename !== sourcePath) {
              const isFolder = folders.some((f) => f.path === sourcePath)
              if (isFolder) renameFolder(sourcePath, basename).catch(() => {})
              else renameFile(sourcePath, basename).catch(() => {})
            }
            setDragPath(null); setDragOverPath(null)
          }}
        >
          {tree.length === 0 && !creating && (
            <p style={{ color: 'var(--text-3)', fontSize: 12, padding: '12px 14px' }}>No courses yet</p>
          )}
          {tree.map((node) => <TreeNodeView key={node.path} node={node} depth={0} state={state} handlers={handlers} />)}
          {creating?.parentPath === null && <CreateRow depth={0} state={state} handlers={handlers} />}
          <div
            style={{ flexGrow: 1, minHeight: 60 }}
            onContextMenu={(e) => {
              e.preventDefault()
              setCtxMenu({ x: e.clientX, y: e.clientY, node: { kind: 'folder', name: '', path: '' } })
            }}
          />
        </div>

        {pendingDelete && (
          <DeleteModal
            node={pendingDelete}
            onConfirm={confirmDelete}
            onCancel={() => setPendingDelete(null)}
          />
        )}

        {/* Context menu */}
        {ctxMenu && (
          <ContextMenu
            menu={ctxMenu}
            clipboard={clipboard}
            onClose={() => setCtxMenu(null)}
            onNewFile={() => startCreating(ctxMenu.node.kind === 'folder' ? ctxMenu.node.path || null : getParent(ctxMenu.node.path), 'file')}
            onNewFolder={() => startCreating(ctxMenu.node.kind === 'folder' ? ctxMenu.node.path || null : getParent(ctxMenu.node.path), 'folder')}
            onRename={() => ctxMenu.node.path ? startRename(ctxMenu.node) : void 0}
            onCopy={() => { setClipboard({ type: 'copy', kind: ctxMenu.node.kind, path: ctxMenu.node.path }); setCtxMenu(null) }}
            onCut={() => { setClipboard({ type: 'cut', kind: ctxMenu.node.kind, path: ctxMenu.node.path }); setCtxMenu(null) }}
            onDuplicate={() => ctxMenu.node.kind === 'file' ? handleDuplicate(ctxMenu.node) : void 0}
            onPaste={() => handlePaste(ctxMenu.node.kind === 'folder' ? ctxMenu.node.path || null : getParent(ctxMenu.node.path))}
            onDelete={() => ctxMenu.node.path ? requestDelete(ctxMenu.node) : void 0}
          />
        )}
      </div>
  )
}

// ─── CreateRow ────────────────────────────────────────────────────────────────

function CreateRow({ depth, state, handlers }: { depth: number; state: TreeState; handlers: TreeHandlers }) {
  const { creating, createValue } = state
  const { setCreateValue, createInputRef, commitCreate, cancelCreate } = handlers
  if (!creating) return null
  const indent = depth * 8 + 4 + 16 + 3  // base + arrow/icon col + gap
  return (
    <div style={{ display: 'flex', alignItems: 'center', paddingLeft: indent, paddingRight: 10, height: 26 }}>
      <input
        ref={createInputRef}
        value={createValue}
        onChange={(e) => setCreateValue(e.target.value)}
        onBlur={commitCreate}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commitCreate()
          if (e.key === 'Escape') cancelCreate()
          e.stopPropagation()
        }}
        placeholder={creating.kind === 'folder' ? 'folder name…' : 'filename.md'}
        style={{ background: 'var(--bg-1)', border: '1px solid #0e639c', color: 'var(--text-bright)', fontSize: 12, padding: '1px 6px', flex: 1, minWidth: 0, outline: 'none', borderRadius: 3 }}
      />
    </div>
  )
}

// ─── TreeNodeView ─────────────────────────────────────────────────────────────

function TreeNodeView({ node, depth, state, handlers }: { node: TreeNode; depth: number; state: TreeState; handlers: TreeHandlers }) {
  const {
    collapsedFolders, activeFilename, focusedPath, ctxMenuPath, renamingPath, renameValue,
    proposals, clipboard, dragPath, dragOverPath, creating,
  } = state
  const {
    toggleFolder, setFocusedPath, setCtxMenu, setActiveFilename, setRenameValue,
    renameInputRef, commitRename, setRenamingPath, handleDragStart, handleDragOver,
    handleDrop, handleDragEnd,
  } = handlers

  const indent = depth * 8 + 4
  const isCollapsed = collapsedFolders.has(node.path)
  const isActive = node.kind === 'file' && node.path === activeFilename
  const isFocused = node.path === focusedPath && node.path !== ctxMenuPath
  const isCtxTarget = !!node.path && node.path === ctxMenuPath
  const isCut = clipboard?.type === 'cut' && clipboard.path === node.path
  const isDragOver = node.kind === 'folder' && dragOverPath === node.path
  const isDragging = dragPath === node.path

  const nodeClass = [
    'tree-node',
    isActive ? 'is-active' : isCtxTarget ? 'is-ctx-target' : isFocused ? 'is-focused' : '',
    isDragOver ? 'is-dragover' : '',
  ].filter(Boolean).join(' ')

  return (
    <>
      <div
        className={nodeClass}
        tabIndex={0}
        draggable
        onDragStart={(e) => { e.stopPropagation(); handleDragStart(e, node.path) }}
        onDragOver={(e) => { e.stopPropagation(); handleDragOver(e, node) }}
        onDrop={(e) => { e.stopPropagation(); handleDrop(e, node) }}
        onDragEnd={handleDragEnd}
        onClick={(e) => {
          e.stopPropagation()
          setFocusedPath(node.path)
          if (node.kind === 'folder') toggleFolder(node.path)
          else setActiveFilename(node.path)
        }}
        onContextMenu={(e) => {
          e.preventDefault()
          e.stopPropagation()
          setFocusedPath(node.path)
          setCtxMenu({ x: e.clientX, y: e.clientY, node })
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 3,
          paddingLeft: indent,
          paddingRight: 8,
          height: 22,
          cursor: isDragging ? 'grabbing' : 'pointer',
          fontSize: 13,
          color: isCut ? 'var(--text-3)' : 'var(--text-1)',
          opacity: isDragging ? 0.4 : isCut ? 0.6 : 1,
          userSelect: 'none',
          outline: 'none',
          fontWeight: node.kind === 'folder' ? 600 : 400,
          borderLeft: isDragOver ? '2px solid #0e639c' : '2px solid transparent',
          transition: 'opacity 0.1s',
          boxSizing: 'border-box',
        }}
      >
        {/* Collapse arrow / file type icon — shared column */}
        <span style={{ width: 16, flexShrink: 0, fontSize: 8, color: 'var(--text-2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {node.kind === 'folder' ? (isCollapsed ? '▶' : '▼') : <FileIcon filename={node.name} />}
        </span>

        {renamingPath === node.path ? (
          <input
            ref={renameInputRef}
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitRename()
              if (e.key === 'Escape') setRenamingPath(null)
              e.stopPropagation()
            }}
            onClick={(e) => e.stopPropagation()}
            style={{ background: 'var(--bg-1)', border: '1px solid #0e639c', color: 'var(--text-bright)', fontSize: 13, padding: '1px 4px', flex: 1, minWidth: 0, outline: 'none', borderRadius: 3 }}
          />
        ) : (
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
            {node.name}
          </span>
        )}

        {node.kind === 'file' && !!proposals[node.path] && (
          <span title="Pending changes" style={{ width: 7, height: 7, borderRadius: '50%', background: '#f9c74f', flexShrink: 0 }} />
        )}
      </div>

      {node.kind === 'folder' && !isCollapsed && (
        <>
          {node.children?.map((child) => <TreeNodeView key={child.path} node={child} depth={depth + 1} state={state} handlers={handlers} />)}
          {creating?.parentPath === node.path && <CreateRow depth={depth + 1} state={state} handlers={handlers} />}
        </>
      )}
    </>
  )
}

// ─── ContextMenu ──────────────────────────────────────────────────────────────

interface ContextMenuProps {
  menu: CtxMenu
  clipboard: { type: string; kind: string; path: string } | null
  onClose: () => void
  onNewFile: () => void
  onNewFolder: () => void
  onRename: () => void
  onCopy: () => void
  onCut: () => void
  onDuplicate: () => void
  onPaste: () => void
  onDelete: () => void
}

function ContextMenu({ menu, clipboard, onClose, onNewFile, onNewFolder, onRename, onCopy, onCut, onDuplicate, onPaste, onDelete }: ContextMenuProps) {
  const hasPath = !!menu.node.path
  const isFolder = menu.node.kind === 'folder'
  const isFile = menu.node.kind === 'file'

  const items: Array<{ label: string; action: () => void; danger?: boolean } | null> = [
    { label: 'New File', action: onNewFile },
    { label: 'New Folder', action: onNewFolder },
    null,
    ...(hasPath ? [
      { label: 'Rename', action: onRename },
      { label: 'Cut', action: onCut },
      { label: 'Copy', action: onCopy },
      ...(isFile ? [{ label: 'Duplicate', action: onDuplicate as () => void }] : []),
    ] : []),
    ...(clipboard ? [{ label: 'Paste', action: onPaste }] : []),
    ...(hasPath ? [null, { label: isFolder ? 'Delete Folder' : 'Delete', action: onDelete as () => void, danger: true }] : []),
  ]

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      style={ctxMenuStyle(menu.x, menu.y)}
    >
      {items.map((item, i) =>
        item === null ? (
          <div key={i} style={{ height: 1, background: 'var(--border)', margin: '3px 0' }} />
        ) : (
          <button
            key={item.label}
            className="ctx-menu-item"
            onClick={() => { onClose(); item.action() }}
            style={{ color: item.danger ? '#f47067' : 'var(--text-1)' }}
          >
            {item.label}
          </button>
        ),
      )}
    </div>
  )
}

// ─── DeleteModal ──────────────────────────────────────────────────────────────

function DeleteModal({ node, onConfirm, onCancel }: { node: TreeNode; onConfirm: () => void; onCancel: () => void }) {
  const deleteBtnRef = useRef<HTMLButtonElement>(null)
  const cancelBtnRef = useRef<HTMLButtonElement>(null)

  useEffect(() => { deleteBtnRef.current?.focus() }, [])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') { e.stopPropagation(); onCancel() }
    if (e.key === 'Tab') {
      e.preventDefault()
      if (document.activeElement === deleteBtnRef.current) cancelBtnRef.current?.focus()
      else deleteBtnRef.current?.focus()
    }
  }

  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}
      onClick={onCancel}
    >
      <div
        style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 6, padding: '20px 24px', minWidth: 280, maxWidth: 360, boxShadow: '0 8px 32px rgba(0,0,0,0.3)', display: 'flex', flexDirection: 'column', gap: 16 }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
          <span style={{ color: '#e5c07b', fontSize: 18, flexShrink: 0 }}>⚠</span>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-1)', marginBottom: 4 }}>
              Delete {node.kind === 'folder' ? 'folder' : 'file'}
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-2)' }}>
              <strong style={{ color: 'var(--text-1)' }}>{node.name}</strong>
              {node.kind === 'folder' && ' and all its contents'}
              {' will be permanently deleted.'}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button ref={cancelBtnRef} onClick={onCancel} style={{ background: 'var(--bg-2)', border: '1px solid var(--border-2)', borderRadius: 3, color: 'var(--text-1)', cursor: 'pointer', fontSize: 13, padding: '5px 14px' }}>
            Cancel
          </button>
          <button ref={deleteBtnRef} onClick={onConfirm} style={{ background: '#c0392b', border: 'none', borderRadius: 3, color: '#fff', cursor: 'pointer', fontSize: 13, padding: '5px 14px' }}>
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}

