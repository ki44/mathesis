import { useCallback, useEffect, useRef, useState } from 'react'
import { useCourseStore } from '../store/courseStore'
import { useContextMenuClose } from '../hooks/useContextMenuClose'
import { ctxMenuStyle } from './ctxMenuStyle'

type CtxMenu = { x: number; y: number; filename: string }

function basename(filename: string) {
  return filename.includes('/') ? filename.split('/').pop()! : filename
}

export function TabBar() {
  const openFiles = useCourseStore((s) => s.openFiles)
  const pinnedFiles = useCourseStore((s) => s.pinnedFiles)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const proposals = useCourseStore((s) => s.proposals)
  const setActiveFilename = useCourseStore((s) => s.setActiveFilename)
  const closeFile = useCourseStore((s) => s.closeFile)
  const closeMultiple = useCourseStore((s) => s.closeMultiple)
  const pinFile = useCourseStore((s) => s.pinFile)
  const unpinFile = useCourseStore((s) => s.unpinFile)
  const renameFile = useCourseStore((s) => s.renameFile)
  const copyFile = useCourseStore((s) => s.copyFile)

  const [ctxMenu, setCtxMenu] = useState<CtxMenu | null>(null)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const renameInputRef = useRef<HTMLInputElement>(null)

  const closeCtxMenu = useCallback(() => setCtxMenu(null), [])
  useContextMenuClose(closeCtxMenu)

  // Ctrl+W closes active tab
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'w' && (e.ctrlKey || e.metaKey) && activeFilename) {
        e.preventDefault()
        closeFile(activeFilename)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeFilename, closeFile])

  useEffect(() => {
    if (renaming) renameInputRef.current?.focus()
  }, [renaming])

  // Sort: pinned first, then in open order
  const tabs = [
    ...openFiles.filter((f) => pinnedFiles.includes(f)),
    ...openFiles.filter((f) => !pinnedFiles.includes(f)),
  ]

  const startRename = (filename: string) => {
    setRenaming(filename)
    setRenameValue(basename(filename))
  }

  const commitRename = async () => {
    if (!renaming || !renameValue.trim()) { setRenaming(null); return }
    const dir = renaming.includes('/') ? renaming.slice(0, renaming.lastIndexOf('/') + 1) : ''
    const newFilename = dir + renameValue.trim()
    if (newFilename !== renaming) {
      await renameFile(renaming, newFilename).catch((e) => console.error(e))
    }
    setRenaming(null)
  }

  const closeOthers = (filename: string) => {
    closeMultiple(openFiles.filter((f) => f !== filename))
  }

  const closeToRight = (filename: string) => {
    const idx = openFiles.indexOf(filename)
    closeMultiple(openFiles.slice(idx + 1))
  }

  const copyPath = (filename: string) => {
    navigator.clipboard.writeText(filename)
  }

  const duplicate = async (filename: string) => {
    await copyFile(filename).catch((e) => console.error(e))
  }

  if (tabs.length === 0) return null

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'row',
        background: 'var(--bg-2)',
        borderBottom: '1px solid var(--border)',
        overflowX: 'auto',
        flexShrink: 0,
        scrollbarWidth: 'none',
      }}
    >
      {tabs.map((filename) => {
        const isActive = filename === activeFilename
        const isPinned = pinnedFiles.includes(filename)
        const hasProposal = !!proposals[filename]

        return (
          <div
            key={filename}
            onClick={() => setActiveFilename(filename)}
            onContextMenu={(e) => {
              e.preventDefault()
              e.stopPropagation()
              setCtxMenu({ x: e.clientX, y: e.clientY, filename })
            }}
            title={filename}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              padding: '0 10px',
              height: 35,
              flexShrink: 0,
              cursor: 'pointer',
              fontSize: 12,
              color: isActive ? 'var(--text-bright)' : 'var(--text-2)',
              background: isActive ? 'var(--bg-1)' : 'transparent',
              borderRight: '1px solid var(--border)',
              borderTop: isActive ? '1px solid #0e639c' : '1px solid transparent',
              userSelect: 'none',
              position: 'relative',
            }}
          >
            {isPinned && <span style={{ fontSize: 9, opacity: 0.7 }}>📌</span>}

            {renaming === filename ? (
              <input
                ref={renameInputRef}
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename()
                  if (e.key === 'Escape') setRenaming(null)
                  e.stopPropagation()
                }}
                onClick={(e) => e.stopPropagation()}
                style={{
                  background: 'var(--bg-3)',
                  border: '1px solid #0e639c',
                  color: 'var(--text-bright)',
                  fontSize: 12,
                  padding: '1px 4px',
                  width: Math.max(80, renameValue.length * 7),
                  outline: 'none',
                }}
              />
            ) : (
              <span style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {basename(filename)}
              </span>
            )}

            {hasProposal && (
              <span
                title="Pending changes"
                style={{ width: 6, height: 6, borderRadius: '50%', background: '#f9c74f', flexShrink: 0 }}
              />
            )}

            <button
              onClick={(e) => { e.stopPropagation(); closeFile(filename) }}
              title="Close"
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-2)',
                fontSize: 14,
                lineHeight: 1,
                padding: '0 2px',
                marginLeft: 2,
                borderRadius: 2,
                flexShrink: 0,
              }}
            >
              ×
            </button>
          </div>
        )
      })}

      {/* Context menu */}
      {ctxMenu && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={ctxMenuStyle(ctxMenu.x, ctxMenu.y)}
        >
          {[
            { label: pinnedFiles.includes(ctxMenu.filename) ? 'Unpin' : 'Pin', action: () => { pinnedFiles.includes(ctxMenu.filename) ? unpinFile(ctxMenu.filename) : pinFile(ctxMenu.filename) } },
            { label: 'Rename', action: () => startRename(ctxMenu.filename) },
            { label: 'Duplicate', action: () => duplicate(ctxMenu.filename) },
            { label: 'Copy Path', action: () => copyPath(ctxMenu.filename) },
            null,
            { label: 'Close', action: () => closeFile(ctxMenu.filename) },
            { label: 'Close Others', action: () => closeOthers(ctxMenu.filename) },
            { label: 'Close to the Right', action: () => closeToRight(ctxMenu.filename) },
          ].map((item, i) =>
            item === null ? (
              <div key={i} style={{ height: 1, background: 'var(--border)', margin: '3px 0' }} />
            ) : (
              <button
                key={item.label}
                onClick={() => { setCtxMenu(null); item.action() }}
                className="ctx-menu-item"
              >
                {item.label}
              </button>
            ),
          )}
        </div>
      )}
    </div>
  )
}
