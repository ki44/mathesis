import { useState, useCallback } from 'react'
import { useCourseStore } from '../store/courseStore'
import { useContextMenuClose } from '../hooks/useContextMenuClose'
import { useThemeStore } from '../store/themeStore'

export function Sidebar() {
  const files = useCourseStore((s) => s.files)
  const proposals = useCourseStore((s) => s.proposals)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const setActiveFilename = useCourseStore((s) => s.setActiveFilename)
  const deleteFile = useCourseStore((s) => s.deleteFile)

  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; filename: string } | null>(null)
  const closeContextMenu = useCallback(() => setContextMenu(null), [])
  useContextMenuClose(closeContextMenu)
  const { theme, toggle } = useThemeStore()

  const handleDelete = async (filename: string) => {
    setContextMenu(null)
    if (!window.confirm(`Delete course "${filename}"?`)) return
    await deleteFile(filename)
  }

  return (
    <div
      style={{
        width: '100%',
        background: 'var(--bg-2)',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--border)',
          fontWeight: 600,
          fontSize: 12,
          color: 'var(--text-2)',
          textTransform: 'uppercase',
          letterSpacing: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        Courses
        <button
          onClick={toggle}
          title="Toggle light/dark mode"
          style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: 0, color: 'var(--text-2)' }}
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </div>

      {/* File list */}
      <div style={{ flex: 1, overflowY: 'auto', paddingTop: 4 }}>
        {files.length === 0 && (
          <p style={{ color: 'var(--text-3)', fontSize: 12, padding: '12px 14px' }}>
            No courses yet
          </p>
        )}
        {files.map((file) => {
          const hasProposal = !!proposals[file.filename]
          const isActive = file.filename === activeFilename
          return (
            <div
              key={file.filename}
              onClick={() => setActiveFilename(file.filename)}
              onContextMenu={(e) => {
                e.preventDefault()
                e.stopPropagation()
                setContextMenu({ x: e.clientX, y: e.clientY, filename: file.filename })
              }}
              className={`sidebar-file${isActive ? ' active' : ''}`}
            >
              <span
                style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                }}
              >
                {file.filename}
              </span>
              {hasProposal && (
                <span
                  title="Pending changes"
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: '#f9c74f',
                    flexShrink: 0,
                  }}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* Context menu */}
      {contextMenu && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'fixed',
            top: contextMenu.y,
            left: contextMenu.x,
            background: 'var(--bg-3)',
            border: '1px solid var(--border-2)',
            borderRadius: 4,
            padding: '4px 0',
            zIndex: 1000,
            minWidth: 160,
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          }}
        >
          <div
            onClick={() => handleDelete(contextMenu.filename)}
            className="sidebar-menu-delete"
          >
            Delete course
          </div>
        </div>
      )}
    </div>
  )
}
