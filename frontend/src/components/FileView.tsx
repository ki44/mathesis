import { useState, useRef, useEffect, useCallback } from 'react'
import Editor, { DiffEditor, type OnMount } from '@monaco-editor/react'
import { MarkdownRenderer } from './MarkdownRenderer'
import { useThemeStore } from '../store/themeStore'
import type * as Monaco from 'monaco-editor'
import { useCourseStore } from '../store/courseStore'
import type { HunkDecision } from '../types'

type ILineChange = Monaco.editor.ILineChange
type IDiffEditor = Monaco.editor.IDiffEditor

// ─── Shared helpers ───────────────────────────────────────────────────────────

const BTN_BASE_STYLE: React.CSSProperties = {
  border: 'none',
  borderRadius: 5,
  color: '#fff',
  padding: '4px 10px',
  cursor: 'pointer',
  fontSize: 12,
  fontWeight: 600,
  whiteSpace: 'nowrap',
  transition: 'opacity 0.1s',
}

const WIDGET_BTN_BASE = 'border-radius:4px;color:#fff;padding:2px 10px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;'

// ─── Pure utility ─────────────────────────────────────────────────────────────

function computeMergedContent(
  decs: HunkDecision[],
  hunks: ILineChange[],
  originalContent: string,
  modifiedContent: string,
): string {
  const originalLines = originalContent.split('\n')
  const modifiedLines = modifiedContent.split('\n')

  const acceptedHunks = decs
    .filter((d) => d.accepted === true)
    .map((d) => hunks[d.hunkIndex])
    .sort((a, b) => b.originalStartLineNumber - a.originalStartLineNumber)

  const result = [...originalLines]
  for (const hunk of acceptedHunks) {
    const origStart = hunk.originalStartLineNumber - 1
    const origEnd = hunk.originalEndLineNumber
    const modStart = hunk.modifiedStartLineNumber - 1
    const modEnd = hunk.modifiedEndLineNumber
    const replacement = modEnd === 0 ? [] : modifiedLines.slice(modStart, modEnd)
    if (origEnd === 0) {
      result.splice(origStart + 1, 0, ...replacement)
    } else {
      result.splice(origStart, origEnd - origStart, ...replacement)
    }
  }
  return result.join('\n')
}

// ─── Plain editor (no proposal) ──────────────────────────────────────────────

function PlainEditor() {
  const files = useCourseStore((s) => s.files)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const saveFile = useCourseStore((s) => s.saveFile)
  const fileRevisions = useCourseStore((s) => s.fileRevisions)

  const [isDirty, setIsDirty] = useState(false)
  const [isPreview, setIsPreview] = useState(false)
  const currentValueRef = useRef<string>('')
  const savedContentRef = useRef<string>('')
  const { theme } = useThemeStore()

  const activeFile = files.find((f) => f.filename === activeFilename)
  const revision = fileRevisions[activeFilename ?? ''] ?? 0

  useEffect(() => {
    setIsDirty(false)
    const content = activeFile?.content ?? ''
    currentValueRef.current = content
    savedContentRef.current = content
  }, [activeFilename, revision]) // eslint-disable-line react-hooks/exhaustive-deps

  const saveRef = useRef<() => Promise<void>>(async () => {})
  saveRef.current = async () => {
    if (!activeFilename || !isDirty) return
    await saveFile(activeFilename, currentValueRef.current)
    savedContentRef.current = currentValueRef.current
    setIsDirty(false)
  }

  const handleMount: OnMount = (editor, monacoInstance) => {
    editor.addCommand(
      monacoInstance.KeyMod.CtrlCmd | monacoInstance.KeyCode.KeyS,
      () => { void saveRef.current() },
    )
  }

  if (!activeFile) return null

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div
        style={{
          padding: '8px 16px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          background: 'var(--bg-2)',
        }}
      >
        <span style={{ fontSize: 13, color: 'var(--text-1)', flex: 1 }}>
          {activeFile.filename}
          {isDirty && (
            <span title="Unsaved changes (Ctrl+S)" style={{ color: '#f9c74f', marginLeft: 6 }}>
              ●
            </span>
          )}
        </span>
        <button
          onClick={() => setIsPreview((v) => !v)}
          style={{ background: 'none', border: '1px solid var(--border-2)', borderRadius: 4, color: 'var(--text-2)', padding: '2px 10px', fontSize: 12, cursor: 'pointer' }}
        >
          {isPreview ? 'Edit' : 'Preview'}
        </button>
      </div>
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {isPreview ? (
          <div style={{ position: 'absolute', inset: 0, overflowY: 'auto', background: 'var(--bg-1)' }}>
            <MarkdownRenderer content={currentValueRef.current} />
          </div>
        ) : (
        <div style={{ position: 'absolute', inset: 0 }}>
          <Editor
            key={`${activeFilename}-${revision}`}
            width="100%"
            height="100%"
            language="markdown"
            theme={theme === 'dark' ? 'vs-dark' : 'vs'}
            defaultValue={activeFile.content}
            onMount={handleMount}
            onChange={(value) => {
              if (value !== undefined) {
                currentValueRef.current = value
                setIsDirty(value !== savedContentRef.current)
              }
            }}
            options={{
              minimap: { enabled: false },
              wordWrap: 'on',
              fontSize: 14,
              scrollBeyondLastLine: false,
              lineNumbers: 'off',
            }}
          />
        </div>
        )}
      </div>
    </div>
  )
}

// ─── Diff editor (proposal present) ──────────────────────────────────────────

function DiffReview() {
  const files = useCourseStore((s) => s.files)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const proposals = useCourseStore((s) => s.proposals)
  const applyChanges = useCourseStore((s) => s.applyChanges)
  const rejectProposal = useCourseStore((s) => s.rejectProposal)

  const editorRef = useRef<IDiffEditor | null>(null)
  const initializedRef = useRef(false)
  const zoneIdsRef = useRef<string[]>([])
  const btnRefsRef = useRef<Array<{ accept: HTMLButtonElement; reject: HTMLButtonElement }>>([])
  const [hunks, setHunks] = useState<ILineChange[]>([])
  const [decisions, setDecisions] = useState<HunkDecision[]>([])  
  const [isApplying, setIsApplying] = useState(false)
  const { theme } = useThemeStore()

  const activeFile = files.find((f) => f.filename === activeFilename)
  const proposal = activeFilename ? proposals[activeFilename] : undefined

  // Reset hunk state when proposal changes
  useEffect(() => {
    initializedRef.current = false
    setHunks([])
    setDecisions([])
  }, [activeFilename, proposal?.proposed_content])

  const handleMount = useCallback((editor: IDiffEditor) => {
    editorRef.current = editor

    const initHunks = () => {
      if (initializedRef.current) return
      const changes = editor.getLineChanges()
      if (changes && changes.length > 0) {
        initializedRef.current = true
        setHunks(changes)
        setDecisions(changes.map((_, i) => ({ hunkIndex: i, accepted: null })))
      }
    }

    editor.onDidUpdateDiff(initHunks)
    // Fallback: onDidUpdateDiff can fire before Monaco populates getLineChanges()
    setTimeout(initHunks, 300)
  }, [])

  // Create per-hunk accept/reject widgets when hunks are initialised
  useEffect(() => {
    if (!editorRef.current || hunks.length === 0) return
    const modEditor = editorRef.current.getModifiedEditor()

    btnRefsRef.current = []
    modEditor.changeViewZones((accessor) => {
      zoneIdsRef.current.forEach((id) => accessor.removeZone(id))
      zoneIdsRef.current = []

      hunks.forEach((hunk, i) => {
        const lineNumber =
          hunk.modifiedEndLineNumber > 0
            ? hunk.modifiedEndLineNumber
            : hunk.modifiedStartLineNumber > 0
              ? hunk.modifiedStartLineNumber
              : 1

        const container = document.createElement('div')
        container.style.cssText =
          'position:relative;z-index:100;display:flex;flex-direction:row;gap:6px;padding:2px 8px;justify-content:flex-end;pointer-events:all;'

        const acceptBtn = document.createElement('button')
        acceptBtn.textContent = 'Accept'
        acceptBtn.style.cssText = `background:#2a2a2a;border:1px solid #555;${WIDGET_BTN_BASE}`
        acceptBtn.onclick = () =>
          setDecisions((prev) => prev.map((d, idx) => (idx === i ? { ...d, accepted: true } : d)))

        const rejectBtn = document.createElement('button')
        rejectBtn.textContent = 'Reject'
        rejectBtn.style.cssText = `background:#2a2a2a;border:1px solid #555;${WIDGET_BTN_BASE}`
        rejectBtn.onclick = () =>
          setDecisions((prev) => prev.map((d, idx) => (idx === i ? { ...d, accepted: false } : d)))

        container.appendChild(acceptBtn)
        container.appendChild(rejectBtn)
        btnRefsRef.current.push({ accept: acceptBtn, reject: rejectBtn })

        zoneIdsRef.current.push(accessor.addZone({ afterLineNumber: lineNumber, heightInPx: 26, domNode: container }))
      })
    })

    return () => {
      if (editorRef.current) {
        const mod = editorRef.current.getModifiedEditor()
        mod.changeViewZones((accessor) => {
          zoneIdsRef.current.forEach((id) => accessor.removeZone(id))
        })
        zoneIdsRef.current = []
        btnRefsRef.current = []
      }
    }
  }, [hunks])

  // Sync button visual states without rebuilding widgets
  useEffect(() => {
    decisions.forEach((dec, i) => {
      const btns = btnRefsRef.current[i]
      if (!btns) return
      btns.accept.style.background = dec.accepted === true ? '#1e7340' : '#2a2a2a'
      btns.accept.style.borderColor = dec.accepted === true ? '#1e7340' : '#555'
      btns.reject.style.background = dec.accepted === false ? '#6b3030' : '#2a2a2a'
      btns.reject.style.borderColor = dec.accepted === false ? '#6b3030' : '#555'
    })
  }, [decisions])

  async function handleApply(decs: HunkDecision[]) {
    if (!activeFilename || !activeFile || !proposal) return
    setIsApplying(true)
    try {
      await applyChanges(activeFilename, computeMergedContent(decs, hunks, activeFile.content, proposal.proposed_content))
    } finally {
      setIsApplying(false)
    }
  }

  if (!activeFile || !proposal) return null

  const allDecided = decisions.length > 0 && decisions.every((d) => d.accepted !== null)

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Toolbar */}
      <div
        style={{
          padding: '8px 16px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          background: 'var(--bg-2)',
        }}
      >
        <span style={{ fontSize: 12, color: 'var(--text-2)', flex: 1 }}>{activeFile.filename}</span>
        {allDecided && (
          <button onClick={() => handleApply(decisions)} disabled={isApplying} style={{ ...BTN_BASE_STYLE, background: isApplying ? '#333' : '#0e639c' }}>
            {isApplying ? '...' : 'Apply selection'}
          </button>
        )}
        <button onClick={() => handleApply(hunks.map((_, i) => ({ hunkIndex: i, accepted: true as const })))} disabled={isApplying} style={{ ...BTN_BASE_STYLE, background: isApplying ? '#333' : '#1e7340' }}>
          {isApplying ? '...' : 'Accept all'}
        </button>
        <button onClick={() => rejectProposal(activeFilename!)} style={{ ...BTN_BASE_STYLE, background: '#6b3030' }}>
          Reject all
        </button>
      </div>

      {/* Description */}
      {proposal.description && (
        <div
          style={{
            padding: '6px 16px',
            background: 'var(--bg-info)',
            borderBottom: '1px solid var(--border)',
            fontSize: 12,
            color: 'var(--text-info)',
          }}
        >
          {proposal.description}
        </div>
      )}

      {/* Monaco DiffEditor */}
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{ position: 'absolute', inset: 0 }}>
          <DiffEditor
            width="100%"
            height="100%"
            language="markdown"
            theme={theme === 'dark' ? 'vs-dark' : 'vs'}
            original={activeFile.content}
            modified={proposal.proposed_content}
            onMount={handleMount}
            options={{
              renderSideBySide: true,
              minimap: { enabled: false },
              wordWrap: 'on',
              fontSize: 14,
              scrollBeyondLastLine: false,
              diffWordWrap: 'on',
            }}
          />
        </div>
      </div>
    </div>
  )
}

// ─── Unified entry point ──────────────────────────────────────────────────────

export function FileView() {
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const proposals = useCourseStore((s) => s.proposals)
  const files = useCourseStore((s) => s.files)

  const hasProposal = activeFilename ? !!proposals[activeFilename] : false
  const hasFile = files.some((f) => f.filename === activeFilename)

  if (!activeFilename || !hasFile) {
    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-3)',
          fontSize: 14,
        }}
      >
        Select a file from the sidebar
      </div>
    )
  }

  return hasProposal ? <DiffReview /> : <PlainEditor />
}
