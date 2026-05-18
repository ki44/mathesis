import { useState, useRef, useEffect, useCallback } from 'react'
import Editor, { type OnMount } from '@monaco-editor/react'
import { DiffEditor, useMonaco } from '@monaco-editor/react'
import type * as Monaco from 'monaco-editor'
import { useCourseStore } from '../store/courseStore'
import type { HunkDecision } from '../types'

type ILineChange = Monaco.editor.ILineChange
type IDiffEditor = Monaco.editor.IDiffEditor

// ─── Shared helpers ───────────────────────────────────────────────────────────

function btnStyle(bg: string): React.CSSProperties {
  return {
    background: bg,
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
}

// ─── Plain editor (no proposal) ──────────────────────────────────────────────

function PlainEditor() {
  const files = useCourseStore((s) => s.files)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const saveFile = useCourseStore((s) => s.saveFile)
  const fileRevisions = useCourseStore((s) => s.fileRevisions)

  const [isDirty, setIsDirty] = useState(false)
  const currentValueRef = useRef<string>('')
  const savedContentRef = useRef<string>('')

  const activeFile = files.find((f) => f.filename === activeFilename)
  const revision = fileRevisions[activeFilename ?? ''] ?? 0

  useEffect(() => {
    setIsDirty(false)
    currentValueRef.current = activeFile?.content ?? ''
    savedContentRef.current = activeFile?.content ?? ''
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
          borderBottom: '1px solid #333',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          background: '#252526',
        }}
      >
        <span style={{ fontSize: 13, color: '#cccccc', flex: 1 }}>
          {activeFile.filename}
          {isDirty && (
            <span title="Modifications non sauvegardées (Ctrl+S)" style={{ color: '#f9c74f', marginLeft: 6 }}>
              ●
            </span>
          )}
        </span>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{ position: 'absolute', inset: 0 }}>
          <Editor
            key={`${activeFilename}-${revision}`}
            width="100%"
            height="100%"
            language="markdown"
            theme="vs-dark"
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
      </div>
    </div>
  )
}

// ─── Diff editor (proposal present) ──────────────────────────────────────────

function DiffReview() {
  const monaco = useMonaco()
  const files = useCourseStore((s) => s.files)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const proposals = useCourseStore((s) => s.proposals)
  const applyChanges = useCourseStore((s) => s.applyChanges)
  const rejectProposal = useCourseStore((s) => s.rejectProposal)

  const editorRef = useRef<IDiffEditor | null>(null)
  const initializedRef = useRef(false)
  const widgetsRef = useRef<Monaco.editor.IContentWidget[]>([])
  const [hunks, setHunks] = useState<ILineChange[]>([])
  const [decisions, setDecisions] = useState<HunkDecision[]>([])
  const [isApplying, setIsApplying] = useState(false)

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
    setTimeout(initHunks, 300)
  }, [])

  // Per-hunk accept/reject widgets
  useEffect(() => {
    if (!monaco || !editorRef.current || hunks.length === 0) return
    const modEditor = editorRef.current.getModifiedEditor()

    widgetsRef.current.forEach((w) => modEditor.removeContentWidget(w))
    widgetsRef.current = []

    hunks.forEach((hunk, i) => {
      const dec = decisions[i]
      const lineNumber =
        hunk.modifiedEndLineNumber > 0
          ? hunk.modifiedEndLineNumber
          : hunk.modifiedStartLineNumber > 0
            ? hunk.modifiedStartLineNumber
            : 1

      const container = document.createElement('div')
      container.style.cssText =
        'display:flex;gap:6px;padding:2px 8px 4px;justify-content:flex-end;pointer-events:all;'

      const makeBtn = (label: string, active: boolean, activeBg: string, onClick: () => void) => {
        const btn = document.createElement('button')
        btn.textContent = label
        btn.style.cssText = [
          `background:${active ? activeBg : '#2a2a2a'}`,
          `border:1px solid ${active ? activeBg : '#555'}`,
          'border-radius:4px',
          'color:#fff',
          'padding:2px 10px',
          'font-size:11px',
          'font-weight:600',
          'cursor:pointer',
          'font-family:inherit',
        ].join(';')
        btn.onclick = onClick
        return btn
      }

      container.appendChild(
        makeBtn('Accepter', dec?.accepted === true, '#1e7340', () =>
          setDecisions((prev) => prev.map((d, idx) => (idx === i ? { ...d, accepted: true } : d))),
        ),
      )
      container.appendChild(
        makeBtn('Refuser', dec?.accepted === false, '#6b3030', () =>
          setDecisions((prev) => prev.map((d, idx) => (idx === i ? { ...d, accepted: false } : d))),
        ),
      )

      const widget: Monaco.editor.IContentWidget = {
        getId: () => `hunk-widget-${i}`,
        getDomNode: () => container,
        getPosition: () => ({
          position: { lineNumber, column: 1 },
          preference: [monaco.editor.ContentWidgetPositionPreference.BELOW],
        }),
      }

      modEditor.addContentWidget(widget)
      widgetsRef.current.push(widget)
    })

    return () => {
      if (editorRef.current) {
        const mod = editorRef.current.getModifiedEditor()
        widgetsRef.current.forEach((w) => mod.removeContentWidget(w))
        widgetsRef.current = []
      }
    }
  }, [monaco, hunks, decisions])

  function computeMergedWith(decs: HunkDecision[]): string {
    if (!activeFile || !proposal) return ''
    const originalLines = activeFile.content.split('\n')
    const modifiedLines = proposal.proposed_content.split('\n')

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

  async function handleApplySelected() {
    if (!activeFilename) return
    setIsApplying(true)
    try {
      await applyChanges(activeFilename, computeMergedWith(decisions))
    } finally {
      setIsApplying(false)
    }
  }

  async function handleAcceptAll() {
    if (!activeFilename) return
    setIsApplying(true)
    try {
      const allAccepted = hunks.map((_, i) => ({ hunkIndex: i, accepted: true as const }))
      await applyChanges(activeFilename, computeMergedWith(allAccepted))
    } finally {
      setIsApplying(false)
    }
  }

  async function handleRejectAll() {
    if (!activeFilename) return
    await rejectProposal(activeFilename)
  }

  if (!activeFile || !proposal) return null

  const anyDecided = decisions.some((d) => d.accepted !== null)

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Toolbar */}
      <div
        style={{
          padding: '8px 16px',
          borderBottom: '1px solid #333',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          background: '#252526',
        }}
      >
        <span style={{ fontSize: 12, color: '#888', flex: 1 }}>{activeFile.filename}</span>
        {anyDecided && (
          <button onClick={handleApplySelected} disabled={isApplying} style={btnStyle(isApplying ? '#333' : '#0e639c')}>
            {isApplying ? '...' : 'Appliquer la sélection'}
          </button>
        )}
        <button onClick={handleAcceptAll} disabled={isApplying} style={btnStyle(isApplying ? '#333' : '#1e7340')}>
          {isApplying ? '...' : 'Tout accepter'}
        </button>
        <button onClick={handleRejectAll} style={btnStyle('#6b3030')}>
          Tout refuser
        </button>
      </div>

      {/* Description */}
      {proposal.description && (
        <div
          style={{
            padding: '6px 16px',
            background: '#1e2a3a',
            borderBottom: '1px solid #333',
            fontSize: 12,
            color: '#9cdcfe',
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
            theme="vs-dark"
            original={activeFile.content}
            modified={proposal.proposed_content}
            onMount={handleMount}
            options={{
              readOnly: false,
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
          color: '#555',
          fontSize: 14,
        }}
      >
        Sélectionnez un fichier dans la sidebar
      </div>
    )
  }

  return hasProposal ? <DiffReview /> : <PlainEditor />
}
