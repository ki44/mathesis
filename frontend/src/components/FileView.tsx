import { useState, useRef, useEffect, useCallback } from 'react'
import Editor, { DiffEditor, type OnMount } from '@monaco-editor/react'
import { MarkdownRenderer, calloutTheme, CALLOUT_RE, calloutMod } from './MarkdownRenderer'
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

const WIDGET_BTN_BASE = 'border-radius:4px;padding:2px 10px;font-size:11px;font-weight:600;cursor:pointer;font-family:inherit;'

// ─── Pure utility ─────────────────────────────────────────────────────────────

type InnerSegment = { text: string; hunkIndex: number | null }
type FlatSegment = { kind: 'flat'; text: string; hunkIndex: number | null }
type CalloutSegment = { kind: 'callout'; title: string; calloutType: string; permanent: boolean; defaultOpen: boolean; inner: InnerSegment[] }
type ContentSegment = FlatSegment | CalloutSegment

/** Split content into annotated segments. Callout blocks (> [!TYPE]) become CalloutSegment with
 *  per-body-paragraph hunk assignment; all other content becomes FlatSegment. */
function buildAnnotatedContent(content: string, hunks: ILineChange[], side: 'original' | 'modified'): ContentSegment[] {
  const lines = content.split('\n')
  const lineHunk = new Map<number, number>()
  hunks.forEach((hunk, idx) => {
    const start = side === 'original' ? hunk.originalStartLineNumber : hunk.modifiedStartLineNumber
    const end   = side === 'original' ? hunk.originalEndLineNumber   : hunk.modifiedEndLineNumber
    if (end > 0) for (let l = start; l <= end; l++) lineHunk.set(l, idx)
  })

  const result: ContentSegment[] = []
  let i = 0
  while (i < lines.length) {
    if (lines[i].startsWith('>')) {
      let j = i
      while (j < lines.length && lines[j].startsWith('>')) j++
      const blockLines = lines.slice(i, j)
      const firstContent = blockLines[0].replace(/^>\s?/, '')
      const calloutM = firstContent.match(CALLOUT_RE)
      if (calloutM) {
        const [, type, mod, rawTitle] = calloutM
        const { permanent, defaultOpen } = calloutMod(mod)
        const inner: InnerSegment[] = []
        let k = 1
        while (k < blockLines.length) {
          const hunkIdx = lineHunk.get(i + k + 1) ?? null
          let m = k
          while (m < blockLines.length && (lineHunk.get(i + m + 1) ?? null) === hunkIdx) m++
          inner.push({ text: blockLines.slice(k, m).map(l => l.replace(/^>\s?/, '')).join('\n'), hunkIndex: hunkIdx })
          k = m
        }
        result.push({ kind: 'callout', title: rawTitle || type, calloutType: type.toLowerCase(), permanent, defaultOpen, inner })
      } else {
        let blockHunk: number | null = null
        for (let l = i; l < j; l++) { const h = lineHunk.get(l + 1); if (h !== undefined) { blockHunk = h; break } }
        result.push({ kind: 'flat', text: blockLines.join('\n'), hunkIndex: blockHunk })
      }
      i = j
    } else {
      const hunkIdx = lineHunk.get(i + 1) ?? null
      let j = i
      while (j < lines.length && !lines[j].startsWith('>') && (lineHunk.get(j + 1) ?? null) === hunkIdx) j++
      result.push({ kind: 'flat', text: lines.slice(i, j).join('\n'), hunkIndex: hunkIdx })
      i = j
    }
  }
  return result
}

function hunkDecisionStyle(type: 'accept' | 'reject', accepted: boolean | null): React.CSSProperties {
  if (type === 'accept') {
    return { ...BTN_BASE_STYLE, background: accepted === true ? '#16a34a' : 'rgba(22,163,74,0.1)', border: `1px solid ${accepted === true ? '#16a34a' : 'rgba(22,163,74,0.5)'}`, color: accepted === true ? '#fff' : '#4ade80', padding: '2px 10px', fontSize: 11 }
  }
  return { ...BTN_BASE_STYLE, background: accepted === false ? '#dc2626' : 'rgba(220,38,38,0.1)', border: `1px solid ${accepted === false ? '#dc2626' : 'rgba(220,38,38,0.5)'}`, color: accepted === false ? '#fff' : '#f87171', padding: '2px 10px', fontSize: 11 }
}

function InlineCalloutContainer({ title, calloutType, permanent, inner, side, open, onToggle, decisions, setDecisions }: {
  title: string
  calloutType?: string
  permanent: boolean
  inner: InnerSegment[]
  side: 'original' | 'modified'
  open: boolean
  onToggle: () => void
  decisions: HunkDecision[]
  setDecisions: React.Dispatch<React.SetStateAction<HunkDecision[]>>
}) {
  const theme = calloutTheme(calloutType)
  const isOpen = permanent || open
  return (
    <div style={{ border: `1px solid ${theme.border}`, borderRadius: 6, margin: '1em 0', overflow: 'hidden', background: theme.bg }}>
      <div onClick={permanent ? undefined : onToggle} style={{ padding: '8px 14px', background: 'var(--bg-2)', color: theme.title, fontWeight: 600, cursor: permanent ? 'default' : 'pointer', display: 'flex', alignItems: 'center', gap: 8, userSelect: 'none' }}>
        {!permanent && <span style={{ fontSize: 11, display: 'inline-block', transform: isOpen ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>▶</span>}
        {title}
      </div>
      {isOpen && (
        <div style={{ padding: '4px 16px 8px', borderTop: '1px solid var(--border)' }}>
          {inner.map((seg, idx) => {
            if (seg.hunkIndex === null) return <div key={idx}><MarkdownRenderer content={seg.text} compact /></div>
            if (side === 'original') {
              return (
                <div key={idx} style={{ background: 'rgba(220,38,38,0.10)', borderLeft: '3px solid rgba(200,60,60,0.55)', paddingLeft: 8, marginLeft: -8 }}>
                  <MarkdownRenderer content={seg.text} compact />
                </div>
              )
            }
            const hunkIdx = seg.hunkIndex
            const dec = decisions[hunkIdx]
            return (
              <div key={idx} style={{ position: 'relative', background: 'rgba(34,197,94,0.10)', borderLeft: '3px solid rgba(34,197,94,0.55)', paddingLeft: 8, marginLeft: -8, paddingRight: 140 }}>
                <div style={{ position: 'absolute', top: 4, right: 4, display: 'flex', gap: 4 }}>
                  <button onClick={() => setDecisions(prev => prev.map((d, k) => k === hunkIdx ? { ...d, accepted: true } : d))} style={hunkDecisionStyle('accept', dec?.accepted ?? null)}>Accept</button>
                  <button onClick={() => setDecisions(prev => prev.map((d, k) => k === hunkIdx ? { ...d, accepted: false } : d))} style={hunkDecisionStyle('reject', dec?.accepted ?? null)}>Reject</button>
                </div>
                <MarkdownRenderer content={seg.text} compact />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

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

function PlainEditor({ isPreview, setIsPreview }: { isPreview: boolean; setIsPreview: React.Dispatch<React.SetStateAction<boolean>> }) {
  const files = useCourseStore((s) => s.files)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const saveFile = useCourseStore((s) => s.saveFile)
  const fileRevisions = useCourseStore((s) => s.fileRevisions)

  const [isDirty, setIsDirty] = useState(false)
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

  // Ctrl+S while in preview mode (Monaco loses focus behind the overlay so its binding doesn't fire)
  useEffect(() => {
    if (!isPreview) return
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        void saveRef.current()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isPreview])

  const handleMount: OnMount = (editor, monacoInstance) => {
    editor.addCommand(
      monacoInstance.KeyMod.CtrlCmd | monacoInstance.KeyCode.KeyS,
      () => { void saveRef.current() },
    )
    editor.onKeyDown((e) => {
      const sel = editor.getSelection()
      const model = editor.getModel()
      if (!sel || !model) return

      // '>' typed over a multi-line selection → prefix each selected line with '> '
      if (e.browserEvent.key === '>' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (sel.startLineNumber < sel.endLineNumber) {
          e.preventDefault()
          e.stopPropagation()
          const ops = []
          for (let line = sel.startLineNumber; line <= sel.endLineNumber; line++)
            ops.push({ range: new monacoInstance.Range(line, 1, line, 1), text: '> ' })
          editor.executeEdits('blockquote-add', ops)
        }
        return
      }

      // Ctrl+< → remove '> '/'>' prefix from selected lines
      if (e.browserEvent.key === '<' && (e.ctrlKey || e.metaKey) && !e.altKey) {
        e.preventDefault()
        e.stopPropagation()
        const ops = []
        for (let line = sel.startLineNumber; line <= sel.endLineNumber; line++) {
          const m = model.getLineContent(line).match(/^(>\s?)/)
          if (m) ops.push({ range: new monacoInstance.Range(line, 1, line, m[1].length + 1), text: '' })
        }
        if (ops.length) editor.executeEdits('blockquote-remove', ops)
        return
      }

      // Enter → continue '>' prefix when current line starts with '> '
      if (e.browserEvent.key === 'Enter' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        if (sel.startLineNumber === sel.endLineNumber) {
          if (/^>\s/.test(model.getLineContent(sel.startLineNumber))) {
            e.preventDefault()
            e.stopPropagation()
            editor.trigger('keyboard', 'type', { text: '\n> ' })
          }
        }
      }
    })
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
        {/* Preview overlay — sits on top of Monaco so the editor stays mounted and undo history is preserved */}
        {isPreview && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 1, overflowY: 'auto', background: 'var(--bg-1)' }}>
            <MarkdownRenderer content={isDirty ? currentValueRef.current : activeFile.content} />
          </div>
        )}
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
      </div>
    </div>
  )
}

// ─── Diff editor (proposal present) ──────────────────────────────────────────

function DiffReview({ isPreview, setIsPreview }: { isPreview: boolean; setIsPreview: React.Dispatch<React.SetStateAction<boolean>> }) {
  const files = useCourseStore((s) => s.files)
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const proposals = useCourseStore((s) => s.proposals)
  const applyChanges = useCourseStore((s) => s.applyChanges)
  const rejectProposal = useCourseStore((s) => s.rejectProposal)

  const editorRef = useRef<IDiffEditor | null>(null)
  const initializedRef = useRef(false)
  const zoneIdsRef = useRef<string[]>([])
  const btnRefsRef = useRef<Array<{ accept: HTMLButtonElement; reject: HTMLButtonElement }>>([])
  const leftScrollRef = useRef<HTMLDivElement | null>(null)
  const rightScrollRef = useRef<HTMLDivElement | null>(null)
  const syncingScroll = useRef(false)
  const [hunks, setHunks] = useState<ILineChange[]>([])
  const [decisions, setDecisions] = useState<HunkDecision[]>([])
  const [openCallouts, setOpenCallouts] = useState<Record<number, boolean>>({})
  const [isApplying, setIsApplying] = useState(false)
  const { theme } = useThemeStore()

  const activeFile = files.find((f) => f.filename === activeFilename)
  const proposal = activeFilename ? proposals[activeFilename] : undefined

  // Reset hunk state when proposal changes
  useEffect(() => {
    initializedRef.current = false
    setHunks([])
    setDecisions([])
    setOpenCallouts({})
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
        acceptBtn.style.cssText = `${WIDGET_BTN_BASE}background:rgba(22,163,74,0.1);border:1px solid rgba(22,163,74,0.5);color:#4ade80;`
        acceptBtn.onclick = () =>
          setDecisions((prev) => prev.map((d, idx) => (idx === i ? { ...d, accepted: true } : d)))

        const rejectBtn = document.createElement('button')
        rejectBtn.textContent = 'Reject'
        rejectBtn.style.cssText = `${WIDGET_BTN_BASE}background:rgba(220,38,38,0.1);border:1px solid rgba(220,38,38,0.5);color:#f87171;`
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
      btns.accept.style.background = dec.accepted === true ? '#16a34a' : 'rgba(22,163,74,0.1)'
      btns.accept.style.borderColor = dec.accepted === true ? '#16a34a' : 'rgba(22,163,74,0.5)'
      btns.accept.style.color = dec.accepted === true ? '#fff' : '#4ade80'
      btns.reject.style.background = dec.accepted === false ? '#dc2626' : 'rgba(220,38,38,0.1)'
      btns.reject.style.borderColor = dec.accepted === false ? '#dc2626' : 'rgba(220,38,38,0.5)'
      btns.reject.style.color = dec.accepted === false ? '#fff' : '#f87171'
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

  async function handleAcceptAll() {
    if (hunks.length > 0) {
      await handleApply(hunks.map((_, i) => ({ hunkIndex: i, accepted: true as const })))
    } else {
      // Fallback: Monaco hasn't computed hunks yet (race), apply full proposal
      if (!activeFilename || !proposal) return
      setIsApplying(true)
      try { await applyChanges(activeFilename, proposal.proposed_content) }
      finally { setIsApplying(false) }
    }
  }

  function syncScroll(from: HTMLDivElement | null, to: HTMLDivElement | null) {
    if (syncingScroll.current || !from || !to) return
    syncingScroll.current = true
    to.scrollTop = from.scrollTop
    syncingScroll.current = false
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
          <button onClick={() => handleApply(decisions)} disabled={isApplying} style={{ ...BTN_BASE_STYLE, background: isApplying ? 'rgba(255,255,255,0.08)' : '#2563eb' }}>
            {isApplying ? '...' : 'Apply selection'}
          </button>
        )}
        <button onClick={handleAcceptAll} disabled={isApplying} style={{ ...BTN_BASE_STYLE, background: isApplying ? 'rgba(255,255,255,0.08)' : '#16a34a' }}>
          {isApplying ? '...' : 'Accept all'}
        </button>
        <button onClick={() => rejectProposal(activeFilename!)} style={{ ...BTN_BASE_STYLE, background: '#dc2626' }}>
          Reject all
        </button>
        <button
          onClick={() => setIsPreview((v) => !v)}
          style={{ background: 'none', border: '1px solid var(--border-2)', borderRadius: 4, color: 'var(--text-2)', padding: '2px 10px', fontSize: 12, cursor: 'pointer' }}
        >
          {isPreview ? 'Code' : 'Preview'}
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

      {/* Content area — Monaco always mounted; preview panels overlay it when active */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {/* Monaco DiffEditor */}
        <div style={{ position: 'absolute', inset: 0, pointerEvents: isPreview ? 'none' : 'auto' }}>
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
        {/* Preview panels overlay — z-index:101 sits above Monaco's view zone buttons (z-index:100) */}
        {isPreview && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 101, display: 'flex', overflow: 'hidden', background: 'var(--bg-1)' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div ref={leftScrollRef} onScroll={() => syncScroll(leftScrollRef.current, rightScrollRef.current)} style={{ flex: 1, overflowY: 'auto', background: 'var(--bg-1)' }}>
                <div style={{ padding: '24px 32px', maxWidth: 800, margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
                  {buildAnnotatedContent(activeFile.content, hunks, 'original').map((seg, i) => {
                    if (seg.kind === 'callout') {
                      const open = openCallouts[i] ?? (seg.inner.some(s => s.hunkIndex !== null) || seg.defaultOpen)
                      return <InlineCalloutContainer key={i} title={seg.title} calloutType={seg.calloutType} permanent={seg.permanent} inner={seg.inner} side="original" open={open} onToggle={() => setOpenCallouts(prev => ({ ...prev, [i]: !open }))} decisions={decisions} setDecisions={setDecisions} />
                    }
                    if (seg.hunkIndex === null) return <div key={i}><MarkdownRenderer content={seg.text} compact /></div>
                    return (
                      <div key={i} style={{ background: 'rgba(220,38,38,0.10)', borderLeft: '3px solid rgba(200,60,60,0.55)', paddingLeft: 8, marginLeft: -11 }}>
                        <MarkdownRenderer content={seg.text} compact />
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
            <div style={{ width: 1, background: 'var(--border)', flexShrink: 0 }} />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div ref={rightScrollRef} onScroll={() => syncScroll(rightScrollRef.current, leftScrollRef.current)} style={{ flex: 1, overflowY: 'auto', background: 'var(--bg-1)' }}>
                <div style={{ padding: '24px 32px', maxWidth: 800, margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
                  {buildAnnotatedContent(proposal.proposed_content, hunks, 'modified').map((seg, i) => {
                    if (seg.kind === 'callout') {
                      const open = openCallouts[i] ?? (seg.inner.some(s => s.hunkIndex !== null) || seg.defaultOpen)
                      return <InlineCalloutContainer key={i} title={seg.title} calloutType={seg.calloutType} permanent={seg.permanent} inner={seg.inner} side="modified" open={open} onToggle={() => setOpenCallouts(prev => ({ ...prev, [i]: !open }))} decisions={decisions} setDecisions={setDecisions} />
                    }
                    if (seg.hunkIndex === null) return <div key={i}><MarkdownRenderer content={seg.text} compact /></div>
                    const hunkIdx = seg.hunkIndex
                    const dec = decisions[hunkIdx]
                    const acceptBtn = <button key="a" onClick={() => setDecisions(prev => prev.map((d, idx) => idx === hunkIdx ? { ...d, accepted: true } : d))} style={hunkDecisionStyle('accept', dec?.accepted ?? null)}>Accept</button>
                    const rejectBtn = <button key="r" onClick={() => setDecisions(prev => prev.map((d, idx) => idx === hunkIdx ? { ...d, accepted: false } : d))} style={hunkDecisionStyle('reject', dec?.accepted ?? null)}>Reject</button>
                    return (
                      <div key={i} style={{ position: 'relative', background: 'rgba(34,197,94,0.10)', borderLeft: '3px solid rgba(34,197,94,0.55)', paddingLeft: 8, marginLeft: -11, paddingRight: 140 }}>
                        <div style={{ position: 'absolute', top: 4, right: 4, display: 'flex', alignItems: 'center', gap: 4 }}>{acceptBtn}{rejectBtn}</div>
                        <MarkdownRenderer content={seg.text} compact />
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Unified entry point ──────────────────────────────────────────────────────

export function FileView() {
  const activeFilename = useCourseStore((s) => s.activeFilename)
  const proposals = useCourseStore((s) => s.proposals)
  const files = useCourseStore((s) => s.files)
  const [isPreview, setIsPreview] = useState(false)
  const hasProposal = activeFilename ? !!proposals[activeFilename] : false
  const hasFile = files.some((f) => f.filename === activeFilename)

  useEffect(() => { setIsPreview(false) }, [activeFilename])
  useEffect(() => { if (hasProposal) setIsPreview(false) }, [hasProposal])

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

  return hasProposal ? <DiffReview isPreview={isPreview} setIsPreview={setIsPreview} /> : <PlainEditor isPreview={isPreview} setIsPreview={setIsPreview} />
}
