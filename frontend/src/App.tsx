import { useCallback, useRef, useState } from 'react'
import { useCourse } from './hooks/useCourse'
import { ChatPanel } from './components/ChatPanel'
import { FileView } from './components/FileView'
import { Sidebar } from './components/Sidebar'

// ─── Resize handle ───────────────────────────────────────────────────────────

interface ResizeHandleProps {
  onMouseDown: (e: React.MouseEvent) => void
}

function ResizeHandle({ onMouseDown }: ResizeHandleProps) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      onMouseDown={onMouseDown}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: 4,
        flexShrink: 0,
        cursor: 'col-resize',
        background: hovered ? '#0e639c' : 'transparent',
        transition: 'background 0.15s',
        zIndex: 10,
      }}
    />
  )
}

// ─── Resize hook ─────────────────────────────────────────────────────────────

function useResizePanel(
  initialWidth: number,
  min: number,
  max: number,
  side: 'left' | 'right',
): [number, (e: React.MouseEvent) => void] {
  const [width, setWidth] = useState(initialWidth)
  const startX = useRef(0)
  const startWidth = useRef(initialWidth)
  const widthRef = useRef(initialWidth)
  widthRef.current = width

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      startX.current = e.clientX
      startWidth.current = widthRef.current
      document.body.style.userSelect = 'none'
      document.body.style.cursor = 'col-resize'

      const onMove = (ev: MouseEvent) => {
        const delta = side === 'left' ? ev.clientX - startX.current : startX.current - ev.clientX
        setWidth(Math.min(max, Math.max(min, startWidth.current + delta)))
      }
      const onUp = () => {
        document.body.style.userSelect = ''
        document.body.style.cursor = ''
        window.removeEventListener('mousemove', onMove)
        window.removeEventListener('mouseup', onUp)
      }
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', onUp)
    },
    [side, min, max],
  )

  return [width, onMouseDown]
}

// ─── App ─────────────────────────────────────────────────────────────────────

export function App() {
  useCourse()

  const [sidebarWidth, onSidebarDrag] = useResizePanel(220, 120, 400, 'left')
  const [chatWidth, onChatDrag] = useResizePanel(360, 240, 600, 'right')

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', width: '100%' }}>
      <div style={{ width: sidebarWidth, flexShrink: 0 }}>
        <Sidebar />
      </div>
      <ResizeHandle onMouseDown={onSidebarDrag} />
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <FileView />
      </div>
      <ResizeHandle onMouseDown={onChatDrag} />
      <div style={{ width: chatWidth, flexShrink: 0 }}>
        <ChatPanel />
      </div>
    </div>
  )
}

