import type React from 'react'

export function ctxMenuStyle(x: number, y: number): React.CSSProperties {
  return {
    position: 'fixed',
    top: y,
    left: x,
    background: 'var(--bg-3)',
    border: '1px solid var(--border-2)',
    borderRadius: 4,
    zIndex: 1000,
    minWidth: 160,
    boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
    overflow: 'hidden',
    padding: '4px 0',
    fontSize: 13,
  }
}
