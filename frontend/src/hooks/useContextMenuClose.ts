import { useEffect } from 'react'

export function useContextMenuClose(close: () => void) {
  useEffect(() => {
    window.addEventListener('click', close)
    window.addEventListener('contextmenu', close)
    return () => {
      window.removeEventListener('click', close)
      window.removeEventListener('contextmenu', close)
    }
  }, [close])
}
