import { useState, useEffect } from 'react'
import { useCourseStore } from '../store/courseStore'

export function Sidebar() {
    const files = useCourseStore((s) => s.files)
    const proposals = useCourseStore((s) => s.proposals)
    const activeFilename = useCourseStore((s) => s.activeFilename)
    const setActiveFilename = useCourseStore((s) => s.setActiveFilename)
    const deleteFile = useCourseStore((s) => s.deleteFile)

    const [contextMenu, setContextMenu] = useState<{ x: number; y: number; filename: string } | null>(null)

    useEffect(() => {
        const close = () => setContextMenu(null)
        window.addEventListener('click', close)
        window.addEventListener('contextmenu', close)
        return () => {
            window.removeEventListener('click', close)
            window.removeEventListener('contextmenu', close)
        }
    }, [])

    const handleDelete = async (filename: string) => {
        setContextMenu(null)
        if (!window.confirm(`Supprimer le cours "${filename}" ?`)) return
        await deleteFile(filename)
    }

    return (
        <>
            <div
                style={{
                    width: '100%',
                    background: '#252526',
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
                        borderBottom: '1px solid #333',
                        fontWeight: 600,
                        fontSize: 12,
                        color: '#888',
                        textTransform: 'uppercase',
                        letterSpacing: 1,
                    }}
                >
                    Cours
                </div>

                {/* File list */}
                <div style={{ flex: 1, overflowY: 'auto', paddingTop: 4 }}>
                    {files.length === 0 && (
                        <p style={{ color: '#555', fontSize: 12, padding: '12px 14px' }}>
                            Aucun cours pour l'instant
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
                                style={{
                                    padding: '7px 14px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 6,
                                    background: isActive ? '#094771' : 'transparent',
                                    color: isActive ? '#ffffff' : '#cccccc',
                                    fontSize: 13,
                                    userSelect: 'none',
                                    transition: 'background 0.1s',
                                }}
                                onMouseEnter={(e) => {
                                    if (!isActive) (e.currentTarget as HTMLDivElement).style.background = '#2a2d2e'
                                }}
                                onMouseLeave={(e) => {
                                    if (!isActive) (e.currentTarget as HTMLDivElement).style.background = 'transparent'
                                }}
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
                                        title="Modifications en attente"
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
            </div>

            {/* Context menu */}
            {contextMenu && (
                <div
                    onClick={(e) => e.stopPropagation()}
                    style={{
                        position: 'fixed',
                        top: contextMenu.y,
                        left: contextMenu.x,
                        background: '#2d2d2d',
                        border: '1px solid #444',
                        borderRadius: 4,
                        padding: '4px 0',
                        zIndex: 1000,
                        minWidth: 160,
                        boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                    }}
                >
                    <div
                        onClick={() => handleDelete(contextMenu!.filename)}
                        style={{ padding: '6px 16px', cursor: 'pointer', fontSize: 13, color: '#f48c8c' }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = '#3a1e1e' }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}
                    >
                        Supprimer le cours
                    </div>
                </div>
            )}
        </>
    )
}
