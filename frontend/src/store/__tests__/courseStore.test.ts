import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { CourseFile, FolderEntry } from '../../types'
import { useCourseStore } from '../courseStore'

const INITIAL_STATE = {
  files: [],
  folders: [],
  activeFilename: null,
  openFiles: [],
  pinnedFiles: [],
  clipboard: null,
  proposals: {},
  fileRevisions: {},
  undoStack: [],
}

const file = (filename: string, content = ''): CourseFile => ({ filename, content, updated_at: '2024-01-01' })
const folder = (path: string): FolderEntry => ({ path, created_at: '2024-01-01' })

beforeEach(() => {
  useCourseStore.setState(INITIAL_STATE)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

// ---------------------------------------------------------------------------
// createFile
// ---------------------------------------------------------------------------

describe('createFile', () => {
  it('pushes an undo entry; undoLast calls DELETE on the file', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify(file('notes.md')), { status: 200 })) // POST /api/courses
        .mockResolvedValueOnce(new Response('', { status: 200 })), // DELETE /api/courses/notes.md
    )

    await useCourseStore.getState().createFile('notes.md')
    expect(useCourseStore.getState().undoStack).toHaveLength(1)
    expect(useCourseStore.getState().files).toHaveLength(1)

    await useCourseStore.getState().undoLast()

    const fetchMock = vi.mocked(global.fetch)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe('/api/courses/notes.md')
    expect((fetchMock.mock.calls[1][1] as RequestInit).method).toBe('DELETE')
    expect(useCourseStore.getState().undoStack).toHaveLength(0)
    expect(useCourseStore.getState().files).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// deleteFile
// ---------------------------------------------------------------------------

describe('deleteFile', () => {
  it('throws and rolls back state when the API returns an error', async () => {
    useCourseStore.setState({ files: [file('notes.md', 'hello')], openFiles: ['notes.md'] })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('Server error', { status: 500 })))

    await expect(useCourseStore.getState().deleteFile('notes.md')).rejects.toThrow()

    expect(useCourseStore.getState().files).toHaveLength(1)
    expect(useCourseStore.getState().openFiles).toContain('notes.md')
    expect(useCourseStore.getState().undoStack).toHaveLength(0)
  })

  it('pushes an undo entry; undoLast re-creates the file via POST', async () => {
    useCourseStore.setState({ files: [file('notes.md', 'hello')] })
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response('', { status: 200 })) // DELETE
        .mockResolvedValueOnce(new Response(JSON.stringify(file('notes.md', 'hello')), { status: 200 })), // POST undo
    )

    await useCourseStore.getState().deleteFile('notes.md')
    expect(useCourseStore.getState().undoStack).toHaveLength(1)

    await useCourseStore.getState().undoLast()

    const fetchMock = vi.mocked(global.fetch)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    const [undoUrl, undoInit] = fetchMock.mock.calls[1] as [string, RequestInit]
    expect(undoUrl).toBe('/api/courses')
    expect((undoInit.method)).toBe('POST')
    expect(JSON.parse(undoInit.body as string)).toMatchObject({ filename: 'notes.md', content: 'hello' })
  })
})

// ---------------------------------------------------------------------------
// renameFile
// ---------------------------------------------------------------------------

describe('renameFile', () => {
  it('pushes an undo entry; undoLast calls rename with swapped filenames', async () => {
    useCourseStore.setState({ files: [file('old.md')] })
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify(file('new.md')), { status: 200 })) // rename
        .mockResolvedValueOnce(new Response(JSON.stringify(file('old.md')), { status: 200 })), // undo rename
    )

    await useCourseStore.getState().renameFile('old.md', 'new.md')
    expect(useCourseStore.getState().undoStack).toHaveLength(1)

    await useCourseStore.getState().undoLast()

    const fetchMock = vi.mocked(global.fetch)
    const undoBody = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string)
    expect(undoBody).toEqual({ old_filename: 'new.md', new_filename: 'old.md' })
  })
})

// ---------------------------------------------------------------------------
// createFolder
// ---------------------------------------------------------------------------

describe('createFolder', () => {
  it('pushes an undo entry; undoLast deletes the folder', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify(folder('docs')), { status: 200 })) // POST /api/folders
        .mockResolvedValueOnce(new Response('', { status: 200 })), // DELETE undo
    )

    await useCourseStore.getState().createFolder('docs')
    expect(useCourseStore.getState().undoStack).toHaveLength(1)

    await useCourseStore.getState().undoLast()

    const fetchMock = vi.mocked(global.fetch)
    expect(fetchMock.mock.calls[1][0]).toBe('/api/folders/docs')
    expect((fetchMock.mock.calls[1][1] as RequestInit).method).toBe('DELETE')
  })
})

// ---------------------------------------------------------------------------
// deleteFolder
// ---------------------------------------------------------------------------

describe('deleteFolder', () => {
  it('undoLast re-creates both the folder AND its empty subfolders', async () => {
    useCourseStore.setState({
      folders: [folder('a'), folder('a/sub')],
      files: [file('a/sub/f.md', 'content')],
    })

    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response('', { status: 200 })) // DELETE a
        .mockResolvedValueOnce(new Response(JSON.stringify(folder('a')), { status: 200 })) // POST a (undo)
        .mockResolvedValueOnce(new Response(JSON.stringify(folder('a/sub')), { status: 200 })) // POST a/sub (undo)
        .mockResolvedValueOnce(new Response(JSON.stringify(file('a/sub/f.md', 'content')), { status: 200 })), // POST file (undo)
    )

    await useCourseStore.getState().deleteFolder('a')
    expect(useCourseStore.getState().undoStack).toHaveLength(1)

    await useCourseStore.getState().undoLast()

    const fetchMock = vi.mocked(global.fetch)
    // Calls: DELETE a, POST a, POST a/sub, POST file — 4 total
    expect(fetchMock).toHaveBeenCalledTimes(4)
    const undoFolderBodies = [fetchMock.mock.calls[2][1] as RequestInit]
      .map((init) => JSON.parse(init.body as string))
    expect(undoFolderBodies[0]).toMatchObject({ path: 'a/sub' })
  })
})

// ---------------------------------------------------------------------------
// renameFolder
// ---------------------------------------------------------------------------

describe('renameFolder', () => {
  it('pushes an undo entry; undoLast calls rename with swapped paths', async () => {
    useCourseStore.setState({
      folders: [folder('a'), folder('a/sub')],
      files: [file('a/file.md')],
    })

    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify([file('b/file.md')]), { status: 200 })) // rename a→b
        .mockResolvedValueOnce(new Response(JSON.stringify([file('a/file.md')]), { status: 200 })), // undo b→a
    )

    await useCourseStore.getState().renameFolder('a', 'b')
    expect(useCourseStore.getState().undoStack).toHaveLength(1)

    await useCourseStore.getState().undoLast()

    const fetchMock = vi.mocked(global.fetch)
    const undoBody = JSON.parse((fetchMock.mock.calls[1][1] as RequestInit).body as string)
    expect(undoBody).toEqual({ old_path: 'b', new_path: 'a' })
  })
})

// ---------------------------------------------------------------------------
// pasteItem — same-parent folder copy
// ---------------------------------------------------------------------------

describe('pasteItem', () => {
  it('appends " copy" to the target name when copying a folder to its own parent', async () => {
    useCourseStore.setState({
      folders: [folder('myFolder')],
      files: [],
      clipboard: { kind: 'folder', type: 'copy', path: 'myFolder' },
    })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(folder('myFolder copy')), { status: 200 })),
    )

    // targetFolder=null means root level (same parent as 'myFolder')
    await useCourseStore.getState().pasteItem(null)

    const fetchMock = vi.mocked(global.fetch)
    const body = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)
    expect(body.path).toBe('myFolder copy')
  })
})
