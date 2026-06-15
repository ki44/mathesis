import JSZip from 'jszip'
import type { CourseFile } from '../types'

function triggerDownload(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function exportFile(filename: string, content: string) {
  const name = filename.split('/').pop() ?? filename
  triggerDownload(new Blob([content], { type: 'text/markdown' }), name)
}

export async function exportFolder(folderPath: string, files: CourseFile[]) {
  const prefix = folderPath + '/'
  const matching = files.filter((f) => f.filename.startsWith(prefix))
  if (matching.length === 0) return

  const zip = new JSZip()
  for (const file of matching) {
    const relativePath = file.filename.slice(prefix.length)
    zip.file(relativePath, file.content)
  }

  const blob = await zip.generateAsync({ type: 'blob' })
  const folderName = folderPath.split('/').pop() ?? folderPath
  triggerDownload(blob, `${folderName}.zip`)
}

export async function exportProject(files: CourseFile[]) {
  if (files.length === 0) return

  const zip = new JSZip()
  for (const file of files) {
    zip.file(file.filename, file.content)
  }

  const blob = await zip.generateAsync({ type: 'blob' })
  triggerDownload(blob, 'project.zip')
}
