import { useState } from 'react'
import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

export function calloutTheme(type?: string): { border: string; bg: string; title: string } {
  switch (type?.toLowerCase()) {
    case 'definition': return { border: 'rgba(37,99,235,0.4)', bg: 'rgba(37,99,235,0.06)', title: '#93c5fd' }
    case 'theorem': case 'proposition': return { border: 'rgba(217,119,6,0.4)', bg: 'rgba(217,119,6,0.06)', title: '#fbbf24' }
    default: return { border: 'var(--border)', bg: 'transparent', title: '#7f6df2' }
  }
}

function CalloutBlock({ title, defaultOpen, permanent, calloutType, sourceLine, children }: { title: string; defaultOpen: boolean; permanent?: boolean; calloutType?: string; sourceLine?: number; children: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen)
  const theme = calloutTheme(calloutType)
  const isOpen = permanent || open
  return (
    <div data-source-line={sourceLine} style={{ border: `1px solid ${theme.border}`, borderRadius: 6, margin: '1em 0', overflow: 'hidden', background: theme.bg }}>
      <div
        onClick={permanent ? undefined : () => setOpen(o => !o)}
        style={{ padding: '8px 14px', background: 'var(--bg-2)', color: theme.title, fontWeight: 600, cursor: permanent ? 'default' : 'pointer', display: 'flex', alignItems: 'center', gap: 8, userSelect: 'none' }}
      >
        {!permanent && <span style={{ fontSize: 11, display: 'inline-block', transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>▶</span>}
        {title}
      </div>
      {isOpen && <div style={{ padding: '4px 16px 8px', borderTop: '1px solid var(--border)' }}>{children}</div>}
    </div>
  )
}

function extractNodeText(nodes: any[]): string {
  return (nodes ?? []).map((n: any): string => {
    if (n.type === 'text') return n.value as string
    if (n.type === 'linkReference') return `[${extractNodeText(n.children)}]`
    if (n.children) return extractNodeText(n.children)
    return ''
  }).join('')
}

export const CALLOUT_RE = /^\[!(\w+)\](-?)\s*(.*)/
export function calloutMod(mod: string) {
  return { permanent: mod === '', defaultOpen: mod !== '-' }
}

export function remarkCallout() {
  return (tree: any) => {
    function walk(nodes: any[]) {
      for (const node of nodes) {
        if (node.type === 'blockquote') {
          const first = node.children?.[0]
          if (first?.type === 'paragraph') {
            const text = extractNodeText(first.children).trim()
            const m = text.match(CALLOUT_RE)
            if (m) {
              const [, type, mod, title] = m
              const { permanent, defaultOpen } = calloutMod(mod)
              node.data = { hName: 'div', hProperties: {
                'data-callout': 'true',
                'data-callout-open': String(defaultOpen),
                'data-callout-permanent': String(permanent),
                'data-callout-title': title || type,
                'data-callout-type': type.toLowerCase(),
                'data-source-line': String(node.position?.start?.line ?? ''),
              }}
              node.children = node.children.slice(1)
            }
          }
        }
        if (node.children) walk(node.children)
      }
    }
    walk(tree.children)
  }
}

const ROOT_STYLE: React.CSSProperties = {
  color: 'var(--text-1)',
  lineHeight: 1.7,
  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  fontSize: 14,
  overflowY: 'auto',
  padding: '24px 32px',
  maxWidth: 800,
  margin: '0 auto',
  width: '100%',
  boxSizing: 'border-box',
}

// Reads source line from hast node for LiveMarkdownPreview cursor tracking
function srcLine(node: any): number | undefined {
  return node?.position?.start?.line
}

export const markdownComponents: React.ComponentProps<typeof ReactMarkdown>['components'] = {
  h1: ({ children, node }: any) => (
    <h1 data-source-line={srcLine(node)} style={{ fontSize: '1.8em', fontWeight: 700, borderBottom: '1px solid var(--border)', paddingBottom: '0.3em', marginTop: '1.5em', marginBottom: '0.8em', color: 'var(--text-bright)' }}>{children}</h1>
  ),
  h2: ({ children, node }: any) => (
    <h2 data-source-line={srcLine(node)} style={{ fontSize: '1.4em', fontWeight: 600, borderBottom: '1px solid var(--border)', paddingBottom: '0.2em', marginTop: '1.4em', marginBottom: '0.6em', color: 'var(--text-bright)' }}>{children}</h2>
  ),
  h3: ({ children, node }: any) => (
    <h3 data-source-line={srcLine(node)} style={{ fontSize: '1.15em', fontWeight: 600, marginTop: '1.2em', marginBottom: '0.4em', color: 'var(--text-bright)' }}>{children}</h3>
  ),
  h4: ({ children, node }: any) => (
    <h4 data-source-line={srcLine(node)} style={{ fontSize: '1em', fontWeight: 600, marginTop: '1em', marginBottom: '0.4em', color: 'var(--text-bright)' }}>{children}</h4>
  ),
  p: ({ children, node }: any) => (
    <p data-source-line={srcLine(node)} style={{ margin: '0.75em 0' }}>{children}</p>
  ),
  a: ({ href, children }) => (
    <a href={href} style={{ color: '#7f6df2', textDecoration: 'none' }} target="_blank" rel="noreferrer">{children}</a>
  ),
  strong: ({ children }) => (
    <strong style={{ color: 'var(--text-bright)', fontWeight: 600 }}>{children}</strong>
  ),
  em: ({ children }) => (
    <em style={{ color: 'var(--text-em)' }}>{children}</em>
  ),
  code: ({ children, className }) => {
    const isBlock = !!className || String(children).includes('\n')
    if (isBlock) return <code style={{ display: 'block', fontFamily: "'Fira Code', 'Cascadia Code', Consolas, monospace", fontSize: '0.9em' }}>{children}</code>
    return <code style={{ background: 'var(--bg-code-inline)', padding: '0.2em 0.4em', borderRadius: 3, fontFamily: "'Fira Code', Consolas, monospace", fontSize: '0.88em', color: 'var(--text-code-inline)' }}>{children}</code>
  },
  pre: ({ children, node }: any) => (
    <pre data-source-line={srcLine(node)} style={{ background: 'var(--bg-code)', padding: '1em', borderRadius: 6, overflowX: 'auto', margin: '1em 0', border: '1px solid var(--border-code)', lineHeight: 1.5 }}>{children}</pre>
  ),
  blockquote: ({ children, node }: any) => (
    <blockquote data-source-line={srcLine(node)} style={{ borderLeft: '3px solid #7f6df2', paddingLeft: '1em', margin: '1em 0', color: 'var(--text-2)', fontStyle: 'italic' }}>{children}</blockquote>
  ),
  ul: ({ children, node }: any) => (
    <ul data-source-line={srcLine(node)} style={{ paddingLeft: '1.5em', margin: '0.5em 0' }}>{children}</ul>
  ),
  ol: ({ children, node }: any) => (
    <ol data-source-line={srcLine(node)} style={{ paddingLeft: '1.5em', margin: '0.5em 0' }}>{children}</ol>
  ),
  li: ({ children, node }: any) => (
    <li data-source-line={srcLine(node)} style={{ margin: '0.2em 0' }}>{children}</li>
  ),
  table: ({ children, node }: any) => (
    <table data-source-line={srcLine(node)} style={{ borderCollapse: 'collapse', width: '100%', margin: '1em 0', fontSize: '0.95em' }}>{children}</table>
  ),
  th: ({ children }) => (
    <th style={{ border: '1px solid var(--border-2)', padding: '0.5em 0.8em', background: 'var(--bg-2)', textAlign: 'left', color: 'var(--text-bright)' }}>{children}</th>
  ),
  td: ({ children }) => (
    <td style={{ border: '1px solid var(--border)', padding: '0.4em 0.8em' }}>{children}</td>
  ),
  hr: () => (
    <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '1.5em 0' }} />
  ),
  div: ({ children, ...props }: any) => {
    if (props['data-callout'] === 'true') {
      return (
        <CalloutBlock
          title={props['data-callout-title'] as string}
          defaultOpen={props['data-callout-open'] === 'true'}
          permanent={props['data-callout-permanent'] === 'true'}
          calloutType={props['data-callout-type'] as string}
          sourceLine={props['data-source-line'] ? Number(props['data-source-line']) : undefined}
        >
          {children}
        </CalloutBlock>
      )
    }
    return <div>{children}</div>
  },
}

export function MarkdownRenderer({ content, compact }: { content: string; compact?: boolean }) {
  const rootStyle: React.CSSProperties = compact
    ? { ...ROOT_STYLE, padding: 0, maxWidth: 'none', margin: 0 }
    : ROOT_STYLE

  return (
    <div style={rootStyle}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkCallout]}
        rehypePlugins={[rehypeKatex]}
        components={markdownComponents}
      >{content}</ReactMarkdown>
    </div>
  )
}
