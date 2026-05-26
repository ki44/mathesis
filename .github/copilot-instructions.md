# Mathesis — Agent Instructions

Mathesis is a chat-driven course authoring tool. Users describe what they want; an LLM agent proposes diffs; users review and accept changes hunk-by-hunk. See [README.md](../README.md) for full setup instructions.

---

## Dev Commands

| Task | Command | CWD |
|---|---|---|
| Start backend | `uvicorn src.main:app --reload` | repo root |
| Start frontend | `npm run dev` | `frontend/` |
| Lint/format (Python) | `ruff check . --fix` | repo root |
| Run tests | `pytest` | repo root |

Both servers must be running simultaneously. Backend: `http://localhost:8000`. Frontend: `http://localhost:5173`.

---

## Configuration

Create `src/.env` (not repo root). Required keys:

```text
MODEL=         # litellm model string, e.g. openai/gpt-4o
API_KEY=       # provider API key
BASE_URL=      # optional, override endpoint
TEMPERATURE=   # optional
MAX_TOKENS=    # optional
MAX_ITERATIONS= # optional, agent loop guard
```

`ModelConfiguration` in [src/schemas/schemas.py](../src/schemas/schemas.py) is a `pydantic-settings` `BaseSettings` that reads this file. The DB path (`mathesis.db`) is resolved relative to `db.py` — it is **not** configurable.

---

## Architecture

```text
src/                        ← Python / FastAPI backend
  main.py                   ← app entry, endpoints, system prompt, Agent construction
  agent.py                  ← Agent class: acompletion() and stream() (SSE generator)
  agent_tools/
    tools_utils.py          ← @tool decorator: builds OpenAI JSON schema from Pydantic model
    storage/
      db.py                 ← get_db() context manager, init_db()
      db_tools.py           ← the three @tool functions (list/read/propose)
      schemas.py            ← Pydantic param models for each tool
  schemas/schemas.py        ← FastAPI-layer Pydantic models + ModelConfiguration

frontend/src/
  App.tsx                   ← layout, resize panel logic
  store/                    ← three Zustand stores (chat, course, theme) — actions inside create()
  hooks/                    ← useChat (orchestration), useCourse (init), useContextMenuClose
  components/
    ChatPanel.tsx           ← conversation list + streaming message view
    FileView.tsx            ← Monaco DiffEditor with hunk accept/reject widgets
    Sidebar.tsx             ← course file list with proposal indicators
    MarkdownRenderer.tsx    ← renders Markdown course content
```

---

## Key Conventions

### Adding a new agent tool (backend)

1. Add a Pydantic `BaseModel` for parameters in [src/agent_tools/storage/schemas.py](../src/agent_tools/storage/schemas.py) — name it `<ToolName>Params`.
2. Write an `async` function decorated with `@tool(description="...", parameters=MyParams)` in [src/agent_tools/storage/db_tools.py](../src/agent_tools/storage/db_tools.py).
3. Import the function and add it to the `tools=[...]` list in [src/main.py](../src/main.py).

There is no auto-discovery — tools must be explicitly listed at `Agent()` construction time.

### Python style

- `snake_case` for functions/variables; `PascalCase` for classes; `_prefix` for module-level private names.
- Line length: 120 (`ruff`). Linting rules: `E`, `F`, `I` — run `ruff check . --fix` before committing.
- Error handling: raise `HTTPException` directly (no custom exceptions). Only validate at boundaries.
- `load_dotenv()` is already called in `main.py` — no need to add it elsewhere.

### TypeScript / React style

- `interface` for object shapes, `type` for unions/aliases. No explicit return types — infer.
- `import type` for type-only imports.
- Three separate Zustand stores by domain; actions defined inside `create()`; use granular selector subscriptions (`useChatStore((s) => s.field)`).
- No CSS modules or utility frameworks — use inline styles + CSS custom properties (`var(--bg-1)`, `var(--border)`) for theming.
- Module-level constants for stable values that would cause re-renders if inlined (e.g., `NO_MESSAGES = []`).

### API / networking

- All API paths use the `/api/` prefix. Vite proxies `/api` → `http://localhost:8000` in dev (see [frontend/vite.config.ts](../frontend/vite.config.ts)).
- Use native `fetch` only — no axios.
- Chat streaming uses a `POST` to `/api/chat/stream`, parsed manually via `ReadableStream` + `TextDecoder`. **Do not use `EventSource`** (GET-only).

---

## Common Pitfalls

- **CORS**: `http://localhost:5173` is hardcoded in [src/main.py](../src/main.py). Update it if the frontend port changes.
- **Monaco hunk race condition**: `onDidUpdateDiff` can fire before Monaco has computed diffs. A 300 ms timeout fallback exists in [FileView.tsx](../frontend/src/components/FileView.tsx) — do not remove it.
- **Hunk merge ordering**: `computeMergedContent()` must iterate hunks in **descending** line-number order to avoid index drift when splicing. Keep this invariant.
- **No test files exist yet**. `pytest` is installed but the test suite is empty. Write tests in `src/tests/` when adding new functionality.

## Coding style

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
