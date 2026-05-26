# Mathesis

An IDE-like environment with agent that creates and maintains course content. Describe what you need, review the proposed diff hunk-by-hunk, and apply only the changes you want.

---

## Features

- **Chat-driven course creation** — describe what you want and the agent writes or edits course files for you
- **Proposal-based workflow** — changes are staged as proposals; nothing is written until you approve
- **Hunk-level review** — accept or reject individual diff hunks before any change is applied (no git required — diffs are computed from the internal proposal system)
- **Obsidian-compatible** — all course content is plain Markdown, readable in any editor

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| An OpenAI-compatible API key | — |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ki44/mathesis.git
cd mathesis
```

### 2. Backend

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
```

---

## Configuration

Create a `.env` file inside the `src/` directory:

```bash
cp src/.env.example src/.env   # if the example exists, otherwise create it manually
```

**`src/.env`**

```env
# Required
MODEL=                   # Any litellm-compatible model string
API_KEY=                 # Your provider API key

# Optional
BASE_URL=https://api.openai.com/v1    # Override for self-hosted or alternative endpoints
TEMPERATURE=0.7                       # Sampling temperature (default: provider default)
```

> litellm supports OpenAI, Anthropic, Azure, Ollama, and many other providers. See [litellm docs](https://docs.litellm.ai/docs/providers) for the full list of model strings.

---

## Running

Start both servers in separate terminals.

**Backend** (from the repo root, with the virtual environment active):

```bash
cd src/
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

**Frontend** (from the `frontend/` directory):

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## Usage

1. Open `http://localhost:5173` in your browser.
2. Select a course from the sidebar, or ask the agent to create a new one.
3. Type a request in the chat panel — e.g. *"Add a section on integration by parts to calculus.md"*.
4. The agent reads the relevant files, then proposes changes.
5. A diff view opens for each modified file. Review each hunk and **Accept** or **Reject** it.
6. Once all hunks are decided, the accepted changes are applied to the course file.

---

## License

[MIT](LICENSE)
