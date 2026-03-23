# ThinkTree v2.0 — MVP Prompt
# =================================================================
# VERSION HISTORY:
#   v9.3 → v2.0: Simplified for rapid MVP development on AI builders
#                (e.g. base44). All core product features preserved.
#
# KEY CHANGES FROM v9.3:
#   - Single-layer workflow: one prompt, one patch file (db/index.ts)
#   - Security: CSP removed, API keys via simple localStorage
#   - Summarizer: retained but simplified (single activeTopic flag,
#     no generation counter, no Guard 2)
#   - MindMap: static layout + whole-canvas pan only (no pinch-zoom,
#     no per-node drag, no coordinate system conversion)
#   - contextBuilder: inline, no stale-summary guard complexity
#   - store/index.ts: inline spec, no separate patch needed
#   - aiAdapter.ts: inline spec, no separate patch needed
#   - Node limits: 200-node hard cap retained; depth limit removed;
#     amber/red warning banners removed
#
# WORKFLOW:
#   STEP 1  — Feed Part 1 (Product Spec) to AI builder
#   STEP 2  — Feed Part 2 (Architecture + Inline Implementations)
#   STEP 3  — Feed Part 3 (Components + UI)
#   STEP 4  — Generate full project
#   STEP 5  — Overwrite ONE file: src/db/index.ts (Part 4 patch)
#   STEP 6  — Run verify checklist (Part 5)
#
# PATCH FILES (only one):
#   src/db/index.ts   ← copy from Part 4 exactly after generation
# =================================================================


================================================================
  PART 1 — PRODUCT SPEC
  Feed to AI builder as Step 1.
  Suggested prompt: "Read this product spec carefully.
  Do not generate any code yet. Confirm you understand
  the product, UI regions, and branching model."
================================================================

---

## PRODUCT VISION

Build a local-first web application called ThinkTree.

ThinkTree is an AI-powered learning and research workspace that replaces
linear AI chat with branching conversations. It combines:
- An AI chat interface
- A visual conversation tree (list view + mind map view)
- A research notebook

ThinkTree is NOT an AI provider. It is a visual interface and context
manager that connects users to their own API keys. All AI intelligence
comes from the user's chosen provider.

---

## CORE PRODUCT RULES — STRICT

- No backend server
- No cloud database
- No server-side API proxy
- All conversation data stays inside the browser (IndexedDB via Dexie.js)
- UI preferences stored in localStorage
- API keys stored in localStorage (MVP simplification — no session/local split)
- Static web application only
- ThinkTree must never send prompts to any server except the user's chosen AI provider
- ThinkTree must never transmit API keys to any server other than the chosen AI provider

---

## CORE IDEA: BRANCHING CONVERSATIONS

Traditional AI chat is linear: Q1 → A1 → Q2 → A2

ThinkTree introduces branching:

```
Root Question
├── Branch A
│   └── Q2 → A2 → Q3 → A3
├── Branch B
│   └── Q2b → A2b
└── Branch C
    └── Q2c
```

- Each node can spawn multiple child branches
- Each path represents a separate thinking direction
- AI context only includes nodes along the current branch path
- Users can start a new branch from ANY previous message

---

## APPLICATION DATA STRUCTURE

```
Workspace
└── Topic (e.g. "Bayesian Regression", "Option Pricing")
    └── Conversation Tree
        └── Node (smallest element)
```

Each Topic has exactly ONE root node and ONE branching tree.
Multiple Topics = multiple independent trees, each with its own root.

---

## NODE LIMIT PER TOPIC

Each topic tree has a hard node limit of 200 non-summary nodes.
When the limit is reached:
- MessageInput textarea: DISABLED
- Send button: DISABLED
- "+ Branch" button on all message bubbles: HIDDEN

No warning banners. No depth limit. No branch-depth tracking.

Node count display in Topic Dropdown:
- count < 200: muted text
- count = 200: red text "200 / 200"

What limits do NOT affect:
- Editing existing nodes
- Deleting nodes / topics
- Switching to a different topic or branch

---

## UI LAYOUT OVERVIEW

```
┌──────────────────────────────────────────────────────────┐
│  TopBar (48px fixed)                                      │
│  [Topic ▾] | [Branch pill]    [spacer]  [⬇ Export] [⚙]  │
├──────────────┬────────────────────────────────────────────┤
│  Left Panel  ║  Center Panel                              │
│  (resizable) ║  Conversation area (scrollable)            │
│  240px       ║                                            │
│  default     ║  Message Input (fixed at bottom)           │
└──────────────╨────────────────────────────────────────────┘
                     Settings Drawer overlays from right
                     edge on demand (z-index above center)
```

---

## TOPBAR

Height: 48px, full width, fixed at top.

Left side (left → right):
- Topic Selector dropdown trigger: [colored dot] [topic name ▾]
  - Max width 200px, truncated with ellipsis
  - Clicking opens Topic Dropdown Panel
- Vertical divider (0.5px, 20px tall)
- Active Branch Pill (purple badge, shows current branch path)

Right side (flex spacer separates):
- Export button (↓ icon)
- Settings button (⚙ icon) → opens Settings Drawer

---

## TOPIC DROPDOWN PANEL

Appears below trigger, width 240px, z-index above Left Panel.

Structure:
1. Header: "TOPICS" (10px uppercase muted)
2. Topic list items
3. "＋ New topic" row
4. Inline new-topic input row (hidden until + clicked)

Per topic row (32px height):
- [colored dot] [topic name] [node count badge]
  Node count badge: muted if < 200, red text "200/200" if at limit
- On hover: reveal [✎ rename] [🗑 delete] buttons

Rename: inline edit → Enter/blur commits → Escape cancels
Delete: confirmation popover "Delete '[name]'? This cannot be undone."
  → auto-switch to nearest remaining topic if deleted was active
  → empty state if no topics remain

"＋ New topic":
- Shows inline input → Enter/Add creates topic with empty root node
- Auto-assigns color from 4-color rotation
- Switches to new topic immediately

Topic color rotation (auto-assigned, wraps after 4):
| # | Color  | Hex     |
|---|--------|---------|
| 1 | Purple | #7F77DD |
| 2 | Teal   | #1D9E75 |
| 3 | Amber  | #EF9F27 |
| 4 | Coral  | #D85A30 |

Persistence: localStorage key "thinktree_active_topic"

---

## LEFT PANEL

Default width: 240px | Min: 150px | Max: 380px

Resize: 4px drag handle between Left and Center panels.
Uses onPointerDown/onPointerMove/onPointerUp (unified mouse + touch).
Persists width: localStorage key "thinktree_left_width".

### Left Panel Header (32px):
- Current topic name (11px, muted, truncated)
- List / Map view toggle (pill-style, right-aligned)

### LIST VIEW (default):
Hierarchical tree navigator as indented list.

Node rendering:
- Indent: 12px per depth level (capped visually at 3 levels)
- Dot colors: root=purple, user_question=blue, ai_response=teal, branch=amber
- Active node: white bg + bold + 2px purple left border
- summary nodes: NEVER rendered in list view

Interactions:
- Click → navigate to that branch (set activeNodeId)
- Right-click / long-press → context menu: "Start new branch"
  "Start new branch" is DISABLED when topic is at NODE_LIMIT_MAX (200)

Section labels (10px uppercase muted): "ROOT", "BRANCH A", etc.

### MAP VIEW:
SVG-based mind map. Same tree data, static radial auto-layout.

Node rendering:
- root → larger circle, purple fill (#EEEDFE, stroke #7F77DD)
- user_question → medium circle, blue fill
- ai_response → medium circle, teal fill
- active node → amber highlight ring
- summary nodes → NOT rendered

Layout: radial auto-layout
- Root at canvas center
- Children spread in 70° arc around parent
- Radius increases by 90px per depth level

Interactions:
- Single pointer drag on EMPTY CANVAS → pan the whole canvas (translate)
- Tap on a node (< 5px movement) → navigate to that branch
- NO pinch-zoom
- NO per-node drag/reposition
- NO scale transforms

SVG element must have: style={{ touchAction: 'none' }}

Persistence: localStorage key "thinktree_mindmap_{topic_id}"
Stores: { pan: {x, y} } — only pan offset, no scale, no per-node positions

---

## CENTER PANEL

Fills remaining width (flex: 1, min-width: 200px).

Structure top to bottom:
1. ConversationFeed (flex: 1, overflow-y: auto)
2. MessageInput (fixed height at bottom)

### CONVERSATION FEED:
Displays nodes along active branch path (root → activeNode), chronological.

User messages: right-aligned, purple bubble (#EEEDFE bg, dark purple text)
AI messages: left-aligned, secondary bg bubble with 0.5px border
summary nodes: NEVER rendered
Only ONE node may be in edit mode at a time

On hover over any message: show timestamp · [Edit] · [+ Branch]
  [+ Branch] is HIDDEN (not disabled) when topic is at NODE_LIMIT_MAX
On hover over AI messages: show "AI" label · [+ Branch]

### MESSAGE EDIT — MID-CONVERSATION TRUNCATION:
Trigger: Edit button (fade in on hover) on any user_question bubble.
Edit is ALWAYS available regardless of node limit.

On clicking Edit:
- Bubble replaced by: textarea (pre-filled) + [Save & regenerate] + [Cancel]

On "Save & regenerate":
1. Update node content in DB
2. Delete ALL descendant nodes (recursive, atomic)
3. Create ONE new ai_response placeholder node as child
4. Set activeNodeId = new AI node
5. Re-render Left Panel (List + Map) IMMEDIATELY
6. Show TruncationNotice bar
7. Fire AI API call with updated context
8. On response: replace placeholder with real content

On "Cancel": restore original bubble, no changes.

Constraints:
- Edit only available on user_question nodes, NOT ai_response nodes
- API failure after save: show error in AI bubble, allow retry

### TRUNCATION NOTICE BAR:
Amber background, 0.5px top amber border, 11px text, flex row.
Content: ⚠ "Messages after the edited question were removed.
            The tree has been updated."  [Dismiss]
Dismiss: hides bar.
Also hides on topic switch.

### MESSAGE INPUT:
- Textarea (height 54px, vertically resizable by user)
- Send button (purple, right-aligned)
- Enter = submit; Shift+Enter = newline
- Disabled states (textarea + Send button both disabled):
  a) No valid API key configured
  b) Topic has reached NODE_LIMIT_MAX (200 non-summary nodes)

---

## SETTINGS DRAWER

Trigger: ⚙ button in TopBar.
Animation: slides in from right edge (CSS translateX, 0.2s ease).
Width: 220px. Overlays Center Panel — does NOT push or resize it.
Close: × button inside / click outside / press Escape.

Contents:
- AI provider selector (OpenAI / DeepSeek / Anthropic)
- Model name (editable, default per provider)
- API key input (masked with ••••, show/hide toggle)
- "Clear key" button
- Context window (read-only: "Last 12 msgs + auto-summary")
- Export JSON button
- Export Markdown button

API key storage: localStorage key "thinktree_api_key_{provider}"
Show masked key on open: last 4 chars visible, rest replaced with ••••

---

## EXPORT

Export dialog (opened from TopBar export button).
Choose scope:
- Current branch path only → Markdown
- Entire topic tree → JSON

JSON format:
```json
{
  "export_version": "1.0",
  "exported_at": "ISO-8601 timestamp",
  "topic": { "id": "...", "name": "...", "created_at": 0 },
  "nodes": [
    { "id": "...", "parent_id": null, "role": "user_question",
      "content": "...", "timestamp": 0, "covers_up_to": null }
  ]
}
```
- summary nodes INCLUDED in JSON
- NEVER include API keys in exported files

Markdown format (current branch path):
```markdown
# [Topic Name]
**Branch:** [path]  **Exported:** [date]
---
**You:** [question]

**[Provider]:** [answer]
---
```
- summary nodes EXCLUDED from Markdown

---

## PRODUCT GOAL

ThinkTree should feel like a structured thinking environment powered
by AI — where users explore ideas through branching conversations
rather than linear chat threads.

Key experience principles:
1. Privacy first — all data stays in the browser, always
2. Non-destructive branching — any message can spawn a new path
3. Reliable persistence — all writes go through Dexie transactions;
   the UI only updates after the DB confirms the write succeeded
4. Bring your own AI — works with any supported provider,
   no account required beyond an API key


================================================================
  PART 2 — ARCHITECTURE + INLINE IMPLEMENTATIONS
  Feed to AI builder as Step 2.
  Suggested prompt: "This is the technical architecture.
  Implement all files exactly as specified. Pay close
  attention to files marked with ⚠ IMPLEMENT EXACTLY."
================================================================

## TECH STACK

| Layer        | Choice                                        |
|--------------|-----------------------------------------------|
| Framework    | React + Next.js (output: "export")            |
| Language     | TypeScript                                    |
| Styling      | TailwindCSS                                   |
| Local DB     | Dexie.js v4 (wraps IndexedDB)                 |
| State        | Zustand                                       |
| AI SDK       | @anthropic-ai/sdk (Anthropic provider only)   |

```bash
npm install dexie zustand @anthropic-ai/sdk
```

---

## DATA MODEL

```typescript
// src/db/types.ts — copy exactly, do not modify

export interface Node {
  id:           string
  topic_id:     string
  parent_id:    string | null
  role:         'user_question' | 'ai_response' | 'summary'
  content:      string
  timestamp:    number
  covers_up_to: number | null   // summary nodes only
}

export interface Topic {
  id:           string
  workspace_id: string
  name:         string
  color_index:  number          // 0–3
  created_at:   number
}

export interface Workspace {
  id:         string
  name:       string
  created_at: number
}
```

---

## NODE LIMIT CONSTANTS

```typescript
// src/lib/nodeLimits.ts — implement in full

export const NODE_LIMIT_MAX = 200

export function getNodeCount(nodes: Node[], topicId: string): number {
  return nodes.filter(n => n.topic_id === topicId && n.role !== 'summary').length
}

export function canAddNode(nodes: Node[], topicId: string): boolean {
  return getNodeCount(nodes, topicId) < NODE_LIMIT_MAX
}
```

---

## AI ADAPTER

⚠ IMPLEMENT EXACTLY — AI builders get the Anthropic format wrong.

```typescript
// src/lib/aiAdapter.ts
import Anthropic from '@anthropic-ai/sdk'

export interface Message {
  role:    'system' | 'user' | 'assistant'
  content: string
}

export async function sendMessage(
  provider: string,
  apiKey:   string,
  messages: Message[]
): Promise<string> {
  switch (provider) {

    case 'openai': {
      const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model:    'gpt-4o-mini',
          messages: messages.map(m => ({ role: m.role, content: m.content })),
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw classifyError(res.status, err)
      }
      const data = await res.json()
      return data.choices[0].message.content as string
    }

    case 'deepseek': {
      const res = await fetch('https://api.deepseek.com/chat/completions', {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model:    'deepseek-chat',
          messages: messages.map(m => ({ role: m.role, content: m.content })),
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw classifyError(res.status, err)
      }
      const data = await res.json()
      return data.choices[0].message.content as string
    }

    case 'anthropic': {
      // ── CRITICAL: Anthropic system prompt is a TOP-LEVEL parameter ──
      // It must NOT be inside the messages array.
      const systemMsg = messages.find(m => m.role === 'system')
      const chatMsgs  = messages
        .filter(m => m.role !== 'system')
        .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }))

      const client = new Anthropic({
        apiKey,
        dangerouslyAllowBrowser: true,
      })

      const res = await client.messages.create({
        model:      'claude-sonnet-4-5',
        max_tokens: 4096,
        system:     systemMsg?.content ?? 'You are a helpful assistant.',
        messages:   chatMsgs,
      })

      // ── CRITICAL: use .find(), NOT [0] — block order is not guaranteed ──
      const textBlock = res.content.find(b => b.type === 'text')
      if (!textBlock || textBlock.type !== 'text') {
        throw new Error('No text block in Anthropic response')
      }
      return textBlock.text
    }

    default:
      throw new Error(`Unknown provider: ${provider}`)
  }
}

function classifyError(status: number, body: any): Error {
  if (status === 401) return Object.assign(new Error('invalid_api_key'), { type: 'invalid_api_key' })
  if (status === 429) return Object.assign(new Error('rate_limit'),       { type: 'rate_limit' })
  return Object.assign(
    new Error(body?.error?.message ?? 'network_failure'),
    { type: 'network_failure' }
  )
}
```

---

## CONTEXT BUILDER

⚠ IMPLEMENT EXACTLY — branch isolation and summary injection must be correct.

```typescript
// src/lib/contextBuilder.ts
import { getNode, getLatestSummaryNode } from '@/db'
import type { Node }    from '@/db/types'
import type { Message } from '@/lib/aiAdapter'

const CONTEXT_TOKEN_BUDGET = 20_000
const MIN_WINDOW_SIZE      = 2

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3)
}

// Traverses parent_id chain from nodeId to root.
// Excludes summary nodes from the returned path.
// Result is ordered root → nodeId.
export async function getBranchPath(nodeId: string): Promise<Node[]> {
  const path: Node[] = []
  let current: string | null = nodeId

  while (current) {
    const node = await getNode(current)
    if (!node) break
    if (node.role !== 'summary') path.push(node)
    current = node.parent_id
  }

  return path.reverse()
}

// Builds the message array to send to the AI provider.
// Includes: system prompt, optional summary pair, recent window.
export async function buildContext(
  nodeId:  string,
  topicId: string
): Promise<Message[]> {

  // Step 1: full branch path (summary nodes excluded)
  const path = await getBranchPath(nodeId)

  // Step 2: take last 12 as the recent window
  let recentWindow = path.slice(-12)

  // Step 3: apply token budget — trim oldest if over limit
  let windowTokens = recentWindow.reduce(
    (sum, n) => sum + estimateTokens(n.content), 0
  )
  while (windowTokens > CONTEXT_TOKEN_BUDGET && recentWindow.length > MIN_WINDOW_SIZE) {
    const removed = recentWindow.shift()!
    windowTokens -= estimateTokens(removed.content)
  }

  // Step 4: find summary for this branch
  const branchRootId = path[0]?.id ?? nodeId
  const summaryNode  = await getLatestSummaryNode(topicId, branchRootId)

  // Step 5: assemble context
  const messages: Message[] = [
    { role: 'system', content: 'You are a helpful assistant.' },
  ]

  if (summaryNode && summaryNode.covers_up_to !== null) {
    // Stale summary guard: only inject if summary does NOT overlap recentWindow.
    // recentWindow starts at index (path.length - recentWindow.length).
    // If covers_up_to >= that start index, the summary overlaps — skip it.
    const recentStartIndex = path.length - recentWindow.length
    if (summaryNode.covers_up_to < recentStartIndex) {
      // Inject as user/assistant pair — accepted by all three providers
      messages.push({ role: 'user',      content: summaryNode.content })
      messages.push({ role: 'assistant', content: 'Understood.' })
    }
  }

  // Step 6: append recent messages
  recentWindow.forEach(n => {
    messages.push({
      role:    n.role === 'user_question' ? 'user' : 'assistant',
      content: n.content,
    })
  })

  return messages
}
```

---

## SUMMARIZER

⚠ IMPLEMENT EXACTLY — the IIFE pattern is required for fire-and-forget correctness.

```typescript
// src/lib/summarizer.ts
import { sendMessage }                    from '@/lib/aiAdapter'
import { upsertSummaryNode }              from '@/db'
import type { Node }                      from '@/db/types'
import type { Message }                   from '@/lib/aiAdapter'

// Single active topic tracker.
// When the user switches topics, setActiveTopic() updates this.
// Any in-flight summary job checks this before writing — if the topic
// has changed, the result is discarded.
let activeSummaryTopicId: string | null = null

export function setActiveTopic(topicId: string): void {
  activeSummaryTopicId = topicId
}

// Called on topic switch to invalidate in-flight jobs for old topic.
export function clearActiveTopic(): void {
  activeSummaryTopicId = null
}

// Fire-and-forget: always called without await at the call site.
// Triggers a summary API call when path.length is a multiple of 10
// and there are more than 12 messages (so there is something to summarize).
export function triggerSummaryIfNeeded(
  path:     Node[],
  provider: string,
  apiKey:   string,
  topicId:  string
): void {
  const toSummarize = path.slice(0, path.length - 12)
  if (toSummarize.length === 0) return

  const branchRootId    = path[0].id
  const capturedTopicId = topicId   // captured in closure before first await

  ;(async () => {
    try {
      const text = await sendMessage(provider, apiKey, buildSummarizationMessages(toSummarize))

      // Guard: discard if user has switched topics while we were waiting
      if (activeSummaryTopicId !== capturedTopicId) {
        console.debug('[summarizer] discarded: topic switched')
        return
      }

      await upsertSummaryNode({
        id:           `summary_${topicId}_${branchRootId}`,
        topic_id:     topicId,
        parent_id:    branchRootId,
        role:         'summary',
        content:      text,
        timestamp:    Date.now(),
        covers_up_to: path.length - 13,
      })

    } catch (err) {
      // Summary failure is always silent — never blocks main message flow
      console.warn('[summarizer] generation failed:', err)
    }
  })()
}

export function buildSummarizationMessages(nodes: Node[]): Message[] {
  const conversation = nodes
    .map(n => {
      const role = n.role === 'user_question' ? 'User' : 'Assistant'
      return `${role}: ${n.content}`
    })
    .join('\n')

  return [
    {
      role:    'system',
      content: 'You are a conversation summarizer. Output only the summary.\nNo preamble, no labels, no bullet points.',
    },
    {
      role:    'user',
      content: [
        'Summarize the following conversation history in 3 concise sentences.',
        'Preserve: key questions asked, conclusions reached, constraints or',
        'assumptions established, and any decisions made.',
        'Do not include greetings or filler content.',
        '',
        'Conversation:',
        conversation,
      ].join('\n'),
    },
  ]
}
```

---

## ZUSTAND STORE

⚠ IMPLEMENT EXACTLY — DB-first rule must not be violated.

```typescript
// src/store/index.ts
import { create } from 'zustand'
import { getNodesByTopic } from '@/db'
import { clearActiveTopic, setActiveTopic } from '@/lib/summarizer'
import type { Node } from '@/db/types'

interface AppStore {
  nodes:        Node[]
  activeNodeId: string | null

  addNode:           (node: Node) => void
  updateNodeContent: (id: string, content: string) => void
  removeDescendants: (nodeId: string) => void
  setActiveNodeId:   (id: string | null) => void
  loadTopicNodes:    (topicId: string) => Promise<void>
}

export const useAppStore = create<AppStore>((set, get) => ({
  nodes:        [],
  activeNodeId: null,

  addNode: (node) =>
    set(state => ({ nodes: [...state.nodes, node] })),

  updateNodeContent: (id, content) =>
    set(state => ({
      nodes: state.nodes.map(n => n.id === id ? { ...n, content } : n),
    })),

  // Removes ALL descendants of nodeId from in-memory array using BFS.
  // Does NOT remove nodeId itself.
  // ⚠ Must use BFS — simple filter(n.parent_id !== nodeId) only removes
  //   direct children, leaving grandchildren as orphans.
  removeDescendants: (nodeId) => {
    const allNodes = get().nodes
    const toRemove = new Set<string>()
    const queue    = [nodeId]

    while (queue.length) {
      const current = queue.shift()!
      allNodes
        .filter(n => n.parent_id === current)
        .forEach(child => {
          toRemove.add(child.id)
          queue.push(child.id)
        })
    }

    set(state => ({
      nodes: state.nodes.filter(n => !toRemove.has(n.id)),
    }))
  },

  setActiveNodeId: (id) => set({ activeNodeId: id }),

  // Replaces entire nodes array on app boot or topic switch.
  // ⚠ MUST call clearActiveTopic() then setActiveTopic() to keep
  //   summarizer in sync with the currently active topic.
  loadTopicNodes: async (topicId) => {
    clearActiveTopic()
    setActiveTopic(topicId)

    const nodes = await getNodesByTopic(topicId)

    set({
      nodes,
      activeNodeId: nodes
        .filter(n => n.role !== 'summary')
        .at(-1)?.id ?? null,
    })
  },
}))
```

---

## MESSAGE FLOW

### Normal message send:

```
Guards (check in order, fail fast):

  Guard 1 — API key:
    apiKey = localStorage.getItem('thinktree_api_key_{provider}')
    if (!apiKey) → show inline notice "Add API key in Settings", return early

  Guard 2 — Topic node limit:
    if (getNodeCount(nodesState, topic_id) >= NODE_LIMIT_MAX)
      show inline notice "Topic full — start a new topic", return early

Main flow:
  1.  userNode = { id: uuid(), topic_id, parent_id: activeNodeId,
                   role: 'user_question', content, timestamp: Date.now(),
                   covers_up_to: null }
  2.  await saveNode(userNode)              // from db/index.ts
  3.  addNode(userNode)                     // Zustand — AFTER saveNode resolves
  4.  path = await getBranchPath(userNode.id)
  5.  if (path.length >= 10 && path.length % 10 === 0):
        triggerSummaryIfNeeded(path, provider, apiKey, topicId)
        // fire-and-forget — do NOT await
  6.  context = await buildContext(userNode.id, topicId)
  7.  reply = await sendMessage(provider, apiKey, context)
  8.  aiNode = { id: uuid(), topic_id, parent_id: userNode.id,
                 role: 'ai_response', content: reply,
                 timestamp: Date.now(), covers_up_to: null }
  9.  await saveNode(aiNode)
  10. addNode(aiNode)                       // Zustand — AFTER saveNode resolves
  11. setActiveNodeId(aiNode.id)
```

### Edit message send:

```
  1.  await atomicEditSave(editedNodeId, newContent, newAiPlaceholder)
  2.  updateNodeContent(editedNodeId, newContent)   // Zustand
  3.  removeDescendants(editedNodeId)               // Zustand
  4.  addNode(newAiPlaceholder)                     // Zustand
  5.  setActiveNodeId(newAiPlaceholder.id)
  6.  show TruncationNotice bar
  7.  reply = await sendMessage(provider, apiKey, await buildContext(editedNodeId, topicId))
  8.  await updateNodeContent(newAiPlaceholder.id, reply)   // db
  9.  updateNodeContent(newAiPlaceholder.id, reply)         // Zustand
```

### Branch creation:

```
  Guard: if (!canAddNode(nodesState, topic_id)) return

  1.  branchNode = { id: uuid(), topic_id, parent_id: clickedNodeId,
                     role: 'user_question', content: '',
                     timestamp: Date.now(), covers_up_to: null }
  2.  await saveNode(branchNode)
  3.  addNode(branchNode)
  4.  setActiveNodeId(branchNode.id)
  5.  focus MessageInput
```

---

## ERROR HANDLING

| Error type           | Behavior                                                     |
|----------------------|--------------------------------------------------------------|
| api_key_missing      | Amber inline notice in MessageInput. No node created.        |
| node_limit_reached   | Amber inline notice. No node created.                        |
| invalid_api_key      | Inline error in message feed, link to Settings               |
| network_failure      | Retry button on failed AI bubble                             |
| rate_limit           | Amber notice with message "Rate limited — please wait"       |
| summary_failure      | Silent — console.warn only, never blocks main message        |
| edit_save_failure    | Error in EditForm, keep edit open, do NOT truncate           |
| db_transaction_fail  | Toast: "Save failed — your data is unchanged."               |

---

## DEPLOYMENT

Static export — no server required.
next.config.js:
```javascript
module.exports = {
  output: 'export',
}
```


================================================================
  PART 3 — COMPONENTS + UI
  Feed to AI builder as Step 3.
  Suggested prompt: "Generate the complete project now.
  Work through the file list in order. Implement all files
  exactly as specified in Parts 1 and 2."
================================================================

## GENERATION ORDER

```
1.  package.json
2.  next.config.js                              (output: 'export' only)
3.  src/db/types.ts                             (interfaces only)
4.  src/db/index.ts                             ⚠ PLACEHOLDER ONLY — see note below
5.  src/lib/nodeLimits.ts                       (from Part 2)
6.  src/lib/aiAdapter.ts                        (from Part 2 — implement exactly)
7.  src/lib/contextBuilder.ts                   (from Part 2 — implement exactly)
8.  src/lib/summarizer.ts                       (from Part 2 — implement exactly)
9.  src/store/index.ts                          (from Part 2 — implement exactly)
10. src/app/layout.tsx
11. src/app/page.tsx
12. src/components/MainLayout.tsx
13. src/components/TopBar/index.tsx
14. src/components/TopBar/TopicSelector.tsx
15. src/components/TopBar/TopicDropdown.tsx
16. src/components/TopBar/TopicListItem.tsx
17. src/components/TopBar/NewTopicInput.tsx
18. src/components/LeftPanel/index.tsx
19. src/components/LeftPanel/PanelHeader.tsx
20. src/components/LeftPanel/ListView.tsx
21. src/components/LeftPanel/TreeNode.tsx
22. src/components/LeftPanel/MindMapView.tsx
23. src/components/LeftPanel/MindMapCanvas.tsx  (see MindMapCanvas spec below)
24. src/components/CenterPanel/index.tsx
25. src/components/CenterPanel/ConversationFeed.tsx
26. src/components/CenterPanel/MessageBubble.tsx
27. src/components/CenterPanel/EditForm.tsx
28. src/components/CenterPanel/BranchButton.tsx
29. src/components/CenterPanel/TruncationNotice.tsx
30. src/components/CenterPanel/MessageInput.tsx
31. src/components/SettingsDrawer/index.tsx
32. src/components/SettingsDrawer/APIKeyManager.tsx
```

⚠ NOTE on src/db/index.ts:
Generate a placeholder file with this content:
```typescript
// src/db/index.ts
// ⚠ This file will be replaced by the patch in Part 4.
// Do not implement. Export stubs so other files compile.
export async function getNode(id: string) { return undefined }
export async function getNodesByTopic(topicId: string) { return [] }
export async function getChildren(parentId: string) { return [] }
export async function getLatestSummaryNode(topicId: string, branchRootId: string) { return undefined }
export async function getAllTopics() { return [] }
export async function saveNode(node: any) {}
export async function updateNodeContent(id: string, content: string) {}
export async function upsertSummaryNode(node: any) {}
export async function deleteDescendants(nodeId: string) {}
export async function atomicEditSave(editedNodeId: string, newContent: string, newAiNode: any) {}
export async function saveTopic(topic: any) {}
export async function deleteTopic(topicId: string) {}
```

---

## package.json

```json
{
  "name": "thinktree",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.x",
    "react": "18.x",
    "react-dom": "18.x",
    "dexie": "^4.0.0",
    "zustand": "^4.5.0",
    "@anthropic-ai/sdk": "^0.24.0"
  },
  "devDependencies": {
    "typescript": "^5",
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "tailwindcss": "^3",
    "postcss": "^8",
    "autoprefixer": "^10",
    "eslint": "^8",
    "eslint-config-next": "14.x"
  }
}
```

---

## UI COMPONENT SPECS

### src/app/page.tsx
- On mount: call `loadTopicNodes` for active topic (localStorage key "thinktree_active_topic")
- If no active topic: call `getAllTopics()`, set first, or show empty state prompt
- Render: TopBar + MainLayout + SettingsDrawer

### src/components/MainLayout.tsx
- flex row filling height below TopBar
- Manages leftWidth state (localStorage "thinktree_left_width", default 240px)
- DragHandle: 4px div, onPointerDown/Move/Up, clamps 150–380px

### TopBar (48px fixed, full width)
Left side: TopicSelector + vertical divider (0.5px) + BranchPill
Right side: ExportButton + SettingsButton

### TopicSelector + TopicDropdown
- Trigger: [colored dot] [topic name ▾], max 200px, overflow ellipsis
- Dropdown (240px): "TOPICS" header + TopicListItem[] + "+ New topic" row
- TopicListItem (32px): dot + name + node count badge
  - badge: muted if < 200, red "200/200" at limit
  - hover: reveal ✎ rename + 🗑 delete buttons
  - rename: inline input, Enter commits, Escape cancels
  - delete: confirmation popover → calls `deleteTopic(id)`
- "+ New topic": input → `saveTopic()` + `saveNode()` (root node) + switch to new topic
- Color rotation: Purple #7F77DD → Teal #1D9E75 → Amber #EF9F27 → Coral #D85A30

### LeftPanel
- Width: leftWidth state (min 150, max 380, default 240)
- Header (32px): topic name (11px muted) + List/Map pill toggle
- List view (default): TreeNode recursive, summary nodes excluded
  - Indent: depth * 12px (cap at 3 levels)
  - Dot colors: root=purple, user_question=blue, ai_response=teal
  - Active: white bg + bold + 2px purple left border
  - Click: setActiveNodeId
  - Right-click/long-press: context menu ("Start new branch" disabled at NODE_LIMIT_MAX)
- Map view: renders MindMapCanvas (see MindMapCanvas spec below)

### MindMapCanvas spec
⚠ The v2.0 MindMapCanvas is SIMPLIFIED vs v9.3.
No pinch-zoom. No per-node drag. No coordinate system conversion.
Only: static auto-layout + whole-canvas pan + tap to navigate.

Props:
```typescript
interface Props {
  nodes:        Node[]          // all non-summary nodes for this topic
  activeNodeId: string | null
  pan:          { x: number; y: number }
  onPanChange:  (pan: { x: number; y: number }) => void
  onNodeTap:    (id: string) => void
}
```

Implementation rules:
1. Auto-layout: radial, root at center, children spread 70° arc,
   radius increases 90px per depth level. Pure calculation, no state.
2. SVG element MUST have: style={{ touchAction: 'none' }}
3. Pan: single pointer drag on any empty area moves the whole canvas
   using SVG `transform="translate(x, y)"` on a wrapper `<g>`.
4. Tap: if pointer moved < 5px from down to up, fire onNodeTap.
5. Pointer events only — no onMouseDown / onTouchStart.
6. Node circles: root=28px radius, others=18px radius.
7. Edges: bezier curves rendered BEFORE nodes in SVG order.
8. Active node: amber ring (3px stroke, no fill, radius+6).
9. summary nodes: filter out before rendering.
10. Pan persisted to localStorage key "thinktree_mindmap_{topic_id}".

```typescript
// Minimal implementation pattern — use this as the basis:

export function MindMapCanvas({ nodes, activeNodeId, pan, onPanChange, onNodeTap }: Props) {
  const isPanning = useRef(false)
  const panStart  = useRef({ x: 0, y: 0 })
  const panOrigin = useRef({ x: 0, y: 0 })
  const tapStart  = useRef({ x: 0, y: 0 })

  const visibleNodes = nodes.filter(n => n.role !== 'summary')
  const layout       = computeAutoLayout(visibleNodes)

  function onPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    e.currentTarget.setPointerCapture(e.pointerId)
    isPanning.current  = true
    panStart.current   = { x: e.clientX, y: e.clientY }
    panOrigin.current  = { ...pan }
    tapStart.current   = { x: e.clientX, y: e.clientY }
  }

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!isPanning.current) return
    onPanChange({
      x: panOrigin.current.x + (e.clientX - panStart.current.x),
      y: panOrigin.current.y + (e.clientY - panStart.current.y),
    })
  }

  function onPointerUp(e: React.PointerEvent<SVGSVGElement>) {
    if (!isPanning.current) return
    isPanning.current = false
    const dx = e.clientX - tapStart.current.x
    const dy = e.clientY - tapStart.current.y
    if (Math.hypot(dx, dy) < 5) {
      // tap — find nearest node to click position
      // convert click to canvas coords: canvas = (screen - pan) / 1
      // (no scale in v2.0, scale is always 1)
      const cx = e.clientX - pan.x - svgCenter.x
      const cy = e.clientY - pan.y - svgCenter.y
      for (const node of visibleNodes) {
        const pos = layout.get(node.id)
        if (!pos) continue
        if (Math.hypot(cx - pos.x, cy - pos.y) <= 22) {
          onNodeTap(node.id)
          break
        }
      }
    }
  }

  return (
    <svg
      style={{ touchAction: 'none', width: '100%', height: '100%' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <g transform={`translate(${pan.x + svgCenter.x}, ${pan.y + svgCenter.y})`}>
        {/* edges first */}
        {/* nodes second */}
      </g>
    </svg>
  )
}
```

### CenterPanel
Structure top to bottom:
1. ConversationFeed (flex:1, overflow-y:auto)
2. TruncationNotice (amber, shown after edit, has [Dismiss])
3. MessageInput (fixed bottom)

### ConversationFeed
- Derives messages from Zustand nodesState via getBranchPath equivalent
  (filter nodes by parent_id chain from activeNodeId to root, reverse)
- User bubbles: right-aligned, #EEEDFE bg
- AI bubbles: left-aligned, secondary bg, 0.5px border
- summary nodes: NEVER rendered
- Hover: timestamp + [Edit] (user only) + [+ Branch] (hidden at NODE_LIMIT_MAX)
- Scrolls to bottom on new message

### EditForm
- Textarea pre-filled, auto-focused
- [Save & regenerate]: calls `atomicEditSave` then Zustand mutations (see message flow)
- [Cancel]: restore bubble

### MessageInput
- Textarea 54px, Enter=submit, Shift+Enter=newline
- Disabled when: no API key, nodeCount >= NODE_LIMIT_MAX
- Send button disabled in same states
- Inline notice area for guard errors (api_key_missing, node_limit_reached)

### SettingsDrawer
- Slides in from right (translateX, 0.2s), width 220px, overlays CenterPanel
- Close: × / outside click / Escape
- Contents: ProviderSelector + ModelInput + ApiKeyInput (masked) +
  ClearKeyButton + ContextWindowNote + Export JSON + Export Markdown
- API key read/write: localStorage.getItem/setItem('thinktree_api_key_{provider}')
- Mask display: show last 4 chars, rest replaced with ••••

---

## CODE QUALITY RULES

TypeScript: strict mode, no `any` types except in classifyError
React: functional components, useCallback for handlers, useMemo for derived data
State: Zustand mutations ONLY after DB write resolves — never before
CSS: TailwindCSS utility classes only


================================================================
  PART 4 — DB PATCH
  After generation is complete, overwrite src/db/index.ts
  with the exact content below.
  This is the ONLY file that needs manual replacement.
================================================================

### FILE: src/db/index.ts

⚠ COPY THIS EXACTLY — do not paraphrase or restructure.
The Dexie transaction patterns here are precise. Incorrect transaction
usage causes silent data corruption that is very hard to debug.

```typescript
// src/db/index.ts
import Dexie, { type Table } from 'dexie'
import type { Node, Topic, Workspace } from './types'

export class ThinkTreeDB extends Dexie {
  workspaces!: Table<Workspace>
  topics!:     Table<Topic>
  nodes!:      Table<Node>

  constructor() {
    super('thinktree_db')
    this.version(1).stores({
      workspaces: 'id',
      topics:     'id, workspace_id',
      nodes:      'id, topic_id, parent_id, role, [topic_id+role]',
    })
  }
}

export const db = new ThinkTreeDB()

// ── Read ──────────────────────────────────────────────────────

export async function getNode(id: string): Promise<Node | undefined> {
  return db.nodes.get(id)
}

export async function getNodesByTopic(topicId: string): Promise<Node[]> {
  return db.nodes.where('topic_id').equals(topicId).toArray()
}

export async function getChildren(parentId: string): Promise<Node[]> {
  return db.nodes
    .where('parent_id').equals(parentId)
    .filter(n => n.role !== 'summary')
    .toArray()
}

export async function getLatestSummaryNode(
  topicId: string,
  branchRootId: string
): Promise<Node | undefined> {
  return db.nodes
    .where('[topic_id+role]')
    .equals([topicId, 'summary'])
    .and(n => n.parent_id === branchRootId)
    .first()
}

export async function getAllTopics(): Promise<Topic[]> {
  return db.topics.toArray()
}

// ── Write ─────────────────────────────────────────────────────

export async function saveNode(node: Node): Promise<void> {
  await db.nodes.put(node)
}

export async function updateNodeContent(
  id: string,
  content: string
): Promise<void> {
  await db.nodes.update(id, { content })
}

export async function upsertSummaryNode(node: Node): Promise<void> {
  // Atomic: delete old summary for this branch + insert new one.
  await db.transaction('rw', db.nodes, async () => {
    await db.nodes
      .where('[topic_id+role]')
      .equals([node.topic_id, 'summary'])
      .and(n => n.parent_id === node.parent_id)
      .delete()
    await db.nodes.put(node)
  })
}

// ── CRITICAL: _recursiveDelete must only run inside a transaction ──
// NEVER call this function directly outside db.transaction().
// NEVER add non-Dexie operations (fetch, setTimeout, Zustand) here.

async function _recursiveDelete(nodeId: string): Promise<void> {
  const children = await db.nodes
    .where('parent_id').equals(nodeId)
    .toArray()
  for (const child of children) {
    await _recursiveDelete(child.id)
  }
  await db.nodes.where('parent_id').equals(nodeId).delete()
}

export async function deleteDescendants(nodeId: string): Promise<void> {
  await db.transaction('rw', db.nodes, async () => {
    await _recursiveDelete(nodeId)
  })
}

// ── atomicEditSave: all 3 steps succeed or all roll back ──────

export async function atomicEditSave(
  editedNodeId: string,
  newContent:   string,
  newAiNode:    Node,
): Promise<void> {
  await db.transaction('rw', db.nodes, async () => {
    await db.nodes.update(editedNodeId, { content: newContent })
    await _recursiveDelete(editedNodeId)
    await db.nodes.put(newAiNode)
  })
}

// ── Topic operations ──────────────────────────────────────────

export async function saveTopic(topic: Topic): Promise<void> {
  await db.topics.put(topic)
}

export async function deleteTopic(topicId: string): Promise<void> {
  await db.transaction('rw', db.topics, db.nodes, async () => {
    await db.nodes.where('topic_id').equals(topicId).delete()
    await db.topics.delete(topicId)
  })
}
```


================================================================
  PART 5 — VERIFY CHECKLIST
  Run after patching src/db/index.ts.
================================================================

----------------------------------------------------------------
  SECTION 1 — DB LAYER  (src/db/index.ts)
----------------------------------------------------------------

□ `_recursiveDelete` is NOT exported
□ `_recursiveDelete` is ONLY called inside `db.transaction()` blocks
□ `atomicEditSave` does NOT contain any Zustand mutations
□ No `new ThinkTreeDB()` anywhere except this file's module scope
□ `[topic_id+role]` compound index present in version(1).stores()
□ No native Promises or setTimeout inside db layer — Dexie only

----------------------------------------------------------------
  SECTION 2 — SUMMARIZER  (src/lib/summarizer.ts)
----------------------------------------------------------------

□ `activeSummaryTopicId` is module-level — NOT inside any function or component
□ `triggerSummaryIfNeeded` return type is `void` — never awaited at call site
□ Async work is inside IIFE `(async () => { ... })()`
□ `capturedTopicId` is captured BEFORE the first `await`
□ Guard present: `activeSummaryTopicId !== capturedTopicId` → return
□ `upsertSummaryNode` called ONLY after guard passes
□ Entire async block wrapped in try/catch → console.warn only, never throws
□ `setActiveTopic(topicId)` called inside `loadTopicNodes` in store
□ `clearActiveTopic()` called inside `loadTopicNodes` BEFORE `setActiveTopic`

----------------------------------------------------------------
  SECTION 3 — STORE  (src/store/index.ts)
----------------------------------------------------------------

□ `addNode` called only AFTER `saveNode()` resolves
□ `updateNodeContent` called only AFTER `db.nodes.update()` resolves
□ `removeDescendants` called only AFTER `atomicEditSave()` resolves
□ `removeDescendants` uses BFS — NOT simple filter(n.parent_id !== nodeId)
□ `loadTopicNodes` calls `clearActiveTopic()` then `setActiveTopic()` before DB read
□ No optimistic updates — UI state only changes after DB confirms

----------------------------------------------------------------
  SECTION 4 — AI ADAPTER  (src/lib/aiAdapter.ts)
----------------------------------------------------------------

□ Anthropic: system message extracted and passed as TOP-LEVEL `system:` param
□ Anthropic: `dangerouslyAllowBrowser: true` in Anthropic constructor
□ Anthropic: model is `claude-sonnet-4-5`
□ Anthropic: response read with `.find(b => b.type === 'text').text`
     NOT `.content[0].text`
□ OpenAI/DeepSeek: system message stays inside messages[] array
□ No API keys in console.log anywhere

----------------------------------------------------------------
  SECTION 5 — CONTEXT BUILDER  (src/lib/contextBuilder.ts)
----------------------------------------------------------------

□ `getBranchPath` traverses parent_id chain (root → nodeId)
□ `getBranchPath` filters out summary nodes
□ Sibling branch nodes are NEVER included in context
□ Stale summary guard present: `summaryNode.covers_up_to < recentStartIndex`
□ Summary injected as user/assistant PAIR — not as a system message
□ Token budget trim loop present, minimum window = 2 messages

----------------------------------------------------------------
  SECTION 6 — MINDMAP  (src/components/LeftPanel/MindMapCanvas.tsx)
----------------------------------------------------------------

□ SVG element has `style={{ touchAction: 'none' }}`
□ Uses `onPointerDown/onPointerMove/onPointerUp` — NOT onMouseDown/onTouchStart
□ Pan moves entire canvas via translate — NOT individual node positions
□ Tap detected by movement threshold < 5px — NOT onClick
□ No pinch-zoom logic anywhere in this file
□ No per-node drag logic anywhere in this file
□ summary nodes filtered out before rendering
□ Bezier edges rendered BEFORE node circles in SVG source order
□ Pan state persisted to localStorage "thinktree_mindmap_{topic_id}"

----------------------------------------------------------------
  SECTION 7 — MESSAGE FLOW
----------------------------------------------------------------

□ Guard 1 (API key) checked before any node is created
□ Guard 2 (node limit) checked before any node is created
□ `triggerSummaryIfNeeded` called fire-and-forget (no await)
□ `triggerSummaryIfNeeded` only triggered when path.length % 10 === 0
□ Zustand `addNode` called only AFTER `saveNode` resolves
□ `setActiveNodeId` set to new AI node after response arrives
□ Edit flow: `atomicEditSave` resolves → THEN Zustand mutations run

----------------------------------------------------------------
  SECTION 8 — FINAL SEARCH CHECKS
  Run in your editor or terminal after generation + patching:
----------------------------------------------------------------

□ `grep -r "new ThinkTreeDB" src/` → only ONE result (in src/db/index.ts)
□ `grep -r "_recursiveDelete" src/` → only results INSIDE src/db/index.ts
□ `grep -r "triggerSummaryIfNeeded" src/` → call site has NO `await` before it
□ `grep -r "activeSummaryTopicId" src/` → only results in src/lib/summarizer.ts
□ `grep -r "onMouseDown\|onTouchStart" src/components/LeftPanel/MindMap` → ZERO results
□ `grep -r "content\[0\]\.text" src/lib/aiAdapter` → ZERO results
□ `next build` completes with zero TypeScript errors

================================================================
  END OF ThinkTree v2.0 MVP PROMPT
================================================================
