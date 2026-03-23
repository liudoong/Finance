# ThinkTree v9.4 — Complete Prompt (7 Parts)
# =================================================================
# VERSION HISTORY:
#   v9.2 → v9.3: Restructured into two-layer workflow
#                All content preserved — reorganized not rewritten
#   v9.3 → v9.4: Five-pit prevention applied
#     Pit 1: function signatures → import statements in feed layer
#            (AI cannot implement what it only sees as imports)
#     Pit 2: path confirmation table added as Step 4.5
#            (fill before executing Step 5 patch)
#     Pit 3: handleSend + handleSave added as copy-paste snippets
#            (critical DB-first ordering locked in Part 3a)
#     Pit 4a: APIKeyManager.tsx moved into patch layer (Part 4b)
#            (prevents direct sessionStorage access in components)
#     Pit 4b: security grep promoted to Step 5 immediate check
#            (catch key leaks before running full checklist)
#     Pit 5: already solved by contextBuilder patch (no change)
#
# TWO-LAYER WORKFLOW:
#   FEED LAYER   (Parts 1, 2a, 3a, 4a) → paste into AI builder in order
#   PATCH LAYER  (Parts 2b, 3b, 4b)    → overwrite AI-generated files directly
#   VERIFY LAYER (Part Checklist)       → run after patching
#
# WHY TWO LAYERS:
#   AI builders generate wrong code for 7 specific files:
#   db/index.ts, keyManager.ts, store/index.ts, aiAdapter.ts,
#   contextBuilder.ts, summarizer.ts, MindMapCanvas.tsx
#   The patch layer bypasses AI generation entirely for these files.
#   The feed layer tells AI builder what to CALL — not how to implement.
#
# PART INDEX:
#   Part 1        — Product Spec          (feed to AI builder, Step 1)
#   Part 2a       — Architecture: Intent  (feed to AI builder, Step 2)
#   Part 3a       — Components: Intent    (feed to AI builder, Step 3)
#   Part 4a       — UI Generation         (feed to AI builder, Step 4)
#   Part 2b       — DB + Security Patch   (overwrite files, Step 5)
#   Part 3b       — Core Logic Patch      (overwrite files, Step 5)
#   Part 4b       — Canvas + API Patch    (overwrite files, Step 5)
#   Part Checklist— Master Checklist      (verify after patching, Step 6)
#
# PATCH FILES (Part 2b + 3b + 4b overwrite these 9 files):
#   src/db/index.ts
#   src/lib/keyManager.ts
#   next.config.js + public/_headers
#   src/store/index.ts
#   src/lib/summarizer.ts
#   src/lib/contextBuilder.ts
#   src/lib/aiAdapter.ts
#   src/components/LeftPanel/MindMapCanvas.tsx
#   src/components/SettingsDrawer/APIKeyManager.tsx  ← new in v9.4
# =================================================================


================================================================
  PART 1 — PRODUCT SPEC
  Feed to AI builder as Step 1.
  Suggested prompt: "Read this product spec carefully.
  Do not generate any code yet. Confirm you understand
  the product, UI regions, and node limit rules."
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
- UI preferences (panel width, active topic, map positions) stored in localStorage
- API keys stored in sessionStorage by default; localStorage only on explicit user opt-in
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
- AI context must only include nodes along the current branch path
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

## NODE LIMITS PER TOPIC

To maintain performance and encourage focused thinking, each topic
tree has the following hard limits:

```
NODE_LIMIT_MAX     = 200   // hard ceiling — no new nodes created above this
NODE_LIMIT_WARNING = 160   // soft threshold — warning banner appears
BRANCH_DEPTH_MAX   = 50    // max depth of any single branch path
```

### What counts toward the limit:
- user_question nodes:  YES — counts toward NODE_LIMIT_MAX
- ai_response nodes:    YES — counts toward NODE_LIMIT_MAX
- summary nodes:        NO  — system-generated, excluded from user quota

### Behavior at NODE_LIMIT_WARNING (160 non-summary nodes):
- All functionality continues normally
- Persistent amber NodeLimitBanner appears above MessageInput:
  "This topic is getting large (160/200 nodes).
   Consider starting a new topic for a fresh direction."
  [New topic] button → opens Topic Dropdown to create a new topic
  [Dismiss] button → hides banner for this session only
  (sessionStorage key: "thinktree_limit_dismissed_{topic_id}")

### Behavior at NODE_LIMIT_MAX (200 non-summary nodes):
- MessageInput textarea: DISABLED
- Send button: DISABLED (grayed out)
- "+ Branch" button on all message bubbles: HIDDEN
- "Start new branch" in context menus: DISABLED
- Red NodeLimitBanner replaces the amber one:
  "This topic has reached the 200-node limit.
   Start a new topic to continue exploring."
  [New topic] button → opens Topic Dropdown
  This banner CANNOT be dismissed — it persists until the user
  deletes nodes or switches to a different topic.

### Behavior at BRANCH_DEPTH_MAX (50 nodes deep on current path):
- MessageInput + Send button: DISABLED for this branch only
- Other branches in the same topic remain fully functional
- "+ Branch" buttons on earlier messages in the conversation remain active
- A red BranchDepthBanner appears above MessageInput:
  "This branch is 50 nodes deep — the maximum depth.
   Start a new branch from an earlier message to continue."

### What limits do NOT affect:
- Editing existing nodes (Edit is always available)
- Deleting nodes / topics
- Exporting
- Switching to a different topic or branch

### Node count display in Topic Dropdown:
Each topic row shows its non-summary node count:
- count < 160:    muted text, no background  e.g. "12"
- count 160–199:  amber text                 e.g. "174"  (warning)
- count = 200:    red text, "/ 200" suffix   e.g. "200 / 200"

---

## UI LAYOUT OVERVIEW

```
┌──────────────────────────────────────────────────────────┐
│  TopBar (48px fixed)                                     │
│  [Topic ▾] | [Branch pill]    [spacer]  [⬇ Export] [⚙] │
├──────────────┬───────────────────────────────────────────┤
│  Left Panel  ║  Center Panel                             │
│  (resizable) ║  Conversation area (scrollable)           │
│  240px       ║  [NodeLimitBanner   — amber/red, optional]│
│  default     ║  [BranchDepthBanner — red,       optional]│
│              ║  [TruncationNotice  — amber,      optional]│
│              ║  Message Input (fixed at bottom)          │
└──────────────╨───────────────────────────────────────────┘
                     Settings Drawer overlays from right
                     edge on demand (z-index above center)
```

Banner stacking order (top to bottom, only shown when relevant):
  NodeLimitBanner → BranchDepthBanner → TruncationNotice → MessageInput

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
  Node count badge color rules (see NODE LIMITS section above)
- On hover: reveal [✎ rename] [🗑 delete] buttons

Rename: inline edit → Enter/blur commits → Escape cancels
Delete: confirmation popover "Delete '[name]'? This cannot be undone."
  → auto-switch to nearest remaining topic if deleted was active
  → empty state if no topics remain

"＋ New topic":
- Shows inline input → Enter/Add creates topic with empty root node
- Auto-assigns color from 4-color rotation (see below)
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

NO topic management in Left Panel.
Topic management lives entirely in the TopBar dropdown.
Full Left Panel height is dedicated to the tree navigator.

### Left Panel Header (32px):
- Current topic name (11px, muted, truncated)
- List / Map view toggle (pill-style, right-aligned)

### LIST VIEW (default):
Hierarchical tree navigator as indented list.

Node rendering:
- Indent: 12px per depth level (capped visually at 3 levels)
- Dot colors: root=purple, user_question=blue, ai_response=teal, branch=amber
- Active node: white bg + bold + 2px purple left border
- Edited node (truncatedBelow=true): small amber "edited" badge
- summary nodes: NEVER rendered in list view

Interactions:
- Click → navigate to that branch (set activeNodeId)
- Right-click / long-press → context menu: "Start new branch" / "Rename"
  "Start new branch" is DISABLED when topic is at NODE_LIMIT_MAX

Section labels (10px uppercase muted): "ROOT", "BRANCH A", etc.

### MAP VIEW:
SVG-based interactive mind map. Same tree data, visual layout.

Node rendering:
- root → larger circle, purple fill (#EEEDFE, stroke #7F77DD)
- user_question → medium circle, blue fill
- ai_response → medium circle, teal fill
- branch → medium circle, amber fill
- active → 2.5px stroke, amber highlight ring
- edited → amber dashed stroke
- summary → NOT rendered

Layout: radial auto-layout
- Root at canvas center-top
- Children spread in 70° arc around parent
- Radius increases by 90px per depth level
- Manually dragged nodes override auto-layout (persisted)

Interactions (unified via Pointer Events API):
- 1 pointer on empty canvas → pan
- 1 pointer on node circle → drag node to reposition
- 2 pointers simultaneously → pinch-zoom (scale clamped 0.3–3.0)
- Tap (< 5px movement) → navigate to that branch
- Long-press → context menu
  "Start new branch" DISABLED in context menu when at NODE_LIMIT_MAX

SVG element must have: style="touch-action: none"
Minimum tap target for nodes: 44×44px

Persistence: localStorage key "thinktree_mindmap_{topic_id}"
Stores: { viewTransform: {x, y, scale}, nodePositions: Map<nodeId, {x,y}> }

---

## CENTER PANEL

Fills remaining width (flex: 1, min-width: 200px).

Structure top to bottom:
1. ConversationFeed (flex: 1, overflow-y: auto)
2. NodeLimitBanner (amber/red, shown when nodeCount >= NODE_LIMIT_WARNING)
3. BranchDepthBanner (red, shown when branch depth >= BRANCH_DEPTH_MAX)
4. TruncationNotice (amber, shown after edit-truncation)
5. MessageInput (fixed height at bottom)

### CONVERSATION FEED:
Displays nodes along active branch path (root → activeNode), chronological.

User messages: right-aligned, purple bubble (#EEEDFE bg, dark purple text)
AI messages: left-aligned, secondary bg bubble with 0.5px border
summary nodes: NEVER rendered
Only ONE node may be in edit mode at a time

On hover over any message: show timestamp · [Edit] · [+ Branch]
  [+ Branch] is HIDDEN (not disabled) when topic is at NODE_LIMIT_MAX
On hover over AI messages: show "AI" label · [+ Branch]

### NODE LIMIT BANNER (NodeLimitBanner):
Warning state (160 ≤ nodeCount < 200):
  Amber background, 0.5px top amber border, 11px text, flex row.
  "This topic is getting large ({nodeCount}/200 nodes).
   Consider starting a new topic for a fresh direction."
  [New topic]  [Dismiss]

Hard limit state (nodeCount ≥ 200):
  Red background (#FCEBEB), 0.5px top red border (#F09595), 11px text.
  "This topic has reached the 200-node limit.
   Start a new topic to continue exploring."
  [New topic]  — no Dismiss button

### BRANCH DEPTH BANNER (BranchDepthBanner):
Shown only when current branch depth ≥ BRANCH_DEPTH_MAX (50).
Red background (#FCEBEB), 0.5px top red border, 11px text.
"This branch is 50 nodes deep — the maximum depth.
 Start a new branch from an earlier message to continue."
No dismiss. Hidden when user navigates to a shallower branch.

### MESSAGE EDIT — MID-CONVERSATION TRUNCATION:
Trigger: Edit button (fade in on hover) on any user_question bubble.
Edit is ALWAYS available regardless of node limit
(editing does not add nodes, it typically reduces node count).

On clicking Edit:
- Bubble replaced by: textarea (pre-filled) + [Save & regenerate] + [Cancel]

On "Save & regenerate":
1. Update node content in DB
2. Delete ALL descendant nodes (recursive, atomic)
3. Create ONE new ai_response placeholder node as child
   (net node count decreases — no limit check needed)
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
Dismiss: hides bar, clears truncatedBelow flags, removes "edited" badges.
Also hides on topic switch.

### MESSAGE INPUT:
- Textarea (height 54px, vertically resizable by user)
- Send button (purple, right-aligned)
- Enter = submit; Shift+Enter = newline
- Disabled states (textarea + Send button both disabled):
  a) No valid API key configured
  b) Topic has reached NODE_LIMIT_MAX (200 non-summary nodes)
  c) Current branch depth ≥ BRANCH_DEPTH_MAX (50 nodes deep)

---

## SETTINGS DRAWER

Trigger: ⚙ button in TopBar.
Animation: slides in from right edge (CSS translateX, 0.2s ease).
Width: 220px. Overlays Center Panel — does NOT push or resize it.
Close: × button inside / click outside / press Escape.

Contents:
- AI provider selector (OpenAI / DeepSeek / Anthropic)
- Model name (editable, default per provider)
- API key input (masked, show/hide toggle)
- "Remember API key" checkbox
- Warning text when "Remember" is checked
- "Clear key" button
- Context window (read-only: "Last 12 msgs + auto-summary")
- Storage status (read-only, live: "session only" or "session + local")
- Export JSON button
- Export Markdown button

---

## EXPORT / IMPORT

Export dialog (opened from TopBar export button).
Choose scope:
- Current branch path only → Markdown (linear, root → active node)
- Entire topic tree → JSON or Markdown

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
- summary nodes INCLUDED in JSON (enables full restore)
- Import: upsert by id, do not overwrite existing nodes with same id
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
3. Controlled truncation — editing mid-conversation is allowed and
   clearly communicated; the tree always reflects the current state
4. Instant sync — the tree navigator (List + Map) always reflects
   the current conversation state with zero delay
5. Bring your own AI — works with any supported provider,
   no account required beyond an API key
6. Reliable persistence — all writes go through Dexie transactions;
   the UI only updates after the DB confirms the write succeeded
7. Secure by default — API keys live in sessionStorage and vanish
   on tab close; CSP blocks exfiltration even if XSS occurs
8. Focused exploration — each topic tree is capped at 200 nodes
   to encourage clear thinking and topic separation

----------------------------------------------------------------

================================================================
  PART 2a — ARCHITECTURE: INTENT
  Feed to AI builder as Step 2.
  Suggested prompt: "This is the technical architecture.
  Understand the tech stack, data model, and function signatures.
  Do not implement any of the listed functions — they will be
  provided as pre-built files. Only call them as specified."
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

## AI PROVIDER MODEL

Three supported providers:
- OpenAI    → api.openai.com
- DeepSeek  → api.deepseek.com
- Anthropic → api.anthropic.com (uses @anthropic-ai/sdk)

API keys: sessionStorage by default, localStorage on opt-in.
All key access goes through `src/lib/keyManager.ts` — never direct storage calls.

---

## FUNCTION SIGNATURES — PRE-BUILT FILES

⚠ DO NOT IMPLEMENT these functions. They are provided as patch files
(Parts 2b, 3b, 4b). Your job is only to CALL them correctly.

The correct way to use pre-built functions is to import them:

```typescript
// DB operations — import from '@/db'
import {
  saveNode, updateNodeContent, upsertSummaryNode,
  deleteDescendants, atomicEditSave,
  getNode, getNodesByTopic, getChildren,
  getLatestSummaryNode, getAllTopics,
  saveTopic, deleteTopic,
} from '@/db'

// API key management — import from '@/lib/keyManager'
import {
  loadApiKey, saveApiKey, clearApiKey, maskApiKey,
  loadRememberPref, saveRememberPref, clearRememberPref,
} from '@/lib/keyManager'

// Zustand store — import from '@/store'
import { useAppStore } from '@/store'
// Destructure what you need:
// const { nodes, activeNodeId, addNode, updateNodeContent,
//         removeDescendants, setActiveNodeId, loadTopicNodes } = useAppStore()

// AI adapter — import from '@/lib/aiAdapter'
import { sendMessage } from '@/lib/aiAdapter'
import type { Message } from '@/lib/aiAdapter'

// Context builder — import from '@/lib/contextBuilder'
import { buildContext, getBranchPath } from '@/lib/contextBuilder'

// Summarizer — import from '@/lib/summarizer'
import {
  triggerSummaryIfNeeded,
  invalidateSummaryGeneration,
  clearAllSummaryGenerations,
} from '@/lib/summarizer'

// Node limits — import from '@/lib/nodeLimits'
import {
  NODE_LIMIT_MAX, NODE_LIMIT_WARNING, BRANCH_DEPTH_MAX,
  getNodeCount, canAddNode, canExtendBranch, getLimitState,
} from '@/lib/nodeLimits'
```

These files already exist in the project (provided as patch files).
Do not create implementations for any of the above functions.
If a file is missing at generation time, create an empty placeholder:
```typescript
// placeholder — will be replaced by patch
export {}
```

---

## SECURITY CONSTRAINTS

- API keys: never in exported files, never in console.log
- CSP headers configured in next.config.js (pre-built — do not modify)
- All storage access for API keys: use keyManager functions only

---

## DEPLOYMENT

Static export — no server required.
- Vercel: automatic from next.config.js
- Cloudflare Pages: public/_headers file (pre-built)


================================================================
  PART 3a — COMPONENTS: INTENT
  Feed to AI builder as Step 3.
  Suggested prompt: "This is the component spec. Understand the
  component tree, message flow, and error handling. Call the
  pre-built functions as specified — do not reimplement them."
================================================================

## NODE LIMIT CONSTANTS

```typescript
// src/lib/nodeLimits.ts — implement this file in full (simple, no risk)
export const NODE_LIMIT_MAX     = 200
export const NODE_LIMIT_WARNING = 160
export const BRANCH_DEPTH_MAX   = 50

export function getNodeCount(nodes: Node[], topicId: string): number {
  return nodes.filter(n => n.topic_id === topicId && n.role !== 'summary').length
}
export function canAddNode(nodes: Node[], topicId: string): boolean {
  return getNodeCount(nodes, topicId) < NODE_LIMIT_MAX
}
export function canExtendBranch(branchPath: Node[]): boolean {
  return branchPath.filter(n => n.role !== 'summary').length < BRANCH_DEPTH_MAX
}
export type LimitState = 'ok' | 'warning' | 'hard_limit'
export function getLimitState(nodes: Node[], topicId: string): LimitState {
  const count = getNodeCount(nodes, topicId)
  if (count >= NODE_LIMIT_MAX)     return 'hard_limit'
  if (count >= NODE_LIMIT_WARNING) return 'warning'
  return 'ok'
}
```

---

## REACT COMPONENT TREE

```
App
├── TopBar
│   ├── TopicSelector
│   │   ├── TopicDropdown
│   │   │   ├── TopicListItem[]    ← node count badge with limit-state color
│   │   │   └── NewTopicInput
│   ├── BranchPill
│   ├── ExportButton
│   └── SettingsButton
│
├── MainLayout
│   ├── LeftPanel
│   │   ├── PanelHeader
│   │   ├── ListView
│   │   │   └── TreeNode           ← "Start new branch" disabled at NODE_LIMIT_MAX
│   │   └── MindMapView
│   │       └── MindMapCanvas      ← ⚠ pre-built — do not generate (see Part 4b)
│   │
│   ├── DragHandle
│   │
│   └── CenterPanel
│       ├── ConversationFeed
│       │   └── MessageBubble[]
│       │       ├── BubbleContent
│       │       ├── EditForm       ← calls invalidateSummaryGeneration + atomicEditSave
│       │       └── BranchButton   ← hidden at NODE_LIMIT_MAX
│       ├── NodeLimitBanner
│       ├── BranchDepthBanner
│       ├── TruncationNotice
│       └── MessageInput
│
└── SettingsDrawer
    └── APIKeyManager              ← uses keyManager functions only
```

---

## TREE ALGORITHM

```typescript
// Implement getBranchPath and getChildrenSync in src/lib/contextBuilder.ts
// ⚠ getBranchPath is pre-built in Part 3b — do not reimplement

getChildrenSync(nodeId, allNodes) → Node[]
  // Sync version for UI rendering (nodesState already in memory)
  return allNodes.filter(n =>
    n.parent_id === nodeId && n.role !== 'summary'
  )
```

---

## MESSAGE FLOW

### Normal message send:

```
Guards (check in order, fail fast):

  Guard 1 — API key:
    apiKey = loadApiKey(provider)      // from keyManager
    if (!apiKey) → show api_key_missing, return early

  Guard 2 — Topic node limit:
    if (getNodeCount(nodesState, topic_id) >= NODE_LIMIT_MAX - 1)
      show node_limit_reached, return early

  Guard 3 — Branch depth:
    path = await getBranchPath(activeNodeId)   // from contextBuilder
    if (!canExtendBranch(path)) → show branch_depth_reached, return early

Main flow:
  1.  userNode = { id: uuid(), topic_id, parent_id: activeNodeId,
                   role: 'user_question', content, timestamp: Date.now(),
                   covers_up_to: null }
  2.  await saveNode(userNode)              // from db/index.ts
  3.  addNode(userNode)                     // Zustand — AFTER saveNode resolves
  4.  path = await getBranchPath(userNode.id)
  5.  if (path.length >= 10 && path.length % 10 === 0):
        triggerSummaryIfNeeded(path, provider, apiKey, topicId)
        // ⚠ fire-and-forget — do NOT await
        // ⚠ do NOT implement this function — it is pre-built in Part 3b
  6.  context = await buildContext(userNode.id, topicId)
        // ⚠ do NOT implement buildContext — pre-built in Part 3b
  7.  reply = await sendMessage(provider, apiKey, context)
        // ⚠ do NOT implement sendMessage — pre-built in Part 4b
  8.  aiNode = { id: uuid(), topic_id, parent_id: userNode.id,
                 role: 'ai_response', content: reply,
                 timestamp: Date.now(), covers_up_to: null }
  9.  await saveNode(aiNode)
  10. addNode(aiNode)                       // Zustand — AFTER saveNode resolves
  11. setActiveNodeId(aiNode.id)
```

### ⚠ COPY-PASTE SNIPPET — MessageInput send handler
Copy this into MessageInput.tsx exactly. The DB-first ordering is critical.
Do not rewrite or restructure this function.

```typescript
// MessageInput.tsx — handleSend
// Copy this exactly. DB writes must complete before Zustand mutations.
const handleSend = async () => {
  if (!content.trim()) return

  // Guard 1: API key
  const apiKey = loadApiKey(provider)
  if (!apiKey) { setError('api_key_missing'); return }

  // Guard 2: node limit (reserve 1 slot for AI response node)
  if (getNodeCount(nodes, topicId) >= NODE_LIMIT_MAX - 1) {
    setError('node_limit_reached'); return
  }

  // Guard 3: branch depth
  const currentPath = await getBranchPath(activeNodeId ?? '')
  if (!canExtendBranch(currentPath)) { setError('branch_depth_reached'); return }

  const userNode: Node = {
    id: crypto.randomUUID(),
    topic_id: topicId,
    parent_id: activeNodeId,
    role: 'user_question',
    content: content.trim(),
    timestamp: Date.now(),
    covers_up_to: null,
  }

  // DB FIRST — then Zustand. Never swap these two lines.
  await saveNode(userNode)
  addNode(userNode)

  setContent('')

  const path = await getBranchPath(userNode.id)
  if (path.length >= 10 && path.length % 10 === 0) {
    triggerSummaryIfNeeded(path, provider, apiKey, topicId) // fire-and-forget
  }

  const context = await buildContext(userNode.id, topicId)
  const reply   = await sendMessage(provider, apiKey, context)

  const aiNode: Node = {
    id: crypto.randomUUID(),
    topic_id: topicId,
    parent_id: userNode.id,
    role: 'ai_response',
    content: reply,
    timestamp: Date.now(),
    covers_up_to: null,
  }

  // DB FIRST — then Zustand.
  await saveNode(aiNode)
  addNode(aiNode)
  setActiveNodeId(aiNode.id)
}
```

### Edit message send:

```
  1.  branchRootId = (await getBranchPath(editedNodeId))[0].id
  2.  invalidateSummaryGeneration(topicId, branchRootId)
        // ⚠ MUST be called BEFORE atomicEditSave
  3.  await atomicEditSave(editedNodeId, newContent, newAiPlaceholder)
  4.  updateNodeContent(editedNodeId, newContent)   // Zustand — after DB
  5.  removeDescendants(editedNodeId)               // Zustand — after DB
  6.  addNode(newAiPlaceholder)                     // Zustand — after DB
  7.  setActiveNodeId(newAiPlaceholder.id)
  8.  show TruncationNotice bar
  9.  reply = await sendMessage(provider, apiKey, await buildContext(editedNodeId, topicId))
  10. await updateNodeContent(newAiPlaceholder.id, reply)   // DB first
  11. updateNodeContent(newAiPlaceholder.id, reply)         // Zustand after DB
```

### ⚠ COPY-PASTE SNIPPET — EditForm save handler
Copy this into EditForm.tsx exactly. Order is critical: invalidate → DB → Zustand.

```typescript
// EditForm.tsx — handleSave
// Copy this exactly. invalidateSummaryGeneration must come before atomicEditSave.
// All Zustand mutations must come after atomicEditSave resolves.
const handleSave = async () => {
  setIsLoading(true)
  try {
    const path        = await getBranchPath(editedNodeId)
    const branchRootId = path[0]?.id ?? editedNodeId

    // Invalidate in-flight summaries BEFORE the DB transaction
    invalidateSummaryGeneration(topicId, branchRootId)

    const newAiPlaceholder: Node = {
      id: crypto.randomUUID(),
      topic_id: topicId,
      parent_id: editedNodeId,
      role: 'ai_response',
      content: '',
      timestamp: Date.now(),
      covers_up_to: null,
    }

    // DB transaction first — atomically: update + delete descendants + insert placeholder
    await atomicEditSave(editedNodeId, newContent, newAiPlaceholder)

    // Zustand mutations only after DB resolves successfully
    updateNodeContent(editedNodeId, newContent)
    removeDescendants(editedNodeId)
    addNode(newAiPlaceholder)
    setActiveNodeId(newAiPlaceholder.id)
    onTruncationOccurred()

    // Fetch AI response
    const apiKey  = loadApiKey(provider)
    const context = await buildContext(editedNodeId, topicId)
    const reply   = await sendMessage(provider, apiKey, context)

    // DB first, then Zustand
    await updateNodeContent(newAiPlaceholder.id, reply)
    updateNodeContent(newAiPlaceholder.id, reply)  // Zustand store update

  } catch (err) {
    setError('edit_save_failure')
  } finally {
    setIsLoading(false)
  }
}
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

| Error type              | Behavior                                                        |
|-------------------------|-----------------------------------------------------------------|
| api_key_missing         | Amber inline notice in MessageInput. No node created.           |
| node_limit_reached      | Amber inline notice. No node created. NodeLimitBanner visible.  |
| branch_depth_reached    | Amber inline notice. No node created. BranchDepthBanner visible.|
| invalid_api_key         | Inline error, link to Settings                                  |
| network_failure         | Retry button on failed AI bubble, node NOT permanently saved    |
| rate_limit              | Amber notice with estimated wait time                           |
| summary_failure         | Silent — console.warn only, never blocks main message           |
| edit_save_failure       | Error in EditForm, keep edit open, do NOT truncate              |
| db_transaction_failure  | Toast: "Save failed — your data is unchanged."                  |
|                         | Skip Zustand mutation after DB failure                          |

---

## LEFT PANEL SYNC

Both ListView and MindMapView read from the same Zustand nodesState.
A single store mutation re-renders both panels simultaneously.
No manual refresh needed.

---

## SETTINGS DRAWER — APIKeyManager

```typescript
// On open:
const key      = loadApiKey(provider)         // keyManager
const remember = loadRememberPref(provider)   // keyManager

// On save:
saveApiKey(provider, newKey, rememberChecked)
saveRememberPref(provider, rememberChecked)

// On clear:
clearApiKey(provider)
clearRememberPref(provider)

// StorageStatusDisplay:
// not remembered → "API key: session only"
// remembered     → "API key: session + local (remembered)"
```


================================================================
  PART 4a — UI GENERATION
  Feed to AI builder as Step 4.
  Suggested prompt: "Generate the complete project now.
  Work through the file list in order. For files marked
  ⚠ PRE-BUILT, create an empty placeholder that only exports
  the module path — do not write any implementation code."

  ─────────────────────────────────────────────────────────
  STEP 4.5 — PATH CONFIRMATION (do before Step 5)
  After AI builder generates the project, fill in this table.
  Use actual generated paths in the right column.
  Step 5 patches must target the ACTUAL paths, not expected.
  ─────────────────────────────────────────────────────────

  PATH CONFIRMATION TABLE (fill after Step 4):

  Expected path (v9.4)                      │ Actual path in project
  ──────────────────────────────────────────┼──────────────────────────────────
  src/db/index.ts                           │ ____________________________
  src/lib/keyManager.ts                     │ ____________________________
  src/lib/aiAdapter.ts                      │ ____________________________
  src/lib/contextBuilder.ts                 │ ____________________________
  src/lib/summarizer.ts                     │ ____________________________
  src/lib/nodeLimits.ts                     │ ____________________________
  src/store/index.ts                        │ ____________________________
  src/components/LeftPanel/MindMapCanvas.tsx│ ____________________________
  src/components/SettingsDrawer/
    APIKeyManager.tsx                       │ ____________________________
  next.config.js                            │ ____________________________
  public/_headers                           │ ____________________________

  Command to generate this list after Step 4:
    find src -name "*.ts" -o -name "*.tsx" | sort

  If Base44 uses different paths, update your patch commands
  in Step 5 to use the ACTUAL paths from the right column.
================================================================

## GENERATION ORDER

```
1.  package.json                              ← implement in full
2.  next.config.js                            ← ⚠ PRE-BUILT (Part 2b)
3.  public/_headers                           ← ⚠ PRE-BUILT (Part 2b)
4.  src/db/types.ts                           ← implement in full (interfaces only)
5.  src/db/index.ts                           ← ⚠ PRE-BUILT (Part 2b)
6.  src/lib/keyManager.ts                     ← ⚠ PRE-BUILT (Part 2b)
7.  src/lib/nodeLimits.ts                     ← implement in full (from Part 3a)
8.  src/store/index.ts                        ← ⚠ PRE-BUILT (Part 3b)
9.  src/lib/aiAdapter.ts                      ← ⚠ PRE-BUILT (Part 4b)
10. src/lib/contextBuilder.ts                 ← ⚠ PRE-BUILT (Part 3b)
11. src/lib/summarizer.ts                     ← ⚠ PRE-BUILT (Part 3b)
12. src/app/layout.tsx                        ← implement in full
13. src/app/page.tsx                          ← implement in full
14. src/components/MainLayout.tsx             ← implement in full
15. src/components/TopBar/index.tsx           ← implement in full
16. src/components/TopBar/TopicSelector.tsx   ← implement in full
17. src/components/TopBar/TopicDropdown.tsx   ← implement in full
18. src/components/TopBar/TopicListItem.tsx   ← implement in full
19. src/components/TopBar/NewTopicInput.tsx   ← implement in full
20. src/components/LeftPanel/index.tsx        ← implement in full
21. src/components/LeftPanel/PanelHeader.tsx  ← implement in full
22. src/components/LeftPanel/ListView.tsx     ← implement in full
23. src/components/LeftPanel/TreeNode.tsx     ← implement in full
24. src/components/LeftPanel/MindMapView.tsx  ← implement in full
25. src/components/LeftPanel/MindMapCanvas.tsx← ⚠ PRE-BUILT (Part 4b)
26. src/components/CenterPanel/index.tsx      ← implement in full
27. src/components/CenterPanel/ConversationFeed.tsx ← implement in full
28. src/components/CenterPanel/MessageBubble.tsx    ← implement in full
29. src/components/CenterPanel/EditForm.tsx         ← implement in full
30. src/components/CenterPanel/BranchButton.tsx     ← implement in full
31. src/components/CenterPanel/TruncationNotice.tsx ← implement in full
32. src/components/CenterPanel/MessageInput.tsx     ← implement in full
33. src/components/SettingsDrawer/index.tsx         ← implement in full
34. src/components/SettingsDrawer/APIKeyManager.tsx ← ⚠ PRE-BUILT (Part 4b)
```

For ⚠ PRE-BUILT files, generate this placeholder pattern:
```typescript
// This file will be replaced by the pre-built patch.
// Export the correct types so other files can import from this path.
export * from './[filename]-types'  // if needed for type imports
```
Or simply leave a comment: `// Pre-built — will be replaced`

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

### src/app/layout.tsx
Root layout with TailwindCSS. No global state here — HTML skeleton only.

### src/app/page.tsx
- On mount: call `loadTopicNodes` for active topic (from localStorage key "thinktree_active_topic")
- If no active topic: call `getAllTopics()`, set first, or show empty state
- Render: TopBar + MainLayout + SettingsDrawer

### src/components/MainLayout.tsx
- flex row filling height below TopBar
- Manages leftWidth state (localStorage key "thinktree_left_width", default 240px)
- DragHandle: 4px div, onPointerDown/Move/Up, clamps 150–380px

### TopBar (48px fixed, full width)
Left side: TopicSelector + vertical divider (0.5px) + BranchPill
Right side: ExportButton + SettingsButton

### TopicSelector + TopicDropdown
- Trigger: [colored dot] [topic name ▾], max 200px, overflow ellipsis
- Dropdown (240px): "TOPICS" header + TopicListItem[] + "+ New topic" row
- TopicListItem (32px): dot + name + node count badge
  - badge color: default < 160, amber 160–199, red 200
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
  - Edited: small amber "edited" badge
  - Click: setActiveNodeId
  - Right-click/long-press: context menu ("Start new branch" disabled at NODE_LIMIT_MAX)
- Map view: renders MindMapView → MindMapCanvas (pre-built)

### CenterPanel
Structure top to bottom:
1. ConversationFeed (flex:1, overflow-y:auto)
2. NodeLimitBanner (amber ≥160, red ≥200, includes [New topic] + [Dismiss])
3. BranchDepthBanner (red, shown when depth ≥ BRANCH_DEPTH_MAX)
4. TruncationNotice (amber, shown after edit, has [Dismiss])
5. MessageInput (fixed bottom)

### ConversationFeed
- Derives messages from Zustand nodesState via getBranchPath equivalent
- User bubbles: right-aligned, #EEEDFE bg
- AI bubbles: left-aligned, secondary bg, 0.5px border
- summary nodes: NEVER rendered
- Hover: timestamp + [Edit] (user only) + [+ Branch] (hidden at NODE_LIMIT_MAX)
- Scrolls to bottom on new message

### EditForm
- Textarea pre-filled, auto-focused
- [Save & regenerate]: calls `invalidateSummaryGeneration` THEN `atomicEditSave`
- [Cancel]: restore bubble

### MessageInput
- Textarea 54px, Enter=submit, Shift+Enter=newline
- Disabled when: no API key, nodeCount ≥ NODE_LIMIT_MAX, depth ≥ BRANCH_DEPTH_MAX
- Send button disabled in same states

### SettingsDrawer
- Slides in from right (translateX, 0.2s), width 220px, overlays CenterPanel
- Close: × / outside click / Escape
- Contents: ProviderSelector + ModelInput + ApiKeyInput (masked) +
  RememberCheckbox + RememberWarning (amber, shown when checked) +
  ClearKeyButton + StorageStatusDisplay + Export JSON + Export Markdown

### Export / Import
JSON scope: current branch OR entire topic (includes summary nodes)
Markdown scope: current branch only (excludes summary nodes)
NEVER include API keys in exports

---

## CODE QUALITY RULES

TypeScript: strict mode, no `any` types
React: functional components, useCallback for handlers, useMemo for derived data
State: Zustand mutations ONLY after DB write resolves — never before
CSS: TailwindCSS utility classes only
Security: all key access via keyManager, zero console.log of keys


================================================================
  PART 2b — DB + SECURITY PATCH
  Step 5: Copy each file below directly into your project.
  Overwrite whatever the AI builder generated for these paths:
    src/db/index.ts
    src/lib/keyManager.ts
    next.config.js
    public/_headers
================================================================


### FILE: src/db/index.ts

⚠ REFERENCE IMPLEMENTATION — copy this exactly. Do NOT paraphrase or restructure.
The transaction logic here is precise. Dexie transactions auto-commit if the
event loop escapes their scheduler. The patterns below prevent that.

```typescript
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
    // Future migrations: APPEND ONLY — never edit version(1)
    // this.version(2).stores({ nodes: '...same + new_field' })
    // this.version(2).upgrade(tx =>
    //   tx.table('nodes').toCollection().modify({ new_field: null })
    // )
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
  // Atomic: delete old summary + insert new one in single transaction.
  // RULE: only Dexie operations inside this block. No Zustand, no side-effects.
  await db.transaction('rw', db.nodes, async () => {
    await db.nodes
      .where('[topic_id+role]')
      .equals([node.topic_id, 'summary'])
      .and(n => n.parent_id === node.parent_id)
      .delete()
    await db.nodes.put(node)
  })
}

// ── CRITICAL: recursive delete must stay inside a transaction ─

// Private helper — only called from inside db.transaction() blocks.
// NEVER call this function directly outside a transaction.
// NEVER add non-Dexie operations (fetch, setTimeout, Zustand, etc.) here.
async function _recursiveDelete(nodeId: string): Promise<void> {
  // Safe to recurse with await here because Dexie's transaction scheduler
  // keeps the transaction alive across awaits — but ONLY when all awaited
  // operations are Dexie operations on the same db instance.
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
  // topicId and branchRootId are NOT used inside the transaction.
  // They are passed here so the CALLER can call
  // invalidateSummaryGeneration() BEFORE calling this function.
  // Do NOT call invalidateSummaryGeneration inside this function.
): Promise<void> {
  // RULE: only db.nodes operations inside this transaction block.
  // invalidateSummaryGeneration() must be called by the caller BEFORE
  // this function — never inside the transaction.
  await db.transaction('rw', db.nodes, async () => {
    // Step 1: update the edited node's content
    await db.nodes.update(editedNodeId, { content: newContent })
    // Step 2: delete all descendants recursively
    await _recursiveDelete(editedNodeId)
    // Step 3: insert the new AI placeholder node
    await db.nodes.put(newAiNode)
    // If any step throws → Dexie auto-rolls back all three steps.
  })
}

// ── Topic operations ──────────────────────────────────────────

export async function saveTopic(topic: Topic): Promise<void> {
  await db.topics.put(topic)
}

export async function deleteTopic(topicId: string): Promise<void> {
  // Atomic: deletes all nodes for this topic AND the topic itself.
  await db.transaction('rw', db.topics, db.nodes, async () => {
    await db.nodes.where('topic_id').equals(topicId).delete()
    await db.topics.delete(topicId)
  })
}
```

INLINE CHECKLIST for src/db/index.ts — verify before moving to next file:
□ `_recursiveDelete` is NOT exported, NOT called outside a `db.transaction()` block
□ `atomicEditSave` does NOT call `invalidateSummaryGeneration` (caller's responsibility)
□ `atomicEditSave` does NOT contain any Zustand store mutations
□ No `new ThinkTreeDB()` anywhere except this file's module scope
□ `db` singleton is imported everywhere else — never re-instantiated
□ `[topic_id+role]` compound index is present in `version(1).stores()`

→ See MASTER ENGINEERING CHECKLIST at end of this file for full verification.

---


---

### FILE: src/lib/keyManager.ts

⚠ REFERENCE IMPLEMENTATION — copy this exactly.
AI builders routinely bypass this module and write sessionStorage.getItem()
directly inside components. This file is the ONLY permitted access point
for API key storage. The reference implementation + checklist enforce that.

```typescript
// src/lib/keyManager.ts
// ALL API key reads and writes in the entire codebase must go through
// this file. Zero direct sessionStorage/localStorage access elsewhere.

const KEY_PREFIX    = 'thinktree_api_key_'
const REMEMBER_PREFIX = 'thinktree_remember_key_'

// Load priority: sessionStorage first → fallback localStorage.
// If found in localStorage, migrate to sessionStorage immediately
// so all subsequent reads in this session go through session storage.
export function loadApiKey(provider: string): string {
  const sessionVal = sessionStorage.getItem(KEY_PREFIX + provider)
  if (sessionVal) return sessionVal

  const localVal = localStorage.getItem(KEY_PREFIX + provider)
  if (localVal) {
    sessionStorage.setItem(KEY_PREFIX + provider, localVal)
    return localVal
  }
  return ''
}

// Always writes to sessionStorage.
// Writes to localStorage only when remember=true.
// Removes from localStorage when remember=false (clears any stale value).
export function saveApiKey(
  provider: string,
  key:      string,
  remember: boolean
): void {
  sessionStorage.setItem(KEY_PREFIX + provider, key)
  if (remember) {
    localStorage.setItem(KEY_PREFIX + provider, key)
  } else {
    localStorage.removeItem(KEY_PREFIX + provider)
  }
}

// Removes from both storages unconditionally.
export function clearApiKey(provider: string): void {
  sessionStorage.removeItem(KEY_PREFIX + provider)
  localStorage.removeItem(KEY_PREFIX + provider)
}

// Display-safe masked version. Never show raw keys in UI.
// Shows provider prefix + bullets + last 4 chars.
export function maskApiKey(key: string): string {
  if (!key || key.length < 8) return '••••••••'
  const prefix = key.startsWith('sk-ant-') ? 'sk-ant-'
               : key.startsWith('sk-')     ? 'sk-'
               : ''
  return `${prefix}••••••••${key.slice(-4)}`
}

// Remember-preference storage (separate from the key itself)
export function loadRememberPref(provider: string): boolean {
  return localStorage.getItem(REMEMBER_PREFIX + provider) === 'true'
}

export function saveRememberPref(provider: string, value: boolean): void {
  localStorage.setItem(REMEMBER_PREFIX + provider, String(value))
}

export function clearRememberPref(provider: string): void {
  localStorage.removeItem(REMEMBER_PREFIX + provider)
}
```

INLINE CHECKLIST for src/lib/keyManager.ts:
□ `sessionStorage` and `localStorage` accessed ONLY in this file
□ `loadApiKey` checks sessionStorage FIRST, then localStorage
□ `loadApiKey` migrates localStorage value into sessionStorage on load
□ `saveApiKey` with remember=false calls `localStorage.removeItem` (not just skips write)
□ `maskApiKey` used in every UI element that displays a key — never raw key string
□ No `console.log` anywhere near key values

---


---

### FILE: next.config.js


```javascript
// next.config.js
const isDev = process.env.NODE_ENV === 'development'

const CSP = [
  "default-src 'self'",
  isDev ? "script-src 'self' 'unsafe-eval'" : "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  // THE CRITICAL RULE: only known AI APIs allowed outbound
  "connect-src 'self' https://api.openai.com https://api.deepseek.com https://api.anthropic.com",
  "frame-src 'none'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join('; ')

module.exports = {
  output: 'export',
  async headers() {
    return [{
      source: '/(.*)',
      headers: [
        { key: 'Content-Security-Policy',   value: CSP },
        { key: 'X-Content-Type-Options',    value: 'nosniff' },
        { key: 'X-Frame-Options',           value: 'DENY' },
        { key: 'Referrer-Policy',           value: 'strict-origin-when-cross-origin' },
      ],
    }]
  },
}
```

IMPORTANT — unsafe-eval in dev only:
Next.js hot reload requires unsafe-eval in development.
It MUST NOT appear in production builds.
The isDev ternary handles this automatically.

IMPORTANT — Cloudflare Pages static hosting:
Cloudflare Pages serves static files directly and does NOT read
next.config.js headers(). Add a `public/_headers` file:

```
/*
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' https://api.openai.com https://api.deepseek.com https://api.anthropic.com; frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self'
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
```

Vercel reads next.config.js headers() automatically — no extra file needed.

---

### FILE: public/_headers

```
/*
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' https://api.openai.com https://api.deepseek.com https://api.anthropic.com; frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self'
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
```


================================================================
  PART 3b — CORE LOGIC PATCH
  Step 5: Copy each file below directly into your project.
  Overwrite whatever the AI builder generated for these paths:
    src/store/index.ts
    src/lib/summarizer.ts
    src/lib/contextBuilder.ts
================================================================


### FILE: src/store/index.ts

⚠ REFERENCE IMPLEMENTATION — copy this exactly.
The critical rule is DB-first: every Zustand mutation that mirrors a DB write
must be called AFTER the Dexie operation resolves. AI builders frequently
invert this order (optimistic updates), producing UI state that diverges from
the actual DB on failure. Also: loadTopicNodes must call clearAllSummaryGenerations.

```typescript
// src/store/index.ts
import { create } from 'zustand'
import { db, getNodesByTopic } from '@/db'
import { clearAllSummaryGenerations } from '@/lib/summarizer'
import type { Node } from '@/db/types'

interface AppStore {
  nodes:        Node[]
  activeNodeId: string | null

  // ── Mutations — ALL called AFTER the Dexie operation resolves ──
  // NEVER call these before the corresponding DB write succeeds.
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

  // Removes nodeId's children, grandchildren, etc. from in-memory array.
  // Does NOT remove nodeId itself — that node was edited, not deleted.
  removeDescendants: (nodeId) => {
    const allNodes = get().nodes

    // Collect all descendant ids via BFS
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

  // Replaces entire nodes array with fresh data from DB.
  // Called on: app boot, topic switch.
  // MUST clear summary generation map to invalidate in-flight summaries
  // from the previous topic.
  loadTopicNodes: async (topicId) => {
    // Invalidate all in-flight summary jobs before switching context
    clearAllSummaryGenerations()

    const nodes = await db.nodes
      .where('topic_id').equals(topicId)
      .toArray()

    set({
      nodes,
      // Reset active node to last non-summary node in the topic,
      // or null if the topic is empty
      activeNodeId: nodes
        .filter(n => n.role !== 'summary')
        .at(-1)?.id ?? null,
    })
  },
}))
```

INLINE CHECKLIST for src/store/index.ts:
□ `addNode` called only AFTER `saveNode(node)` resolves
□ `updateNodeContent` called only AFTER `db.nodes.update()` resolves
□ `removeDescendants` called only AFTER `atomicEditSave()` resolves
□ `loadTopicNodes` calls `clearAllSummaryGenerations()` BEFORE the DB read
□ `removeDescendants` uses BFS — NOT a simple filter(n.parent_id !== nodeId)
     (that only removes direct children, not grandchildren)
□ No optimistic updates anywhere — UI state changes only after DB confirms

---


---

### FILE: src/lib/contextBuilder.ts

⚠ REFERENCE IMPLEMENTATION — copy this exactly.
Two bugs AI builders reliably introduce here:
(1) Missing stale-summary guard — using a summary whose covers_up_to overlaps
    with the recentWindow causes the AI to see the same messages twice.
(2) Missing token budget — 12 messages with large content can exceed provider limits.

```typescript
// src/lib/contextBuilder.ts
import { getNode, getLatestSummaryNode } from '@/db'
import type { Node }    from '@/db/types'
import type { Message } from '@/lib/aiAdapter'

// ── Token budget ───────────────────────────────────────────────
// Conservative estimate: chars / 3 (errs on the high side, no library needed).
// If recentWindow exceeds budget, trim oldest messages until it fits.
// Minimum window: 2 messages (never trim below this).

const CONTEXT_TOKEN_BUDGET = 20_000
const MIN_WINDOW_SIZE      = 2

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3)
}

// ── Branch path traversal ─────────────────────────────────────

export async function getBranchPath(nodeId: string): Promise<Node[]> {
  const path: Node[] = []
  let current: string | null = nodeId

  while (current) {
    const node = await getNode(current)
    if (!node) break
    if (node.role !== 'summary') path.push(node)
    current = node.parent_id
  }

  return path.reverse()  // root → current node
}

// ── Main context builder ──────────────────────────────────────

export async function buildContext(
  nodeId:  string,
  topicId: string
): Promise<Message[]> {

  // Step 1: full branch path (summary nodes excluded)
  const path = await getBranchPath(nodeId)

  // Step 2: take last 12 as the recent window
  let recentWindow = path.slice(-12)

  // Step 3: apply token budget — trim from oldest if over limit
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
    // ── STALE SUMMARY GUARD ─────────────────────────────────
    // The recentWindow starts at index (path.length - recentWindow.length).
    // If the summary covers messages UP TO OR BEYOND that start index,
    // the summary content overlaps with recentWindow — the AI would see
    // the same messages in both the summary and the explicit history.
    // In that case, skip the summary entirely.
    const recentStartIndex = path.length - recentWindow.length
    const summaryIsValid   = summaryNode.covers_up_to < recentStartIndex

    if (summaryIsValid) {
      // Inject summary as a user/assistant pair so all three providers accept it
      messages.push({ role: 'user',      content: summaryNode.content })
      messages.push({ role: 'assistant', content: 'Understood.' })
    }
    // If not valid: silently skip — the recentWindow already contains those messages
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

INLINE CHECKLIST for src/lib/contextBuilder.ts:
□ Stale summary guard present: `summaryNode.covers_up_to < recentStartIndex`
     (NOT just checking if summaryNode exists)
□ Token budget loop present: trims recentWindow when estimated tokens > 20,000
□ Minimum window preserved: loop condition includes `recentWindow.length > MIN_WINDOW_SIZE`
□ Summary injected as user/assistant PAIR — not as a system message
□ `getBranchPath` filters out summary nodes (`role !== 'summary'`)
□ Sibling branch nodes never included — traversal goes strictly via parent_id

---


---

### FILE: src/lib/summarizer.ts

⚠ REFERENCE IMPLEMENTATION — copy this exactly. Do NOT paraphrase or restructure.
A plain async function here will NOT work: it has no version control, so stale
summaries from in-flight jobs will silently overwrite correct state after edits
or rapid sends. The IIFE + generation_id pattern below is the only correct approach.

```typescript
// src/lib/summarizer.ts

import { sendMessage }       from '@/lib/aiAdapter'
import { upsertSummaryNode, getNodesByTopic } from '@/db'
import type { Node }         from '@/db/types'
import type { Message }      from '@/lib/aiAdapter'

// ── Generation counter ────────────────────────────────────────
// Module-level Map — NOT React state, NOT Zustand, NOT persisted.
// Lives only in memory for the lifetime of the browser tab.
// Key: "topicId::branchRootId"
// Value: monotonically increasing integer

const summaryGenerationMap = new Map<string, number>()

function genKey(topicId: string, branchRootId: string): string {
  return `${topicId}::${branchRootId}`
}

function currentGen(topicId: string, branchRootId: string): number {
  return summaryGenerationMap.get(genKey(topicId, branchRootId)) ?? 0
}

function bumpGen(topicId: string, branchRootId: string): number {
  const key  = genKey(topicId, branchRootId)
  const next = (summaryGenerationMap.get(key) ?? 0) + 1
  summaryGenerationMap.set(key, next)
  return next
}

// ── Public invalidation API ───────────────────────────────────
// Call this in EXACTLY two places (see Block 3 SUMMARY VERSION CONTROL):
//   1. Before atomicEditSave() in the EditForm handler
//   2. Inside loadTopicNodes() Zustand action on topic switch
// Do NOT call this anywhere else.

export function invalidateSummaryGeneration(
  topicId:      string,
  branchRootId: string
): void {
  bumpGen(topicId, branchRootId)
}

// Call this on topic switch to wipe all in-flight summaries for old topic.
export function clearAllSummaryGenerations(): void {
  summaryGenerationMap.clear()
}

// ── Main export ───────────────────────────────────────────────
// This function is ALWAYS called fire-and-forget (no await at call site).
// It returns void synchronously; all async work runs inside an IIFE.

export function triggerSummaryIfNeeded(
  path:        Node[],
  provider:    string,
  apiKey:      string,
  topicId:     string
): void {
  const toSummarize = path.slice(0, path.length - 12)
  if (toSummarize.length === 0) return

  const branchRootId = path[0].id

  // Snapshot both generation id and path length at call time.
  // These are captured in closure — they do NOT update if the world
  // changes while we are waiting for the AI response.
  const capturedGenId  = currentGen(topicId, branchRootId)
  const capturedLength = path.length

  // IIFE: fire-and-forget async block. The outer function returns void
  // immediately; this block runs concurrently.
  ;(async () => {
    try {
      const text = await sendMessage(
        provider,
        apiKey,
        buildSummarizationMessages(toSummarize)
      )

      // ── GUARD 1: edit invalidation ──────────────────────────
      // An atomicEditSave() called invalidateSummaryGeneration()
      // while we were waiting. Our result is based on a branch that
      // no longer exists in its original form. Discard.
      if (currentGen(topicId, branchRootId) !== capturedGenId) {
        console.debug('[summarizer] discarded: generation invalidated')
        return
      }

      // ── GUARD 2: rapid-send staleness ──────────────────────
      // More messages arrived while we were summarizing. Our
      // covers_up_to would be wrong relative to the new window.
      // Discard — a new trigger will fire at the next multiple-of-10.
      const currentNodes = await getNodesByTopic(topicId)
      const nonSummary   = currentNodes.filter(n => n.role !== 'summary')
      if (nonSummary.length > capturedLength) {
        console.debug('[summarizer] discarded: path grew during generation')
        return
      }

      // ── GUARD 3: topic switch ───────────────────────────────
      // clearAllSummaryGenerations() was called (topic switched).
      // currentGen returns 0, capturedGenId was > 0, so Guard 1
      // already catches this. Guard 3 is implicit — no extra code needed.

      await upsertSummaryNode({
        id:           `summary_${topicId}_${branchRootId}`,
        topic_id:     topicId,
        parent_id:    branchRootId,
        role:         'summary',
        content:      text,
        timestamp:    Date.now(),
        covers_up_to: capturedLength - 13,
      })

    } catch (err) {
      // Summary failure is always silent — never block the main message flow.
      console.warn('[summarizer] generation failed:', err)
    }
  })()
}

// ── Summarization prompt builder ──────────────────────────────

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

INLINE CHECKLIST for src/lib/summarizer.ts — verify before moving to next file:
□ `summaryGenerationMap` is module-level — NOT inside any function, component, or hook
□ `triggerSummaryIfNeeded` return type is `void` — caller never awaits it
□ Async work is inside an IIFE `(async () => { ... })()`
□ `capturedGenId` and `capturedLength` are captured BEFORE the first `await`
□ Guard 1 present: `currentGen(...) !== capturedGenId` → return
□ Guard 2 present: `nonSummary.length > capturedLength` → return
□ `upsertSummaryNode` is called ONLY after both guards pass
□ Entire async block is wrapped in `try/catch` → `console.warn` only, never throws
□ `invalidateSummaryGeneration` is called in EditForm handler BEFORE `atomicEditSave`
□ `clearAllSummaryGenerations` is called inside `loadTopicNodes` Zustand action

→ See MASTER ENGINEERING CHECKLIST Section 2 at end of this file for full verification.

---

================================================================
  PART 4b — CANVAS + API PATCH
  Step 5: Copy each file below directly into your project.
  Overwrite whatever the AI builder generated for these paths:
    src/lib/aiAdapter.ts
    src/components/LeftPanel/MindMapCanvas.tsx
    src/components/SettingsDrawer/APIKeyManager.tsx  ← new in v9.4
================================================================


### FILE: src/lib/aiAdapter.ts

⚠ REFERENCE IMPLEMENTATION — copy this exactly.
The Anthropic provider has a unique message format: the system prompt must be
a top-level parameter, NOT a message in the messages array. AI builders
consistently get this wrong, causing all Claude requests to fail silently
or return unexpected responses. The reference below handles all three providers.

```typescript
// src/lib/aiAdapter.ts
import Anthropic from '@anthropic-ai/sdk'

export interface Message {
  role:    'system' | 'user' | 'assistant'
  content: string
}

// Unified adapter — the ONLY place that calls provider APIs directly.
// All other modules use this function.
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
      // Extract system message first, then pass remaining messages separately.
      const systemMsg = messages.find(m => m.role === 'system')
      const chatMsgs  = messages
        .filter(m => m.role !== 'system')
        .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }))

      const client = new Anthropic({
        apiKey,
        dangerouslyAllowBrowser: true,  // required for browser-side SDK use
      })

      const res = await client.messages.create({
        model:      'claude-sonnet-4-5',
        max_tokens: 4096,
        system:     systemMsg?.content ?? 'You are a helpful assistant.',
        messages:   chatMsgs,
      })

      // Response content is an array of blocks; text is in the first text block
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

// Maps HTTP status codes to typed error strings for the error handling table.
function classifyError(status: number, body: any): Error {
  if (status === 401) return Object.assign(new Error('invalid_api_key'), { type: 'invalid_api_key' })
  if (status === 429) return Object.assign(new Error('rate_limit'),       { type: 'rate_limit' })
  return Object.assign(
    new Error(body?.error?.message ?? 'network_failure'),
    { type: 'network_failure' }
  )
}
```

INLINE CHECKLIST for src/lib/aiAdapter.ts:
□ Anthropic: system message extracted from messages[] and passed as TOP-LEVEL `system:` param
□ Anthropic: `dangerouslyAllowBrowser: true` set in Anthropic constructor
□ Anthropic: model is `claude-sonnet-4-5` (not claude-3, not claude-2)
□ Anthropic: response read from `res.content.find(b => b.type === 'text').text`
     NOT from `res.content[0].text` (order not guaranteed)
□ OpenAI/DeepSeek: system message stays inside messages[] (different format from Anthropic)
□ All three branches throw typed errors that match the error handling table in Block 3
□ No API keys logged anywhere in this file

---

---

### FILE: src/components/LeftPanel/MindMapCanvas.tsx

MindMapCanvas.tsx:

⚠ REFERENCE IMPLEMENTATION — copy this exactly.
Five specific bugs appear when AI builders generate this from description alone.
The implementation below prevents all five. Do not paraphrase or restructure.

```typescript
// src/components/LeftPanel/MindMapCanvas.tsx

import { useRef, useCallback } from 'react'
import type { Node } from '@/db/types'

// ── Types ──────────────────────────────────────────────────────

interface ViewTransform { x: number; y: number; scale: number }
interface Pos           { x: number; y: number }

interface Props {
  nodes:           Node[]                        // all non-summary nodes for this topic
  activeNodeId:    string | null
  viewTransform:   ViewTransform
  nodePositions:   Map<string, Pos>              // manual overrides; absent = use auto-layout
  onViewChange:    (vt: ViewTransform) => void
  onNodeMove:      (id: string, pos: Pos) => void
  onNodeTap:       (id: string) => void          // navigate to branch
}

// ── Radial auto-layout ─────────────────────────────────────────

function computeAutoLayout(nodes: Node[]): Map<string, Pos> {
  const positions = new Map<string, Pos>()
  const root = nodes.find(n => n.parent_id === null)
  if (!root) return positions

  const RADIUS_STEP = 90   // px per depth level
  const ARC_DEG     = 70   // degrees spread per parent

  function place(nodeId: string, depth: number, angleDeg: number) {
    if (depth === 0) {
      positions.set(nodeId, { x: 0, y: 0 })
    } else {
      const rad = (angleDeg * Math.PI) / 180
      positions.set(nodeId, {
        x: Math.cos(rad) * RADIUS_STEP * depth,
        y: Math.sin(rad) * RADIUS_STEP * depth,
      })
    }
    const children = nodes.filter(n => n.parent_id === nodeId)
    const startAngle = angleDeg - (ARC_DEG * (children.length - 1)) / 2
    children.forEach((child, i) => {
      place(child.id, depth + 1, startAngle + ARC_DEG * i)
    })
  }

  place(root.id, 0, -90)
  return positions
}

// ── Distance helper for pinch-zoom ────────────────────────────

function dist(a: React.Touch | PointerEvent, b: React.Touch | PointerEvent) {
  return Math.hypot(
    (a as any).clientX - (b as any).clientX,
    (a as any).clientY - (b as any).clientY
  )
}

// ── Component ─────────────────────────────────────────────────

export function MindMapCanvas({
  nodes, activeNodeId, viewTransform, nodePositions,
  onViewChange, onNodeMove, onNodeTap,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)

  // ── GAP 2 FIX: activePointers Map tracks ALL live pointers ──
  // Key = pointerId, Value = latest PointerEvent for that pointer.
  // This is the ONLY correct way to detect 1-finger vs 2-finger.
  const activePointers = useRef<Map<number, PointerEvent>>(new Map())

  // Drag state (single pointer on a node)
  const dragState = useRef<{
    nodeId:     string
    startPtr:   Pos        // pointer position at drag start (screen coords)
    startNode:  Pos        // node position at drag start (canvas coords)
  } | null>(null)

  // Pan state (single pointer on empty canvas)
  const panState = useRef<{
    startPtr: Pos
    startVT:  ViewTransform
  } | null>(null)

  // Pinch state (two pointers)
  const pinchState = useRef<{
    startDist:  number
    startScale: number
    midpoint:   Pos        // screen midpoint of the two pointers at pinch start
    startVT:    ViewTransform
  } | null>(null)

  // Tap detection
  const tapState = useRef<{
    nodeId:   string | null
    startPos: Pos
  } | null>(null)

  // ── GAP 3 FIX: coordinate conversion helpers ────────────────
  // All pointer positions arrive in screen (client) coords.
  // Node positions live in canvas coords (after translate + scale).
  // Converting between them:
  //   canvas = (screen - translate) / scale
  //   screen = canvas * scale + translate

  function screenToCanvas(screenX: number, screenY: number, vt: ViewTransform): Pos {
    return {
      x: (screenX - vt.x) / vt.scale,
      y: (screenY - vt.y) / vt.scale,
    }
  }

  // ── GAP 1 FIX: node hit-test in canvas coords ───────────────
  // We determine "did the pointer land on a node?" by checking
  // canvas-coordinate distance — not by relying on SVG event bubbling
  // or e.target, both of which are unreliable in nested SVG groups.

  const NODE_HIT_RADIUS = 22   // px in canvas coords (matches 44px tap target)

  function hitTestNode(screenX: number, screenY: number, vt: ViewTransform): string | null {
    const c = screenToCanvas(screenX, screenY, vt)
    const autoLayout = computeAutoLayout(nodes)
    for (const node of nodes) {
      const pos = nodePositions.get(node.id) ?? autoLayout.get(node.id) ?? { x: 0, y: 0 }
      const dx = c.x - pos.x
      const dy = c.y - pos.y
      if (Math.hypot(dx, dy) <= NODE_HIT_RADIUS) return node.id
    }
    return null
  }

  // ── Pointer event handlers ──────────────────────────────────

  const onPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    // GAP 4 FIX: capture pointer so move/up fire even outside SVG bounds
    e.currentTarget.setPointerCapture(e.pointerId)

    activePointers.current.set(e.pointerId, e.nativeEvent)
    const count = activePointers.current.size

    if (count === 1) {
      // GAP 1 FIX: use canvas-coord hit-test, not e.target
      const hitId = hitTestNode(e.clientX, e.clientY, viewTransform)

      // GAP 5 FIX: record tap start for movement threshold check
      tapState.current = {
        nodeId:   hitId,
        startPos: { x: e.clientX, y: e.clientY },
      }

      if (hitId) {
        // Drag mode: pointer landed on a node
        const autoLayout = computeAutoLayout(nodes)
        const nodePos = nodePositions.get(hitId) ?? autoLayout.get(hitId) ?? { x: 0, y: 0 }
        dragState.current = {
          nodeId:    hitId,
          startPtr:  { x: e.clientX, y: e.clientY },
          startNode: { ...nodePos },
        }
        panState.current = null
      } else {
        // Pan mode: pointer landed on empty canvas
        panState.current = {
          startPtr: { x: e.clientX, y: e.clientY },
          startVT:  { ...viewTransform },
        }
        dragState.current = null
      }
      pinchState.current = null

    } else if (count === 2) {
      // GAP 2 FIX: switch to pinch-zoom when second pointer arrives
      dragState.current = null
      panState.current  = null

      const ptrs = Array.from(activePointers.current.values())
      const initialDist = dist(ptrs[0], ptrs[1])
      const midX = (ptrs[0].clientX + ptrs[1].clientX) / 2
      const midY = (ptrs[0].clientY + ptrs[1].clientY) / 2

      pinchState.current = {
        startDist:  initialDist,
        startScale: viewTransform.scale,
        midpoint:   { x: midX, y: midY },
        startVT:    { ...viewTransform },
      }
    }
  }, [viewTransform, nodePositions, nodes])

  const onPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    activePointers.current.set(e.pointerId, e.nativeEvent)
    const count = activePointers.current.size

    if (count === 2 && pinchState.current) {
      // GAP 2 FIX: pinch-zoom with correct midpoint pivot
      const ptrs = Array.from(activePointers.current.values())
      const currentDist = dist(ptrs[0], ptrs[1])
      const { startDist, startScale, midpoint, startVT } = pinchState.current

      const rawScale = startScale * (currentDist / startDist)
      const newScale = Math.min(3.0, Math.max(0.3, rawScale))

      // Zoom toward the pinch midpoint (not the canvas origin)
      // GAP 3 FIX: pivot calculation in screen coords
      const newX = midpoint.x - (midpoint.x - startVT.x) * (newScale / startScale)
      const newY = midpoint.y - (midpoint.y - startVT.y) * (newScale / startScale)

      onViewChange({ x: newX, y: newY, scale: newScale })

    } else if (count === 1) {
      if (dragState.current) {
        // GAP 3 FIX: divide screen delta by scale to get canvas delta
        const { nodeId, startPtr, startNode } = dragState.current
        const dx = (e.clientX - startPtr.x) / viewTransform.scale
        const dy = (e.clientY - startPtr.y) / viewTransform.scale
        onNodeMove(nodeId, { x: startNode.x + dx, y: startNode.y + dy })

      } else if (panState.current) {
        // Pan: screen delta applied directly to translate (no scale division)
        const { startPtr, startVT } = panState.current
        onViewChange({
          ...startVT,
          x: startVT.x + (e.clientX - startPtr.x),
          y: startVT.y + (e.clientY - startPtr.y),
        })
      }
    }
  }, [viewTransform, onViewChange, onNodeMove])

  const onPointerUp = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    activePointers.current.delete(e.pointerId)

    // GAP 5 FIX: tap detection — only fire if movement < 5px
    if (tapState.current) {
      const { nodeId, startPos } = tapState.current
      const moved = Math.hypot(e.clientX - startPos.x, e.clientY - startPos.y)
      if (moved < 5 && nodeId) {
        onNodeTap(nodeId)
      }
      tapState.current = null
    }

    if (activePointers.current.size < 2) {
      pinchState.current = null
    }
    if (activePointers.current.size === 0) {
      dragState.current = null
      panState.current  = null
    }
  }, [onNodeTap])

  // ── Render ─────────────────────────────────────────────────

  const autoLayout  = computeAutoLayout(nodes)
  const { x, y, scale } = viewTransform

  return (
    <svg
      ref={svgRef}
      style={{ width: '100%', height: '100%', touchAction: 'none' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <g transform={`translate(${x},${y}) scale(${scale})`}>
        {/* Bezier edges rendered BEFORE nodes (nodes sit on top) */}
        {nodes.map(node => {
          if (!node.parent_id) return null
          const from = nodePositions.get(node.parent_id) ?? autoLayout.get(node.parent_id) ?? { x: 0, y: 0 }
          const to   = nodePositions.get(node.id)        ?? autoLayout.get(node.id)        ?? { x: 0, y: 0 }
          const mx   = (from.x + to.x) / 2
          const my   = (from.y + to.y) / 2
          return (
            <path
              key={`edge-${node.id}`}
              d={`M ${from.x} ${from.y} Q ${mx} ${from.y} ${mx} ${my} Q ${mx} ${to.y} ${to.x} ${to.y}`}
              fill="none"
              stroke="var(--color-border-secondary)"
              strokeWidth={1 / scale}   // keep edge width visually constant when zoomed
            />
          )
        })}

        {/* Nodes */}
        {nodes.map(node => {
          const pos      = nodePositions.get(node.id) ?? autoLayout.get(node.id) ?? { x: 0, y: 0 }
          const isActive = node.id === activeNodeId
          const radius   = node.parent_id === null ? 18 : 12   // root is larger
          const fill = {
            user_question: '#E6F1FB',
            ai_response:   '#E1F5EE',
            summary:       'transparent',
          }[node.role] ?? '#F1EFE8'

          return (
            <g key={node.id} transform={`translate(${pos.x},${pos.y})`}>
              {/* 44px minimum hit area (transparent, always on top of circle) */}
              <circle r={Math.max(radius, 22)} fill="transparent" />
              {/* Visible circle */}
              <circle
                r={radius}
                fill={fill}
                stroke={isActive ? '#EF9F27' : '#7F77DD'}
                strokeWidth={isActive ? 2.5 / scale : 0.5 / scale}
                strokeDasharray={node.role === 'user_question' && (node as any).truncatedBelow
                  ? `${4 / scale} ${2 / scale}`
                  : undefined}
              />
              <text
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={10 / scale}
                fill="var(--color-text-secondary)"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                {node.content.slice(0, 12)}{node.content.length > 12 ? '…' : ''}
              </text>
            </g>
          )
        })}
      </g>
    </svg>
  )
}
```

MindMapNode.tsx:
- Not needed as a separate component — node rendering is inlined in MindMapCanvas
  (separating it into its own component would require passing scale as a prop to
  every node, and forwardRef on SVG <g> elements is unreliable across React versions)
- If the builder insists on a separate MindMapNode component, it MUST receive
  `scale` as a prop and use it for strokeWidth and fontSize calculations

---

### FILE: src/components/SettingsDrawer/APIKeyManager.tsx

⚠ REFERENCE IMPLEMENTATION — copy this exactly.
Prevents AI builder from writing sessionStorage.getItem() directly
in components. All storage access goes through keyManager functions.

```typescript
// src/components/SettingsDrawer/APIKeyManager.tsx
import { useState, useEffect } from 'react'
import {
  loadApiKey, saveApiKey, clearApiKey, maskApiKey,
  loadRememberPref, saveRememberPref, clearRememberPref,
} from '@/lib/keyManager'

const PROVIDERS = ['openai', 'deepseek', 'anthropic'] as const
type Provider = typeof PROVIDERS[number]

const DEFAULT_MODELS: Record<Provider, string> = {
  openai:    'gpt-4o-mini',
  deepseek:  'deepseek-chat',
  anthropic: 'claude-sonnet-4-5',
}

interface Props {
  provider:  Provider
  onProviderChange: (p: Provider) => void
  model:     string
  onModelChange: (m: string) => void
}

export function APIKeyManager({ provider, onProviderChange, model, onModelChange }: Props) {
  const [keyInput,    setKeyInput]    = useState('')
  const [showKey,     setShowKey]     = useState(false)
  const [remember,    setRemember]    = useState(false)
  const [isSaved,     setIsSaved]     = useState(false)

  // Load key and remember pref whenever provider changes
  useEffect(() => {
    const existing = loadApiKey(provider)
    const pref     = loadRememberPref(provider)
    // Always display masked version — never show raw key in state
    setKeyInput(existing ? maskApiKey(existing) : '')
    setRemember(pref)
    setIsSaved(!!existing)
    setShowKey(false)
  }, [provider])

  const handleSave = () => {
    // Only save if user typed a new key (not the masked placeholder)
    if (!keyInput || keyInput.includes('••')) return
    saveApiKey(provider, keyInput, remember)
    saveRememberPref(provider, remember)
    setKeyInput(maskApiKey(keyInput))
    setShowKey(false)
    setIsSaved(true)
  }

  const handleClear = () => {
    clearApiKey(provider)
    clearRememberPref(provider)
    setKeyInput('')
    setRemember(false)
    setIsSaved(false)
  }

  const handleRememberChange = (checked: boolean) => {
    setRemember(checked)
    saveRememberPref(provider, checked)
    // If key already saved, update storage immediately
    if (isSaved) {
      const existing = loadApiKey(provider)
      if (existing) saveApiKey(provider, existing, checked)
    }
  }

  const storageStatus = remember
    ? 'API key: session + local (remembered)'
    : 'API key: session only'

  return (
    <div className="flex flex-col gap-4 p-4">

      {/* Provider selector */}
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500 uppercase tracking-wide">Provider</label>
        <select
          value={provider}
          onChange={e => {
            onProviderChange(e.target.value as Provider)
            onModelChange(DEFAULT_MODELS[e.target.value as Provider])
          }}
          className="border rounded px-2 py-1 text-sm"
        >
          {PROVIDERS.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>

      {/* Model input */}
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500 uppercase tracking-wide">Model</label>
        <input
          type="text"
          value={model}
          onChange={e => onModelChange(e.target.value)}
          className="border rounded px-2 py-1 text-sm"
        />
      </div>

      {/* API key input */}
      <div className="flex flex-col gap-1">
        <label className="text-xs text-gray-500 uppercase tracking-wide">API Key</label>
        <div className="flex gap-1">
          <input
            type={showKey ? 'text' : 'password'}
            value={keyInput}
            onChange={e => setKeyInput(e.target.value)}
            onFocus={() => {
              // Clear mask when user clicks to edit
              if (keyInput.includes('••')) setKeyInput('')
            }}
            placeholder="Paste your API key…"
            className="border rounded px-2 py-1 text-sm flex-1"
          />
          <button
            onClick={() => setShowKey(v => !v)}
            className="border rounded px-2 py-1 text-xs"
          >{showKey ? 'Hide' : 'Show'}</button>
        </div>
        <div className="flex gap-2 mt-1">
          <button
            onClick={handleSave}
            className="text-xs bg-purple-600 text-white rounded px-3 py-1"
          >Save</button>
          {isSaved && (
            <button
              onClick={handleClear}
              className="text-xs border rounded px-3 py-1"
            >Clear</button>
          )}
        </div>
      </div>

      {/* Remember checkbox */}
      <label className="flex items-center gap-2 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={remember}
          onChange={e => handleRememberChange(e.target.checked)}
        />
        Remember across browser restarts
      </label>
      {remember && (
        <p className="text-xs text-amber-600">
          Key stored in localStorage. Clear it when using shared devices.
        </p>
      )}

      {/* Storage status */}
      <p className="text-xs text-gray-400">{storageStatus}</p>

    </div>
  )
}
```

INLINE CHECKLIST for APIKeyManager.tsx:
□ `loadApiKey` / `saveApiKey` / `clearApiKey` all imported from '@/lib/keyManager'
□ No `sessionStorage` or `localStorage` calls anywhere in this file
□ Key displayed using `maskApiKey()` — raw key never stored in component state
□ `handleRememberChange` calls `saveRememberPref` immediately on checkbox change
□ `handleSave` does NOT save if keyInput contains '••' (masked value, not new input)

---


================================================================
  PART CHECKLIST — MASTER ENGINEERING CHECKLIST
  Step 6: Run after completing Step 5 (patch layer applied).
  A single ✗ means the file is not safe — fix before running.
================================================================

----------------------------------------------------------------
  STEP 5 IMMEDIATE CHECK — Run these BEFORE the full checklist.
  These catch the most critical issues introduced by AI builders.
  If any returns unexpected results, fix before proceeding.
----------------------------------------------------------------

```bash
# Run immediately after Step 5 patch is applied:

# 1. API key isolation — only keyManager.ts should touch storage
grep -r "sessionStorage\|localStorage" src/components/
# Expected: ZERO results
# If results found: AI builder wrote direct storage access in a component.
# Fix: replace with loadApiKey() / saveApiKey() from keyManager.

# 2. DB singleton — only one instantiation allowed
grep -r "new ThinkTreeDB" src/
# Expected: exactly ONE result (in src/db/index.ts)

# 3. No raw IndexedDB in DB layer
grep -r "new Promise\|setTimeout" src/db/
# Expected: ZERO results

# 4. summarizer map is module-level only
grep -r "summaryGenerationMap" src/
# Expected: only results in src/lib/summarizer.ts

# 5. Anthropic system message format
grep -r "content\[0\]\.text" src/lib/aiAdapter.ts
# Expected: ZERO results (must use .find(b => b.type === 'text').text)
```

If all five pass, proceed to full checklist below.
If any fail, fix the affected file (re-apply the relevant patch) before continuing.

----------------------------------------------------------------

  SECTION 1 — DATABASE LAYER  (src/db/index.ts)
  Check after: initial generation AND any later modification
----------------------------------------------------------------

Dexie transaction correctness:
□ `_recursiveDelete` function is NOT exported (no `export` keyword)
□ `_recursiveDelete` is ONLY called from inside `db.transaction()` blocks
□ `_recursiveDelete` contains NO non-Dexie operations
     (no fetch, no setTimeout, no Zustand calls, no console side-effects)
□ `atomicEditSave` calls `db.transaction('rw', db.nodes, async () => {...})`
     and ALL three steps (update + recursiveDelete + put) are inside that block
□ `atomicEditSave` does NOT call `invalidateSummaryGeneration()`
     (that call belongs in the component/hook that calls atomicEditSave)
□ `atomicEditSave` does NOT call any Zustand store mutation
□ `upsertSummaryNode` wraps its delete+put in `db.transaction()`
□ `deleteTopic` wraps its node-delete+topic-delete in
     `db.transaction('rw', db.topics, db.nodes, async () => {...})`

Singleton pattern:
□ `new ThinkTreeDB()` appears exactly ONCE in the entire codebase
     (in src/db/index.ts module scope as `export const db = new ThinkTreeDB()`)
□ Search result for `new ThinkTreeDB` in all other files: ZERO matches

Schema:
□ `[topic_id+role]` compound index declared in `version(1).stores()` for nodes
□ No existing `version(1)` block has been modified
     (new fields use a new `version(2).stores(...).upgrade(...)` block)

----------------------------------------------------------------
  SECTION 2 — SUMMARY RACE CONDITION  (src/lib/summarizer.ts)
  Check after: initial generation AND any modification to summary logic
----------------------------------------------------------------

generation_id pattern:
□ `summaryGenerationMap` is module-level (not inside a React component or hook)
□ `invalidateSummaryGeneration()` is called in EXACTLY two places:
     a) Inside `atomicEditSave` caller (component/hook), BEFORE the await
     b) Inside `loadTopicNodes` Zustand action, on topic switch
□ `invalidateSummaryGeneration()` is NOT called anywhere else
□ `triggerSummaryIfNeeded` captures `capturedGenId` and `capturedLength`
     at call time (before the async work begins)

Stale result guards (inside the async IIFE in triggerSummaryIfNeeded):
□ Guard 1 present: checks `currentGeneration(...) !== capturedGenId` → discard
□ Guard 2 present: checks current node count > capturedLength → discard
□ Both guards appear BEFORE `upsertSummaryNode()` is called
□ The function is truly fire-and-forget: caller does NOT await it
□ All errors caught with try/catch → console.warn only, never thrown

----------------------------------------------------------------
  SECTION 3 — API KEY SECURITY  (src/lib/keyManager.ts)
  Check after: initial generation AND any Settings-related changes
----------------------------------------------------------------

Storage isolation:
□ `sessionStorage` and `localStorage` for API keys are accessed ONLY
     in src/lib/keyManager.ts — zero direct access anywhere else
□ Search for `sessionStorage.setItem` outside keyManager.ts: ZERO matches
□ Search for `localStorage.setItem.*api_key` outside keyManager.ts: ZERO matches
□ `loadApiKey` checks sessionStorage FIRST before falling back to localStorage
□ `loadApiKey` copies a localStorage hit into sessionStorage before returning it
□ `saveApiKey` with remember=false explicitly calls `localStorage.removeItem`
     (not just skipping the write — stale values must be cleared)

Key masking:
□ `maskApiKey()` is called wherever a key is displayed in the UI
□ No raw API key string ever passed to React state that feeds a visible text node
□ Zero `console.log` calls that could print an API key

CSP (next.config.js):
□ `script-src 'self'` present (no `unsafe-inline` in production)
□ `unsafe-eval` is inside an `isDev` ternary — NOT in the production string
□ `connect-src` whitelist contains exactly:
     'self'  https://api.openai.com  https://api.deepseek.com  https://api.anthropic.com
     and NO other domains
□ public/_headers file exists for Cloudflare Pages deployment
     and its CSP string matches next.config.js (without unsafe-eval)

----------------------------------------------------------------
  SECTION 4 — SHARED STATE  (src/store/index.ts)
  Check after: initial generation AND any store modification
----------------------------------------------------------------

DB-first rule:
□ Every Zustand mutation that corresponds to a DB write is called AFTER
     the Dexie operation resolves — never before (no optimistic updates)
□ In the edit flow: `atomicEditSave()` resolves → THEN store mutations run
□ In the normal send flow: `saveNode()` resolves → THEN `addNode()` runs

Derived state:
□ `removeDescendants` uses BFS to collect ALL descendant ids before filtering
     NOT a simple `.filter(n => n.parent_id !== nodeId)` (that only removes
     direct children, leaving grandchildren as orphans in the UI)
□ `loadTopicNodes` replaces the ENTIRE nodes array (not appends)
□ `loadTopicNodes` calls `clearAllSummaryGenerations()` BEFORE the DB read
     to invalidate in-flight summaries from the previous topic

----------------------------------------------------------------
  SECTION 4b — AI ADAPTER  (src/lib/aiAdapter.ts)
  Check after: initial generation AND any provider-related changes
----------------------------------------------------------------

Anthropic format (most commonly broken):
□ System message is passed as TOP-LEVEL `system:` parameter in the SDK call
     NOT as a message with role: 'system' inside the messages array
□ `dangerouslyAllowBrowser: true` is set in the Anthropic constructor
□ Model is `claude-sonnet-4-5` (not claude-3-opus, not claude-2, not claude-3-haiku)
□ Response text read as `res.content.find(b => b.type === 'text').text`
     NOT `res.content[0].text` (content block order is not guaranteed)

OpenAI / DeepSeek format:
□ System message stays inside messages[] for OpenAI and DeepSeek
     (only Anthropic requires it at the top level)
□ `response.choices[0].message.content` used to extract text

All providers:
□ HTTP errors mapped to typed error objects matching the error table in Block 3
□ No API keys appear in error messages or logs

----------------------------------------------------------------
  SECTION 5 — NODE LIMIT GUARDS  (src/lib/nodeLimits.ts + MessageInput)
  Check after: initial generation AND any changes to message send logic
----------------------------------------------------------------

Guard order in MessageInput send handler:
□ Guard 1 (API key): `loadApiKey()` → empty → show notice, return early
□ Guard 2 (node limit): `getNodeCount() >= NODE_LIMIT_MAX - 1` → return early
     (-1 reserves a slot for the AI response node)
□ Guard 3 (branch depth): `canExtendBranch(path)` → false → return early
□ Guards are checked in this ORDER before any node is created

UI consistency:
□ Send button is disabled when nodeCount >= NODE_LIMIT_MAX
□ Send button is disabled when branch depth >= BRANCH_DEPTH_MAX
□ `+ Branch` button is HIDDEN (not just disabled) at NODE_LIMIT_MAX
□ "Start new branch" in context menus is DISABLED at NODE_LIMIT_MAX
□ NodeLimitBanner shows amber at 160–199 nodes, red at 200 nodes
□ Topic Dropdown node count badge: amber at 160–199, red at 200

----------------------------------------------------------------
  SECTION 6 — CONTEXT BUILDER  (src/lib/contextBuilder.ts)
  Check after: initial generation AND any changes to buildContext
----------------------------------------------------------------

Branch isolation:
□ buildContext() only includes nodes from the current branch path
     (root → activeNode via parent_id traversal)
□ Sibling branch nodes are NEVER included
□ getBranchPath filters out summary nodes before returning

Stale summary guard:
□ `recentStartIndex = path.length - recentWindow.length` computed first
□ Summary used ONLY if `summaryNode.covers_up_to < recentStartIndex`
     NOT just `if (summaryNode)` — that allows stale/overlapping summaries
□ Summary is injected as a user/assistant pair (not as a system message)

Token budget:
□ Token estimation: `Math.ceil(content.length / 3)` — no external library
□ Trim loop: removes oldest recentWindow entries until under 20,000 tokens
□ Minimum window: `recentWindow.length > MIN_WINDOW_SIZE (2)` — never trim to 0 or 1

----------------------------------------------------------------
  SECTION 7 — COMPONENT INTEGRATION
  Spot-check after: generating any component that touches DB or state
----------------------------------------------------------------

EditForm (src/components/CenterPanel/EditForm.tsx):
□ Calls `invalidateSummaryGeneration(topicId, branchRootId)` BEFORE
     calling `atomicEditSave()`
□ Does NOT call any Dexie function directly (all DB via src/db/index.ts)
□ Zustand mutations run AFTER atomicEditSave resolves

MessageInput (src/components/CenterPanel/MessageInput.tsx):
□ Calls `loadApiKey()` from keyManager — not sessionStorage directly
□ Three guards checked before creating any node (see Section 5)
□ `triggerSummaryIfNeeded()` called fire-and-forget (no await)

MindMapCanvas (src/components/LeftPanel/MindMapCanvas.tsx):
□ SVG element has `style={{ touchAction: 'none' }}` (React camelCase, not HTML attribute)
□ Uses `onPointerDown/onPointerMove/onPointerUp/onPointerCancel` — NOT separate
     mouse and touch handlers, NOT onClick for drag detection
□ summary nodes are filtered out before rendering

GAP 1 — pan vs drag discrimination:
□ Node hit-testing uses canvas-coordinate distance check (`hitTestNode` function)
     NOT `e.target` comparison and NOT SVG event bubbling / stopPropagation
□ `hitTestNode` converts screen coords to canvas coords before comparing to node positions
□ Hit radius is defined in canvas coords (not screen coords), e.g. 22px canvas = 44px touch target

GAP 2 — pinch-zoom multi-pointer tracking:
□ `activePointers` is a `Map<number, PointerEvent>` (ref, not state)
□ `activePointers.set(e.pointerId, e.nativeEvent)` in onPointerDown AND onPointerMove
□ `activePointers.delete(e.pointerId)` in onPointerUp AND onPointerCancel
□ Pinch mode activates when `activePointers.size === 2`
□ Scale computed as `startScale * (currentDist / startDist)`
□ Scale clamped to range [0.3, 3.0]
□ When second pointer arrives, drag and pan states are both cleared to null

GAP 3 — coordinate system conversion:
□ Drag delta divides screen delta by `viewTransform.scale`:
     `dx = (e.clientX - startPtr.x) / viewTransform.scale`
□ Pan delta does NOT divide by scale (translate is in screen space)
□ Pinch zoom pivot uses midpoint formula:
     `newX = midX - (midX - startVT.x) * (newScale / startScale)`
     NOT `newX = midX * (1 - newScale)` (common wrong formula)

GAP 4 — pointer capture:
□ `e.currentTarget.setPointerCapture(e.pointerId)` called in onPointerDown
□ This ensures onPointerMove and onPointerUp fire even when pointer leaves SVG bounds

GAP 5 — tap vs drag distinction:
□ `tapState` records `{ nodeId, startPos }` on every onPointerDown
□ onPointerUp checks `Math.hypot(dx, dy) < 5` before firing `onNodeTap`
□ `tapState` is cleared in onPointerUp regardless of outcome
□ Long-press (if implemented) uses a setTimeout started in onPointerDown
     and cleared in onPointerMove (if movement > 5px) and onPointerUp

Rendering correctness:
□ Bezier edges are rendered BEFORE node circles in SVG source order
     (later elements render on top — nodes must cover edge endpoints)
□ Edge `strokeWidth` is `1 / scale` to stay visually constant when zoomed
□ Node `strokeWidth` and `fontSize` also divide by scale
□ `strokeDasharray` values also divide by scale for edited nodes
□ `computeAutoLayout` is called with the full node list and returns a
     `Map<string, Pos>` — per-node position falls back to auto if not in `nodePositions`

APIKeyManager (src/components/SettingsDrawer/APIKeyManager.tsx):
□ All key reads/writes go through keyManager functions
□ "Remember" checkbox state saved to localStorage separately from the key
□ Displayed key is always the masked version via `maskApiKey()`

----------------------------------------------------------------
  SECTION 8 — FINAL INTEGRATION CHECKS
  Run once after ALL files have been generated
----------------------------------------------------------------

Search-based verification (run in your editor or terminal):

□ `grep -r "new ThinkTreeDB" src/` → only ONE result (in src/db/index.ts)
□ `grep -r "sessionStorage" src/` → only results in src/lib/keyManager.ts
□ `grep -r "localStorage.*api_key" src/` → only results in src/lib/keyManager.ts
□ `grep -r "console.log.*key\|console.log.*apiKey" src/` → ZERO results
□ `grep -r "_recursiveDelete" src/` → only results INSIDE src/db/index.ts
□ `grep -r "invalidateSummaryGeneration" src/` → exactly TWO call sites:
     one in the EditForm handler, one in loadTopicNodes
□ `grep -r "new Promise\|Promise.resolve\|setTimeout" src/db/` → ZERO results
     (no native Promises inside the db layer — Dexie only)
□ `grep -r "summaryGenerationMap" src/` → only results in src/lib/summarizer.ts
     (the Map must not be imported or re-declared elsewhere)
□ `grep -r "system.*role.*messages\|messages.*system" src/lib/aiAdapter` → ZERO results
     (system prompt must be top-level param for Anthropic, not in messages array)
□ `grep -r "content\[0\]\.text" src/lib/aiAdapter` → ZERO results
     (must use .find(b => b.type === 'text').text, not [0] index)
□ `grep -r "activePointers" src/` → only results in MindMapCanvas.tsx
□ `grep -r "onMouseDown\|onTouchStart" src/components/LeftPanel/MindMap` → ZERO results
     (must use Pointer Events only — no separate mouse/touch handlers)

Build verification:
□ `next build` completes with zero TypeScript errors
□ No `any` types in src/db/index.ts, src/lib/keyManager.ts, src/store/index.ts,
     src/lib/aiAdapter.ts, src/lib/contextBuilder.ts, src/lib/summarizer.ts
□ output: 'export' is set in next.config.js

================================================================
  END OF MASTER ENGINEERING CHECKLIST
================================================================
