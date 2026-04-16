# longmem — cross-project memory

You have a persistent memory MCP server. Two rules are mandatory on every chat.

## Rule 1 — search FIRST, every time
BEFORE reasoning about any bug, error, or implementation question, call:
```
search_similar(problem="<what is broken or needed>", category="<domain>", tags=[...], language="<lang>")
```
→ similarity ≥ 85%: present cached solution first, check `edge_cases` for caveats  
→ similarity < 85%: solve normally, then apply Rule 2

## Rule 2 — save on success
When user signals success ("works", "fixed", "perfect", "thanks" at end of thread), call:
```
confirm_solution(solution="<reusable synthesis>", project="<repo name>")
```
Metadata is auto-filled from the last `search_similar`. If search was skipped, use `save_solution` with all params.

Write solutions as general patterns first, then specific detail — so they're reusable across projects.

## Other tools (when needed)
- `correct_solution` — user corrects a saved fact → call immediately, no confirmation needed
- `enrich_solution` — new context after a save
- `add_edge_case` — cached result didn't fully work → record why, then `confirm_solution` with the fix
- `search_by_project` / `list_recent` / `stats` — audit memory
- `delete_solution` — remove a bad entry
- `rebuild_index` — call once at 256+ entries to compact and index
