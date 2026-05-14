# NorthStar Connector Layer

NorthStar connectors are a context layer for animation planning. They let a creator bring outside knowledge into the prompt-to-Manim workflow without turning every integration into a special-case UI.

## Product Goal

Use connected context to make better videos:

- Google Drive / Docs: lecture notes, cheat sheets, PDFs, slides.
- Notion: research pages, course databases, prompt packs.
- GitHub: code demos, README explainers, technical launch videos.
- Slack / Gmail: user questions, team feedback, launch context.
- Supabase: creator memory, render history, project gallery metadata.
- YouTube / Vimeo / Frame.io: publishing metadata, review notes, video assets.
- Whisper: lecture audio or voice notes to animation scripts.

## Current Implementation

The current implementation is intentionally local-first and safe:

- `backend/connectors.py` defines the connector catalog.
- `/api/connectors` returns connector metadata, setup status, and enabled state.
- `/api/connectors` `POST` saves selected connector IDs in local settings.
- `/api/connectors/context` returns the prompt-ready connector context block.
- `web/index.html` shows connectors in Settings and in the right-side Sources & Context panel.
- Selected connector context is injected into the director brief for the next plan.

This does not yet perform OAuth or fetch private data. It tells the planner which connected systems should shape the workflow and warns it not to invent private facts.

## Prompt Contract

Connector context should be treated as grounding instructions, not evidence. The model may say “use Google Drive cheat sheets as source material,” but it must not cite exact file contents unless the user has attached or indexed the actual file, transcript, page, or notes.

## Next Implementation Step

Add real adapters behind the existing catalog:

```text
connector -> auth setup -> search/list -> select source -> summarize/index -> source card -> director brief -> plan
```

Recommended first production connectors:

1. Supabase for persistent memory, projects, gallery metadata, and render history.
2. Google Drive / Docs for cheat sheets, PDFs, and notes.
3. YouTube Data for publish-ready uploads and descriptions.
4. Frame.io for review comments and video feedback.

## Security Requirements

- Never expose provider secrets to the browser.
- Store only OAuth tokens or API keys server-side.
- Keep source cards small and summarized.
- Ask user confirmation before external publishing.
- Separate “connector installed” from “specific source attached.”
- Log source IDs and summaries, not full private documents by default.
