# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An Obsidian vault. All notes are Markdown files. Obsidian-specific syntax:
- Wikilinks: `[[Note Name]]` and `[[Note Name|display text]]`
- Frontmatter: YAML block at top of file between `---` delimiters (used for properties/tags/dates)
- Embeds: `![[Note Name]]` to inline another note
- Tags: `#tag` inline or `tags:` in frontmatter

## Vault structure

- `Welcome.md` — default starter note (can be deleted)
- `.obsidian/` — vault config (plugins, workspace state, appearance) — do not edit manually

## Active plugins

**Core:** file-explorer, global-search, graph, backlink, canvas, outgoing-link, tag-pane, properties, daily-notes, templates, note-composer, command-palette, bookmarks, outline, word-count, file-recovery, sync, bases

**Community:**
- `terminal` (polyipseity v3.26.0) — integrated zsh shell inside Obsidian; default profile runs `/bin/zsh --login` with cwd set to vault root. Terminal panel is currently pinned and titled "Claude Code".

## Notes on working with this vault

- Obsidian Sync is enabled — changes sync across devices automatically.
- The workspace has an integrated terminal panel open running Claude Code. Claude Code tasks run from the vault root (`/Users/jamesraphel/Documents/Claude/Obsidian File`).
- When creating notes, prefer frontmatter properties over inline tags for metadata that isn't meant to appear in graph view.
