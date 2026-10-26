# Round 2 -> 3 changes

Round 2 scores: Home 6.4 (regression: figures rendered as raw markdown, since fixed), Learn 6.9, Reference 6.6, Features 6.8, Resources 6.1, About 6.9, Release Notes 6.8.

Applied
- Home: figures as real <img> (md_in_html is not enabled), benchmark chart moved to the first screen with a larger-font log chart and legend above, shorter lead.
- Site: navigation.indexes removed so every section's overview page shows in its sidebar; floating back-to-top button removed (it covered text on mobile); footer pager labels kept on small screens; code-styled links inherit link colour; tables at 0.75rem with non-wrapping identifier columns; card body text darker in light mode.
- Learn: 3-column equal-height grid on wide screens, eyebrow only on "Start here", honest data tags, new trajectory thumbnail from a GeoLife outing.
- Features: arrows on all cards, charts in scroll wrapper, Validated links to dev setup, Open source links to GitHub.
- Reference: lede example defines df, "Start from your input" before the module table.
- Resources: cards reordered (PyPI, Source, Help), glossary ordered along the hierarchy with TrajDataFrame, link to citation.
- About: BibTeX version, bug-report checklist.
- Release Notes: version and date in one cell, plain-language highlights linked to docs, concrete (non-invented) upgrade policy, per-version nav children removed.

Deferred: logo asset, black light header, route/map-match thumbnail, notebook thumbnails.
