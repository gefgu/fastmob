# Round 1 -> 2 changes (all pages)

Round 1 scores: Home 7.0, Learn 5.5, Reference 5.5, Features 4.4, Resources 4.6, About 5.0, Release Notes 6.0.

Applied
- Nav: 7 tabs -> 5 (Home, Features, Learn, Reference, About). Resources and Release Notes now live under About. Section index entries renamed ("Learn overview", ...) so footer prev/next read clearly. Reference sidebar groups collapse by default (navigation.sections removed). Visualization added to Reference nav; contact-network.md moved to reference/measures/.
- Theme: one link colour per theme (dark gold light / yellow dark), dark footer matches page, footer copyright links to PyPI/GitHub, tables fill the column, figures scroll sideways on mobile instead of shrinking.
- Home: log-scale benchmark chart (published speedups), hierarchy diagram redrawn (Locations branch off Staypoints; softer dark mode), hero output matches code, outputs inside tabs, nav hidden, shorter "Why", next steps point at new destinations.
- Features: card grid, import-switch snippet, speed chart, Dig deeper row; dropped the false "<30KB wheel" claim (wheels are ~21 MB) and "500x" (now median 426x from the published chart).
- Learn: sections Guides / Notebook examples, one card per sidebar entry, nav labels = card titles, metadata row, thumbnails from real data (GeoLife, Brightkite) plus a backend diagram, "Start here" marker, TOC hidden, duplicate tip removed.
- Reference: API | Description table of all 12 modules, one-line lede and code example.
- About: license, relationship to scikit-mobility, citation, maintainer, badges. Resources: external links, datasets, linked glossary. Release Notes: dates, v0.2.1-v0.3.1 pages, latest-release callout, concrete upgrade policy.

Declined / deferred
- Header logo text ("Fastkit Mobilit") and white box in dark mode: needs the source asset re-exported.
- Black header in light mode kept (brand).
- Mobile floating TOC overlap, split of the 55,000px Individual measures page: later rounds.
