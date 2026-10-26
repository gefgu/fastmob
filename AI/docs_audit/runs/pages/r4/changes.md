# Round 4 -> 5 changes (extra round beyond the 4-round cap; each page was still gaining)

Round 4 scores: Home 7.8, Learn 7.5, Release Notes 7.5, About 7.4, Features 7.1, Resources 6.9, Reference 6.8.

Root cause found for the mobile overlap that four rounds of critics flagged: the floating 38px button was the collapsed secondary sidebar (`.md-sidebar--secondary`), not `.md-sidebar-button`; it is now hidden below 76.19em.

Applied
- Home: real-data GeoLife figure (staypoints + triplegs for the busiest user-day), redrawn hierarchy diagrams (no crossings; Locations -> Tours dashed link kept on mobile), output blocks lose their copy button, darker chart label colours, hero fits one row on phones.
- Reference: Key functions column (names checked against the package), code lines under 40 chars.
- Learn: Route card thumbnail (illustrative SVG, not real data), uniform "Data: ..." meta lines, Binder as plain text.
- Features: portrait chart on phones via <picture>.
- Release Notes: single nav entry, 0.1.x as a table row, closing link to issues and GitHub releases.
- Resources/About: identifiers stay whole in tables and cards; tighter list spacing.
