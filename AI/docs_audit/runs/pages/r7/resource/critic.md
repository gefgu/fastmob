```
PAGE: docs/src/resource/index.md   ROUND: 7
SCORES: first_impression=7 engagement=7 navigation=8 scannability=8 visual=7 ia_role=7 conciseness=8 trust=7
OVERALL: 7.4/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [ia_role] [P1] The intro sentence calls the library "fastmob" in prose ("Datasets, packages, community and background reading for fastmob."). You can see it in all three fold shots, directly under the H1. Every other prose mention on the page says Fastkit-Mobility, so the first sentence a visitor reads breaks the naming rule.
   FIX: In docs/src/resource/index.md line 3, change it to "Datasets, packages, community and background reading for Fastkit-Mobility." Keep `pip install fastmob` in the PyPI card as code.
2. [visual] [P1] On mobile (mobile-full, Class/Meaning table) rows are still 3-4 lines tall. The Class column holds one short identifier but takes about 45% of the width. The table runs about 1.3 phone screens, the longest block on the page. This was flagged in round 6 and is not fixed.
   FIX: In docs/src/assets/stylesheets/extra.css, add a page-scoped rule: `@media (max-width:600px){ .glossary-table td:first-child, .glossary-table th:first-child { width:1%; white-space:nowrap; padding-right:.6em } }`. Wrap the table in `<div class="glossary-table" markdown>`. Alternatively, drop the bold "**Trajectory**:" term prefix on phones, since the class name already says it.
3. [visual] [P2] On desktop (desktop-light-full, desktop-dark-full, "Terms used in these docs") the horizontal hierarchy diagram is still about 480px wide in an about 580px column. Its sublabels ("raw GPS / check-ins", "trips back to a place") are about 8px and hard to read. The Triplegs sublabel now fits its box.
   FIX: In extra.css, let `.figure img` fill the content column (`width:100%; max-width:none; height:auto`) so the SVG scales up about 20%.
4. [visual] [P2] The "Citing Fastkit-Mobility" tip (bottom of desktop-light-full) still has a mint/teal body fill. Only the icon and the title bar were moved to the accent, so it still reads as a foreign colour on the yellow/olive theme. In dark mode the body is dark teal under an olive title bar.
   FIX: In extra.css, also override the tip body background (`.md-typeset .admonition.tip { background-color: color-mix(in srgb, var(--md-accent-fg-color) 8%, transparent); border-color: var(--md-accent-fg-color) }`), not only the title.
5. [trust] [P2] The license line under Datasets ("Check each dataset's own page for its license...") is generic. A researcher still learns nothing about whether Brightkite or GeoLife can be used commercially.
   FIX: Add one short clause per dataset in the existing bullets where the terms are known from the linked source page (for example "SNAP, research use"). Otherwise leave the line as it is. Do not add a new section.
6. [engagement] [P2] (Deferred, not scored.) There are no dataset thumbnails, no card icons, and no DOI links on Papers.
   FIX: None this round. Keep the deferred list.

PREVIOUS ISSUES STATUS (rounds >= 2):
1 (first_impression, uneven PyPI card / pip line): fixed. The PyPI card now opens with `pip install fastmob`, and all three cards are the same height on desktop and stack evenly on mobile.
2 (trust, papers without links / license placement): partial. The license line now sits directly under the Datasets list. It is still generic. DOI links are deferred.
3 (visual, diagram scale and Triplegs sublabel overflow): partial. The Triplegs sublabel ("moving segments") now fits in both variants, and the mobile vertical diagram is clean. The desktop figure is still undersized.
4 (visual, tall mobile glossary rows): not fixed.
5 (engagement, datasets thumbnail): deferred (not scored).
6 (visual, teal tip admonition): partial. The icon and title use the accent, but the body fill is still mint/teal.

WHAT WORKS (max 3, so it is not broken in the next round):
- The three equal-height link cards with `pip install fastmob` as the PyPI body make the first screen actionable in one glance on both desktop and mobile.
- The mobile vertical hierarchy diagram (served via <picture>) is readable at 390px, with every node and the dashed Tours to Locations relation visible.
- Clear wayfinding: the About sidebar highlights Resources, the right TOC (Datasets / Background reading > Papers / Terms) mirrors the page, and the footer pager stacks cleanly on phones.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none this round (the datasets thumbnail strip remains deferred).

CROSS-PAGE NOTES:
- The About section sidebar entry and footer pager read "About fastmob" (sidebar, all screenshots; pager "Previous: About fastmob" in full shots). This is prose/nav naming, so retitle that page/nav entry "About Fastkit-Mobility" in zensical.toml or the About page H1.
- Tip admonition body colour is still teal site-wide. Fix it once in extra.css.
- Deferred (P2 only): clipped header logo image ("Mobility" cut at the right edge of the logo box), black light-mode header, footer Previous crossing sections, no card icons, no dataset thumbnails, papers without DOI links.
```
