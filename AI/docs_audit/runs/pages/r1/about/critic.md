PAGE: about/index.md   ROUND: 1
SCORES: first_impression=5 engagement=3 navigation=5 scannability=6 visual=6 ia_role=5 conciseness=6 trust=4
OVERALL: 5.0/10   VERDICT: REVISE
(PASS only if OVERALL >= 8.0 AND no dimension < 6)

TOP ISSUES (max 7, ordered by impact on a first-time reader):
1. [trust] [P0] The page is missing everything a reader expects from an About page in research software. It has no license, no maintainer or author, no "how to cite" block, no mention of scikit-mobility as the API ancestor, no acknowledgements or third-party credits (e.g. the MoveTK Apache-2.0 notice), and no current version. All three full-page screenshots end after "Contribute and report issues", and none of these items appear anywhere.
   FIX: In docs/src/about/index.md, replace "What it supports" with four short sections, each 1-3 lines. (a) "Relationship to scikit-mobility": one sentence saying it uses the same measure API and is a Rust reimplementation, with a link to skmob. (b) "License": name the license, link LICENSE on GitHub, and link the third-party notices. (c) "Cite fastmob": a fenced BibTeX block plus a link to CITATION.cff if one exists. (d) "Maintainers": name and GitHub handle.
2. [ia_role] [P1] "What it supports" (desktop light, y≈350-520) repeats the landing and Features pitch: Narwhals backends, trajectory hierarchy, measures. On a meta page this is marketing copy in the wrong place, and it is also the weakest version of that pitch (plain semicolon bullets, no links).
   FIX: Delete the "What it supports" section and keep the one-sentence intro paragraph. If a capability pointer is wanted, add a single line: "See [Features](../features/index.md) for what fastmob can do."
3. [navigation] [P1] The left sidebar (desktop, x≈123,y≈165) lists only "Development setup". The "Overview" entry, which is this page, is folded into the section index, so the reader sees a one-item sidebar with no section title and no "you are here" marker. The About tab in the top nav is the only orientation cue.
   FIX: Retitle the zensical.toml entry `{ "Development setup" = "about/development.md" }` to "Contributing / development setup" and add a short "Contributing" lead-in on this page that links to it. Better still, collapse About into a single leaf page (see CROSS-PAGE NOTES) so the one-item sidebar disappears.
4. [engagement] [P1] The page is text only: an H1, three H2s, bullets and a sentence. There is no logo, badge row or visual anchor anywhere above or below the fold. It reads as a placeholder, which matches the known "About looks low-value" weakness.
   FIX: Add a compact badge row under the H1: PyPI version, Python versions, license, CI status, and GitHub stars, using shields.io images. These double as live trust signals. Do not add a hero image.
5. [visual] [P1] On mobile (mobile-full, y≈770-830) the floating table-of-contents button overlaps body text in "Project status": "optional extras" is clipped behind the round button. In the mobile footer, the Previous link shows only a bare left arrow with no label, while Next is labelled "Development setup".
   FIX: In docs/src/stylesheets/extra.css, give the floating TOC button a solid background with a shadow and right padding on content, or hide it on pages with 3 or fewer headings. Restore the "Previous" label in the mobile footer, or hide the orphan arrow (site-wide).
6. [scannability] [P2] "Contribute and report issues" mixes three link bullets with a dense sentence of bug-report instructions. "Release history" duplicates the "release notes" link two lines above. "Documentation recipes" is a vague label for the Learn section.
   FIX: Drop the "Release history" bullet. Rename "Documentation recipes" to "Learn (tutorials)". Turn the bug-report sentence into a short checklist (version, backend, minimal input, expected vs actual, traceback), or move it into a GitHub issue template and link that instead.
7. [visual] [P2] In the light theme, links are olive-yellow underlines on white (e.g. "reference" and "release notes", y≈655). They read as muted and off-brand next to the dark header, while dark mode uses blue links, so the two themes look like different sites. The header logo reads "Fastkit-Mobility" while the wordmark says "fastmob", which undermines the identity on a page whose job is identity.
   FIX: Pick one accent colour per theme in extra.css that meets WCAG AA on white (a darker gold/amber or the dark-mode blue). Make the logo asset match the "fastmob" name, or explain the Fastkit-Mobility umbrella in one line on this page.

PREVIOUS ISSUES STATUS (rounds >= 2): n/a (round 1)

WHAT WORKS (max 3, so it is not broken in the next round):
- The opening paragraph is a clear two-sentence definition that names Rust kernels and dataframe-facing API. Keep it as the page lead.
- "Pin a released version for reproducible research or production pipelines" is an honest, useful stability statement. Keep the Project status framing.
- The page is short with a sensible right-hand "On this page" TOC, and dark mode has good contrast throughout.

IMAGE REQUESTS (only where a visual would explain something words cannot; max 3):
- none. This meta page needs badges and facts (license, citation, maintainers), not illustrations.

CROSS-PAGE NOTES:
- Merge "About" and "Release Notes" in the top nav, or make About a single leaf. Currently About is a two-item submenu (Overview + Development setup), Resources is a leaf, and Release Notes is a long submenu, so the right end of the nav mixes leaf and submenu items. Suggestion: one "About" section holding Overview, Contributing, Release notes and Resources.
- The mobile footer drops the "Previous" label (bare arrow), and the floating TOC button overlaps body text. This is theme-level and likely affects every page.
- The light-theme link colour (olive-yellow) differs from the dark-theme link colour (blue). Unify the accent colour in extra.css.
- The "Fastkit-Mobility" logo vs the "fastmob" wordmark mismatch appears in the header on every page.
